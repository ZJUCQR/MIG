import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "evaluation" / "multi_image_prompts.json"


def extract_json_from_text(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    raw_obj = re.search(r"\{.*\}", text, re.DOTALL)
    if raw_obj:
        return raw_obj.group(0)
    raw_arr = re.search(r"\[.*\]", text, re.DOTALL)
    if raw_arr:
        return raw_arr.group(0)
    return text


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def build_system_prompt() -> str:
    return """你是一个专业的图像生成 prompt 重写助手。
你的任务是重写一个多图角色一致性 prompt 条目，并把所有子图都改成“同一角色的正脸照”，同时尽量保留原本的人物身份、服装、场景、风格和多图结构。

严格要求：
1. 所有英文和中文 prompt 都必须明确强调：同一角色、正脸、正面朝向镜头/观众、脸部清晰、以脸部人像为主体。
2. 不允许输出侧脸、背影、低头遮脸、远景全身主体、看向别处等描述。
3. 如果原 prompt 是绘画、涂鸦、浮世绘、雕塑等风格，也要改写成该风格下的正脸肖像。
4. 必须保留原来的 num_images，不得增减图片数量。
5. sub_prompts 和 sub_prompts_cn 的长度必须严格等于 num_images。
6. prompt、prompt_cn、global_prompt、global_prompt_cn 必须是单行字符串，不能包含换行符。
7. 只输出 JSON 对象，不要输出解释、注释或 markdown。
8. 不要输出任何额外字段。

输出 JSON 格式必须严格为：
{
  "prompt": "...",
  "prompt_cn": "...",
  "global_prompt": "...",
  "global_prompt_cn": "...",
  "num_images": 4,
  "sub_prompts": ["..."],
  "sub_prompts_cn": ["..."]
}
"""


def build_user_prompt(item: Dict[str, Any]) -> str:
    return (
        "请重写下面这个多图 prompt 条目。\n"
        "目标：把整组图全部改成同一角色的正脸照，要求角色正面朝向镜头，脸部清晰可见，以脸部肖像或半身肖像为主体。\n"
        "同时尽量保留原本的人物身份、服装、场景、艺术风格和多图差异。\n"
        f"必须保持 num_images={int(item.get('num_images', 0))} 不变。\n"
        "请返回严格 JSON。\n"
        f"原始条目: {json.dumps(item, ensure_ascii=False)}"
    )


def repair_result_schema(result_dict: Dict[str, Any], *, original_item: Dict[str, Any]) -> Dict[str, Any]:
    num_images = int(original_item.get("num_images", 1))

    repaired = {
        "id": original_item.get("id", ""),
        "prompt": normalize_text(result_dict.get("prompt", original_item.get("prompt", ""))),
        "prompt_cn": normalize_text(result_dict.get("prompt_cn", original_item.get("prompt_cn", ""))),
        "global_prompt": normalize_text(result_dict.get("global_prompt", original_item.get("global_prompt", ""))),
        "global_prompt_cn": normalize_text(result_dict.get("global_prompt_cn", original_item.get("global_prompt_cn", ""))),
        "num_images": num_images,
    }

    sub_prompts = result_dict.get("sub_prompts", [])
    sub_prompts_cn = result_dict.get("sub_prompts_cn", [])
    original_sub_prompts = original_item.get("sub_prompts", [])
    original_sub_prompts_cn = original_item.get("sub_prompts_cn", [])

    if not isinstance(sub_prompts, list):
        sub_prompts = []
    if not isinstance(sub_prompts_cn, list):
        sub_prompts_cn = []
    if not isinstance(original_sub_prompts, list):
        original_sub_prompts = []
    if not isinstance(original_sub_prompts_cn, list):
        original_sub_prompts_cn = []

    repaired["sub_prompts"] = [
        normalize_text(sub_prompts[i]) if i < len(sub_prompts) else normalize_text(original_sub_prompts[i] if i < len(original_sub_prompts) else "")
        for i in range(num_images)
    ]
    repaired["sub_prompts_cn"] = [
        normalize_text(sub_prompts_cn[i]) if i < len(sub_prompts_cn) else normalize_text(original_sub_prompts_cn[i] if i < len(original_sub_prompts_cn) else "")
        for i in range(num_images)
    ]
    return repaired


def rewrite_item(client: OpenAI, item: Dict[str, Any], *, model: str) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(item)},
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": True, "thinking_budget": 4096},
        )
        answer_content = completion.choices[0].message.content or "{}"
    except Exception:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"enable_thinking": True, "thinking_budget": 4096},
        )
        answer_content = completion.choices[0].message.content or "{}"

    try:
        result = json.loads(extract_json_from_text(answer_content))
    except json.JSONDecodeError as error:
        print(f"JSON parse failed for {item.get('id', '')}: {error}")
        print(answer_content)
        return None

    return repair_result_schema(result, original_item=item)


def rewrite_items(client: OpenAI, items: List[Dict[str, Any]], *, model: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rewritten_items: List[Dict[str, Any]] = []
    target_items = items[:limit] if limit is not None else items
    total = len(target_items)

    for index, item in enumerate(target_items, start=1):
        print(f"[{index}/{total}] Rewriting {item.get('id', '')} | num_images={item.get('num_images', '')}")
        rewritten = rewrite_item(client, item, model=model)
        if rewritten is None:
            raise RuntimeError(f"Failed to rewrite item: {item.get('id', '')}")
        rewritten_items.append(rewritten)

    if limit is None:
        return rewritten_items

    merged = list(items)
    for index, item in enumerate(rewritten_items):
        merged[index] = item
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite multi-image prompts into frontal-face character portraits and overwrite the source JSON by default."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to multi_image_prompts.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path. Defaults to overwriting --input.",
    )
    parser.add_argument("--model", type=str, default="qwen3.6-plus", help="Qwen model name.")
    parser.add_argument("--limit", type=int, default=None, help="Only rewrite the first N items for testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("Error: DASHSCOPE_API_KEY environment variable is required.")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.exists():
        raise SystemExit(f"Error: input file does not exist: {input_path}")

    items = load_json_array(input_path)
    print(f"Loaded {len(items)} prompt items from {input_path}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    rewritten_items = rewrite_items(client, items, model=args.model, limit=args.limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rewritten_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rewritten_items)} items to {output_path}")


if __name__ == "__main__":
    main()
