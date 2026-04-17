import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


DEFAULT_ERROR_JSON = Path(__file__).resolve().parent / "error.json"
DEFAULT_FULL_JSON = Path(__file__).resolve().parent / "evaluation" / "multi_image_prompts.json"


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


def normalize_text(text: Any) -> str:
    return " ".join(str(text).split())


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def collect_existing_celebrities(items: List[Dict[str, Any]]) -> List[str]:
    names = set()
    patterns = [
        r"Albert Einstein|Barack Obama|Andrew Ng|Elon Mask|Elon Musk|Feifei Li|Geoffrey Hinton|Joe Biden|Kamala Harris|Kaiming He|Yann LeCun|Michelle Obama|Sam Altman|Scarlett Johansson|Taylor Swift|Dwayne Johnson",
    ]
    for item in items:
        text_candidates = [
            item.get("prompt", ""),
            item.get("global_prompt", ""),
            *item.get("sub_prompts", []),
        ]
        joined = " ".join(normalize_text(x) for x in text_candidates)
        for pattern in patterns:
            for match in re.findall(pattern, joined):
                names.add(match)
    return sorted(names)


def choose_new_celebrity(current_item: Dict[str, Any], candidate_names: List[str], rng: random.Random) -> str:
    text_candidates = [
        current_item.get("prompt", ""),
        current_item.get("global_prompt", ""),
        *current_item.get("sub_prompts", []),
    ]
    joined = " ".join(normalize_text(x) for x in text_candidates)
    current_names = set()
    for name in candidate_names:
        if name in joined:
            current_names.add(name)
    pool = [name for name in candidate_names if name not in current_names]
    if not pool:
        raise ValueError("No alternative celebrity available for replacement.")
    return rng.choice(pool)


def build_system_prompt() -> str:
    return '''你是一个专业的 prompt 重写助手。

你的任务是重写一个失败的文生组图 prompt，把其中的名人替换成另一个不同的名人，同时保持组图结构、图片数量、场景变化和风格变化尽量不变。

严格输出 JSON，不能输出任何解释、注释、markdown 代码块。

输出格式必须严格为：
{
  "prompt": "英文完整 prompt，单行字符串",
  "prompt_cn": "中文完整 prompt，单行字符串",
  "global_prompt": "英文全局描述，单行字符串",
  "global_prompt_cn": "中文全局描述，单行字符串",
  "num_images": 4,
  "sub_prompts": ["..."],
  "sub_prompts_cn": ["..."]
}

硬性要求：
1. 必须把原 prompt 中的人物替换成新的名人
2. 新名人必须和原名人不同
3. `num_images` 必须严格保持不变
4. `sub_prompts` 和 `sub_prompts_cn` 的长度必须严格等于 `num_images`
5. 保留原有组图结构、场景变化和风格变化
6. 不要输出任何额外字段
'''


def build_user_prompt(item: Dict[str, Any], new_celebrity: str) -> str:
    return (
        f"请把下面这个失败的组图 prompt 重写，并把其中的名人替换成：{new_celebrity}。"
        f"\n必须保持 num_images={item['num_images']} 不变，并尽量保持场景和风格结构不变。"
        f"\n原始 JSON: {json.dumps(item, ensure_ascii=False)}"
    )


def repair_result_schema(result_dict: Dict[str, Any], *, original_item: Dict[str, Any]) -> Dict[str, Any]:
    num_images = int(original_item.get("num_images", 1))
    repaired = {
        "id": original_item["id"],
        "prompt": normalize_text(result_dict.get("prompt", "")),
        "prompt_cn": normalize_text(result_dict.get("prompt_cn", "")),
        "global_prompt": normalize_text(result_dict.get("global_prompt", "")),
        "global_prompt_cn": normalize_text(result_dict.get("global_prompt_cn", "")),
        "num_images": num_images,
    }

    sub_prompts = result_dict.get("sub_prompts", [])
    sub_prompts_cn = result_dict.get("sub_prompts_cn", [])
    if not isinstance(sub_prompts, list):
        sub_prompts = []
    if not isinstance(sub_prompts_cn, list):
        sub_prompts_cn = []

    repaired["sub_prompts"] = [normalize_text(sub_prompts[i]) if i < len(sub_prompts) else "" for i in range(num_images)]
    repaired["sub_prompts_cn"] = [normalize_text(sub_prompts_cn[i]) if i < len(sub_prompts_cn) else "" for i in range(num_images)]
    return repaired


def rewrite_item(client: OpenAI, item: Dict[str, Any], *, model: str, new_celebrity: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(item, new_celebrity)},
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

    result = json.loads(extract_json_from_text(answer_content))
    return repair_result_schema(result, original_item=item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite failed prompts with a different random celebrity and write replacements back into the full JSON.")
    parser.add_argument("--error-json", type=str, default=str(DEFAULT_ERROR_JSON), help="Path to error.json.")
    parser.add_argument("--full-json", type=str, default=str(DEFAULT_FULL_JSON), help="Path to the full multi_image_prompts.json.")
    parser.add_argument("--model", type=str, default="qwen3.6-plus", help="Qwen model name.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for celebrity replacement.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("错误: 找不到 DASHSCOPE_API_KEY 环境变量。")

    error_json = Path(args.error_json)
    full_json = Path(args.full_json)
    if not error_json.exists():
        raise SystemExit(f"错误: 文件不存在: {error_json}")
    if not full_json.exists():
        raise SystemExit(f"错误: 文件不存在: {full_json}")

    failed_items = load_json_array(error_json)
    all_items = load_json_array(full_json)
    all_by_id = {str(item.get("id")): item for item in all_items}

    candidate_names = collect_existing_celebrities(all_items)
    if len(candidate_names) < 2:
        raise SystemExit("错误: 可替换的名人数量不足。")

    rng = random.Random(args.seed)
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    for index, failed_item in enumerate(failed_items, start=1):
        prompt_id = str(failed_item.get("id", "")).strip()
        if not prompt_id or prompt_id not in all_by_id:
            print(f"[{index}/{len(failed_items)}] 跳过未知 id: {prompt_id}")
            continue

        target_item = all_by_id[prompt_id]
        new_celebrity = choose_new_celebrity(target_item, candidate_names, rng)
        print(f"[{index}/{len(failed_items)}] 重写 {prompt_id} -> {new_celebrity}")
        rewritten = rewrite_item(client, target_item, model=args.model, new_celebrity=new_celebrity)
        all_by_id[prompt_id] = rewritten

    merged_items = []
    for item in all_items:
        prompt_id = str(item.get("id", ""))
        merged_items.append(all_by_id.get(prompt_id, item))

    full_json.write_text(json.dumps(merged_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已写回完整文件: {full_json}")


if __name__ == "__main__":
    main()
