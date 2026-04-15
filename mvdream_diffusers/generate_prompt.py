import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from openai import OpenAI


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


def norm_line(s: str) -> str:
    return " ".join(s.strip().split())


def build_system_prompt(batch_size: int) -> str:
    return f'''你是一个专业的 multi-view prompt 设计师。

你的任务是一次性生成 {batch_size} 条高质量、多样化的 multi-view prompt，用于“单次生成同一主体的多个视角图像”。

严格输出 JSON，不能输出任何解释、注释、markdown 代码块。

输出格式必须严格为：
{{
  "items": [
    {{
      "prompt": "英文完整 prompt，单行字符串",
      "prompt_cn": "中文完整 prompt，单行字符串"
    }}
  ]
}}

硬性要求：
1. `items` 长度必须严格等于 {batch_size}
2. 每条数据都必须是 multi-view，不是时序，不是剧情，不是不同物体拼接
3. 每条数据都必须明确表示：生成同一主体的四个固定视角，顺序是 front, left, back, right
4. 四张图必须是同一个主体，只允许视角变化，不允许主体、颜色、材质、配件、数量发生变化
5. `prompt` 必须适合直接用于图像生成，写成自然、具体、清晰的英文
6. `prompt_cn` 必须是对应的自然中文完整 prompt，不要只是英文直拷贝
7. 主体要非常多样，必须覆盖并混合：动物、车辆、家具、玩具、日用品、植物、雕塑、机器人、食物、建筑小品、幻想生物、武器道具、服饰配件、电子产品，以及人物
8. 人物必须明显纳入生成范围，包括但不限于：young woman, old man, boy, girl, warrior, wizard, astronaut, chef, dancer, scientist, knight, cyberpunk character, cartoon character
9. 尽量避免复杂多人场景，优先单主体
10. 不要生成重复或近似重复的主体
11. 所有字符串必须是单行，不允许出现换行符
12. 不要输出 `dimension`、`global_prompt`、`global_prompt_cn`、`sub_prompts`、`sub_prompts_cn`、`auxiliary_info`、`auxiliary_info_cn`
13. 每条 prompt 自身要写完整，不依赖其他字段
14. `prompt` 里应明确提到 multi-view / four views / front left back right / same subject / consistent appearance 这类信息
15. `prompt` 和 `prompt_cn` 要尽量简洁，写成短一些、直白一些的生成指令，不要过度堆砌形容词
16. 避免复杂背景、镜头语言、电影化描述、长串风格词，优先写清主体和四视角要求
'''


def build_user_prompt(batch_size: int, existing_prompts: List[str]) -> str:
    recent = existing_prompts[-30:]
    avoid_text = ""
    if recent:
        avoid_text = f"\n请避免重复或近似重复以下 prompt：{json.dumps(recent, ensure_ascii=False)}"

    return (
        f"请生成 {batch_size} 条 multi-view prompt。"
        f" 每条都必须是同一主体的四个固定视角：front, left, back, right。"
        f" 需要显著提高主体多样性，并明确加入人物类主体。"
        f" 输出为严格 JSON 对象，顶层键为 items，每个 item 只保留 prompt 和 prompt_cn 两个字段。"
        f"{avoid_text}"
    )


def repair_generated_item(item: Dict[str, Any], prompt_id: int) -> Dict[str, Any]:
    prompt = norm_line(str(item.get("prompt", "")))
    prompt_cn = norm_line(str(item.get("prompt_cn", "")))

    if not prompt:
        prompt = (
            "A multi-view image set of the same subject, showing four consistent views in the exact order: "
            "front, left, back, right, with consistent appearance and identity."
        )
    if not prompt_cn:
        prompt_cn = "同一主体的多视角图像，严格按照正视图、左视图、后视图、右视图的顺序生成四个一致视角，外观与身份保持一致。"

    return {
        "id": f"mv_{prompt_id}",
        "prompt": prompt,
        "prompt_cn": prompt_cn,
        "num_images": 4,
    }


def parse_items_from_response(answer_content: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(extract_json_from_text(answer_content))
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        print(answer_content)
        return []

    if isinstance(payload, dict):
        items = payload.get("items", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    return []


def request_prompt_batch(
    client: OpenAI,
    *,
    model: str,
    batch_size: int,
    existing_prompts: List[str],
) -> List[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": build_system_prompt(batch_size)},
        {"role": "user", "content": build_user_prompt(batch_size, existing_prompts)},
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

    return parse_items_from_response(answer_content)


def make_dedupe_key(item: Dict[str, Any]) -> str:
    return norm_line(str(item.get("prompt", ""))).lower()


def generate_multi_view_prompts(
    client: OpenAI,
    *,
    model: str,
    count: int,
    batch_size: int,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    attempts = 0

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        current_batch_size = min(batch_size, count - len(results))
        existing_prompts = [item["prompt"] for item in results]
        print(f"[attempt {attempts}] requesting {current_batch_size} prompts...")

        raw_items = request_prompt_batch(
            client,
            model=model,
            batch_size=current_batch_size,
            existing_prompts=existing_prompts,
        )

        added = 0
        for raw_item in raw_items:
            repaired = repair_generated_item(raw_item, len(results) + 1)
            key = make_dedupe_key(repaired)
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(repaired)
            added += 1
            if len(results) >= count:
                break

        print(f"added {added}, total {len(results)}/{count}")

    if len(results) < count:
        raise RuntimeError(
            f"Only generated {len(results)} prompts after {attempts} attempts. "
            f"Try increasing --max-attempts or lowering --batch-size."
        )

    return results[:count]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate multi-view prompts with Qwen.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "multi_view_prompts.json"),
        help="Output JSON file path.",
    )
    parser.add_argument("--model", type=str, default="qwen3.6-plus", help="Model name.")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of multi-view prompts to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of prompts requested from Qwen per API call.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Maximum number of batch requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("错误: 找不到 DASHSCOPE_API_KEY 环境变量。")

    if args.count <= 0:
        raise SystemExit("错误: --count 必须大于 0。")
    if args.batch_size <= 0:
        raise SystemExit("错误: --batch-size 必须大于 0。")
    if args.max_attempts <= 0:
        raise SystemExit("错误: --max-attempts 必须大于 0。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    payload = generate_multi_view_prompts(
        client,
        model=args.model,
        count=args.count,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(payload)} multi-view prompts to {output_path}")


if __name__ == "__main__":
    main()
