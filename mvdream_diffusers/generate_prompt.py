import argparse
import json
import os
import random
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
1. `items` 长度必须严格等于 {batch_size}，每个 item 只包含 `prompt` 和 `prompt_cn`
2. 每条数据都必须是同一主体的 four-view multi-view prompt，顺序为 front, left, back, right，主体外观保持一致
3. `prompt` 和 `prompt_cn` 要写成完整自然的一句话，清晰说明主体、四视角和一致性要求
4. 主体要多样，允许自由发挥，覆盖人物、动物、车辆、产品、幻想生物等不同类型
5. 背景不固定，可以自由变化
6. 所有字符串必须是单行；不要输出解释
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
        f" prompt 和 prompt_cn 不要太短，写成自然完整的一句话。"
        f" 但也不要过长，不要堆很多风格形容词，不要复杂长段落。"
        f" 背景可以自由发挥，不要让所有 prompt 都变成纯白背景。"
        f" 输出为严格 JSON 对象，顶层键为 items，每个 item 只保留 prompt 和 prompt_cn 两个字段。"
        f" 不要生成 category。"
        f"{avoid_text}"
    )


def repair_generated_item(item: Dict[str, Any], prompt_id: int) -> Dict[str, Any]:
    prompt = norm_line(str(item.get("prompt", "")))
    prompt_cn = norm_line(str(item.get("prompt_cn", "")))

    if not prompt:
        prompt = (
            "Generate four consistent views of the same subject in the exact order front, left, back, and right, "
            "with a coherent appearance and a natural scene that matches the subject."
        )
    if not prompt_cn:
        prompt_cn = "请生成同一主体的四个一致视角，顺序为正视图、左视图、后视图和右视图，主体外观保持连贯，背景场景自然匹配主体。"

    return {
        "id": f"mv_{prompt_id}",
        "category": "Multi-View Consistency",
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
    seed: int,
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
            extra_body={"enable_thinking": True, "thinking_budget": 4096, "seed": seed},
        )
        answer_content = completion.choices[0].message.content or "{}"
    except Exception:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"enable_thinking": True, "thinking_budget": 4096, "seed": seed},
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
    rng = random.SystemRandom()

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        current_batch_size = min(batch_size, count - len(results))
        existing_prompts = [item["prompt"] for item in results]
        seed = rng.randint(1, 2**31 - 1)
        print(f"[attempt {attempts}] requesting {current_batch_size} prompts with seed {seed}...")

        raw_items = request_prompt_batch(
            client,
            model=model,
            batch_size=current_batch_size,
            existing_prompts=existing_prompts,
            seed=seed,
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
