import argparse
import csv
import json
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "evaluation" / "EvaluationIMGs_stars_prompts.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "evaluation" / "multi_image_prompts.json"


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


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("_") or "subject"


def human_name_from_image_name(name: str) -> str:
    stem = Path(name).stem.replace("_", " ").strip()
    return " ".join(part.capitalize() for part in stem.split()) or "the same person"


def load_csv_groups(csv_path: Path) -> "OrderedDict[str, List[str]]":
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = norm_line(str(row.get("Image_Name", "")))
            prompt = norm_line(str(row.get("Prompt", "")))
            if not image_name or not prompt:
                continue
            groups.setdefault(image_name, []).append(prompt)
    return groups


def choose_chunk_size(remaining: int, rng: random.Random, min_group_size: int, max_group_size: int) -> int:
    upper = min(max_group_size, remaining)
    if remaining <= max_group_size:
        if remaining >= min_group_size:
            return remaining
        return 0

    valid_sizes: List[int] = []
    for size in range(min_group_size, upper + 1):
        leftover = remaining - size
        if leftover == 0 or leftover >= min_group_size:
            valid_sizes.append(size)

    if not valid_sizes:
        return 0
    return rng.choice(valid_sizes)


def build_grouped_entries(
    grouped_prompts: "OrderedDict[str, List[str]]",
    *,
    rng: random.Random,
    min_group_size: int,
    max_group_size: int,
    limit_people: Optional[int],
    limit_groups: Optional[int],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for person_index, (image_name, prompts) in enumerate(grouped_prompts.items(), start=1):
        if limit_people is not None and person_index > limit_people:
            break

        prompts_copy = list(prompts)
        rng.shuffle(prompts_copy)
        chunk_index = 1

        while len(prompts_copy) >= min_group_size:
            if limit_groups is not None and len(entries) >= limit_groups:
                return entries

            chunk_size = choose_chunk_size(len(prompts_copy), rng, min_group_size, max_group_size)
            if chunk_size < min_group_size:
                break

            chunk = prompts_copy[:chunk_size]
            prompts_copy = prompts_copy[chunk_size:]
            entries.append(
                {
                    "id": f"cn_{len(entries) + 1:03d}",
                    "image_name": image_name,
                    "num_images": chunk_size,
                    "source_prompts": chunk,
                }
            )
            chunk_index += 1

    return entries


def build_system_prompt() -> str:
    return '''你是一个专业的文生组图 prompt 设计师。

你的任务是把输入的一组英文 prompt 改写成“单次生成多张图片”的组图 prompt。注意，这组 prompt 都对应同一个人，只是服装、动作、风格、场景等发生变化。你必须在输出的 prompt 和 global_prompt 中明确写出这个人物是谁，例如 Albert Einstein，这样生成模型能知道整组图片都是同一个指定人物。

严格输出 JSON，不能输出任何解释、注释、markdown 代码块。

输出格式必须严格为：
{
  "prompt": "英文完整 prompt，必须是单行字符串，不允许出现\\n。结构应为：先写 global prompt，再按顺序列出每个子图指令，明确这是一组同一人物的多图/组图/sequence 图片。",
  "prompt_cn": "中文完整 prompt，必须是单行字符串，不允许出现\\n",
  "global_prompt": "英文全局描述，强调同一个人物身份一致、明确人物姓名，并描述整组图片的共同设定",
  "global_prompt_cn": "中文全局描述，明确人物是谁",
  "num_images": 4,
  "sub_prompts": [
    "English instruction for image 1",
    "English instruction for image 2"
  ],
  "sub_prompts_cn": [
    "第1张图中文子指令",
    "第2张图中文子指令"
  ]
}

硬性要求：
1. `num_images` 必须严格等于输入 prompt 的条数
2. `sub_prompts` 和 `sub_prompts_cn` 的长度必须严格等于 `num_images`
3. 输出的 `prompt` 和 `global_prompt` 必须明确写出人物姓名，例如 Albert Einstein；中文字段也要明确写出人物姓名，例如 爱因斯坦
4. `prompt`、`prompt_cn`、`global_prompt`、`global_prompt_cn` 都必须是单行字符串，不能含换行符
5. `sub_prompts` 是英文列表，`sub_prompts_cn` 是中文列表
6. 不要输出任何额外字段，也不要丢失 JSON 结构
'''


def build_user_prompt(entry: Dict[str, Any]) -> str:
    person_name = human_name_from_image_name(entry["image_name"])
    prompt_lines = "\n".join(f"{idx}. {prompt}" for idx, prompt in enumerate(entry["source_prompts"], start=1))
    return (
        f"请把下面这些属于同一个人物 `{entry['image_name']}` 的英文 prompt，合并改写成一个适合文生组图的 prompt。"
        f"\n这个人物明确是：{person_name}。"
        f"\n输出的英文 prompt 和 global_prompt 必须明确写出 {person_name}；中文字段也必须明确写出这个人物是谁。"
        f"\n这些 prompt 都必须对应同一个人，最终输出的 num_images 必须等于 {entry['num_images']}。"
        f"\n原始 prompts:\n{prompt_lines}"
    )


def repair_result_schema(result_dict: Dict[str, Any], *, entry: Dict[str, Any]) -> Dict[str, Any]:
    num_images = entry["num_images"]

    repaired = {
        "id": entry["id"],
        "prompt": " ".join(str(result_dict.get("prompt", "")).split()),
        "prompt_cn": " ".join(str(result_dict.get("prompt_cn", "")).split()),
        "global_prompt": " ".join(str(result_dict.get("global_prompt", "")).split()),
        "global_prompt_cn": " ".join(str(result_dict.get("global_prompt_cn", "")).split()),
        "num_images": num_images,
    }

    sub_prompts = result_dict.get("sub_prompts", [])
    sub_prompts_cn = result_dict.get("sub_prompts_cn", [])
    if not isinstance(sub_prompts, list):
        sub_prompts = []
    if not isinstance(sub_prompts_cn, list):
        sub_prompts_cn = []

    normalized_sub_prompts: List[str] = []
    normalized_sub_prompts_cn: List[str] = []
    for index in range(num_images):
        normalized_sub_prompts.append(" ".join(str(sub_prompts[index]).split()) if index < len(sub_prompts) else "")
        normalized_sub_prompts_cn.append(" ".join(str(sub_prompts_cn[index]).split()) if index < len(sub_prompts_cn) else "")

    repaired["sub_prompts"] = normalized_sub_prompts
    repaired["sub_prompts_cn"] = normalized_sub_prompts_cn
    return repaired


def rewrite_single_entry(
    client: OpenAI,
    entry: Dict[str, Any],
    *,
    model: str,
) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(entry)},
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
        print(f"❌ JSON parse failed for {entry['id']}: {error}")
        print(answer_content)
        return None

    return repair_result_schema(result, entry=entry)


def rewrite_entries(
    client: OpenAI,
    entries: List[Dict[str, Any]],
    *,
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{total}] {entry['id']} | {entry['image_name']} | {entry['num_images']} prompts")
        rewritten = rewrite_single_entry(client, entry, model=model)
        if rewritten is not None:
            results.append(rewritten)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite evaluation CSV prompts into same-person multi-image prompt JSON.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help="Path to EvaluationIMGs_stars_prompts.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON file path.",
    )
    parser.add_argument("--model", type=str, default="qwen3.6-plus", help="Qwen model name.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed used for grouping prompts.")
    parser.add_argument("--min-group-size", type=int, default=2, help="Minimum prompts per group.")
    parser.add_argument("--max-group-size", type=int, default=10, help="Maximum prompts per group.")
    parser.add_argument("--limit-people", type=int, default=None, help="Only process the first N people.")
    parser.add_argument("--limit-groups", type=int, default=None, help="Only process the first N grouped entries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_group_size < 2:
        raise SystemExit("错误: --min-group-size 不能小于 2。")
    if args.max_group_size < args.min_group_size:
        raise SystemExit("错误: --max-group-size 不能小于 --min-group-size。")
    if args.max_group_size > 10:
        raise SystemExit("错误: --max-group-size 不能大于 10。")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("错误: 找不到 DASHSCOPE_API_KEY 环境变量。")

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"错误: CSV 文件不存在: {csv_path}")

    grouped_prompts = load_csv_groups(csv_path)
    rng = random.Random(args.seed)
    entries = build_grouped_entries(
        grouped_prompts,
        rng=rng,
        min_group_size=args.min_group_size,
        max_group_size=args.max_group_size,
        limit_people=args.limit_people,
        limit_groups=args.limit_groups,
    )

    print(f"Loaded {len(grouped_prompts)} people from CSV.")
    print(f"Built {len(entries)} grouped prompt entries.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    payload = rewrite_entries(client, entries, model=args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(payload)} rewritten prompts to {output_path}")


if __name__ == "__main__":
    main()
