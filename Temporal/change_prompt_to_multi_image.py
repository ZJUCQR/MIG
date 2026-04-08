import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI

SUPPORTED_DIMENSIONS = [
    "Complex_Plot",
    "Dynamic_Attribute",
    "Dynamic_Spatial_Relationship",
    "Motion_Order_Understanding",
]


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


def normalize_dimension_value(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else "unknown"
    return str(value)


def norm_line(s: str) -> str:
    return " ".join(s.strip().split())


def load_meta_json(meta_path: Path) -> List[Dict[str, Any]]:
    if not meta_path.exists():
        return []
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {meta_path}")
    return data


def load_prompt_entries(meta_dir: Path, dimensions: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    next_id = 1

    for dimension in dimensions:
        meta_path = meta_dir / f"{dimension}.json"
        meta_items = load_meta_json(meta_path)
        for item in meta_items:
            prompt_en = norm_line(str(item.get("prompt_en", "")))
            entries.append(
                {
                    "id": f"tem_{next_id}",
                    "dimension": normalize_dimension_value(item.get("dimension", dimension)),
                    "prompt_en": prompt_en,
                    "auxiliary_info": item.get("auxiliary_info", None),
                }
            )
            next_id += 1

    return entries


def build_system_prompt() -> str:
    return '''你是一个专业的多图生成 prompt 设计师。

你的任务是把输入的一条时序 prompt 改写成“单次生成多张图片”的 prompt。目标是让模型一次生成多张彼此相关、具有时间推进、状态变化、或剧情连续性的图片。

你需要自己判断最合适的图片数量，图片数量限制在 2 到 12 张之间。

严格输出 JSON，不能输出任何解释、注释、markdown 代码块。

输出格式必须严格为：
{
  "prompt": "英文完整 prompt，必须是单行字符串，不允许出现\\n。结构应为：先写 global prompt，再按顺序列出每个子指令，明确这是一组多图/分镜/sequence 图片。",
  "prompt_cn": "中文完整 prompt，必须是单行字符串，不允许出现\\n",
  "global_prompt": "英文全局场景描述",
  "global_prompt_cn": "中文全局场景描述",
  "num_images": 4,
  "sub_prompts": [
    "English instruction for image 1",
    "English instruction for image 2"
  ],
  "sub_prompts_cn": [
    "第1张图中文子指令",
    "第2张图中文子指令"
  ],
  "auxiliary_info": [
    "rewritten auxiliary item 1",
    "rewritten auxiliary item 2"
  ],
  "auxiliary_info_cn": [
    "中文辅助信息1",
    "中文辅助信息2"
  ]
}

硬性要求：
1. `num_images` 由你根据内容复杂度自行判断，范围 2~12
2. `sub_prompts` 和 `sub_prompts_cn` 的长度必须严格等于 `num_images`
3. `prompt` 的大概结构必须是：先 global_prompt，再依次写每张图的子指令
4. `global_prompt` 是英文，`global_prompt_cn` 是中文
5. `prompt` 是英文完整 prompt，`prompt_cn` 是中文完整 prompt，均不能包含换行符 \n
6. `sub_prompts` 是英文子指令列表，`sub_prompts_cn` 是中文子指令列表
7. `auxiliary_info` 与 `auxiliary_info_cn` 是改写后的辅助信息，应概括每张图要检查的关键时序/关系/状态点，长度可与 `num_images` 相同或更短，但必须是列表
8. 保留原 prompt 的核心主体和变化
'''


def build_user_prompt(entry: Dict[str, Any]) -> str:
    aux = entry.get("auxiliary_info")
    aux_text = ""
    if aux is not None:
        aux_text = f"\n辅助信息(auxiliary_info): {json.dumps(aux, ensure_ascii=False)}"

    return (
        f"请把下面这条 {entry['dimension']} 维度的 prompt 改写成适合单次生成多张图片的多图 prompt。"
        f"\n原始英文 prompt: {entry['prompt_en']}"
        f"{aux_text}"
    )


def repair_result_schema(result_dict: Dict[str, Any], *, entry: Dict[str, Any]) -> Dict[str, Any]:
    result_dict["id"] = entry["id"]
    result_dict["category"] = "Temporal Consistency"
    result_dict["dimension"] = entry["dimension"]

    result_dict.setdefault("prompt", entry["prompt_en"])
    result_dict.setdefault("prompt_cn", "")
    result_dict.setdefault("global_prompt", entry["prompt_en"])
    result_dict.setdefault("global_prompt_cn", "")
    result_dict.setdefault("auxiliary_info", [])
    result_dict.setdefault("auxiliary_info_cn", [])

    num_images = result_dict.get("num_images")
    num_images = int(num_images)
    result_dict["num_images"] = num_images

    sub_prompts = result_dict.get("sub_prompts", [])
    sub_prompts_cn = result_dict.get("sub_prompts_cn", [])
    if not isinstance(sub_prompts, list):
        sub_prompts = []
    if not isinstance(sub_prompts_cn, list):
        sub_prompts_cn = []

    normalized_sub_prompts = []
    normalized_sub_prompts_cn = []
    for i in range(num_images):
        normalized_sub_prompts.append(sub_prompts[i] if i < len(sub_prompts) else "")
        normalized_sub_prompts_cn.append(sub_prompts_cn[i] if i < len(sub_prompts_cn) else "")

    aux = result_dict.get("auxiliary_info", [])
    aux_cn = result_dict.get("auxiliary_info_cn", [])
    if not isinstance(aux, list):
        aux = []
    if not isinstance(aux_cn, list):
        aux_cn = []

    result_dict["prompt"] = " ".join(str(result_dict.get("prompt", "")).split())
    result_dict["prompt_cn"] = " ".join(str(result_dict.get("prompt_cn", "")).split())
    result_dict["global_prompt"] = " ".join(str(result_dict.get("global_prompt", "")).split())
    result_dict["global_prompt_cn"] = " ".join(str(result_dict.get("global_prompt_cn", "")).split())
    result_dict["sub_prompts"] = [" ".join(str(x).split()) for x in normalized_sub_prompts]
    result_dict["sub_prompts_cn"] = [" ".join(str(x).split()) for x in normalized_sub_prompts_cn]
    result_dict["auxiliary_info"] = [" ".join(str(x).split()) for x in aux]
    result_dict["auxiliary_info_cn"] = [" ".join(str(x).split()) for x in aux_cn]
    return result_dict


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
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed for {entry['id']} {entry['dimension']}: {e}")
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
    for idx, entry in enumerate(entries, start=1):
        print(f"[{idx}/{total}] {entry['dimension']} | {entry['prompt_en']}")
        rewritten = rewrite_single_entry(client, entry, model=model)
        if rewritten is not None:
            results.append(rewritten)
    return results


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Rewrite meta_info entries into multi-image prompt JSON.")
    parser.add_argument(
        "--meta-dir",
        type=str,
        default=str(base_dir / "prompts" / "meta_info"),
        help="Directory containing per-dimension meta_info json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "temporal_prompts.json"),
        help="Output JSON file path.",
    )
    parser.add_argument("--model", type=str, default="qwen3.6-plus", help="Model name.")
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=SUPPORTED_DIMENSIONS,
        help="Optional subset of dimensions to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total number of prompts processed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("错误: 找不到 DASHSCOPE_API_KEY 环境变量。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    entries = load_prompt_entries(
        meta_dir=Path(args.meta_dir),
        dimensions=list(args.dimensions),
    )

    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"Loaded {len(entries)} prompts from meta_info.")
    payload = rewrite_entries(client, entries, model=args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(payload)} rewritten prompts to {output_path}")


if __name__ == "__main__":
    main()
