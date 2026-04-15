import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

PROMPT_FILE = Path("prompts") / "VBench2_full_text_info.json"
OUTPUT_ROOT = Path("wan_t2v_videos")
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def normalize_dimension(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else "unknown"
    return str(value)



def norm_line(text: str) -> str:
    return " ".join(text.strip().split())



def load_prompt_entries(prompt_file: Path, dimensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    data = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {prompt_file}")

    allowed = set(dimensions or [])
    entries: List[Dict[str, Any]] = []
    for idx, (prefix, item) in enumerate(data.items(), start=1):
        if not isinstance(item, dict):
            continue
        caption = norm_line(str(item.get("caption", "")))
        if not caption:
            continue
        dimension = normalize_dimension(item.get("dimension", []))
        if allowed and dimension not in allowed:
            continue
        entries.append(
            {
                "index": idx,
                "benchmark_prefix": prefix,
                "caption": caption,
                "dimension": dimension,
                "auxiliary_info": item.get("auxiliary_info", []),
            }
        )
    return entries



def build_output_path(output_root: Path, entry: Dict[str, Any], variant_index: int) -> Path:
    dimension_dir = output_root / entry["dimension"]
    filename = f"{entry['benchmark_prefix']}-{variant_index}.mp4"
    return dimension_dir / filename



def load_wan_pipeline(args: argparse.Namespace) -> WanVideoPipeline:
    torch_dtype = getattr(torch, args.torch_dtype)
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="google/umt5-xxl/"),
    )



def run_single_inference(
    args: argparse.Namespace,
    pipe: WanVideoPipeline,
    entry: Dict[str, Any],
    variant_index: int,
) -> None:
    output_path = build_output_path(Path(args.output_root), entry, variant_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_path.exists():
        print(f"跳过已存在文件: {output_path}")
        return

    seed = args.base_seed + (entry["index"] - 1) * args.variants_per_prompt + variant_index

    print(f"\n==== {entry['dimension']} | #{entry['index']} | variant {variant_index} ====")
    print(f"输出文件: {output_path}")
    print(f"种子: {seed}")

    video = pipe(
        prompt=entry["caption"],
        negative_prompt=args.negative_prompt,
        seed=seed,
        tiled=args.tiled,
    )
    save_video(video, str(output_path), fps=args.fps, quality=args.quality)



def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Batch-generate benchmark-compatible Wan T2V videos.")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(base_dir / PROMPT_FILE),
        help="Prompt metadata JSON file. Default: prompts/VBench2_full_text_info.json",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(base_dir / OUTPUT_ROOT),
        help="Output root for generated videos.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=None,
        help="Optional subset of dimensions to generate.",
    )
    parser.add_argument(
        "--variants-per-prompt",
        type=int,
        default=3,
        help="Number of videos to generate per prompt. Default 3 for vbench_standard evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total prompts for smoke testing.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=123,
        help="Base random seed.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Wan model id used by diffsynth.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used when loading WanVideoPipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device passed to WanVideoPipeline.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt passed to Wan generation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Saved video FPS.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=5,
        help="Saved video quality passed to save_video.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        default=True,
        help="Enable tiled inference. Defaults to true.",
    )
    parser.add_argument(
        "--no-tiled",
        action="store_false",
        dest="tiled",
        help="Disable tiled inference.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already generated videos.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    entries = load_prompt_entries(prompt_file, dimensions=args.dimensions)
    if args.limit is not None:
        entries = entries[: args.limit]

    if not entries:
        raise SystemExit("没有可生成的 prompt。")

    print(f"读取到 {len(entries)} 条 prompt")
    print(f"每条生成 {args.variants_per_prompt} 个视频")
    print(f"输出目录: {args.output_root}")

    pipe = load_wan_pipeline(args)

    for entry in entries:
        for variant_index in range(args.variants_per_prompt):
            run_single_inference(args, pipe, entry, variant_index)

    print("\n✅ Wan T2V 视频生成完成")


if __name__ == "__main__":
    main()
