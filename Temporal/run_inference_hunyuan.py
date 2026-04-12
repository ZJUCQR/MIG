import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_FILE = Path("prompts") / "VBench2_full_text_info.json"
OUTPUT_ROOT = Path("hunyuan_videos")


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


def build_command(args: argparse.Namespace, entry: Dict[str, Any], output_path: Path, seed: int) -> List[str]:
    command = [
        args.torchrun_bin,
        f"--nproc_per_node={args.nproc_per_node}",
        "generate.py",
        "--prompt",
        entry["caption"],
        "--image_path",
        "none",
        "--resolution",
        args.resolution,
        "--aspect_ratio",
        args.aspect_ratio,
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--video_length",
        str(args.video_length),
        "--rewrite",
        "false",
        "--output_path",
        str(output_path),
        "--model_path",
        args.model_path,
        "--seed",
        str(seed),
    ]
    if args.dtype:
        command.extend(["--dtype", args.dtype])
    if args.disable_sr:
        command.extend(["--sr", "false"])
    if args.cfg_distilled:
        command.extend(["--cfg_distilled", "true"])
    return command


def run_single_inference(args: argparse.Namespace, entry: Dict[str, Any], variant_index: int) -> None:
    output_path = build_output_path(Path(args.output_root), entry, variant_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_path.exists():
        print(f"跳过已存在文件: {output_path}")
        return

    seed = args.base_seed + (entry["index"] - 1) * args.variants_per_prompt + variant_index
    command = build_command(args, entry, output_path, seed)

    print(f"\n==== {entry['dimension']} | #{entry['index']} | variant {variant_index} ====")
    print(f"输出文件: {output_path}")
    print(f"种子: {seed}")

    subprocess.run(command, cwd=args.hunyuan_root, check=True)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Batch-generate benchmark-compatible Hunyuan videos.")
    parser.add_argument(
        "--hunyuan-root",
        type=str,
        required=True,
        help="Path to the HunyuanVideo-1.5 repository containing generate.py",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to HunyuanVideo checkpoints directory (for example ./ckpts)",
    )
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
        "--resolution",
        type=str,
        default="720p",
        help="Hunyuan output resolution, e.g. 480p or 720p.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="16:9",
        help="Video aspect ratio.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=121,
        help="Number of frames to generate.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of GPUs passed to torchrun.",
    )
    parser.add_argument(
        "--torchrun-bin",
        type=str,
        default="torchrun",
        help="torchrun executable name or absolute path.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="Transformer dtype passed to Hunyuan generate.py",
    )
    parser.add_argument(
        "--disable-sr",
        action="store_true",
        help="Pass --sr false to disable super-resolution during generation.",
    )
    parser.add_argument(
        "--cfg-distilled",
        action="store_true",
        help="Enable CFG distilled mode if your checkpoint supports it.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already generated videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hunyuan_root = Path(args.hunyuan_root)
    generate_py = hunyuan_root / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(f"generate.py not found under Hunyuan root: {generate_py}")

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

    for entry in entries:
        for variant_index in range(args.variants_per_prompt):
            run_single_inference(args, entry, variant_index)

    print("\n✅ Hunyuan 视频生成完成")


if __name__ == "__main__":
    main()
