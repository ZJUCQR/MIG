import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
from diffusers import EulerDiscreteScheduler
from diffusers.utils import load_image

from pipline_StableDiffusionXL_ConsistentID import ConsistentIDStableDiffusionXLPipeline


DEFAULT_BASE_MODEL = "./pretrained_models/sdxl-base"
DEFAULT_CONSISTENTID_CKPT = "./pretrained_models/JackAILab_ConsistentID/ConsistentID_SDXL-v1.bin"
DEFAULT_IMAGE_ENCODER = "./pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEFAULT_FACE_PARSING = "./pretrained_models/JackAILab_ConsistentID/face_parsing.pth"
DEFAULT_INPUT_DIR = "./ref_images"
DEFAULT_OUTPUT_DIR = "./outputs_sdxl"
DEFAULT_PROMPT = "A woman wearing a santa hat"
DEFAULT_NEGATIVE_PROMPT = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a ConsistentID SDXL image from a reference portrait.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base SDXL model path or Hugging Face repo.")
    parser.add_argument(
        "--consistentid_ckpt",
        type=str,
        default=DEFAULT_CONSISTENTID_CKPT,
        help="Path like ./pretrained_models/.../ConsistentID_SDXL-v1.bin or repo/file spec like JackAILab/ConsistentID/ConsistentID_SDXL-v1.bin.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=DEFAULT_IMAGE_ENCODER,
        help="CLIP image encoder path or Hugging Face repo.",
    )
    parser.add_argument(
        "--face_parsing_path",
        type=str,
        default=DEFAULT_FACE_PARSING,
        help="Local path to face_parsing.pth.",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Reference image path. If omitted, the script resolves --star_name from --input_dir or ./examples.",
    )
    parser.add_argument(
        "--input_dir",
        "--ref_dir",
        dest="input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing reference images for --star_name resolution.",
    )
    parser.add_argument(
        "--star_name",
        type=str,
        default="scarlett_johansson",
        help="Reference subject stem for fallback resolution when --input_image is omitted.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Generation prompt.")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used when --output_path is omitted.",
    )
    parser.add_argument("--seed", type=int, default=222, help="Random seed.")
    parser.add_argument("--width", type=int, default=864, help="Generated image width.")
    parser.add_argument("--height", type=int, default=1152, help="Generated image height.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--merge_steps", type=int, default=30, help="ConsistentID merge step.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device. The current repo is intended to run on CUDA.")
    return parser.parse_args()


def ensure_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the current ConsistentID SDXL pipeline, but no CUDA device is available.")
        return torch.float16
    raise RuntimeError("The current ConsistentID SDXL inference path is CUDA-only. Please run with --device cuda.")


def split_ckpt_spec(ckpt_spec: str) -> Tuple[str, str]:
    ckpt_dir = os.path.dirname(ckpt_spec) or "."
    weight_name = os.path.basename(ckpt_spec)
    if not weight_name:
        raise ValueError(f"Invalid checkpoint spec: {ckpt_spec}")
    return ckpt_dir, weight_name


def safe_name(text: str, limit: int = 120) -> str:
    compact = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return compact[:limit] or "value"


def resolve_reference_image(name_or_path: Optional[str], search_dir: str, fallback_examples: bool = True) -> Path:
    if not name_or_path:
        raise FileNotFoundError("No reference image or subject name was provided.")

    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    search_dirs = []
    input_dir = Path(search_dir)
    if input_dir.exists():
        search_dirs.append(input_dir)
    if fallback_examples:
        examples_dir = Path("./examples")
        if examples_dir.exists() and examples_dir not in search_dirs:
            search_dirs.append(examples_dir)

    target_name = candidate.name
    target_stem = candidate.stem if candidate.suffix else candidate.name

    for directory in search_dirs:
        exact_path = directory / target_name
        if exact_path.exists():
            return exact_path

        for extension in SUPPORTED_EXTENSIONS:
            by_stem = directory / f"{target_stem}{extension}"
            if by_stem.exists():
                return by_stem

    searched = ", ".join(str(path) for path in search_dirs) or "<none>"
    raise FileNotFoundError(f"Could not find reference image '{name_or_path}' in: {searched}")


def default_output_path(output_dir: str, base_model: str, reference_image_path: Path, prompt: str) -> Path:
    base_name = safe_name(Path(base_model).name or base_model)
    subject = reference_image_path.stem
    file_name = f"{base_name}__{subject}__{safe_name(prompt)}.png"
    return Path(output_dir) / file_name


def load_pipeline(args: argparse.Namespace) -> ConsistentIDStableDiffusionXLPipeline:
    torch_dtype = ensure_device(args.device)

    face_parsing_path = Path(args.face_parsing_path)
    if not face_parsing_path.exists():
        raise FileNotFoundError(
            f"face_parsing.pth was not found at {face_parsing_path}. Download it locally and pass --face_parsing_path."
        )

    ckpt_dir, weight_name = split_ckpt_spec(args.consistentid_ckpt)

    pipe = ConsistentIDStableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        variant="fp16" if torch_dtype == torch.float16 else None,
    ).to(args.device)

    pipe.load_ConsistentID_model(
        ckpt_dir,
        image_encoder_path=args.image_encoder_path,
        bise_net_cp=str(face_parsing_path),
        subfolder="",
        weight_name=weight_name,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


def main() -> None:
    args = parse_args()
    pipe = load_pipeline(args)

    reference_image_path = resolve_reference_image(args.input_image or args.star_name, args.input_dir)
    output_path = Path(args.output_path) if args.output_path else default_output_path(args.output_dir, args.base_model, reference_image_path, args.prompt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_image = load_image(str(reference_image_path))
    image = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        input_id_images=reference_image,
        input_image_path=str(reference_image_path),
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=args.num_steps,
        start_merge_step=args.merge_steps,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
    ).images[0]

    image.save(output_path)
    print(f"Saved image to: {output_path}")
    print(f"Reference image: {reference_image_path}")
    print(f"Prompt used: {args.prompt}")


if __name__ == "__main__":
    main()
