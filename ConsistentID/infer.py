import argparse
import csv
import os
import re
from pathlib import Path
from typing import Optional

import torch
from diffusers import EulerDiscreteScheduler
from diffusers.utils import load_image

from pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline


DEFAULT_BASE_MODEL = "./pretrained_models/Realistic_Vision_V6.0_B1_noVAE"
DEFAULT_CONSISTENTID_CKPT = "./pretrained_models/JackAILab_ConsistentID/ConsistentID-v1.bin"
DEFAULT_IMAGE_ENCODER = "./pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEFAULT_FACE_PARSING = "./pretrained_models/JackAILab_ConsistentID/face_parsing.pth"
DEFAULT_INPUT_DIR = "./ref_images"
DEFAULT_OUTPUT_DIR = "./outputs_sd15"
DEFAULT_PROMPT = "A man, in a forest, adventuring"
DEFAULT_NEGATIVE_PROMPT = (
    "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    ", ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers"
    ", mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed)))"
    ", ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face"
    ", (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions"
    ", (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs)))"
    ", mutated hands, (fused fingers), (too many fingers), (((long neck)))"
)
PROMPT_TEMPLATE = (
    "cinematic photo, {prompt}, 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed"
)
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ConsistentID SD1.5 images for single prompts or CSV-driven evaluation."
    )
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base SD1.5 model path or Hugging Face repo.")
    parser.add_argument(
        "--consistentid_ckpt",
        type=str,
        default=DEFAULT_CONSISTENTID_CKPT,
        help="Path like ./pretrained_models/.../ConsistentID-v1.bin or repo/file spec like JackAILab/ConsistentID/ConsistentID-v1.bin.",
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
        help="Single reference image path. If omitted, the script resolves --star_name from --input_dir or ./examples.",
    )
    parser.add_argument(
        "--input_dir",
        "--ref_dir",
        dest="input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing reference images for --star_name resolution or CSV evaluation.",
    )
    parser.add_argument(
        "--star_name",
        type=str,
        default="albert_einstein",
        help="Reference subject stem for single-image mode. Resolved from --input_dir or ./examples.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt used in single-image mode.")
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default=None,
        help="CSV file with Image_Name and Prompt columns for batch evaluation generation.",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output file path for single-image mode.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for single-image defaults and CSV batch generation.",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Base random seed.")
    parser.add_argument("--width", type=int, default=512, help="Generated image width.")
    parser.add_argument("--height", type=int, default=768, help="Generated image height.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--merge_steps", type=int, default=30, help="ConsistentID merge step.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument(
        "--use_prompt_template",
        action="store_true",
        help="Wrap each prompt with the original cinematic prompt template used by the demo script.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate the first N CSV rows. Useful for smoke tests.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device. The current repo is intended to run on CUDA.")
    args = parser.parse_args()

    if args.prompt_csv and args.output_path:
        parser.error("--output_path can only be used in single-image mode.")

    return args


def ensure_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the current ConsistentID pipeline, but no CUDA device is available.")
        return torch.float16
    raise RuntimeError("The current ConsistentID inference path is CUDA-only. Please run with --device cuda.")


def split_ckpt_spec(ckpt_spec: str) -> tuple[str, str]:
    ckpt_dir = os.path.dirname(ckpt_spec) or "."
    weight_name = os.path.basename(ckpt_spec)
    if not weight_name:
        raise ValueError(f"Invalid checkpoint spec: {ckpt_spec}")
    return ckpt_dir, weight_name


def safe_name(text: str, limit: int = 120) -> str:
    compact = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return compact[:limit] or "prompt"


def apply_prompt_template(prompt: str, use_prompt_template: bool) -> str:
    return PROMPT_TEMPLATE.format(prompt=prompt) if use_prompt_template else prompt


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


def load_pipeline(args: argparse.Namespace) -> ConsistentIDStableDiffusionPipeline:
    torch_dtype = ensure_device(args.device)

    face_parsing_path = Path(args.face_parsing_path)
    if not face_parsing_path.exists():
        raise FileNotFoundError(
            f"face_parsing.pth was not found at {face_parsing_path}. Download it locally and pass --face_parsing_path."
        )

    ckpt_dir, weight_name = split_ckpt_spec(args.consistentid_ckpt)

    pipe = ConsistentIDStableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
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


def make_generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def default_single_output_path(output_dir: str, reference_image_path: Path, prompt: str) -> Path:
    subject = reference_image_path.stem
    return Path(output_dir) / f"{subject}__{safe_name(prompt)}.png"


def generate_image(
    pipe: ConsistentIDStableDiffusionPipeline,
    reference_image_path: Path,
    prompt: str,
    args: argparse.Namespace,
    output_path: Path,
    seed: int,
) -> Path:
    prompt_text = apply_prompt_template(prompt, args.use_prompt_template)
    reference_image = load_image(str(reference_image_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = pipe(
        prompt=prompt_text,
        width=args.width,
        height=args.height,
        input_id_images=reference_image,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=args.num_steps,
        start_merge_step=args.merge_steps,
        generator=make_generator(args.device, seed),
    ).images[0]
    image.save(output_path)
    return output_path


def run_single(pipe: ConsistentIDStableDiffusionPipeline, args: argparse.Namespace) -> None:
    reference_image_path = resolve_reference_image(args.input_image or args.star_name, args.input_dir)
    output_path = Path(args.output_path) if args.output_path else default_single_output_path(args.output_dir, reference_image_path, args.prompt)
    saved_path = generate_image(pipe, reference_image_path, args.prompt, args, output_path, args.seed)
    print(f"Saved image to: {saved_path}")
    print(f"Reference image: {reference_image_path}")
    print(f"Prompt used: {apply_prompt_template(args.prompt, args.use_prompt_template)}")


def run_batch(pipe: ConsistentIDStableDiffusionPipeline, args: argparse.Namespace) -> None:
    csv_path = Path(args.prompt_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generated_count = 0
    skipped_count = 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            if args.limit is not None and row_index > args.limit:
                break

            image_name = row.get("Image_Name")
            prompt = row.get("Prompt")
            if not image_name or not prompt:
                skipped_count += 1
                print(f"[skip] row {row_index}: missing Image_Name or Prompt")
                continue

            try:
                reference_image_path = resolve_reference_image(image_name, args.input_dir)
            except FileNotFoundError as error:
                skipped_count += 1
                print(f"[skip] row {row_index}: {error}")
                continue

            subject = Path(image_name).stem
            subject_dir = output_root / subject
            output_path = subject_dir / f"{row_index:04d}__{safe_name(prompt)}.png"
            row_seed = args.seed + row_index - 1

            saved_path = generate_image(pipe, reference_image_path, prompt, args, output_path, row_seed)
            generated_count += 1
            print(f"[saved] row {row_index}: {saved_path}")

    print(f"Batch generation finished. Generated {generated_count} images, skipped {skipped_count} rows.")


if __name__ == "__main__":
    arguments = parse_args()
    pipeline = load_pipeline(arguments)
    if arguments.prompt_csv:
        run_batch(pipeline, arguments)
    else:
        run_single(pipeline, arguments)
