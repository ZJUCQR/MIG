import argparse
import json
import os
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import dashscope
import requests
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message


dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs_wan_2_7_image_pro"
DEFAULT_MODEL_NAME = "wan2.7-image-pro"
DEFAULT_IMAGE_SIZE = "2K"
DEFAULT_NUM_IMAGES_PER_PROMPT = 4
DEFAULT_API_BATCH_SIZE = 4


def find_default_prompt_file() -> Path:
    candidates = [
        PROJECT_ROOT / "a.e_prompts.txt",
        PROJECT_ROOT / "evaluation" / "a.e_prompts.txt",
        PROJECT_ROOT / "evaluation" / "prompts.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return PROJECT_ROOT / "a.e_prompts.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WAN image generation on the Attend-and-Excite benchmark prompts."
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(find_default_prompt_file()),
        help="Path to prompt file. Supports .txt (one prompt per line) or .json.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory to save outputs in the benchmark-compatible layout.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="DashScope image generation model name.",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default=DEFAULT_IMAGE_SIZE,
        help="Image size passed to DashScope, for example 1024*1024 or 2K.",
    )
    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=DEFAULT_NUM_IMAGES_PER_PROMPT,
        help="Target number of images to generate for each prompt.",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=DEFAULT_API_BATCH_SIZE,
        help="How many images to request in one API call. Use this if the model has a per-call image limit.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this prompt index in the prompt file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N prompts after start-index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate prompts even if the output directory already contains enough images.",
    )
    return parser.parse_args()


def load_prompt_items(prompt_file: Path) -> List[str]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    raw_text = prompt_file.read_text(encoding="utf-8")

    # The official Attend-and-Excite prompt file is named .txt but its contents are JSON.
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        prompts: List[str] = []
        for value in data.values():
            if not isinstance(value, list):
                continue
            for item in value:
                prompt = ""
                if isinstance(item, str):
                    prompt = " ".join(item.split())
                elif isinstance(item, dict):
                    prompt = " ".join(str(item.get("prompt", "")).split())
                if prompt:
                    prompts.append(prompt)
        return prompts

    if isinstance(data, list):
        prompts: List[str] = []
        for item in data:
            if isinstance(item, str):
                prompt = " ".join(item.split())
            elif isinstance(item, dict):
                prompt = " ".join(str(item.get("prompt", "")).split())
            else:
                prompt = ""
            if prompt:
                prompts.append(prompt)
        return prompts

    if prompt_file.suffix.lower() == ".txt":
        prompts = []
        for line in raw_text.splitlines():
            prompt = " ".join(line.strip().split())
            if prompt and not prompt.startswith("#"):
                prompts.append(prompt)
        return prompts

    if prompt_file.suffix.lower() == ".json":
        raise ValueError(f"Unsupported JSON structure in {prompt_file}")

    raise ValueError(f"Unsupported prompt file format: {prompt_file}")


def get_existing_image_count(save_dir: Path) -> int:
    if not save_dir.exists():
        return 0
    image_paths = [
        path
        for path in save_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    return len(image_paths)


def ensure_clean_dir(save_dir: Path) -> None:
    if not save_dir.exists():
        return
    for path in save_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".json"}:
            path.unlink()


def download_image(url: str, filename: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as handle:
        handle.write(response.content)
    print(f"Saved: {filename}")


def parse_duration_seconds(scheduled_time_str: Optional[str], end_time_str: Optional[str]) -> Optional[float]:
    if not scheduled_time_str or not end_time_str:
        return None

    def parse_time(value: str) -> datetime:
        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in value else "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(value, fmt)

    return (parse_time(end_time_str) - parse_time(scheduled_time_str)).total_seconds()


def submit_generation(
    api_key: str,
    prompt_text: str,
    *,
    model_name: str,
    image_size: str,
    batch_size: int,
) -> Tuple[str, List[str], Optional[float]]:
    # DashScope model naming uses wan2.7-image-pro, not wan-2.7-image-pro.
    if model_name == "wan-2.7-image-pro":
        model_name = "wan2.7-image-pro"

    message = Message(role="user", content=[{"text": prompt_text}])

    response = ImageGeneration.async_call(
        model=model_name,
        api_key=api_key,
        messages=[message],
        enable_sequential=True,
        n=batch_size,
        size=image_size,
    )

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Task submission failed: code={getattr(response, 'code', None)} message={getattr(response, 'message', None)}"
        )

    task_id = response.output.task_id
    print(f"Submitted task: {task_id}")

    status = ImageGeneration.wait(task=response, api_key=api_key)
    task_status = getattr(status.output, "task_status", None)
    if task_status != "SUCCEEDED":
        raise RuntimeError(
            f"Task failed: task_id={task_id} status={task_status} "
            f"code={getattr(status, 'code', None)} message={getattr(status, 'message', None)}"
        )

    content_list = status.output.choices[0].message.content
    image_urls = [item.get("image") for item in content_list if item.get("image")]
    duration_seconds = parse_duration_seconds(
        getattr(status.output, "scheduled_time", None),
        getattr(status.output, "end_time", None),
    )
    return task_id, image_urls, duration_seconds


def append_manifest_record(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_selected_prompts(prompts: List[str], start_index: int, limit: Optional[int]) -> Iterable[Tuple[int, str]]:
    selected = prompts[start_index:]
    if limit is not None:
        selected = selected[:limit]
    for offset, prompt in enumerate(selected, start=start_index):
        yield offset, prompt


def run_single_prompt(
    api_key: str,
    prompt_index: int,
    prompt_text: str,
    output_root: Path,
    *,
    model_name: str,
    image_size: str,
    num_images_per_prompt: int,
    api_batch_size: int,
    overwrite: bool,
    manifest_path: Path,
) -> None:
    save_dir = output_root / prompt_text
    existing_count = get_existing_image_count(save_dir)

    if overwrite:
        ensure_clean_dir(save_dir)
        existing_count = 0

    if existing_count >= num_images_per_prompt:
        print(f"Skip [{prompt_index}] {prompt_text}: already has {existing_count} images")
        return

    if not prompt_text:
        print(f"Skip [{prompt_index}]: empty prompt")
        return

    print(f"\n==== Prompt {prompt_index} ====")
    print(f"Prompt: {prompt_text}")
    print(f"Output: {save_dir}")
    print(f"Existing images: {existing_count}")
    print(f"Target images: {num_images_per_prompt}")

    while existing_count < num_images_per_prompt:
        batch_size = min(api_batch_size, num_images_per_prompt - existing_count)
        print(f"Requesting batch of {batch_size} images")

        task_id, image_urls, duration_seconds = submit_generation(
            api_key,
            prompt_text,
            model_name=model_name,
            image_size=image_size,
            batch_size=batch_size,
        )

        if duration_seconds is not None:
            print(f"Model generation time: {duration_seconds:.2f}s")

        if not image_urls:
            raise RuntimeError(f"No image URLs returned for task {task_id}")

        downloaded_files = []
        for image_url in image_urls:
            filename = save_dir / f"{existing_count}.png"
            download_image(image_url, filename)
            downloaded_files.append(str(filename))
            existing_count += 1
            if existing_count >= num_images_per_prompt:
                break

        append_manifest_record(
            manifest_path,
            {
                "prompt_index": prompt_index,
                "prompt": prompt_text,
                "task_id": task_id,
                "requested_batch_size": batch_size,
                "returned_image_count": len(image_urls),
                "downloaded_files": downloaded_files,
                "duration_seconds": duration_seconds,
                "model_name": model_name,
                "image_size": image_size,
            },
        )


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Please set the DASHSCOPE_API_KEY environment variable.")
        return

    prompt_file = Path(args.prompt_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.num_images_per_prompt <= 0:
        raise ValueError("--num-images-per-prompt must be positive")
    if args.api_batch_size <= 0:
        raise ValueError("--api-batch-size must be positive")

    prompts = load_prompt_items(prompt_file)
    manifest_path = output_root / "_manifest.jsonl"

    selected_prompts = list(iter_selected_prompts(prompts, args.start_index, args.limit))

    print(f"Prompt file: {prompt_file}")
    print(f"Loaded prompts: {len(prompts)}")
    print(f"Selected prompts: {len(selected_prompts)}")
    print(f"Output root: {output_root}")
    print(f"Model: {args.model_name}")
    print(f"Image size: {args.image_size}")
    print(f"Images per prompt: {args.num_images_per_prompt}")
    print(f"API batch size: {args.api_batch_size}")

    for prompt_index, prompt_text in selected_prompts:
        try:
            run_single_prompt(
                api_key,
                prompt_index,
                prompt_text,
                output_root,
                model_name=args.model_name,
                image_size=args.image_size,
                num_images_per_prompt=args.num_images_per_prompt,
                api_batch_size=args.api_batch_size,
                overwrite=args.overwrite,
                manifest_path=manifest_path,
            )
        except Exception as error:
            print(f"Prompt [{prompt_index}] failed: {prompt_text}")
            print(f"Error: {error}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
