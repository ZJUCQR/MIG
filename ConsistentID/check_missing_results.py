import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check generated result folders against prompt JSON and write failed entries to error.json."
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing generated result folders, usually one folder per prompt id.",
    )
    parser.add_argument(
        "--prompt-json",
        type=str,
        required=True,
        help="Path to the prompt JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="error.json",
        help="Path to write failed prompt entries.",
    )
    return parser.parse_args()


def load_prompt_items(prompt_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(prompt_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {prompt_json}")
    return data


def expected_image_count(item: Dict[str, Any]) -> int:
    num_images = item.get("num_images", 0)
    try:
        value = int(num_images)
    except Exception:
        value = len(item.get("sub_prompts", []))
    return max(1, value)


def count_generated_images(result_dir: Path) -> int:
    if not result_dir.exists() or not result_dir.is_dir():
        return 0
    return sum(1 for path in result_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def collect_failed_items(video_dir: Path, prompt_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failed_items: List[Dict[str, Any]] = []

    for item in prompt_items:
        prompt_id = str(item.get("id", "")).strip()
        if not prompt_id:
            failed_items.append(item)
            continue

        result_dir = video_dir / prompt_id
        generated_count = count_generated_images(result_dir)
        required_count = expected_image_count(item)

        if generated_count < required_count:
            failed_items.append(item)

    return failed_items


def main() -> None:
    args = parse_args()

    video_dir = Path(args.video_dir)
    prompt_json = Path(args.prompt_json)
    output_path = Path(args.output)

    if not prompt_json.exists():
        raise SystemExit(f"Prompt JSON not found: {prompt_json}")
    if not video_dir.exists():
        raise SystemExit(f"Video/result directory not found: {video_dir}")

    prompt_items = load_prompt_items(prompt_json)
    failed_items = collect_failed_items(video_dir, prompt_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(failed_items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total prompts: {len(prompt_items)}")
    print(f"Failed prompts: {len(failed_items)}")
    print(f"Saved failed entries to: {output_path}")


if __name__ == "__main__":
    main()
