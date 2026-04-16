import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.multimodal import CLIPScore
from torch_fidelity import calculate_metrics

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_image_tensor(self.image_paths[index])


def load_prompt_items(prompt_file: Path) -> List[Dict[str, Any]]:
    if not prompt_file.exists():
        raise SystemExit(f"未找到 prompt 文件: {prompt_file}")

    data = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {prompt_file}")
    return [item for item in data if isinstance(item, dict)]


def normalize_prompt(item: Dict[str, Any]) -> str:
    return " ".join(str(item.get("prompt", "")).split())


def build_prompt_map(items: List[Dict[str, Any]]) -> Dict[str, str]:
    prompt_by_id: Dict[str, str] = {}
    for index, item in enumerate(items, start=1):
        prompt_id = str(item.get("id") or f"item_{index}")
        prompt = normalize_prompt(item)
        if prompt:
            prompt_by_id[prompt_id] = prompt
    return prompt_by_id


def image_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 0, path.name)


def iter_image_files(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.iterdir(), key=image_sort_key):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def collect_image_prompt_pairs(
    result_root: Path,
    prompt_by_id: Dict[str, str],
    max_images: int | None = None,
) -> Tuple[List[Path], List[str], List[str], List[str], List[str], List[str]]:
    image_paths: List[Path] = []
    prompts: List[str] = []
    image_ids: List[str] = []
    matched_ids: List[str] = []
    missing_prompt_ids: List[str] = []
    missing_image_ids: List[str] = []

    expected_ids = set(prompt_by_id.keys())
    discovered_ids = set()

    if not result_root.exists():
        return image_paths, prompts, image_ids, matched_ids, missing_prompt_ids, sorted(expected_ids)

    for child in sorted(result_root.iterdir()):
        if not child.is_dir():
            continue

        prompt_id = child.name
        discovered_ids.add(prompt_id)
        prompt = prompt_by_id.get(prompt_id)
        if not prompt:
            missing_prompt_ids.append(prompt_id)
            continue

        files = list(iter_image_files(child))
        if not files:
            missing_image_ids.append(prompt_id)
            continue

        matched_ids.append(prompt_id)
        for image_path in files:
            image_paths.append(image_path)
            prompts.append(prompt)
            image_ids.append(prompt_id)
            if max_images is not None and len(image_paths) >= max_images:
                break
        if max_images is not None and len(image_paths) >= max_images:
            break

    unseen_expected_ids = sorted(expected_ids - discovered_ids)
    missing_image_ids.extend(unseen_expected_ids)

    return image_paths, prompts, image_ids, matched_ids, missing_prompt_ids, sorted(set(missing_image_ids))


def select_is_image_paths(image_paths: List[Path], image_ids: List[str], *, per_id_limit: int = 3) -> List[Path]:
    selected: List[Path] = []
    counts: Dict[str, int] = {}
    for image_path, prompt_id in zip(image_paths, image_ids):
        current = counts.get(prompt_id, 0)
        if current >= per_id_limit:
            continue
        selected.append(image_path)
        counts[prompt_id] = current + 1
    return selected


def compute_inception_score(image_paths: List[Path], batch_size: int, *, use_cuda: bool) -> Dict[str, float]:
    dataset = ImagePathDataset(image_paths)
    metrics = calculate_metrics(
        input1=dataset,
        isc=True,
        fid=False,
        kid=False,
        ppl=False,
        batch_size=batch_size,
        samples_shuffle=False,
        cuda=use_cuda,
        verbose=False,
    )
    return {
        "inception_score_mean": float(metrics["inception_score_mean"]),
        "inception_score_std": float(metrics["inception_score_std"]),
    }


def load_image_tensor(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1).contiguous()
    return tensor


def compute_clip_score(
    image_paths: List[Path],
    prompts: List[str],
    image_ids: List[str],
    *,
    device: torch.device,
    clip_model: str,
) -> Dict[str, Any]:
    metric = CLIPScore(model_name_or_path=clip_model).to(device)
    metric.eval()
    scores: List[float] = []
    per_id_scores: Dict[str, List[float]] = {}

    for image_path, prompt, prompt_id in zip(image_paths, prompts, image_ids):
        image_tensor = load_image_tensor(image_path).unsqueeze(0).to(device)
        metric.reset()
        with torch.no_grad():
            score = metric(image_tensor, [prompt])
        score_value = float(score.detach().cpu().item())
        scores.append(score_value)
        per_id_scores.setdefault(prompt_id, []).append(score_value)

    per_id_summary = {}
    reduced_scores: List[float] = []
    for prompt_id, id_scores in sorted(per_id_scores.items()):
        sorted_scores = sorted(id_scores, reverse=True)
        top2_scores = sorted_scores[:2]
        top2_mean = mean(top2_scores) if top2_scores else 0.0
        reduced_scores.append(top2_mean)
        per_id_summary[prompt_id] = {
            "num_images": len(id_scores),
            "clip_score_mean": mean(id_scores),
            "clip_score_std": pstdev(id_scores) if len(id_scores) > 1 else 0.0,
            "clip_score_top2_mean": top2_mean,
        }

    return {
        "clip_score_mean": mean(reduced_scores) if reduced_scores else 0.0,
        "clip_score_std": pstdev(reduced_scores) if len(reduced_scores) > 1 else 0.0,
        "per_id": per_id_summary,
    }


def evaluate_result_root(
    result_root: Path,
    prompt_by_id: Dict[str, str],
    *,
    device: torch.device,
    clip_model: str,
    batch_size: int,
    max_images: int | None,
) -> Dict[str, Any]:
    image_paths, prompts, image_ids, matched_ids, missing_prompt_ids, missing_image_ids = collect_image_prompt_pairs(
        result_root,
        prompt_by_id,
        max_images=max_images,
    )

    if not image_paths:
        return {
            "result_root": str(result_root),
            "num_ids_matched": 0,
            "num_images": 0,
            "num_images_for_is": 0,
            "inception_score_mean": None,
            "inception_score_std": None,
            "clip_score_mean": None,
            "clip_score_std": None,
            "missing_prompt_ids": missing_prompt_ids,
            "missing_image_ids": missing_image_ids,
        }

    is_image_paths = select_is_image_paths(image_paths, image_ids, per_id_limit=3)
    is_metrics = compute_inception_score(
        is_image_paths,
        batch_size=batch_size,
        use_cuda=device.type == "cuda",
    )
    clip_metrics = compute_clip_score(
        image_paths,
        prompts,
        image_ids,
        device=device,
        clip_model=clip_model,
    )

    return {
        "result_root": str(result_root),
        "num_ids_matched": len(set(matched_ids)),
        "num_images": len(image_paths),
        "num_images_for_is": len(is_image_paths),
        "inception_score_mean": is_metrics["inception_score_mean"],
        "inception_score_std": is_metrics["inception_score_std"],
        "clip_score_mean": clip_metrics["clip_score_mean"],
        "clip_score_std": clip_metrics["clip_score_std"],
        "per_id_clip_scores": clip_metrics["per_id"],
        "missing_prompt_ids": missing_prompt_ids,
        "missing_image_ids": missing_image_ids,
    }


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate generated images with IS and CLIP Score.")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(base_dir / "multi_view_prompts.json"),
        help="Prompt JSON generated by generate_prompt.py.",
    )
    parser.add_argument(
        "--result-roots",
        nargs="+",
        default=[str(base_dir / "wan_result")],
        help="One or more result directories to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for CLIP Score computation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for Inception Score computation.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model used by torchmetrics CLIPScore.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(base_dir / "wan_result_metrics.json"),
        help="Path to save the metric report as JSON.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of images evaluated per result root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("错误: 当前环境不可用 CUDA，但 --device 设置为 cuda。")

    prompt_items = load_prompt_items(Path(args.prompt_file))
    prompt_by_id = build_prompt_map(prompt_items)
    if not prompt_by_id:
        raise SystemExit("错误: prompt 文件中没有有效 prompt。")

    report = {
        "prompt_file": str(Path(args.prompt_file)),
        "results": {},
    }

    for result_root_str in args.result_roots:
        result_root = Path(result_root_str)
        print(f"\n=== Evaluating {result_root} ===")
        metrics = evaluate_result_root(
            result_root,
            prompt_by_id,
            device=device,
            clip_model=args.clip_model,
            batch_size=args.batch_size,
            max_images=args.max_images,
        )
        report["results"][result_root.name] = metrics

        print(f"matched ids: {metrics['num_ids_matched']}")
        print(f"images: {metrics['num_images']}")
        print(f"images used for IS: {metrics['num_images_for_is']}")
        if metrics["inception_score_mean"] is not None:
            print(f"IS: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
            print(f"CLIP Score (top-2 per id mean): {metrics['clip_score_mean']:.4f} ± {metrics['clip_score_std']:.4f}")
        else:
            print("No valid images found for this root.")

        if metrics["missing_prompt_ids"]:
            print(f"missing prompts for ids: {metrics['missing_prompt_ids']}")
        if metrics["missing_image_ids"]:
            print(f"missing images for ids: {metrics['missing_image_ids']}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved metric report to {output_path}")


if __name__ == "__main__":
    main()
