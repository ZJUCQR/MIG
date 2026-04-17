import json
from pathlib import Path

from evaluation.evaluate_multi_image import (
    PromptSpec,
    build_image_rows,
    group_status,
    image_sort_key,
    list_group_images,
    load_prompt_specs,
    parse_requested_metrics,
    prompt_text_for_image,
)


def test_parse_requested_metrics_expands_clip_t_alias() -> None:
    metrics = parse_requested_metrics("clip_t,clip_i_anchor,clip_t_global")
    assert metrics == ("clip_t_sub", "clip_t_global", "clip_i_anchor")


def test_image_sort_key_orders_numeric_names() -> None:
    names = [Path("10.png"), Path("2.png"), Path("1.png")]
    ordered = sorted(names, key=image_sort_key)
    assert [path.name for path in ordered] == ["1.png", "2.png", "10.png"]


def test_prompt_text_for_image_prefers_sub_prompt_then_global_then_prompt() -> None:
    with_sub = PromptSpec(
        prompt_id="p1",
        prompt="fallback prompt",
        global_prompt="global prompt",
        sub_prompts=("first prompt", "second prompt"),
        expected_images=2,
    )
    assert prompt_text_for_image(with_sub, 0) == "first prompt"
    assert prompt_text_for_image(with_sub, 1) == "second prompt"
    assert prompt_text_for_image(with_sub, 2) == "global prompt"

    no_sub = PromptSpec(
        prompt_id="p2",
        prompt="fallback prompt",
        global_prompt="",
        sub_prompts=(),
        expected_images=1,
    )
    assert prompt_text_for_image(no_sub, 0) == "fallback prompt"


def test_load_prompt_specs_and_group_files(tmp_path: Path) -> None:
    prompt_file = tmp_path / "multi_image_prompts.json"
    prompt_file.write_text(
        json.dumps(
            [
                {
                    "id": "group_1",
                    "prompt": "full prompt",
                    "global_prompt": "global prompt",
                    "sub_prompts": ["prompt one", "prompt two"],
                    "num_images": 2,
                }
            ]
        ),
        encoding="utf-8",
    )

    specs = load_prompt_specs(prompt_file)
    assert len(specs) == 1
    assert specs[0].prompt_id == "group_1"
    assert specs[0].sub_prompts == ("prompt one", "prompt two")

    group_dir = tmp_path / "group_1"
    group_dir.mkdir()
    for name in ("2.png", "10.png", "1.png"):
        (group_dir / name).write_bytes(b"test")

    image_paths = list_group_images(group_dir)
    assert [path.name for path in image_paths] == ["1.png", "2.png", "10.png"]
    assert group_status(expected_images=3, found_images=len(image_paths)) == "ok"

    rows = build_image_rows(specs[0], image_paths, anchor_position=0)
    assert rows[0]["is_anchor"] is True
    assert rows[1]["sub_prompt_used"] == "prompt two"
    assert rows[2]["sub_prompt_used"] == "global prompt"
