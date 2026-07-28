"""Freeze a development-only Gate A scene registry from image filenames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.reproducibility import sha256_file, sha256_strings
from spectrum_semcom.stage4_development import make_scene_records, validate_development_registry
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage4_protocol.json")
    parser.add_argument(
        "--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2"
    )
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "configs" / "stage4_development_registry_v1.json")
    args = parser.parse_args()

    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_stage4_protocol(protocol)
    settings = protocol["gate_a_development_registry"]
    selection_path = PROJECT_DIR / settings["selection_result"]
    selection_result = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_task = settings["selected_task"]
    for key in ("sources_per_scene", "n_channels", "demand_channels"):
        if int(selection_result["selection"][key]) != int(selected_task[key]):
            raise ValueError(f"frozen task disagrees with validation selection for {key}")

    splits: dict[str, dict] = {}
    sources = int(selected_task["sources_per_scene"])
    for split_name, allocation in settings["allocations"].items():
        image_dir = args.root / "images" / allocation["source_split"]
        all_frame_ids = [path.stem for path in sorted(image_dir.glob("*.png"))]
        offset = int(allocation["frame_offset"])
        count = int(allocation["scene_count"])
        selected_ids = [
            f"{allocation['source_split']}:{frame_id}"
            for frame_id in all_frame_ids[offset : offset + count * sources]
        ]
        scenes = make_scene_records(
            selected_ids,
            split_name=split_name,
            sources_per_scene=sources,
            scene_count=count,
        )
        scene_ids = [scene["scene_id"] for scene in scenes]
        source_frame_ids = [frame_id for scene in scenes for frame_id in scene["source_frame_ids"]]
        splits[split_name] = {
            "source_split": allocation["source_split"],
            "frame_offset": offset,
            "scene_count": count,
            "scene_ids_sha256": sha256_strings(scene_ids),
            "source_frame_ids_sha256": sha256_strings(source_frame_ids),
            "scenes": scenes,
        }

    registry = {
        "version": "1.0",
        "split_id": "stage4_development_registry_v1",
        "parent_split_id": "stage4_scene_registry_v1",
        "status": "frozen_development_only",
        "protocol_sha256": sha256_file(args.config),
        "selection_result": settings["selection_result"],
        "selection_result_sha256": sha256_file(selection_path),
        "selected_task": selected_task,
        "splits": splits,
        "final_holdout_included": False,
        "final_holdout_accessed": False,
        "claim_boundary": "Development pressure scenes support H2 method development only; unrelated source-frame composition cannot support H3.",
    }
    errors = validate_development_registry(registry)
    if errors:
        raise ValueError("invalid generated development registry: " + "; ".join(errors))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
