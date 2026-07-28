from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.research_scope import assert_valid_experiment_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen stage-1 reproducibility baseline and write an auditable record.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage1_baseline.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage1" / "reproducibility_smoke")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    assert_valid_experiment_metadata(config)

    registry_path = PROJECT_DIR / "configs" / "dataset_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("dataset registry is missing; run scripts/freeze_dataset_registry.py first")

    command = [sys.executable, str(PROJECT_DIR / config["command"][0])]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_DIR, env=env, capture_output=True, text=True, check=False)
    duration_s = time.perf_counter() - started

    output_path = PROJECT_DIR / config["output"]
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"baseline failed with code {completed.returncode}: {completed.stderr}")
    result = json.loads(output_path.read_text(encoding="utf-8"))

    semantic_payload = next(item for item in result["payloads"] if item["name"] == "semantic_packet")
    config["metrics"] = {
        "best_tf_iou": result["best_tf_iou"],
        "total_transmitted_bits_per_decision": semantic_payload["bits_per_frame"],
    }
    record = {
        "config": config,
        "config_sha256": sha256_file(args.config),
        "dataset_registry_sha256": sha256_file(registry_path),
        "environment": environment_snapshot(["numpy", "scipy", "matplotlib", "pillow", "torch", "pytest", "torchvision", "ultralytics"]),
        "execution": {
            "command": command,
            "returncode": completed.returncode,
            "duration_s": duration_s,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        "artifacts": {
            "result": output_path.relative_to(PROJECT_DIR).as_posix(),
            "result_sha256": sha256_file(output_path),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.out_dir / "run_record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {record_path}")


if __name__ == "__main__":
    main()
