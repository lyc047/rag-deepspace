#!/usr/bin/env python
"""Build a mobile-friendly code-and-document research handoff bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import date
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAMP = "20260717"
OUTPUT_DIR = PROJECT_DIR / f"mobile_handoff_{STAMP}"

ROOT_FILES = (
    ".gitignore",
    ".python-version",
    "pyproject.toml",
    "requirements-lock.txt",
    "README_CODE_START.md",
    "RESEARCH_PLAN_UAV_SPECTRUM_SEMCOM.md",
    "MOBILE_HANDOFF_README.md",
    "MOBILE_GPT_ANALYSIS_PROMPT.md",
)

SOURCE_DIRS = ("src", "scripts", "configs", "tests", "docs")
KEY_RESULT_PREFIX = "results/stage4/paper_assets_v1/"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", "data"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak", ".pem", ".key"}
SENSITIVE_PATH = re.compile(r"(^|[._-])(secret|token|credential|private[_-]?key|id_rsa)([._-]|$)", re.I)
SENSITIVE_TEXT = re.compile(r"OPENAI_API_KEY\s*=|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|\bsk-[A-Za-z0-9_-]{20,}")
TEXT_SUFFIXES = {".py", ".json", ".toml", ".md", ".txt", ".yaml", ".yml", ".csv"}
CODEBOOK_SUFFIXES = {".py", ".json", ".toml", ".md", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_DIR).as_posix()


def allowed(path: Path) -> bool:
    rel = path.relative_to(PROJECT_DIR)
    if any(part in EXCLUDED_PARTS or part.startswith("mobile_handoff_") for part in rel.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if SENSITIVE_PATH.search(path.name):
        return False
    return path.is_file()


def collect(directories: tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()
    for name in ROOT_FILES:
        path = PROJECT_DIR / name
        if allowed(path):
            paths.add(path)
    for directory in directories:
        for path in (PROJECT_DIR / directory).rglob("*"):
            if allowed(path):
                paths.add(path)
    return sorted(paths, key=relative)


def scan_sensitive(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 4 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if SENSITIVE_TEXT.search(text):
            findings.append(relative(path))
    return findings


def code_fence(path: Path) -> str:
    return {
        ".py": "python",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }.get(path.suffix.lower(), "text")


def build_codebook(paths: list[Path], output: Path) -> None:
    priority_docs = {
        "README_CODE_START.md",
        "RESEARCH_PLAN_UAV_SPECTRUM_SEMCOM.md",
        "MOBILE_HANDOFF_README.md",
        "MOBILE_GPT_ANALYSIS_PROMPT.md",
        "docs/THESIS_STAGE0_SCOPE.md",
        "docs/STAGE4_MASTER_PLAN.md",
        "docs/STAGE4_DEVELOPMENT_SYNTHESIS.md",
        "docs/STAGE4_PAPER_STYLE_OVERVIEW.md",
        "docs/STAGE4_FINAL_HOLDOUT_RUNBOOK.md",
        "docs/STAGE4_FINAL_METRICS_SCHEMA.md",
    }
    selected = [
        path
        for path in paths
        if path.suffix.lower() in CODEBOOK_SUFFIXES
        and (
            relative(path) in priority_docs
            or relative(path).startswith(("src/", "scripts/", "configs/", "tests/"))
        )
    ]
    lines = [
        "# UAV频谱语义通信项目移动端源码Codebook",
        "",
        f"生成日期：{date.today().isoformat()}  ",
        f"收录文本文件：{len(selected)}  ",
        "用途：无需解压ZIP即可在ChatGPT手机端搜索和分析项目代码。",
        "",
        "> 证据边界：开发集结论不等于final；C1保留，C2/C3为负结果；final访问计数为0。",
        "",
        "## 文件目录",
        "",
    ]
    lines.extend(f"- `{relative(path)}`" for path in selected)
    for path in selected:
        rel = relative(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        fence = code_fence(path)
        lines.extend(["", f"## FILE: `{rel}`", "", f"```{fence}", text.rstrip(), "```"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_records(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.append(
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "in_code_docs_bundle": True,
            }
        )
    return rows


def write_zip(output: Path, paths: list[Path], manifest_bytes: bytes) -> dict[str, object]:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        for path in paths:
            archive.write(path, f"uav_spectrum_semcom_project/{relative(path)}")
        archive.writestr("uav_spectrum_semcom_project/handoff/HANDOFF_MANIFEST.json", manifest_bytes)
    with zipfile.ZipFile(output, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC validation failed: {output.name}: {bad}")
        members = len(archive.infolist())
    return {
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "members": members,
        "crc_check": "passed",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("MOBILE_HANDOFF_README.md", "MOBILE_GPT_ANALYSIS_PROMPT.md"):
        shutil.copy2(PROJECT_DIR / name, OUTPUT_DIR / name)

    source_paths = collect(SOURCE_DIRS)
    key_assets = [path for path in (PROJECT_DIR / "results/stage4/paper_assets_v1").rglob("*") if allowed(path)]
    source_paths = sorted(set(source_paths + key_assets), key=relative)
    findings = scan_sensitive(source_paths)
    if findings:
        raise RuntimeError("sensitive-looking content found in: " + ", ".join(findings))

    codebook_path = OUTPUT_DIR / "UAV_MOBILE_SOURCE_CODEBOOK.md"
    build_codebook(source_paths, codebook_path)

    records = manifest_records(source_paths)
    manifest = {
        "handoff_id": f"uav_spectrum_semcom_mobile_handoff_{STAMP}",
        "created_date": date.today().isoformat(),
        "project": "uav_spectrum_semcom_project",
        "scope": "source, scripts, configs, tests, research documents, and lightweight paper assets",
        "exclusions": ["data/", "results/ except stage4 paper assets", "model checkpoints", "experiment caches", "__pycache__/", ".pytest_cache/", ".git/", "*.pyc", "*.pem", "*.key", "mobile_handoff_*/"],
        "research_status": {
            "tests_passed": 149,
            "development_acceptance": True,
            "final_access_count": 0,
            "final_independent_scenes": 0,
            "final_required_scenes": 200,
            "C1": "retained_on_development_Gate_A",
            "C2": "rejected_by_Gate_B_negative_result",
            "C3": "rejected_by_Gate_C_negative_result",
        },
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_path = OUTPUT_DIR / "HANDOFF_MANIFEST.json"
    manifest_path.write_bytes(manifest_bytes)

    csv_path = OUTPUT_DIR / "HANDOFF_MANIFEST.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    bundle_specs = {
        f"uav_spectrum_semcom_code_docs_{STAMP}.zip": source_paths,
    }
    bundles = [write_zip(OUTPUT_DIR / name, paths, manifest_bytes) for name, paths in bundle_specs.items()]

    hashes = [f"{item['sha256']}  {item['file']}" for item in bundles]
    hashes.extend(
        [
            f"{sha256(codebook_path)}  {codebook_path.name}",
            f"{sha256(manifest_path)}  {manifest_path.name}",
            f"{sha256(csv_path)}  {csv_path.name}",
        ]
    )
    (OUTPUT_DIR / "HANDOFF_SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    summary = {
        "output_directory": str(OUTPUT_DIR),
        "bundles": bundles,
        "codebook": {
            "file": codebook_path.name,
            "bytes": codebook_path.stat().st_size,
            "sha256": sha256(codebook_path),
        },
        "manifest_file_count": len(records),
        "excluded": ["third-party data", "model checkpoints", "experiment caches and binary arrays"],
        "sensitive_content_scan": "passed",
    }
    (OUTPUT_DIR / "HANDOFF_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
