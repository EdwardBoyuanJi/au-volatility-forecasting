#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
STRATEGY_NAME = "au_final_strategy_10pct_2d_20260831"
MODEL_NAME = "au_final_prediction_model_20260831"
DOCS = [
    "沪金期权最终策略说明书_10pct风险预算_每2日对冲_20260831.docx",
    "沪金期权波动率预测项目完整研究报告_20260831.docx",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_destination(path: Path) -> None:
    if path.parent != DELIVERABLES:
        raise RuntimeError(f"Refusing to clean unexpected path: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def _manifest(package: Path, kind: str, version: str) -> Path:
    files: dict[str, str] = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name == "package_manifest.json":
            continue
        relative = path.relative_to(package).as_posix()
        if relative in {
            "data/outputs/final_prediction_model/predictions.csv",
            "data/outputs/final_prediction_model/predictions.json",
        }:
            continue
        files[relative] = _sha256(path)
    payload = {
        "package_kind": kind,
        "version": version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "files": files,
    }
    destination = package / "package_manifest.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _scan_secrets(package: Path) -> None:
    forbidden_names = {".env", ".env.local", ".env.production"}
    found_names = [str(path.relative_to(package)) for path in package.rglob("*") if path.is_file() and path.name in forbidden_names]
    if found_names:
        raise RuntimeError(f"Secret environment files entered package: {found_names}")
    example = package / ".env.example"
    if example.exists():
        for line in example.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name in {"TQ_USER", "TQ_PASSWORD", "DATABENTO_API_KEY", "FRED_API_KEY"} and not value.startswith("replace_with_"):
                raise RuntimeError(f"Non-placeholder credential in .env.example: {name}")


def _zip(package: Path) -> Path:
    destination = DELIVERABLES / f"{package.name}.zip"
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{package.name}/{path.relative_to(package).as_posix()}")
    return destination


def build_strategy_package() -> tuple[Path, Path]:
    package = DELIVERABLES / STRATEGY_NAME
    _clean_destination(package)
    _copy_file(ROOT / "packaging/STRATEGY_README.md", package / "README.md")
    _copy_file(ROOT / "packaging/DATA_AND_LICENSE_NOTICE.md", package / "DATA_AND_LICENSE_NOTICE.md")
    for name in (".env.example", "requirements-lock.txt", "final_strategy_10pct_2d_config.yaml"):
        _copy_file(ROOT / name, package / name)
    for name in ("27_run_final_strategy_10pct_2d.py", "28_verify_final_strategy_package.py"):
        _copy_file(ROOT / "scripts" / name, package / "scripts" / name)
    _copy_tree(ROOT / "src", package / "src")
    input_files = {
        "data/outputs/strategy_v6_5y/opportunities.parquet": "data/outputs/strategy_v6_5y/opportunities.parquet",
        "data/raw/tqsdk_au_option_daily_strategy.parquet": "data/raw/tqsdk_au_option_daily_strategy.parquet",
        "data/raw/tqsdk_au_futures_daily_all_contracts.parquet": "data/raw/tqsdk_au_futures_daily_all_contracts.parquet",
        "data/raw/strategy_v6_real_topbook/futures_hedge_topbook_ticks.parquet": "data/raw/strategy_v6_real_topbook/futures_hedge_topbook_ticks.parquet",
        "data/raw/strategy_v6_real_topbook/futures_hedge_topbook_coverage.csv": "data/raw/strategy_v6_real_topbook/futures_hedge_topbook_coverage.csv",
    }
    for source, destination in input_files.items():
        _copy_file(ROOT / source, package / destination)
    _copy_tree(ROOT / "data/raw/strategy_real_quotes", package / "data/raw/strategy_real_quotes")
    _copy_tree(ROOT / "data/outputs/final_strategy_10pct_2d", package / "reference_results")
    for document in DOCS:
        _copy_file(DELIVERABLES / document, package / "docs" / document)
    _scan_secrets(package)
    _manifest(package, "final_strategy", "final_narrow_rb10_hedge2d_20260830")
    return package, _zip(package)


def build_model_package() -> tuple[Path, Path]:
    package = DELIVERABLES / MODEL_NAME
    _clean_destination(package)
    _copy_file(ROOT / "packaging/MODEL_README.md", package / "README.md")
    _copy_file(ROOT / "packaging/DATA_AND_LICENSE_NOTICE.md", package / "DATA_AND_LICENSE_NOTICE.md")
    for name in (".env.example", "requirements-lock.txt", "final_prediction_model_config.yaml"):
        _copy_file(ROOT / name, package / name)
    for name in (
        "32_train_final_prediction_model.py",
        "33_predict_final_model.py",
        "34_serve_final_model_dashboard.py",
        "35_verify_final_prediction_model.py",
    ):
        _copy_file(ROOT / "scripts" / name, package / "scripts" / name)
    _copy_tree(ROOT / "src", package / "src")
    _copy_tree(ROOT / "models/final_prediction_model", package / "models/final_prediction_model")
    _copy_tree(ROOT / "ui/final_model_dashboard", package / "ui/final_model_dashboard")
    _copy_file(
        ROOT / "data/processed/strategy_v6_5y/model_features_daily.parquet",
        package / "data/processed/strategy_v6_5y/model_features_daily.parquet",
    )
    _copy_file(
        ROOT / "data/history/walk_forward_forecasts.parquet",
        package / "data/history/walk_forward_forecasts.parquet",
    )
    _copy_file(
        ROOT / "data/model_inputs/latest_feature_snapshot.csv",
        package / "data/model_inputs/latest_feature_snapshot.csv",
    )
    output_names = (
        "latest_predictions.csv",
        "latest_predictions.json",
        "walk_forward_metrics.csv",
        "model_coefficients.csv",
    )
    for name in output_names:
        _copy_file(
            ROOT / "data/outputs/final_prediction_model" / name,
            package / "data/outputs/final_prediction_model" / name,
        )
    _copy_file(DELIVERABLES / DOCS[1], package / "docs" / DOCS[1])
    _scan_secrets(package)
    _manifest(package, "final_prediction_model", "final_dual_har_x_20260831")
    return package, _zip(package)


def main() -> int:
    strategy, strategy_zip = build_strategy_package()
    model, model_zip = build_model_package()
    print(f"strategy_package: {strategy}")
    print(f"strategy_zip: {strategy_zip}")
    print(f"model_package: {model}")
    print(f"model_zip: {model_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
