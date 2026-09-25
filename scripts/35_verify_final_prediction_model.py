#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.final_model.inference import load_model_bundles, predict_feature_rows
from au_rv.final_model.spec import validate_final_model_settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_package_manifest(root: Path) -> int:
    path = root / "package_manifest.json"
    if not path.exists():
        return 0
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        candidate = root / relative
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise RuntimeError(f"Package manifest mismatch: {relative}")
    return len(manifest["files"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the frozen final prediction-model package.")
    parser.add_argument("--package-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.package_root).expanduser().resolve()
    settings = yaml.safe_load((root / "final_prediction_model_config.yaml").read_text(encoding="utf-8"))
    validate_final_model_settings(settings)
    model_manifest = json.loads((root / settings["outputs"]["manifest_path"]).read_text(encoding="utf-8"))
    for entry in model_manifest["models"]:
        if _sha256(root / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Model hash mismatch: {entry['path']}")
    bundles = load_model_bundles(root / settings["outputs"]["model_dir"])
    robust_features = set().union(*(bundle["feature_order"] for (role, _), bundle in bundles.items() if role == "robust"))
    forbidden = {feature for feature in robust_features if "gvz" in feature or "slv" in feature or "implied" in feature}
    if forbidden:
        raise RuntimeError(f"IV feature leaked into robust model: {sorted(forbidden)}")
    latest_input = pd.read_csv(root / settings["outputs"]["latest_input_path"])
    actual = predict_feature_rows(latest_input, bundles)
    expected = pd.read_csv(root / settings["outputs"]["reference_csv_path"])
    aligned = actual.merge(expected, on=["forecast_date", "horizon", "role"], suffixes=("_actual", "_expected"))
    if len(aligned) != 6:
        raise RuntimeError("Reference prediction alignment did not produce six model-horizon rows.")
    maximum_error = float(np.max(np.abs(aligned["forecast_rv_actual"] - aligned["forecast_rv_expected"])))
    if maximum_error > 1.0e-12:
        raise RuntimeError(f"Reference prediction mismatch: {maximum_error}")
    metrics = pd.read_csv(root / settings["outputs"]["evaluation_path"])
    for horizon in (5, 20, 40):
        scoped = metrics[metrics["horizon"].astype(int).eq(horizon)].set_index("model_name")
        persistence = float(scoped.loc["persistence__raw__none", "qlike"])
        for model_name in (
            "har__mse_log__macro+gvz+slv_iv+us_epu",
            "har__qlike__macro+us_epu",
        ):
            if float(scoped.loc[model_name, "qlike"]) >= persistence:
                raise RuntimeError(f"{model_name} failed to beat persistence at {horizon}d.")
    result = {
        "model_version": settings["project"]["version"],
        "bundles_verified": len(bundles),
        "reference_rows_verified": len(aligned),
        "maximum_reference_rv_error": maximum_error,
        "robust_model_iv_features": 0,
        "package_manifest_files_verified": _verify_package_manifest(root),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
