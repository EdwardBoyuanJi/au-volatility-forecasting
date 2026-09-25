#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.final_strategy import EXPECTED_REFERENCE_METRICS, FINAL_STRATEGY_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path) -> int:
    manifest_path = root / "package_manifest.json"
    if not manifest_path.exists():
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Manifest file missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Manifest hash mismatch: {relative}")
    return len(manifest["files"])


def _verify_results(output_dir: Path, verify_reference: bool) -> dict:
    metrics = pd.read_csv(output_dir / "metrics.csv")
    orders = pd.read_csv(output_dir / "orders.csv")
    futures = pd.read_csv(output_dir / "futures_execution_audit.csv")
    reconciliation = pd.read_csv(output_dir / "order_reconciliation.csv")
    if len(metrics) != 1 or metrics.iloc[0]["strategy"] != FINAL_STRATEGY_NAME:
        raise RuntimeError("Output does not contain exactly the frozen final strategy.")
    if not bool(futures["execution_valid"].all()):
        raise RuntimeError("At least one futures execution failed the top-book audit.")
    if not bool(
        ((futures["side"] != "BUY") | (futures["effective_price"] == futures["ask_price1"])).all()
    ):
        raise RuntimeError("A futures buy did not execute at the real ask1.")
    if not bool(
        ((futures["side"] != "SELL") | (futures["effective_price"] == futures["bid_price1"])).all()
    ):
        raise RuntimeError("A futures sell did not execute at the real bid1.")
    forbidden = futures["price_source"].str.contains(
        "MODELED|DAILY_CLOSE|PLUS_1_TICK|SLIPPAGE", case=False, regex=True
    )
    if bool(forbidden.any()):
        raise RuntimeError("A modeled futures price leaked into the final output.")
    if reconciliation["cost_reconciliation_error"].abs().max() > 1.0e-6:
        raise RuntimeError("Order commissions do not reconcile to trade costs.")
    if verify_reference:
        row = metrics.iloc[0]
        for field, expected in EXPECTED_REFERENCE_METRICS.items():
            if not math.isclose(
                float(row[field]), float(expected), rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise RuntimeError(f"Reference mismatch for {field}.")
    return {
        "strategy": FINAL_STRATEGY_NAME,
        "order_rows": int(len(orders)),
        "futures_orders": int(len(futures)),
        "invalid_futures_orders": int((~futures["execution_valid"]).sum()),
        "net_profit": float(metrics.iloc[0]["net_profit"]),
        "sharpe": float(metrics.iloc[0]["sharpe"]),
        "reference_verified": bool(verify_reference),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the frozen final strategy package.")
    parser.add_argument("--package-root", default=str(ROOT))
    parser.add_argument(
        "--output-dir",
        default="data/outputs/final_strategy_10pct_2d",
    )
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    root = Path(args.package_root).expanduser().resolve()
    manifest_count = _verify_manifest(root)
    result = _verify_results(root / args.output_dir, args.verify_reference)
    result["manifest_files_verified"] = manifest_count
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
