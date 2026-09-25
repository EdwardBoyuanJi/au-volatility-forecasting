#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from au_rv.final_model.inference import load_model_bundles, predict_feature_rows


FEATURE_META = {
    "log_rv_1d": ("HAR", "当日对数 RV"),
    "log_rv_5d": ("HAR", "5 日平均对数 RV"),
    "log_rv_22d": ("HAR", "22 日平均对数 RV"),
    "log_comex_nonoverlap_rv_1d": ("跨市场", "COMEX 非重叠时段对数 RV"),
    "abs_broad_dollar_ret_1d": ("宏观", "美元指数单日绝对收益"),
    "abs_us10y_real_chg_1d": ("宏观", "美国 10 年实际利率单日绝对变化"),
    "abs_usdcny_ret_1d": ("宏观", "美元兑人民币单日绝对收益"),
    "log_gvz_implied_var_daily": ("隐含波动率", "GVZ 对数隐含方差"),
    "gvz_log_change_1d": ("隐含波动率", "GVZ 单日对数变化"),
    "gvz_log_change_5d": ("隐含波动率", "GVZ 5 日对数变化"),
    "log_slv_implied_var_daily": ("隐含波动率", "SLV 对数隐含方差"),
    "slv_iv_log_change_1d": ("隐含波动率", "SLV IV 单日对数变化"),
    "slv_iv_log_change_5d": ("隐含波动率", "SLV IV 5 日对数变化"),
    "log_us_epu": ("政策不确定性", "美国 EPU 对数水平"),
    "us_epu_log_change_5d": ("政策不确定性", "美国 EPU 5 日对数变化"),
}


def _records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].astype(str)
    clean = clean.replace([np.inf, -np.inf], np.nan).where(pd.notna(clean), None)
    return clean.to_dict("records")


class DashboardState:
    def __init__(self, root: Path):
        self.root = root
        self.ui_dir = root / "ui/final_model_dashboard"
        self.bundles = load_model_bundles(root / "models/final_prediction_model")

    def payload(self) -> dict:
        reference = pd.read_csv(self.root / "data/outputs/final_prediction_model/latest_predictions.csv")
        metrics = pd.read_csv(self.root / "data/outputs/final_prediction_model/walk_forward_metrics.csv")
        latest_input = pd.read_csv(self.root / "data/model_inputs/latest_feature_snapshot.csv")
        history_path = self.root / "data/history/walk_forward_forecasts.parquet"
        history_columns = [
            "trade_date", "horizon", "model_name", "forecast_vol_pct",
            "actual_vol_pct", "current_vol_pct",
        ]
        if history_path.exists():
            history = pd.read_parquet(history_path)
            history["trade_date"] = pd.to_datetime(history["trade_date"])
            history = history[history["actual_rv"].notna()].copy()
            keep: list[pd.DataFrame] = []
            for horizon in (5, 20, 40):
                scoped = history[history["horizon"].astype(int).eq(horizon)]
                dates = scoped["trade_date"].drop_duplicates().sort_values().tail(320)
                keep.append(scoped[scoped["trade_date"].isin(dates)])
            history = pd.concat(keep, ignore_index=True)
            annualization = 252.0
            history["forecast_vol_pct"] = 100.0 * np.sqrt(annualization * history["forecast_rv"].astype(float))
            history["actual_vol_pct"] = 100.0 * np.sqrt(annualization * history["actual_rv"].astype(float))
            history["current_vol_pct"] = 100.0 * np.sqrt(annualization * history["rv_at_signal"].astype(float))
            history = history[history_columns]
        else:
            # The public portfolio intentionally excludes derived histories that
            # may be restricted by market-data licenses. Forecasts, metrics, and
            # interactive inference remain fully available.
            history = pd.DataFrame(columns=history_columns)
        manifest = json.loads((self.root / "models/final_prediction_model/model_manifest.json").read_text(encoding="utf-8"))
        features = []
        row = latest_input.iloc[0]
        required = sorted(set(feature for bundle in self.bundles.values() for feature in bundle["feature_order"]))
        for feature in required:
            group, label = FEATURE_META.get(feature, ("其他", feature))
            features.append({"name": feature, "label": label, "group": group, "value": float(row[feature])})
        identity = {
            "trade_date": str(row.get("trade_date", "")),
            "selected_au_contract": str(row.get("selected_au_contract", "")),
            "rv": float(row.get("rv")),
        }
        return {
            "manifest": manifest,
            "reference": _records(reference),
            "metrics": _records(metrics),
            "history": _records(history),
            "input": {"identity": identity, "features": features},
        }

    def predict(self, payload: dict) -> list[dict]:
        features = payload.get("features", payload)
        if not isinstance(features, dict):
            raise ValueError("features must be a JSON object.")
        frame = pd.DataFrame([features])
        return _records(predict_feature_rows(frame, self.bundles))


class Handler(BaseHTTPRequestHandler):
    state: DashboardState

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            try:
                self._json(self.state.payload())
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        relative = "index.html" if path in ("", "/") else path.lstrip("/")
        candidate = (self.state.ui_dir / relative).resolve()
        if self.state.ui_dir.resolve() not in candidate.parents and candidate != self.state.ui_dir.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        mime, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Request body must be between 1 byte and 1 MB.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._json({"predictions": self.state.predict(payload)})
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt, *args):
        print(f"dashboard: {self.address_string()} - {fmt % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local AU volatility model dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1 unless you add authentication.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        print("WARNING: this research dashboard has no authentication; do not expose it to an untrusted network.")
    Handler.state = DashboardState(ROOT)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AU volatility dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
