#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x: float, y: float, value: object, size: int = 14, weight: int = 400,
             anchor: str = "start", fill: str = "#26354a") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{esc(value)}</text>'
    )


def write_svg(path: Path, body: list[str], width: int, height: int) -> None:
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="22" fill="#f6f8fb"/>',
        *body,
        '</svg>',
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def model_chart() -> None:
    rows = read_rows(ROOT / "data/outputs/final_prediction_model/walk_forward_metrics.csv")
    main_name = "har__mse_log__macro+gvz+slv_iv+us_epu"
    robust_name = "har__qlike__macro+us_epu"
    by_key = {(int(r["horizon"]), r["model_name"]): r for r in rows}
    width, height = 1120, 560
    body = [
        svg_text(54, 58, "Out-of-sample volatility forecast performance", 26, 700),
        svg_text(54, 86, "Expanded walk-forward evaluation; higher is better", 14, 400, fill="#657386"),
    ]
    panels = [
        ("OOS R² vs persistence", "oos_r2_vs_persistence", 100.0, "%"),
        ("Direction accuracy", "direction_accuracy", 100.0, "%"),
    ]
    colors = {main_name: "#b37b18", robust_name: "#27796e"}
    labels = {main_name: "Main HAR-X", robust_name: "IV-free robust HAR-X"}
    for panel_index, (title, field, scale, suffix) in enumerate(panels):
        left = 54 + panel_index * 535
        top = 130
        body.append(svg_text(left, top, title, 18, 650))
        for group_index, horizon in enumerate((5, 20, 40)):
            y = top + 62 + group_index * 102
            body.append(svg_text(left, y + 23, f"{horizon}D", 14, 700, fill="#4d5c70"))
            for bar_index, model_name in enumerate((main_name, robust_name)):
                value = float(by_key[(horizon, model_name)][field]) * scale
                yy = y + bar_index * 31
                bar_width = 350 * value / 100.0
                body.append(f'<rect x="{left + 52}" y="{yy}" width="350" height="20" rx="10" fill="#e4e9f0"/>')
                body.append(f'<rect x="{left + 52}" y="{yy}" width="{bar_width:.1f}" height="20" rx="10" fill="{colors[model_name]}"/>')
                body.append(svg_text(left + 416, yy + 15, f"{value:.1f}{suffix}", 13, 700, fill=colors[model_name]))
    legend_y = 506
    for index, model_name in enumerate((main_name, robust_name)):
        x = 54 + index * 200
        body.append(f'<circle cx="{x + 7}" cy="{legend_y}" r="7" fill="{colors[model_name]}"/>')
        body.append(svg_text(x + 22, legend_y + 5, labels[model_name], 13, 600))
    body.append(svg_text(1064, 511, "As of 2026-08-24", 12, 400, "end", "#7b8798"))
    write_svg(ASSETS / "model_performance.svg", body, width, height)


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.5) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def strategy_chart() -> None:
    rows = read_rows(ROOT / "results/final_strategy_daily.csv")
    dates = [date.fromisoformat(r["trade_date"]) for r in rows]
    equities = [float(r["equity"]) for r in rows]
    drawdowns = [100.0 * float(r["drawdown"]) for r in rows]
    width, height = 1120, 610
    left, right = 70, 1060
    top, equity_bottom = 120, 390
    dd_top, bottom = 445, 555
    span = max((dates[-1] - dates[0]).days, 1)
    x = lambda d: left + (right - left) * (d - dates[0]).days / span
    min_eq, max_eq = min(equities), max(equities)
    eq_pad = max((max_eq - min_eq) * 0.15, 10_000)
    min_eq -= eq_pad
    max_eq += eq_pad
    y_eq = lambda v: top + (equity_bottom - top) * (max_eq - v) / (max_eq - min_eq)
    min_dd = min(drawdowns)
    y_dd = lambda v: dd_top + (bottom - dd_top) * (0.0 - v) / max(0.0 - min_dd, 0.01)
    body = [
        svg_text(54, 58, "Final strategy: equity and drawdown", 26, 700),
        svg_text(54, 86, "CNY 10m initial research capital; costs and observed crossing friction included", 14, 400, fill="#657386"),
    ]
    for ratio in (0.0, 0.5, 1.0):
        value = min_eq + (max_eq - min_eq) * ratio
        yy = y_eq(value)
        body.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#dce3ec" stroke-width="1"/>')
        body.append(svg_text(left - 12, yy + 5, f"¥{value / 1_000_000:.2f}m", 12, 500, "end", "#6d7989"))
    body.append(polyline([(x(d), y_eq(v)) for d, v in zip(dates, equities)], "#b37b18", 3.0))
    dd_points = [(x(d), y_dd(v)) for d, v in zip(dates, drawdowns)]
    dd_polygon = " ".join(f"{px:.1f},{py:.1f}" for px, py in dd_points)
    dd_polygon += f" {right},{dd_top} {left},{dd_top}"
    body.append(f'<polygon points="{dd_polygon}" fill="#dbece9"/>')
    body.append(polyline(dd_points, "#27796e", 2.0))
    body.append(svg_text(left - 12, dd_top + 5, "0%", 12, 500, "end", "#6d7989"))
    body.append(svg_text(left - 12, bottom + 5, f"{min_dd:.2f}%", 12, 500, "end", "#6d7989"))
    for year in range(dates[0].year, dates[-1].year + 1):
        marker = date(year, 7, 1)
        if dates[0] <= marker <= dates[-1]:
            xx = x(marker)
            body.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" stroke="#e3e8ef" stroke-width="1"/>')
            body.append(svg_text(xx, 582, str(year), 12, 500, "middle", "#6d7989"))
    final_value = equities[-1]
    body.append(f'<circle cx="{x(dates[-1]):.1f}" cy="{y_eq(final_value):.1f}" r="5" fill="#b37b18"/>')
    body.append(svg_text(right - 8, y_eq(final_value) - 12, f"Net profit ¥{final_value - equities[0]:,.0f}", 14, 700, "end", "#8d6113"))
    body.append(svg_text(left, 420, "Drawdown", 13, 650, fill="#27796e"))
    body.append(svg_text(right, 420, "12 trades · Sharpe 1.085 · Max DD -0.91%", 13, 600, "end", "#4f5d70"))
    write_svg(ASSETS / "strategy_equity.svg", body, width, height)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    model_chart()
    strategy_chart()
    print("Generated README assets in", ASSETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

