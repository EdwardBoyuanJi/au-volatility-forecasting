#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "doc_assets"
OUT.mkdir(parents=True, exist_ok=True)
FONT_REGULAR = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"

NAVY = "#162738"
BLUE = "#2C5F7C"
GOLD = "#C79A3B"
GREEN = "#25836B"
RED = "#B94A48"
INK = "#23303B"
MUTED = "#6E7B86"
GRID = "#D9E0E5"
PAPER = "#F7F9FA"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def title(draw: ImageDraw.ImageDraw, text: str, subtitle: str | None = None) -> None:
    draw.text((80, 45), text, fill=NAVY, font=font(42, True))
    if subtitle:
        draw.text((80, 105), subtitle, fill=MUTED, font=font(24))


def save(image: Image.Image, name: str) -> None:
    image.save(OUT / name, format="PNG", optimize=True)


def equity_drawdown() -> None:
    path = ROOT / "data/outputs/final_strategy_10pct_2d/daily.csv"
    data = pd.read_csv(path)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    image = Image.new("RGB", (1800, 1050), PAPER)
    draw = ImageDraw.Draw(image)
    title(
        draw,
        "最终策略累计权益与回撤",
        "10%风险预算 · 每2个交易日Delta对冲 · 真实期货买一/卖一",
    )
    left, right = 110, 1710
    top1, bottom1 = 180, 650
    top2, bottom2 = 760, 980
    dates = data["trade_date"]
    x_values = [
        left + int((value - dates.min()).days / max((dates.max() - dates.min()).days, 1) * (right - left))
        for value in dates
    ]

    def draw_panel(top: int, bottom: int, values, color: str, formatter) -> None:
        lo, hi = float(values.min()), float(values.max())
        if math.isclose(lo, hi):
            hi = lo + 1.0
        for i in range(5):
            y = top + int(i * (bottom - top) / 4)
            draw.line((left, y, right, y), fill=GRID, width=2)
            value = hi - i * (hi - lo) / 4
            draw.text((20, y - 14), formatter(value), fill=MUTED, font=font(21))
        points = [
            (x, bottom - int((float(v) - lo) / (hi - lo) * (bottom - top)))
            for x, v in zip(x_values, values)
        ]
        draw.line(points, fill=color, width=5, joint="curve")

    draw_panel(top1, bottom1, data["equity"], BLUE, lambda v: f"{v/1e6:.2f}m")
    draw.text((left, top1 - 45), "账户权益（人民币）", fill=INK, font=font(25, True))
    draw_panel(top2, bottom2, data["drawdown"] * 100, RED, lambda v: f"{v:.1f}%")
    draw.text((left, top2 - 45), "回撤", fill=INK, font=font(25, True))
    for year in range(dates.min().year, dates.max().year + 1):
        dt = pd.Timestamp(year=year, month=1, day=1)
        if dt < dates.min() or dt > dates.max():
            continue
        x = left + int((dt - dates.min()).days / max((dates.max() - dates.min()).days, 1) * (right - left))
        draw.text((x - 28, 995), str(year), fill=MUTED, font=font(20))
    save(image, "final_equity_drawdown.png")


def trade_pnl() -> None:
    path = ROOT / "data/outputs/final_strategy_10pct_2d/trades.csv"
    data = pd.read_csv(path).sort_values("entry_date")
    image = Image.new("RGB", (1800, 950), PAPER)
    draw = ImageDraw.Draw(image)
    title(draw, "最终策略逐笔净盈亏", "12笔已到期交易；绿色为盈利，红色为亏损")
    left, right, top, bottom = 120, 1730, 190, 760
    values = data["net_pnl"].astype(float).tolist()
    max_abs = max(abs(min(values)), abs(max(values))) * 1.12
    zero = int((top + bottom) / 2)
    draw.line((left, zero, right, zero), fill=INK, width=3)
    width = (right - left) / len(values)
    for i, (row, value) in enumerate(zip(data.itertuples(index=False), values)):
        x0 = left + i * width + 10
        x1 = left + (i + 1) * width - 10
        height = abs(value) / max_abs * (bottom - top) / 2
        y0, y1 = (zero - height, zero) if value >= 0 else (zero, zero + height)
        color = GREEN if value >= 0 else RED
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=color)
        label = f"{value/1000:.0f}k"
        draw.text((x0 + 2, y0 - 30 if value >= 0 else y1 + 5), label, fill=color, font=font(18, True))
        date = str(row.entry_date)[2:7]
        draw.text((x0, bottom + 35), date, fill=MUTED, font=font(17))
        draw.text((x0 + 12, bottom + 70), f"{int(row.horizon)}d", fill=INK, font=font(17, True))
    draw.text((20, zero - 18), "0", fill=MUTED, font=font(20))
    draw.text((80, 865), "注：各交易受真实盘口容量约束，手数并不相同。", fill=MUTED, font=font(20))
    save(image, "final_trade_pnl.png")


def grid_heatmap() -> None:
    path = ROOT / "data/outputs/strategy_v8_narrow_risk_hedge_grid/grid_sharpe_matrix.csv"
    data = pd.read_csv(path)
    columns = ["hedge_every_1d", "hedge_every_2d", "hedge_every_3d"]
    values = data[columns].to_numpy(dtype=float)
    lo, hi = float(values.min()), float(values.max())
    image = Image.new("RGB", (1500, 980), PAPER)
    draw = ImageDraw.Draw(image)
    title(draw, "风险预算 × 对冲频率：Sharpe网格", "18组控制变量实验；颜色越深表示风险调整收益越高")
    x0, y0 = 330, 230
    cw, ch = 330, 100
    for j, text in enumerate(["每日", "每2个交易日", "每3个交易日"]):
        draw.text((x0 + j * cw + 55, y0 - 60), text, fill=INK, font=font(25, True))
    for i, row in data.iterrows():
        draw.text((95, y0 + i * ch + 32), f"风险预算 {row['risk_budget_percent']:g}%", fill=INK, font=font(25, True))
        for j, column in enumerate(columns):
            value = float(row[column])
            t = (value - lo) / (hi - lo)
            base = (231, 238, 241)
            target = (39, 104, 91)
            color = tuple(int(base[k] + t * (target[k] - base[k])) for k in range(3))
            box = (x0 + j * cw, y0 + i * ch, x0 + (j + 1) * cw - 12, y0 + (i + 1) * ch - 12)
            draw.rounded_rectangle(box, radius=16, fill=color)
            text_color = WHITE if t > 0.50 else INK
            draw.text((box[0] + 110, box[1] + 25), f"{value:.3f}", fill=text_color, font=font(30, True))
    draw.rounded_rectangle((x0 + cw, y0 + 2 * ch, x0 + 2 * cw - 12, y0 + 3 * ch - 12), radius=16, outline=GOLD, width=8)
    draw.text((1050, 865), "最高Sharpe：10% / 每2日", fill=GOLD, font=font(28, True))
    save(image, "grid_sharpe_heatmap.png")


def model_qlike() -> None:
    path = ROOT / "data/outputs/control_experiment_metrics.csv"
    data = pd.read_csv(path)
    data = data[data["scope"].eq("all")]
    models = [
        ("主模型", "har__mse_log__macro+gvz+slv_iv+us_epu", BLUE, 14),
        ("无IV稳健模型", "har__qlike__macro+us_epu", GREEN, -34),
        ("GARCH基准", "garch__raw__none", GOLD, 18),
        ("Persistence", "persistence__raw__none", MUTED, -30),
    ]
    image = Image.new("RGB", (1700, 980), PAPER)
    draw = ImageDraw.Draw(image)
    title(draw, "模型QLIKE相对Persistence", "全样本描述；数值越低越好，100%代表Persistence")
    left, right, top, bottom = 160, 1610, 220, 760
    horizons = [5, 20, 40]
    y_max = 125.0
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((55, y - 12), f"{i * y_max / 5:.0f}%", fill=MUTED, font=font(20))
    x_positions = [350, 850, 1350]
    for x, horizon in zip(x_positions, horizons):
        draw.text((x - 35, bottom + 35), f"{horizon}日", fill=INK, font=font(24, True))
    for label, model, color, label_offset in models:
        ratios = []
        for horizon in horizons:
            q = float(data[(data.model_name == model) & (data.horizon == horizon)].iloc[0].qlike)
            p = float(data[(data.model_name == "persistence__raw__none") & (data.horizon == horizon)].iloc[0].qlike)
            ratios.append(q / p * 100)
        points = []
        for x, ratio in zip(x_positions, ratios):
            y = bottom - ratio / y_max * (bottom - top)
            points.append((x, y))
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)
            draw.text((x + 16, y + label_offset), f"{ratio:.1f}%", fill=color, font=font(19, True))
        draw.line(points, fill=color, width=5)
    for i, (label, _, color, _) in enumerate(models):
        x = 220 + i * 350
        draw.rounded_rectangle((x, 870, x + 28, 898), radius=8, fill=color)
        draw.text((x + 42, 865), label, fill=INK, font=font(21))
    save(image, "model_qlike_comparison.png")


def research_timeline() -> None:
    stages = [
        ("01", "问题定义", "预测5/20/40日RV\n再交易IV-RV错价"),
        ("02", "模型工程", "Ridge-HAR-X\nQLIKE与HARQ"),
        ("03", "变量实验", "HAR/HARQ/GARCH\n256子集与IV对照"),
        ("04", "经济检验", "VRP校准\nATM短跨+Delta对冲"),
        ("05", "现实成交", "期权真实Bid/Ask\n期货真实Top-book"),
        ("06", "策略筛选", "窄盘口≤25%\n容量与风险预算"),
        ("07", "最终冻结", "10%风险预算\n每2日对冲"),
    ]
    image = Image.new("RGB", (1900, 760), PAPER)
    draw = ImageDraw.Draw(image)
    title(draw, "项目研究演进路线", "每一步都保留旧版本，失败实验同样进入结论")
    y = 360
    draw.line((120, y, 1780, y), fill=GRID, width=8)
    gap = 1660 / (len(stages) - 1)
    for i, (num, name, desc) in enumerate(stages):
        x = 120 + i * gap
        draw.ellipse((x - 32, y - 32, x + 32, y + 32), fill=GOLD if i == len(stages)-1 else BLUE)
        draw.text((x - 19, y - 19), num, fill=WHITE, font=font(20, True))
        top = 210 if i % 2 == 0 else 435
        draw.rounded_rectangle((x - 105, top, x + 105, top + 135), radius=18, fill=WHITE, outline=GRID, width=2)
        draw.text((x - 85, top + 16), name, fill=NAVY, font=font(24, True))
        draw.multiline_text((x - 85, top + 55), desc, fill=MUTED, font=font(18), spacing=6)
    save(image, "research_timeline.png")


def main() -> int:
    equity_drawdown()
    trade_pnl()
    grid_heatmap()
    model_qlike()
    research_timeline()
    for path in sorted(OUT.glob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
