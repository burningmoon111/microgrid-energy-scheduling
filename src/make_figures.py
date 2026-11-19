"""使用真实求解结果生成论文主图。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import solve_c as m


OUT = Path(os.environ.get("CUMCM_FIG_OUT", str(m.PROJECT / "论文" / "附图")))
COLORS = {"blue": "#0072B2", "green": "#009E73", "orange": "#E69F00", "red": "#D55E00", "gray": "#7A8793", "purple": "#75638B"}


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.sans-serif": ["Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"],
        "axes.unicode_minus": False, "font.size": 9.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.55,
        "axes.axisbelow": True,
    })


def q1_dispatch(fixed_price, load, pv):
    d = np.load(m.CACHE / "q1.npz")
    x = np.arange(144) / 6
    fig = plt.figure(figsize=(9.2, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(x, load / 6, color=COLORS["blue"], lw=1.7, label="负荷")
    ax0.plot(x, pv / 6, color=COLORS["green"], lw=1.7, label="光伏")
    ax0.plot(x, d["purchase"], color=COLORS["orange"], lw=1.5, label="购电")
    ax0.set_ylabel("十分钟电量 / kWh")
    ax0.legend(ncol=3, loc="upper right", frameon=False)
    ax0.text(0.01, 0.93, "(a) 供需与购电", transform=ax0.transAxes, fontsize=9, fontweight="bold")

    ax1 = fig.add_subplot(gs[1, 0])
    width = 1 / 6
    ax1.bar(x, d["charge"], width=width, color=COLORS["green"], label="充电")
    ax1.bar(x, -d["discharge"], width=width, color=COLORS["red"], label="放电")
    ax1.axhline(0, color="#333333", lw=0.7)
    ax1.set_xlabel("时刻 / h"); ax1.set_ylabel("充放电量 / kWh")
    ax1.legend(loc="lower left", frameon=False)
    ax1.text(0.02, 0.91, "(b) 储能功率", transform=ax1.transAxes, fontsize=9, fontweight="bold")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2b = ax2.twinx()
    ax2.step(np.arange(145) / 6, d["soc"], where="post", color=COLORS["blue"], lw=1.8, label="SOC")
    ax2b.plot(x, fixed_price, color=COLORS["orange"], lw=1.4, label="电价")
    ax2.set_xlabel("时刻 / h"); ax2.set_ylabel("储电量 / kWh"); ax2b.set_ylabel("电价 / 元·kWh$^{-1}$")
    ax2.set_xlim(0, 24); ax2.set_xticks(np.arange(0, 25, 4))
    lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower left", frameon=False)
    ax2.text(0.02, 0.91, "(c) SOC与电价", transform=ax2.transAxes, fontsize=9, fontweight="bold")
    fig.savefig(OUT / "图1_问题1调度_终稿.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def policy_tradeoff():
    c = json.loads((m.CACHE / "comparisons.json").read_text(encoding="utf-8"))["q2_policy_comparison"]
    keys = ["deterministic", "stochastic_risk_neutral", "cvar_0.04"]
    names = ["确定性", "随机优化", "CVaR"]
    costs = np.array([c[k]["total_cost"] for k in keys]) / 1e4
    emergency = np.array([c[k]["emergency_kwh"] for k in keys]) / 1e4
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw={"wspace": 0.32})
    colors = [COLORS["gray"], COLORS["blue"], COLORS["red"]]
    y = np.arange(len(names))
    bars0 = ax0.barh(y, costs, color=colors, height=0.56)
    bars1 = ax1.barh(y, emergency, color=colors, height=0.56)
    for ax, bars, values in [(ax0, bars0, costs), (ax1, bars1, emergency)]:
        ax.set_yticks(y, names); ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f" {val:.1f}", va="center", fontsize=8.5)
    ax0.set_xlabel("全年总费用 / 万元"); ax1.set_xlabel("紧急购电量 / 万kWh")
    ax0.text(0.02, 0.94, "(a) 费用", transform=ax0.transAxes, fontsize=9, fontweight="bold")
    ax1.text(0.02, 0.94, "(b) 风险", transform=ax1.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "图2_费用风险权衡_终稿.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def q3_information_value(fixed_price, load, pv):
    d = np.load(m.CACHE / "q3.npz")
    dates = pd.to_datetime(d["dates"])
    adjusted_cost = np.sum(fixed_price[None, :] * d["adjusted"] + 0.5 * fixed_price[None, :] * np.abs(d["adjusted"] - d["purchase"]) + 5 * fixed_price[None, :] * d["emergency"], axis=1)
    no_cost = np.zeros(len(dates))
    for i, day in enumerate(range(31, 365)):
        rec = m.actual_recourse(fixed_price, load[day], pv[day], d["purchase"][i])
        no_cost[i] = np.sum(fixed_price * d["purchase"][i]) + np.sum(5 * fixed_price * rec["emergency"])
    diff = np.cumsum(adjusted_cost - no_cost) / 1e4
    energy_delta = (np.sum(fixed_price[None, :] * d["adjusted"]) - np.sum(fixed_price[None, :] * d["purchase"])) / 1e4
    penalty = np.sum(0.5 * fixed_price[None, :] * np.abs(d["adjusted"] - d["purchase"])) / 1e4
    emergency_delta = (np.sum(5 * fixed_price[None, :] * d["emergency"]) - np.sum(5 * fixed_price[None, :] * np.array([m.actual_recourse(fixed_price, load[day], pv[day], d["purchase"][i])["emergency"] for i, day in enumerate(range(31, 365))]))) / 1e4
    components = np.array([energy_delta, penalty, emergency_delta, energy_delta + penalty + emergency_delta])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.5, 4.0), gridspec_kw={"width_ratios": [0.9, 1.35], "wspace": 0.30})
    labels = ["能源费\n变化", "调整\n偏差费", "紧急费\n变化", "总费用\n变化"]
    colors = [COLORS["orange"], COLORS["red"], COLORS["green"], COLORS["purple"]]
    bars = ax0.bar(np.arange(4), components, color=colors, width=0.62)
    ax0.axhline(0, color="#333333", lw=0.8)
    ax0.set_xticks(np.arange(4), labels); ax0.set_ylabel("相对不调整方案的变化 / 万元")
    for bar, val in zip(bars, components):
        ax0.text(bar.get_x() + bar.get_width()/2, val + (1.4 if val >= 0 else -2.0), f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8.5)
    ax0.text(0.02, 0.94, "(a) 费用分解", transform=ax0.transAxes, fontsize=9, fontweight="bold")
    ax1.plot(dates, diff, color=COLORS["red"], lw=1.5)
    ax1.axhline(0, color="#444444", lw=0.8)
    ax1.fill_between(dates, 0, diff, where=diff >= 0, color=COLORS["red"], alpha=0.12)
    ax1.set_ylabel("累计增加费用 / 万元"); ax1.set_xlabel("日期")
    ax1.text(0.02, 0.94, "(b) 日累计变化", transform=ax1.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "图3_滚动预报信息价值_终稿.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def q4_heatmaps(price):
    q4d = np.load(m.CACHE / "q4_2.npz")
    q4 = q4d["purchase"]
    dates = pd.to_datetime(q4d["dates"])
    p = price[31:]
    months = np.arange(2, 13)
    price_m = np.vstack([np.median(p[dates.month == mo], axis=0) for mo in months])
    purchase_m = np.vstack([np.median(q4[dates.month == mo], axis=0) for mo in months])
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.3), sharex=True, gridspec_kw={"hspace": 0.28})
    im0 = axes[0].imshow(price_m, aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=np.quantile(price_m, 0.98))
    im1 = axes[1].imshow(purchase_m, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=np.quantile(purchase_m, 0.98))
    for ax, title, im in [(axes[0], "(a) 月内典型电价 / 元·kWh$^{-1}$", im0), (axes[1], "(b) 月内典型计划购电量 / kWh", im1)]:
        ax.set_ylabel("月份"); ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_yticks(np.arange(len(months)), [f"{mo}月" for mo in months])
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.018)
    axes[-1].set_xticks(np.arange(0, 145, 24)); axes[-1].set_xticklabels([f"{h}:00" for h in range(0, 25, 4)])
    axes[-1].set_xlabel("日内时刻")
    fig.tight_layout()
    fig.savefig(OUT / "图4_波动电价购电时移热图_终稿.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    setup()
    _, fixed_price, base_load, base_pv, load, pv, _, price = m.read_inputs()
    q1_dispatch(fixed_price, base_load, base_pv)
    policy_tradeoff()
    q3_information_value(fixed_price, load, pv)
    q4_heatmaps(price)


if __name__ == "__main__":
    main()
