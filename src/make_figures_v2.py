# -*- coding: utf-8 -*-
"""论文主图 v2 —— 按图表库模板风格(OKABE-ITO 色板、去边框、浅灰网格)用真实结果重绘,共 5 张。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import solve_c as m

OUT = m.PROJECT / "论文" / "附图"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
GREY = "#B7BEC6"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已输出:", name)


def fig1(fixed_price, base_load, base_pv):
    """问题1确定性日前调度:电价/负荷光伏购电/充放电/SOC 四面板,无双轴。"""
    d = np.load(m.CACHE / "q1.npz")
    x = np.arange(144) / 6
    fig, axes = plt.subplots(4, 1, figsize=(9.4, 8.6), sharex=True, gridspec_kw={"hspace": 0.12})
    axes[0].step(x, fixed_price, where="post", color=OI[1], lw=1.8)
    axes[0].set_ylabel("电价\n(元/kWh)")
    axes[1].plot(x, base_load / 6, color=OI[0], lw=1.3, label="负荷")
    axes[1].plot(x, base_pv / 6, color=OI[2], lw=1.3, label="光伏")
    axes[1].plot(x, d["purchase"], color=OI[3], lw=1.3, label="购电")
    axes[1].set_ylabel("电量\n(kWh)")
    axes[1].legend(ncol=3, loc="upper left", fontsize=9)
    axes[2].bar(x, d["charge"], width=1 / 6, color=OI[2], label="充电")
    axes[2].bar(x, -d["discharge"], width=1 / 6, color=OI[3], label="放电")
    axes[2].axhline(0, color="#555555", lw=0.6)
    axes[2].set_ylabel("充放电\n(kWh)")
    axes[2].legend(ncol=2, fontsize=9)
    axes[3].step(np.arange(145) / 6, d["soc"], where="post", color=OI[0], lw=1.8, label="储电量")
    axes[3].axhline(6000, color=GREY, lw=0.8, ls="--")
    axes[3].set_ylabel("储电量\n(kWh)")
    axes[3].set_xlabel("时刻 (h)")
    axes[3].set_xlim(0, 24)
    axes[3].set_xticks(np.arange(0, 25, 4))
    axes[3].legend(loc="upper right", fontsize=9)
    save(fig, "图1_问题1确定性日前调度.png")


def fig2():
    """问题2三策略费用—供电风险权衡(越小越好)。"""
    c = json.loads((m.CACHE / "comparisons.json").read_text(encoding="utf-8"))["q2_policy_comparison"]
    rows = [
        ("确定性点预测", c["deterministic"]),
        ("风险中性场景", c["stochastic_risk_neutral"]),
        ("CVaR 风险厌恶", c["cvar_0.04"]),
    ]
    em = np.array([r[1]["emergency_kwh"] for r in rows]) / 1e4
    cost = np.array([r[1]["total_cost"] for r in rows]) / 1e4
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    # 确定性点被随机方案支配(成本与紧急购电均更高),用灰;两个非支配点用主色
    ax.scatter(em[1:], cost[1:], s=110, color=[OI[0], OI[3]], zorder=3)
    ax.scatter(em[0], cost[0], s=110, color=GREY, zorder=3)
    for i, (name, _) in enumerate(rows):
        dx, dy = (8, 8) if i == 0 else (8, 10)
        ax.annotate(name, (em[i], cost[i]), xytext=(dx, dy), textcoords="offset points", fontsize=9.5)
    ax.plot(em, cost, color=GREY, lw=0.8, zorder=1)
    ax.set_xlabel("全年紧急购电量 (万 kWh)")
    ax.set_ylabel("全年总费用 (万元)")
    ax.invert_xaxis()
    save(fig, "图2_三策略费用风险权衡.png")


def fig3(fixed_price, load, pv):
    """问题3滚动调整相对不调整的累计经济效果(时间序列)。"""
    d = np.load(m.CACHE / "q3.npz")
    dates = pd.to_datetime(d["dates"])
    adj_cost = np.sum(fixed_price[None, :] * d["adjusted"] + 0.5 * fixed_price[None, :] * np.abs(d["adjusted"] - d["purchase"]) + 5 * fixed_price[None, :] * d["emergency"], axis=1)
    no_cost = np.zeros(len(dates))
    for i, day in enumerate(range(31, 365)):
        rec = m.actual_recourse(fixed_price, load[day], pv[day], d["purchase"][i])
        no_cost[i] = np.sum(fixed_price * d["purchase"][i]) + np.sum(5 * fixed_price * rec["emergency"])
    diff = np.cumsum(adj_cost - no_cost) / 1e4
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(dates, diff, color=OI[3], lw=1.6)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.fill_between(dates, 0, diff, where=diff >= 0, color=OI[3], alpha=0.15)
    ax.set_ylabel("累计增加费用 (万元)")
    ax.set_xlabel("日期")
    save(fig, "图3_滚动调整累计经济效果.png")


def fig4(price):
    """问题4波动电价驱动全年购电时移:电价/波动电价购电/相对固定电价的变化。"""
    q2 = np.load(m.CACHE / "q2.npz")["purchase"]
    q4 = np.load(m.CACHE / "q4_2.npz")["purchase"]
    diff = q4 - q2
    items = [
        ("实际波动电价 (元/kWh)", price[31:], "YlOrRd", 0),
        ("波动电价下购电量 (kWh)", q4, "Blues", 0),
        ("相对固定电价的变化 (kWh)", diff, "RdBu_r", 1),
    ]
    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.08], wspace=0.22)
    for k, (title, mat, cmap, div) in enumerate(items):
        ax = fig.add_subplot(gs[0, k])
        vmax = np.quantile(np.abs(mat), 0.99) if div else np.quantile(mat, 0.99)
        vmin = -vmax if div else 0
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("日内时刻")
        ax.set_ylabel("日序" if k == 0 else "")
        ax.set_xticks(np.arange(0, 145, 36))
        ax.set_xticklabels(["0:00", "6:00", "12:00", "18:00", "24:00"])
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    save(fig, "图4_波动电价购电时移热图.png")


def fig5(load, pv, load_pred, pv_pred):
    """因果预测 vs 实测(代表性一周,负荷与光伏)。"""
    d0, nd = 152, 7
    days = np.arange(d0, d0 + nd)
    t = np.arange(nd * 144) / 6
    act_l = load[days].reshape(-1)
    prd_l = load_pred[days].reshape(-1)
    act_p = pv[days].reshape(-1)
    prd_p = pv_pred[days].reshape(-1)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.6), sharex=True, gridspec_kw={"hspace": 0.14})
    axes[0].plot(t, act_l, color=OI[0], lw=1.1, label="实测负荷")
    axes[0].plot(t, prd_l, color=OI[1], lw=1.0, ls="--", label="因果预测")
    axes[0].set_ylabel("负荷 (kW)")
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].plot(t, act_p, color=OI[2], lw=1.1, label="实测光伏")
    axes[1].plot(t, prd_p, color=OI[3], lw=1.0, ls="--", label="因果预测")
    axes[1].set_ylabel("光伏 (kW)")
    axes[1].set_xlabel("时刻 (h)")
    axes[1].set_xlim(0, nd * 24)
    axes[1].set_xticks(np.arange(0, nd * 24 + 1, 24))
    axes[1].legend(ncol=2, fontsize=9)
    save(fig, "图5_因果预测与实测对比.png")


def main():
    _, fixed_price, base_load, base_pv, load, pv, _, price = m.read_inputs()
    fc = np.load(m.CACHE / "forecasts.npz")
    fig1(fixed_price, base_load, base_pv)
    fig2()
    fig3(fixed_price, load, pv)
    fig4(price)
    fig5(load, pv, fc["load_pred"], fc["pv_pred"])


if __name__ == "__main__":
    main()
