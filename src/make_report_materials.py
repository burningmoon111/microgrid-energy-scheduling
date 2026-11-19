"""从可追溯的 npz/json 结果生成论文用 Markdown 数值素材。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "C题项目"
CACHE = PROJECT / "中间数据"
ATTACH = ROOT / "题目" / "C题" / "附件"
SELECTED = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def money(x):
    return f"{x:,.2f}"


def num(x):
    return f"{x:,.3f}"


def selected_daily(label, price, adjusted=False):
    data = np.load(CACHE / f"{label}.npz")
    dates = pd.to_datetime(data["dates"])
    rows = []
    for date in SELECTED:
        i = int(np.where(dates == pd.Timestamp(date))[0][0])
        p = price if price.ndim == 1 else price[i]
        q = data["adjusted"][i] if adjusted else data["purchase"][i]
        fee = np.sum(p * q) + (np.sum(0.5 * p * np.abs(data["adjusted"][i] - data["purchase"][i])) if adjusted else 0)
        emergency_fee = np.sum(5 * p * data["emergency"][i])
        rows.append([date, np.sum(q), fee, np.sum(data["emergency"][i]), emergency_fee, fee + emergency_fee])
    return rows


def main():
    a1 = pd.read_excel(ATTACH / "附件1.xlsx")
    fixed_price = a1.iloc[:, 1].to_numpy(float)
    price_all = pd.read_excel(ATTACH / "附件4.xlsx", index_col=0).to_numpy(float)[31:]
    summaries = json.loads((CACHE / "summaries.json").read_text(encoding="utf-8"))
    comparison = json.loads((CACHE / "comparisons.json").read_text(encoding="utf-8"))
    validation = json.loads((CACHE / "validation.json").read_text(encoding="utf-8"))
    q1 = np.load(CACHE / "q1.npz")

    lines = ["# C题结果汇总", "", "所有数值均由 `代码/solve_c.py` 生成，并由 `代码/prepare_outputs.py` 独立复核。", ""]
    lines += ["## 问题1", "", "| 时间段 | 购电量/kWh |", "|---|---:|"]
    for idx, label in zip([59, 71, 83, 95, 107, 119], ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]):
        lines.append(f"| {label} | {num(q1['purchase'][idx])} |")
    lines += ["", f"全天计划购电量为 **{num(np.sum(q1['purchase']))} kWh**，购电费为 **{money(np.sum(fixed_price*q1['purchase']))} 元**。", "", "| 时间段 | 充电量/kWh | 放电量/kWh |", "|---|---:|---:|"]
    for b in range(6):
        a, z = b * 24, (b + 1) * 24
        lines.append(f"| {4*b}:00-{4*(b+1)}:00 | {num(np.sum(q1['charge'][a:z]))} | {num(np.sum(q1['discharge'][a:z]))} |")
    lines += ["", "0:00与24:00储电量均为6000.000 kWh。", ""]

    for title, label, p, adj in [
        ("问题2（固定电价，0:00计划）", "q2", fixed_price, False),
        ("问题3（固定电价，滚动调整）", "q3", fixed_price, True),
        ("问题4-2（波动电价，0:00计划）", "q4_2", price_all, False),
        ("问题4-3（波动电价，滚动调整）", "q4_3", price_all, True),
    ]:
        lines += [f"## {title}", "", "| 日期 | 购电量/kWh | 计划或调整费用/元 | 紧急购电量/kWh | 紧急购电费/元 | 总费用/元 |", "|---|---:|---:|---:|---:|---:|"]
        for row in selected_daily(label, p, adj):
            lines.append(f"| {row[0]} | {num(row[1])} | {money(row[2])} | {num(row[3])} | {money(row[4])} | {money(row[5])} |")
        s = summaries[label]
        lines += ["", f"全年（2月1日至12月31日）总购电量 {num(s['purchase_kwh'])} kWh，紧急购电量 {num(s['emergency_kwh'])} kWh，总费用 {money(s['total_cost'])} 元。", ""]

    c = comparison["q2_policy_comparison"]
    lines += ["## 问题2模型对照与CVaR敏感性", "", "| 策略 | 总费用/元 | 紧急购电量/kWh | 紧急购电天数 | 弃电量/kWh |", "|---|---:|---:|---:|---:|"]
    for key, name in [("deterministic", "确定性点预测"), ("stochastic_risk_neutral", "风险中性场景优化"), ("cvar_0.04", "CVaR风险厌恶")]:
        r = c[key]
        lines.append(f"| {name} | {money(r['total_cost'])} | {num(r['emergency_kwh'])} | {r['emergency_days']} | {num(r['spill_kwh'])} |")

    q3c = comparison["q3_cost_decomposition"]
    lines += ["", "## 问题3信息价值", "", f"不调整时总费用为 {money(q3c['no_adjust_total_cost'])} 元；三次滚动调整后为 {money(q3c['adjusted_total_cost'])} 元，其中调整偏差费用为 {money(q3c['adjustment_penalty'])} 元。使用其他时刻预报反而增加 {money(-q3c['net_value_of_adjustment'])} 元，因此在给定结算规则下不建议全量启用三次调整。", "", "## 校验", "", f"全部约束校验通过：`{validation['all_constraints_passed']}`。全年模型最大能量平衡残差不超过 {max(validation[k]['balance_max_abs_kwh'] for k in ['q2','q3','q4_2','q4_3']):.3e} kWh。", ""]
    (PROJECT / "结果汇总.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
