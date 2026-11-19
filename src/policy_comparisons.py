"""确定性/随机/CVaR对照，以及滚动光伏预报的信息价值诊断。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

import solve_c as m


def aggregate(rows):
    return {
        "total_cost": float(sum(x["total_cost"] for x in rows)),
        "plan_cost": float(sum(x["planned_or_adjusted_cost"] for x in rows)),
        "emergency_cost": float(sum(x["emergency_cost"] for x in rows)),
        "emergency_kwh": float(sum(x["emergency_kwh"] for x in rows)),
        "spill_kwh": float(sum(x["spill_kwh"] for x in rows)),
        "emergency_days": int(sum(x["emergency_kwh"] > 1e-5 for x in rows)),
    }


def q2_comparison(fixed_price, load, pv, load_pred, pv_pred):
    settings = [
        ("deterministic", None),
        ("stochastic_risk_neutral", 0.0),
        ("cvar_0.04", 0.04),
        ("cvar_0.08", 0.08),
        ("cvar_0.16", 0.16),
    ]
    out = {}
    for name, risk in settings:
        rows = []
        for i, d in enumerate(range(31, 365)):
            if risk is None:
                ls, ps = load_pred[d][None, :], pv_pred[d][None, :]
                lam = 0.0
            else:
                ls, ps, _ = m.make_scenarios(d, load_pred[d], pv_pred[d], load, pv, load_pred, pv_pred)
                lam = risk
            q, _, _ = m.solve_dispatch(fixed_price, ls, ps, cvar_weight=lam)
            rec = m.actual_recourse(fixed_price, load[d], pv[d], q)
            rows.append(m.summarize_recourse(fixed_price, q, rec))
            if i % 100 == 0:
                print(f"{name}: {i+1}/334", flush=True)
        out[name] = aggregate(rows)
    return out


def issue_forecast_metrics(pv, forecast_hourly):
    pred = m.pv_issue_history(pv, forecast_hourly)
    out = {}
    for issue in range(4):
        start = issue * 36
        e = pred[31:, issue, start:] - pv[31:, start:]
        out[f"{6*issue:02d}:00"] = {
            "remaining_hours": 24 - 6 * issue,
            "mae_kw": float(np.mean(np.abs(e))),
            "rmse_kw": float(np.sqrt(np.mean(e**2))),
            "nrmse_mean": float(np.sqrt(np.mean(e**2)) / max(np.mean(pv[31:, start:]), 1e-9)),
        }
    return out


def q3_cost_decomposition(fixed_price, load, pv):
    data = np.load(m.CACHE / "q3.npz")
    plan = data["purchase"]
    adjusted = data["adjusted"]
    price = np.repeat(fixed_price[None, :], len(plan), axis=0)
    plan_cost = float(np.sum(price * plan))
    adjusted_energy_cost = float(np.sum(price * adjusted))
    adjustment_penalty = float(np.sum(0.5 * price * np.abs(adjusted - plan)))
    adjusted_emergency_cost = float(np.sum(5 * price * data["emergency"]))
    no_emergency = 0.0
    no_emergency_kwh = 0.0
    for i, d in enumerate(range(31, 365)):
        rec = m.actual_recourse(fixed_price, load[d], pv[d], plan[i])
        no_emergency += float(np.sum(5 * fixed_price * rec["emergency"]))
        no_emergency_kwh += float(np.sum(rec["emergency"]))
    return {
        "plan_cost": plan_cost,
        "adjusted_energy_cost": adjusted_energy_cost,
        "adjustment_penalty": adjustment_penalty,
        "adjusted_emergency_cost": adjusted_emergency_cost,
        "adjusted_total_cost": adjusted_energy_cost + adjustment_penalty + adjusted_emergency_cost,
        "no_adjust_emergency_cost": no_emergency,
        "no_adjust_emergency_kwh": no_emergency_kwh,
        "no_adjust_total_cost": plan_cost + no_emergency,
        "net_value_of_adjustment": plan_cost + no_emergency - (adjusted_energy_cost + adjustment_penalty + adjusted_emergency_cost),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q3-only", action="store_true")
    args = parser.parse_args()
    dates, fixed_price, base_load, base_pv, load, pv, forecast_hourly, varying_price = m.read_inputs()
    f = np.load(m.CACHE / "forecasts.npz")
    out = m.CACHE / "comparisons.json"
    report = json.loads(out.read_text(encoding="utf-8")) if args.q3_only and out.exists() else {}
    if not args.q3_only:
        report["q2_policy_comparison"] = q2_comparison(fixed_price, load, pv, f["load_pred"], f["pv_pred"])
    report["q3_issue_forecast_metrics"] = issue_forecast_metrics(pv, forecast_hourly)
    report["q3_cost_decomposition"] = q3_cost_decomposition(fixed_price, load, pv)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
