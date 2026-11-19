"""2026 C题：微网购电、储能与紧急购电的统一求解程序。

所有计划预测严格只使用决策时刻以前的数据。功率统一除以6转为
10分钟电量；充放电变量均定义在交流母线侧。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


DT = 1 / 6
ETA = 0.90
SOC_MIN, SOC_MAX, SOC_REF = 1200.0, 10800.0, 6000.0
FLOW_MAX = 5000.0 * DT
ALPHA = 0.90
# 主提交按题目“尽可能节省费用”采用风险中性期望成本；CVaR仅作风险偏好扩展。
CVaR_WEIGHT = 0.0
N_SCENARIOS = 5
EPS = 1e-7


ROOT = Path(__file__).resolve().parents[2]
ATTACH = ROOT / "题目" / "C题" / "附件"
PROJECT = ROOT / "C题项目"
CACHE = PROJECT / "中间数据"


def read_inputs():
    a1 = pd.read_excel(ATTACH / "附件1.xlsx")
    a2_load = pd.read_excel(ATTACH / "附件2.xlsx", sheet_name="小区负载", index_col=0)
    a2_pv = pd.read_excel(ATTACH / "附件2.xlsx", sheet_name="光伏发电实际功率", index_col=0)
    a4 = pd.read_excel(ATTACH / "附件4.xlsx", index_col=0)

    fixed_price = a1.iloc[:, 1].to_numpy(float)
    base_load = a1.iloc[:, 2].to_numpy(float)
    base_pv = a1.iloc[:, 3].to_numpy(float)
    load = a2_load.to_numpy(float)
    pv = a2_pv.to_numpy(float)
    price = a4.to_numpy(float)
    dates = pd.to_datetime(a2_load.index).to_numpy()

    f = pd.read_excel(ATTACH / "附件3.xlsx")
    forecast = np.zeros((365, 4, 24), dtype=float)
    for d in range(365):
        forecast[d] = f.iloc[4 * d : 4 * d + 4, 2:26].to_numpy(float)
    return dates, fixed_price, base_load, base_pv, load, pv, forecast, price


def causal_profiles(data: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """近期+同星期日轮廓预测，日d只读取[0,d)实际数据。"""
    out = np.zeros_like(data, dtype=float)
    out[0] = baseline
    for d in range(1, len(data)):
        recent = data[max(0, d - 14) : d]
        weekly_idx = np.arange(d - 7, -1, -7)[:6]
        weekly = data[weekly_idx] if len(weekly_idx) else recent
        recent_profile = np.median(recent, axis=0)
        weekly_profile = np.mean(weekly, axis=0)
        # 同星期日捕捉工作日效应，近期中位数降低异常天气影响。
        out[d] = 0.65 * weekly_profile + 0.35 * recent_profile
    return np.maximum(out, 0.0)


def representative_indices(score: np.ndarray, count: int) -> np.ndarray:
    """按净负荷误差日能量的分位点选代表场景，确定且可复现。"""
    order = np.argsort(score)
    if len(order) <= count:
        return order
    pos = np.linspace(0, len(order) - 1, count)
    return order[np.rint(pos).astype(int)]


def make_scenarios(
    day: int,
    load_fc: np.ndarray,
    pv_fc: np.ndarray,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
    load_pred_hist: np.ndarray,
    pv_pred_hist: np.ndarray,
    start: int = 0,
    price_fc: np.ndarray | None = None,
    price_actual: np.ndarray | None = None,
    price_pred_hist: np.ndarray | None = None,
):
    pool = np.arange(max(0, day - 42), day)
    if len(pool) == 0:
        load_s = load_fc[None, :]
        pv_s = pv_fc[None, :]
        price_s = price_fc[None, :] if price_fc is not None else None
        return load_s, pv_s, price_s

    lr = load_actual[pool, start : start + len(load_fc)] - load_pred_hist[pool, start : start + len(load_fc)]
    pr = pv_actual[pool, start : start + len(pv_fc)] - pv_pred_hist[pool, start : start + len(pv_fc)]
    score = np.sum(lr - pr, axis=1)
    pick = representative_indices(score, N_SCENARIOS)
    load_s = np.maximum(load_fc[None, :] + lr[pick], 0.0)
    pv_s = np.maximum(pv_fc[None, :] + pr[pick], 0.0)

    if price_fc is None:
        price_s = None
    else:
        rr = price_actual[pool, start : start + len(price_fc)] - price_pred_hist[pool, start : start + len(price_fc)]
        price_s = np.maximum(price_fc[None, :] + rr[pick], 0.001)
    return load_s, pv_s, price_s


def solve_dispatch(
    price: np.ndarray,
    load_s_kw: np.ndarray,
    pv_s_kw: np.ndarray,
    initial_soc: float = SOC_REF,
    terminal_soc: float = SOC_REF,
    cvar_weight: float = CVaR_WEIGHT,
    baseline: np.ndarray | None = None,
    fixed_purchase: np.ndarray | None = None,
    allow_emergency: bool = True,
):
    """共享购电计划+场景储能追索的稀疏线性规划。"""
    load_s = np.atleast_2d(load_s_kw) * DT
    pv_s = np.atleast_2d(pv_s_kw) * DT
    s_count, T = load_s.shape
    price_s = np.atleast_2d(price)
    if price_s.shape[0] == 1:
        price_s = np.repeat(price_s, s_count, axis=0)
    mean_price = price_s.mean(axis=0)

    # q | scenario blocks(c,d,e,w,soc[0:T+1]) | plus,minus(optional) | z,xi(optional)
    block = 4 * T + T + 1
    q0 = 0
    scen0 = T
    adj0 = scen0 + s_count * block
    use_adjust = baseline is not None
    risk0 = adj0 + (2 * T if use_adjust else 0)
    use_cvar = allow_emergency and cvar_weight > 0 and s_count > 1
    nvar = risk0 + (1 + s_count if use_cvar else 0)

    obj = np.zeros(nvar)
    if use_adjust:
        obj[q0 : q0 + T] = mean_price
        obj[adj0 : adj0 + 2 * T] = np.tile(0.5 * mean_price, 2)
    else:
        obj[q0 : q0 + T] = mean_price

    for s in range(s_count):
        b = scen0 + s * block
        e = b + 2 * T
        obj[e : e + T] = 5.0 * price_s[s] / s_count
        # 极小循环代价只用于消除充放电退化解，不影响经济目标。
        obj[b : b + 2 * T] += 1e-8

    if use_cvar:
        z = risk0
        xi = risk0 + 1
        obj[z] = cvar_weight
        obj[xi : xi + s_count] = cvar_weight / ((1 - ALPHA) * s_count)

    neq = s_count * (2 * T) + (T if use_adjust else 0)
    Aeq = lil_matrix((neq, nvar))
    beq = np.zeros(neq)
    r = 0
    for s in range(s_count):
        b = scen0 + s * block
        c, d, e, w, soc = b, b + T, b + 2 * T, b + 3 * T, b + 4 * T
        for t in range(T):
            Aeq[r, q0 + t] = 1
            Aeq[r, c + t] = -1
            Aeq[r, d + t] = 1
            Aeq[r, e + t] = 1
            Aeq[r, w + t] = -1
            beq[r] = load_s[s, t] - pv_s[s, t]
            r += 1
        for t in range(T):
            Aeq[r, soc + t + 1] = 1
            Aeq[r, soc + t] = -1
            Aeq[r, c + t] = -ETA
            Aeq[r, d + t] = 1 / ETA
            r += 1
    if use_adjust:
        plus, minus = adj0, adj0 + T
        for t in range(T):
            Aeq[r, q0 + t] = 1
            Aeq[r, plus + t] = -1
            Aeq[r, minus + t] = 1
            beq[r] = baseline[t]
            r += 1

    Aub = None
    bub = None
    if use_cvar:
        Aub = lil_matrix((s_count, nvar))
        bub = np.zeros(s_count)
        z, xi = risk0, risk0 + 1
        for s in range(s_count):
            b = scen0 + s * block
            e = b + 2 * T
            Aub[s, e : e + T] = 5.0 * price_s[s]
            Aub[s, z] = -1
            Aub[s, xi + s] = -1

    bounds = [(0, None)] * nvar
    if fixed_purchase is not None:
        for t, v in enumerate(fixed_purchase):
            bounds[q0 + t] = (float(v), float(v))
    for s in range(s_count):
        b = scen0 + s * block
        c, d, e, w, soc = b, b + T, b + 2 * T, b + 3 * T, b + 4 * T
        for t in range(T):
            bounds[c + t] = (0, FLOW_MAX)
            bounds[d + t] = (0, FLOW_MAX)
            bounds[e + t] = (0, None if allow_emergency else 0)
            bounds[w + t] = (0, None)
        for t in range(T + 1):
            bounds[soc + t] = (SOC_MIN, SOC_MAX)
        bounds[soc] = (initial_soc, initial_soc)
        bounds[soc + T] = (terminal_soc, terminal_soc)
    if use_cvar:
        bounds[risk0] = (0, None)

    res = linprog(
        obj,
        A_ub=Aub.tocsr() if Aub is not None else None,
        b_ub=bub,
        A_eq=Aeq.tocsr(),
        b_eq=beq,
        bounds=bounds,
        method="highs",
        options={"dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7},
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x
    q = x[:T]
    scen = []
    for s in range(s_count):
        b = scen0 + s * block
        scen.append(
            {
                "charge": x[b : b + T],
                "discharge": x[b + T : b + 2 * T],
                "emergency": x[b + 2 * T : b + 3 * T],
                "spill": x[b + 3 * T : b + 4 * T],
                "soc": x[b + 4 * T : b + 5 * T + 1],
            }
        )
    return q, scen, res.fun


def actual_recourse(price, load, pv, purchase, initial=SOC_REF, terminal=SOC_REF):
    q, scen, _ = solve_dispatch(
        price,
        load[None, :],
        pv[None, :],
        initial_soc=initial,
        terminal_soc=terminal,
        cvar_weight=0,
        fixed_purchase=purchase,
        allow_emergency=True,
    )
    return scen[0]


def hourly_to_ten_min(hourly: np.ndarray, remaining: int | None = None) -> np.ndarray:
    x = np.repeat(np.asarray(hourly, float), 6)
    return x if remaining is None else x[:remaining]


def pv_issue_history(pv_actual: np.ndarray, forecast_hourly: np.ndarray):
    out = np.zeros((365, 4, 144))
    for d in range(365):
        for k in range(4):
            start = 36 * k
            n = 144 - start
            out[d, k, start:] = hourly_to_ten_min(forecast_hourly[d, k], n)
    return out


def summarize_recourse(price, purchase, rec, baseline=None):
    settlement = np.sum(price * purchase)
    if baseline is not None:
        settlement += np.sum(0.5 * price * np.abs(purchase - baseline))
    emergency_cost = np.sum(5 * price * rec["emergency"])
    return {
        "purchase_kwh": float(np.sum(purchase)),
        "planned_or_adjusted_cost": float(settlement),
        "emergency_kwh": float(np.sum(rec["emergency"])),
        "emergency_cost": float(emergency_cost),
        "total_cost": float(settlement + emergency_cost),
        "charge_kwh": float(np.sum(rec["charge"])),
        "discharge_kwh": float(np.sum(rec["discharge"])),
        "spill_kwh": float(np.sum(rec["spill"])),
    }


def solve_q1(fixed_price, base_load, base_pv):
    q, scen, objective = solve_dispatch(
        fixed_price,
        base_load[None, :],
        base_pv[None, :],
        cvar_weight=0,
        allow_emergency=False,
    )
    rec = scen[0]
    balance = q + base_pv * DT + rec["discharge"] - base_load * DT - rec["charge"] - rec["spill"]
    summary = summarize_recourse(fixed_price, q, rec)
    summary.update(
        {
            "objective": float(objective),
            "balance_max_abs": float(np.max(np.abs(balance))),
            "soc_close_error": float(abs(rec["soc"][-1] - rec["soc"][0])),
            "soc_min": float(rec["soc"].min()),
            "soc_max": float(rec["soc"].max()),
        }
    )
    np.savez_compressed(CACHE / "q1.npz", purchase=q, **rec)
    return summary


def solve_q2_or_q4(
    dates,
    fixed_price,
    load,
    pv,
    load_pred,
    pv_pred,
    varying_price=None,
    price_pred=None,
    label="q2",
):
    days = range(31, 365)
    n = len(list(days))
    purchase = np.zeros((n, 144))
    charge = np.zeros_like(purchase)
    discharge = np.zeros_like(purchase)
    emergency = np.zeros_like(purchase)
    spill = np.zeros_like(purchase)
    soc = np.zeros((n, 145))
    daily = []

    for out_i, d in enumerate(days):
        if varying_price is None:
            p_fc = fixed_price
            p_actual = fixed_price
            p_s = None
        else:
            p_fc = price_pred[d]
            p_actual = varying_price[d]
            p_s = True
        ls, ps, prs = make_scenarios(
            d,
            load_pred[d],
            pv_pred[d],
            load,
            pv,
            load_pred,
            pv_pred,
            price_fc=p_fc if p_s else None,
            price_actual=varying_price if p_s else None,
            price_pred_hist=price_pred if p_s else None,
        )
        plan_price = prs if prs is not None else p_fc
        q, _, _ = solve_dispatch(plan_price, ls, ps)
        rec = actual_recourse(p_actual, load[d], pv[d], q)
        purchase[out_i] = q
        for key, arr in [("charge", charge), ("discharge", discharge), ("emergency", emergency), ("spill", spill)]:
            arr[out_i] = rec[key]
        soc[out_i] = rec["soc"]
        daily.append(summarize_recourse(p_actual, q, rec))
        if out_i % 40 == 0:
            print(f"{label}: {out_i + 1}/{n}", flush=True)

    np.savez_compressed(
        CACHE / f"{label}.npz",
        dates=dates[31:], purchase=purchase, charge=charge, discharge=discharge,
        emergency=emergency, spill=spill, soc=soc,
    )
    return aggregate_daily(daily)


def make_issue_scenarios(day, issue, load_fc, pv_fc, load, pv, load_pred, pv_issue_pred):
    start = issue * 36
    pool = np.arange(max(0, day - 42), day)
    if len(pool) == 0:
        return load_fc[None, :], pv_fc[None, :]
    lr = load[pool, start:] - load_pred[pool, start:]
    pr = pv[pool, start:] - pv_issue_pred[pool, issue, start:]
    score = np.sum(lr - pr, axis=1)
    pick = representative_indices(score, N_SCENARIOS)
    return np.maximum(load_fc[None, :] + lr[pick], 0), np.maximum(pv_fc[None, :] + pr[pick], 0)


def solve_q3_or_q43(
    dates,
    fixed_price,
    load,
    pv,
    forecast_hourly,
    load_pred,
    varying_price=None,
    price_pred=None,
    label="q3",
):
    pv_issue_pred = pv_issue_history(pv, forecast_hourly)
    days = range(31, 365)
    n = len(list(days))
    planned = np.zeros((n, 144))
    adjusted = np.zeros_like(planned)
    charge = np.zeros_like(planned)
    discharge = np.zeros_like(planned)
    emergency = np.zeros_like(planned)
    spill = np.zeros_like(planned)
    soc = np.zeros((n, 145))
    daily, no_adjust_daily = [], []

    for out_i, d in enumerate(days):
        actual_price = fixed_price if varying_price is None else varying_price[d]
        p0 = fixed_price if varying_price is None else price_pred[d]
        pv0 = pv_issue_pred[d, 0]
        ls, ps = make_issue_scenarios(d, 0, load_pred[d], pv0, load, pv, load_pred, pv_issue_pred)
        q0, scen0, _ = solve_dispatch(p0, ls, ps)
        planned[out_i] = q0

        no_rec = actual_recourse(actual_price, load[d], pv[d], q0)
        no_adjust_daily.append(summarize_recourse(actual_price, q0, no_rec, baseline=q0))

        current_soc = SOC_REF
        soc[out_i, 0] = current_soc
        for issue in range(4):
            start = issue * 36
            end = min(start + 36, 144)
            remaining = 144 - start
            if issue == 0:
                candidate = q0[start:]
                plan_scen = scen0
            else:
                # 已实现负荷对剩余负荷预测作水平校正，限制比例防止单点异常放大。
                ratio = np.mean(load[d, :start]) / max(np.mean(load_pred[d, :start]), EPS)
                ratio = float(np.clip(ratio, 0.80, 1.20))
                lf = load_pred[d, start:] * ratio
                pf = pv_issue_pred[d, issue, start:]
                ls, ps = make_issue_scenarios(d, issue, lf, pf, load, pv, load_pred, pv_issue_pred)
                if varying_price is None:
                    pfc = fixed_price[start:]
                else:
                    observed_ratio = np.mean(varying_price[d, :start]) / max(np.mean(price_pred[d, :start]), EPS)
                    observed_ratio = float(np.clip(observed_ratio, 0.70, 1.30))
                    pfc = price_pred[d, start:] * observed_ratio
                candidate, plan_scen, _ = solve_dispatch(
                    pfc, ls, ps, initial_soc=current_soc, terminal_soc=SOC_REF,
                    baseline=q0[start:],
                )
            target_idx = end - start
            target_soc = float(np.mean([s["soc"][target_idx] for s in plan_scen]))
            if end == 144:
                target_soc = SOC_REF
            exec_q = candidate[: end - start]
            adjusted[out_i, start:end] = exec_q
            rec = actual_recourse(
                actual_price[start:end], load[d, start:end], pv[d, start:end], exec_q,
                initial=current_soc, terminal=target_soc,
            )
            charge[out_i, start:end] = rec["charge"]
            discharge[out_i, start:end] = rec["discharge"]
            emergency[out_i, start:end] = rec["emergency"]
            spill[out_i, start:end] = rec["spill"]
            soc[out_i, start : end + 1] = rec["soc"]
            current_soc = target_soc

        rec_all = {"charge": charge[out_i], "discharge": discharge[out_i], "emergency": emergency[out_i], "spill": spill[out_i], "soc": soc[out_i]}
        daily.append(summarize_recourse(actual_price, adjusted[out_i], rec_all, baseline=q0))
        if out_i % 30 == 0:
            print(f"{label}: {out_i + 1}/{n}", flush=True)

    np.savez_compressed(
        CACHE / f"{label}.npz", dates=dates[31:], purchase=planned, adjusted=adjusted,
        charge=charge, discharge=discharge, emergency=emergency, spill=spill, soc=soc,
    )
    result = aggregate_daily(daily)
    result["no_adjust_total_cost"] = float(sum(x["total_cost"] for x in no_adjust_daily))
    result["forecast_adjustment_saving"] = result["no_adjust_total_cost"] - result["total_cost"]
    return result


def aggregate_daily(rows):
    keys = rows[0].keys()
    out = {k: float(sum(r[k] for r in rows)) for k in keys}
    out["days"] = len(rows)
    out["emergency_days"] = int(sum(r["emergency_kwh"] > 1e-5 for r in rows))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="all", help="q1,q2,q3,q4 or all")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    dates, fixed_price, base_load, base_pv, load, pv, forecast_hourly, varying_price = read_inputs()
    load_pred = causal_profiles(load, base_load)
    pv_pred = causal_profiles(pv, base_pv)
    price_pred = causal_profiles(varying_price, fixed_price)
    np.savez_compressed(CACHE / "forecasts.npz", load_pred=load_pred, pv_pred=pv_pred, price_pred=price_pred)

    wanted = {x.strip() for x in args.questions.split(",")}
    if "all" in wanted:
        wanted = {"q1", "q2", "q3", "q4"}
    summary_path = CACHE / "summaries.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if "q1" in wanted:
        summaries["q1"] = solve_q1(fixed_price, base_load, base_pv)
    if "q2" in wanted:
        summaries["q2"] = solve_q2_or_q4(dates, fixed_price, load, pv, load_pred, pv_pred, label="q2")
    if "q3" in wanted:
        summaries["q3"] = solve_q3_or_q43(dates, fixed_price, load, pv, forecast_hourly, load_pred, label="q3")
    if "q4" in wanted:
        summaries["q4_2"] = solve_q2_or_q4(
            dates, fixed_price, load, pv, load_pred, pv_pred,
            varying_price=varying_price, price_pred=price_pred, label="q4_2",
        )
        summaries["q4_3"] = solve_q3_or_q43(
            dates, fixed_price, load, pv, forecast_hourly, load_pred,
            varying_price=varying_price, price_pred=price_pred, label="q4_3",
        )
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
