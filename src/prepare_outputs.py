"""校验数值结果并生成供 Artifact Tool 写入模板的 JSON 载荷。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


DT, ETA = 1 / 6, 0.9
ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "C题项目"
CACHE = PROJECT / "中间数据"
ATTACH = ROOT / "题目" / "C题" / "附件"


def clean(a):
    return np.where(np.abs(a) < 5e-7, 0.0, a)


def load_source():
    a1 = pd.read_excel(ATTACH / "附件1.xlsx")
    load = pd.read_excel(ATTACH / "附件2.xlsx", sheet_name="小区负载", index_col=0).to_numpy(float)
    pv = pd.read_excel(ATTACH / "附件2.xlsx", sheet_name="光伏发电实际功率", index_col=0).to_numpy(float)
    price = pd.read_excel(ATTACH / "附件4.xlsx", index_col=0).to_numpy(float)
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    return (
        dates,
        a1.iloc[:, 1].to_numpy(float),
        a1.iloc[:, 2].to_numpy(float),
        a1.iloc[:, 3].to_numpy(float),
        load,
        pv,
        price,
    )


def validate_one(label, data, price, load_kw, pv_kw, adjusted=False):
    purchase = data["adjusted"] if adjusted else data["purchase"]
    bal = purchase + pv_kw * DT + data["discharge"] + data["emergency"] - load_kw * DT - data["charge"] - data["spill"]
    transition = data["soc"][:, 1:] - data["soc"][:, :-1] - ETA * data["charge"] + data["discharge"] / ETA
    simultaneous = np.minimum(data["charge"], data["discharge"])
    return {
        "label": label,
        "shape": list(purchase.shape),
        "balance_max_abs_kwh": float(np.max(np.abs(bal))),
        "soc_transition_max_abs_kwh": float(np.max(np.abs(transition))),
        "soc_min_kwh": float(np.min(data["soc"])),
        "soc_max_kwh": float(np.max(data["soc"])),
        "daily_close_max_abs_kwh": float(np.max(np.abs(data["soc"][:, -1] - 6000))),
        "simultaneous_charge_discharge_max_kwh": float(np.max(simultaneous)),
        "negative_min_kwh": float(min(np.min(purchase), np.min(data["charge"]), np.min(data["discharge"]), np.min(data["emergency"]))),
        "actual_total_cost": float(
            np.sum(price * purchase)
            + np.sum(5 * price * data["emergency"])
            + (np.sum(0.5 * price * np.abs(data["adjusted"] - data["purchase"])) if adjusted else 0)
        ),
    }


def time_labels():
    wb = load_workbook(ATTACH / "附件5" / "result2.xlsx", read_only=True, data_only=True)
    ws = wb["计划购电量"]
    return [ws.cell(1, c).value for c in range(2, 146)]


def emergency_rows(dates, emergency, labels):
    rows = []
    for i, day in enumerate(dates):
        active = emergency[i] > 5e-6
        t = 0
        while t < 144:
            if not active[t]:
                t += 1
                continue
            start = t
            amount = 0.0
            while t < 144 and active[t]:
                amount += emergency[i, t]
                t += 1
            # 模板标签的左端点与原始附件顺序完全一致。
            start_text = labels[start].split("-")[0]
            end_text = labels[t - 1].split("-")[-1]
            rows.append([day.strftime("%Y-%m-%d"), f"{start_text}-{end_text}", round(float(amount), 6)])
    return rows


def charge_rows(dates, data):
    rows = []
    for i, day in enumerate(dates):
        for block in range(6):
            a, b = 24 * block, 24 * (block + 1)
            rows.append(
                [
                    day.strftime("%Y-%m-%d") if block == 0 else None,
                    f"{4*block}:00-{4*(block+1)}:00",
                    round(float(np.sum(data["charge"][i, a:b])), 6),
                    round(float(np.sum(data["discharge"][i, a:b])), 6),
                    "0:00" if block == 0 else ("24:00" if block == 1 else None),
                    round(float(data["soc"][i, 0 if block == 0 else -1]), 6) if block < 2 else None,
                ]
            )
    return rows


def plan_rows(dates, purchase, price, baseline=None):
    rows = []
    for i, day in enumerate(dates):
        fee = float(np.sum(price[i] * purchase[i]))
        if baseline is not None:
            fee += float(np.sum(0.5 * price[i] * np.abs(purchase[i] - baseline[i])))
        rows.append(
            [day.strftime("%Y-%m-%d")]
            + [round(float(x), 6) for x in clean(purchase[i])]
            + [round(float(np.sum(purchase[i])), 6), round(fee, 6)]
        )
    return rows


def main():
    dates_all, fixed_price, base_load, base_pv, load, pv, varying_price = load_source()
    dates = dates_all[31:]
    labels = time_labels()
    summaries = json.loads((CACHE / "summaries.json").read_text(encoding="utf-8"))
    report = {}

    q1 = np.load(CACHE / "q1.npz")
    q1_bal = q1["purchase"] + base_pv * DT + q1["discharge"] - base_load * DT - q1["charge"] - q1["spill"]
    q1_trans = q1["soc"][1:] - q1["soc"][:-1] - ETA * q1["charge"] + q1["discharge"] / ETA
    report["q1"] = {
        "balance_max_abs_kwh": float(np.max(np.abs(q1_bal))),
        "soc_transition_max_abs_kwh": float(np.max(np.abs(q1_trans))),
        "soc_min_kwh": float(np.min(q1["soc"])),
        "soc_max_kwh": float(np.max(q1["soc"])),
        "soc_close_abs_kwh": float(abs(q1["soc"][-1] - q1["soc"][0])),
        "simultaneous_charge_discharge_max_kwh": float(np.max(np.minimum(q1["charge"], q1["discharge"]))),
    }

    payload = {"summary": summaries, "time_labels": labels}
    q1_charge = []
    for block in range(6):
        a, b = 24 * block, 24 * (block + 1)
        q1_charge.append(
            [
                f"{4*block}:00-{4*(block+1)}:00",
                round(float(np.sum(q1["charge"][a:b])), 6),
                round(float(np.sum(q1["discharge"][a:b])), 6),
                "0:00" if block == 0 else ("24:00" if block == 1 else None),
                6000.0 if block < 2 else None,
            ]
        )
    payload["q1"] = {
        "plan": [[labels[i], round(float(clean(q1["purchase"])[i]), 6)] for i in range(144)],
        "charge": q1_charge,
    }

    specs = [
        ("q2", fixed_price, False),
        ("q3", fixed_price, True),
        ("q4_2", varying_price[31:], False),
        ("q4_3", varying_price[31:], True),
    ]
    for label, p, adjusted in specs:
        data = np.load(CACHE / f"{label}.npz")
        price_matrix = np.repeat(p[None, :], len(dates), axis=0) if p.ndim == 1 else p
        report[label] = validate_one(label, data, price_matrix, load[31:], pv[31:], adjusted=adjusted)
        item = {
            "plan": plan_rows(dates, data["purchase"], price_matrix),
            "charge": charge_rows(dates, data),
            "emergency": emergency_rows(dates, data["emergency"], labels),
        }
        if adjusted:
            item["adjusted"] = plan_rows(dates, data["adjusted"], price_matrix, baseline=data["purchase"])
        payload[label] = item

    forecast = np.load(CACHE / "forecasts.npz")
    for key, actual in [("load", load), ("pv", pv), ("price", varying_price)]:
        pred = forecast[f"{key}_pred"][31:]
        obs = actual[31:]
        report[f"{key}_forecast"] = {
            "mae": float(np.mean(np.abs(pred - obs))),
            "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "nrmse_mean": float(np.sqrt(np.mean((pred - obs) ** 2)) / max(np.mean(obs), 1e-9)),
        }

    passed = True
    for name, row in report.items():
        if name.endswith("forecast"):
            continue
        passed &= row.get("balance_max_abs_kwh", 0) < 1e-4
        passed &= row.get("soc_transition_max_abs_kwh", 0) < 1e-4
        passed &= row.get("soc_min_kwh", 1200) >= 1200 - 1e-4
        passed &= row.get("soc_max_kwh", 10800) <= 10800 + 1e-4
        passed &= row.get("daily_close_max_abs_kwh", row.get("soc_close_abs_kwh", 0)) < 1e-4
        passed &= row.get("simultaneous_charge_discharge_max_kwh", 0) < 1e-4
        passed &= row.get("negative_min_kwh", 0) >= -1e-5
    report["all_constraints_passed"] = bool(passed)
    (CACHE / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (CACHE / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
