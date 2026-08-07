from __future__ import annotations

import pandas as pd


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Tenure" in data.columns:
        data["TenureGroup"] = pd.cut(
            data["Tenure"],
            bins=[-1, 2, 5, 8, 100],
            labels=["new", "developing", "loyal", "veteran"],
        ).astype(str)
    if {"Balance", "EstimatedSalary"}.issubset(data.columns):
        data["BalanceToSalaryRatio"] = data["Balance"] / data["EstimatedSalary"].replace(0, 1)
    if "Balance" in data.columns:
        data["BalanceBucket"] = pd.cut(
            data["Balance"],
            bins=[-1, 0, 50000, 100000, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(int)
    return data
