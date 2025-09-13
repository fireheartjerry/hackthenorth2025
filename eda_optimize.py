# eda_optimize.py
import numpy as np, pandas as pd
from utils_data import load_df

def evaluate(df, max_lr=0.6, premium_min=100_000, premium_max=2_000_000,
             tiv_max=150_000_000, min_win=0.5, year_min=2000):
    d = df.copy()
    d = d[d["loss_ratio"] <= max_lr]
    d = d[(d["total_premium"] >= premium_min) & (d["total_premium"] <= premium_max)]
    d = d[d["tiv"] <= tiv_max]
    d = d[(d["winnability"] >= min_win)]
    d = d[(d["building_year"] >= year_min) | (d["building_year"].isna())]
    # objective: expected premium proxy
    d["exp_value"] = d["total_premium"] * d["winnability"]
    return {
        "count": len(d),
        "avg_lr": float(d["loss_ratio"].mean()) if len(d) else None,
        "avg_exp_value": float(d["exp_value"].mean()) if len(d) else 0.0,
        "sum_exp_value": float(d["exp_value"].sum()) if len(d) else 0.0,
        "premium_range": (premium_min, premium_max),
        "max_lr": max_lr,
        "tiv_max": tiv_max,
        "min_win": min_win,
        "year_min": year_min
    }

def grid_search(df):
    lr_grid = [0.4, 0.5, 0.6, 0.7]
    prem_bands = [(100_000, 400_000), (150_000, 800_000), (200_000, 1_800_000)]
    tiv_grid = [75_000_000, 100_000_000, 150_000_000]
    win_grid = [0.4, 0.5, 0.6, 0.7]
    year_grid = [1990, 2000, 2010]

    rows = []
    for lr in lr_grid:
        for pr in prem_bands:
            for tiv in tiv_grid:
                for w in win_grid:
                    for yr in year_grid:
                        r = evaluate(df, max_lr=lr, premium_min=pr[0], premium_max=pr[1],
                                     tiv_max=tiv, min_win=w, year_min=yr)
                        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["sum_exp_value","avg_exp_value","count"], ascending=False)
    return out

if __name__ == "__main__":
    df = load_df("data.json")
    report = grid_search(df)
    print(report.head(20).to_string(index=False))
