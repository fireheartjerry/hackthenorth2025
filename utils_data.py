# utils_data.py
import json, pandas as pd, numpy as np
from datetime import datetime, timezone

ACCEPT_CONSTRUCTION = {
    "Joisted Masonry", "Non-Combustible", "Masonry Non-Combustible",
    "Fire Resistive", "Non Combustible", "Non Combustible/Steel"
}

def load_df(path="data.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    if isinstance(raw, dict) and "output" in raw and raw["output"]:
        items = raw["output"][0].get("data", [])
    elif isinstance(raw, dict) and "data" in raw:
        items = raw["data"]

    df = pd.DataFrame(items)
    if df.empty:
        return df

    # types
    for c in ["tiv", "total_premium", "winnability"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "loss_value" in df.columns:
        df["loss_value"] = pd.to_numeric(df["loss_value"], errors="coerce")

    for c in ["created_at", "effective_date", "expiration_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    now = datetime.now(timezone.utc)
    df["fresh_days"] = (now - df["created_at"]).dt.days.replace({pd.NaT: np.nan})

    if "oldest_building" in df.columns:
        df["building_year"] = pd.to_numeric(df["oldest_building"], errors="coerce")
    else:
        df["building_year"] = np.nan

    # normalize winnability 0-1
    if "winnability" in df.columns:
        df["winnability"] = df["winnability"].apply(
            lambda x: np.nan if pd.isna(x) else (x/100.0 if x>1 else float(x))
        )

    # derived
    df["loss_ratio"] = (df["loss_value"] / df["total_premium"]).replace([np.inf, -np.inf], np.nan)
    df["is_good_construction"] = df["construction_type"].isin(ACCEPT_CONSTRUCTION)

    return df
