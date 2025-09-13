import json
import pandas as pd


def summarize_dataframe(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {
            "count": 0,
            "quantiles": {},
            "top_states": [],
            "construction_mix": {},
            "lob_mix": {},
        }

    def q(obj, cols):
        out = {}
        for c, _ in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            qs = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            out[c] = {"p05": float(qs.loc[0.05]), "p25": float(qs.loc[0.25]), "p50": float(qs.loc[0.5]), "p75": float(qs.loc[0.75]), "p95": float(qs.loc[0.95])}
        return out

    quants = q(df, [
        ("total_premium", True),
        ("tiv", True),
        ("loss_ratio", True),
        ("winnability", True),
        ("building_year", True),
    ])

    top_states = (
        df["primary_risk_state"].fillna("UNK").astype(str).str.upper().value_counts().head(10).to_dict()
        if "primary_risk_state" in df.columns else {}
    )
    construction_mix = (
        df["construction_type"].fillna("UNK").astype(str).value_counts().head(10).to_dict()
        if "construction_type" in df.columns else {}
    )
    lob_mix = (
        df["line_of_business"].fillna("UNK").astype(str).str.upper().value_counts().head(10).to_dict()
        if "line_of_business" in df.columns else {}
    )

    return {
        "count": int(len(df)),
        "quantiles": quants,
        "top_states": top_states,
        "construction_mix": construction_mix,
        "lob_mix": lob_mix,
    }


