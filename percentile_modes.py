import numpy as np
import pandas as pd


DEFAULT_PCTS = {
    "unicorn": {"top_pct": 1, "max_lr_pct": 60, "min_win_pct": 70, "max_tiv_pct": 80, "fresh_pct": 40},
    "balanced": {"top_pct": 20, "max_lr_pct": 70, "min_win_pct": 50, "max_tiv_pct": 90, "fresh_pct": 60},
    "loose": {"top_pct": 80, "max_lr_pct": 90, "min_win_pct": 30, "max_tiv_pct": 100, "fresh_pct": 80},
    "turnaround": {"top_pct": 40, "max_lr_pct": 85, "min_win_pct": 40, "max_tiv_pct": 70, "fresh_pct": 40},
}


def _pct(series, p):
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    try:
        return float(np.nanpercentile(s.values, p))
    except Exception:
        return None


def summarize_percentiles(df: pd.DataFrame):
    """Return selected percentile summaries for key metrics.

    Used for transparency and optional UI overlays.
    """
    return {
        "premium": {k: _pct(df.get("total_premium"), k) for k in (1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 95, 99)},
        "tiv": {k: _pct(df.get("tiv"), k) for k in (1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 95, 99)},
        "lr": {k: _pct(df.get("loss_ratio"), k) for k in (10, 20, 40, 50, 60, 70, 80, 90)},
        "win": {k: _pct(df.get("winnability"), k) for k in (10, 20, 40, 50, 60, 70, 80, 90)},
        "fresh": {k: _pct(df.get("fresh_days"), k) for k in (10, 20, 40, 50, 60, 70, 80, 90)},
    }


def build_percentile_filters(df: pd.DataFrame, kind: str = "balanced", overrides: dict | None = None):
    """Build soft filters using dataset percentiles.

    kind: one of [unicorn, balanced, loose, turnaround]
    overrides: optional dict to override percentiles (e.g., {"top_pct": 1})
    """
    cfg = DEFAULT_PCTS.get(kind, DEFAULT_PCTS["balanced"]).copy()
    if overrides:
        # Only apply recognized keys
        for k in ("top_pct", "max_lr_pct", "min_win_pct", "max_tiv_pct", "fresh_pct"):
            if k in overrides:
                try:
                    cfg[k] = int(overrides[k])
                except Exception:
                    pass

    s = summarize_percentiles(df)

    max_lr = s["lr"].get(cfg["max_lr_pct"]) if s["lr"] else None
    min_win = s["win"].get(cfg["min_win_pct"]) if s["win"] else None
    max_tiv = s["tiv"].get(cfg["max_tiv_pct"]) if s["tiv"] else None
    max_fresh_days = s["fresh"].get(cfg["fresh_pct"]) if s["fresh"] else None

    # premium band per mode
    if kind == "unicorn":
        lo = s["premium"].get(90)
        hi = s["premium"].get(99)
    elif kind == "turnaround":
        lo = s["premium"].get(20)
        hi = s["premium"].get(60)
    elif kind == "loose":
        lo = s["premium"].get(10)
        hi = s["premium"].get(90)
    else:  # balanced
        lo = s["premium"].get(40)
        hi = s["premium"].get(80)

    # Fallbacks if percentiles missing
    if max_lr is None:
        max_lr = 0.8
    if min_win is None:
        min_win = 0.5
    if max_tiv is None:
        try:
            max_tiv = float(pd.to_numeric(df.get("tiv"), errors="coerce").dropna().max())
        except Exception:
            max_tiv = None
    prem_series = pd.to_numeric(df.get("total_premium"), errors="coerce")
    try:
        prem_min = float(prem_series.dropna().min())
        prem_max = float(prem_series.dropna().max())
    except Exception:
        prem_min, prem_max = None, None

    lo = lo if lo is not None else prem_min
    hi = hi if hi is not None else prem_max
    # Ensure ordered bounds if both present
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo

    filters = {
        "loss_ratio_max": max_lr,
        "min_winnability": min_win,
        "tiv_max": max_tiv,
        "premium_range": [lo, hi],
        "max_fresh_days": max_fresh_days if max_fresh_days is not None else df.get("fresh_days").quantile(0.8),
        # keep optional constraints flexible by default
        # consumers may still pass "lob_in" or "good_construction_only" explicitly
    }
    return filters, s

