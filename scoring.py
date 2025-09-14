# scoring.py
import numpy as np

def zscore(x, lo, hi):
    if x is None or np.isnan(x): return 0.0
    if lo == hi: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def sweet_spot(x, lo, mid_lo, mid_hi, hi):
    if x is None or np.isnan(x): return 0.0
    if x < lo or x > hi: return 0.0
    if x <= mid_lo: return (x - lo) / (mid_lo - lo + 1e-9)
    if x >= mid_hi: return (hi - x) / (hi - mid_hi + 1e-9)
    return 1.0

def geo_penalty(state_counts, state):
    if not state or state not in state_counts: return 0.0
    total = sum(state_counts.values()) or 1
    share = state_counts[state] / total
    if share <= 0.15: return 0.0
    if share <= 0.25: return 0.1
    if share <= 0.35: return 0.2
    return 0.3  # heavy concentration penalty

def score_row(r, weights, state_counts):
    prem = r.get("total_premium")
    win = r.get("winnability")
    year = r.get("building_year")
    good_constr = 1.0 if r.get("is_good_construction") else 0.0
    tiv = r.get("tiv")
    fresh = r.get("fresh_days")

    s_prem = sweet_spot(prem, weights["premium_lo"], weights["premium_mid_lo"], weights["premium_mid_hi"], weights["premium_hi"])
    s_win  = 0.0 if win is None or np.isnan(win) else float(win)  # already 0..1
    s_year = 0.0 if year is None or np.isnan(year) else zscore(year, 1990, 2022)
    s_con  = good_constr
    s_tiv  = 1.0 - zscore(tiv, 0, weights["tiv_hi"]) if tiv is not None and not np.isnan(tiv) else 0.0
    s_fresh= 1.0 - zscore(fresh, 0, 90) if fresh is not None and not np.isnan(fresh) else 0.5

    base = (
        weights["w_prem"]*s_prem +
        weights["w_win"]*s_win +
        weights["w_year"]*s_year +
        weights["w_con"]*s_con +
        weights["w_tiv"]*s_tiv +
        weights["w_fresh"]*s_fresh
    )
    penalty = geo_penalty(state_counts, r.get("primary_risk_state"))
    return max(0.0, base - penalty)


def compute_priority_score(appetite: float, win_norm: float, prem_norm: float, fresh_norm: float) -> float:
    """Blend core components into a 0-10 priority score.

    Args:
        appetite: Appetite or mode score on a 0-10 scale.
        win_norm: Winnability normalized to 0-1.
        prem_norm: Premium factor normalized to 0-1.
        fresh_norm: Freshness factor normalized to 0-1.

    Returns:
        Priority score clamped to the 0-10 range.
    """

    priority = (
        appetite * 0.4
        + win_norm * 10.0 * 0.3
        + prem_norm * 10.0 * 0.2
        + fresh_norm * 10.0 * 0.1
    )
    return max(0.1, min(10.0, priority))
