import os
import json
from typing import Dict

import google.generativeai as genai


ALLOWED_FILTERS = {
    "lob_in", "new_business_only", "loss_ratio_max", "tiv_max",
    "premium_range", "min_winnability", "min_year", "good_construction_only",
}
ALLOWED_WEIGHTS = {
    "w_prem", "w_win", "w_year", "w_con", "w_tiv", "w_fresh",
    "premium_lo", "premium_mid_lo", "premium_mid_hi", "premium_hi", "tiv_hi",
}


def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(str(x).replace(",", "").strip())
    except Exception:
        return default


def _normalize_weights(d: Dict) -> Dict:
    out = dict(d or {})
    # numeric clamps
    for k in list(out.keys()):
        if k in {"premium_range"}:
            continue
        if k in ALLOWED_WEIGHTS:
            out[k] = _num(out.get(k), 0.0) or 0.0

    # ensure premium band
    plo = _num(out.get("premium_lo"), 100_000) or 100_000
    phi = _num(out.get("premium_hi"), 1_500_000) or 1_500_000
    if phi < plo:
        plo, phi = phi, plo
    span = max(1.0, phi - plo)
    pml = _num(out.get("premium_mid_lo"), plo + 0.30 * span)
    pmh = _num(out.get("premium_mid_hi"), plo + 0.70 * span)
    ordered = sorted([plo, pml, pmh, phi])
    out["premium_lo"], out["premium_mid_lo"], out["premium_mid_hi"], out["premium_hi"] = [float(x) for x in ordered]

    # tiv
    tiv_hi = _num(out.get("tiv_hi"), max(ordered[-1] * 10.0, 1_000_000.0)) or max(ordered[-1] * 10.0, 1_000_000.0)
    out["tiv_hi"] = float(max(1_000_000.0, tiv_hi))

    # core weights normalization
    core = ["w_prem", "w_win", "w_year", "w_con", "w_tiv", "w_fresh"]
    total = sum(float(out.get(k, 0.0) or 0.0) for k in core)
    if total <= 0:
        for k in core:
            out[k] = 1.0 / len(core)
    else:
        for k in core:
            out[k] = float(out.get(k, 0.0) or 0.0) / total
    return out


def _sanitize(d: Dict) -> Dict:
    if not isinstance(d, dict):
        return {"filters": {}, "weights": {}}
    f = d.get("filters", {}) or {}
    w = d.get("weights", {}) or {}

    # keep only allowed
    f2 = {}
    for k, v in f.items():
        if k in ALLOWED_FILTERS:
            if k == "premium_range" and isinstance(v, (list, tuple)) and len(v) == 2:
                lo = _num(v[0], 0.0) or 0.0
                hi = _num(v[1], lo) or lo
                if hi < lo: lo, hi = hi, lo
                f2[k] = [float(lo), float(hi)]
            elif k in {"lob_in"} and isinstance(v, list):
                f2[k] = [str(x).upper() for x in v]
            elif k in {"new_business_only", "good_construction_only"}:
                f2[k] = bool(v)
            elif k in {"loss_ratio_max", "tiv_max", "min_winnability"}:
                f2[k] = float(_num(v, 0.0) or 0.0)
            elif k in {"min_year"}:
                try:
                    f2[k] = int(_num(v, 2000) or 2000)
                except Exception:
                    f2[k] = 2000
    w2 = {k: v for k, v in w.items() if k in ALLOWED_WEIGHTS}
    w2 = _normalize_weights(w2)
    return {"filters": f2, "weights": w2}


def propose_with_gemini(summary_json: dict, seed_rules: dict, goal_text: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    if not api_key:
        # No Gemini available: return seed as proposal
        return _sanitize(seed_rules)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    guard = (
        "Return JSON ONLY with keys 'filters' and 'weights'. "
        "Do not include backticks or code fences. "
        "filters must only contain: " + ", ".join(sorted(list(ALLOWED_FILTERS))) + ". "
        "weights must only contain: " + ", ".join(sorted(list(ALLOWED_WEIGHTS))) + ". "
        "Clamp numeric values to reasonable ranges; use numbers, not strings. "
        "premium_range is [lo, hi] with lo<=hi, in dollars. "
        "loss_ratio_max is 0.0..1.5; min_winnability is 0.0..1.0. "
        "Renormalize core weights (w_*) to sum to 1.0."
    )
    prompt = (
        f"{guard}\nDataset summary: {json.dumps(summary_json)}\nSeed rules: {json.dumps(seed_rules)}\n"
        f"Goal: {(goal_text or '').strip()}\n"
        "Adjust the seed to meet the goal and dataset context, but keep it realistic."
    )

    try:
        resp = model.generate_content(prompt)
        text = resp.text or ""
    except Exception:
        return _sanitize(seed_rules)

    if not text:
        return _sanitize(seed_rules)

    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`\n ")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        start = s.index("{")
        end = s.rindex("}")
        s = s[start : end + 1]
    except ValueError:
        pass

    try:
        raw = json.loads(s)
    except Exception:
        return _sanitize(seed_rules)

    return _sanitize(raw)


