import os
import json
import math
import requests
import pandas as pd
import numpy as np


def _num(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        return float(str(x).replace(",", "").strip())
    except Exception:
        return default


def summarize_dataframe(df: "pd.DataFrame") -> dict:
    """
    Returns a compact JSON summary of df distributions:
    premium/tiv/loss_ratio/winnability/building_year quantiles,
    top_states, construction_mix, lob_mix, count.
    """
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
        for c, qlist in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            qs = np.nanpercentile(s.values, [5, 25, 50, 75, 95]) if s.notna().any() else [None]*5
            out[c] = {
                "p05": _num(qs[0], None),
                "p25": _num(qs[1], None),
                "p50": _num(qs[2], None),
                "p75": _num(qs[3], None),
                "p95": _num(qs[4], None),
            }
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


def _normalize_weights(w: dict) -> dict:
    keys = ["w_prem", "w_win", "w_year", "w_con", "w_tiv", "w_fresh"]
    # clamp to [0,1]
    for k in keys:
        v = _num(w.get(k), 0.0) or 0.0
        w[k] = max(0.0, min(1.0, float(v)))
    s = sum(w[k] for k in keys)
    if s <= 0:
        for k in keys:
            w[k] = 1.0/len(keys)
    else:
        for k in keys:
            w[k] = w[k] / s
    # ensure numeric bands
    for k in ["premium_lo", "premium_mid_lo", "premium_mid_hi", "premium_hi", "tiv_hi"]:
        if k in w:
            w[k] = float(_num(w.get(k), 0.0) or 0.0)
    # fix premium band order
    plo = _num(w.get("premium_lo"), 0.0) or 0.0
    pml = _num(w.get("premium_mid_lo"), plo)
    pmh = _num(w.get("premium_mid_hi"), pml)
    phi = _num(w.get("premium_hi"), pmh)
    ordered = sorted([plo, pml, pmh, phi])
    w["premium_lo"], w["premium_mid_lo"], w["premium_mid_hi"], w["premium_hi"] = [float(x) for x in ordered]
    # tiv_hi sanity
    th = _num(w.get("tiv_hi"), None)
    if th is None or th <= 0:
        w["tiv_hi"] = float(max(ordered[-1]*10, 1_000_000))
    return w


def _sanitize_filters(f: dict) -> dict:
    out = {}
    # pass-through allowed keys only
    allowed = {
        "lob_in", "new_business_only", "loss_ratio_max", "tiv_max",
        "premium_range", "min_winnability", "min_year", "good_construction_only"
    }
    for k, v in (f or {}).items():
        if k in allowed:
            out[k] = v

    # types & clamps
    if "lob_in" in out and isinstance(out["lob_in"], list):
        out["lob_in"] = [str(x).upper() for x in out["lob_in"]]
    if "new_business_only" in out:
        out["new_business_only"] = bool(out["new_business_only"])
    if "loss_ratio_max" in out:
        out["loss_ratio_max"] = float(max(0.0, min(2.0, _num(out["loss_ratio_max"], 0.7) or 0.7)))
    if "tiv_max" in out:
        out["tiv_max"] = float(max(0.0, _num(out["tiv_max"], 150_000_000) or 150_000_000))
    if "premium_range" in out and isinstance(out["premium_range"], (list, tuple)) and len(out["premium_range"]) == 2:
        lo = float(_num(out["premium_range"][0], 0.0) or 0.0)
        hi = float(_num(out["premium_range"][1], lo) or lo)
        if hi < lo:
            lo, hi = hi, lo
        out["premium_range"] = [lo, hi]
    if "min_winnability" in out:
        out["min_winnability"] = float(max(0.0, min(1.0, _num(out["min_winnability"], 0.5) or 0.5)))
    if "min_year" in out:
        out["min_year"] = int(_num(out["min_year"], 2000) or 2000)
    if "good_construction_only" in out:
        out["good_construction_only"] = bool(out["good_construction_only"])
    return out


def _sanitize_proposal(d: dict) -> dict:
    if not isinstance(d, dict):
        return {"filters": {}, "weights": {}}
    f = _sanitize_filters(d.get("filters", {}))
    w = _normalize_weights(d.get("weights", {}))
    # ensure required bands if missing
    if "premium_lo" not in w or "premium_hi" not in w:
        lo = float(f.get("premium_range", [0.0, 1_500_000])[0])
        hi = float(f.get("premium_range", [0.0, 1_500_000])[1])
        mid_lo = lo + 0.3 * (hi - lo)
        mid_hi = lo + 0.7 * (hi - lo)
        w.update({
            "premium_lo": lo,
            "premium_mid_lo": mid_lo,
            "premium_mid_hi": mid_hi,
            "premium_hi": hi,
        })
        w = _normalize_weights(w)
    if "tiv_hi" not in w:
        w["tiv_hi"] = float(f.get("tiv_max", 150_000_000))
    return {"filters": f, "weights": w}


def seeds_from_answers(answers: dict) -> dict:
    """
    Deterministically maps questionnaire answers to initial
    {'filters': {...}, 'weights': {...}} seed rules.
    """
    a = answers or {}
    aggr = int(a.get("aggressiveness", 3) or 3)
    aggr = max(1, min(5, aggr))
    objective = (a.get("objective") or "balance").strip().lower()
    prem_lo = _num(a.get("premium_lo"), 100_000) or 100_000
    prem_hi = _num(a.get("premium_hi"), 1_500_000) or 1_500_000
    if prem_hi < prem_lo:
        prem_lo, prem_hi = prem_hi, prem_lo
    max_lr = _num(a.get("max_lr"), 0.65) or 0.65
    tiv_max = _num(a.get("tiv_max"), 150_000_000) or 150_000_000
    new_only = bool(a.get("new_only", True))
    strict_construction = bool(a.get("strict_construction", False))
    con_pref = a.get("construction_pref") or []

    # map aggressiveness → default thresholds
    lr_map = {1: 0.50, 2: 0.60, 3: 0.70, 4: 0.80, 5: 0.90}
    win_map = {1: 0.60, 2: 0.55, 3: 0.50, 4: 0.45, 5: 0.40}
    year_map = {1: 2010, 2: 2005, 3: 2000, 4: 1995, 5: 1990}

    filters = {
        "lob_in": ["COMMERCIAL PROPERTY"],
        "new_business_only": new_only,
        "loss_ratio_max": float(min(max_lr, lr_map[aggr])),
        "tiv_max": float(tiv_max),
        "premium_range": [float(prem_lo), float(prem_hi)],
        "min_winnability": float(win_map[aggr]),
        "min_year": int(year_map[aggr]),
        "good_construction_only": bool(strict_construction),
    }

    # weights by objective
    if objective == "expected_premium":
        core = dict(w_prem=0.30, w_win=0.25, w_year=0.15, w_con=0.15, w_tiv=0.10, w_fresh=0.05)
    elif objective == "win_rate":
        core = dict(w_prem=0.20, w_win=0.35, w_year=0.15, w_con=0.15, w_tiv=0.10, w_fresh=0.05)
    elif objective == "freshness":
        core = dict(w_prem=0.15, w_win=0.25, w_year=0.10, w_con=0.10, w_tiv=0.05, w_fresh=0.35)
    else:  # balance
        core = dict(w_prem=0.25, w_win=0.25, w_year=0.15, w_con=0.15, w_tiv=0.10, w_fresh=0.10)

    span = prem_hi - prem_lo
    mid_lo = prem_lo + 0.30 * span
    mid_hi = prem_lo + 0.70 * span

    weights = {
        **core,
        "premium_lo": float(prem_lo),
        "premium_mid_lo": float(mid_lo),
        "premium_mid_hi": float(mid_hi),
        "premium_hi": float(prem_hi),
        "tiv_hi": float(tiv_max),
    }

    return _sanitize_proposal({"filters": filters, "weights": weights})


def propose_with_gemini(summary_json: dict, seed_rules: dict, goal_text: str) -> dict:
    """
    Calls Gemini (model from env GEMINI_MODEL, key GEMINI_API_KEY).
    Prompt instructs STRICT JSON with keys 'filters' and 'weights'.
    Returns sanitized dict suitable for scoring engine.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
    if not api_key:
        # No Gemini available: return seed as proposal
        return _sanitize_proposal(seed_rules)

    # Build strict prompt
    allowed_filters = [
        "lob_in","new_business_only","loss_ratio_max","tiv_max",
        "premium_range","min_winnability","min_year","good_construction_only"
    ]
    allowed_weights = [
        "w_prem","w_win","w_year","w_con","w_tiv","w_fresh",
        "premium_lo","premium_mid_lo","premium_mid_hi","premium_hi","tiv_hi"
    ]
    guard = (
        "Return JSON ONLY with keys 'filters' and 'weights'. "
        "Do not include backticks or code fences. "
        "filters must only contain: " + ", ".join(allowed_filters) + ". "
        "weights must only contain: " + ", ".join(allowed_weights) + ". "
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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {"temperature": 0.2}
    }

    text = None
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        j = resp.json()
        cands = (j.get("candidates") or [])
        if cands:
            parts = ((cands[0] or {}).get("content") or {}).get("parts") or []
            if parts and isinstance(parts[0], dict):
                text = parts[0].get("text")
    except Exception:
        return _sanitize_proposal(seed_rules)

    if not text:
        return _sanitize_proposal(seed_rules)

    # Strip code fences and extract JSON object
    s = str(text).strip()
    if s.startswith("```"):
        s = s.strip("`\n ")
        # remove possible 'json' language tag
        if s.lower().startswith("json"):
            s = s[4:].strip()
    # Extract first {...}
    try:
        start = s.index("{")
        end = s.rindex("}")
        s = s[start:end+1]
    except ValueError:
        pass

    try:
        raw = json.loads(s)
    except Exception:
        # very defensive fallback
        return _sanitize_proposal(seed_rules)

    return _sanitize_proposal(raw)


def explain_with_gemini(item: dict, rule_text: str, label: str, raw_reason: list) -> dict:
    """
    Calls Gemini to generate a concise explanation for why a submission 
    is classified as IN/OUT/TARGET appetite.
    
    Returns dict with keys: 'explanation', 'ai_used', 'fallback_reason'
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Fallback explanation
    fallback_explanation = f"{label}: " + (
        "; ".join(raw_reason) 
        if raw_reason 
        else "Meets appetite and target criteria"
    )
    
    if not api_key:
        return {
            "explanation": fallback_explanation,
            "ai_used": False,
            "fallback_reason": "No API key provided"
        }
    
    # Build prompt for explanation
    prompt = (
        f"You are an underwriting assistant. Explain in one concise sentence why this submission is {label}. "
        f"Use the guidelines and the submission fields to provide a clear, professional explanation. "
        f"Focus on the key factors that led to this classification.\n\n"
        f"Guidelines: {rule_text}\n\n"
        f"Submission Details: {json.dumps(item, default=str)}\n\n"
        f"Classification: {label}\n"
        f"Rule-based reasons: {'; '.join(raw_reason) if raw_reason else 'Meets all criteria'}"
    )
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 120}
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        j = resp.json()
        cands = (j.get("candidates") or [])
        if cands:
            parts = ((cands[0] or {}).get("content") or {}).get("parts") or []
            if parts and isinstance(parts[0], dict):
                text = parts[0].get("text", "").strip()
                if text:
                    return {
                        "explanation": text,
                        "ai_used": True,
                        "fallback_reason": None
                    }
    except Exception as e:
        return {
            "explanation": fallback_explanation,
            "ai_used": False,
            "fallback_reason": f"API error: {str(e)}"
        }
    
    return {
        "explanation": fallback_explanation,
        "ai_used": False,
        "fallback_reason": "No response from API"
    }


def nlq_with_gemini(query: str) -> dict:
    """
    Calls Gemini to parse natural language queries into structured filters.
    
    Returns dict with keys: 'filters', 'ai_used', 'fallback_reason', 'ai_summary'
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Heuristic fallback parsing
    def heuristic_parse(q):
        st = None
        status = None
        min_p = None
        max_p = None
        search = None
        
        tokens = q.lower().split()
        for tok in tokens:
            if len(tok) == 2 and tok.upper() in {"OH", "PA", "MD", "CO", "CA", "FL", "NC", "SC", "GA", "VA", "UT", "TX", "NY"}:
                st = tok.upper()
            elif "target" in tok:
                status = "TARGET"
            elif "out" in tok:
                status = "OUT"
            elif "in" in tok and status is None:
                status = "IN"
            elif "k" in tok or "$" in tok:
                try:
                    num = float(tok.replace("k", "").replace("$", "").replace(",", ""))
                    if "k" in tok:
                        num *= 1000
                    if min_p is None:
                        min_p = num
                    else:
                        max_p = num
                except:
                    pass
            elif not tok in {"show", "find", "get", "premium", "over", "under", "above", "below", "with", "in", "at"}:
                if search is None:
                    search = tok
                else:
                    search += " " + tok
        
        return {
            "state": st,
            "status": status,
            "min_premium": min_p,
            "max_premium": max_p,
            "search": search
        }
    
    fallback_filters = heuristic_parse(query)
    
    if not api_key:
        return {
            "filters": fallback_filters,
            "ai_used": False,
            "fallback_reason": "No API key provided",
            "ai_summary": "Used heuristic parsing to extract filters from query"
        }
    
    # Build prompt for NLQ parsing
    system_prompt = (
        "You are an expert at parsing natural language queries for underwriting data filtering. "
        "Extract structured filters from the user's query and respond with ONLY valid JSON. "
        "Do not include backticks or code fences. "
        "\n\nSupported filters:"
        "\n- state: 2-letter US state code or null"
        "\n- status: 'IN' | 'OUT' | 'TARGET' | null (appetite classification)"
        "\n- min_premium: number or null (minimum premium in dollars)"
        "\n- max_premium: number or null (maximum premium in dollars)"  
        "\n- search: string or null (general text search)"
        "\n\nExamples:"
        "\n'Show targets in CA over $100k premium' → {\"state\": \"CA\", \"status\": \"TARGET\", \"min_premium\": 100000, \"max_premium\": null, \"search\": null}"
        "\n'Out of appetite submissions' → {\"state\": null, \"status\": \"OUT\", \"min_premium\": null, \"max_premium\": null, \"search\": null}"
        "\n'Florida properties under 50k' → {\"state\": \"FL\", \"status\": null, \"min_premium\": null, \"max_premium\": 50000, \"search\": \"properties\"}"
    )
    
    prompt = f"Query: {query}"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": system_prompt + "\n\n" + prompt}]
        }],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 120}
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        j = resp.json()
        cands = (j.get("candidates") or [])
        if cands:
            parts = ((cands[0] or {}).get("content") or {}).get("parts") or []
            if parts and isinstance(parts[0], dict):
                text = parts[0].get("text", "").strip()
                
                # Strip code fences and extract JSON
                if text.startswith("```"):
                    text = text.strip("`\n ")
                    if text.lower().startswith("json"):
                        text = text[4:].strip()
                
                # Extract first {...}
                try:
                    start = text.index("{")
                    end = text.rindex("}")
                    text = text[start:end+1]
                except ValueError:
                    pass
                
                try:
                    parsed_filters = json.loads(text)
                    # Validate the structure
                    valid_keys = {"state", "status", "min_premium", "max_premium", "search"}
                    filtered_result = {k: v for k, v in parsed_filters.items() if k in valid_keys}
                    
                    return {
                        "filters": filtered_result,
                        "ai_used": True,
                        "fallback_reason": None,
                        "ai_summary": f"Gemini parsed query '{query}' into structured filters"
                    }
                except json.JSONDecodeError:
                    pass
                    
    except Exception as e:
        return {
            "filters": fallback_filters,
            "ai_used": False,
            "fallback_reason": f"API error: {str(e)}",
            "ai_summary": "Used heuristic parsing due to API error"
        }
    
    return {
        "filters": fallback_filters,
        "ai_used": False,
        "fallback_reason": "Invalid response from API",
        "ai_summary": "Used heuristic parsing due to invalid API response"
    }
