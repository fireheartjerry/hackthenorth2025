from flask import Flask, request, jsonify, render_template, make_response
import os, json, time, datetime as dt
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from collections import Counter

# Custom modules
from utils_data import load_df
from scoring import score_row
from guidelines import classify_for_mode_row
from modes import MODES
from persona_store import load_store as persona_load_store, save_store as persona_save_store, list_personas as persona_list, add_persona as persona_add, get_persona as persona_get, set_active as persona_set_active, get_active as persona_get_active
from ai_suggest import summarize_dataframe
from persona_seed import seeds_from_answers
from ai_constraints import propose_with_gemini
from gemini_test import explain_with_gemini, nlq_with_gemini
from utils_json import df_records_to_builtin, to_builtin
from percentile_modes import build_percentile_filters, summarize_percentiles
from blend import blend_modes

load_dotenv()

app = Flask(__name__)

# Custom Jinja2 filter for date formatting
@app.template_filter('dateformat')
def dateformat(value, format='%b %d, %Y'):
    """Format a date that could be a string or datetime object."""
    if not value:
        return '—'
    
    # If it's already a datetime object, use strftime directly
    if hasattr(value, 'strftime'):
        return value.strftime(format)
    
    # If it's a string, try to parse it first
    if isinstance(value, str):
        try:
            # Try parsing common datetime formats
            for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    dt_obj = dt.datetime.strptime(value, fmt)
                    return dt_obj.strftime(format)
                except ValueError:
                    continue
            # If parsing fails, return the string as-is
            return value
        except:
            return value
    
    return '—'

# Custom Jinja2 filter for comma formatting of large numbers
@app.template_filter('commafy')
def commafy(value, decimals=0):
    """Format a number with thousands separators."""
    try:
        fmt = f"{float(value):,.{decimals}f}"
        return fmt
    except (TypeError, ValueError):
        return value

FEDERATO_TOKEN = os.environ.get("FEDERATO_TOKEN")
CLIENT_ID = os.environ.get("FEDERATO_CLIENT_ID")
CLIENT_SECRET = os.environ.get("FEDERATO_CLIENT_SECRET")
AUDIENCE = os.environ.get("FEDERATO_AUDIENCE", "https://product.federato.ai/core-api")
AUTH_URL = os.environ.get(
    "FEDERATO_AUTH_URL", "https://product-federato.us.auth0.com/oauth/token"
)
POLICIES_URL_TYPO = os.environ.get(
    "FEDERATO_POLICIES_URL_TYPO",
    "https://product.federato.ai/integrations-api/handlers/all-pollicies?outputOnly=true",
)
POLICIES_URL = os.environ.get(
    "FEDERATO_POLICIES_URL",
    "https://product.federato.ai/integrations-api/handlers/all-policies?outputOnly=true",
)
USE_LOCAL_DATA = os.environ.get("USE_LOCAL_DATA", "false").lower() == "true"

# AI functionality now handled by Gemini
# GEMINI_API_KEY and GEMINI_MODEL are read directly in gemini_test.py

CACHE_TTL = 300
_cache_data = {"ts": 0, "df": None, "raw": None}

# Persona registry (slugs loaded from personas.json). We mirror saved personas into MODES
_PERSONA_INDEX = {}  # slug -> title

def _load_personas_into_modes():
    global _PERSONA_INDEX
    store = persona_load_store()
    items = store.get("items", {}) or {}
    _PERSONA_INDEX = {slug: rec.get("title") or slug for slug, rec in items.items()}
    # Mirror saved personas into MODES by their slug key
    for slug, rec in items.items():
        preset = rec.get("preset") or {}
        if isinstance(preset, dict) and preset:
            MODES[slug] = preset
    # Set MODES["custom"] to active persona if present
    active_slug = store.get("active")
    if active_slug and active_slug in items:
        active_preset = items[active_slug].get("preset")
        if isinstance(active_preset, dict) and active_preset:
            MODES["custom"] = active_preset

# Load personas at startup
try:
    _load_personas_into_modes()
except Exception:
    pass

ACCEPT_STATES = {"OH", "PA", "MD", "CO", "CA", "FL", "NC", "SC", "GA", "VA", "UT"}
TARGET_STATES = {"OH", "PA", "MD", "CO", "CA", "FL"}
ACCEPT_CONSTRUCTION = {
    "Joisted Masonry",
    "Non-Combustible",
    "Masonry Non-Combustible",
    "Fire Resistive",
    "Non Combustible",
    "Non Combustible/Steel",
}
CURRENT_YEAR = dt.datetime.now().year


def getAccessToken():
    if FEDERATO_TOKEN:
        return FEDERATO_TOKEN
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience": AUDIENCE,
        "grant_type": "client_credentials",
    }
    r = requests.post(
        AUTH_URL, json=payload, headers={"Content-Type": "application/json"}
    )
    r.raise_for_status()
    return r.json().get("access_token")


def fetchPolicies(access_token):
    h = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    r = requests.post(POLICIES_URL_TYPO, headers=h, json={})
    if r.status_code == 404 or r.status_code == 405:
        r = requests.post(POLICIES_URL, headers=h, json={})
    r.raise_for_status()
    return r.json()


def loadLocalData():
    if not os.path.exists("data.json"):
        return None
    with open("data.json", "r", encoding="utf-8") as f:
        return json.load(f)


def normalizePayload(raw):
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    if isinstance(raw, dict) and "output" in raw:
        o = raw["output"]
        if (
            isinstance(o, list)
            and len(o) > 0
            and isinstance(o[0], dict)
            and "data" in o[0]
        ):
            return o[0]["data"]
    return []


def toDataFrame(items):
    df = pd.DataFrame(items)
    if df.empty:
        return df
    # Ensure expected columns exist to avoid KeyErrors downstream
    expected_num = [
        "tiv",
        "total_premium",
        "winnability",
        "loss_value",
        "oldest_building",
    ]
    expected_str = [
        "account_name",
        "primary_risk_state",
        "line_of_business",
        "renewal_or_new_business",
        "construction_type",
    ]
    expected_id = ["id"]
    for c in expected_num:
        if c not in df.columns:
            df[c] = np.nan
    for c in expected_str:
        if c not in df.columns:
            df[c] = ""
    for c in expected_id:
        if c not in df.columns:
            df[c] = pd.NA
    if "loss_value" in df.columns:
        df["loss_value"] = pd.to_numeric(df["loss_value"], errors="coerce")
    for col in ["total_premium", "winnability", "tiv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["created_at", "effective_date", "expiration_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "oldest_building" in df.columns:
        df["building_age"] = CURRENT_YEAR - pd.to_numeric(
            df["oldest_building"], errors="coerce"
        )
    else:
        df["building_age"] = np.nan
    return df


def get_role():
    """Return a simple role flag for insights tabs.
    Prefers explicit `?role=` query, then cookie, else defaults to 'uw'.
    """
    try:
        role_q = (request.args.get("role") or "").strip().lower()
        if role_q in ("uw", "leader", "admin"):
            return role_q
        cookie_val = (request.cookies.get("role") or "").strip().lower()
        if cookie_val in ("uw", "leader", "admin"):
            return cookie_val
    except Exception:
        pass
    return "uw"


def classifyRow(row):
    """
    Streamlined appetite classification and priority scoring.
    
    Returns (status, reasons, appetite_score, priority_score)
    Range: 0.0 - 10.0, combining appetite (50%), winnability (30%), premium (20%)
    """
    reasons = []
    score = 5.0
    multiplier = 1.0
    
    # === CRITICAL REQUIREMENTS ===
    lob = str(row.get("line_of_business", "")).strip().upper()
    if lob != "COMMERCIAL PROPERTY":
        return "OUT", ["Line of Business not Commercial Property"], 1.0, 0.1
    
    sub_type = str(row.get("renewal_or_new_business", "")).strip().upper()
    if sub_type == "RENEWAL":
        score -= 1.5
        reasons.append("Renewal business (prefer new)")
    elif sub_type == "NEW_BUSINESS":
        score += 1.0
        reasons.append("New business (preferred)")
    
    # === CORE FACTORS ===
    # State scoring
    st = str(row.get("primary_risk_state", "")).strip().upper()
    if st in TARGET_STATES:
        score += 2.0
        multiplier *= 1.2
        reasons.append(f"Target state: {st}")
    elif st in ACCEPT_STATES:
        score += 1.0
        reasons.append(f"Acceptable state: {st}")
    else:
        score -= 2.0
        reasons.append(f"Non-preferred state: {st}")
    
    # TIV scoring
    tiv = row.get("tiv", np.nan)
    if pd.notna(tiv):
        if 50_000_000 <= tiv <= 100_000_000:
            score += 2.0
            multiplier *= 1.15
            reasons.append("TIV in target range")
        elif tiv <= 150_000_000:
            score += 0.5
            reasons.append("TIV acceptable")
        else:
            score -= 2.0
            reasons.append("TIV exceeds limit")
    
    # Premium scoring
    premium = row.get("total_premium", np.nan)
    if pd.notna(premium):
        if 75_000 <= premium <= 100_000:
            score += 2.0
            multiplier *= 1.3
            reasons.append("Premium in target range")
        elif 50_000 <= premium <= 175_000:
            score += 0.5
            reasons.append("Premium acceptable")
        else:
            score -= 1.0
            reasons.append("Premium outside range")
        
        # High value bonus
        if premium >= 1_000_000:
            score += 1.0
            multiplier *= 1.1
            reasons.append("High-value premium")
    
    # Building age scoring
    year = row.get("oldest_building", np.nan)
    if pd.notna(year):
        y = int(float(year)) if not pd.isna(year) else 0
        if y > 2010:
            score += 1.5
            reasons.append("Modern building")
        elif y > 1990:
            score += 0.5
            reasons.append("Acceptable building age")
        else:
            score -= 1.5
            reasons.append("Older building")
    
    # Construction type
    ctype = str(row.get("construction_type", "")).strip()
    if ctype in ACCEPT_CONSTRUCTION:
        score += 0.5
        reasons.append("Preferred construction")
    
    # Loss history
    loss = row.get("loss_value", np.nan)
    if pd.notna(loss):
        if loss < 100_000:
            score += 1.0 if loss < 25_000 else 0.5
            reasons.append("Good loss history")
        else:
            score -= 2.0
            reasons.append("High loss history")
    
    # Winnability
    w = row.get("winnability", 0.5)
    if pd.notna(w):
        w = w / 100.0 if w > 1 else w
        w = max(0.0, min(1.0, float(w)))
        if w >= 0.8:
            score += 1.0
            multiplier *= 1.15
            reasons.append("High win probability")
        elif w >= 0.6:
            score += 0.5
            reasons.append("Good win probability")
    else:
        w = 0.5
    
    # === FINAL CALCULATION ===
    appetite_score = max(1.0, min(10.0, score * multiplier))
    
    # Composite priority score: appetite (50%) + winnability (30%) + premium factor (20%)
    premium_factor = min(premium / 150_000, 1.0) if pd.notna(premium) else 0.5
    priority_score = max(0.1, min(10.0, 
        appetite_score * 0.5 + 
        w * 10.0 * 0.3 + 
        premium_factor * 10.0 * 0.2
    ))
    
    # === STATUS DETERMINATION ===
    severe_issues = sum(1 for r in reasons if any(word in r.lower() 
                       for word in ['exceeds', 'outside', 'non-preferred', 'high loss']))
    
    if severe_issues >= 2 or appetite_score < 3.0:
        status = "OUT"
    elif st in TARGET_STATES and appetite_score >= 7.0 and priority_score >= 7.0:
        status = "TARGET"
    else:
        status = "IN"
    
    return status, reasons, appetite_score, priority_score



def classifyDataFrame(df):
    if df.empty:
        return df
    statuses = []
    reasons = []
    appetites = []
    priorities = []
    for _, r in df.iterrows():
        s, rs, a, p = classifyRow(r)
        statuses.append(s)
        reasons.append(rs)
        appetites.append(a)
        priorities.append(p)
    df = df.copy()
    df["appetite_status"] = statuses
    df["appetite_reasons"] = reasons
    df["appetite_score"] = appetites
    df["priority_score"] = priorities
    # Mark target opportunities globally on the full dataset (top 30% of in-appetite by priority)
    try:
        df = mark_target_opportunities_df(df)
    except Exception:
        # Non-fatal if marking fails; continue without the flag
        pass
    return df


def mark_target_opportunities_df(frame):
    """Return a copy of frame with boolean column 'target_opportunity' set True
    for the top ceil(40%) rows among in-appetite (TARGET/IN) by priority_score.

    This operates on the provided frame only; callers should pass a filtered
    subset when computing list-specific target opportunities.
    """
    if frame is None or len(frame) == 0:
        return frame
    d = frame.copy()
    # Initialize column
    d["target_opportunity"] = False
    try:
        in_mask = d["appetite_status"].isin(["TARGET", "IN"]).fillna(False)
        in_df = d[in_mask]
        n = len(in_df)
        if n > 0:
            top_n = int(np.ceil(0.40 * n))  # Increased from 30% to 40%
            # Sort by priority_score descending; NaNs go last
            in_sorted = in_df.sort_values(["priority_score"], ascending=[False])
            ids = in_sorted.head(max(1, top_n))["id"].tolist()
            if ids:
                d.loc[d["id"].isin(ids), "target_opportunity"] = True
    except Exception:
        # If anything goes wrong, leave column as False
        pass
    return d


def refreshCache():
    now = time.time()
    if now - _cache_data["ts"] < CACHE_TTL and _cache_data["df"] is not None:
        return _cache_data["df"], _cache_data["raw"]
    raw = None
    if USE_LOCAL_DATA:
        raw = loadLocalData()
    else:
        token = getAccessToken()
        if token:
            try:
                raw = fetchPolicies(token)
            except Exception:
                raw = loadLocalData()
        else:
            raw = loadLocalData()
    items = normalizePayload(raw or {})
    df = toDataFrame(items)
    df = classifyDataFrame(df)
    _cache_data["ts"] = now
    _cache_data["df"] = df
    _cache_data["raw"] = raw
    return df, raw


def _safe_json_value(v):
    # Handle missing values first where applicable
    try:
        if pd.isna(v):
            return None
    except Exception:
        # pd.isna may return an array for list-like; ignore here
        pass

    # Datetime-like
    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    # Numpy scalars
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        try:
            if np.isnan(v):
                return None
        except Exception:
            pass
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # List-like / ndarray / set / tuple
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return list(v)

    return v


def dfToDict(df):
    cols = [c for c in df.columns]
    records = []
    for _, r in df.iterrows():
        x = {c: _safe_json_value(r[c]) for c in cols}
        records.append(x)
    return records


@app.route("/")
def home():
    df, _ = refreshCache()
    total = len(df)
    # In-appetite includes both IN and TARGET statuses
    in_ct = int(df["appetite_status"].isin(["IN", "TARGET"]).sum()) if not df.empty else 0
    # Target opportunities are the top 30% of in-appetite by priority_score
    if not df.empty and "target_opportunity" in df.columns:
        tgt_ct = int((df["target_opportunity"] == True).sum())
    else:
        try:
            _tmp = mark_target_opportunities_df(df)
            tgt_ct = int((_tmp["target_opportunity"] == True).sum())
        except Exception:
            tgt_ct = 0
    out_ct = int((df["appetite_status"] == "OUT").sum()) if not df.empty else 0
    avg_premium = float(df["total_premium"].mean()) if not df.empty else 0.0
    avg_premium_str = f"${avg_premium:,.0f}"
    # Percent breakdowns from current dataset
    def _pct(part, whole):
        try:
            return int(round((part / whole) * 100.0)) if whole else 0
        except Exception:
            return 0
    pct_tgt = _pct(tgt_ct, total)
    pct_in = _pct(in_ct, total)
    pct_out = _pct(out_ct, total)

    # 7-day vs prior 7-day deltas based on created_at
    trend_target = trend_in = trend_out = trend_total = 0
    if not df.empty and "created_at" in df.columns:
        now = pd.Timestamp.utcnow()
        w1_start = now - pd.Timedelta(days=7)
        w2_start = now - pd.Timedelta(days=14)
        recent = df[df["created_at"] >= w1_start]
        prior = df[(df["created_at"] >= w2_start) & (df["created_at"] < w1_start)]

        def _delta(cur, prev):
            try:
                if prev == 0:
                    return 0
                return int(round(((cur - prev) / prev) * 100.0))
            except Exception:
                return 0

        trend_total = _delta(len(recent), len(prior))
        trend_target = _delta(int((recent["appetite_status"] == "TARGET").sum()), int((prior["appetite_status"] == "TARGET").sum()))
        trend_in = _delta(int((recent["appetite_status"] == "IN").sum()), int((prior["appetite_status"] == "IN").sum()))
        trend_out = _delta(int((recent["appetite_status"] == "OUT").sum()), int((prior["appetite_status"] == "OUT").sum()))

    # Winnability average (normalized 0..1)
    avg_win = 0.0
    if not df.empty and "winnability" in df.columns:
        s = pd.to_numeric(df["winnability"], errors="coerce")
        try:
            s = s.apply(lambda x: np.nan if pd.isna(x) else (x/100.0 if x>1 else float(x)))
            avg_win = float(s.mean(skipna=True)) if len(s) else 0.0
        except Exception:
            avg_win = 0.0

    # Opportunities = all in-appetite (includes targets)
    opp_ct = int(in_ct)
    return render_template(
        "index.html",
        total=total,
        in_ct=in_ct,
        tgt_ct=tgt_ct,
        out_ct=out_ct,
        avg_premium=avg_premium,
        avg_premium_str=avg_premium_str,
        pct_tgt=pct_tgt,
        pct_in=pct_in,
        pct_out=pct_out,
        trend_target=trend_target,
        trend_in=trend_in,
        trend_out=trend_out,
        trend_total=trend_total,
        avg_win=avg_win,
        opp_ct=opp_ct,
        full_width=True,
    )


@app.route("/submissions")
def submissions():
    df, _ = refreshCache()
    state = request.args.get("state")
    status = request.args.get("status")
    q = request.args.get("q")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    d = df
    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if status:
        d = d[d["appetite_status"] == status]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]  # Handle NaN values explicitly
    # Recompute target opportunities within the filtered set to keep 30% rule list-specific
    try:
        d = mark_target_opportunities_df(d)
    except Exception:
        pass

    # Sort: target_opportunity first, then TARGET -> IN -> OUT, then by priority score desc, premium desc
    try:
        d = d.copy()
        d["__top"] = d["target_opportunity"].astype(int)
        d["__s"] = pd.Categorical(
            d["appetite_status"], categories=["TARGET", "IN", "OUT"], ordered=True
        )
        d = d.sort_values(["__top", "__s", "priority_score", "total_premium"], ascending=[False, True, False, False])
        d = d.drop(columns=["__top", "__s"])
    except Exception:
        d = d.sort_values(["priority_score"], ascending=False)
    rows = dfToDict(d)
    return render_template("submissions.html", rows=rows)


@app.route("/insights")
def insights():
    # Portfolio Insights landing with simple role-based access for tabs
    role = get_role()
    tabs = [
        {"id": "overview", "title": "Triage", "protected": False},
        {"id": "premium_by_state", "title": "Pricing", "protected": False},
        {"id": "status_mix", "title": "Exposure", "protected": False},
        {"id": "tiv_bands", "title": "Renewals", "protected": False},
        {"id": "mix", "title": "Mix", "protected": False},
        {"id": "risk_signals", "title": "Risk Signals", "protected": False},
        {"id": "geo", "title": "US Map", "protected": False},
        {"id": "scatter3d", "title": "3D Scatter", "protected": False},
        {"id": "flows", "title": "Flows (Sankey)", "protected": False},
    ]
    resp = make_response(render_template("insights.html", role=role, tabs=tabs))
    # Persist role if provided via query
    role_q = request.args.get("role")
    if role_q:
        resp.set_cookie("role", role, max_age=7*24*3600, httponly=False, samesite="Lax")
    return resp


@app.route("/detail/<int:pid>")
def detail(pid):
    mode = request.args.get("mode")
    if mode and mode in MODES:
        try:
            ddf = load_df("data.json")
            dd = ddf[ddf["id"] == pid]
            if dd.empty:
                return render_template("detail.html", item=None)
            scounts = Counter(ddf["primary_risk_state"].fillna("UNK"))
            preset = MODES[mode]
            r = dd.iloc[0].to_dict()
            status, reasons, s, pscore = classify_for_mode_row(
                r, preset["weights"], preset["filters"], scounts
            )
            r["appetite_status"] = status
            r["appetite_reasons"] = reasons
            r["priority_score"] = float(pscore)
            return render_template("detail.html", item=r)
        except Exception:
            pass
    df, _ = refreshCache()
    d = df[df["id"] == pid]
    if d.empty:
        return render_template("detail.html", item=None)
    item = dfToDict(d)[0]
    return render_template("detail.html", item=item)


@app.route("/api/policies")
def apiPolicies():
    df, raw = refreshCache()
    return jsonify({"count": len(df), "data": dfToDict(df)})


@app.route("/api/classified")
def apiClassified():
    df, _ = refreshCache()
    state = request.args.get("state")
    status = request.args.get("status")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")
    d = df
    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if status:
        d = d[d["appetite_status"] == status]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]
    # Mark targets within the filtered set then sort with them first
    try:
        d = mark_target_opportunities_df(d)
        d = d.copy()
        d["__top"] = d["target_opportunity"].astype(int)
        d = d.sort_values(["__top", "priority_score"], ascending=[False, False]).drop(columns=["__top"])
    except Exception:
        d = d.sort_values(["priority_score"], ascending=False)
    return jsonify({"count": len(d), "data": dfToDict(d)})


@app.route("/api/priority-accounts")
def apiPriorityAccounts():
    """API endpoint for Priority Account Review - returns top priority submissions
    Uses the same mode-aware classification as the main dashboard for consistency"""
    
    # Get the current active mode to ensure consistency with dashboard
    from flask import g
    
    # Use same logic as main dashboard
    mode = request.args.get("mode") or "balanced_growth"
    df = load_df("data.json")
    if df.empty:
        return jsonify({"data": [], "pagination": {"page": 1, "per_page": 20, "total_count": 0, "total_pages": 0, "has_next": False, "has_prev": False}})

    # Map legacy/new names → percentile kind (same as main dashboard)
    name_map = {
        "unicorn_hunting": "unicorn",
        "balanced_growth": "balanced", 
        "loose_fits": "loose",
        "turnaround_bets": "turnaround",
        "unicorn": "unicorn",
        "balanced": "balanced",
        "loose": "loose", 
        "turnaround": "turnaround",
    }

    # Use the same mode classification logic as the main dashboard
    if mode in MODES:
        preset = MODES[mode]
        filters = (preset or {}).get("filters", {})
        weights = (preset or {}).get("weights", MODES.get("balanced_growth", {}).get("weights", {}))
        d = apply_hard_filters(df, filters)
        mode_explanation = f"Preset mode — {mode.replace('_',' ').title()}"
    else:
        # Fall back to percentile-driven dynamic filters
        kind = name_map.get(mode, "balanced")
        overrides = {}
        for k in ("top_pct", "max_lr_pct", "min_win_pct", "max_tiv_pct", "fresh_pct"):
            if k in request.args:
                overrides[k] = request.args.get(k)
        filters, _summary = build_percentile_filters(df, kind=kind, overrides=overrides)
        d = apply_hard_filters(df, filters)
        weights = MODES.get("balanced_growth", {}).get("weights", {})
        mode_explanation = f"{kind.title()} mode — dynamic percentile-based filters"

    # Classify all rows using the same logic as main dashboard
    scounts = Counter(d["primary_risk_state"].fillna("UNK"))
    classified_rows = []
    
    for r in d.to_dict(orient="records"):
        status, reasons, s, pscore = classify_for_mode_row(r, weights, filters, scounts)
        r["appetite_status"] = status
        r["appetite_reasons"] = reasons if isinstance(reasons, list) else []
        r["mode_score"] = s
        r["priority_score"] = pscore
        classified_rows.append(r)
    
    # Convert back to DataFrame for filtering
    d = pd.DataFrame(classified_rows)
    
    # Get query parameters for additional filtering
    state = request.args.get("state")
    status = request.args.get("status")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=20, type=int)
    sort_by = request.args.get("sort_by", default="priority_score")
    sort_dir = request.args.get("sort_dir", default="desc")
    
    # Apply user filters on top of mode classification
    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if status:
        d = d[d["appetite_status"] == status]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]
    
    # Focus on in-appetite submissions (TARGET and IN) for priority review
    d = d[d["appetite_status"].isin(["TARGET", "IN"])]

    # Apply sorting
    ascending = sort_dir.lower() == "asc"
    # Mark target opportunities and sort them first
    try:
        d = mark_target_opportunities_df(d)
        d = d.copy()
        d["__top"] = d["target_opportunity"].astype(int)
        if sort_by in d.columns:
            d = d.sort_values(["__top", sort_by], ascending=[False, ascending])
        else:
            d = d.sort_values(["__top", "priority_score"], ascending=[False, False])
        d = d.drop(columns=["__top"])
    except Exception as e:
        print(f"Error in target opportunity sorting: {e}")
        if sort_by in d.columns:
            d = d.sort_values([sort_by], ascending=ascending)
        else:
            d = d.sort_values(["priority_score"], ascending=False)
    
    # Calculate pagination
    total_count = len(d)
    total_pages = (total_count + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get paginated data
    paginated_data = d.iloc[start_idx:end_idx]
    
    # Convert to response format
    response_data = []
    for _, row in paginated_data.iterrows():
        response_data.append({
            "id": int(row["id"]),
            "account_name": str(row["account_name"]),
            "appetite_status": str(row["appetite_status"]),
            "total_premium": float(row["total_premium"]),
            "primary_risk_state": str(row["primary_risk_state"]),
            "winnability": float(row["winnability"]) if pd.notna(row["winnability"]) else 0.0,
            "priority_score": float(row["priority_score"]) if pd.notna(row["priority_score"]) else 0.0,
            "tiv": float(row["tiv"]) if pd.notna(row["tiv"]) else 0.0,
            "line_of_business": str(row["line_of_business"]) if pd.notna(row["line_of_business"]) else "",
            "target_opportunity": bool(row.get("target_opportunity", False)),
            "appetite_reasons": row.get("appetite_reasons", []) if isinstance(row.get("appetite_reasons"), list) else [],
            "created_at": str(row["created_at"]) if pd.notna(row["created_at"]) else "",
            "effective_date": str(row["effective_date"]) if pd.notna(row["effective_date"]) else "",
        })
    
    return jsonify({
        "data": response_data,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        },
        "filters": {
            "state": state,
            "status": status,
            "min_premium": min_p,
            "max_premium": max_p,
            "search": q,
            "sort_by": sort_by,
            "sort_dir": sort_dir
        }
    })


@app.route("/api/metrics_mode")
def api_metrics_mode():
    """Return KPI metrics for the given analysis mode without trimming to top_pct.

    Provides counts and percents for TARGET/IN/OUT based on mode-driven classification,
    plus totals, avg premium, avg winnability and opportunities.
    """
    mode = (request.args.get("mode") or "balanced_growth").strip()
    df = load_df("data.json")
    if df.empty:
        return jsonify({"ok": True, "metrics": {
            "total": 0, "targets": 0, "inCt": 0, "outCt": 0,
            "pctTgt": 0, "pctIn": 0, "pctOut": 0,
            "avgPremium": 0.0, "winRate": 0.0, "opportunities": 0,
        }})

    # Determine filters/weights for the requested mode
    name_map = {
        "unicorn_hunting": "unicorn",
        "balanced_growth": "balanced",
        "loose_fits": "loose",
        "turnaround_bets": "turnaround",
        "unicorn": "unicorn",
        "balanced": "balanced",
        "loose": "loose",
        "turnaround": "turnaround",
    }

    if mode == "custom" and "custom" in MODES:
        preset = MODES["custom"]
        filters = preset.get("filters", {})
        weights = preset.get("weights", MODES.get("balanced_growth", {}).get("weights", {}))
        d = apply_hard_filters(df, filters)
    else:
        kind = name_map.get(mode, "balanced")
        filters, _summary = build_percentile_filters(df, kind=kind, overrides=None)
        d = apply_hard_filters(df, filters)
        weights = MODES.get("balanced_growth", {}).get("weights", {})

    if d.empty:
        return jsonify({"ok": True, "metrics": {
            "total": 0, "targets": 0, "inCt": 0, "outCt": 0,
            "pctTgt": 0, "pctIn": 0, "pctOut": 0,
            "avgPremium": 0.0, "winRate": 0.0, "opportunities": 0,
        }})

    # Classify all rows (no top_pct cut) to get status counts
    scounts = Counter(d["primary_risk_state"].fillna("UNK"))
    t = i = o = 0
    wins = []
    prems = []
    for r in d.to_dict(orient="records"):
        status, reasons, s, pscore = classify_for_mode_row(r, weights, filters, scounts)
        if status == "TARGET":
            t += 1
        elif status == "IN":
            i += 1
        else:
            o += 1
        try:
            v = r.get("winnability")
            if v is not None and v == v:
                wins.append(float(v))
        except Exception:
            pass
        try:
            v = r.get("total_premium")
            if v is not None and v == v:
                prems.append(float(v))
        except Exception:
            pass

    total = int(t + i + o)
    def _pct(n, d):
        try:
            return int(round(((n / d) * 100.0))) if d else 0
        except Exception:
            return 0

    metrics = {
        "total": total,
        "targets": int(t),
        "inCt": int(i),
        "outCt": int(o),
        "pctTgt": _pct(t, total),
        "pctIn": _pct(i, total),
        "pctOut": _pct(o, total),
        "avgPremium": float(np.mean(prems)) if prems else 0.0,
        "winRate": float(np.mean(wins) * 100.0) if wins else 0.0,
        "opportunities": int(t + i),
    }
    return jsonify({"ok": True, "metrics": metrics})


@app.route("/api/refresh", methods=["POST"]) 
def apiRefresh():
    # Reset cache and refetch so UI can manually refresh
    _cache_data["ts"] = 0
    df, _ = refreshCache()
    return jsonify({"ok": True, "count": len(df)})


def buildGuidelineSummary():
    t = []
    t.append("Line of Business must be Commercial Property.")
    t.append("Submission type must be New Business.")
    t.append(
        "Acceptable states: OH, PA, MD, CO, CA, FL, NC, SC, GA, VA, UT; targets include OH, PA, MD, CO, CA, FL."
    )
    t.append("TIV acceptable up to 150M; target 50M–100M; over 150M not acceptable.")
    t.append("Premium acceptable 50k–175k; target 75k–100k; outside not acceptable.")
    t.append(
        "Building age newer than 1990 acceptable; newer than 2010 target; older than 1990 not acceptable."
    )
    t.append(
        "Construction acceptable when >50% JM, Non-Combustible/Steel, Masonry Non-Combustible, Fire Resistive."
    )
    t.append("Loss value must be < 100k.")
    return " ".join(t)


def buildModeGuidelineSummary(mode_name: str):
    p = MODES.get(mode_name)
    if not p:
        return "Mode not found; using default heuristic filters and weights."
    f = p.get("filters", {})
    parts = [f"Mode {mode_name} — Dynamic guideline summary:"]
    if f.get("lob_in"):
        parts.append(f"Allowed LOB: {', '.join(f['lob_in'])}.")
    if f.get("new_business_only"):
        parts.append("Only NEW_BUSINESS submissions.")
    if "loss_ratio_max" in f:
        parts.append(f"Loss ratio ≤ {f['loss_ratio_max']:.2f}.")
    if "tiv_max" in f:
        parts.append(f"TIV ≤ ${f['tiv_max']:,}.")
    if "premium_range" in f:
        lo, hi = f["premium_range"]
        parts.append(f"Premium in ${lo:,}–${hi:,} targetable window.")
    if "min_winnability" in f:
        parts.append(f"Winnability ≥ {f['min_winnability']:.2f}.")
    if "min_year" in f:
        parts.append(f"Building year ≥ {int(f['min_year'])} (or missing).")
    if f.get("good_construction_only"):
        parts.append("Acceptable construction types only.")
    parts.append(
        "Weighted by premium, winnability, year, construction, TIV, freshness; premium sweet spot defined by mid band."
    )
    return " ".join(parts)


def makePolicyDict(pid):
    df, _ = refreshCache()
    d = df[df["id"] == pid]
    if d.empty:
        return None
    return dfToDict(d)[0]


@app.route("/api/explain/<int:pid>", methods=["POST"])
def apiExplain(pid):
    mode = request.args.get("mode")
    if mode and mode in MODES:
        ddf = load_df("data.json")
        d = ddf[ddf["id"] == pid]
        if d.empty:
            return jsonify({"ok": False, "error": "Not found"}), 404
        preset = MODES[mode]
        scounts = Counter(ddf["primary_risk_state"].fillna("UNK"))
        r = d.iloc[0].to_dict()
        status, reasons, s, pscore = classify_for_mode_row(
            r, preset["weights"], preset["filters"], scounts
        )
        r["appetite_status"] = status
        r["appetite_reasons"] = reasons
        r["priority_score"] = float(pscore)
        item = r
        rule_text = buildModeGuidelineSummary(mode)
    else:
        item = makePolicyDict(pid)
        if not item:
            return jsonify({"ok": False, "error": "Not found"}), 404
        rule_text = buildGuidelineSummary()
    raw_reason = item.get("appetite_reasons", [])
    label = item.get("appetite_status", "IN")
    
    # Use Gemini for explanation
    result = explain_with_gemini(item, rule_text, label, raw_reason)
    
    response = {
        "ok": True, 
        "explanation": result["explanation"],
        "ai_used": result["ai_used"]
    }
    
    if result["fallback_reason"]:
        response["ai_note"] = result["fallback_reason"]
    
    return jsonify(response)


@app.route("/api/nlq", methods=["POST"])
def apiNlq():
    data = request.get_json(silent=True) or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"filters": {}, "ai_used": False, "ai_summary": "Empty query provided"})
    
    # Use Gemini for NLQ parsing
    result = nlq_with_gemini(q)
    
    response = {
        "filters": result["filters"],
        "ai_used": result["ai_used"],
        "ai_summary": result["ai_summary"]
    }
    
    if result["fallback_reason"]:
        response["ai_note"] = result["fallback_reason"]
    
    return jsonify(response)


@app.route("/api/modes")
def api_modes():
    # Return an array of objects so the UI can render key/label pairs
    arr = []
    for k in MODES.keys():
        if k in _PERSONA_INDEX:
            label = f"Persona: {_PERSONA_INDEX[k]}"
        elif k == "custom":
            # If custom is active persona, reflect it
            slug, rec = persona_get_active()
            if slug and rec:
                label = f"Persona: {rec.get('title') or slug} (Active)"
            else:
                label = k.replace("_", " ").title()
        else:
            label = k.replace("_", " ").title()
        arr.append({"key": k, "label": label})
    return jsonify(arr)


@app.route("/api/mode/percentiles")
def api_mode_percentiles():
    try:
        df = load_df("data.json")
        if df.empty:
            return jsonify({"ok": True, "summary": {}})
        s = summarize_percentiles(df)
        return jsonify({"ok": True, "summary": to_builtin(s)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/mode/blend", methods=["POST"])
def api_mode_blend():
    try:
        data = request.get_json(silent=True) or {}
        left = str(data.get("left") or "balanced_growth")
        right = str(data.get("right") or "unicorn_hunting")
        t = float(data.get("t") or 0.5)
        lm = MODES.get(left)
        rm = MODES.get(right)
        if not lm or not rm:
            return jsonify({"ok": False, "error": "Unknown mode(s)"}), 400
        blended = blend_modes(lm, rm, t)
        MODES["blend"] = blended
        return jsonify({"ok": True, "mode": "blend", "blended": to_builtin(blended)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/ai-sort", methods=["POST"])
def api_ai_sort():
    """AI-powered sorting based on selected categories"""
    try:
        data = request.get_json(silent=True) or {}
        categories = data.get("categories", [])
        current_filters = data.get("current_filters", {})
        sort_preferences = data.get("sort_preferences", {})
        
        if not categories:
            return jsonify({
                "ok": False, 
                "error": "No categories provided for AI sorting"
            }), 400
        
        # Load and filter data based on current filters
        df, _ = refreshCache()
        if df.empty:
            return jsonify({
                "ok": True,
                "data": [],
                "ai_used": False,
                "ai_summary": "No data available for sorting"
            })
        
        # Apply current filters
        filtered_df = df.copy()
        
        # Apply basic filters
        if current_filters.get("state"):
            filtered_df = filtered_df[
                filtered_df["primary_risk_state"].astype(str).str.upper() == 
                current_filters["state"].upper()
            ]
        
        if current_filters.get("status"):
            filtered_df = filtered_df[
                filtered_df["appetite_status"] == current_filters["status"]
            ]
        
        if current_filters.get("min_premium"):
            min_p = float(current_filters["min_premium"])
            filtered_df = filtered_df[filtered_df["total_premium"] >= min_p]
        
        if current_filters.get("max_premium"):
            max_p = float(current_filters["max_premium"])
            filtered_df = filtered_df[filtered_df["total_premium"] <= max_p]
        
        if current_filters.get("q"):
            query = current_filters["q"]
            mask = filtered_df["account_name"].astype(str).str.contains(
                query, case=False, na=False
            )
            filtered_df = filtered_df[mask.fillna(False)]
        
        # Apply category-specific filters and create AI-powered sorting strategy
        category_weights = {}
        category_filters = {}
        ai_strategy_parts = []
        
        for category in categories:
            if category == "targets":
                category_filters["targets"] = filtered_df["appetite_status"] == "TARGET"
                category_weights["priority_score"] = category_weights.get("priority_score", 0) + 0.4
                category_weights["appetite_alignment"] = category_weights.get("appetite_alignment", 0) + 0.3
                ai_strategy_parts.append("prioritizing TARGET status submissions")
                
            elif category == "in-appetite":
                category_filters["in_appetite"] = filtered_df["appetite_status"] == "IN"
                category_weights["appetite_alignment"] = category_weights.get("appetite_alignment", 0) + 0.25
                ai_strategy_parts.append("including IN-APPETITE submissions")
                
            elif category == "high-value":
                high_value_threshold = filtered_df["total_premium"].quantile(0.75)
                category_filters["high_value"] = filtered_df["total_premium"] >= high_value_threshold
                category_weights["premium_size"] = category_weights.get("premium_size", 0) + 0.35
                ai_strategy_parts.append(f"emphasizing high-value premiums (${high_value_threshold:,.0f}+)")
                
            elif category == "new-business":
                category_filters["new_business"] = (
                    filtered_df["renewal_or_new_business"].astype(str).str.upper() == "NEW_BUSINESS"
                )
                category_weights["business_type"] = category_weights.get("business_type", 0) + 0.2
                ai_strategy_parts.append("favoring new business opportunities")
                
            elif category == "recent":
                if "created_at" in filtered_df.columns:
                    recent_threshold = pd.Timestamp.utcnow() - pd.Timedelta(days=7)
                    category_filters["recent"] = filtered_df["created_at"] >= recent_threshold
                    category_weights["freshness"] = category_weights.get("freshness", 0) + 0.15
                    ai_strategy_parts.append("prioritizing recent submissions (last 7 days)")
                
            elif category == "geographic":
                target_states = {"OH", "PA", "MD", "CO", "CA", "FL"}
                category_filters["geographic"] = filtered_df["primary_risk_state"].isin(target_states)
                category_weights["geographic_preference"] = category_weights.get("geographic_preference", 0) + 0.2
                ai_strategy_parts.append("focusing on target geographic regions")
        
        # Create AI-powered composite score
        ai_scores = []
        explanations = []
        
        for _, row in filtered_df.iterrows():
            composite_score = row.get("priority_score", 0.0)
            score_components = []
            
            # Apply category-specific scoring
            for category, condition in category_filters.items():
                if condition.loc[row.name] if hasattr(condition, 'loc') else condition:
                    if category == "targets":
                        composite_score += 3.0
                        score_components.append("TARGET status (+3.0)")
                    elif category == "in_appetite":
                        composite_score += 1.5
                        score_components.append("IN-APPETITE (+1.5)")
                    elif category == "high_value":
                        premium_bonus = min(2.0, (row.get("total_premium", 0) / 100000) * 0.5)
                        composite_score += premium_bonus
                        score_components.append(f"High premium (+{premium_bonus:.1f})")
                    elif category == "new_business":
                        composite_score += 1.0
                        score_components.append("New business (+1.0)")
                    elif category == "recent":
                        composite_score += 0.8
                        score_components.append("Recent submission (+0.8)")
                    elif category == "geographic":
                        composite_score += 1.2
                        score_components.append("Target geography (+1.2)")
            
            # Winnability boost
            winnability = row.get("winnability", 0.5)
            if isinstance(winnability, (int, float)) and winnability > 1:
                winnability = winnability / 100.0
            winnability_bonus = winnability * 2.0
            composite_score += winnability_bonus
            score_components.append(f"Winnability (+{winnability_bonus:.1f})")
            
            ai_scores.append(composite_score)
            explanations.append("; ".join(score_components))
        
        # Add AI scores to dataframe
        filtered_df = filtered_df.copy()
        filtered_df["ai_composite_score"] = ai_scores
        filtered_df["ai_score_explanation"] = explanations
        
        # Sort by AI composite score
        filtered_df = filtered_df.sort_values("ai_composite_score", ascending=False)
        
        # Convert to response format
        response_data = dfToDict(filtered_df)
        
        # Generate AI summary
        ai_summary = f"Applied intelligent sorting strategy: {', '.join(ai_strategy_parts)}. " \
                    f"Analyzed {len(filtered_df)} submissions with composite scoring algorithm."
        
        return jsonify({
            "ok": True,
            "data": response_data,
            "ai_used": True,
            "ai_summary": ai_summary,
            "categories_applied": categories,
            "total_submissions": len(response_data),
            "strategy_components": ai_strategy_parts
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "ai_used": False,
            "ai_summary": f"AI sorting failed: {str(e)}"
        }), 500


def apply_hard_filters(df, f):
    d = df.copy()
    if "lob_in" in f:
        d = d[d["line_of_business"].str.upper().isin([x.upper() for x in f["lob_in"]])]
    if f.get("new_business_only"):
        d = d[d["renewal_or_new_business"].str.upper()=="NEW_BUSINESS"]
    if "loss_ratio_max" in f:
        d = d[(d["loss_ratio"].notna()) & (d["loss_ratio"]<=f["loss_ratio_max"])]
    if "tiv_max" in f:
        d = d[d["tiv"]<=f["tiv_max"]]
    if "premium_range" in f:
        lo, hi = f["premium_range"]
        d = d[(d["total_premium"]>=lo) & (d["total_premium"]<=hi)]
    if "min_winnability" in f:
        d = d[d["winnability"]>=f["min_winnability"]]
    if "max_fresh_days" in f and f.get("max_fresh_days") is not None:
        # keep only fresh submissions where we have the metric
        d = d[(d["fresh_days"].notna()) & (d["fresh_days"] <= f["max_fresh_days"])]
    if "min_year" in f:
        d = d[(d["building_year"]>=f["min_year"]) | (d["building_year"].isna())]
    if f.get("good_construction_only"):
        d = d[d["is_good_construction"]==True]
    if "fresh_days_max" in f:
        d = d[d["fresh_days"]<=f["fresh_days_max"]]
    return d


def sanitize_row(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            out[k] = float(v)
        elif isinstance(v, (pd.Timestamp, dt.datetime)):
            out[k] = None if pd.isna(v) else v.isoformat()
        elif v is None:
            out[k] = None
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def generate_explanation(r, qs):
    parts = []
    prem = r.get("total_premium")
    if prem is not None and not (isinstance(prem, float) and np.isnan(prem)):
        if prem >= qs["premium"].loc[0.9]:
            parts.append(f"premium ${prem:,.0f} is in top 10%")
        elif prem >= qs["premium"].loc[0.5]:
            parts.append(f"premium ${prem:,.0f} above median")
    tiv = r.get("tiv")
    if tiv is not None and not (isinstance(tiv, float) and np.isnan(tiv)):
        if tiv <= 150_000_000:
            parts.append(f"TIV ${tiv:,.0f} ≤ cap")
    win = r.get("winnability")
    if win is not None and not (isinstance(win, float) and np.isnan(win)):
        med_win = qs["winnability"].loc[0.5]
        if win >= med_win:
            parts.append(f"winnability {win:.2f} ≥ median")
    if not parts:
        return "No clear reasons available."
    prefix = (
        "In Appetite because "
        if r.get("appetite_status") != "OUT"
        else "Out of Appetite because "
    )
    return prefix + ", ".join(parts) + "."


@app.route("/api/classified_mode")
def api_classified_mode():
    """Percentile-driven modes and persona custom.

    Query params:
      - mode: one of legacy [unicorn_hunting, balanced_growth, loose_fits, turnaround_bets] or new [unicorn, balanced, loose, turnaround, custom]
      - overrides: top_pct, max_lr_pct, min_win_pct, max_tiv_pct, fresh_pct
      - filters: state, status, min_premium, max_premium, q
      - sorting: sort_by, sort_dir
    """
    mode = (request.args.get("mode") or "balanced_growth").strip()
    df = load_df("data.json")
    if df.empty:
        return jsonify({"count": 0, "data": []})

    # Map legacy/new names → percentile kind
    name_map = {
        "unicorn_hunting": "unicorn",
        "balanced_growth": "balanced",
        "loose_fits": "loose",
        "turnaround_bets": "turnaround",
        "unicorn": "unicorn",
        "balanced": "balanced",
        "loose": "loose",
        "turnaround": "turnaround",
    }

    # Check if all user filters are empty (reset state)
    user_filters_active = any([
        request.args.get("state"),
        request.args.get("status"), 
        request.args.get("min_premium"),
        request.args.get("max_premium"),
        request.args.get("q")
    ])

    # 1) If mode refers to a concrete preset in MODES (including saved personas), use it
    if mode in MODES:
        preset = MODES[mode]
        filters = (preset or {}).get("filters", {})
        weights = (preset or {}).get("weights", MODES.get("balanced_growth", {}).get("weights", {}))
        
        # If no user filters are active, show ALL data but use mode weights for scoring
        if not user_filters_active:
            d = df  # Don't apply hard filters - show all data
        else:
            d = apply_hard_filters(df, filters)
            
        if mode in _PERSONA_INDEX or mode == "custom":
            # Persona-selected
            title = _PERSONA_INDEX.get(mode)
            if mode == "custom":
                _slug, _rec = persona_get_active()
                if _rec:
                    title = _rec.get("title") or title
            mode_explanation = f"Custom persona — {title or mode}"
        else:
            mode_explanation = f"Preset mode — {mode.replace('_',' ').title()}"
    else:
        # 2) Otherwise, fall back to percentile-driven dynamic filters by kind
        kind = name_map.get(mode, "balanced")
        overrides = {}
        for k in ("top_pct", "max_lr_pct", "min_win_pct", "max_tiv_pct", "fresh_pct"):
            if k in request.args:
                overrides[k] = request.args.get(k)
        filters, _summary = build_percentile_filters(df, kind=kind, overrides=overrides)
        
        # If no user filters are active, show ALL data but use mode weights for scoring
        if not user_filters_active:
            d = df  # Don't apply hard filters - show all data
        else:
            d = apply_hard_filters(df, filters)
            
        weights = MODES.get("balanced_growth", {}).get("weights", {})
        mode_explanation = f"{kind.title()} mode — dynamic percentile-based filters"

    # Optional UI filters
    state = request.args.get("state")
    status_filter = request.args.get("status")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")

    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]

    # Score and rank
    scounts = Counter(d["primary_risk_state"].fillna("UNK"))
    rows = []
    for r in d.to_dict(orient="records"):
        stat, reasons, s, pscore = classify_for_mode_row(
            r, weights, filters, scounts
        )
        r["appetite_status"] = stat
        r["appetite_reasons"] = reasons
        r["priority_score"] = float(pscore)
        r["mode_score"] = round(float(pscore), 4)
        rows.append(sanitize_row(r))

    # Mark target opportunities in this list (top 30% by priority among in-appetite)
    try:
        in_rows = [r for r in rows if str(r.get("appetite_status")) in ("TARGET", "IN")]
        k = len(in_rows)
        if k > 0:
            top_n = max(1, int(np.ceil(0.30 * k)))
            in_rows_sorted = sorted(in_rows, key=lambda r: (r.get("priority_score") or 0.0), reverse=True)
            top_set = {r.get("id") for r in in_rows_sorted[:top_n]}
            for r in rows:
                r["target_opportunity"] = bool(r.get("id") in top_set)
        else:
            for r in rows:
                r["target_opportunity"] = False
    except Exception:
        for r in rows:
            r["target_opportunity"] = False

    # Cut to top N% if provided in filters
    top_pct = None
    try:
        if isinstance(filters.get("top_pct"), (int, float)):
            top_pct = float(filters.get("top_pct"))
    except Exception:
        top_pct = None
    if top_pct:
        n = max(1, int(len(rows) * (top_pct / 100.0)))
        rows.sort(key=lambda r: r.get("priority_score") or 0.0, reverse=True)
        rows = rows[:n]

    # Filter by status if requested
    if status_filter:
        rows = [r for r in rows if str(r.get("appetite_status")) == status_filter]

    # Sorting (always put target_opportunity first)
    sort_by = request.args.get("sort_by")
    sort_dir = (request.args.get("sort_dir") or "desc").lower()
    reverse = sort_dir != "asc"

    def sort_key(row, key):
        if key == "appetite_status":
            if reverse:
                order_map = {"TARGET": 0, "IN": 1, "OUT": 2}
            else:
                order_map = {"OUT": 0, "IN": 1, "TARGET": 2}
            return order_map.get(str(row.get("appetite_status") or ""), 99)
        v = row.get(key)
        if isinstance(v, str):
            return v.lower()
        try:
            if v is None:
                return float("-inf") if reverse else float("inf")
            import math
            if isinstance(v, float) and math.isnan(v):
                return float("-inf") if reverse else float("inf")
        except Exception:
            pass
        return v

    valid_keys = {
        "id", "account_name", "primary_risk_state", "line_of_business",
        "total_premium", "tiv", "winnability", "appetite_status",
        "priority_score", "mode_score",
    }
    # Primary sort by target_opportunity desc, then requested sort
    rows.sort(
        key=lambda r: (1 if r.get("target_opportunity") else 0,
                       sort_key(r, sort_by) if sort_by in valid_keys else (r.get("priority_score") or 0.0)),
        reverse=True,
    )

    return jsonify({
        "count": len(rows),
        "data": rows,
        "mode": mode,
        "mode_explanation": mode_explanation,
    })


@app.route("/api/persona/list")
def api_persona_list():
    """List saved personas."""
    items = persona_list()
    # Shallow list for UI
    out = []
    for slug, rec in items.items():
        out.append({
            "slug": slug,
            "title": rec.get("title") or slug,
            "saved_at": rec.get("saved_at"),
        })
    active_slug, _rec = persona_get_active()
    return jsonify({"active": active_slug, "items": out})


@app.route("/api/persona/save", methods=["POST"])
def api_persona_save():
    """Save the current custom persona (or provided preset) under a name and make it selectable."""
    try:
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip() or None
        preset = data.get("preset")
        # If no preset provided, try using current MODES["custom"]
        if not isinstance(preset, dict) or not preset:
            preset = MODES.get("custom")
        if not isinstance(preset, dict) or not preset:
            return jsonify({"ok": False, "error": "No custom persona available to save"}), 400
        if not name:
            name = time.strftime("Persona %Y-%m-%d %H:%M")
        slug, rec = persona_add(name, preset)
        # Mirror into MODES and refresh index
        _load_personas_into_modes()
        return jsonify({"ok": True, "slug": slug, "record": rec})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/persona/activate", methods=["POST"])
def api_persona_activate():
    """Set a saved persona as active (also mirrors to MODES['custom'])."""
    try:
        data = request.get_json(silent=True) or {}
        slug = data.get("slug")
        if not slug:
            return jsonify({"ok": False, "error": "Missing slug"}), 400
        rec = persona_get(slug)
        if not rec:
            return jsonify({"ok": False, "error": "Persona not found"}), 404
        persona_set_active(slug)
        # Mirror into MODES
        preset = rec.get("preset") or {}
        if isinstance(preset, dict) and preset:
            MODES[slug] = preset
            MODES["custom"] = preset
        _load_personas_into_modes()
        return jsonify({"ok": True, "active": slug})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/persona/seed", methods=["POST"])
def api_persona_seed():
    try:
        data = request.get_json(silent=True) or {}
        answers = data.get("answers") or {}
        df = load_df("data.json")
        # Build seed deterministically from answers
        seed = seeds_from_answers(answers)
        # Register as custom mode
        MODES["custom"] = seed
        return jsonify({"ok": True, "mode": "custom", "seed": seed})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/persona/propose", methods=["POST"])
def api_persona_propose():
    try:
        data = request.get_json(silent=True) or {}
        answers = data.get("answers") or {}
        goal_text = data.get("goal_text") or ""
        # Load dataset and summarize
        df = load_df("data.json")
        summary = summarize_dataframe(df)
        # Seed and propose
        seed = seeds_from_answers(answers)
        proposal = propose_with_gemini(summary, seed, goal_text)
        # Register proposal as custom mode
        MODES["custom"] = proposal
        return jsonify({
            "ok": True,
            "mode": "custom",
            "seed": seed,
            "proposal": proposal,
            "debug_summary": summary,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


def _apply_common_filters(df):
    d = df
    state = request.args.get("state")
    status = request.args.get("status")
    lob = request.args.get("lob")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")
    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if status:
        d = d[d["appetite_status"].astype(str).str.upper() == status.upper()]
    if lob:
        d = d[d["line_of_business"].astype(str).str.upper() == lob.upper()]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]
    return d


@app.route("/api/aggregate")
def api_aggregate():
    """Generic aggregation endpoint for Portfolio Insights.
    Params:
      - group: column to group by (default primary_risk_state)
      - metric: one of count | sum:total_premium | sum:tiv | avg:total_premium
      - series_by: optional column to pivot series (e.g., appetite_status)
      - common filters: state,status,lob,min_premium,max_premium,q
    """
    df, _ = refreshCache()
    if df.empty:
        return jsonify({"labels": [], "series": []})
    group = request.args.get("group", "primary_risk_state")
    metric = request.args.get("metric", "count")
    series_by = request.args.get("series_by")
    d = _apply_common_filters(df)

    # Validate columns
    cols = set(d.columns)
    if group not in cols:
        return jsonify({"error": f"Unknown group '{group}'"}), 400
    if series_by and series_by not in cols:
        return jsonify({"error": f"Unknown series_by '{series_by}'"}), 400

    # Build aggregation
    def agg_series(frame):
        if metric == "count":
            return frame.size
        if metric.startswith("sum:"):
            col = metric.split(":", 1)[1]
            if col not in cols:
                return 0
            return frame[col].sum(min_count=1)
        if metric.startswith("avg:"):
            col = metric.split(":", 1)[1]
            if col not in cols:
                return 0
            return frame[col].mean()
        # default to count
        return frame.size

    if series_by:
        piv = (
            d.groupby([group, series_by])
            .apply(lambda x: agg_series(x))
            .unstack(fill_value=0)
        )
        piv = piv.sort_index()
        labels = [str(x) for x in list(piv.index)]
        series = []
        for col in piv.columns:
            series.append({
                "name": str(col),
                "data": [float(v) if v == v else 0.0 for v in list(piv[col].values)],
            })
    else:
        agg = d.groupby(group).apply(lambda x: agg_series(x)).sort_values(ascending=False)
        labels = [str(k) for k in list(agg.index)]
        series = [{
            "name": metric,
            "data": [float(v) if v == v else 0.0 for v in list(agg.values)],
        }]

    return jsonify({
        "labels": labels,
        "series": series,
        "meta": {"group": group, "metric": metric, "series_by": series_by},
    })


@app.route("/api/underlying")
def api_underlying():
    """Return underlying rows for a chart segment.
       Accepts same filters plus optional label (value for the group) and series_val.
    """
    df, _ = refreshCache()
    if df.empty:
        return jsonify({"count": 0, "data": []})
    group = request.args.get("group", "primary_risk_state")
    label = request.args.get("label")
    series_by = request.args.get("series_by")
    series_val = request.args.get("series_val")
    d = _apply_common_filters(df)
    if label is not None and group in d.columns:
        d = d[d[group].astype(str) == label]
    if series_by and series_val is not None and series_by in d.columns:
        d = d[d[series_by].astype(str) == series_val]
    try:
        d = mark_target_opportunities_df(d)
        d = d.copy()
        d["__top"] = d["target_opportunity"].astype(int)
        d = d.sort_values(["__top", "priority_score"], ascending=[False, False]).drop(columns=["__top"])
    except Exception:
        d = d.sort_values(["priority_score"], ascending=False)
    return jsonify({"count": len(d), "data": dfToDict(d)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
