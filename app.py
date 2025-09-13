from flask import Flask, request, jsonify, render_template
import os, json, time, datetime as dt
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

CACHE_TTL = 300
_cache_data = {"ts": 0, "df": None, "raw": None}

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


def classifyRow(row):
    reasons = []
    status = "IN"
    score = 0

    lob = str(row.get("line_of_business", "") or "").strip().upper()
    if lob != "COMMERCIAL PROPERTY":
        reasons.append("Line of Business not acceptable")

    sub_type = str(row.get("renewal_or_new_business", "") or "").strip().upper()
    if sub_type == "RENEWAL":
        reasons.append("Renewal business not acceptable")

    st = str(row.get("primary_risk_state", "") or "").strip().upper()
    if st not in ACCEPT_STATES:
        reasons.append("Primary risk state not acceptable")

    tiv = row.get("tiv", np.nan)
    if pd.notna(tiv):
        if tiv > 150_000_000:
            reasons.append("TIV exceeds 150M cap")
        elif 50_000_000 <= tiv <= 100_000_000:
            score += 2
        elif tiv <= 150_000_000:
            score += 1

    premium = row.get("total_premium", np.nan)
    if pd.notna(premium):
        if premium < 50_000 or premium > 175_000:
            reasons.append("Premium outside acceptable range")
        elif 75_000 <= premium <= 100_000:
            score += 2
        else:
            score += 1

    age = row.get("building_age", np.nan)
    if pd.notna(age):
        # newer than 2010 => target; newer than 1990 => acceptable; older than 1990 => not acceptable
        if age <= (CURRENT_YEAR - 2010):
            score += 2
        elif age <= (CURRENT_YEAR - 1990):
            score += 1
        else:
            reasons.append("Building age older than 1990")

    ctype = str(row.get("construction_type", "") or "").strip()
    if ctype in ACCEPT_CONSTRUCTION:
        score += 1
    else:
        reasons.append("Construction type not acceptable")

    loss = row.get("loss_value", np.nan)
    if pd.notna(loss):
        # exact match fix: >= 100_000 is Not Acceptable
        if loss >= 100_000:
            reasons.append("Loss value >= 100k")
        else:
            score += 1

    if reasons:
        status = "OUT"
    else:
        status = "IN"
        if (
            st in TARGET_STATES
            and 75_000 <= (premium or 0) <= 100_000
            and 50_000_000 <= (tiv or 0) <= 100_000_000
            and age is not np.nan and age <= (CURRENT_YEAR - 2010)
        ):
            status = "TARGET"

    w = row.get("winnability", np.nan)
    if pd.notna(w):
        if isinstance(w, (int, float)) and w > 1:
            w = w / 100.0
        w = max(0.0, min(1.0, float(w)))
    else:
        w = 0.5

    appetite_score = score
    priority_score = appetite_score * 0.6 + w * 0.4

    try:
        with open("scores_debug.txt", "a", encoding="utf-8") as f:
            f.write(
                f"id={row.get('id')}, status={status}, appetite_score={appetite_score}, priority_score={priority_score}, reasons={reasons}\n"
            )
    except Exception:
        pass

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
    return df


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
    in_ct = int((df["appetite_status"] == "IN").sum()) if not df.empty else 0
    tgt_ct = int((df["appetite_status"] == "TARGET").sum()) if not df.empty else 0
    out_ct = int((df["appetite_status"] == "OUT").sum()) if not df.empty else 0
    avg_premium = float(df["total_premium"].mean()) if not df.empty else 0.0
    return render_template(
        "index.html",
        total=total,
        in_ct=in_ct,
        tgt_ct=tgt_ct,
        out_ct=out_ct,
        avg_premium=avg_premium,
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
    # Sort: TARGET -> IN -> OUT, then by priority score desc, premium desc
    try:
        d = d.copy()
        d["__s"] = pd.Categorical(
            d["appetite_status"], categories=["TARGET", "IN", "OUT"], ordered=True
        )
        d = d.sort_values(["__s", "priority_score", "total_premium"], ascending=[True, False, False])
        d = d.drop(columns=["__s"])
    except Exception:
        d = d.sort_values(["priority_score"], ascending=False)
    rows = dfToDict(d)
    return render_template("submissions.html", rows=rows)


@app.route("/detail/<int:pid>")
def detail(pid):
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
    d = d.sort_values(["priority_score"], ascending=False)
    return jsonify({"count": len(d), "data": dfToDict(d)})


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


def makePolicyDict(pid):
    df, _ = refreshCache()
    d = df[df["id"] == pid]
    if d.empty:
        return None
    return dfToDict(d)[0]


@app.route("/api/explain/<int:pid>", methods=["POST"])
def apiExplain(pid):
    item = makePolicyDict(pid)
    if not item:
        return jsonify({"ok": False, "error": "Not found"}), 404
    base = buildGuidelineSummary()
    rule_text = base
    raw_reason = item.get("appetite_reasons", [])
    label = item.get("appetite_status", "IN")
    if not OPENAI_API_KEY:
        s = f"{label}: " + (
            "; ".join(raw_reason)
            if raw_reason
            else "Meets appetite and target criteria"
        )
        return jsonify({"ok": True, "explanation": s})
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"You are an underwriting assistant. Explain in one concise sentence why this submission is {label}. Use the guidelines and the fields. Guidelines: {rule_text}. Fields: {json.dumps(item, default=str)}."
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        txt = resp.choices[0].message.content.strip()
        return jsonify({"ok": True, "explanation": txt})
    except Exception as e:
        s = f"{label}: " + ("; ".join(raw_reason) if raw_reason else "Meets appetite and target criteria")
        return jsonify({"ok": True, "explanation": s, "note": "AI fallback"})


@app.route("/api/nlq", methods=["POST"])
def apiNlq():
    data = request.get_json(silent=True) or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"filters": {}, "note": "Empty query"})
    if not OPENAI_API_KEY:
        st = None
        status = None
        min_p = None
        max_p = None
        for tok in q.split():
            if tok.upper() in ACCEPT_STATES:
                st = tok.upper()
        if "out" in q.lower():
            status = "OUT"
        elif "target" in q.lower():
            status = "TARGET"
        elif "in" in q.lower():
            status = "IN"
        return jsonify(
            {
                "filters": {
                    "state": st,
                    "status": status,
                    "min_premium": min_p,
                    "max_premium": max_p,
                },
                "note": "Heuristic",
            }
        )
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    sys = "Extract filters from a natural-language underwriting query. Respond as JSON with keys: state (2-letter or null), status (IN|OUT|TARGET|null), min_premium (number|null), max_premium (number|null), search (string|null)."
    u = f"Query: {q}"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": u}],
            temperature=0.1,
            max_tokens=120,
        )
        txt = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = {
                "state": None,
                "status": None,
                "min_premium": None,
                "max_premium": None,
                "search": None,
            }
        return jsonify({"filters": parsed})
    except Exception:
        # Heuristic fallback on failure
        st = None
        status = None
        min_p = None
        max_p = None
        for tok in q.split():
            if tok.upper() in ACCEPT_STATES:
                st = tok.upper()
        if "out" in q.lower():
            status = "OUT"
        elif "target" in q.lower():
            status = "TARGET"
        elif "in" in q.lower():
            status = "IN"
        return jsonify(
            {
                "filters": {
                    "state": st,
                    "status": status,
                    "min_premium": min_p,
                    "max_premium": max_p,
                    "search": None,
                },
                "note": "Heuristic fallback",
            }
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
