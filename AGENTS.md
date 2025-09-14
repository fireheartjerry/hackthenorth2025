# AGENTS.md — Federato AppetizerIQ Hackathon Brief (for Agents/Codex)

> Sources: **Hack The North Federato Challenge.pdf**, **Hack The North – Federato Challenge – API Documentation.pdf**, **Hack The North – Federato Challenge – Underwriting Guidelines Sample.pdf**, **Hack The North – Federato Challenge – Underwriting Guidelines Explained.pdf**, **Hack The North – Glossary Federato Challenge.pdf**, **API workshop.pptx**.

---

## 1) Mission & Objective 
Design a **next‑generation AppetizerIQ underwriting dashboard** that, on user login, **surfaces and prioritizes high‑value, in‑appetite submissions** so underwriters can act faster and more effectively.

- **Primary Goal:** Reimagine the *landing page* experience to intelligently curate submissions aligned with **carrier appetite**.
- **Real‑world framing:** Mirrors challenges Federato’s **Forward Deployed Engineering (FDE)** team tackles.
- **Deliverable:** A working, visually polished B2B SaaS‑style dashboard + backend logic aligning to appetite rules.

(Ref: *Hack The North Federato Challenge.pdf*)

---

## 2) Provided Resources
1. **Sample Dataset via API** (read‑only).
2. **Endpoint Documentation** with field definitions and usage.
3. **$50 OpenAI credit per team**.
4. **Sample Appetite Guidelines** mapping to fields in the data.

(Refs: *Hack The North Federato Challenge.pdf*, *API Documentation.pdf*, *Underwriting Guidelines Sample.pdf*, *Underwriting Guidelines Explained.pdf*)

---

## 3) Data Access — API & Auth
- **Auth:** OAuth 2.0 **Client Credentials**. Obtain **Bearer access_token** from Auth0, then call the integrations API.
  - Auth URL: `https://product-federato.us.auth0.com/oauth/token`
  - Body: `client_id`, `client_secret`, `audience="https://product.federato.ai/core-api"`, `grant_type="client_credentials"`
- **Endpoint (Policies):** Base `https://product.federato.ai/integrations-api`
  - Preferred: `POST /handlers/all-policies?outputOnly=true`
  - Note: Some environments expose a working **typo** path: `POST /handlers/all-pollicies?outputOnly=true`
- **Headers:** `Authorization: Bearer <token>`, `Content-Type: application/json`
- **Response Shape:** JSON object. Official docs show `{ "data": [ { policy... } ] }`. Some exports appear as `{ "output": [ { "data": [ ... ] } ] }` — normalize both.
- **Policy Fields (non‑exhaustive):**
  - `id` (int)
  - `tiv` (number) — Total Insured Value
  - `created_at` (timestamp UTC)
  - `loss_value` (string → parse to number)
  - `winnability` (0–1 or %‑like integer → normalize to 0–1)
  - `account_name` (string)
  - `total_premium` (number)
  - `effective_date`, `expiration_date` (timestamps)
  - `oldest_building` (int year)
  - `primary_risk_state` (2‑letter)
  - `renewal_or_new_business` ("NEW_BUSINESS" | "RENEWAL")
  - (Some data samples include `line_of_business`, `construction_type`)
  
(Ref: *API Documentation.pdf*)

---

## 4) Underwriting Appetite — Rules to Encode
Implement the **Commercial Property** appetite rules exactly; use them to compute **status** and **explanations**. Keep logic explainable and traceable for UI/AI.

### 4.1 General
- **Risk Selection Philosophy:** Favor profitable, sustainable risks with sound practices, good construction, manageable losses.
- **Adherence:** Deviations require manager approval (note for future; for hackathon treat as hard rules).

### 4.2 Core Data Required
`Account Name`, `Primary Risk State`, `Line of Business`, `Effective/Expiration Dates`, `TIV`, `Construction Type`, `Building Year`, `Premium`, `Loss Value`.

### 4.3 Accept/Target/Not‑Acceptable Thresholds
- **Submission Type:** NEW_BUSINESS = Acceptable; RENEWAL = Not acceptable.
- **Line of Business:** COMMERCIAL PROPERTY = Acceptable; others = Not acceptable.
- **Primary Risk State:**
  - **Acceptable:** `OH, PA, MD, CO, CA, FL, NC, SC, GA, VA, UT`
  - **Target:** `OH, PA, MD, CO, CA, FL`
  - **Not acceptable:** All others
- **TIV:**
  - `≤ 150M` Acceptable
  - `50M – 100M` Target
  - `> 150M` Not acceptable
- **Total Premium:**
  - `50k – 175k` Acceptable
  - `75k – 100k` Target
  - `< 50k` or `> 175k` Not acceptable
- **Building Age:** (by `oldest_building`)
  - Newer than **1990** Acceptable
  - Newer than **2010** Target
  - Older than **1990** Not acceptable
- **Construction Type:** Acceptable if **≥50%** are **JM, Non‑Combustible/Steel, Masonry Non‑Combustible, Fire Resistive**. (Sample data has a single value; treat listed types as acceptable.)
- **Loss Value:**
  - `< 100k` Acceptable
  - `≥ 100k` Not acceptable

(Ref: *Underwriting Guidelines Sample.pdf*; application guidance in *Underwriting Guidelines Explained.pdf*)

### 4.4 Classification Output
- **appetite_status:** `"TARGET" | "IN" | "OUT"`
- **appetite_reasons:** list of triggered rule explanations
- Optional **appetite_score:** additive score from target/acceptable hits
- Optional **priority_score:** combine appetite_score with `winnability` to sort

---

## 5) UX/UI — Landing Page & Dashboard Expectations
- **Landing Experience:** On login, show a **curated worklist** of the most aligned, highest‑priority submissions with **badges**, **scores**, and **quick context** (state, LOB, premium, TIV).
- **Transparency:** Every prioritized item should display or provide access to a clear **“why”** (trace to rules).
- **Controls:** Prominent filtering/sorting (state, appetite status, premium range, search), fast drill‑down to details.
- **Modern B2B SaaS Quality:** Clean typography, grid/spacing discipline, command‑palette or global search (optional), loading skeletons, subtle micro‑interactions.
- **Portfolio Alignment:** Show small KPI tiles (counts by status, avg premium, distribution). Optional portfolio summary (by state, TIV bands).
- **Accessibility & Speed:** Keyboard‑friendly, responsive, high‑contrast/dark mode friendly.

(Refs: *Hack The North Federato Challenge.pdf*, *Glossary Federato Challenge.pdf*, plus best‑practice interpretations)

---

## 6) AI (Gemini) — Enhanced AI Features
- **One‑liner Appetite Explanations:** Given a submission's fields + the guideline text, generate a concise reason for `"TARGET"/"IN"/"OUT"` using Gemini Flash 2.0.
- **Natural‑Language Query (NLQ):** Convert English queries like "out‑of‑appetite in FL > 100k premium" into structured filters with Gemini intelligence.
- **Assistant Q&A:** "Why was submission 123 out‑of‑appetite?" returns rule‑referenced explanation; "Show top 5 targets this week" returns a filtered/sorted set.
- **AI Activity Tracking:** Real-time widget displays what AI operations were performed, whether AI was used or fallback applied, and relevant notes.

(Powered by Google's Gemini Flash 2.0 for faster, more efficient AI processing)

---

## 7) Engineering Requirements (Backend + Frontend)
### 7.1 Backend (Flask or Django; sample uses Flask)
- **Secrets:** Store `client_id`, `client_secret`, `audience` in env; optionally accept direct `FEDERATO_TOKEN` for speed.
- **OAuth Flow:** Fetch token, cache until expiry; add `Authorization: Bearer` on each call.
- **Fetching:** Call `POST /handlers/all-policies?outputOnly=true` (fallback to typo path as needed). Normalize `{ data: [...] }` vs `{ output: [{ data: [...] }] }`.
- **DataFrame Layer:** Load into **pandas**; fix types (`loss_value` to float, timestamps to datetime, winnability to 0–1). Derive `building_age`.
- **Rules Engine:** Implement exact thresholds; compute `appetite_status`, `appetite_reasons`, `appetite_score`, `priority_score`.
- **Caching:** Memory or Redis cache of raw + classified DataFrame (e.g., 5–10 min TTL). Fallback to local `data.json` if API issues.
- **API Endpoints:**
  - `/api/policies` — raw+classified
  - `/api/classified` — supports filters: `state`, `status`, `min_premium`, `max_premium`, `q`
  - `/api/explain/<id>` — Gemini AI one‑liner explanation with usage tracking
  - `/api/nlq` — English → filter JSON with Gemini intelligence and activity logs
- **Error Handling:** Graceful 401/5xx handling; surface UI toasts/messages.

### 7.2 Frontend (templates or SPA; sample uses server‑rendered + Tailwind)
- **Pages:** `/` overview with KPI tiles + prioritized list; `/submissions` table with filters; `/detail/<id>` drawer/page.
- **Interactions:** Quick filters, search, sort; “Explain” action calls AI endpoint; optional command palette.
- **Visual System:** Tailwind utilities, dark theme, consistent chips/badges for `"TARGET"/"IN"/"OUT"`.

---

## 8) Non‑Functional & Deliverable Notes
- **Explainability first:** Every classification is auditably tied to the written guideline rules.
- **Performance:** Cached fetch + vectorized pandas transforms for snappy UI.
- **Security:** No secrets on client; HTTPS; minimal logs redact tokens.
- **Extensibility:** Rules kept modular for easy tuning; add states/thresholds via config later.
- **Demo Polish:** Loading states, empty‑state messaging, keyboard nav, responsive layout.

---

## 9) Environment & Config (suggested)
```
FEDERATO_TOKEN=...                    # or use CLIENT_ID/SECRET to fetch
FEDERATO_CLIENT_ID=...
FEDERATO_CLIENT_SECRET=...
FEDERATO_AUDIENCE=https://product.federato.ai/core-api
FEDERATO_AUTH_URL=https://product-federato.us.auth0.com/oauth/token
FEDERATO_POLICIES_URL=https://product.federato.ai/integrations-api/handlers/all-policies?outputOnly=true
FEDERATO_POLICIES_URL_TYPO=https://product.federato.ai/integrations-api/handlers/all-pollicies?outputOnly=true
USE_LOCAL_DATA=false
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash-exp
```
(Refs: *API Documentation.pdf*; typo path observed in real responses.)

---

## 10) Step‑By‑Step Implementation Plan
1. **Auth & Fetch:** Implement OAuth client‑credentials; fetch policies; normalize payload shape.
2. **DataFrame Build:** Parse types; derive `building_age`; fix `winnability` scale.
3. **Rules Engine:** Encode appetite thresholds; produce status + reasons + scores.
4. **Cache Layer:** Memoize raw+classified; scheduled refresh or manual button.
5. **API Surface:** `/api/classified` with filters; `/api/explain` and `/api/nlq` using Gemini Flash 2.0 with activity tracking.
6. **UI Build:** KPI tiles, prioritized list, filters+table, detail view with “Explain” CTA.
7. **Polish & QA:** Empty states, loading skeletons, accessibility, responsive, sanity tests on rules.
8. **Demo Scripts:** Seed `.env`, run app, present curated “wow” scenarios (targets vs outs).

---

## 11) Acceptance Criteria
- Landing page shows **prioritized, in‑appetite** submissions at the top with reasons available.
- Filters work (state, status, premium range, search).
- At least one **AI‑generated** rationale and **NLQ** path demonstrated using the credit.
- Rules match **Underwriting Guidelines Sample.pdf** thresholds; decisions are explainable.
- API auth works per **API Documentation.pdf**; if API unavailable, local `data.json` fallback works.
- UI meets **modern B2B** quality: clean, responsive, accessible, polished.

---

## 12) Glossary (select)
- **AppetizerIQ:** Tools/workflows aligning individual underwriting to portfolio strategy.
- **Appetite:** Types of risk a carrier wants to write (by product, geography, size, etc.).
- **In‑Appetite / Out‑of‑Appetite:** Matches vs violates appetite rules.
- **Submission:** A broker/agent request to insure an exposure.
- **TIV:** Total Insured Value.
- **Winnability:** Likelihood of winning the business (0–1), provided in data.

(Ref: *Glossary Federato Challenge.pdf*)

---

## 13) Notes & Caveats
- Sample data includes additional LOBs (e.g., UMBRELLA, CYBER). For the challenge, **only COMMERCIAL PROPERTY** is acceptable per guidelines; others should classify Out unless instructed otherwise.
- Real‑world guidelines mention **“>50% acceptable construction types”** across locations; sample has one `construction_type`. Treat listed acceptable types as pass, others fail.
- `loss_value` input is a **string**; always parse to number.
- Some records show `winnability` as an integer percent (e.g., 86); normalize to 0–1.
- If `state` like `CE` appears, treat as non‑acceptable.
- Keep a central **explain()** that maps triggered rule checks to human words (and feeds AI).

