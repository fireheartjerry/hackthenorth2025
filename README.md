RiskOps Underwriting Dashboard

Quick Start

1. pip install -r requirements.txt
2. Create .env with: GEMINI_API_KEY=replace_me, GEMINI_MODEL=gemini-1.5-pro, USE_LOCAL_DATA=true
3. Run: python app.py and open http://localhost:5000

Features

- Percentile modes: unicorn, balanced, loose, turnaround
- Persona CUSTOM via Gemini (fallback to seed if no key)
- Gemini NLQ/explain + AI activity widget
- Live AI Chatbot with streaming responses and dashboard integration
- Auto-refresh with backoff and change detection
- Shortcuts: 1/2/3/4/C, / search, ? help
- Real-time chat with confirmation dialogs for actions

Key Endpoints

- GET /api/modes
- GET /api/classified_mode
- GET /api/mode/percentiles
- POST /api/persona/seed
- POST /api/persona/propose
- POST /api/mode/blend
- POST /api/chat (live AI chat)
- GET /api/chat/stream (streaming responses)
- POST /api/chat/execute (action execution)
- GET /api/chat/suggestions (dynamic suggestions)

Notes

- Source data: data.json supports {output:[{data:[...]}]} or {data:[...]}
- Works without Gemini key (deterministic seed; heuristic fallbacks)

