import json
import os
import re
import time
from typing import Dict, Tuple, Optional


PERSONAS_PATH = os.path.join(os.getcwd(), "personas.json")


def _slugify(name: str) -> str:
    name = (name or "").strip().lower()
    # replace non-alphanum with underscore, collapse repeats
    s = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if not s:
        s = f"persona_{int(time.time())}"
    return s


def load_store() -> Dict:
    if not os.path.exists(PERSONAS_PATH):
        return {"active": None, "items": {}}
    try:
        with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"active": None, "items": {}}
        # normalize
        items = data.get("items") or {}
        if not isinstance(items, dict):
            items = {}
        active = data.get("active")
        return {"active": active, "items": items}
    except Exception:
        return {"active": None, "items": {}}


def save_store(store: Dict) -> None:
    data = {
        "active": store.get("active"),
        "items": store.get("items", {}),
    }
    with open(PERSONAS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add_persona(name: str, preset: Dict, meta: Optional[Dict] = None) -> Tuple[str, Dict]:
    store = load_store()
    slug = _slugify(name)
    # Ensure uniqueness by suffixing
    base = slug
    i = 2
    while slug in store.get("items", {}):
        slug = f"{base}_{i}"
        i += 1
    rec = {
        "title": name or slug,
        "slug": slug,
        "preset": preset or {},
        "meta": meta or {},
        "saved_at": int(time.time()),
    }
    store.setdefault("items", {})[slug] = rec
    store.setdefault("active", slug)
    save_store(store)
    return slug, rec


def list_personas() -> Dict[str, Dict]:
    store = load_store()
    return store.get("items", {})


def get_persona(slug: str) -> Optional[Dict]:
    return list_personas().get(slug)


def set_active(slug: Optional[str]) -> Optional[str]:
    store = load_store()
    items = store.get("items", {})
    if slug is None:
        store["active"] = None
    elif slug in items:
        store["active"] = slug
    else:
        return None
    save_store(store)
    return store.get("active")


def get_active() -> Tuple[Optional[str], Optional[Dict]]:
    store = load_store()
    slug = store.get("active")
    if slug and slug in store.get("items", {}):
        return slug, store["items"][slug]
    return None, None

