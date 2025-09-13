// Auto-refresh helper with visibility pause, backoff, and stable hashing
// Usage:
// import { startAutoRefresh } from "/static/js/refresh.js";
// const ctl = startAutoRefresh({ targetId, endpointBuilder, onUpdate, hashSelector });

function djb2(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) + str.charCodeAt(i);
    hash = hash & 0xffffffff;
  }
  return (hash >>> 0).toString(16);
}

function prettyMode(name) {
  const s = String(name || "");
  return s.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export function startAutoRefresh({ targetId, endpointBuilder, onUpdate, hashSelector }) {
  const baseIntervalMs = 45_000;
  const maxIntervalMs = 300_000; // 5 min
  const state = {
    timer: null,
    errors: 0,
    lastHash: null,
    active: true,
    lastCount: 0,
  };

  const target = typeof targetId === 'string' ? document.getElementById(targetId) : targetId;

  function computeHash(rows) {
    try {
      const lean = Array.isArray(rows)
        ? rows.map(r => hashSelector ? hashSelector(r) : r)
        : rows;
      const s = JSON.stringify(lean);
      return djb2(s);
    } catch (e) {
      return String(Date.now());
    }
  }

  function intervalMs() {
    const i = baseIntervalMs * Math.pow(2, Math.max(0, state.errors));
    return Math.min(i, maxIntervalMs);
  }

  async function tick(initial = false) {
    if (!state.active || document.visibilityState === 'hidden') {
      scheduleNext();
      return;
    }
    let url = "";
    try {
      target?.classList?.add('is-loading');
      url = endpointBuilder();
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const json = await res.json();
      const rows = json?.data || [];
      const h = computeHash(rows);
      const changed = h !== state.lastHash;
      state.lastHash = h;
      state.lastCount = Array.isArray(rows) ? rows.length : 0;
      state.errors = 0;

      // Notify caller
      try { onUpdate?.(json, changed); } catch (e) { console.debug('onUpdate error', e); }

      // Toasts
      const mode = (localStorage.getItem('ACTIVE_MODE') || 'balanced_growth');
      if (!initial) {
        if (changed) {
          window.toast?.(`Updated ${state.lastCount} items — ${prettyMode(mode)}`, 'success');
        } else {
          // Quietly log for demo clarity
          console.debug('[refresh] No change; updated timestamp');
        }
      }
    } catch (err) {
      state.errors += 1;
      const wait = Math.round(intervalMs() / 1000);
      window.toast?.(`Connection lost, retrying in ${wait}s…`, 'error');
      console.debug('[refresh] error', err, 'url=', url);
    } finally {
      target?.classList?.remove('is-loading');
      scheduleNext();
    }
  }

  function scheduleNext() {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => tick(false), intervalMs());
    console.debug('[refresh] next in', Math.round(intervalMs()/1000), 's');
  }

  function resumeNow() {
    clearTimeout(state.timer);
    tick(false);
  }

  // Visibility handling
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      console.debug('[refresh] visible → immediate fetch');
      resumeNow();
    } else {
      console.debug('[refresh] hidden → paused');
    }
  });

  // LocalStorage mode change handling
  window.addEventListener('storage', (e) => {
    if (e.key === 'ACTIVE_MODE') {
      console.debug('[refresh] ACTIVE_MODE changed → refetch');
      state.errors = 0;
      resumeNow();
    }
  });

  // Public controller
  const controller = {
    triggerNow: resumeNow,
    stop: () => { state.active = false; clearTimeout(state.timer); },
    start: () => { if (!state.active) { state.active = true; resumeNow(); } },
    getLastCount: () => state.lastCount,
  };

  // Kick off
  tick(true);
  return controller;
}

