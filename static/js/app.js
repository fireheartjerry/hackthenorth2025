// Minimal helpers for consistent UX
window.toast = function(msg, type = "info") {
  const host = document.getElementById("toasts") || (function(){
    const d = document.createElement("div");
    d.id = "toasts"; d.className = "toasts"; document.body.appendChild(d); return d;
  })();
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  host.appendChild(el);
  setTimeout(()=>{ el.classList.add("out"); setTimeout(()=>host.removeChild(el), 200); }, 2800);
}

window.fetchJSON = async function(url, opts={}){
  const res = await fetch(url, opts);
  if(!res.ok){
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if(j && j.error) msg = j.error; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

window.withLoading = async function(el, fn){
  try{ el?.classList?.add("is-loading"); return await fn(); } finally { el?.classList?.remove("is-loading"); }
}

// Subtle hover glow interaction for elements with .hover-glow
document.addEventListener('pointermove', (e)=>{
  const target = e.target.closest('.hover-glow');
  if(!target) return;
  const rect = target.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 100;
  target.style.setProperty('--x', x + '%');
});

// Footer year helper
(()=>{
  try{
    const y = new Date().getFullYear();
    const el = document.getElementById('footer_year');
    if (el) el.textContent = String(y);
  }catch(_){/* noop */}
})();
