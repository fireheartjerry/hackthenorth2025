// Keyboard shortcuts
// 1 Unicorn, 2 Balanced, 3 Loose, 4 Turnaround, C Custom, / search, ? help
(function(){
  function setMode(m){ try{ localStorage.setItem('ACTIVE_MODE', m); window.toast?.('Mode → ' + m.replace(/_/g,' '), 'success'); }catch(e){} }
  function focusSearch(){ const q = document.getElementById('f_q'); if(q){ q.focus(); q.select(); } }
  function showHelp(){ alert('Shortcuts:\n1 Unicorn\n2 Balanced\n3 Loose\n4 Turnaround\nC Custom\n/ Search'); }
  document.addEventListener('keydown', (e)=>{
    if (e.target && ['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key==='1'){ setMode('unicorn_hunting'); e.preventDefault(); }
    else if (e.key==='2'){ setMode('balanced_growth'); e.preventDefault(); }
    else if (e.key==='3'){ setMode('loose_fits'); e.preventDefault(); }
    else if (e.key==='4'){ setMode('turnaround_bets'); e.preventDefault(); }
    else if (e.key.toLowerCase()==='c'){ setMode('custom'); e.preventDefault(); }
    else if (e.key==='/'){ focusSearch(); e.preventDefault(); }
    else if (e.key==='?'){ showHelp(); e.preventDefault(); }
  });
})();


