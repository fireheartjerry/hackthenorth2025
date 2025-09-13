// Enhanced UX helpers with modern features
window.toast = function(msg, type = "info") {
  const host = document.getElementById("toasts") || (function(){
    const d = document.createElement("div");
    d.id = "toasts"; d.className = "toasts"; document.body.appendChild(d); return d;
  })();
  
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  
  // Enhanced toast with icon
  const icons = {
    success: '✅',
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️'
  };
  
  el.innerHTML = `
    <div class="toast-content">
      <span class="toast-icon">${icons[type] || icons.info}</span>
      <span class="toast-message">${msg}</span>
    </div>
  `;
  
  host.appendChild(el);
  
  // Enhanced animations
  requestAnimationFrame(() => {
    el.style.transform = 'translateX(0)';
    el.style.opacity = '1';
  });
  
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, 300);
  }, 3500);
  
  // Click to dismiss
  el.addEventListener('click', () => {
    el.classList.add("out");
    setTimeout(() => {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, 300);
  });
}

window.fetchJSON = async function(url, opts = {}) {
  // Add request interceptor for better UX
  const startTime = Date.now();
  
  try {
    const res = await fetch(url, {
      ...opts,
      headers: {
        'Content-Type': 'application/json',
        ...opts.headers
      }
    });
    
    const duration = Date.now() - startTime;
    
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try { 
        const j = await res.json(); 
        if (j && j.error) msg = j.error; 
      } catch {}
      
      // Log network errors for debugging
      console.error(`Network error: ${msg} (${duration}ms)`);
      throw new Error(msg);
    }
    
    // Log successful requests in dev mode
    if (duration > 1000) {
      console.warn(`Slow request: ${url} took ${duration}ms`);
    }
    
    return res.json();
  } catch (error) {
    // Enhanced error handling
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network connection failed. Please check your internet connection.');
    }
    throw error;
  }
}

window.withLoading = async function(el, fn) {
  try {
    el?.classList?.add("is-loading");
    
    // Add loading skeleton if element supports it
    if (el && el.classList.contains('table-body')) {
      const skeleton = createTableSkeleton();
      el.appendChild(skeleton);
    }
    
    return await fn();
  } finally {
    el?.classList?.remove("is-loading");
    
    // Remove skeleton
    const skeleton = el?.querySelector('.loading-skeleton');
    if (skeleton) skeleton.remove();
  }
}

// Enhanced loading states
function createTableSkeleton() {
  const skeleton = document.createElement('tr');
  skeleton.className = 'loading-skeleton';
  skeleton.innerHTML = `
    <td colspan="100%">
      <div class="skeleton-rows">
        <div class="skeleton-row"></div>
        <div class="skeleton-row"></div>
        <div class="skeleton-row"></div>
      </div>
    </td>
  `;
  return skeleton;
}

// Enhanced hover effects
document.addEventListener('pointermove', (e) => {
  const target = e.target.closest('.hover-glow');
  if (!target) return;
  
  const rect = target.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 100;
  const y = ((e.clientY - rect.top) / rect.height) * 100;
  
  target.style.setProperty('--x', x + '%');
  target.style.setProperty('--y', y + '%');
});

// Modern interaction utilities
window.debounce = function(func, wait, immediate) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      timeout = null;
      if (!immediate) func(...args);
    };
    const callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    if (callNow) func(...args);
  };
}

window.throttle = function(func, limit) {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }
}

// Enhanced animation utilities
window.animateValue = function(start, end, duration, callback) {
  const startTime = performance.now();
  const change = end - start;
  
  function animate(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Easing function (ease-out cubic)
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = start + (change * easeOut);
    
    callback(current);
    
    if (progress < 1) {
      requestAnimationFrame(animate);
    }
  }
  
  requestAnimationFrame(animate);
}

// Performance monitoring
window.performanceMonitor = {
  marks: new Map(),
  
  start(name) {
    this.marks.set(name, performance.now());
  },
  
  end(name) {
    const start = this.marks.get(name);
    if (start) {
      const duration = performance.now() - start;
      this.marks.delete(name);
      return duration;
    }
    return null;
  },
  
  log(name) {
    const duration = this.end(name);
    if (duration !== null) {
      console.log(`⏱️ ${name}: ${duration.toFixed(2)}ms`);
    }
  }
};

// Global error handler for better UX
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error);
  
  // Don't show error toasts for script loading errors
  if (event.error && !event.filename.includes('.js')) {
    toast('An unexpected error occurred. Please try refreshing the page.', 'error');
  }
});

// Enhanced keyboard navigation
window.addKeyboardNavigation = function(container, selector) {
  let currentIndex = -1;
  const items = () => container.querySelectorAll(selector);
  
  container.addEventListener('keydown', (e) => {
    const itemList = items();
    if (itemList.length === 0) return;
    
    switch(e.key) {
      case 'ArrowDown':
        e.preventDefault();
        currentIndex = Math.min(currentIndex + 1, itemList.length - 1);
        updateFocus();
        break;
        
      case 'ArrowUp':
        e.preventDefault();
        currentIndex = Math.max(currentIndex - 1, 0);
        updateFocus();
        break;
        
      case 'Enter':
        e.preventDefault();
        if (currentIndex >= 0 && itemList[currentIndex]) {
          itemList[currentIndex].click();
        }
        break;
        
      case 'Escape':
        currentIndex = -1;
        updateFocus();
        break;
    }
  });
  
  function updateFocus() {
    const itemList = items();
    itemList.forEach((item, index) => {
      item.classList.toggle('keyboard-focus', index === currentIndex);
    });
    
    if (currentIndex >= 0 && itemList[currentIndex]) {
      itemList[currentIndex].scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      });
    }
  }
};

// Modern clipboard utility
window.copyToClipboard = async function(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied to clipboard', 'success');
  } catch (err) {
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    toast('Copied to clipboard', 'success');
  }
};

// Initialize enhanced features when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Add enhanced CSS for toasts
  const style = document.createElement('style');
  style.textContent = `
    .toast {
      transform: translateX(100%);
      opacity: 0;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      cursor: pointer;
      min-width: 300px;
    }
    
    .toast-content {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .toast-icon {
      font-size: 1.1rem;
    }
    
    .toast-message {
      flex: 1;
    }
    
    .skeleton-rows {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      padding: 1rem;
    }
    
    .keyboard-focus {
      outline: 2px solid var(--indigo);
      outline-offset: 2px;
    }
  `;
  document.head.appendChild(style);
  
  console.log('🚀 Enhanced UX features loaded');
});
