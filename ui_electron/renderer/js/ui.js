// Shared UI building blocks.
import { html, useEffect, useLayoutEffect, useRef, useState } from './lib.js';
import { getState, setState, useStore } from './store.js';

// ── Small controls ──────────────────────────────────────────────────────────

export function Label({ children, right }) {
  if (right === undefined) return html`<div class="label">${children}</div>`;
  return html`<div class="label-row"><div class="label">${children}</div>${right}</div>`;
}

export function Btn({ children, variant = '', size = '', busy = false, class: cls = '', ...props }) {
  const classes = ['btn', variant, size, cls].filter(Boolean).join(' ');
  return html`<button class=${classes} ...${props}>${busy ? html`<span class="spinner"></span>` : null}${children}</button>`;
}

export function Seg({ value, options, onChange, disabled = false, class: cls = '' }) {
  return html`<div class=${`seg ${cls}`} role="radiogroup">
    ${options.map((opt) => {
      const o = typeof opt === 'string' ? { value: opt, label: opt } : opt;
      return html`<button
        role="radio"
        aria-checked=${o.value === value}
        class=${o.value === value ? 'on' : ''}
        disabled=${disabled || o.disabled}
        title=${o.title || ''}
        onClick=${() => o.value !== value && onChange && onChange(o.value)}
      >${o.label}</button>`;
    })}
  </div>`;
}

export function Check({ checked, onChange, children, disabled = false, aside, asideClass = '', large = false, title = '' }) {
  return html`<button
    class=${`check ${checked ? 'on' : ''} ${large ? 'lg' : ''}`}
    disabled=${disabled}
    title=${title}
    onClick=${() => onChange && onChange(!checked)}
  ><span class="box"></span><span>${children}</span>${aside ? html`<span class=${`aside ${asideClass}`}>${aside}</span>` : null}</button>`;
}

export function Radio({ checked, onChange, children }) {
  return html`<button class=${`radio ${checked ? 'on' : ''}`} onClick=${() => !checked && onChange && onChange()}>
    <span class="r"></span><span>${children}</span>
  </button>`;
}

export function Dot({ kind = '', large = false }) {
  return html`<span class=${`dotc ${kind} ${large ? 'lg' : ''}`}></span>`;
}

export function Bar({ value = 0, kind = '', large = false, style }) {
  return html`<div class=${`bar ${kind} ${large ? 'lg' : ''}`} style=${style}><div style=${{ width: `${Math.max(0, Math.min(100, value))}%` }}></div></div>`;
}

export function Slider({ value, min = 0, max = 1, step = 0.01, onChange, onCommit, disabled = false }) {
  const pctValue = ((Number(value) - min) / (max - min)) * 100;
  return html`<div class="slider" style=${{ '--pct': `${pctValue}%` }}>
    <input type="range" min=${min} max=${max} step=${step} value=${value} disabled=${disabled}
      onInput=${(e) => onChange && onChange(Number(e.currentTarget.value))}
      onChange=${(e) => onCommit && onCommit(Number(e.currentTarget.value))} />
  </div>`;
}

export function Select({ value, options, onChange, disabled = false, large = false, title = '' }) {
  return html`<select class=${`select ${large ? 'lg' : ''}`} value=${value} disabled=${disabled} title=${title}
    onChange=${(e) => onChange && onChange(e.currentTarget.value)}>
    ${options.map((opt) => {
      const o = typeof opt === 'string' ? { value: opt, label: opt } : opt;
      return html`<option value=${o.value} disabled=${o.disabled} selected=${o.value === value}>${o.label}</option>`;
    })}
  </select>`;
}

/** Text input that keeps local state while typing and reports on commit. */
export function TextField({ value, onCommit, onInput, placeholder = '', large = false, multiline = false, rows = 3, disabled = false, onEnter, class: cls = '', title = '' }) {
  const [local, setLocal] = useState(value ?? '');
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setLocal(value ?? '');
  }, [value]);
  const commit = () => {
    if ((local ?? '') !== (value ?? '') && onCommit) onCommit(local);
  };
  const common = {
    value: local,
    placeholder,
    disabled,
    title,
    onFocus: () => { focused.current = true; },
    onBlur: () => { focused.current = false; commit(); },
    onInput: (e) => { setLocal(e.currentTarget.value); onInput && onInput(e.currentTarget.value); },
  };
  if (multiline) {
    return html`<textarea class=${`textarea ${cls}`} rows=${rows} ...${common}></textarea>`;
  }
  return html`<input class=${`input ${large ? 'lg' : ''} ${cls}`} ...${common}
    onKeyDown=${(e) => { if (e.key === 'Enter') { commit(); onEnter && onEnter(local); } }} />`;
}

export function NumberField({ value, onCommit, min, max, step = 1, disabled = false, special, title = '' }) {
  const [local, setLocal] = useState(String(value ?? ''));
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setLocal(String(value ?? ''));
  }, [value]);
  const commit = () => {
    let n = step < 1 ? parseFloat(local) : parseInt(local, 10);
    if (Number.isNaN(n)) n = value;
    if (min != null) n = Math.max(min, n);
    if (max != null) n = Math.min(max, n);
    setLocal(String(n));
    if (n !== value && onCommit) onCommit(n);
  };
  const showSpecial = special && Number(local) === min && !focused.current;
  return html`<input class="input num" type=${showSpecial ? 'text' : 'number'} min=${min} max=${max} step=${step}
    disabled=${disabled} title=${title}
    value=${showSpecial ? special : local}
    onFocus=${() => { focused.current = true; }}
    onBlur=${() => { focused.current = false; commit(); }}
    onInput=${(e) => setLocal(e.currentTarget.value)}
    onKeyDown=${(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }} />`;
}

// ── Toasts ───────────────────────────────────────────────────────────────────

let toastId = 0;
export function toast(message, level = 'info', timeout = 5000) {
  const id = ++toastId;
  setState((s) => ({ toasts: [...(s.toasts || []), { id, message, level }].slice(-5) }));
  if (timeout) setTimeout(() => dismissToast(id), timeout);
  return id;
}

export function dismissToast(id) {
  setState((s) => ({ toasts: (s.toasts || []).filter((t) => t.id !== id) }));
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts || []);
  if (!toasts.length) return null;
  return html`<div class="toasts">
    ${toasts.map((t) => html`<div class=${`toast ${t.level}`} key=${t.id}>
      <div>${t.message}</div>
      <button class="xbtn x" onClick=${() => dismissToast(t.id)}>✕</button>
    </div>`)}
  </div>`;
}

/** Show an error from an RPC failure as a toast. */
export function reportError(err, prefix = '') {
  const message = err && err.message ? err.message : String(err);
  toast(prefix ? `${prefix}: ${message}` : message, 'error', 8000);
}

// ── Modals ───────────────────────────────────────────────────────────────────

export function Modal({ title, subtitle, onClose, width = 640, children, footer, head, closeOnBackdrop = true, bodyStyle }) {
  useEffect(() => {
    if (!onClose) return undefined;
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);
  return html`<div class="backdrop" onMouseDown=${(e) => { if (closeOnBackdrop && e.target === e.currentTarget && onClose) onClose(); }}>
    <div class="modal" style=${{ width: `${width}px` }}>
      <div class="modal-head">
        <div class="t">${title}</div>
        ${subtitle ? html`<div class="s ellipsis">${subtitle}</div>` : null}
        <div class="spacer"></div>
        ${head || null}
        ${onClose ? html`<button class="xbtn" onClick=${onClose}>✕</button>` : null}
      </div>
      <div class="modal-body" style=${bodyStyle}>${children}</div>
      ${footer ? html`<div class="modal-foot">${footer}</div>` : null}
    </div>
  </div>`;
}

/** Promise-based confirmation. Resolves true when confirmed. */
export function confirmDialog({ title, message, confirmLabel = 'Continue', cancelLabel = 'Cancel', variant = 'primary', detail }) {
  return new Promise((resolve) => {
    const finish = (result) => {
      setState({ confirm: null });
      resolve(result);
    };
    setState({ confirm: { title, message, confirmLabel, cancelLabel, variant, detail, finish } });
  });
}

export function ConfirmHost() {
  const confirm = useStore((s) => s.confirm);
  if (!confirm) return null;
  return html`<${Modal} title=${confirm.title} onClose=${() => confirm.finish(false)} width=${480}
    footer=${html`<div class="spacer"></div>
      <${Btn} onClick=${() => confirm.finish(false)}>${confirm.cancelLabel}</${Btn}>
      <${Btn} variant=${confirm.variant} onClick=${() => confirm.finish(true)}>${confirm.confirmLabel}</${Btn}>`}>
    <div style=${{ font: '400 12px/1.7 var(--sans)', color: 'var(--tx2)', whiteSpace: 'pre-wrap' }}>${confirm.message}</div>
    ${confirm.detail ? html`<pre class="logbox" style=${{ marginTop: '12px', maxHeight: '220px' }}>${confirm.detail}</pre>` : null}
  </${Modal}>`;
}

// ── Virtualized list and grid ────────────────────────────────────────────────

function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return size;
}

export function VirtualList({ items, rowHeight = 32, renderRow, overscan = 8, class: cls = '', scrollToIndex = null, empty = null }) {
  const ref = useRef(null);
  const { height } = useElementSize(ref);
  const [scrollTop, setScrollTop] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el || scrollToIndex == null || scrollToIndex < 0) return;
    const top = scrollToIndex * rowHeight;
    if (top < el.scrollTop || top + rowHeight > el.scrollTop + el.clientHeight) {
      el.scrollTop = Math.max(0, top - el.clientHeight / 3);
    }
  }, [scrollToIndex]);
  const total = items.length;
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(total, Math.ceil((scrollTop + height) / rowHeight) + overscan);
  const rows = [];
  for (let i = start; i < end; i += 1) rows.push(renderRow(items[i], i));
  return html`<div class=${`scroll ${cls}`} ref=${ref} style=${{ flex: 1, position: 'relative' }}
    onScroll=${(e) => setScrollTop(e.currentTarget.scrollTop)}>
    ${total === 0 && empty ? empty : html`<div style=${{ height: `${total * rowHeight}px`, position: 'relative' }}>
      <div style=${{ position: 'absolute', top: `${start * rowHeight}px`, left: 0, right: 0 }}>${rows}</div>
    </div>`}
  </div>`;
}

export function VirtualGrid({ items, minWidth = 200, gap = 10, padding = 12, captionHeight = 40, renderCell, overscanRows = 2, onVisible, empty = null, scrollKey }) {
  const ref = useRef(null);
  const { width, height } = useElementSize(ref);
  const [scrollTop, setScrollTop] = useState(0);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = 0;
    setScrollTop(0);
  }, [scrollKey]);
  const inner = Math.max(0, width - padding * 2);
  const cols = Math.max(1, Math.floor((inner + gap) / (minWidth + gap)));
  const cellWidth = cols ? (inner - gap * (cols - 1)) / cols : minWidth;
  const imageHeight = Math.round(cellWidth * 0.66);
  const rowHeight = imageHeight + captionHeight + gap;
  const rowsTotal = Math.ceil(items.length / cols);
  const firstRow = Math.max(0, Math.floor((scrollTop - padding) / rowHeight) - overscanRows);
  const lastRow = Math.min(rowsTotal, Math.ceil((scrollTop + height) / rowHeight) + overscanRows);
  const startIndex = firstRow * cols;
  const endIndex = Math.min(items.length, lastRow * cols);
  const visible = items.slice(startIndex, endIndex);
  useEffect(() => {
    if (onVisible) onVisible(visible);
  }, [startIndex, endIndex, items]);
  return html`<div class="scroll" ref=${ref} style=${{ flex: 1, position: 'relative' }}
    onScroll=${(e) => setScrollTop(e.currentTarget.scrollTop)}>
    ${items.length === 0 && empty ? empty : html`<div style=${{ height: `${rowsTotal * rowHeight + padding * 2}px`, position: 'relative' }}>
      <div style=${{
        position: 'absolute', top: `${padding + firstRow * rowHeight}px`, left: `${padding}px`, right: `${padding}px`,
        display: 'grid', gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: `${gap}px`,
      }}>
        ${visible.map((item, i) => renderCell(item, startIndex + i, imageHeight))}
      </div>
    </div>`}
  </div>`;
}

// ── Misc ──────────────────────────────────────────────────────────────────────

export function useInterval(fn, ms, active = true) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => saved.current(), ms);
    return () => clearInterval(id);
  }, [ms, active]);
}

export function Img({ src, alt = '', style, class: cls = '', onError }) {
  const [failed, setFailed] = useState(false);
  // An <img> keeps painting its previous picture until a new src decodes;
  // dim it meanwhile so a slow load never passes as the new image.
  const [shownSrc, setShownSrc] = useState(null);
  useEffect(() => setFailed(false), [src]);
  if (!src || failed) return null;
  const stale = shownSrc !== null && shownSrc !== src;
  return html`<img src=${src} alt=${alt} class=${`${cls}${stale ? ' stale' : ''}`} style=${style} loading="lazy" decoding="async" draggable=${false}
    onLoad=${() => setShownSrc(src)}
    onError=${() => { setFailed(true); onError && onError(); }} />`;
}

export function currentState() {
  return getState();
}
