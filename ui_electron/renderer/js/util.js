// Formatting helpers.

export const fmtInt = (n) => (n == null ? '—' : Number(n).toLocaleString('en-US'));

export function fmtBytes(bytes) {
  if (bytes == null || Number.isNaN(bytes)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${Math.round(value)} B` : `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

export function fmtGB(bytes) {
  if (bytes == null) return '—';
  return (bytes / 1024 ** 3).toFixed(1);
}

export function fmtDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function fmtRate(rate) {
  if (!rate) return '—';
  if (rate >= 1) return `${rate.toFixed(2)} img/s`;
  return `${rate.toFixed(2)} img/s`;
}

export function basename(path) {
  if (!path) return '';
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1];
}

export function dirname(path) {
  if (!path) return '';
  const idx = Math.max(String(path).lastIndexOf('/'), String(path).lastIndexOf('\\'));
  return idx > 0 ? String(path).slice(0, idx) : '';
}

/** Path relative to root when inside it, otherwise the file name. */
export function relPath(path, root) {
  if (!path) return '';
  if (root) {
    const base = root.endsWith('/') || root.endsWith('\\') ? root : `${root}/`;
    if (path.startsWith(base)) return path.slice(base.length);
    const winBase = root.endsWith('\\') ? root : `${root}\\`;
    if (path.startsWith(winBase)) return path.slice(winBase.length);
  }
  return basename(path);
}

export function shortHash(hash) {
  if (!hash) return '—';
  return `${hash.slice(0, 6)}…${hash.slice(-4)}`;
}

export function fmtTime(value) {
  if (!value) return '—';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(String(value).replace(' ', 'T'));
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleTimeString('en-GB', { hour12: false });
}

export function fmtDateTime(value) {
  if (!value) return '—';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(String(value).replace(' ', 'T'));
  if (Number.isNaN(date.getTime())) return String(value);
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function pct(part, whole) {
  if (!whole) return 0;
  return Math.max(0, Math.min(100, (part / whole) * 100));
}

export function plural(n, word, pluralWord) {
  return `${fmtInt(n)} ${n === 1 ? word : pluralWord || `${word}s`}`;
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function debounce(fn, ms) {
  let timer = null;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(timer);
  return wrapped;
}

export const modelShort = (name) => (name ? String(name).split('/').pop() : '');
