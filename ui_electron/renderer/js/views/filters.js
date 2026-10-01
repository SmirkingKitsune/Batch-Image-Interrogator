// Tag filter editor, shared by the FLT rail sheets and Settings → Tag filters.
import { html, useState } from '../lib.js';
import { rpc } from '../api.js';
import { useStore } from '../store.js';
import { Btn, Check, confirmDialog, reportError, Seg, TextField } from '../ui.js';

const KINDS = [
  { value: 'prefix', label: 'Prefix', help: 'Prepended to every generated .txt file — trigger words or consistent tags.' },
  { value: 'remove', label: 'Remove', help: 'Excluded from .txt output. Useful for unwanted tags from skewed models.' },
  { value: 'replace', label: 'Replace', help: "Rewritten before writing, e.g. 'girl' → 'female'." },
  { value: 'keep', label: 'Keep', help: 'Always written, whatever the confidence threshold.' },
];

export function filterSummary(stats) {
  if (!stats) return '—';
  const parts = [];
  if (stats.prefix_count) parts.push(`${stats.prefix_count} prefix`);
  if (stats.remove_count) parts.push(`${stats.remove_count} remove`);
  if (stats.replace_count) parts.push(`${stats.replace_count} replace`);
  if (stats.keep_count) parts.push(`${stats.keep_count} keep`);
  return parts.length ? parts.join(' · ') : 'none';
}

export function FiltersEditor({ compact = false }) {
  const filters = useStore((s) => s.filters);
  const [kind, setKind] = useState('remove');
  const [tag, setTag] = useState('');
  const [replacement, setReplacement] = useState('');
  const [inputKey, setInputKey] = useState(0);
  const info = KINDS.find((k) => k.value === kind);

  const entries = kind === 'replace'
    ? Object.entries(filters.replace || {}).map(([from, to]) => ({ key: from, label: html`${from} <span class="dim">→</span> <span style=${{ color: 'var(--orange-tx)' }}>${to}</span>` }))
    : (filters[kind] || []).map((t) => ({ key: t, label: t }));

  const add = async () => {
    const value = tag.trim();
    if (!value) return;
    try {
      await rpc('filters.add', { kind, tag: value, replacement: replacement.trim() });
      setTag('');
      setReplacement('');
      setInputKey((k) => k + 1);
    } catch (err) {
      reportError(err);
    }
  };
  const remove = (key) => rpc('filters.remove', { kind, tag: key }).catch(reportError);
  const clear = async () => {
    if (!entries.length) return;
    const ok = await confirmDialog({ title: `Clear ${info.label.toLowerCase()} list`, message: `Remove all ${entries.length} entries from the ${info.label.toLowerCase()} list?`, confirmLabel: 'Clear', variant: 'danger' });
    if (ok) rpc('filters.clear', { kind }).catch(reportError);
  };

  return html`<div class="col gap10">
    <${Seg} class="sans" value=${kind} onChange=${(k) => { setKind(k); setTag(''); setReplacement(''); }}
      options=${KINDS.map((k) => ({ value: k.value, label: `${k.label} ${(k.value === 'replace' ? Object.keys(filters.replace || {}).length : (filters[k.value] || []).length) || ''}`.trim() }))} />
    <div class="help">${info.help}</div>
    <div class="row gap6" key=${inputKey}>
      <div class="grow"><${TextField} placeholder=${kind === 'replace' ? 'Original…' : `Tag to ${kind === 'prefix' ? 'prepend' : kind}…`} value=${tag} onInput=${setTag} onEnter=${kind === 'replace' ? null : add} /></div>
      ${kind === 'replace' ? html`<div class="grow"><${TextField} placeholder="Replacement…" value=${replacement} onInput=${setReplacement} onEnter=${add} /></div>` : null}
      <${Btn} onClick=${add} disabled=${!tag.trim() || (kind === 'replace' && !replacement.trim())}>Add</${Btn}>
    </div>
    <div class="listbox" style=${{ maxHeight: compact ? '240px' : '320px', minHeight: '60px' }}>
      ${entries.length ? entries.map((e) => html`<div class="li" key=${e.key}>
        <span class="grow ellipsis">${e.label}</span>
        <button class="rm" title="Remove" onClick=${() => remove(e.key)}>✕</button>
      </div>`) : html`<div class="li off">No ${info.label.toLowerCase()} rules.</div>`}
    </div>
    <div class="row gap8">
      <${Check} checked=${Boolean(filters.underscores)} onChange=${(v) => rpc('filters.set_underscores', { enabled: v }).catch(reportError)}
        title="Replace underscores with spaces in written tags (long_hair → long hair). Emoji tags like ^_^ are kept.">Replace underscores with spaces</${Check}>
      <div class="spacer"></div>
      <${Btn} size="sm" onClick=${clear} disabled=${!entries.length}>Clear list</${Btn}>
    </div>
    <div class="note">Active: ${filterSummary(filters.stats)}${filters.underscores ? ' · underscores → spaces' : ''}. Filters apply to .txt output only; the database keeps raw results.</div>
  </div>`;
}
