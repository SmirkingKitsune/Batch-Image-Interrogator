// Gallery tab — 1e: tag filter left, contact sheet centre, inspector right.
import { html, useEffect, useLayoutEffect, useMemo, useRef, useState } from '../lib.js';
import { imageUrl, native, rpc, thumbUrl } from '../api.js';
import { getState, patch, useStore } from '../store.js';
import { openInspect, openInspectMulti, openModal } from '../state.js';
import { Btn, Img, reportError, Seg, Slider, TextField, toast, VirtualGrid } from '../ui.js';
import { basename, debounce, fmtBytes, fmtInt, modelShort, relPath } from '../util.js';

async function refreshList() {
  const g = getState().gallery;
  patch('gallery', { loading: true });
  try {
    const list = await rpc('gallery.list', { sort: g.sort, show: g.show, tags: g.tags, search: g.search });
    patch('gallery', { list, loading: false });
  } catch (err) {
    patch('gallery', { loading: false });
    reportError(err);
  }
}
const refreshListSoon = debounce(refreshList, 180);

// Metadata (dimensions, model count) for visible thumbnails, fetched in batches.
const pendingMeta = new Set();
const flushMeta = debounce(async () => {
  const paths = [...pendingMeta];
  pendingMeta.clear();
  if (!paths.length) return;
  try {
    const rows = await rpc('gallery.meta', { paths });
    patch('gallery', (g) => {
      const meta = { ...g.meta };
      for (const row of rows) meta[row.p] = row;
      return { meta };
    });
  } catch {
    /* thumbnails still show without metadata */
  }
}, 120);

function requestMeta(items) {
  const meta = getState().gallery.meta;
  let queued = false;
  for (const item of items) {
    if (!meta[item.p] && !pendingMeta.has(item.p)) {
      pendingMeta.add(item.p);
      queued = true;
    }
  }
  if (queued) flushMeta();
}

async function loadDetail(path) {
  try {
    const detail = await rpc('gallery.detail', { path });
    if (getState().gallery.selected === path) patch('gallery', { detail });
  } catch (err) {
    reportError(err);
  }
}

function select(path, event) {
  const g = getState().gallery;
  if (g.multi) {
    const items = g.list?.items || [];
    let next;
    if (event?.shiftKey && g.multiSel.length) {
      const last = items.findIndex((i) => i.p === g.multiSel[g.multiSel.length - 1]);
      const here = items.findIndex((i) => i.p === path);
      const [a, b] = last < here ? [last, here] : [here, last];
      next = [...new Set([...g.multiSel, ...items.slice(a, b + 1).map((i) => i.p)])];
    } else if (g.multiSel.includes(path)) {
      next = g.multiSel.filter((p) => p !== path);
    } else {
      next = [...g.multiSel, path];
    }
    patch('gallery', { multiSel: next, selected: next.length === 1 ? next[0] : null, detail: next.length === 1 ? g.detail : null });
    if (next.length === 1) loadDetail(next[0]);
    return;
  }
  patch('gallery', { selected: path, detail: g.selected === path ? g.detail : null });
  loadDetail(path);
}

export function GalleryPage() {
  const { dirVersion, galleryVersion, sort, show, tags, search, path } = useStore((s) => ({
    dirVersion: s.dir.version, galleryVersion: s.gallery.version, sort: s.gallery.sort, show: s.gallery.show,
    tags: s.gallery.tags, search: s.gallery.search, path: s.dir.path,
  }));
  useEffect(() => { if (path) refreshListSoon(); }, [dirVersion, galleryVersion, sort, show, tags.join('\u0001'), search, path]);
  useEffect(() => {
    // A changed sidecar can change the tag editor too.
    const selected = getState().gallery.selected;
    if (selected) loadDetail(selected);
  }, [galleryVersion]);

  if (!path) {
    return html`<div class="page"><div class="empty">
      <div class="big">No directory selected</div>
      <div>The gallery shows the directory chosen on the Interrogation tab.</div>
      <${Btn} variant="primary" onClick=${() => import('../state.js').then((m) => m.chooseDirectory())}>Select Directory…</${Btn}>
    </div></div>`;
  }
  return html`<div class="page"><div class="workspace">
    <${TagSidebar} />
    <${ContactSheet} />
    <${Inspector} />
  </div></div>`;
}

// ── Left: tag filter ──────────────────────────────────────────────────────────

function TagSidebar() {
  const { list, tags, sort, show } = useStore((s) => ({ list: s.gallery.list, tags: s.gallery.tags, sort: s.gallery.sort, show: s.gallery.show }));
  const selected = list?.selected_tags || tags.map((t) => [t, 0]);
  const toggle = (tag) => {
    const current = getState().gallery.tags;
    patch('gallery', { tags: current.includes(tag) ? current.filter((t) => t !== tag) : [...current, tag] });
  };
  return html`<div class="side" style=${{ width: '236px' }}>
    <div style=${{ padding: '12px 12px 10px', borderBottom: '1px solid var(--line)' }}>
      <div class="label" style=${{ marginBottom: '7px' }}>Search tags</div>
      <${TextField} placeholder="Type to search tags…" value=${getState().gallery.search}
        onInput=${(v) => { patch('gallery', { search: v }); }} />
    </div>
    <div class="label-row" style=${{ padding: '11px 12px 0', marginBottom: '8px' }}>
      <div class="label">Filter by tag</div>
      ${tags.length ? html`<button class="btn link" onClick=${() => patch('gallery', { tags: [] })}>Clear ${tags.length}</button>` : null}
    </div>
    <div class="scroll grow col" style=${{ padding: '0 12px', gap: '1px' }}>
      ${selected.map(([tag, count]) => html`<button class="ftag on" key=${`s-${tag}`} onClick=${() => toggle(tag)}>
        <span class="box"></span><span class="ellipsis">${tag}</span><span class="c">${fmtInt(count)}</span></button>`)}
      ${(list?.tags || []).map(([tag, count]) => html`<button class="ftag" key=${tag} onClick=${() => toggle(tag)}>
        <span class="box"></span><span class="ellipsis">${tag}</span><span class="c">${fmtInt(count)}</span></button>`)}
      ${list && list.hidden_tags ? html`<div class="note" style=${{ padding: '8px 6px 0', color: 'var(--tx7)' }}>+${fmtInt(list.hidden_tags)} more — search to narrow</div>` : null}
      ${list && !list.unique_tags ? html`<div class="help" style=${{ padding: '6px' }}>No tags yet — .txt sidecars are read from this directory.</div>` : null}
    </div>
    <div class="col gap10" style=${{ padding: '12px', borderTop: '1px solid var(--line)' }}>
      <div><div class="label" style=${{ marginBottom: '6px' }}>Sort</div>
        <${Seg} class="sans" value=${sort} onChange=${(v) => patch('gallery', { sort: v })} options=${[{ value: 'name', label: 'Name' }, { value: 'date', label: 'Date' }, { value: 'size', label: 'Size' }]} /></div>
      <div><div class="label" style=${{ marginBottom: '6px' }}>Show</div>
        <${Seg} class="sans" value=${show} onChange=${(v) => patch('gallery', { show: v })} options=${[{ value: 'all', label: 'All' }, { value: 'tagged', label: 'Tagged' }, { value: 'untagged', label: 'Untagged' }]} /></div>
    </div>
  </div>`;
}

// ── Centre: contact sheet ─────────────────────────────────────────────────────

function ContactSheet() {
  const { list, loading, selected, multi, multiSel, thumb, root, meta } = useStore((s) => ({
    list: s.gallery.list, loading: s.gallery.loading, selected: s.gallery.selected, multi: s.gallery.multi,
    multiSel: s.gallery.multiSel, thumb: s.settings.gallery_thumb || 200, root: s.dir.path, meta: s.gallery.meta,
  }));
  const [menu, setMenu] = useState(null);
  const [size, setSize] = useState(thumb);
  useEffect(() => setSize(thumb), [thumb]);
  const items = list?.items || [];
  const selSet = useMemo(() => new Set(multiSel), [multiSel]);
  const paths = useMemo(() => items.map((i) => i.p), [items]);

  useEffect(() => {
    if (!menu) return undefined;
    const close = () => setMenu(null);
    window.addEventListener('click', close);
    window.addEventListener('blur', close);
    return () => { window.removeEventListener('click', close); window.removeEventListener('blur', close); };
  }, [menu]);

  const renderCell = (item, index, imageHeight) => {
    const m = meta[item.p];
    const isSel = multi ? selSet.has(item.p) : selected === item.p;
    const sub = !item.t
      ? (m?.db_only ? `${fmtInt(m.models)} ${m.models === 1 ? 'model' : 'models'} · unwritten` : 'untagged')
      : `${fmtInt(item.n)} tags${m?.w ? ` · ${m.w}×${m.h}` : ''}`;
    return html`<div class=${`thumb ${isSel ? 'sel' : ''}`} key=${item.p}
      onClick=${(e) => select(item.p, e)}
      onDblClick=${() => openInspect(item.p, paths)}
      onContextMenu=${(e) => { e.preventDefault(); if (!isSel) select(item.p, e); setMenu({ x: e.clientX, y: e.clientY, path: item.p }); }}>
      <div class=${`img ${item.t ? '' : 'untagged'}`} style=${{ height: `${imageHeight}px` }}>
        <${Img} src=${thumbUrl(item.p, size <= 200 ? 256 : 512)} />
        <div class="badges">
          ${item.t ? html`<span class="badge txt">.txt</span>` : null}
          ${m?.models > 1 || (m?.models && item.t) ? html`<span class="badge models">${m.models} ${m.models === 1 ? 'model' : 'models'}</span>` : null}
          ${m?.db_only ? html`<span class="badge dbonly">db only</span>` : null}
        </div>
      </div>
      <div class="cap"><div class="n" title=${item.p}>${relPath(item.p, root)}</div><div class="s">${sub}</div></div>
    </div>`;
  };

  return html`<div class="grow col" style=${{ minWidth: 0 }}>
    <div class="row gap12" style=${{ height: '38px', padding: '0 14px', borderBottom: '1px solid var(--line)', background: 'var(--panel)', flex: 'none' }}>
      <div class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx3)' }}>${list ? `${fmtInt(items.length)} of ${fmtInt(list.total)} images` : loading ? 'loading…' : ''}</div>
      <div style=${{ width: '1px', height: '18px', background: 'var(--line)' }}></div>
      <div class="note" style=${{ fontSize: '10px' }}>thumb</div>
      <div style=${{ width: '80px' }}><${Slider} value=${size} min=${100} max=${400} step=${10} onChange=${setSize}
        onCommit=${(v) => { patch('settings', { gallery_thumb: v }); rpc('app.set_setting', { key: 'gallery_thumb', value: v }).catch(() => {}); }} /></div>
      <div class="mono" style=${{ fontSize: '10px', color: 'var(--tx3)', width: '38px' }}>${size}px</div>
      <div class="spacer"></div>
      <button class=${`pill ${multi ? 'on' : ''}`} style=${{ borderRadius: '4px' }} title="Ctrl/Shift-click to select several images"
        onClick=${() => patch('gallery', (g) => ({ multi: !g.multi, multiSel: g.multi ? [] : (g.selected ? [g.selected] : []) }))}>
        <span class="row gap6"><span style=${{ width: '10px', height: '10px', borderRadius: '2px', background: multi ? 'var(--blue)' : 'transparent', border: multi ? 0 : '1px solid var(--box)' }}></span>Multi-select${multi ? ` · ${multiSel.length}` : ''}</span>
      </button>
      <${Btn} size="sm" onClick=${() => openModal('organize', {})} style=${{ height: '26px' }}>Organize by Tags…</${Btn}>
    </div>
    <${VirtualGrid} items=${items} minWidth=${size} captionHeight=${44} renderCell=${renderCell} onVisible=${requestMeta}
      scrollKey=${`${list?.total}-${getState().gallery.sort}-${getState().gallery.show}-${getState().gallery.tags.join(',')}`}
      empty=${html`<div class="empty">${loading ? html`<span class="spinner"></span>` : 'No images match these filters.'}</div>`} />
    ${menu ? html`<div style=${{ position: 'fixed', left: `${menu.x}px`, top: `${menu.y}px`, zIndex: 60, background: '#1a1e24', border: '1px solid var(--line-strong)', borderRadius: '6px', padding: '4px', boxShadow: '0 10px 30px rgba(0,0,0,.5)', minWidth: '210px' }}>
      ${multi && multiSel.length > 1
        ? html`<${MenuItem} onClick=${() => openInspectMulti(multiSel)}>Edit selected tags (${multiSel.length} images)…</${MenuItem}>`
        : html`<${MenuItem} onClick=${() => openInspect(menu.path, paths)}>Advanced inspection…</${MenuItem}>`}
      ${native.available ? html`<${MenuItem} onClick=${() => native.showItem(menu.path)}>Show in folder</${MenuItem}>` : null}
    </div>` : null}
  </div>`;
}

function MenuItem({ onClick, children }) {
  return html`<button class="btn" style=${{ width: '100%', justifyContent: 'flex-start', border: 0, height: '28px', fontWeight: 400 }} onClick=${onClick}>${children}</button>`;
}

// ── Right: inspector ──────────────────────────────────────────────────────────

function ZoomPreview({ path, size, dims }) {
  const [zoom, setZoom] = useState(0); // 0 = fit
  const [box, setBox] = useState(null);
  const boxRef = useRef(null);
  useEffect(() => setZoom(0), [path]);
  // The box is only measurable after mount; track it so the fit % is right on
  // the first paint and after the pane resizes.
  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return undefined;
    const measure = () => setBox((b) => (b && b.w === el.clientWidth && b.h === el.clientHeight ? b : { w: el.clientWidth, h: el.clientHeight }));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const fitPct = dims?.w && box?.w ? Math.min(box.w / dims.w, box.h / dims.h, 1) : null;
  const label = zoom === 0 ? (fitPct ? `${Math.round(fitPct * 100)}%` : 'fit') : `${Math.round(zoom * 100)}%`;
  const step = (delta) => setZoom((z) => {
    const base = z === 0 ? (fitPct || 1) : z;
    return Math.max(0.25, Math.min(4, Math.round((base + delta) * 100) / 100));
  });
  return html`<div>
    <div class="preview" ref=${boxRef} style=${{ height: '210px', overflow: zoom ? 'auto' : 'hidden', alignItems: zoom ? 'flex-start' : 'center', justifyContent: zoom ? 'flex-start' : 'center' }}>
      ${zoom === 0
        ? html`<${Img} src=${thumbUrl(path, 800)} />`
        : html`<img src=${imageUrl(path)} draggable=${false} style=${{ maxWidth: 'none', maxHeight: 'none', width: dims?.w ? `${Math.round(dims.w * zoom)}px` : `${zoom * 100}%`, display: 'block' }} />`}
    </div>
    <div class="row gap6" style=${{ marginTop: '9px' }}>
      <button class="iconbtn" onClick=${() => step(-0.25)}>−</button>
      <div class="mono" style=${{ width: '44px', textAlign: 'center', fontSize: '10.5px', color: 'var(--tx3)' }}>${label}</div>
      <button class="iconbtn" onClick=${() => step(0.25)}>+</button>
      <button class="iconbtn" style=${{ width: 'auto', padding: '0 9px' }} onClick=${() => setZoom(0)}>Fit</button>
      <button class="iconbtn" style=${{ width: 'auto', padding: '0 9px' }} onClick=${() => setZoom(1)}>1:1</button>
      <div class="spacer"></div>
      <div class="note">${size != null ? fmtBytes(size) : ''}</div>
    </div>
  </div>`;
}

function Inspector() {
  const { selected, detail, multi, multiSel, filtersVersion } = useStore((s) => ({
    selected: s.gallery.selected, detail: s.gallery.detail, multi: s.gallery.multi, multiSel: s.gallery.multiSel, filtersVersion: s.filters,
  }));
  if (multi && multiSel.length > 1) {
    return html`<div class="col" style=${{ width: '372px', flex: 'none', background: 'var(--panel)', borderLeft: '1px solid var(--line)' }}>
      <div style=${{ padding: '12px 14px' }}>
        <div class="preview" style=${{ height: '210px' }}><${Img} src=${thumbUrl(multiSel[0], 800)} /></div>
      </div>
      <div class="block col gap10">
        <div style=${{ font: '500 13px/1.3 var(--sans)', color: 'var(--tx1)' }}>${fmtInt(multiSel.length)} images selected</div>
        <div class="help">Edit the tags these images share. Each image keeps its own unique tags.</div>
        <${Btn} variant="primary" onClick=${() => openInspectMulti(multiSel)}>Edit common tags…</${Btn}>
        <${Btn} onClick=${() => patch('gallery', { multiSel: [] })}>Clear selection</${Btn}>
      </div>
    </div>`;
  }
  if (!selected) {
    return html`<div class="col" style=${{ width: '372px', flex: 'none', background: 'var(--panel)', borderLeft: '1px solid var(--line)' }}>
      <div class="empty" style=${{ flex: 1 }}><div>Select an image to inspect its tags.</div><div class="dim">Double-click opens advanced inspection.</div></div>
    </div>`;
  }
  return html`<div class="col" style=${{ width: '372px', flex: 'none', background: 'var(--panel)', borderLeft: '1px solid var(--line)', minHeight: 0 }}>
    <div style=${{ padding: '12px 14px 10px' }}>
      <${ZoomPreview} path=${selected} size=${detail?.meta?.size} dims=${detail?.meta} />
    </div>
    ${detail ? html`<${Results} detail=${detail} />` : html`<div class="empty"><span class="spinner"></span></div>`}
    ${detail ? html`<${TagEditor} detail=${detail} key=${`${selected}-${(detail.file_tags || []).join(',')}`} />` : null}
  </div>`;
}

function Results({ detail }) {
  const [tab, setTab] = useState('all');
  const models = detail.interrogations || [];
  useEffect(() => setTab('all'), [detail.path]);
  const rows = useMemo(() => {
    const out = [];
    const seen = new Map();
    for (const m of models) {
      if (tab !== 'all' && m.model_name !== tab) continue;
      const scores = m.confidence_scores || {};
      for (const tag of m.tags || []) {
        const conf = scores[tag];
        const prev = seen.get(tag);
        if (prev && (prev.conf ?? -1) >= (conf ?? -1)) continue;
        const row = { tag, model: m.model_type === 'LlamaCpp' ? 'llama' : m.model_type, conf };
        seen.set(tag, row);
      }
    }
    for (const row of seen.values()) out.push(row);
    out.sort((a, b) => (b.conf ?? -1) - (a.conf ?? -1));
    return out;
  }, [models, tab]);
  return html`<div style=${{ padding: '2px 14px 10px', borderBottom: '1px solid var(--line)' }}>
    <div class="row gap4" style=${{ marginBottom: '9px', flexWrap: 'wrap' }}>
      <button class=${`task ${tab === 'all' ? 'on' : ''}`} onClick=${() => setTab('all')}>All models</button>
      ${models.map((m) => html`<button class=${`task ${tab === m.model_name ? 'on' : ''}`} title=${m.model_name} onClick=${() => setTab(m.model_name)}>
        ${m.model_type === 'LlamaCpp' ? 'llama' : `${m.model_type} ${modelShort(m.model_name).replace(/^wd-|-tagger|tagger-/g, '').slice(0, 14)}`.trim()}</button>`)}
    </div>
    <div class="scroll" style=${{ maxHeight: '150px' }}>
      ${rows.length ? rows.slice(0, 200).map((r) => html`<div class="row gap9" style=${{ gap: '9px', padding: '4px 0', borderBottom: '1px solid var(--line-soft)' }} key=${r.tag}>
        <div class="grow ellipsis mono" style=${{ fontSize: '11px', color: 'var(--tx2)' }}>${r.tag}</div>
        <div class="mono" style=${{ fontSize: '10px', color: 'var(--tx7)' }}>${r.model}</div>
        <div class="mono" style=${{ width: '30px', textAlign: 'right', fontSize: '10px', color: r.conf != null ? 'var(--tx3)' : 'var(--tx7)' }}>${r.conf != null ? Number(r.conf).toFixed(2) : '—'}</div>
      </div>`) : html`<div class="help">No interrogation results stored for this image yet.</div>`}
    </div>
  </div>`;
}

function TagEditor({ detail }) {
  const [checked, setChecked] = useState(() => new Set(detail.editor.selected));
  const [extra, setExtra] = useState([]);
  const [adding, setAdding] = useState('');
  const [saving, setSaving] = useState(false);
  const all = [...detail.editor.all, ...extra.filter((t) => !detail.editor.all.includes(t))];
  const toggle = (tag) => setChecked((prev) => {
    const next = new Set(prev);
    if (next.has(tag)) next.delete(tag); else next.add(tag);
    return next;
  });
  const add = () => {
    const tag = adding.trim();
    if (!tag) return;
    if (!all.includes(tag)) setExtra((e) => [...e, tag]);
    setChecked((prev) => new Set([...prev, tag]));
    setAdding('');
  };
  const save = async () => {
    // Keep the sidecar's own order for tags already there; append new ones.
    const current = detail.file_tags.filter((t) => checked.has(t));
    const added = all.filter((t) => checked.has(t) && !detail.file_tags.includes(t));
    setSaving(true);
    try {
      await rpc('gallery.save_tags', { path: detail.path, tags: [...current, ...added] });
      toast(`Saved ${fmtInt(current.length + added.length)} tags to ${basename(detail.path).replace(/\.[^.]+$/, '')}.txt`, 'ok', 3000);
    } catch (err) {
      reportError(err);
    } finally {
      setSaving(false);
    }
  };
  const count = all.filter((t) => checked.has(t)).length;
  return html`<div class="grow col" style=${{ padding: '11px 14px', minHeight: 0 }}>
    <div class="label-row"><div class="label">Tag editor</div><div class="note" style=${{ color: 'var(--tx5)' }}>total ${fmtInt(all.length)} · selected ${fmtInt(count)}</div></div>
    <div class="scroll grow" style=${{ display: 'flex', flexWrap: 'wrap', gap: '5px', alignContent: 'flex-start' }}>
      ${all.map((tag) => html`<span class=${`tagchip toggle ${checked.has(tag) ? 'on' : 'off'}`} key=${tag} onClick=${() => toggle(tag)}>
        <span class="cb"></span>${tag}</span>`)}
      ${!all.length ? html`<div class="help">No tags yet. Add one below or interrogate this image.</div>` : null}
    </div>
    <div class="row gap6" style=${{ marginTop: '8px' }}>
      <div class="grow"><${TextField} placeholder="Add a tag…" value=${adding} onInput=${setAdding} onEnter=${add} /></div>
      <${Btn} size="sm" style=${{ height: '28px' }} onClick=${add} disabled=${!adding.trim()}>Add</${Btn}>
    </div>
    <div class="row gap8" style=${{ marginTop: '10px' }}>
      <${Btn} class="grow" size="tall" variant="success" busy=${saving} disabled=${saving} onClick=${save}>Save ${fmtInt(count)} tags to .txt</${Btn}>
      <${Btn} size="tall" onClick=${() => setChecked(new Set(all))}>All</${Btn}>
      <${Btn} size="tall" onClick=${() => setChecked(new Set())}>None</${Btn}>
    </div>
    <div class="note" style=${{ marginTop: '8px', color: 'var(--tx7)' }}>saves bypass filter rules — what is checked is what lands on disk</div>
  </div>`;
}
