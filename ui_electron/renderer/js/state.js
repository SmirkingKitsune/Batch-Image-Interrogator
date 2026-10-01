// Initial state, backend event wiring and shared actions.
import { on, onConnection, rpc } from './api.js';
import { getState, initState, patch, setState } from './store.js';
import { reportError, toast } from './ui.js';
import { basename, fmtInt, plural } from './util.js';

// Per-image queue state lives outside the store: thousands of rows update
// several times a second, so copying a big object per event would be waste.
// The store only carries a version counter that bumps when this changes.
export const itemStates = new Map();
const pathIndex = new Map();

export function initialState() {
  return {
    connected: false,
    loaded: false,
    tab: 'interrogation',
    device: {},
    hw: {},
    settings: {},
    activity: { interrogating: false, inquiring: false },
    dir: { path: '', recursive: false, scanning: false, paths: [], hasTxt: [], dirCount: 0, progress: null, version: 0 },
    filters: { prefix: [], remove: [], replace: {}, keep: [], underscores: false, stats: {} },
    interrog: {
      options: null,
      model: null,
      loading: false,
      loadingType: null,
      loadError: null,
      running: false,
      cancelling: false,
      progress: null,
      tags: null,
      itemsVersion: 0,
      panel: 'dir',
      queueFilter: 'all',
      selected: null,
      cfgType: 'WD',
      cfg: null,
      txtMode: 'merge',
      lastRun: null,
    },
    inquiry: {
      loaded: false,
      config: {},
      options: {},
      tasks: ['describe', 'ocr', 'vqa', 'custom', 'audit'],
      model: null,
      loading: false,
      loadError: null,
      panel: 'gguf',
      mode: 'single',
      single: { path: null, info: null, turns: [], pending: null, stream: '', thinking: '', thinkingStartedAt: null, thinkingEndedAt: null, thinkingSeconds: null, raw: '', showRaw: false, running: false, error: null },
      batch: { progress: null, cards: [], running: false, cancelling: false, showRaw: false, raw: '', lastRun: null },
      sources: null,
      sourcesScanning: false,
      metadata: null,
    },
    runtime: { summary: null, update: null, health: null, checkingUpdate: false, checkingHealth: false, provision: { running: false, step: null, logs: [], result: null } },
    gallery: { sort: 'name', show: 'all', tags: [], search: '', list: null, loading: false, selected: null, multi: false, multiSel: [], detail: null, meta: {}, version: 0 },
    settingsPage: 'hardware',
    settingsData: { db: null, queue: null, providers: null, cache: null },
    modals: { inspect: null, organize: null, firstRun: false, provision: null },
    busy: [],
    confirm: null,
    toasts: [],
  };
}

export function bumpItems() {
  patch('interrog', (s) => ({ itemsVersion: s.itemsVersion + 1 }));
}

function rebuildPathIndex(paths) {
  pathIndex.clear();
  paths.forEach((p, i) => pathIndex.set(p, i));
}

function setHasTxt(path, hasTxt) {
  const index = pathIndex.get(path);
  if (index === undefined) return;
  patch('dir', (d) => {
    if (d.hasTxt[index] === hasTxt) return null;
    const next = d.hasTxt.slice();
    next[index] = hasTxt;
    return { hasTxt: next, version: d.version + 1 };
  });
}

function applyDirectory(payload) {
  rebuildPathIndex(payload.paths || []);
  itemStates.clear();
  setState((s) => ({
    dir: {
      path: payload.path || '',
      recursive: Boolean(payload.recursive),
      scanning: Boolean(payload.scanning),
      paths: payload.paths || [],
      hasTxt: payload.has_txt || [],
      dirCount: payload.dir_count || 0,
      progress: null,
      version: (s.dir?.version || 0) + 1,
    },
    interrog: { ...s.interrog, itemsVersion: s.interrog.itemsVersion + 1, selected: null },
    gallery: { ...s.gallery, selected: null, multiSel: [], detail: null, version: s.gallery.version + 1 },
  }));
}

/** Load everything the window needs. Called on start and on reconnect. */
export async function loadAppState() {
  const snapshot = await rpc('app.state');
  const interrogation = snapshot.interrogation || {};
  setState((s) => ({
    loaded: true,
    device: snapshot.device || {},
    hw: snapshot.hardware || {},
    settings: snapshot.settings || {},
    filters: snapshot.filters || s.filters,
    tab: s.loaded ? s.tab : (snapshot.settings?.active_tab || 'interrogation'),
    interrog: {
      ...s.interrog,
      model: interrogation.model || null,
      loading: Boolean(interrogation.loading),
      running: Boolean(interrogation.running),
      progress: interrogation.progress || s.interrog.progress,
      tags: interrogation.tags || s.interrog.tags,
      txtMode: interrogation.txt_mode || s.interrog.txtMode,
    },
    inquiry: {
      ...s.inquiry,
      model: snapshot.inquiry?.model || null,
      loading: Boolean(snapshot.inquiry?.loading),
    },
    runtime: { ...s.runtime, summary: snapshot.runtime || null },
    modals: {
      ...s.modals,
      firstRun: s.loaded ? s.modals.firstRun : !snapshot.settings?.first_run_done,
    },
  }));
  if (snapshot.directory) applyDirectory(snapshot.directory);
  if (snapshot.queue?.pending_count > 0 && !getState().queuePrompted) {
    setState({ queuePrompted: true });
    promptQueue(snapshot.queue);
  }
  return snapshot;
}

async function promptQueue(queue) {
  const { confirmDialog } = await import('./ui.js');
  const process = await confirmDialog({
    title: 'Queued database operations',
    message: `${plural(queue.pending_count, 'database write')} could not complete last time because the database was busy. Process them now?`,
    confirmLabel: 'Process now',
    cancelLabel: 'Later',
  });
  if (process) {
    try {
      await rpc('db.process_queue');
      toast('Processing queued operations…');
    } catch (err) {
      reportError(err);
    }
  }
}

export function setTab(tab) {
  setState({ tab });
  rpc('app.set_setting', { key: 'active_tab', value: tab }).catch(() => {});
}

export async function chooseDirectory() {
  const { native } = await import('./api.js');
  const s = getState();
  const picked = await native.pickDirectory({ title: 'Select Image Directory', defaultPath: s.dir.path || undefined });
  if (!picked) return;
  await openDirectory(picked, s.dir.recursive);
}

export async function openDirectory(path, recursive) {
  try {
    await rpc('dir.open', { path, recursive: Boolean(recursive) });
  } catch (err) {
    reportError(err);
  }
}

export async function setRecursive(recursive) {
  try {
    await rpc('dir.set_recursive', { recursive: Boolean(recursive) });
    patch('dir', { recursive: Boolean(recursive) });
  } catch (err) {
    reportError(err);
  }
}

export function openInspect(path, list) {
  const s = getState();
  const images = list && list.length ? list : s.dir.paths;
  setState({ modals: { ...s.modals, inspect: { path, list: images, multi: null } } });
}

export function openInspectMulti(paths) {
  const s = getState();
  setState({ modals: { ...s.modals, inspect: { path: null, list: [], multi: paths } } });
}

export function openModal(name, value = true) {
  setState((s) => ({ modals: { ...s.modals, [name]: value } }));
}

export function closeModal(name) {
  setState((s) => ({ modals: { ...s.modals, [name]: name === 'firstRun' ? false : null } }));
}

// ── Event wiring ──────────────────────────────────────────────────────────────

export function wireEvents() {
  onConnection((connected) => {
    setState({ connected });
    if (connected && getState().loaded) {
      loadAppState().catch(() => {});
    }
  });

  on('hardware', (sample) => setState((s) => ({ hw: { ...s.hw, ...sample } })));
  on('toast', (t) => toast(t.message, t.level || 'info'));
  on('activity', (a) => setState((s) => ({ activity: { ...s.activity, ...a } })));

  // Directory
  on('dir.scanning', (d) => {
    itemStates.clear();
    patch('dir', { path: d.path, recursive: d.recursive, scanning: true, progress: { count: 0, current: '' }, paths: [], hasTxt: [] });
  });
  on('dir.progress', (p) => patch('dir', { progress: p }));
  on('dir.loaded', (d) => applyDirectory(d));
  on('dir.error', (e) => {
    patch('dir', { scanning: false });
    toast(e.message, 'error');
  });

  // Interrogation
  on('interrogate.model', (m) => {
    if (m.state === 'loading') patch('interrog', { loading: true, loadingType: m.type, loadError: null });
    else if (m.state === 'loaded') {
      patch('interrog', { loading: false, loadingType: null, model: m, loadError: null });
      toast(`Model loaded: ${m.short || m.label}`, 'ok', 3000);
    } else if (m.state === 'unloaded') {
      patch('interrog', { loading: false, model: null });
      if (m.reason === 'auto') toast('Batch complete — model auto-unloaded', 'info', 4000);
    } else if (m.state === 'error') {
      patch('interrog', { loading: false, loadingType: null, model: null, loadError: m.error });
      toast(`Failed to load model: ${m.error}`, 'error', 10000);
    }
  });
  on('interrogate.clip_models', (models) => patch('interrog', (s) => ({ options: s.options ? { ...s.options, clip_models: models } : s.options })));
  on('interrogate.started', (e) => {
    for (const path of e.paths || []) itemStates.set(path, { state: 'queued' });
    patch('interrog', (s) => ({
      running: true, cancelling: false, lastRun: null, tags: { unique: 0, rows: [] },
      progress: { total: e.total, done: 0, cached: 0, failed: 0, processed: 0, running: null, rate: 0, eta: null, paused: false, active: true },
      itemsVersion: s.itemsVersion + 1,
    }));
  });
  on('interrogate.item', (item) => {
    if (!item.path) return;
    const prev = itemStates.get(item.path) || {};
    itemStates.set(item.path, { ...prev, ...item });
    bumpItems();
  });
  on('interrogate.progress', (p) => patch('interrog', { progress: p }));
  on('interrogate.tags', (t) => patch('interrog', { tags: t }));
  on('interrogate.cancelling', () => patch('interrog', { cancelling: true }));
  on('interrogate.finished', (f) => {
    for (const [path, value] of itemStates) {
      if (value.state === 'running') itemStates.set(path, { ...value, state: 'queued' });
    }
    patch('interrog', (s) => ({ running: false, cancelling: false, lastRun: f, progress: s.progress ? { ...s.progress, active: false, paused: false, running: null } : null, itemsVersion: s.itemsVersion + 1 }));
    const parts = [`${fmtInt(f.done)} interrogated`, `${fmtInt(f.cached)} cached`];
    if (f.failed) parts.push(`${fmtInt(f.failed)} failed`);
    toast(`${f.cancelled ? 'Batch cancelled' : 'Batch complete'} — ${parts.join(' · ')}`, f.failed ? 'warn' : 'ok', 6000);
  });
  on('image_result_ready', (e) => {
    setHasTxt(e.path, Boolean(e.has_txt));
    patch('gallery', (g) => ({ version: g.version + 1 }));
  });
  on('gallery.tags_saved', (e) => {
    setHasTxt(e.path, Boolean(e.has_txt));
    patch('gallery', (g) => ({ version: g.version + 1 }));
  });
  on('filters.changed', (f) => setState({ filters: f }));

  // Inquiry
  on('inquiry.model', (m) => {
    if (m.state === 'loading') patch('inquiry', { loading: true, loadError: null });
    else if (m.state === 'loaded') {
      patch('inquiry', { loading: false, model: m, loadError: null });
      toast(`Inquiry model loaded: ${m.file}`, 'ok', 3000);
    } else if (m.state === 'unloaded') patch('inquiry', { loading: false, model: null });
    else if (m.state === 'error') {
      patch('inquiry', { loading: false, model: null, loadError: { message: m.error, logs: m.logs } });
    }
  });
  on('inquiry.single.started', (e) => patch('inquiry', (q) => ({
    single: { ...q.single, running: true, pending: e.turn, stream: '', thinking: '', thinkingStartedAt: null, thinkingEndedAt: null, thinkingSeconds: null, notice: null, error: null, startedAt: Date.now() },
  })));
  on('inquiry.single.stream', (e) => patch('inquiry', (q) => ({
    single: { ...q.single, stream: e.text, ...thinkingEnds(q.single, e.text) },
  })));
  on('inquiry.single.reasoning', (e) => patch('inquiry', (q) => ({
    single: { ...q.single, ...appendThinking(q.single, e) },
  })));
  on('inquiry.single.done', (e) => patch('inquiry', (q) => {
    const sameImage = q.single.path === e.path;
    return {
      single: {
        ...q.single,
        running: false,
        pending: null,
        stream: '',
        thinking: '',
        thinkingStartedAt: null,
        thinkingEndedAt: null,
        raw: e.raw || q.single.raw,
        turns: sameImage ? [...q.single.turns, e.turn] : q.single.turns,
        refresh: (q.single.refresh || 0) + 1,
      },
    };
  }));
  on('inquiry.single.done', (e) => {
    if (e.removed && e.removed.length) toast(`Audit removed ${plural(e.removed.length, 'sidecar tag')} from ${basename(e.path)}`, 'ok');
    else if (e.warnings && e.warnings.length) toast(`Inquiry completed with warnings: ${e.warnings.slice(0, 2).join(', ')}`, 'warn');
  });
  on('inquiry.single.error', (e) => patch('inquiry', (q) => ({
    single: {
      ...q.single, running: false, pending: null, stream: '', thinking: '', thinkingStartedAt: null, thinkingEndedAt: null,
      error: { message: e.error, logs: e.logs, turn: q.single.pending, thinking: e.thinking || '', thinking_seconds: e.thinking_seconds },
    },
  })));
  on('inquiry.sources', (e) => patch('inquiry', { sources: e.sources, sourcesScanning: Boolean(e.scanning) }));
  on('inquiry.batch.started', (e) => patch('inquiry', (q) => ({
    batch: { ...q.batch, running: true, cancelling: false, cards: [], lastRun: null, progress: { total: e.total, done: 0, cached: 0, failed: 0, processed: 0, rejected: 0, task: e.task, tags: [], active: true } },
  })));
  on('inquiry.batch.progress', (p) => patch('inquiry', (q) => ({ batch: { ...q.batch, progress: p } })));
  on('inquiry.batch.turn', (e) => patch('inquiry', (q) => ({
    batch: { ...q.batch, cards: [...q.batch.cards.filter((c) => c.path !== e.path), { path: e.path, state: 'live', turn: e.turn, stream: '', thinking: '', started: Date.now() }].slice(-300) },
  })));
  on('inquiry.batch.stream', (e) => patch('inquiry', (q) => ({
    batch: { ...q.batch, cards: q.batch.cards.map((c) => (c.path === e.path && c.state === 'live' ? { ...c, stream: e.text, ...thinkingEnds(c, e.text) } : c)) },
  })));
  on('inquiry.batch.reasoning', (e) => patch('inquiry', (q) => ({
    batch: { ...q.batch, cards: q.batch.cards.map((c) => (c.path === e.path && c.state === 'live' ? { ...c, ...appendThinking(c, e) } : c)) },
  })));
  on('inquiry.batch.result', (e) => patch('inquiry', (q) => ({
    batch: {
      ...q.batch,
      raw: e.raw || q.batch.raw,
      progress: { ...(q.batch.progress || {}), ...pickProgress(e) },
      cards: [...q.batch.cards.filter((c) => c.path !== e.path), { path: e.path, state: 'done', turn: e.turn }].slice(-300),
    },
  })));
  on('inquiry.batch.error', (e) => patch('inquiry', (q) => ({
    batch: {
      ...q.batch,
      progress: { ...(q.batch.progress || {}), ...pickProgress(e) },
      cards: [...q.batch.cards.filter((c) => c.path !== e.path), { path: e.path, state: 'error', error: e.error }].slice(-300),
    },
  })));
  on('inquiry.batch.cancelling', () => patch('inquiry', (q) => ({ batch: { ...q.batch, cancelling: true } })));
  on('inquiry.batch.finished', (e) => {
    patch('inquiry', (q) => ({
      batch: {
        ...q.batch,
        running: false,
        cancelling: false,
        lastRun: e,
        progress: { ...(q.batch.progress || {}), ...pickProgress(e), active: false },
        cards: q.batch.cards.map((c) => (c.state === 'live' ? { ...c, state: 'error', error: 'Batch inquiry ended before this image returned a response.' } : c)),
      },
    }));
    const summary = e.cancelled
      ? `Batch inquiry cancelled after ${fmtInt(e.processed)} of ${fmtInt(e.total)} images. Partial results were kept.`
      : `Batch inquiry complete — ${fmtInt(e.done + e.cached)} done${e.failed ? ` · ${fmtInt(e.failed)} failed` : ''}`;
    toast(summary, e.failed ? 'warn' : 'ok', 7000);
  });

  // Runtime
  on('runtime.summary', (summary) => patch('runtime', { summary }));
  on('runtime.update', (update) => patch('runtime', { update, checkingUpdate: false }));
  on('runtime.health', (health) => patch('runtime', { health, checkingHealth: false }));
  on('runtime.provision.started', () => patch('runtime', { provision: { running: true, step: null, logs: [], result: null } }));
  on('runtime.provision.progress', (p) => patch('runtime', (r) => ({ provision: { ...r.provision, step: p } })));
  on('runtime.provision.log', (l) => patch('runtime', (r) => ({ provision: { ...r.provision, logs: [...r.provision.logs, l].slice(-1500) } })));
  on('runtime.provision.finished', (result) => {
    patch('runtime', (r) => ({ provision: { ...r.provision, running: false, result } }));
    if (result.binary) toast(result.mismatch ? 'Installed a fallback runtime — see the warning.' : 'llama.cpp runtime installed.', result.mismatch ? 'warn' : 'ok', 8000);
    else if (result.error) toast(result.error, 'error', 10000);
  });

  // Organize
  on('organize.progress', (p) => setState((s) => ({ modals: s.modals.organize ? { ...s.modals, organize: { ...s.modals.organize, progress: p } } : s.modals })));
  on('organize.finished', (f) => {
    setState((s) => ({ modals: s.modals.organize ? { ...s.modals, organize: { ...s.modals.organize, progress: null, finished: f } } : s.modals }));
    toast(`Moved ${plural(f.moved, 'image')}${f.errors.length ? ` · ${f.errors.length} errors` : ''}`, f.errors.length ? 'warn' : 'ok', 7000);
  });

  // Database busy (request/reply)
  on('database_busy', (request) => setState((s) => ({ busy: [...s.busy, request] })));
  on('db.queue_processed', (q) => {
    setState((s) => ({ settingsData: { ...s.settingsData, queue: q } }));
    toast(`Queue processed: ${fmtInt(q.success)} succeeded${q.failed > 0 ? `, ${fmtInt(q.failed)} failed` : ''}`, q.failed > 0 ? 'warn' : 'ok');
  });
}

// Thinking arrives as appended fragments with their offset; slicing to the
// offset first makes a repeated fragment harmless. The backend closes it with
// `done` and the measured duration when the first answer token arrives, or
// voids it with `reset` when an attempt that looped starts over.
function appendThinking(holder, e) {
  if (e.reset) return { thinking: '', thinkingStartedAt: null, thinkingEndedAt: null, thinkingSeconds: null, notice: e.notice || null };
  if (e.done) return { thinkingEndedAt: holder.thinkingEndedAt || Date.now(), thinkingSeconds: e.seconds };
  const thinking = (holder.thinking || '').slice(0, e.offset) + (e.text || '');
  return { thinking, thinkingStartedAt: holder.thinkingStartedAt || Date.now() };
}

// Fallback end marker: readable answer text has started to stream.
function thinkingEnds(holder, text) {
  return text && holder.thinkingStartedAt && !holder.thinkingEndedAt ? { thinkingEndedAt: Date.now() } : null;
}

function pickProgress(e) {
  const keys = ['total', 'done', 'cached', 'failed', 'processed', 'rejected', 'task', 'rate', 'eta', 'active', 'tags', 'running'];
  const out = {};
  for (const k of keys) if (k in e) out[k] = e[k];
  return out;
}

export function bootstrapState() {
  initState(initialState());
}
