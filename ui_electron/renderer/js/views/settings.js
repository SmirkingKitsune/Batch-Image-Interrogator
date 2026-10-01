// Database / Settings — 1f's settings index with a live hardware block, and
// 2c's llama.cpp runtime page.
import { html, useEffect, useState } from '../lib.js';
import { native, rpc } from '../api.js';
import { patch, setState, useStore } from '../store.js';
import { openModal } from '../state.js';
import { Btn, Check, confirmDialog, Dot, Radio, reportError, toast } from '../ui.js';
import { fmtBytes, fmtGB, fmtInt, fmtTime, plural } from '../util.js';
import { FiltersEditor } from './filters.js';
import { checkHealth, openLogFolder, RuntimeCard, updateLevelColor } from './runtime.js';

const PAGES = [
  ['hardware', 'Hardware & providers'],
  ['llama', 'llama.cpp runtime'],
  ['database', 'Database'],
  ['filters', 'Tag filters'],
  ['cache', 'Model cache'],
  ['queue', 'Operation queue'],
  ['application', 'Application'],
];

async function loadSettingsData(keys = ['db', 'queue', 'providers', 'cache']) {
  const calls = { db: 'db.stats', queue: 'db.queue', providers: 'providers.state', cache: 'cache.models' };
  await Promise.all(keys.map(async (key) => {
    try {
      const value = await rpc(calls[key]);
      setState((s) => ({ settingsData: { ...s.settingsData, [key]: value } }));
    } catch (err) {
      reportError(err);
    }
  }));
}

export function SettingsPage() {
  const { page, runtime } = useStore((s) => ({ page: s.settingsPage, runtime: s.runtime.summary }));
  useEffect(() => {
    loadSettingsData();
    rpc('runtime.summary').then((summary) => patch('runtime', { summary })).catch(() => {});
  }, []);
  return html`<div class="page"><div class="workspace">
    <div class="side" style=${{ width: '196px' }}>
      <div class="navlist">
        ${PAGES.map(([id, label]) => html`<button class=${page === id ? 'on' : ''} onClick=${() => setState({ settingsPage: id })}>
          ${label}
          ${id === 'llama' && runtime ? html`<${Dot} kind=${!runtime.installed ? 'err' : runtime.fallback_from || !runtime.is_gpu ? 'warn' : 'ok'} />` : null}
        </button>`)}
      </div>
    </div>
    <div class="scroll grow" style=${{ padding: '18px 20px' }}>
      ${page === 'hardware' ? html`<${HardwarePage} />` : null}
      ${page === 'llama' ? html`<${LlamaPage} />` : null}
      ${page === 'database' ? html`<div class="col gap12" style=${{ maxWidth: '980px' }}><${DatabaseCard} /><${QueueCard} /></div>` : null}
      ${page === 'filters' ? html`<div class="card" style=${{ maxWidth: '720px' }}><div class="label" style=${{ marginBottom: '12px' }}>Tag filters</div><${FiltersEditor} /></div>` : null}
      ${page === 'cache' ? html`<${CachePage} />` : null}
      ${page === 'queue' ? html`<div style=${{ maxWidth: '720px' }}><${QueueCard} detailed /></div>` : null}
      ${page === 'application' ? html`<${ApplicationPage} />` : null}
    </div>
  </div></div>`;
}

// ── Hardware overview ─────────────────────────────────────────────────────────

function ComputeCard() {
  const hw = useStore((s) => s.hw);
  const pctUsed = hw.total ? Math.round((hw.used / hw.total) * 100) : 0;
  const onnxGpu = (hw.onnx_providers || []).find((p) => p !== 'CPUExecutionProvider');
  return html`<div class="card grow">
    <div class="label" style=${{ marginBottom: '10px' }}>Compute</div>
    <div class="row gap8" style=${{ marginBottom: '9px', alignItems: 'baseline' }}>
      <div style=${{ font: '500 15px/1 var(--sans)', color: 'var(--tx1)' }}>${hw.gpu_name ? `${/nvidia/i.test(hw.gpu_name) ? '' : ''}${hw.gpu_name}` : 'No GPU detected'}</div>
      ${hw.torch_cuda ? html`<span class="badge" style=${{ background: 'rgba(62,207,142,.14)', color: 'var(--green)', fontSize: '9.5px' }}>CUDA ${hw.cuda_version}</span>` : null}
    </div>
    ${hw.total ? html`<div class="bar lg" style=${{ marginBottom: '7px' }}><div style=${{ width: `${pctUsed}%` }}></div></div>
      <div class="kvm"><span>${fmtGB(hw.used)} / ${fmtGB(hw.total)} GB ${hw.unified ? 'unified memory' : 'VRAM'}</span><span>${hw.driver ? `driver ${hw.driver}` : ''}</span></div>` : null}
    <div class="col gap6" style=${{ marginTop: '12px', paddingTop: '11px', borderTop: '1px solid var(--line)' }}>
      <div class="row gap8 mono" style=${{ fontSize: '10.5px', color: hw.torch_cuda ? 'var(--tx3)' : 'var(--amber-tx)' }} title=${hw.torch_error || ''}>
        <${Dot} kind=${hw.torch_cuda ? 'ok' : 'warn'} />torch ${hw.torch_version || '—'} · ${hw.torch_cuda ? 'cuda available' : 'CPU only'}</div>
      <div class="row gap8 mono" style=${{ fontSize: '10.5px', color: hw.onnx_cuda ? 'var(--tx3)' : 'var(--amber-tx)' }} title=${hw.onnx_error || ''}>
        <${Dot} kind=${hw.onnx_cuda ? 'ok' : 'warn'} />onnxruntime ${hw.onnx_version || '—'} · ${onnxGpu || 'CPUExecutionProvider only'}</div>
      ${(hw.onnx_providers || []).includes('TensorrtExecutionProvider') ? html`<div class="row gap8 mono" style=${{ fontSize: '10.5px', color: hw.tensorrt_engines ? 'var(--tx3)' : 'var(--amber-tx)' }}>
        <${Dot} kind=${hw.tensorrt_engines ? 'ok' : 'warn'} />${hw.tensorrt_engines ? `TensorRT engine cache · ${plural(hw.tensorrt_engines, 'entry', 'entries')}` : 'TensorRT engine cache empty — first run will compile'}</div>` : null}
    </div>
  </div>`;
}

function ProviderCard() {
  const providers = useStore((s) => s.settingsData.providers);
  if (!providers) return html`<div class="card" style=${{ width: '340px', flex: 'none' }}><div class="note">Loading providers…</div></div>`;
  const pref = providers.preference;
  const trtOn = pref === 'tensorrt_cuda_cpu';
  const cudaOn = pref !== 'cpu_only';
  const setPref = async (value) => {
    try {
      const next = await rpc('providers.set', { preference: value });
      setState((s) => ({ settingsData: { ...s.settingsData, providers: next } }));
    } catch (err) {
      reportError(err);
    }
  };
  const row = (name, on, available, onToggle, note) => html`<div class=${`provrow ${on ? '' : 'off'}`}>
    <span class="dim" style=${{ fontSize: '10px' }}>⠿</span>
    <span class="grow">${name}</span>
    ${note ? html`<span class="dim" style=${{ fontSize: '9.5px' }}>${note}</span>` : null}
    ${onToggle ? html`<${Check} checked=${on} onChange=${onToggle} disabled=${!available && !on} title=${available ? '' : 'Not available on this machine'} />` : null}
  </div>`;
  return html`<div class="card" style=${{ width: '340px', flex: 'none' }}>
    <div class="label-row" style=${{ marginBottom: '10px' }}><div class="label">Execution provider order</div>
      <button class="btn link" onClick=${() => rpc('providers.refresh').then((p) => setState((s) => ({ settingsData: { ...s.settingsData, providers: p } }))).catch(reportError)}>Refresh</button></div>
    <div class="col gap4" style=${{ gap: '5px' }}>
      ${row('TensorrtExecutionProvider', trtOn, providers.tensorrt, (v) => setPref(v ? 'tensorrt_cuda_cpu' : 'cuda_cpu'), providers.tensorrt ? '' : 'not installed')}
      ${row('CUDAExecutionProvider', cudaOn, providers.cuda, (v) => setPref(v ? 'cuda_cpu' : 'cpu_only'), providers.cuda ? '' : 'not available')}
      ${row('CPUExecutionProvider', true, true, null, 'always last')}
    </div>
    <div class="note" style=${{ marginTop: '11px', lineHeight: 1.6 }}>${providers.description}</div>
    <div class="note" style=${{ marginTop: '4px' }}>applies on next model load</div>
  </div>`;
}

function HardwarePage() {
  return html`<div class="col gap14" style=${{ gap: '16px', maxWidth: '1100px' }}>
    <div class="row gap12" style=${{ alignItems: 'stretch' }}><${ComputeCard} /><${ProviderCard} /></div>
    <div class="row gap12" style=${{ alignItems: 'stretch' }}><div class="grow"><${DatabaseCard} /></div><div style=${{ width: '340px', flex: 'none' }}><${QueueCard} /></div></div>
    <${CacheSummary} />
    <div class="row gap10"><${Btn} onClick=${() => openModal('firstRun', true)}>Run environment check…</${Btn}>
      <div class="note">The first-run screen: PyTorch, ONNX Runtime and llama.cpp, detected rather than assumed.</div></div>
  </div>`;
}

// ── Database and queue ────────────────────────────────────────────────────────

function DatabaseCard() {
  const db = useStore((s) => s.settingsData.db);
  const [busy, setBusy] = useState('');
  if (!db) return html`<div class="card"><div class="note">Loading database statistics…</div></div>`;
  const setMode = async (local) => {
    try {
      const next = await rpc('db.set_mode', { local });
      setState((s) => ({ settingsData: { ...s.settingsData, db: next } }));
      toast(`Database mode: ${local ? 'local (per directory)' : 'global (shared)'}\n${next.location}`, 'ok', 5000);
    } catch (err) {
      reportError(err);
    }
  };
  const vacuum = async () => {
    setBusy('vacuum');
    try {
      const r = await rpc('db.vacuum');
      toast(`Database optimized: ${fmtBytes(r.before)} → ${fmtBytes(r.after)} (saved ${fmtBytes(r.saved)})`, 'ok', 6000);
      loadSettingsData(['db']);
    } catch (err) {
      reportError(err);
    } finally {
      setBusy('');
    }
  };
  const exportJson = async () => {
    const target = await native.pickSavePath({ title: 'Export database to JSON', defaultPath: 'interrogations-export.json', filters: [{ name: 'JSON', extensions: ['json'] }] });
    if (!target) return;
    setBusy('export');
    try {
      const r = await rpc('db.export', { path: target });
      toast(`Exported ${fmtInt(r.images)} images and ${fmtInt(r.interrogations)} interrogations to ${r.path}`, 'ok', 7000);
    } catch (err) {
      reportError(err);
    } finally {
      setBusy('');
    }
  };
  return html`<div class="card">
    <div class="label" style=${{ marginBottom: '11px' }}>Database</div>
    <div class="statgrid" style=${{ marginBottom: '12px' }}>
      <div><b>${fmtInt(db.total_images)}</b><span>images</span></div>
      <div><b>${fmtInt(db.total_interrogations)}</b><span>interrogations</span></div>
      <div><b>${fmtInt(db.unique_models_used)}</b><span>models</span></div>
      <div><b>${fmtBytes(db.size)}</b><span>on disk</span></div>
    </div>
    <div class="col gap6" style=${{ paddingTop: '11px', borderTop: '1px solid var(--line)' }}>
      <${Radio} checked=${!db.local} onChange=${() => setMode(false)}>Global database — one file for all directories</${Radio}>
      <${Radio} checked=${db.local} onChange=${() => setMode(true)}>Local database — stored beside each image set</${Radio}>
      <div class="note selectable" style=${{ marginTop: '4px' }}>${db.location}</div>
    </div>
    <div class="row gap8" style=${{ marginTop: '12px' }}>
      <${Btn} busy=${busy === 'vacuum'} disabled=${Boolean(busy)} onClick=${vacuum}>Vacuum</${Btn}>
      <${Btn} busy=${busy === 'export'} disabled=${Boolean(busy)} onClick=${exportJson}>Export JSON</${Btn}>
      ${native.available ? html`<${Btn} onClick=${() => native.showItem(db.location)}>Reveal file</${Btn}>` : null}
    </div>
  </div>`;
}

function QueueCard({ detailed = false }) {
  const queue = useStore((s) => s.settingsData.queue);
  if (!queue) return html`<div class="card"><div class="note">Loading queue…</div></div>`;
  const pending = queue.pending_count || 0;
  const process = async () => {
    try {
      await rpc('db.process_queue');
      setState((s) => ({ settingsData: { ...s.settingsData, queue: { ...queue, processing: true } } }));
    } catch (err) {
      reportError(err);
    }
  };
  const clear = async () => {
    const ok = await confirmDialog({ title: 'Clear queue', message: `Discard all ${plural(pending, 'queued operation')}? This permanently discards the data and cannot be undone.`, confirmLabel: 'Discard operations', variant: 'destructive' });
    if (!ok) return;
    try {
      const next = await rpc('db.clear_queue');
      setState((s) => ({ settingsData: { ...s.settingsData, queue: next } }));
      toast(`Cleared ${fmtInt(next.cleared)} queued operations.`, 'ok');
    } catch (err) {
      reportError(err);
    }
  };
  return html`<div class="card col" style=${{ height: '100%' }}>
    <div class="label" style=${{ marginBottom: '11px' }}>Operation queue</div>
    <div class="row gap8" style=${{ marginBottom: '10px' }}>
      <${Dot} kind=${pending ? (queue.at_capacity ? 'err' : 'warn') : 'ok'} />
      <div style=${{ font: '400 11px/1 var(--sans)', color: 'var(--tx3)' }}>${queue.processing ? 'Processing…' : pending ? `${plural(pending, 'pending operation')}` : 'Idle — 0 pending operations'}</div>
    </div>
    <div class="note" style=${{ fontSize: '10.5px', lineHeight: 1.7 }}>Writes are queued when the database is locked by another process${queue.at_capacity ? ' — the queue is at capacity' : queue.near_capacity ? ' — the queue is nearly full' : ''}.</div>
    ${detailed && pending ? html`<div class="listbox" style=${{ marginTop: '10px', maxHeight: '240px' }}>
      ${(queue.operations || []).slice(0, 50).map((op) => html`<div class="li"><span class="grow ellipsis">${op.summary || op.operation}</span>
        <span class="aside">${op.timestamp ? fmtTime(op.timestamp) : ''}${op.retry_count ? ` · retried ${op.retry_count}×` : ''}</span></div>`)}
    </div>` : null}
    <div class="spacer"></div>
    <div class="row gap8" style=${{ marginTop: '12px' }}>
      <${Btn} disabled=${!pending || queue.processing} onClick=${process}>Process queue</${Btn}>
      <${Btn} disabled=${!pending || queue.processing} onClick=${clear}>Clear</${Btn}>
      <${Btn} size="sm" onClick=${() => loadSettingsData(['queue'])} style=${{ marginLeft: 'auto' }}>Refresh</${Btn}>
    </div>
  </div>`;
}

// ── Model cache ───────────────────────────────────────────────────────────────

function CacheSummary() {
  const cache = useStore((s) => s.settingsData.cache);
  const cached = (cache?.models || []).filter((m) => m.cached || m.trt);
  return html`<div class="card">
    <div class="row gap12" style=${{ marginBottom: '11px' }}>
      <div class="label">Model cache</div>
      <div class="note" style=${{ fontSize: '10.5px', color: 'var(--tx5)' }}>${cache ? `${fmtBytes(cache.total)} total · HuggingFace + TensorRT engines` : 'calculating…'}</div>
      <div class="spacer"></div>
      <button class="btn link" onClick=${() => setState({ settingsPage: 'cache' })}>Open cache manager</button>
    </div>
    <div class="row gap8" style=${{ flexWrap: 'wrap' }}>
      ${cached.slice(0, 6).map((m) => html`<div class="cachecard"><div class="h ellipsis">${m.name}</div>
        <div class="d">${fmtBytes(m.size + m.trt_size)} · onnx${m.trt ? ' + trt engine' : ''}</div></div>`)}
      ${(cache?.gguf || []).slice(0, 1).map((g) => html`<div class="cachecard"><div class="h ellipsis" title=${g.path}>${g.name}</div>
        <div class="d">${fmtBytes(cache.gguf.reduce((a, b) => a + b.size, 0))} · gguf${cache.gguf.length > 1 ? ' + mmproj' : ''}</div></div>`)}
      ${cache && !cached.length && !(cache.gguf || []).length ? html`<div class="note">No tagger models are cached yet; they download on first load.</div>` : null}
    </div>
  </div>`;
}

function CachePage() {
  const { cache, loadedModel } = useStore((s) => ({ cache: s.settingsData.cache, loadedModel: s.interrog.model?.label }));
  const [busy, setBusy] = useState('');
  const remove = async (model, tensorrt) => {
    const ok = await confirmDialog({
      title: `Delete ${model.name}?`,
      message: `Delete the downloaded ${model.name} files${tensorrt && model.trt ? ' and its TensorRT engine' : ''} (${fmtBytes(model.size + (tensorrt ? model.trt_size : 0))})? The model downloads again the next time it is loaded.`,
      confirmLabel: 'Delete', variant: 'destructive',
    });
    if (!ok) return;
    setBusy(model.id);
    try {
      const next = await rpc('cache.delete', { model_id: model.id, tensorrt });
      setState((s) => ({ settingsData: { ...s.settingsData, cache: next } }));
      toast(`Deleted ${model.name}.`, 'ok');
    } catch (err) {
      reportError(err);
    } finally {
      setBusy('');
    }
  };
  if (!cache) return html`<div class="note">Scanning caches…</div>`;
  return html`<div class="col gap12" style=${{ maxWidth: '980px' }}>
    <div class="card">
      <div class="statgrid">
        <div><b>${fmtBytes(cache.total)}</b><span>total</span></div>
        <div><b>${fmtBytes(cache.huggingface)}</b><span>huggingface</span></div>
        <div><b>${fmtBytes(cache.tensorrt)}</b><span>tensorrt engines</span></div>
        <div><b>${fmtInt(cache.models.filter((m) => m.cached).length)}</b><span>cached taggers</span></div>
      </div>
    </div>
    <div class="card" style=${{ padding: 0 }}>
      ${cache.models.map((m) => html`<div class="row gap10" style=${{ padding: '10px 16px', borderBottom: '1px solid var(--line-soft)' }}>
        <${Dot} kind=${m.cached ? 'ok' : ''} />
        <div class="grow"><div class="mono" style=${{ fontSize: '11px', color: m.cached ? 'var(--tx1)' : 'var(--tx5)' }}>${m.id}</div>
          <div class="note">${m.type} · ${m.cached ? fmtBytes(m.size) : 'not downloaded'}${m.trt ? ` · TensorRT engine ${fmtBytes(m.trt_size)}` : ''}${loadedModel === m.id ? ' · loaded' : ''}</div></div>
        ${m.cached ? html`<${Btn} size="sm" busy=${busy === m.id} disabled=${Boolean(busy) || loadedModel === m.id} onClick=${() => remove(m, true)}>Delete</${Btn}>` : null}
      </div>`)}
    </div>
    ${(cache.gguf || []).length ? html`<div class="card"><div class="label" style=${{ marginBottom: '8px' }}>Configured GGUF files</div>
      ${cache.gguf.map((g) => html`<div class="kvm"><span class="ellipsis selectable">${g.path}</span><span>${fmtBytes(g.size)}</span></div>`)}
      <div class="note" style=${{ marginTop: '6px' }}>Your own files; managed from the Inquiry tab, never deleted here.</div></div>` : null}
  </div>`;
}

// ── llama.cpp runtime page (2c) ───────────────────────────────────────────────

function LlamaPage() {
  const { summary, update, health, checkingUpdate, config } = useStore((s) => ({
    summary: s.runtime.summary, update: s.runtime.update, health: s.runtime.health,
    checkingUpdate: s.runtime.checkingUpdate, config: s.inquiry.config,
  }));
  const [mode, setMode] = useState(null);
  useEffect(() => {
    rpc('inquiry.state').then((state) => setMode(state.config.llama_runtime_mode)).catch(() => {});
  }, []);
  const changeMode = async (value) => {
    setMode(value);
    try {
      await rpc('inquiry.save', { config: { llama_runtime_mode: value } });
      patch('inquiry', (q) => ({ config: { ...q.config, llama_runtime_mode: value } }));
      if (value === 'custom') toast('Custom mode: set the llama-server path in Inquiry → GGUF.', 'info', 5000);
    } catch (err) {
      reportError(err);
    }
  };
  const ladder = summary?.ladder || [];
  return html`<div class="row gap14" style=${{ alignItems: 'flex-start', maxWidth: '1100px' }}>
    <div class="grow col gap12">
      <div class="card" style=${{ padding: '16px' }}>
        <${RuntimeCard} summary=${summary} mode=${mode || config.llama_runtime_mode} onModeChange=${changeMode} />
      </div>
      <div class="card">
        <div class="label" style=${{ marginBottom: '10px' }}>Update check</div>
        ${checkingUpdate ? html`<div class="note row gap6"><span class="spinner"></span>Asking upstream…</div>`
          : update ? html`<div class="mono" style=${{ fontSize: '11px', lineHeight: 1.6, color: updateLevelColor(update.level) }}>${update.text}</div>
            ${update.warning ? html`<div class="note warn" style=${{ marginTop: '5px' }}>${update.warning}</div>` : null}`
          : html`<div class="note">Not checked yet.</div>`}
        <div class="help" style=${{ marginTop: '5px' }}>Keeps the install method: a source build is never swapped for a generic archive.</div>
      </div>
      <div class="card">
        <div class="label-row" style=${{ marginBottom: '10px' }}><div class="label">Health</div>
          <button class="btn link" onClick=${checkHealth}>Run check</button></div>
        ${health ? html`<div class="row gap8 mono" style=${{ fontSize: '11px', lineHeight: 1.6, color: health.ok ? 'var(--tx3)' : 'var(--red-tx)' }}>
          <${Dot} kind=${health.ok ? 'ok' : 'err'} /><span class="selectable" style=${{ whiteSpace: 'pre-wrap' }}>${health.message}</span></div>`
          : html`<div class="note">Starts the installed llama-server to confirm it actually runs.</div>`}
      </div>
    </div>
    <div class="card col" style=${{ width: '360px', flex: 'none' }}>
      <div class="label" style=${{ marginBottom: '12px' }}>Acquisition ladder · last run</div>
      <div class="col gap6">
        ${ladder.length ? ladder.map((rung, i) => {
          const last = i === ladder.length - 1;
          const fallback = rung.ok && summary?.fallback_from;
          const cls = rung.ok ? (fallback || (rung.accelerator === 'cpu' && summary?.detected !== 'cpu') ? 'fallback' : 'ok') : rung.skipped ? 'fail skip' : 'fail';
          return html`<div class=${`ladder ${cls}`}>
            <div class="mk">${rung.ok ? '✓' : rung.skipped ? '–' : '✕'}</div>
            <div class="grow"><div class="h">${rung.method} · ${rung.accelerator}</div><div class="d">${rung.detail}${last && rung.ok && cls === 'fallback' ? '' : ''}</div></div>
          </div>`;
        }) : html`<div class="note" style=${{ lineHeight: 1.6 }}>${summary?.installed
          ? `Installed ${summary.method === 'legacy' ? 'before the ladder was recorded' : 'without a recorded ladder'}. The next install or update records each rung here.`
          : 'Nothing installed yet. Installing tries a matched release, then a source build, then a CPU release.'}</div>`}
      </div>
      <div class="help" style=${{ marginTop: '14px' }}>Read from active-runtime.json, so this shows what is installed rather than what was configured. The one-time warning from the provisioning dialog is gone by the time a slow batch runs. This panel is not.</div>
      ${summary?.log_dir ? html`<button class="btn link" style=${{ marginTop: '10px', alignSelf: 'flex-start' }} onClick=${() => openLogFolder(summary)}>Open log folder</button>` : null}
    </div>
  </div>`;
}

// ── Application ───────────────────────────────────────────────────────────────

function ApplicationPage() {
  const settings = useStore((s) => s.settings);
  const set = (key, value) => {
    patch('settings', { [key]: value });
    rpc('app.set_setting', { key, value }).catch(reportError);
  };
  return html`<div class="col gap12" style=${{ maxWidth: '720px' }}>
    <div class="card col gap10">
      <div class="label">Performance</div>
      <${Check} checked=${settings.auto_unload !== false} onChange=${(v) => set('auto_unload', v)}>Auto-unload model after batch interrogation</${Check}>
      <div class="help">Frees GPU memory when not actively interrogating images.</div>
    </div>
    <div class="card col gap10">
      <div class="label">First run</div>
      <div class="help">Re-run the environment check: GPU, PyTorch, ONNX Runtime and the llama.cpp runtime.</div>
      <div><${Btn} onClick=${() => openModal('firstRun', true)}>Run environment check…</${Btn}></div>
    </div>
    <div class="card col gap8">
      <div class="label">About</div>
      <div class="help">Image Interrogator — batch image tagging with CLIP, WD and Camie taggers and llama.cpp multimodal inquiry. This window is the opt-in Electron front end, started by <span class="mono">./run.sh --electron</span>. Without the flag the PyQt6 interface opens as before; both share the same database and settings.</div>
    </div>
  </div>`;
}
