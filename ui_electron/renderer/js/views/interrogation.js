// Interrogation tab — direction 1a: run bar + config rail + live queue + now processing.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { rpc, thumbUrl } from '../api.js';
import { getState, patch, setState, useStore } from '../store.js';
import { chooseDirectory, itemStates, openInspect, openModal, setRecursive, setTab } from '../state.js';
import { Bar, Btn, Check, confirmDialog, Dot, Img, Label, reportError, Seg, Select, Slider, useInterval, VirtualList } from '../ui.js';
import { basename, fmtBytes, fmtDuration, fmtInt, fmtRate, pct, relPath, shortHash } from '../util.js';
import { FiltersEditor, filterSummary } from './filters.js';

const TXT_LABEL = { none: 'no .txt output', merge: 'write / merge', overwrite: 'overwrite' };

// ── Options / model config ────────────────────────────────────────────────────

async function ensureOptions() {
  const s = getState();
  if (s.interrog.options) return s.interrog.options;
  try {
    const options = await rpc('interrogate.options');
    patch('interrog', (i) => ({
      options,
      cfg: i.cfg || JSON.parse(JSON.stringify(options.defaults)),
      cfgType: i.cfg ? i.cfgType : (options.selected || 'WD'),
    }));
    return options;
  } catch (err) {
    reportError(err, 'Could not load model options');
    return null;
  }
}

function updateCfg(type, change) {
  patch('interrog', (i) => ({ cfg: { ...i.cfg, [type]: { ...i.cfg[type], ...change } } }));
}

async function loadModel() {
  const { cfg, cfgType } = getState().interrog;
  if (!cfg) return;
  try {
    await rpc('interrogate.load', { config: { type: cfgType, ...cfg[cfgType] } });
  } catch (err) {
    reportError(err);
  }
}

async function unloadModel() {
  try {
    await rpc('interrogate.unload');
  } catch (err) {
    reportError(err);
  }
}

async function startBatch() {
  const s = getState();
  const txtMode = s.interrog.txtMode || 'merge';
  const total = s.dir.paths.length;
  if (txtMode === 'overwrite') {
    const ok = await confirmDialog({
      title: 'Overwrite .txt files?',
      message: `This batch replaces the existing .txt sidecar of every one of the ${fmtInt(total)} images with fresh model output.`,
      confirmLabel: `Overwrite ${fmtInt(total)} files`,
      variant: 'destructive',
    });
    if (!ok) return;
  }
  try {
    await rpc('interrogate.start', { txt_mode: txtMode });
  } catch (err) {
    reportError(err);
  }
}

// ── Page ──────────────────────────────────────────────────────────────────────

export function InterrogationPage() {
  const panel = useStore((s) => s.interrog.panel);
  useEffect(() => { ensureOptions(); }, []);
  return html`<div class="page">
    <${RunBar} />
    <div class="workspace">
      <${Rail} />
      ${panel === 'dir' ? html`<${DirPanel} />` : null}
      ${panel === 'mdl' ? html`<${ModelSheet} />` : null}
      ${panel === 'flt' ? html`<${FilterSheet} />` : null}
      ${panel === 'out' ? html`<${OutputSheet} />` : null}
      <${LiveQueue} />
      <${NowProcessing} />
    </div>
  </div>`;
}

function Rail() {
  const panel = useStore((s) => s.interrog.panel);
  const toggle = (name) => patch('interrog', (i) => ({ panel: i.panel === name ? null : name }));
  const item = (name, label, title) => html`<button class=${`rail-btn ${panel === name ? 'active' : ''}`} title=${title} onClick=${() => toggle(name)}>${label}</button>`;
  return html`<div class="rail">
    ${item('dir', 'DIR', 'Directory and queue')}
    ${item('mdl', 'MDL', 'Model configuration')}
    ${item('flt', 'FLT', 'Tag filters')}
    ${item('out', 'OUT', '.txt output')}
    <div class="rail-sep"></div>
    <button class="rail-btn" title="Organize by tags" onClick=${() => openModal('organize', {})}>ORG</button>
    <div class="spacer"></div>
    <button class="rail-btn" title="About" onClick=${() => openModal('about', true)}>?</button>
  </div>`;
}

// ── Run bar ───────────────────────────────────────────────────────────────────

function RunBar() {
  const { progress, running, cancelling, lastRun, model, total, scanning, loading } = useStore((s) => ({
    progress: s.interrog.progress,
    running: s.interrog.running,
    cancelling: s.interrog.cancelling,
    lastRun: s.interrog.lastRun,
    model: s.interrog.model,
    total: s.dir.paths.length,
    scanning: s.dir.scanning,
    loading: s.interrog.loading,
  }));
  const p = progress && (running || lastRun) ? progress : null;
  const shownTotal = p ? p.total : total;
  const done = p ? p.done : 0;
  const cached = p ? p.cached : 0;
  const failed = p ? p.failed : 0;
  const processed = done + cached + failed;
  const queued = Math.max(0, shownTotal - processed - (running && p?.running ? 1 : 0));

  let hint = null;
  if (!running) {
    if (scanning) hint = 'scanning directory…';
    else if (!total) hint = 'select a directory with images';
    else if (loading) hint = 'model loading…';
    else if (!model) hint = html`<a onClick=${() => patch('interrog', { panel: 'mdl' })}>load a model</a> to start`;
  }

  return html`<div class="runbar">
    <div class="count"><b>${fmtInt(processed)}</b><span>/ ${fmtInt(shownTotal)} images</span></div>
    <div class="grow col gap6">
      <div class="segbar">
        <div class="s-done" style=${{ width: `${pct(done, shownTotal)}%` }}></div>
        <div class="s-cached" style=${{ width: `${pct(cached, shownTotal)}%` }}></div>
        ${running && p?.running ? html`<div class="s-run" style=${{ width: `${Math.max(0.4, pct(1, shownTotal))}%` }}></div>` : null}
        <div class="s-fail" style=${{ width: `${pct(failed, shownTotal)}%` }}></div>
      </div>
      <div class="legend">
        <span style=${{ color: 'var(--green)' }}>■ ${fmtInt(done)} interrogated</span>
        <span style=${{ color: 'var(--green-dk)' }}>■ ${fmtInt(cached)} cache hits</span>
        <span style=${{ color: 'var(--red)' }}>■ ${fmtInt(failed)} failed</span>
        <span>${fmtInt(queued)} queued</span>
        ${hint ? html`<span class="dim">· ${hint}</span>` : null}
        ${!running && lastRun ? html`<span class="dim">· last run ${lastRun.cancelled ? 'cancelled' : 'finished'} in ${fmtDuration(lastRun.elapsed)}</span>` : null}
      </div>
    </div>
    <div class="stats">
      <div class="stat"><div class="label">Throughput</div><div class="v">${running ? fmtRate(p?.rate) : '—'}</div></div>
      <div class="stat"><div class="label">ETA</div><div class="v">${running ? (p?.paused ? 'paused' : fmtDuration(p?.eta)) : '—'}</div></div>
      <div class="row gap8">
        ${running ? html`
          <${Btn} disabled=${cancelling} onClick=${() => rpc(p?.paused ? 'interrogate.resume' : 'interrogate.pause').catch(reportError)}>${p?.paused ? 'Resume' : 'Pause'}</${Btn}>
          <${Btn} variant="danger" disabled=${cancelling} onClick=${() => rpc('interrogate.cancel').catch(reportError)}>${cancelling ? 'Cancelling…' : 'Cancel'}</${Btn}>`
        : html`<${Btn} variant="primary" disabled=${!model || !total || scanning} onClick=${startBatch}>Start Batch</${Btn}>`}
      </div>
    </div>
  </div>`;
}

// ── Left column ───────────────────────────────────────────────────────────────

function useQueueCounts() {
  const { version, itemsVersion, paths, hasTxt } = useStore((s) => ({
    version: s.dir.version, itemsVersion: s.interrog.itemsVersion, paths: s.dir.paths, hasTxt: s.dir.hasTxt,
  }));
  return useMemo(() => {
    let done = 0; let failed = 0; let untagged = 0;
    paths.forEach((p, i) => {
      const st = itemStates.get(p)?.state;
      if (st === 'done' || st === 'cached') done += 1;
      else if (st === 'failed') failed += 1;
      if (!hasTxt[i]) untagged += 1;
    });
    return { all: paths.length, done, failed, untagged };
  }, [version, itemsVersion, paths, hasTxt]);
}

function DirPanel() {
  const { dir, model, txtMode, filters, filter, progress } = useStore((s) => ({
    dir: s.dir, model: s.interrog.model, txtMode: s.interrog.txtMode, filters: s.filters.stats,
    filter: s.interrog.queueFilter, progress: s.interrog.progress,
  }));
  const counts = useQueueCounts();
  const [queue, setQueue] = useState(null);
  useEffect(() => {
    rpc('db.queue').then(setQueue).catch(() => {});
  }, [progress?.processed === progress?.total]);
  const hits = progress ? progress.cached : 0;
  const hitBase = progress ? progress.done + progress.cached : 0;
  const cfg = model?.config || {};

  return html`<div class="side" style=${{ width: '308px' }}>
    <div style=${{ padding: '12px 14px 0' }}><${Label}>Directory</${Label}></div>
    <div class="col gap8" style=${{ padding: '9px 14px 13px', borderBottom: '1px solid var(--line)' }}>
      ${dir.path
        ? html`<div class="mono selectable" style=${{ fontSize: '11px', lineHeight: 1.5, color: 'var(--tx2)', wordBreak: 'break-all' }}>${dir.path}</div>`
        : html`<div class="help">No directory selected.</div>`}
      <div class="row gap8">
        <button class=${`pill ${dir.recursive ? 'on' : ''}`} style=${{ borderRadius: '4px' }} onClick=${() => setRecursive(!dir.recursive)}>
          <span class="row gap6"><span class=${`dotc ${dir.recursive ? '' : ''}`} style=${{ borderRadius: '2px', width: '9px', height: '9px', background: dir.recursive ? 'var(--blue)' : 'transparent', border: dir.recursive ? '0' : '1px solid var(--box)' }}></span>Recursive</span>
        </button>
        <span class="note" style=${{ fontSize: '10.5px' }}>
          ${dir.scanning
            ? html`<span class="row gap6"><span class="spinner" style=${{ color: 'var(--blue)' }}></span>found ${fmtInt(dir.progress?.count || 0)}…</span>`
            : dir.path ? `${fmtInt(dir.paths.length)} images · ${fmtInt(dir.dirCount)} ${dir.dirCount === 1 ? 'dir' : 'dirs'}` : ''}
        </span>
      </div>
      <${Btn} class="block" variant="fill" onClick=${chooseDirectory}>${dir.path ? 'Change Directory…' : 'Select Directory…'}</${Btn}>
    </div>

    <div class="label-row" style=${{ padding: '12px 14px 0', marginBottom: '3px' }}>
      <div class="label">Run config</div>
      <button class="btn link" onClick=${() => patch('interrog', { panel: 'mdl' })}>Edit</button>
    </div>
    <div style=${{ padding: '0 14px 12px', borderBottom: '1px solid var(--line)' }}>
      <div class="kv"><span>Model</span><span>${model ? `${model.type === 'WD' ? 'WD Tagger' : model.type === 'Camie' ? 'Camie Tagger' : 'CLIP'}` : 'not loaded'}</span></div>
      ${model ? html`<div class="kv"><span>Checkpoint</span><span class="ellipsis" style=${{ maxWidth: '170px' }} title=${model.label}>${model.short}</span></div>` : null}
      ${model && model.type !== 'CLIP' ? html`<div class="kv"><span>Threshold</span><span>${Number(model.threshold ?? cfg.threshold ?? 0).toFixed(2)}</span></div>` : null}
      ${model?.type === 'CLIP' ? html`<div class="kv"><span>Mode</span><span>${cfg.mode || 'best'}</span></div>` : null}
      <div class="kv"><span>Device</span><span>${model ? model.device : '—'}</span></div>
      <div class="kv"><span>.txt output</span><span>${TXT_LABEL[txtMode] || txtMode}</span></div>
      <div class="kv"><span>Filters</span><span>${filterSummary(filters)}</span></div>
    </div>

    <div style=${{ padding: '12px 14px 9px' }}><${Label}>Queue filter</${Label}></div>
    <div class="row gap6" style=${{ padding: '0 14px', flexWrap: 'wrap' }}>
      ${[['all', 'All', counts.all], ['done', 'Done', counts.done], ['failed', 'Failed', counts.failed], ['untagged', 'Untagged', counts.untagged]].map(([id, label, n]) => html`
        <button class=${`pill ${filter === id ? 'on' : ''}`} onClick=${() => patch('interrog', { queueFilter: id })}>${label} ${fmtInt(n)}</button>`)}
    </div>
    <div class="spacer"></div>
    <div class="col gap8" style=${{ padding: '12px 14px', borderTop: '1px solid var(--line)' }}>
      <div class="kvm"><span>db queue</span><span>${queue ? `${fmtInt(queue.pending_count)} pending` : '—'}</span></div>
      <div class="kvm"><span>cache hit rate</span><span>${hitBase ? `${((hits / hitBase) * 100).toFixed(1)}%` : '—'}</span></div>
    </div>
  </div>`;
}

function OutputSheet() {
  const { txtMode, autoUnload } = useStore((s) => ({ txtMode: s.interrog.txtMode, autoUnload: s.settings.auto_unload }));
  const setMode = (mode) => {
    patch('interrog', { txtMode: mode });
    rpc('app.set_setting', { key: 'txt_mode', value: mode }).catch(() => {});
  };
  return html`<div class="sheet" style=${{ width: '308px' }}>
    <div class="sheet-head"><div class="t">.txt Output</div><button class="xbtn" onClick=${() => patch('interrog', { panel: 'dir' })}>✕</button></div>
    <div class="block col gap10">
      <${Label}>Sidecar files</${Label}>
      <${Seg} class="pad" value=${txtMode} onChange=${setMode} options=${[
        { value: 'none', label: 'none' }, { value: 'merge', label: 'merge' }, { value: 'overwrite', label: 'overwrite' }]} />
      <div class="help">${txtMode === 'none'
        ? 'Results go to the database only; no .txt file is written or changed.'
        : txtMode === 'merge'
          ? 'New tags are merged into existing .txt files, keeping the tags already there.'
          : 'Each .txt file is replaced with the filtered model output.'}</div>
    </div>
    <div class="block col gap8">
      <${Label}>After the batch</${Label}>
      <${Check} checked=${autoUnload !== false} onChange=${(v) => {
        patch('settings', { auto_unload: v });
        rpc('app.set_setting', { key: 'auto_unload', value: v }).catch(reportError);
      }}>Auto-unload model after batch interrogation</${Check}>
      <div class="help">Frees GPU memory when not actively interrogating images.</div>
    </div>
  </div>`;
}

function FilterSheet() {
  return html`<div class="sheet" style=${{ width: '340px' }}>
    <div class="sheet-head"><div class="t">Tag Filters</div><button class="xbtn" onClick=${() => patch('interrog', { panel: 'dir' })}>✕</button></div>
    <div class="block scroll grow"><${FiltersEditor} compact /></div>
  </div>`;
}

// ── Model sheet (1b's Model Configuration) ────────────────────────────────────

function DeviceField({ value, onChange, cudaAvailable, statusOk, statusText, error }) {
  return html`<div>
    <div class="label" style=${{ marginBottom: '6px' }}>Device</div>
    <${Seg} class="pad" value=${value} onChange=${onChange} options=${[
      { value: 'cuda', label: 'cuda', disabled: !cudaAvailable, title: cudaAvailable ? '' : 'CUDA not available' },
      { value: 'cpu', label: 'cpu' }]} />
    <div class="row gap6" style=${{ marginTop: '7px' }} title=${error || ''}>
      <${Dot} kind=${statusOk ? 'ok' : 'warn'} /><span class="help" style=${{ color: statusOk ? 'var(--green-tx)' : 'var(--amber-tx)' }}>${statusText}</span>
    </div>
  </div>`;
}

function ModelSheet() {
  const { options, cfg, cfgType, model, loading, loadError } = useStore((s) => ({
    options: s.interrog.options, cfg: s.interrog.cfg, cfgType: s.interrog.cfgType,
    model: s.interrog.model, loading: s.interrog.loading, loadError: s.interrog.loadError,
  }));
  if (!options || !cfg) {
    return html`<div class="sheet" style=${{ width: '340px' }}><div class="empty"><span class="spinner"></span>Loading model options…</div></div>`;
  }
  const loadedSame = model && model.type === cfgType;
  const tabs = [['WD', 'WD Tagger'], ['CLIP', 'CLIP'], ['Camie', 'Camie']];
  return html`<div class="sheet" style=${{ width: '340px' }}>
    <div class="sheet-head"><div class="t">Model Configuration</div><button class="xbtn" onClick=${() => patch('interrog', { panel: 'dir' })}>✕</button></div>
    <div class="row gap4" style=${{ padding: '10px 14px 0' }}>
      ${tabs.map(([id, label]) => html`<button onClick=${() => patch('interrog', { cfgType: id })}
        style=${{
          padding: '5px 10px', borderRadius: '5px 5px 0 0', border: cfgType === id ? '1px solid var(--line-strong)' : '1px solid transparent', borderBottom: 'none',
          background: cfgType === id ? 'var(--hover)' : 'transparent', font: `${cfgType === id ? 500 : 400} 11px/1 var(--sans)`,
          color: cfgType === id ? 'var(--tx1)' : 'var(--tx4)', marginBottom: '-1px',
        }}>${label}</button>`)}
    </div>
    <div style=${{ margin: '0 14px', borderTop: '1px solid var(--line-strong)' }}></div>
    <div class="scroll grow col gap14" style=${{ padding: '14px' }}>
      ${cfgType === 'WD' ? html`<${WDConfig} options=${options} cfg=${cfg.WD} />` : null}
      ${cfgType === 'CLIP' ? html`<${ClipConfig} options=${options} cfg=${cfg.CLIP} />` : null}
      ${cfgType === 'Camie' ? html`<${CamieConfig} options=${options} cfg=${cfg.Camie} />` : null}
      ${cfgType !== 'CLIP' ? html`<${ProviderOrder} providers=${options.providers} />` : null}
      ${loadError ? html`<div class="banner red"><div class="mono selectable" style=${{ whiteSpace: 'pre-wrap' }}>${loadError}</div></div>` : null}
    </div>
    <div class="row gap8" style=${{ padding: '12px 14px', borderTop: '1px solid var(--line)' }}>
      <${Btn} class="grow" size="tall" variant="primary" busy=${loading} disabled=${loading} onClick=${loadModel}>
        ${loading ? 'Loading…' : loadedSame ? 'Reload Model' : model ? `Load ${cfgType} (replace ${model.type})` : 'Load Model'}
      </${Btn}>
      <${Btn} size="tall" disabled=${!model || loading} onClick=${unloadModel}>Unload</${Btn}>
    </div>
  </div>`;
}

function ThresholdField({ label, value, onChange, hint = true }) {
  return html`<div>
    <div class="label-row" style=${{ marginBottom: '7px' }}>
      <div class="label">${label}</div>
      <div class="mono" style=${{ font: '500 10.5px/1 var(--mono)', color: 'var(--tx1)' }}>${Number(value).toFixed(2)}</div>
    </div>
    <${Slider} value=${value} min=${0} max=${1} step=${0.01} onChange=${onChange} />
    ${hint ? html`<div class="slider-ends"><span>0.2 more tags</span><span>0.8 confident only</span></div>` : null}
  </div>`;
}

function WDConfig({ options, cfg }) {
  const set = (change) => updateCfg('WD', change);
  const devices = options.devices;
  return html`
    <div>
      <div class="label" style=${{ marginBottom: '6px' }}>WD model</div>
      <${Select} large value=${cfg.wd_model} onChange=${(v) => set({ wd_model: v })} options=${options.wd_models} />
      <div class="help" style=${{ marginTop: '6px' }}>${options.wd_notes[cfg.wd_model] || ''}</div>
    </div>
    <${ThresholdField} label="Confidence threshold" value=${cfg.threshold} onChange=${(v) => set({ threshold: v })} />
    <${DeviceField} value=${cfg.device} onChange=${(v) => set({ device: v })} cudaAvailable=${devices.onnx_cuda}
      statusOk=${devices.onnx_cuda} statusText=${devices.onnx_cuda ? 'ONNX Runtime CUDA available' : 'CUDA not available — using CPU'} error=${devices.onnx_error} />`;
}

function clipModelOptions(models) {
  if (!models) return [{ value: '', label: 'Loading models…', disabled: true }];
  const groups = [['sd_1x', 'SD 1.x Models (Recommended)'], ['sd_20', 'SD 2.0 Models'], ['sdxl', 'SDXL Models'], ['other', 'Other Models']];
  const out = [];
  for (const [key, label] of groups) {
    const list = models[key] || [];
    if (!list.length) continue;
    out.push({ value: `__${key}`, label: `── ${label} ──`, disabled: true });
    for (const m of list) {
      let text = m;
      if (m === 'ViT-L-14/openai') text = `${m} (Default)`;
      if (m === 'ViT-bigG-14/laion2b_s39b_b160k') text = `${m} (SDXL Default)`;
      out.push({ value: m, label: text });
    }
  }
  return out;
}

function ClipConfig({ options, cfg }) {
  const set = (change) => updateCfg('CLIP', change);
  const devices = options.devices;
  return html`
    <div>
      <div class="label" style=${{ marginBottom: '6px' }}>CLIP model</div>
      <${Select} large value=${cfg.clip_model} onChange=${(v) => set({ clip_model: v })} options=${clipModelOptions(options.clip_models)} disabled=${!options.clip_models} />
      <div class="help" style=${{ marginTop: '6px' }}>ViT-L-14 is a good balance; ViT-H/g are higher quality, ViT-B faster.</div>
    </div>
    <div>
      <div class="label" style=${{ marginBottom: '6px' }}>Caption model</div>
      <${Select} large value=${cfg.caption_model || 'None'} onChange=${(v) => set({ caption_model: v })} options=${options.caption_models} />
      <div class="help" style=${{ marginTop: '6px' }}>None uses CLIP only; BLIP2 variants give the best captions.</div>
    </div>
    <div>
      <div class="label" style=${{ marginBottom: '6px' }}>Mode</div>
      <${Seg} class="pad" value=${cfg.mode} onChange=${(v) => set({ mode: v })} options=${options.clip_modes} />
      <div class="help" style=${{ marginTop: '6px' }}>${{ best: 'Highest quality, slowest.', fast: 'Quick processing, good quality.', classic: 'Traditional approach.', negative: 'Generates negative prompts.' }[cfg.mode] || ''}</div>
    </div>
    <${DeviceField} value=${cfg.device} onChange=${(v) => set({ device: v })} cudaAvailable=${devices.pytorch_cuda}
      statusOk=${devices.pytorch_cuda} statusText=${devices.pytorch_cuda ? 'PyTorch CUDA available' : 'CUDA not available — using CPU'} error=${devices.pytorch_error} />`;
}

function CamieConfig({ options, cfg }) {
  const set = (change) => updateCfg('Camie', change);
  const devices = options.devices;
  const enabled = new Set(cfg.enabled_categories || []);
  const categoryThresholds = cfg.category_thresholds || {};
  const specific = cfg.threshold_profile === 'category_specific';
  return html`
    <div>
      <div class="label" style=${{ marginBottom: '6px' }}>Camie model</div>
      <${Select} large value=${cfg.camie_model} onChange=${(v) => set({ camie_model: v })} options=${options.camie_models} />
      <div class="help" style=${{ marginTop: '6px' }}>~70,527 tags across 7 categories. v2 is recommended.</div>
    </div>
    <div>
      <div class="label" style=${{ marginBottom: '6px' }}>Threshold profile</div>
      <${Select} large value=${cfg.threshold_profile} onChange=${(v) => set({ threshold_profile: v })} options=${options.camie_profiles} />
    </div>
    ${specific ? null : html`<${ThresholdField} label="Base threshold" value=${cfg.threshold} onChange=${(v) => set({ threshold: v })} />`}
    <div>
      <div class="label" style=${{ marginBottom: '8px' }}>Categories</div>
      <div style=${{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0,1fr))', gap: '7px 12px' }}>
        ${options.camie_categories.map((c) => html`<${Check} checked=${enabled.has(c)} onChange=${(v) => {
          const next = new Set(enabled);
          if (v) next.add(c); else next.delete(c);
          set({ enabled_categories: options.camie_categories.filter((x) => next.has(x)) });
        }}>${c}</${Check}>`)}
      </div>
    </div>
    ${specific ? html`<div class="col gap10">
      <div class="label">Category thresholds</div>
      ${options.camie_categories.map((c) => html`<div class="row gap10">
        <span class="mono" style=${{ width: '74px', fontSize: '10.5px', color: 'var(--tx3)' }}>${c}</span>
        <div class="grow"><${Slider} value=${categoryThresholds[c] ?? 0.5} min=${0} max=${1} step=${0.01}
          onChange=${(v) => set({ category_thresholds: { ...categoryThresholds, [c]: v } })} /></div>
        <span class="mono" style=${{ width: '32px', textAlign: 'right', fontSize: '10.5px' }}>${Number(categoryThresholds[c] ?? 0.5).toFixed(2)}</span>
      </div>`)}
    </div>` : null}
    <${DeviceField} value=${cfg.device} onChange=${(v) => set({ device: v })} cudaAvailable=${devices.onnx_cuda}
      statusOk=${devices.onnx_cuda} statusText=${devices.onnx_cuda ? 'ONNX Runtime CUDA available' : 'CUDA not available — using CPU'} error=${devices.onnx_error} />`;
}

function ProviderOrder({ providers }) {
  if (!providers) return null;
  const available = new Set(providers.available || []);
  const chain = providers.chain || [];
  return html`<div>
    <div class="label-row" style=${{ marginBottom: '6px' }}>
      <div class="label">Execution provider order</div>
      <button class="btn link" onClick=${() => { setState({ settingsPage: 'hardware' }); setTab('settings'); }}>Change</button>
    </div>
    <div class="col gap4">
      ${chain.map((name, index) => {
        const isCpu = name === 'CPUExecutionProvider';
        const ok = available.has(name);
        return html`<div class=${`provrow ${isCpu ? 'off' : ''}`}>
          <span class="dim" style=${{ fontSize: '10px' }}>${index + 1}</span>
          <span class="grow">${name}</span>
          <span style=${{ fontSize: '9.5px', color: isCpu ? 'var(--tx6)' : ok ? 'var(--green-tx)' : 'var(--amber-tx)' }}>${isCpu ? 'fallback' : ok ? 'ok' : 'missing'}</span>
        </div>`;
      })}
    </div>
  </div>`;
}

// ── Live queue ────────────────────────────────────────────────────────────────

const STATE_LABEL = { queued: 'QUEUED', running: 'RUNNING', done: 'DONE', cached: 'CACHED', failed: 'FAILED' };

function ElapsedMs({ since }) {
  const [, setTick] = useState(0);
  useInterval(() => setTick((t) => t + 1), 200);
  return html`${fmtInt(Date.now() - since)}`;
}

function LiveQueue() {
  const { paths, hasTxt, root, filter, itemsVersion, selected, runningPath, dirVersion } = useStore((s) => ({
    paths: s.dir.paths, hasTxt: s.dir.hasTxt, root: s.dir.path, filter: s.interrog.queueFilter,
    itemsVersion: s.interrog.itemsVersion, selected: s.interrog.selected,
    runningPath: s.interrog.running ? s.interrog.progress?.running : null, dirVersion: s.dir.version,
  }));
  const startedAt = useRef(new Map());

  const rows = useMemo(() => {
    const out = [];
    paths.forEach((p, i) => {
      const item = itemStates.get(p);
      const st = item?.state;
      if (filter === 'done' && st !== 'done' && st !== 'cached') return;
      if (filter === 'failed' && st !== 'failed') return;
      if (filter === 'untagged' && hasTxt[i]) return;
      out.push(i);
    });
    return out;
  }, [paths, hasTxt, filter, itemsVersion, dirVersion]);

  const runningIndex = runningPath ? rows.indexOf(paths.indexOf(runningPath)) : -1;

  const renderRow = (index) => {
    const p = paths[index];
    const item = itemStates.get(p) || { state: 'queued' };
    const st = item.state || 'queued';
    if (st === 'running' && !startedAt.current.has(p)) startedAt.current.set(p, Date.now());
    if (st !== 'running') startedAt.current.delete(p);
    const rowClass = `qrow ${st} ${selected === p ? 'sel' : ''}`;
    const src = item.source || (st === 'queued' && hasTxt[index] ? '.txt' : '—');
    return html`<div class=${rowClass} key=${p}
      onClick=${() => patch('interrog', { selected: p })}
      onDblClick=${() => openInspect(p)}>
      <div class=${`st st-${st}`}><${Dot} kind=${st === 'done' ? 'ok' : st === 'cached' ? '' : st === 'running' ? 'run' : st === 'failed' ? 'err' : ''} />${STATE_LABEL[st] || st.toUpperCase()}</div>
      <div class="file" title=${item.error || p}>${relPath(p, root)}${item.error ? html`<span class="err"> · ${item.error}</span>` : null}</div>
      <div class="src" style=${src === '.txt' ? { color: 'var(--green-tx)' } : null}>${src}</div>
      <div class="n" style=${item.tags == null ? { color: 'var(--tx7)' } : null}>${item.tags == null ? '—' : fmtInt(item.tags)}</div>
      <div class="ms" style=${st === 'running' ? { color: 'var(--blue-tx)' } : item.ms == null ? { color: 'var(--tx7)' } : null}>
        ${st === 'running' ? html`<${ElapsedMs} since=${startedAt.current.get(p)} />` : item.ms == null ? '—' : fmtInt(Math.round(item.ms))}
      </div>
    </div>`;
  };

  return html`<div class="grow col" style=${{ background: 'var(--bg)', minWidth: 0 }}>
    <div class="row gap12" style=${{ height: '34px', padding: '0 14px', borderBottom: '1px solid var(--line)', flex: 'none' }}>
      <div class="label">Live queue</div>
      <div class="spacer"></div>
      <div class="note" style=${{ fontSize: '10.5px', color: 'var(--tx7)' }}>double-click for advanced inspection</div>
    </div>
    <div class="qhead"><div>STATE</div><div>FILE</div><div>SOURCE</div><div style=${{ textAlign: 'right' }}>TAGS</div><div style=${{ textAlign: 'right' }}>MS</div></div>
    <${VirtualList} items=${rows} rowHeight=${32} renderRow=${renderRow} scrollToIndex=${runningIndex}
      empty=${html`<div class="empty">
        ${paths.length ? html`<div>No images match this filter.</div>` : html`<div class="big">No images queued</div>
        <div>Choose a directory to fill the queue. Every image gets a live state: queued, running, cached, done or failed.</div>
        <${Btn} variant="primary" onClick=${chooseDirectory}>Select Directory…</${Btn}>`}
      </div>`} />
  </div>`;
}

// ── Now processing ────────────────────────────────────────────────────────────

const PREVIEW_SIZE = 640;
const PREVIEW_PREFETCH = 2;

function NowProcessing() {
  const { runningPath, selected, tags, running, lastPath } = useStore((s) => ({
    runningPath: s.interrog.running ? s.interrog.progress?.running : null,
    selected: s.interrog.selected,
    tags: s.interrog.tags,
    running: s.interrog.running,
    lastPath: s.dir.paths[0] || null,
  }));
  const shown = runningPath || selected || null;
  const [meta, setMeta] = useState(null);
  useEffect(() => {
    let cancelled = false;
    setMeta(null);
    if (!shown) return undefined;
    const timer = setTimeout(() => {
      rpc('gallery.meta', { paths: [shown] }).then((r) => { if (!cancelled) setMeta(r[0] || null); }).catch(() => {});
    }, running ? 50 : 120);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [shown]);

  // Warm the next queued thumbnails so the preview keeps pace with a fast batch.
  useEffect(() => {
    if (!runningPath) return;
    const { paths } = getState().dir;
    let warmed = 0;
    for (let i = paths.indexOf(runningPath) + 1; i > 0 && i < paths.length && warmed < PREVIEW_PREFETCH; i++) {
      if (itemStates.get(paths[i])?.state !== 'queued') continue;
      const img = new Image();
      img.decoding = 'async';
      img.src = thumbUrl(paths[i], PREVIEW_SIZE);
      warmed++;
    }
  }, [runningPath]);

  const rows = tags?.rows || [];
  return html`<div class="col" style=${{ width: '352px', flex: 'none', background: 'var(--panel)', borderLeft: '1px solid var(--line)', minHeight: 0 }}>
    <div style=${{ padding: '12px 14px 9px' }}><${Label}>${runningPath ? 'Now processing' : selected ? 'Selected image' : 'Now processing'}</${Label}></div>
    <div style=${{ padding: '0 14px 12px' }}>
      <div class="preview" style=${{ height: '196px' }}>
        ${shown ? html`<${Img} src=${thumbUrl(shown, PREVIEW_SIZE)} />` : html`<div class="ph">${lastPath ? 'select a row or start a batch' : 'no image'}</div>`}
      </div>
    </div>
    <div class="col gap6" style=${{ padding: '0 14px 12px', borderBottom: '1px solid var(--line)' }}>
      <div class="kvm"><span>file</span><span class="ellipsis" style=${{ maxWidth: '240px' }}>${shown ? basename(shown) : '—'}</span></div>
      <div class="kvm"><span>hash</span><span>${meta?.hash ? shortHash(meta.hash) : '—'}</span></div>
      <div class="kvm"><span>size</span><span>${meta ? `${fmtBytes(meta.size)}${meta.w ? ` · ${meta.w}×${meta.h}` : ''}` : '—'}</span></div>
    </div>
    <div class="label-row" style=${{ padding: '12px 14px 0', marginBottom: '6px' }}>
      <div class="label">Discovered tags</div>
      <div class="note" style=${{ color: 'var(--tx7)' }}>${tags ? `${fmtInt(tags.unique)} unique` : ''}</div>
    </div>
    <div class="scroll grow taglist" style=${{ padding: '0 14px' }}>
      ${rows.length ? rows.map((r) => html`<div class="trow" key=${r.tag} title=${`${r.tag} · seen ${r.count}×${r.category ? ` · ${r.category}` : ''}${r.replaced ? ` · written as “${r.replaced}”` : ''}`}>
        <div class=${`t ${r.removed ? 'struck' : ''}`}>${r.tag}${r.replaced ? html`<span class="dim"> → ${r.replaced}</span>` : null}</div>
        <div class="k">${r.count > 1 ? `${fmtInt(r.count)}×` : ''}</div>
        <${Bar} value=${r.conf * 100} kind=${r.removed ? 'muted' : ''} />
        <div class="c" style=${r.removed ? { color: 'var(--tx6)' } : null}>${r.conf > 0 ? r.conf.toFixed(2) : 'N/A'}</div>
      </div>`) : html`<div class="help" style=${{ padding: '6px 0' }}>Tags appear here as the batch runs, most frequent first.</div>`}
    </div>
    <div class="note" style=${{ padding: '10px 14px', borderTop: '1px solid var(--line)', fontSize: '10.5px', color: 'var(--tx7)' }}>strikethrough = removed by filter rules before .txt write</div>
  </div>`;
}
