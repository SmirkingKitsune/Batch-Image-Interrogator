// Inquiry tab — 2b (batch) and 1d's single-image transcript, rewired to the
// managed llama.cpp runtime.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { native, rpc, thumbUrl } from '../api.js';
import { getState, patch, useStore } from '../store.js';
import { chooseDirectory, openInspect, openModal, setRecursive } from '../state.js';
import { Btn, Check, confirmDialog, Dot, Img, NumberField, reportError, Seg, TextField, toast, useInterval } from '../ui.js';
import { basename, debounce, fmtBytes, fmtDuration, fmtInt, pct, plural, relPath, shortHash } from '../util.js';
import { FiltersEditor } from './filters.js';
import { RuntimeCard } from './runtime.js';

// ── Data ──────────────────────────────────────────────────────────────────────

async function loadInquiryState() {
  try {
    const state = await rpc('inquiry.state');
    patch('inquiry', (q) => ({
      loaded: true,
      config: state.config,
      options: state.options,
      tasks: state.tasks,
      model: state.model,
      loading: state.loading,
      sources: state.sources ?? q.sources,
      mode: q.loaded ? q.mode : (state.options.active_tab === 1 ? 'batch' : 'single'),
      batch: state.batch ? { ...q.batch, progress: state.batch, running: state.batch.active } : q.batch,
    }));
    if (state.sources == null) rpc('inquiry.sources').catch(() => {});
  } catch (err) {
    reportError(err, 'Could not load Inquiry settings');
  }
}

const saveConfigSoon = debounce((config) => rpc('inquiry.save', { config }).catch(reportError), 400);
const saveOptionsSoon = debounce((options) => rpc('inquiry.save', { options }).catch(() => {}), 400);

function setConfig(change) {
  patch('inquiry', (q) => {
    const config = { ...q.config, ...change };
    saveConfigSoon(config);
    return { config };
  });
}

function setOptions(change) {
  patch('inquiry', (q) => {
    const options = { ...q.options, ...change };
    saveOptionsSoon(change);
    return { options };
  });
}

async function loadLlama() {
  const { config } = getState().inquiry;
  saveConfigSoon.cancel?.();
  try {
    await rpc('inquiry.load', { config });
  } catch (err) {
    if (err.code === 'no_runtime') {
      const ok = await confirmDialog({
        title: 'Install llama.cpp runtime?',
        message: `No llama.cpp runtime is installed.\n\nDetected accelerator: ${err.data?.accelerator || 'unknown'}.\n\nInstall one now? A matched release is downloaded when available; otherwise it is built from source, which can take several minutes.`,
        confirmLabel: 'Choose install method…',
      });
      if (ok) openModal('provision', {});
      return;
    }
    reportError(err);
  }
}

async function selectImage(path) {
  patch('inquiry', (q) => ({ single: { ...q.single, path, info: null, turns: [], raw: '', error: null } }));
  if (!path) return;
  try {
    const info = await rpc('inquiry.image', { path });
    if (getState().inquiry.single.path !== path) return;
    patch('inquiry', (q) => ({
      single: { ...q.single, info, turns: info.history || [], raw: info.latest_raw || '', priorChecked: new Set() },
    }));
  } catch (err) {
    reportError(err);
  }
}

// ── Page ──────────────────────────────────────────────────────────────────────

export function InquiryPage() {
  const { loaded, panel, mode, paths, modelLabel, singlePath } = useStore((s) => ({
    loaded: s.inquiry.loaded, panel: s.inquiry.panel, mode: s.inquiry.mode, paths: s.dir.paths,
    modelLabel: s.inquiry.model?.label || '', singlePath: s.inquiry.single.path,
  }));
  useEffect(() => { loadInquiryState(); }, []);
  // Keep a valid selection as the shared queue changes.
  useEffect(() => {
    if (!paths.length) { if (singlePath) selectImage(null); return; }
    if (!singlePath || !paths.includes(singlePath)) selectImage(paths[0]);
  }, [paths]);
  // History is scoped to the loaded model.
  useEffect(() => { if (singlePath) selectImage(singlePath); }, [modelLabel]);

  if (!loaded) return html`<div class="page"><div class="empty"><span class="spinner"></span>Loading Inquiry…</div></div>`;
  return html`<div class="page"><div class="workspace">
    <${Rail} />
    ${panel === 'dir' ? html`<${SourcePanel} />` : null}
    ${panel === 'gguf' ? html`<${GgufSheet} />` : null}
    ${panel === 'flt' ? html`<div class="sheet" style=${{ width: '340px' }}>
      <div class="sheet-head"><div class="t">Tag Filters</div><button class="xbtn" onClick=${() => patch('inquiry', { panel: null })}>✕</button></div>
      <div class="block scroll grow"><${FiltersEditor} compact /></div></div>` : null}
    <${Controls} />
    ${mode === 'batch' ? html`<${BatchTranscript} />` : html`<${SingleTranscript} />`}
  </div></div>`;
}

function Rail() {
  const panel = useStore((s) => s.inquiry.panel);
  const toggle = (name) => patch('inquiry', (q) => ({ panel: q.panel === name ? null : name }));
  const item = (name, label, title, cls = '') => html`<button class=${`rail-btn ${cls} ${panel === name ? 'active' : ''}`} title=${title} onClick=${() => toggle(name)}>${label}</button>`;
  return html`<div class="rail">
    ${item('dir', 'DIR', 'Shared image source')}
    ${item('gguf', 'GGUF', 'Runtime, model and inference', 'small')}
    ${item('flt', 'FLT', 'Tag filters')}
  </div>`;
}

function SourcePanel() {
  const dir = useStore((s) => s.dir);
  return html`<div class="side" style=${{ width: '300px' }}>
    <div class="block col gap8">
      <div class="label">Shared image source</div>
      ${dir.path ? html`<div class="mono selectable" style=${{ fontSize: '11px', lineHeight: 1.5, color: 'var(--tx2)', wordBreak: 'break-all' }}>${dir.path}</div>`
        : html`<div class="help">No directory selected.</div>`}
      <div class="row gap8">
        <${Check} checked=${dir.recursive} onChange=${setRecursive}>Recursive</${Check}>
        <span class="note">${dir.path ? `${fmtInt(dir.paths.length)} images` : ''}</span>
      </div>
      <${Btn} variant="fill" onClick=${chooseDirectory}>${dir.path ? 'Change Directory…' : 'Select Directory…'}</${Btn}>
      <div class="help">The queue is shared with the Interrogation tab, so prior tagger results for these images can be used as context.</div>
    </div>
  </div>`;
}

// ── GGUF sheet: runtime, model, inference ─────────────────────────────────────

function PathField({ value, placeholder, onCommit, title, filters }) {
  return html`<div class="row gap6">
    <div class="grow"><${TextField} large value=${value || ''} placeholder=${placeholder} title=${value || ''} onCommit=${onCommit} /></div>
    <button class="iconbtn" style=${{ height: '30px', width: '30px' }} title="Browse…"
      onClick=${async () => {
        const picked = await native.pickFile({ title, defaultPath: value || undefined, filters });
        if (picked) onCommit(picked);
      }}>…</button>
  </div>`;
}

function GgufSheet() {
  const { config, model, loading, loadError, runtime, singleRunning, batchRunning } = useStore((s) => ({
    config: s.inquiry.config, model: s.inquiry.model, loading: s.inquiry.loading, loadError: s.inquiry.loadError,
    runtime: s.runtime.summary, singleRunning: s.inquiry.single.running, batchRunning: s.inquiry.batch.running,
  }));
  const [metadata, setMetadata] = useState(null);
  const custom = config.llama_runtime_mode === 'custom';
  const busy = singleRunning || batchRunning;
  const readMetadata = async (path) => {
    try {
      setMetadata({ loading: true });
      setMetadata(await rpc('inquiry.metadata', { model_path: path || config.llama_model_path, ctx_size: config.ctx_size, max_tokens: config.max_tokens }));
    } catch (err) {
      setMetadata({ ok: false, lines: [err.message] });
    }
  };
  const gguf = [{ name: 'GGUF model', extensions: ['gguf'] }, { name: 'All files', extensions: ['*'] }];

  return html`<div class="sheet" style=${{ width: '316px' }}>
    <div class="scroll grow">
      <div class="block">
        <div class="label-row"><div class="label">Runtime</div><button class="xbtn" onClick=${() => patch('inquiry', { panel: null })}>✕</button></div>
        <${Seg} value=${custom ? 'custom' : 'managed'} onChange=${(v) => setConfig({ llama_runtime_mode: v })}
          options=${[{ value: 'managed', label: 'managed' }, { value: 'custom', label: 'custom path' }]} />
        <div style=${{ marginTop: '10px' }}>
          ${custom ? html`<div class="col gap6">
            <${PathField} value=${config.llama_binary_path} placeholder="/path/to/llama-server" title="Select llama-server binary"
              onCommit=${(v) => setConfig({ llama_binary_path: v })} />
            <div class="note">Your own llama-server build. The managed runtime card applies only in managed mode.</div>
          </div>` : html`<${RuntimeCard} summary=${runtime} compact />`}
        </div>
      </div>

      <div class="block col gap6">
        <div class="label">Model</div>
        <${PathField} value=${config.llama_model_path} placeholder="Multimodal .gguf model (required)" title="Select multimodal GGUF model"
          filters=${gguf} onCommit=${(v) => { setConfig({ llama_model_path: v }); if (v) readMetadata(v); }} />
        <${PathField} value=${config.llama_mmproj_path} placeholder="mmproj .gguf (optional)" title="Select multimodal projector (optional)"
          filters=${gguf} onCommit=${(v) => setConfig({ llama_mmproj_path: v })} />
        ${model?.notes?.length ? html`<div class="note">${model.notes.map((n) => html`<div>${n}</div>`)}</div>` : null}
        <div class="row gap8">
          <button class="btn link" disabled=${!config.llama_model_path} onClick=${() => readMetadata()}>Read metadata</button>
          ${metadata?.suggested ? html`<button class="btn link" onClick=${() => { setConfig({ ctx_size: metadata.suggested }); setMetadata({ ...metadata, suggested: null, lines: [`Context size set to ${fmtInt(metadata.suggested)}.`] }); }}>Apply suggested ctx ${fmtInt(metadata.suggested)}</button>` : null}
        </div>
        ${metadata ? html`<div class=${`note ${metadata.loading ? '' : metadata.ok ? 'ok' : 'warn'}`} style=${{ lineHeight: 1.55 }}>
          ${metadata.loading ? 'Reading GGUF metadata…' : metadata.lines.map((l) => html`<div>${l}</div>`)}
        </div>` : null}
      </div>

      <div class="block">
        <div class="label" style=${{ marginBottom: '9px' }}>Inference</div>
        <div style=${{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 12px', alignItems: 'center' }}>
          <span class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx5)' }}>ctx</span>
          <${NumberField} value=${config.ctx_size} min=${256} max=${131072} step=${512} onCommit=${(v) => setConfig({ ctx_size: v })} />
          <span class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx5)' }}>gpu_layers</span>
          <${NumberField} value=${config.gpu_layers} min=${-1} max=${999} onCommit=${(v) => setConfig({ gpu_layers: v })} title="-1 offloads every layer the runtime can fit on the GPU." />
          <span class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx5)' }}>temp</span>
          <${NumberField} value=${config.temperature} min=${0} max=${2} step=${0.05} onCommit=${(v) => setConfig({ temperature: v })} />
          <span class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx5)' }}>max_tokens</span>
          <${NumberField} value=${config.max_tokens} min=${16} max=${131072} step=${32} onCommit=${(v) => setConfig({ max_tokens: v })} />
          <span class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx5)' }}>port</span>
          <${NumberField} value=${config.server_port} min=${1024} max=${65535} onCommit=${(v) => setConfig({ server_port: v })} title="Reassigned automatically when the port is already in use." />
        </div>
        <div class="col gap6" style=${{ marginTop: '10px' }}>
          <${Check} checked=${Boolean(config.disable_reasoning)} aside="live" asideClass="live"
            title="Ask the chat template not to emit a thinking block. Takes effect on the next request without a reload."
            onChange=${(v) => { patch('inquiry', (q) => ({ config: { ...q.config, disable_reasoning: v } })); rpc('inquiry.set_reasoning', { disabled: v }).catch(reportError); }}>Disable reasoning</${Check}>
          <${Check} checked=${Boolean(config.no_reasoning_preserve)} aside="reload" asideClass="reload"
            title="Passes --no-reasoning-preserve. Changing this restarts the server."
            onChange=${(v) => setConfig({ no_reasoning_preserve: v })}>No reasoning preserve</${Check}>
          <${Check} checked=${config.repetition_guard !== false} aside="live" asideClass="live"
            title="Adds llama.cpp's DRY sampler to every request, which penalizes runaway repetition, and stops a reply that still gets stuck repeating itself, retrying it once with the model's recommended sampling. Takes effect on the next request."
            onChange=${(v) => { patch('inquiry', (q) => ({ config: { ...q.config, repetition_guard: v } })); rpc('inquiry.set_guard', { enabled: v }).catch(reportError); }}>Repetition guard</${Check}>
          <div class="row gap8" style=${{ justifyContent: 'space-between' }}>
            <span class="mono" style=${{ fontSize: '10.5px', color: config.disable_reasoning ? 'var(--tx7)' : 'var(--tx5)' }}>reasoning budget</span>
            ${config.disable_reasoning
              ? html`<span class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx7)' }}>${config.reasoning_budget} · n/a while disabled</span>`
              : html`<${NumberField} value=${config.reasoning_budget} min=${-1} max=${32768} step=${128} special="unrestricted"
                  title="Cap thinking at N tokens. -1 is unrestricted; reload the model to apply." onCommit=${(v) => setConfig({ reasoning_budget: v })} />`}
          </div>
        </div>
      </div>
    </div>
    <div class="col gap8" style=${{ padding: '12px 14px', borderTop: '1px solid var(--line)' }}>
      <div class="row gap8">
        <${Btn} class="grow" size="tall" variant=${model ? 'fill' : 'primary'} busy=${loading} disabled=${loading || busy} onClick=${loadLlama}>
          ${loading ? 'Starting server…' : model ? 'Reload Llama Model' : 'Load Llama Model'}</${Btn}>
        <${Btn} class="grow" size="tall" disabled=${!model || loading || busy} onClick=${() => rpc('inquiry.unload').catch(reportError)}>Unload Model</${Btn}>
      </div>
      ${model ? html`<div class="note ok ellipsis" title=${model.log_path || ''}>Connected: ${model.file} · pid ${model.pid}</div>`
        : loadError ? html`<div class="note err" style=${{ whiteSpace: 'pre-wrap' }}>
            ${loadError.message}
            ${loadError.logs ? html`<details style=${{ marginTop: '6px' }}><summary style=${{ cursor: 'pointer', color: 'var(--tx5)' }}>recent llama logs</summary><pre class="logbox" style=${{ maxHeight: '180px', marginTop: '6px' }}>${loadError.logs}</pre></details>` : null}
          </div>`
        : html`<div class="note">Configure and load a llama model to begin inquiries.</div>`}
    </div>
  </div>`;
}

// ── Controls column ────────────────────────────────────────────────────────────

/**
 * Thinking on/off and effort, for chat templates that expose them. Both ride
 * on each request, so a change needs no reload; mid-batch it applies from the
 * next image. The choices come from the loaded model's template, because one
 * that validates its effort (Qwen3.8) fails a request with any other value.
 */
function ReasoningControl() {
  const { controls, disabled, requested } = useStore((s) => ({
    controls: s.inquiry.model?.reasoning || null,
    disabled: Boolean(s.inquiry.config.disable_reasoning),
    requested: s.inquiry.config.reasoning_effort || '',
  }));
  const effort = controls?.effort || null;
  if (!controls || (!effort && !controls.thinking_toggle)) return null;

  const options = [];
  if (controls.thinking_toggle) options.push({ value: 'off', label: 'off', title: 'Answer without a thinking block (enable_thinking = false).' });
  if (effort) {
    if (!effort.default) options.push({ value: '', label: 'auto', title: 'Leave reasoning_effort to the chat template.' });
    for (const v of effort.values) {
      options.push({ value: v, label: v, title: `reasoning_effort = ${v}${v === effort.default ? ' (template default)' : ''}` });
    }
  } else {
    options.push({ value: 'on', label: 'on', title: 'Think before answering (template default).' });
  }
  let value = 'on';
  if (disabled) value = 'off';
  else if (effort) value = effort.values.includes(requested) ? requested : (effort.aliases?.[requested] || effort.default || '');

  const choose = (next) => {
    const setsEffort = effort && next !== 'off' && next !== 'on';
    const change = { disable_reasoning: next === 'off', ...(setsEffort ? { reasoning_effort: next } : {}) };
    patch('inquiry', (q) => ({ config: { ...q.config, ...change } }));
    rpc('inquiry.set_reasoning', { disabled: change.disable_reasoning, ...(setsEffort ? { effort: next } : {}) }).catch(reportError);
  };
  return html`<div class="block tight">
    <div class="label-row" style=${{ marginBottom: '8px' }}>
      <div class="label">${effort ? 'Reasoning effort' : 'Thinking'}</div>
      <div class="note" style=${{ color: 'var(--tx7)' }} title="Sent with each request, so no reload is needed. During a batch it applies from the next image.">
        ${effort?.default ? `default ${effort.default} · ` : ''}per request</div>
    </div>
    <${Seg} value=${value} options=${options} onChange=${choose} />
  </div>`;
}

function TaskChips({ value, onChange, tasks, disabled }) {
  return html`<div class="row gap4" style=${{ flexWrap: 'wrap', gap: '5px' }}>
    ${tasks.map((t) => html`<button class=${`task ${value === t ? 'on' : ''} ${t === 'audit' ? 'audit' : ''}`} disabled=${disabled}
      onClick=${() => onChange(t)}>${t}</button>`)}
  </div>`;
}

function Controls() {
  const mode = useStore((s) => s.inquiry.mode);
  const setMode = (m) => {
    patch('inquiry', { mode: m });
    rpc('inquiry.save', { options: { active_tab: m === 'batch' ? 1 : 0 } }).catch(() => {});
  };
  return html`<div class="col" style=${{ width: '300px', flex: 'none', borderRight: '1px solid var(--line)', background: 'var(--panel)', minHeight: 0 }}>
    <div class="row gap4" style=${{ padding: '10px 12px 0' }}>
      ${['single', 'batch'].map((m) => html`<button onClick=${() => setMode(m)} style=${{
        padding: '6px 11px', borderRadius: '5px', font: `${mode === m ? 500 : 400} 11px/1 var(--sans)`,
        background: mode === m ? 'var(--sel)' : 'transparent', border: mode === m ? '1px solid var(--line-sel)' : '1px solid transparent',
        color: mode === m ? 'var(--blue-tx)' : 'var(--tx4)',
      }}>${m === 'single' ? 'Single' : 'Batch'}</button>`)}
    </div>
    ${mode === 'single' ? html`<${SingleControls} />` : html`<${BatchControls} />`}
  </div>`;
}

function ImagePicker({ paths, root, value, onChange }) {
  const [query, setQuery] = useState('');
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q ? paths.filter((p) => relPath(p, root).toLowerCase().includes(q)) : paths;
    return list.slice(0, 1500);
  }, [paths, root, query]);
  const index = paths.indexOf(value);
  return html`<div class="col gap6">
    <div class="row gap6">
      <button class="iconbtn" style=${{ height: '30px' }} disabled=${index <= 0} onClick=${() => onChange(paths[index - 1])}>←</button>
      <select class="select lg grow" value=${value || ''} onChange=${(e) => onChange(e.currentTarget.value)}>
        ${value && !filtered.includes(value) ? html`<option value=${value}>${relPath(value, root)}</option>` : null}
        ${filtered.map((p) => html`<option value=${p} key=${p}>${relPath(p, root)}</option>`)}
      </select>
      <button class="iconbtn" style=${{ height: '30px' }} disabled=${index < 0 || index >= paths.length - 1} onClick=${() => onChange(paths[index + 1])}>→</button>
    </div>
    ${paths.length > 30 ? html`<${TextField} placeholder=${`Filter ${fmtInt(paths.length)} images…`} value=${query} onInput=${setQuery} />` : null}
  </div>`;
}

function SingleControls() {
  const { paths, root, single, options, tasks, model, busy } = useStore((s) => ({
    paths: s.dir.paths, root: s.dir.path, single: s.inquiry.single, options: s.inquiry.options, tasks: s.inquiry.tasks,
    model: s.inquiry.model, busy: s.inquiry.single.running || s.inquiry.batch.running,
  }));
  const [checked, setChecked] = useState(() => new Set());
  useEffect(() => setChecked(new Set()), [single.info?.hash]);
  const task = options.single_task || 'describe';
  const info = single.info;
  const priorTurns = single.turns.length;

  const send = async () => {
    try {
      await rpc('inquiry.send', {
        path: single.path,
        task,
        prompt: options.single_prompt || '',
        prior_indices: [...checked],
        include_transcripts: Boolean(options.single_include_prior_transcripts),
      });
    } catch (err) {
      reportError(err);
    }
  };
  const reset = async () => {
    const ok = await confirmDialog({ title: 'Reset image context', message: `Clear the saved inquiry session for ${basename(single.path)}? Earlier turns will no longer be sent as context.`, confirmLabel: 'Reset context', variant: 'danger' });
    if (!ok) return;
    try {
      await rpc('inquiry.reset', { path: single.path });
      patch('inquiry', (q) => ({ single: { ...q.single, turns: [] } }));
      toast('Image context reset.', 'ok', 2500);
    } catch (err) {
      reportError(err);
    }
  };

  if (!paths.length) {
    return html`<div class="empty" style=${{ flex: 1 }}><div>No images in the shared queue.</div>
      <${Btn} variant="primary" onClick=${chooseDirectory}>Select Directory…</${Btn}></div>`;
  }
  return html`<div class="scroll grow col" style=${{ minHeight: 0 }}>
    <div class="block tight" style=${{ paddingTop: '14px' }}>
      <div class="label-row"><div class="label">Image</div><button class="btn link" disabled=${!single.path} onClick=${() => openInspect(single.path)}>Open advanced</button></div>
      <${ImagePicker} paths=${paths} root=${root} value=${single.path} onChange=${selectImage} />
      <div class="preview" style=${{ height: '150px', marginTop: '8px' }}>
        ${single.path ? html`<${Img} src=${thumbUrl(single.path, 600)} />` : html`<div class="ph">no image</div>`}
      </div>
      <div class="note" style=${{ marginTop: '6px' }}>${info?.meta?.width ? `${info.meta.width}×${info.meta.height} · ${fmtBytes(info.meta.file_size)}` : ''}</div>
    </div>
    <div class="block tight">
      <div class="label" style=${{ marginBottom: '7px' }}>Task</div>
      <${TaskChips} value=${task} tasks=${tasks} onChange=${(t) => setOptions({ single_task: t })} />
      <div class="label" style=${{ margin: '11px 0 7px' }}>Prompt</div>
      <${TextField} multiline rows=${3} value=${options.single_prompt || ''}
        placeholder=${task === 'vqa' ? 'Ask a visual question…' : task === 'custom' ? 'Custom instructions…' : 'Optional guidance for this image…'}
        onCommit=${(v) => setOptions({ single_prompt: v })} />
    </div>
    <${ReasoningControl} />
    <div class="block tight">
      <div class="label" style=${{ marginBottom: '8px' }}>Context sources</div>
      <div class="col gap4">
        ${(info?.prior || []).map((row) => html`<button class=${`srcrow ${checked.has(row.index) ? 'on' : ''}`} key=${row.index}
          onClick=${() => setChecked((prev) => { const n = new Set(prev); if (n.has(row.index)) n.delete(row.index); else n.add(row.index); return n; })}>
          <span class="box"></span><span class="grow ellipsis" title=${row.model_name}>${row.model_type} · ${row.display || String(row.model_name || '').split('/').pop()}</span>
          <span class="aside">${plural(row.tags, 'tag')}</span></button>`)}
        <button class=${`srcrow ${options.single_include_prior_transcripts ? 'on' : ''}`}
          onClick=${() => setOptions({ single_include_prior_transcripts: !options.single_include_prior_transcripts })}>
          <span class="box"></span><span class="grow">prior transcripts</span><span class="aside">${plural(priorTurns, 'turn')}</span></button>
        ${info && !info.prior.length ? html`<div class="note">No prior tagger results for this image yet.</div>` : null}
      </div>
    </div>
    <div class="spacer"></div>
    <div class="row gap8" style=${{ padding: '12px', borderTop: '1px solid var(--line)' }}>
      <${Btn} class="grow" size="tall" variant="primary" busy=${single.running} disabled=${!model || !single.path || busy} onClick=${send}
        title=${model ? '' : 'Load a llama model first'}>${single.running ? 'Waiting for model…' : 'Send Inquiry'}</${Btn}>
      <${Btn} size="tall" disabled=${!single.path || busy} onClick=${reset}>Reset Context</${Btn}>
    </div>
  </div>`;
}

function BatchControls() {
  const { options, tasks, sources, scanning, model, batch, total, singleRunning } = useStore((s) => ({
    options: s.inquiry.options, tasks: s.inquiry.tasks, sources: s.inquiry.sources, scanning: s.inquiry.sourcesScanning,
    model: s.inquiry.model, batch: s.inquiry.batch, total: s.dir.paths.length, singleRunning: s.inquiry.single.running,
  }));
  const task = options.batch_task || 'describe';
  const audit = task === 'audit';
  const selectedKeys = new Set(options.batch_context_source_keys || []);
  const txtMode = audit ? 'merge' : (options.txt_output_mode || 'merge');
  const running = batch.running;
  const toggleSource = (key) => {
    const next = new Set(selectedKeys);
    if (next.has(key)) next.delete(key); else next.add(key);
    setOptions({ batch_context_source_keys: [...next], batch_include_prior_tables: next.size > 0 });
  };
  const start = async () => {
    if (!audit && txtMode === 'overwrite') {
      const ok = await confirmDialog({ title: 'Overwrite .txt files?', message: `Every one of the ${fmtInt(total)} images gets its .txt sidecar replaced with the model's tags.`, confirmLabel: 'Overwrite', variant: 'destructive' });
      if (!ok) return;
    }
    if (audit) {
      const ok = await confirmDialog({ title: 'Run an audit batch?', message: `Audit mode DELETES sidecar tags the model judges erroneous, on ${plural(total, 'image')}. Rejected tags are removed from each .txt file.`, confirmLabel: 'Start audit', variant: 'amber' });
      if (!ok) return;
    }
    try {
      await rpc('inquiry.batch_start', {
        task,
        prompt: options.batch_prompt || '',
        source_keys: [...selectedKeys],
        include_transcripts: Boolean(options.batch_include_prior_transcripts),
        carry_context: Boolean(options.batch_carry_context),
        use_cache: Boolean(options.batch_use_cache),
        txt_mode: txtMode,
      });
    } catch (err) {
      reportError(err);
    }
  };

  return html`<div class="scroll grow col" style=${{ minHeight: 0 }}>
    <div class="col gap10 block tight" style=${{ paddingTop: '14px' }}>
      <div><div class="label" style=${{ marginBottom: '7px' }}>Task</div>
        <${TaskChips} value=${task} tasks=${tasks} disabled=${running} onChange=${(t) => setOptions({ batch_task: t })} /></div>
      <${TextField} multiline rows=${2} value=${options.batch_prompt || ''} placeholder="Optional prompt/question for all images in batch."
        onCommit=${(v) => setOptions({ batch_prompt: v })} />
    </div>
    <${ReasoningControl} />
    <div class="block tight">
      <div class="label-row" style=${{ marginBottom: '8px' }}><div class="label">Context sources</div><div class="note" style=${{ color: 'var(--tx7)' }}>images</div></div>
      <div class="col gap4">
        ${scanning ? html`<div class="note row gap6"><span class="spinner"></span>scanning the queue for prior results…</div>` : null}
        ${(sources || []).map((src) => html`<button class=${`srcrow ${selectedKeys.has(src.source_key) ? 'on' : ''}`} key=${src.source_key} onClick=${() => toggleSource(src.source_key)}>
          <span class="box"></span><span class="grow ellipsis" title=${src.model_name}>${src.display || String(src.model_name || '').split('/').pop()} (${src.model_type})</span>
          <span class="aside">${fmtInt(src.count)}</span></button>`)}
        ${sources && !sources.length && !scanning ? html`<div class="note">No prior results in this queue yet.</div>` : null}
      </div>
      <div class="col gap6" style=${{ marginTop: '9px' }}>
        <${Check} checked=${Boolean(options.batch_include_prior_transcripts)} onChange=${(v) => setOptions({ batch_include_prior_transcripts: v })}>Include prior inquiry transcripts</${Check}>
        <${Check} checked=${Boolean(options.batch_carry_context)} onChange=${(v) => setOptions({ batch_carry_context: v })}>Carry context across batch images</${Check}>
        <${Check} checked=${Boolean(options.batch_use_cache)} aside="temp 0" disabled=${Boolean(options.batch_carry_context)}
          title=${options.batch_carry_context ? 'Exact-match cache is bypassed while context carries across images.' : 'Reuse a stored answer when every input matches exactly.'}
          onChange=${(v) => setOptions({ batch_use_cache: v })}>Use exact-match cache</${Check}>
      </div>
    </div>
    <div class="block tight">
      <div class="label" style=${{ marginBottom: '8px' }}>.txt output</div>
      <${Seg} class=${audit ? 'locked' : ''} value=${txtMode} disabled=${audit || running}
        onChange=${(v) => setOptions({ txt_output_mode: v })} options=${['none', 'merge', 'overwrite']} />
      ${audit ? html`<div class="note warn" style=${{ marginTop: '7px' }}>locked — audit always mutates sidecars by deleting rejected tags</div>` : null}
    </div>
    <div class="spacer"></div>
    <div class="row gap8" style=${{ padding: '12px', borderTop: '1px solid var(--line)' }}>
      <${Btn} class="grow" size="tall" variant=${running ? 'fill' : 'primary'} disabled=${!model || !total || running || singleRunning} onClick=${start}
        title=${model ? '' : 'Load a llama model first'}>Start Batch Inquiry</${Btn}>
      <${Btn} size="tall" variant="danger" disabled=${!running || batch.cancelling} onClick=${() => rpc('inquiry.batch_cancel').catch(reportError)}>
        ${batch.cancelling ? 'Cancelling…' : 'Cancel'}</${Btn}>
    </div>
  </div>`;
}

// ── Transcripts ────────────────────────────────────────────────────────────────

function Elapsed({ since }) {
  const [, setTick] = useState(0);
  useInterval(() => setTick((t) => t + 1), 500);
  return html`${fmtDuration((Date.now() - since) / 1000)}`;
}

function TranscriptHeader({ title, meta, showRaw, onToggleRaw }) {
  return html`<div class="row gap12" style=${{ height: '34px', padding: '0 16px', borderBottom: '1px solid var(--line)', flex: 'none' }}>
    <div class="label">${title}</div>
    ${meta ? html`<div class="note ellipsis" style=${{ fontSize: '10.5px', color: 'var(--tx7)' }}>${meta}</div>` : null}
    <div class="spacer"></div>
    <button class="btn link" style=${{ fontSize: '10.5px', color: 'var(--blue-tx)' }} onClick=${onToggleRaw}>${showRaw ? 'Hide raw response' : 'Show raw response'}</button>
  </div>`;
}

/** Where a pending turn is: before the first token, thinking, or answering. */
function phase(pending, idle) {
  if (pending.stream) return 'streaming';
  if (pending.thinkingEndedAt) return 'answering';
  return pending.thinking ? 'thinking' : idle;
}

function fmtThought(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '';
  return seconds < 10 ? `${seconds.toFixed(1)}s` : fmtDuration(seconds);
}

/** The image a message is about, beside its text; opens advanced inspection. */
function TurnImage({ path }) {
  if (!path) return null;
  return html`<button class="turn-img" title=${`${basename(path)} · open advanced inspection`} onClick=${() => openInspect(path)}>
    <${Img} src=${thumbUrl(path, 320)} alt=${basename(path)} />
  </button>`;
}

/**
 * The model's thinking: streamed live ahead of the answer, then folded into
 * "Thought for …" once the answer begins. Kept for this session only.
 */
function Thinking({ text, live = false, startedAt, endedAt, seconds }) {
  const [open, setOpen] = useState(false);
  const bodyRef = useRef(null);
  const pinned = useRef(true);
  const streaming = live && !endedAt;
  useEffect(() => {
    const el = bodyRef.current;
    if (el && streaming && pinned.current) el.scrollTop = el.scrollHeight;
  }, [text, streaming]);
  if (!text) return null;
  if (streaming) {
    // The status line above already says "thinking"; this is just the stream.
    return html`<div class="thinking live">
      <div class="th-body" ref=${bodyRef} title="The model's thinking, streamed as it happens"
        onScroll=${(e) => { const el = e.currentTarget; pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24; }}>${text}</div>
    </div>`;
  }
  const secs = seconds != null ? seconds : (startedAt && endedAt ? (endedAt - startedAt) / 1000 : null);
  return html`<div class="thinking">
    <button class="th-head toggle" aria-expanded=${open} onClick=${() => setOpen((o) => !o)}>
      <span class="caret">${open ? '▾' : '▸'}</span>${secs ? `Thought for ${fmtThought(secs)}` : 'Thinking'}
    </button>
    ${open ? html`<div class="th-body">${text}</div>
      <div class="note" style=${{ marginTop: '5px', color: 'var(--tx7)' }}>kept for this session; not saved with the turn</div>` : null}
  </div>`;
}

function TurnTags({ turn }) {
  if (turn.task === 'audit') {
    const removed = turn.removed?.length ? turn.removed : turn.delete_tags || [];
    const kept = turn.remaining?.length || turn.tags.length;
    return html`<div class="sect"><div class="label" style=${{ marginBottom: '7px' }}>Audit · ${plural(removed.length, 'rejection')}</div>
      <div class="row" style=${{ flexWrap: 'wrap', gap: '5px' }}>
        ${removed.map((t) => html`<span class="tagchip struck">${t}</span>`)}
        <span class="tagchip plain">${fmtInt(kept)} kept</span>
      </div></div>`;
  }
  if (!turn.tags.length) return null;
  return html`<div class="sect"><div class="label" style=${{ marginBottom: '7px' }}>Extracted tags · ${fmtInt(turn.tags.length)}</div>
    <div class="row" style=${{ flexWrap: 'wrap', gap: '5px' }}>${turn.tags.map((t) => html`<span class="tagchip">${t}</span>`)}</div></div>`;
}

function ModelBubble({ turn }) {
  const text = turn.unusual ? (turn.raw || turn.comment || '[raw]') : (turn.comment || '[no comment]');
  return html`<div class=${`bubble-m with-img ${turn.unusual ? 'unusual' : ''}`}>
    <div class="bm-main">
      <div class="meta"><span title=${turn.model}>${turn.model_display || String(turn.model || '').split('/').pop()}</span>
        ${turn.elapsed != null ? html`<span class="dim">${turn.elapsed}s</span>` : null}
        ${turn.unusual ? html`<span style=${{ color: 'var(--red-tx)' }}>non-JSON response</span>` : null}
        ${turn.loop_retried ? html`<span class="dim" style=${{ color: 'var(--amber-tx)' }} title="The first attempt got stuck repeating itself; this answer comes from the retry with the model's recommended sampling.">retried after a loop</span>` : null}
      </div>
      <${Thinking} text=${turn.thinking} seconds=${turn.thinking_seconds} />
      <div class="txt" style=${turn.thinking ? { marginTop: '8px' } : null}>${text}</div>
      <${TurnTags} turn=${turn} />
      ${turn.reasoning ? html`<div class="reason">reasoning summary · ${turn.reasoning}</div>` : null}
    </div>
    <${TurnImage} path=${turn.image_path} />
  </div>`;
}

function UserBubble({ turn, index, pending }) {
  return html`<div class="bubble-u">
    <div class="meta">turn ${index} · ${turn.task}${pending ? ' · sending…' : ''}${turn.context_tables ? ` · ${plural(turn.context_tables, 'source')}` : ''}${turn.context_transcripts ? ` · ${turn.context_transcripts} prior turns` : ''}</div>
    ${turn.prompt ? html`<div class="txt">${turn.prompt}</div>` : html`<div class="txt" style=${{ color: 'var(--tx4)' }}>${turn.summary || turn.task}</div>`}
  </div>`;
}

function useStickToBottom(dep) {
  const ref = useRef(null);
  const atBottom = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [dep]);
  const onScroll = (e) => {
    const el = e.currentTarget;
    atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };
  return [ref, onScroll];
}

function SingleTranscript() {
  const { single, task, model } = useStore((s) => ({ single: s.inquiry.single, task: s.inquiry.options.single_task, model: s.inquiry.model }));
  const turns = single.turns;
  const [ref, onScroll] = useStickToBottom(`${turns.length}-${single.stream?.length}-${single.thinking?.length}-${single.pending ? 1 : 0}`);
  const info = single.info;
  const meta = info ? `session img:${shortHash(info.hash)} · ${plural(turns.length, 'turn')} · persisted` : '';
  return html`<div class="grow col" style=${{ background: 'var(--bg)', minWidth: 0 }}>
    <${TranscriptHeader} title="Transcript" meta=${meta} showRaw=${single.showRaw}
      onToggleRaw=${() => patch('inquiry', (q) => ({ single: { ...q.single, showRaw: !q.single.showRaw } }))} />
    <div class="scroll grow col gap12" style=${{ padding: '16px' }} ref=${ref} onScroll=${onScroll}>
      ${!model && !turns.length ? html`<div class="empty"><div class="big">No llama model loaded</div>
        <div>Load a multimodal GGUF from the GGUF sheet, then ask about the selected image.</div>
        <${Btn} onClick=${() => patch('inquiry', { panel: 'gguf' })}>Open GGUF settings</${Btn}></div>` : null}
      ${model && !turns.length && !single.pending ? html`<div class="empty"><div>No turns for this image yet.</div><div class="dim">Pick a task and send an inquiry.</div></div>` : null}
      ${turns.map((turn, i) => html`<${UserBubble} turn=${turn} index=${i + 1} /><${ModelBubble} turn=${turn} />`)}
      ${single.pending ? html`<${UserBubble} turn=${single.pending} index=${turns.length + 1} pending />
        <div class="bubble-m with-img">
          <div class="bm-main">
            <div class="row gap10"><${Dot} kind="run" /><span class="mono" style=${{ fontSize: '11px', color: 'var(--tx4)' }}>
              ${phase(single, 'generating')} · <${Elapsed} since=${single.startedAt || Date.now()} /> · ${single.notice ? 'retry after a loop' : 'attempt 1 of 2 (120s, then 300s retry)'}</span></div>
            ${single.notice ? html`<div class="note warn" style=${{ marginTop: '7px' }}>↻ ${single.notice}</div>` : null}
            <${Thinking} text=${single.thinking} live startedAt=${single.thinkingStartedAt} endedAt=${single.thinkingEndedAt} seconds=${single.thinkingSeconds} />
            ${single.stream ? html`<div class="txt" style=${{ marginTop: '8px', color: 'var(--tx3)' }}>${single.stream}</div>` : null}
          </div>
          <${TurnImage} path=${single.pending.image_path} />
        </div>` : null}
      ${single.error ? html`<div class="bubble-m with-img unusual">
        <div class="bm-main">
          <div class="meta" style=${{ color: 'var(--red-tx)' }}>inquiry failed</div>
          <div class="txt">${single.error.message}</div>
          <${Thinking} text=${single.error.thinking} seconds=${single.error.thinking_seconds} />
          ${single.error.logs ? html`<details style=${{ marginTop: '8px' }}><summary class="note" style=${{ cursor: 'pointer' }}>recent llama logs</summary><pre class="logbox" style=${{ maxHeight: '200px', marginTop: '6px' }}>${single.error.logs}</pre></details>` : null}
        </div>
        <${TurnImage} path=${single.error.turn?.image_path} />
      </div>` : null}
      <div class="spacer"></div>
      ${task === 'audit' ? html`<div class="banner amber" style=${{ alignItems: 'center' }}>
        <${Dot} kind="warn" /><div class="mono grow">audit mode will DELETE sidecar tags the model judges erroneous.</div>
      </div>` : null}
    </div>
    ${single.showRaw ? html`<div style=${{ borderTop: '1px solid var(--line)', padding: '10px 16px', background: 'var(--panel)', flex: 'none' }}>
      <pre class="rawbox" style=${{ maxHeight: '220px' }}>${single.raw || 'Raw llama response will appear here.'}</pre></div>` : null}
  </div>`;
}

function BatchCard({ card, root }) {
  if (card.state === 'live') {
    return html`<div class="tcard live with-img">
      <div class="tc-main">
        <div class="hd"><${Dot} kind="run" /><div class="f">${relPath(card.path, root)}</div>
          <div class="m" style=${{ color: 'var(--blue-tx)' }}>${phase(card, 'waiting')} · <${Elapsed} since=${card.started} /></div></div>
        ${card.notice ? html`<div class="note warn" style=${{ marginBottom: '6px' }}>↻ ${card.notice}</div>` : null}
        <${Thinking} text=${card.thinking} live startedAt=${card.thinkingStartedAt} endedAt=${card.thinkingEndedAt} seconds=${card.thinkingSeconds} />
        ${card.stream ? html`<div class="body" style=${card.thinking ? { marginTop: '8px' } : null}>${card.stream}</div>`
          : card.thinking ? null : html`<div class="sub">${card.turn?.summary || ''}</div>`}
      </div>
      <${TurnImage} path=${card.path} />
    </div>`;
  }
  if (card.state === 'error') {
    return html`<div class="tcard err with-img">
      <div class="tc-main">
        <div class="hd"><div class="f">${relPath(card.path, root)}</div><div class="m" style=${{ color: 'var(--red-tx)' }}>failed</div></div>
        <div class="sub" style=${{ whiteSpace: 'pre-wrap' }}>${card.error}</div>
        <div class="sub">sidecar untouched</div>
      </div>
      <${TurnImage} path=${card.path} />
    </div>`;
  }
  const turn = card.turn;
  const audit = turn.task === 'audit';
  const removed = turn.removed || [];
  return html`<div class="tcard with-img">
    <div class="tc-main">
      <div class="hd"><div class="f">${relPath(card.path, root)}</div>
        <div class="m">${turn.task} · ${turn.cached ? 'cache hit' : `${turn.elapsed}s`}</div>
        ${turn.unusual ? html`<div class="m" style=${{ color: 'var(--red-tx)' }}>non-JSON</div>` : null}
        ${turn.loop_retried ? html`<div class="m" style=${{ color: 'var(--amber-tx)' }} title="The first attempt got stuck repeating itself; this answer comes from the retry with the model's recommended sampling.">retried after a loop</div>` : null}</div>
      ${turn.thinking ? html`<div style=${{ marginBottom: '7px' }}><${Thinking} text=${turn.thinking} seconds=${turn.thinking_seconds} /></div>` : null}
      ${audit ? html`<div class="row" style=${{ flexWrap: 'wrap', gap: '5px' }}>
          ${removed.map((t) => html`<span class="tagchip struck">${t}</span>`)}
          ${removed.length ? html`<span class="tagchip plain">${fmtInt(turn.tags.length)} kept</span>` : html`<span class="sub">no rejections · ${fmtInt(turn.tags.length)} kept</span>`}
        </div>`
        : html`${turn.comment ? html`<div class="body">${turn.comment}</div>` : null}
          ${turn.tags.length ? html`<div class="row" style=${{ flexWrap: 'wrap', gap: '5px', marginTop: turn.comment ? '8px' : 0 }}>
            ${turn.tags.slice(0, 40).map((t) => html`<span class="tagchip">${t}</span>`)}
            ${turn.tags.length > 40 ? html`<span class="tagchip plain">+${turn.tags.length - 40}</span>` : null}</div>` : null}`}
    </div>
    <${TurnImage} path=${card.path} />
  </div>`;
}

function BatchTranscript() {
  const { batch, root } = useStore((s) => ({ batch: s.inquiry.batch, root: s.dir.path }));
  const p = batch.progress;
  const last = batch.cards[batch.cards.length - 1];
  const [ref, onScroll] = useStickToBottom(`${batch.cards.length}-${last?.stream?.length || 0}-${last?.thinking?.length || 0}`);
  const tags = p?.tags || [];
  return html`<div class="grow col" style=${{ background: 'var(--bg)', minWidth: 0 }}>
    ${p ? html`<div class="row gap14" style=${{ padding: '12px 16px', borderBottom: '1px solid var(--line)', background: 'var(--raised)', flex: 'none' }}>
      <div class="mono" style=${{ font: '700 17px/1 var(--mono)', color: 'var(--tx1)' }}>${fmtInt(p.processed)}<span style=${{ fontWeight: 400, fontSize: '11px', color: 'var(--tx6)' }}> / ${fmtInt(p.total)}</span></div>
      <div class="segbar sm grow">
        <div class="s-done" style=${{ width: `${pct(p.done, p.total)}%` }}></div>
        <div class="s-cached" style=${{ width: `${pct(p.cached, p.total)}%` }}></div>
        ${batch.running ? html`<div class="s-run" style=${{ width: `${Math.max(0.5, pct(1, p.total))}%` }}></div>` : null}
        <div class="s-fail" style=${{ width: `${pct(p.failed, p.total)}%` }}></div>
      </div>
      <div class="mono" style=${{ fontSize: '10.5px', color: 'var(--tx3)' }}>${batch.running ? `${p.rate ? `${p.rate.toFixed(2)} img/s` : '—'} · eta ${fmtDuration(p.eta)}` : batch.lastRun ? (batch.lastRun.cancelled ? 'cancelled' : 'finished') : ''}</div>
    </div>` : null}
    <${TranscriptHeader} title="Batch transcript" showRaw=${batch.showRaw}
      onToggleRaw=${() => patch('inquiry', (q) => ({ batch: { ...q.batch, showRaw: !q.batch.showRaw } }))} />
    <div class="scroll grow col gap10" style=${{ padding: '14px 16px' }} ref=${ref} onScroll=${onScroll}>
      ${batch.cards.length ? batch.cards.map((c) => html`<${BatchCard} card=${c} root=${root} key=${c.path} />`)
        : html`<div class="empty"><div class="big">No batch has run yet</div><div>Cards stream in here as each image returns: extracted tags, audit rejections and failures.</div></div>`}
    </div>
    ${batch.showRaw ? html`<div style=${{ borderTop: '1px solid var(--line)', padding: '10px 16px', background: 'var(--panel)', flex: 'none' }}>
      <pre class="rawbox" style=${{ maxHeight: '200px' }}>${batch.raw || 'Raw llama response will appear here.'}</pre></div>` : null}
    <div style=${{ height: '120px', flex: 'none', borderTop: '1px solid var(--line)', background: 'var(--panel)', padding: '9px 16px', overflow: 'auto' }}>
      <div class="label" style=${{ marginBottom: '8px' }}>Batch tags (real-time)</div>
      <div class="grid3">
        ${tags.map(([tag, count]) => html`<div class="gtag"><span class="ellipsis">${tag}</span><span>${fmtInt(count)}</span></div>`)}
        ${p?.task === 'audit' && p.rejected ? html`<div class="gtag red"><span>rejected</span><span>${fmtInt(p.rejected)}</span></div>` : null}
      </div>
      ${!tags.length ? html`<div class="help">Tag frequencies across the batch appear here.</div>` : null}
    </div>
  </div>`;
}
