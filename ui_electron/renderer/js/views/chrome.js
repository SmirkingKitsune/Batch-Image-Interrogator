// Title bar and tab strip with always-on status chips.
import { html, useEffect, useState } from '../lib.js';
import { native } from '../api.js';
import { useStore } from '../store.js';
import { setTab } from '../state.js';
import { Dot } from '../ui.js';
import { fmtDuration, fmtGB, fmtInt, fmtRate, modelShort } from '../util.js';

export function gpuShort(name) {
  if (!name) return '';
  return String(name).replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '').replace(/^AMD\s+/i, '');
}

export function TitleBar() {
  const { path, tab, interrog, inquiryBatch } = useStore((s) => ({
    path: s.dir.path,
    tab: s.tab,
    interrog: s.interrog.running ? s.interrog.progress : null,
    inquiryBatch: s.inquiry.batch.running ? s.inquiry.batch.progress : null,
  }));
  const [maximized, setMaximized] = useState(false);
  useEffect(() => {
    native.isMaximized().then((m) => setMaximized(Boolean(m)));
    return native.onWindowState((state) => setMaximized(Boolean(state.maximized)));
  }, []);

  let activity = null;
  if (interrog && tab !== 'interrogation') {
    activity = `interrogating · ${fmtInt(interrog.processed)}/${fmtInt(interrog.total)} · ${fmtRate(interrog.rate)} · eta ${fmtDuration(interrog.eta)}`;
  } else if (inquiryBatch && tab !== 'inquiry') {
    activity = `inquiry · ${fmtInt(inquiryBatch.processed)}/${fmtInt(inquiryBatch.total)} · eta ${fmtDuration(inquiryBatch.eta)}`;
  }

  return html`<div class="titlebar" onDblClick=${(e) => { if (e.target === e.currentTarget) native.toggleMaximize(); }}>
    <img src="/assets/icon.png" alt="" />
    <div class="title">Image Interrogator</div>
    ${path ? html`<div class="path ellipsis">— ${path}</div>` : null}
    <div class="spacer"></div>
    ${activity ? html`<div class="activity"><${Dot} kind="run" /><span>${activity}</span></div>` : null}
    <div class="spacer"></div>
    ${native.available ? html`<div class="winctl">
      <button title="Minimize" onClick=${() => native.minimize()}><span class="min"></span></button>
      <button title=${maximized ? 'Restore' : 'Maximize'} onClick=${() => native.toggleMaximize()}><span class=${`max ${maximized ? 'restored' : ''}`}></span></button>
      <button class="close" title="Close" onClick=${() => native.close()}>✕</button>
    </div>` : html`<div style=${{ width: '12px' }}></div>`}
  </div>`;
}

const TABS = [
  ['interrogation', 'Interrogation'],
  ['inquiry', 'Inquiry'],
  ['gallery', 'Gallery'],
  ['settings', 'Database / Settings'],
];

export function TabStrip() {
  const { tab, inquiring, interrogating } = useStore((s) => ({
    tab: s.tab,
    inquiring: s.activity.inquiring,
    interrogating: s.activity.interrogating,
  }));
  return html`<div class="tabstrip">
    ${TABS.map(([id, label]) => html`<button class=${`tab ${tab === id ? 'active' : ''}`} onClick=${() => setTab(id)}>
      ${label}
      ${(id === 'inquiry' && inquiring) || (id === 'interrogation' && interrogating) ? html`<span class="dot pulse"></span>` : null}
    </button>`)}
    <${StatusChips} tab=${tab} />
  </div>`;
}

function GpuChip() {
  const hw = useStore((s) => s.hw);
  if (!hw.gpu_name && !hw.total) return null;
  const pctUsed = hw.total ? Math.round((hw.used / hw.total) * 100) : 0;
  return html`<div class="chip" title=${hw.unified ? 'Unified memory: the GPU shares system RAM' : 'GPU memory in use'}>
    <span>${gpuShort(hw.gpu_name) || 'GPU'}</span>
    ${hw.total ? html`<div class="vram"><div style=${{ width: `${pctUsed}%` }}></div></div>
      <b>${fmtGB(hw.used)}/${fmtGB(hw.total)} GB${hw.unified ? ' unified' : ''}</b>` : null}
  </div>`;
}

function DeviceChip() {
  const hw = useStore((s) => s.hw);
  if (hw.torch_cuda) return html`<div class="chip"><${Dot} kind="ok" /><b>CUDA ${hw.cuda_version || ''}</b></div>`;
  return html`<div class="chip" title=${hw.torch_error || ''}><${Dot} kind="warn" /><b class="warn">CPU</b></div>`;
}

function ModelChip() {
  const { model, loading, loadingType } = useStore((s) => ({ model: s.interrog.model, loading: s.interrog.loading, loadingType: s.interrog.loadingType }));
  if (loading) return html`<div class="chip"><${Dot} kind="run" /><b>loading ${loadingType || 'model'}…</b></div>`;
  if (!model) return html`<div class="chip"><${Dot} /><span>no model loaded</span></div>`;
  return html`<div class="chip" title=${model.label}><${Dot} kind="ok" /><b>${model.type} · ${model.short}</b></div>`;
}

function RuntimeChip({ compact = false }) {
  const summary = useStore((s) => s.runtime.summary);
  if (!summary) return null;
  if (!summary.installed) return html`<div class="chip"><${Dot} kind="err" /><b>llama.cpp · not installed</b></div>`;
  if (summary.fallback_from) {
    return html`<div class="chip"><${Dot} kind="warn" /><b class="warn">llama.cpp · ${summary.accelerator} (fallback)</b></div>`;
  }
  const kind = summary.is_gpu ? 'ok' : 'warn';
  return html`<div class="chip"><${Dot} kind=${kind} /><b>llama.cpp · ${summary.accelerator || 'unknown'}${compact ? '' : ` · ${summary.short_version || ''}`}</b></div>`;
}

function ServerChip() {
  const { model, loading } = useStore((s) => ({ model: s.inquiry.model, loading: s.inquiry.loading }));
  if (loading) return html`<div class="chip"><${Dot} kind="run" /><b>starting llama-server…</b></div>`;
  if (!model) return html`<div class="chip"><span>server stopped</span></div>`;
  const host = (model.url || '').replace(/^https?:\/\//, '');
  return html`<div class="chip" title=${model.log_path || ''}>
    <b>${host}</b>
    ${model.requested_port && model.port !== model.requested_port ? html`<span class="warn">${model.requested_port} busy</span>` : null}
  </div>`;
}

function IdleChip() {
  const { hw, model, running } = useStore((s) => ({ hw: s.hw, model: s.interrog.model, running: s.interrog.running }));
  const device = hw.torch_cuda ? 'CUDA' : 'CPU';
  const state = running ? 'interrogating' : 'idle';
  return html`<div class="chip"><${Dot} kind=${hw.torch_cuda ? 'ok' : 'warn'} />
    <span>${device}${hw.gpu_name ? ` · ${gpuShort(hw.gpu_name)}` : ''} · ${state} · ${model ? `${model.type} ${modelShort(model.label)}` : 'no model loaded'}</span>
  </div>`;
}

function StatusChips({ tab }) {
  if (tab === 'inquiry') return html`<div class="chips"><${RuntimeChip} /><${GpuChip} /><${ServerChip} /></div>`;
  if (tab === 'gallery') return html`<div class="chips"><${IdleChip} /></div>`;
  if (tab === 'settings') return html`<div class="chips"><${RuntimeChip} compact /></div>`;
  return html`<div class="chips"><${DeviceChip} /><${GpuChip} /><${ModelChip} /></div>`;
}
