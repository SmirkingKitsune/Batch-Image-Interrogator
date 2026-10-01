// The managed llama.cpp runtime card, shared by Inquiry (GGUF sheet) and
// Settings → llama.cpp runtime. It reads provisioner state, so it shows what
// is installed — including a CPU fallback — rather than what was configured.
import { html } from '../lib.js';
import { native, rpc } from '../api.js';
import { patch, useStore } from '../store.js';
import { openModal } from '../state.js';
import { Btn, Dot, reportError, toast } from '../ui.js';

export function runtimeDetail(summary) {
  if (!summary) return '';
  const parts = [];
  if (summary.short_version) parts.push(summary.short_version);
  if (summary.method) parts.push(`via ${summary.method}`);
  if (summary.cuda_architectures) parts.push(`arch ${summary.cuda_architectures}`);
  if (summary.installed_short) parts.push(`installed ${summary.installed_short}`);
  return parts.join(' · ') || summary.executable;
}

export async function checkForUpdates() {
  patch('runtime', { checkingUpdate: true, update: null });
  try {
    await rpc('runtime.check_updates');
  } catch (err) {
    patch('runtime', { checkingUpdate: false });
    reportError(err);
  }
}

export async function checkHealth() {
  patch('runtime', { checkingHealth: true });
  try {
    const result = await rpc('runtime.health');
    if (result && result.message) patch('runtime', { health: result, checkingHealth: false });
  } catch (err) {
    patch('runtime', { checkingHealth: false });
    reportError(err);
  }
}

export function openLogFolder(summary) {
  if (!summary?.log_dir) return;
  if (native.available) native.openFolder(summary.log_dir);
  else toast(`Logs: ${summary.log_dir}`, 'info', 8000);
}

const LEVEL_COLOR = { ok: 'var(--green-tx)', warn: 'var(--amber-tx)', error: 'var(--red-tx)', muted: 'var(--tx5)' };

export function RuntimeCard({ summary, compact = false, onModeChange, mode }) {
  const { update, checkingUpdate, provisioning } = useStore((s) => ({
    update: s.runtime.update, checkingUpdate: s.runtime.checkingUpdate, provisioning: s.runtime.provision.running,
  }));
  if (!summary) return html`<div class="note row gap6"><span class="spinner"></span>Reading runtime state…</div>`;

  if (!summary.installed) {
    return html`<div class="col gap6">
      <div class="row gap8"><${Dot} kind="err" /><div style=${{ font: `600 ${compact ? 12 : 15}px/1 var(--sans)`, color: 'var(--red-tx)' }}>No runtime installed</div></div>
      <div class="note" style=${{ fontSize: '10.5px', lineHeight: 1.6 }}>Detected accelerator: ${summary.detected}. Installing fetches a matched release, or builds from source when none is published.</div>
      <div class="row gap6" style=${{ marginTop: '4px' }}>
        <${Btn} size=${compact ? 'sm' : ''} variant="primary" busy=${provisioning} onClick=${() => openModal('provision', {})}>${provisioning ? 'Installing…' : 'Install Runtime'}</${Btn}>
      </div>
    </div>`;
  }

  const fallback = Boolean(summary.fallback_from);
  const color = summary.is_gpu ? 'var(--green)' : 'var(--amber)';
  return html`<div class="col">
    <div class="row gap8" style=${{ marginBottom: compact ? '4px' : '6px' }}>
      <${Dot} kind=${summary.is_gpu ? 'ok' : 'warn'} large=${!compact} />
      <div style=${{ font: `600 ${compact ? 12 : 15}px/1 var(--sans)`, color }}>Ready — ${summary.accelerator || 'unknown'}</div>
      ${!compact && onModeChange ? html`<div class="spacer"></div>
        <div class="seg inline"><button class=${mode !== 'custom' ? 'on' : ''} onClick=${() => onModeChange('managed')}>managed</button><button class=${mode === 'custom' ? 'on' : ''} onClick=${() => onModeChange('custom')}>custom</button></div>` : null}
    </div>
    <div class="note selectable" style=${{ fontSize: compact ? '10.5px' : '11px', lineHeight: 1.6, color: 'var(--tx5)' }} title=${summary.executable}>${runtimeDetail(summary)}</div>
    ${fallback ? html`<div class="banner amber" style=${{ marginTop: '10px', fontSize: compact ? '10.5px' : '11px' }}>
      <div>Running on CPU although this machine supports <span class="mono">${summary.fallback_from}</span>. Inference will be far slower. Reinstall to retry the GPU build.</div>
    </div>` : null}
    <div class="row gap6" style=${{ marginTop: compact ? '9px' : '12px', flexWrap: 'wrap', gap: compact ? '6px' : '8px' }}>
      <${Btn} size=${compact ? 'sm' : ''} variant=${fallback && !compact ? 'amber' : ''} busy=${provisioning} disabled=${provisioning} onClick=${() => openModal('provision', {})}>
        ${provisioning ? 'Installing…' : 'Reinstall / Update'}</${Btn}>
      <${Btn} size=${compact ? 'sm' : ''} busy=${checkingUpdate} disabled=${checkingUpdate} onClick=${checkForUpdates}>Check for Updates</${Btn}>
      ${compact ? null : html`<${ManageExtras} summary=${summary} />`}
    </div>
    ${compact && update ? html`<div class="note" style=${{ marginTop: '7px', color: LEVEL_COLOR[update.level] || 'var(--tx5)' }}>${update.text}</div>` : null}
  </div>`;
}

function ManageExtras({ summary }) {
  const checkingHealth = useStore((s) => s.runtime.checkingHealth);
  return html`
    <${Btn} onClick=${() => openModal('provision', { advanced: true })}>Manage…</${Btn}>
    <${Btn} busy=${checkingHealth} disabled=${checkingHealth} onClick=${checkHealth}>Check Health</${Btn}>
    <${Btn} onClick=${() => openLogFolder(summary)}>Open Log Folder</${Btn}>`;
}

export function updateLevelColor(level) {
  return LEVEL_COLOR[level] || 'var(--tx5)';
}
