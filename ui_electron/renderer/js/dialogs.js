// Modal dialogs: advanced inspection (1g), organize (1i), first run (2d),
// runtime provisioning, database busy, about.
import { html, useEffect, useMemo, useRef, useState } from './lib.js';
import { native, rpc, thumbUrl } from './api.js';
import { getState, patch, setState, useStore } from './store.js';
import { chooseDirectory, closeModal, openDirectory, setTab } from './state.js';
import { Bar, Btn, Check, confirmDialog, Dot, Img, Modal, reportError, Seg, TextField, toast } from './ui.js';
import { basename, debounce, fmtBytes, fmtDateTime, fmtInt, modelShort, plural, relPath, shortHash } from './util.js';

export function DialogHost() {
  const { modals, busy } = useStore((s) => ({ modals: s.modals, busy: s.busy }));
  return html`
    ${modals.inspect ? html`<${InspectDialog} ...${modals.inspect} key=${modals.inspect.multi ? 'multi' : 'single'} />` : null}
    ${modals.organize ? html`<${OrganizeDialog} state=${modals.organize} />` : null}
    ${modals.firstRun ? html`<${FirstRunDialog} />` : null}
    ${modals.provision ? html`<${ProvisionDialog} options=${modals.provision} />` : null}
    ${modals.about ? html`<${AboutDialog} />` : null}
    ${busy.length ? html`<${BusyDialog} request=${busy[0]} />` : null}`;
}

// ── Advanced inspection (1g) ──────────────────────────────────────────────────

const STATUS = {
  in_both: { color: 'var(--green)', label: 'in both' },
  db_only: { color: 'var(--amber)', label: 'db only — will be written' },
  removed_by_filter: { color: 'var(--red)', label: 'removed by filter' },
  replaced: { color: 'var(--orange)', label: 'replaced' },
  file_only: { color: 'var(--blue)', label: 'file only — added by hand' },
  manually_added: { color: 'var(--blue)', label: 'kept by hand despite a filter' },
};

function Square({ color }) {
  return html`<span style=${{ width: '8px', height: '8px', borderRadius: '2px', background: color, flex: 'none', display: 'inline-block' }}></span>`;
}

function InspectDialog({ path, list, multi }) {
  if (multi) return html`<${CommonTagsDialog} paths=${multi} />`;
  return html`<${SingleInspect} initialPath=${path} list=${list} />`;
}

function SingleInspect({ initialPath, list }) {
  const { root, filters, llama } = useStore((s) => ({ root: s.dir.path, filters: s.filters, llama: s.inquiry.model }));
  const [path, setPath] = useState(initialPath);
  const [data, setData] = useState(null);
  const [tab, setTabName] = useState(null);
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const [editor, setEditor] = useState(null);
  const index = list.indexOf(path);

  const load = async (target) => {
    setError(null);
    try {
      const result = await rpc('inspect.image', { path: target });
      setData(result);
      setModel((current) => (result.models.some((m) => m.model_name === current) ? current : result.models[0]?.model_name || null));
      setTabName((current) => current || (result.models.length ? 'diff' : 'editor'));
      setEditor({ checked: new Set(result.editor.selected), extra: [] });
    } catch (err) {
      setError(err.message);
    }
  };
  useEffect(() => { setData(null); load(path); }, [path, filters]);

  const close = () => closeModal('inspect');
  const go = (delta) => {
    const next = index + delta;
    if (next >= 0 && next < list.length) setPath(list[next]);
  };
  const current = data?.models.find((m) => m.model_name === model) || null;

  const saveEditor = async () => {
    if (!data || !editor) return;
    const all = [...data.editor.all, ...editor.extra];
    const kept = data.file_tags.filter((t) => editor.checked.has(t));
    const added = all.filter((t) => editor.checked.has(t) && !data.file_tags.includes(t));
    try {
      await rpc('gallery.save_tags', { path, tags: [...kept, ...added] });
      toast(`Saved ${plural(kept.length + added.length, 'tag')}.`, 'ok', 2500);
      load(path);
    } catch (err) {
      reportError(err);
    }
  };
  const apply = async () => {
    if (!current) return;
    try {
      await rpc('inspect.apply', { path, model_name: current.model_name });
      toast(`Applied ${modelShort(current.model_name)} to ${basename(path).replace(/\.[^.]+$/, '')}.txt`, 'ok', 3000);
      load(path);
    } catch (err) {
      reportError(err);
    }
  };

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape' && !getState().confirm) { e.preventDefault(); close(); return; }
      if (e.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        if (tab === 'editor') saveEditor();
        else if (tab === 'diff') apply();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  });

  const tabs = [['results', 'Model Results'], ['diff', 'Database vs File'], ['editor', 'Tag Editor'], ['inquiry', 'Multimodal Inquiry']];
  return html`<div class="backdrop" onMouseDown=${(e) => { if (e.target === e.currentTarget) close(); }}>
    <div class="modal" style=${{ width: '1080px', height: 'min(760px, 100%)' }}>
      <div class="modal-head">
        <div class="t">Advanced Image Inspection</div>
        <div class="s ellipsis">${relPath(path, root)}${index >= 0 ? ` · ${fmtInt(index + 1)} of ${fmtInt(list.length)}` : ''}</div>
        <div class="spacer"></div>
        <div class="row gap6">
          <button class="iconbtn" disabled=${index <= 0} onClick=${() => go(-1)} title="Previous (←)">←</button>
          <button class="iconbtn" disabled=${index < 0 || index >= list.length - 1} onClick=${() => go(1)} title="Next (→)">→</button>
          <button class="iconbtn" onClick=${close} title="Close (Esc)">✕</button>
        </div>
      </div>
      <div class="mtabs">${tabs.map(([id, label]) => html`<button class=${tab === id ? 'on' : ''} onClick=${() => setTabName(id)}>${label}</button>`)}</div>
      <div class="row grow" style=${{ minHeight: 0, alignItems: 'stretch' }}>
        <div class="col gap12" style=${{ width: '340px', flex: 'none', borderRight: '1px solid var(--line)', padding: '14px', overflow: 'auto' }}>
          <div class="preview" style=${{ height: '260px', flex: 'none' }}><${Img} src=${thumbUrl(path, 800)} /></div>
          ${current?.ratings ? html`<div>
            <div class="label" style=${{ marginBottom: '9px' }}>WD sensitivity</div>
            <div class="col gap6" style=${{ gap: '7px' }}>
              ${[['general', 'green'], ['sensitive', 'amber'], ['questionable', 'red'], ['explicit', 'red']].map(([k, kind]) => html`<div class="row gap9" style=${{ gap: '9px' }}>
                <div class="mono" style=${{ width: '78px', fontSize: '10.5px', color: 'var(--tx3)' }}>${k}</div>
                <div class="grow"><${Bar} value=${(current.ratings[k] || 0) * 100} kind=${kind} large /></div>
                <div class="mono" style=${{ width: '32px', textAlign: 'right', fontSize: '10px', color: 'var(--tx3)' }}>${(current.ratings[k] || 0).toFixed(2)}</div>
              </div>`)}
            </div></div>` : null}
          <div class="col gap6" style=${{ paddingTop: '11px', borderTop: '1px solid var(--line)' }}>
            <div class="kvm"><span>hash</span><span>${shortHash(data?.meta?.hash)}</span></div>
            <div class="kvm"><span>size</span><span>${data ? `${fmtBytes(data.meta.size)}${data.meta.w ? ` · ${data.meta.w}×${data.meta.h}` : ''}` : '—'}</span></div>
            <div class="kvm"><span>models run</span><span class="ellipsis" style=${{ maxWidth: '200px' }}>${data?.models.length ? [...new Set(data.models.map((m) => (m.model_type === 'LlamaCpp' ? 'llama' : m.model_type)))].join(', ') : 'none'}</span></div>
            <div class="kvm"><span>last write</span><span>${data?.last_write ? fmtDateTime(data.last_write) : 'no .txt'}</span></div>
          </div>
        </div>
        <div class="grow col" style=${{ minWidth: 0 }}>
          ${error ? html`<div class="empty">${error}</div>` : !data ? html`<div class="empty"><span class="spinner"></span></div>`
            : tab === 'results' ? html`<${ResultsTab} data=${data} model=${model} setModel=${setModel} />`
            : tab === 'diff' ? html`<${DiffTab} data=${data} current=${current} setModel=${setModel} filters=${filters} />`
            : tab === 'editor' ? html`<${EditorTab} data=${data} editor=${editor} setEditor=${setEditor} />`
            : html`<${InquiryTab} path=${path} llama=${llama} />`}
        </div>
      </div>
      <div class="modal-foot">
        <div class="note" style=${{ fontSize: '10.5px' }}>←/→ navigate · ${native.platform === 'darwin' ? '⌘' : 'Ctrl+'}S save · esc close</div>
        <div class="spacer"></div>
        ${tab !== 'editor' ? html`<${Btn} onClick=${() => setTabName('editor')}>Open Tag Editor</${Btn}>` : null}
        ${tab === 'editor' ? html`<${Btn} variant="success" onClick=${saveEditor} disabled=${!data}>Save tags to .txt</${Btn}>` : null}
        ${tab === 'diff' || tab === 'results' ? html`<${Btn} variant="success" onClick=${apply} disabled=${!current || !current.plan.changes}
          title=${current && !current.plan.changes ? 'The .txt file already matches this result' : ''}>Apply to .txt</${Btn}>` : null}
      </div>
    </div>
  </div>`;
}

function ModelChips({ models, value, onChange }) {
  if (models.length <= 1) return null;
  return html`<div class="row gap4" style=${{ flexWrap: 'wrap' }}>
    ${models.map((m) => html`<button class=${`task ${value === m.model_name ? 'on' : ''}`} title=${m.model_name} onClick=${() => onChange(m.model_name)}>
      ${m.model_type === 'LlamaCpp' ? 'llama' : m.model_type} · ${modelShort(m.model_name).slice(0, 22)}</button>`)}
  </div>`;
}

function ResultsTab({ data, model, setModel }) {
  const current = data.models.find((m) => m.model_name === model);
  if (!current) return html`<div class="empty"><div class="big">No interrogations for this image</div><div>Run a tagger from the Interrogation tab, or tag it by hand in the editor.</div></div>`;
  const scores = current.confidence_scores || {};
  const rows = [...current.tags].sort((a, b) => (scores[b] ?? -1) - (scores[a] ?? -1));
  return html`<div class="col grow" style=${{ minHeight: 0 }}>
    <div class="row gap12" style=${{ padding: '12px 16px', borderBottom: '1px solid var(--line)', background: 'var(--panel)', flexWrap: 'wrap' }}>
      <${ModelChips} models=${data.models} value=${model} onChange=${setModel} />
      <div class="note">${current.model_name} · ${plural(current.tags.length, 'tag')}${current.interrogated_at ? ` · ${fmtDateTime(current.interrogated_at)}` : ''}</div>
    </div>
    <div class="scroll grow taglist" style=${{ padding: '8px 16px' }}>
      ${rows.map((tag) => html`<div class="trow" key=${tag}>
        <div class="t">${tag}</div>
        ${scores[tag] != null ? html`<${Bar} value=${scores[tag] * 100} /><div class="c">${Number(scores[tag]).toFixed(4)}</div>` : html`<div class="c" style=${{ width: 'auto' }}>—</div>`}
      </div>`)}
    </div>
  </div>`;
}

function normalizeForCompare(tag, underscores) {
  const lower = String(tag).toLowerCase();
  return underscores ? lower.replace(/_/g, ' ') : lower;
}

function DiffTab({ data, current, setModel, filters }) {
  if (!current) return html`<div class="empty"><div class="big">Nothing to compare</div><div>This image has no stored interrogation. The file column still shows what is on disk.</div></div>`;
  const underscores = filters.underscores;
  const prefix = new Set((data.prefix_tags || []).map((t) => normalizeForCompare(t, underscores)));
  const statusByTag = new Map();
  for (const row of current.comparison) {
    const key = normalizeForCompare(row.tag, underscores);
    if (!statusByTag.has(key) || row.status === 'in_both') statusByTag.set(key, row.status);
  }
  const scores = current.confidence_scores || {};
  const plan = current.plan;
  const summaryParts = [];
  if (plan.added.length) summaryParts.push(`add ${plural(plan.added.length, 'tag')} (${plan.added.slice(0, 4).join(', ')}${plan.added.length > 4 ? '…' : ''})`);
  if (plan.rewritten.length) summaryParts.push(`rewrite ${plan.rewritten.length} (${plan.rewritten.slice(0, 2).map(([a, b]) => `${a} → ${b}`).join(', ')})`);
  const manual = plan.kept_manual.length;

  return html`<div class="col grow" style=${{ minHeight: 0 }}>
    <div class="row gap14" style=${{ padding: '12px 16px', borderBottom: '1px solid var(--line)', background: 'var(--panel)', flexWrap: 'wrap', gap: '14px' }}>
      ${['in_both', 'db_only', 'removed_by_filter', 'replaced', 'file_only'].map((k) => html`<div class="row gap6 mono" style=${{ fontSize: '10.5px', color: 'var(--tx3)' }}><${Square} color=${STATUS[k].color} />${STATUS[k].label}</div>`)}
    </div>
    ${data.models.length > 1 ? html`<div style=${{ padding: '8px 16px', borderBottom: '1px solid var(--line-soft)' }}><${ModelChips} models=${data.models} value=${current.model_name} onChange=${setModel} /></div>` : null}
    <div class="row grow" style=${{ minHeight: 0, alignItems: 'stretch' }}>
      <div class="col grow" style=${{ minWidth: 0, borderRight: '1px solid var(--line-soft)' }}>
        <div class="row gap8" style=${{ padding: '10px 16px', borderBottom: '1px solid var(--line-soft)' }}><span class="label">Database · ${plural(current.tags.length, 'tag')}</span><span class="note" style=${{ color: 'var(--tx7)' }}>${modelShort(current.model_name)}</span></div>
        <div class="scroll grow" style=${{ padding: '6px 16px' }}>
          ${current.comparison.filter((r) => r.status !== 'file_only').map((r) => {
            const st = STATUS[r.status] || STATUS.in_both;
            let aside = r.confidence != null && r.confidence > 0 ? Number(r.confidence).toFixed(2) : '';
            if (r.status === 'replaced') aside = 'rule';
            else if (r.status === 'removed_by_filter') aside = filters.remove.includes(r.tag.toLowerCase()) ? 'remove' : (r.confidence ? `${Number(r.confidence).toFixed(2)} < thr` : 'filter');
            else if (current.model_type === 'LlamaCpp') aside = 'llama';
            return html`<div class="row gap9" style=${{ gap: '9px', padding: '5px 0' }} title=${st.label}>
              <${Square} color=${st.color} />
              <div class="grow mono ellipsis" style=${{ fontSize: '11px', color: r.status === 'removed_by_filter' ? 'var(--tx4)' : 'var(--tx2)', textDecoration: r.status === 'removed_by_filter' ? 'line-through' : 'none' }}>
                ${r.original_tag ? html`${r.original_tag} → <span style=${{ color: 'var(--orange-tx)' }}>${r.tag}</span>` : r.tag}</div>
              <div class="mono" style=${{ fontSize: '10px', color: 'var(--tx5)' }}>${aside}</div>
            </div>`;
          })}
        </div>
      </div>
      <div class="col grow" style=${{ minWidth: 0 }}>
        <div class="label" style=${{ padding: '10px 16px', borderBottom: '1px solid var(--line-soft)' }}>${basename(data.path).replace(/\.[^.]+$/, '')}.txt · ${plural(data.file_tags.length, 'tag')}</div>
        <div class="scroll grow" style=${{ padding: '6px 16px' }}>
          ${data.file_tags.map((tag) => {
            const key = normalizeForCompare(tag, underscores);
            const st = statusByTag.get(key);
            const isPrefix = prefix.has(key);
            const color = st === 'in_both' ? STATUS.in_both.color : st === 'replaced' ? STATUS.replaced.color : 'var(--blue)';
            const aside = isPrefix ? 'prefix' : st === 'in_both' || st === 'replaced' ? '' : st === 'manually_added' ? 'kept' : 'manual';
            return html`<div class="row gap9" style=${{ gap: '9px', padding: '5px 0' }}>
              <${Square} color=${color} /><div class="grow mono ellipsis" style=${{ fontSize: '11px', color: 'var(--tx2)' }}>${tag}</div>
              <div class="mono" style=${{ fontSize: '10px', color: 'var(--blue-tx)' }}>${aside}</div></div>`;
          })}
          ${!data.file_tags.length ? html`<div class="help" style=${{ padding: '6px 0' }}>No .txt file yet.</div>` : null}
          <div class="banner amber" style=${{ marginTop: '12px' }}><div class="mono">
            ${plan.changes
              ? `Writing now would ${summaryParts.join(', ') || 'reorder tags'}${manual ? ` and keep ${manual === 1 ? 'the' : manual === 2 ? 'both' : `all ${manual}`} manual ${manual === 1 ? 'tag' : 'tags'}` : ''}.`
              : 'The .txt file already matches this result after filters.'}
          </div></div>
        </div>
      </div>
    </div>
  </div>`;
}

function EditorTab({ data, editor, setEditor }) {
  const [adding, setAdding] = useState('');
  if (!editor) return null;
  const all = [...data.editor.all, ...editor.extra.filter((t) => !data.editor.all.includes(t))];
  const toggle = (tag) => {
    const checked = new Set(editor.checked);
    if (checked.has(tag)) checked.delete(tag); else checked.add(tag);
    setEditor({ ...editor, checked });
  };
  const add = () => {
    const tag = adding.trim();
    if (!tag) return;
    setEditor({ extra: all.includes(tag) ? editor.extra : [...editor.extra, tag], checked: new Set([...editor.checked, tag]) });
    setAdding('');
  };
  const count = all.filter((t) => editor.checked.has(t)).length;
  return html`<div class="col grow" style=${{ padding: '14px 16px', minHeight: 0 }}>
    <div class="label-row"><div class="label">Tag editor</div><div class="note">total ${fmtInt(all.length)} · selected ${fmtInt(count)}</div></div>
    <div class="scroll grow" style=${{ display: 'flex', flexWrap: 'wrap', gap: '5px', alignContent: 'flex-start' }}>
      ${all.map((tag) => html`<span class=${`tagchip toggle ${editor.checked.has(tag) ? 'on' : 'off'}`} key=${tag} onClick=${() => toggle(tag)}><span class="cb"></span>${tag}</span>`)}
    </div>
    <div class="row gap8" style=${{ marginTop: '10px' }}>
      <div class="grow"><${TextField} placeholder="Add a tag…" value=${adding} onInput=${setAdding} onEnter=${add} /></div>
      <${Btn} onClick=${add} disabled=${!adding.trim()}>Add</${Btn}>
      <${Btn} onClick=${() => setEditor({ ...editor, checked: new Set(all) })}>All</${Btn}>
      <${Btn} onClick=${() => setEditor({ ...editor, checked: new Set() })}>None</${Btn}>
    </div>
    <div class="note" style=${{ marginTop: '8px' }}>saves bypass filter rules — what is checked is what lands on disk</div>
  </div>`;
}

function InquiryTab({ path, llama }) {
  const { single, tasks } = useStore((s) => ({ single: s.inquiry.single, tasks: s.inquiry.tasks }));
  const [info, setInfo] = useState(null);
  const [task, setTask] = useState('describe');
  const [prompt, setPrompt] = useState('');
  const running = single.running && single.path === path;
  const reload = () => rpc('inquiry.image', { path }).then(setInfo).catch((err) => setInfo({ error: err.message }));
  useEffect(() => { reload(); }, [path, llama?.label, single.refresh]);
  if (!llama) {
    return html`<div class="empty"><div class="big">No llama model loaded</div><div>Load a multimodal model on the Inquiry tab to ask about this image here.</div>
      <${Btn} onClick=${() => { closeModal('inspect'); setTab('inquiry'); patch('inquiry', { panel: 'gguf' }); }}>Open Inquiry</${Btn}></div>`;
  }
  const send = async () => {
    try {
      patch('inquiry', (q) => ({ single: { ...q.single, path } }));
      await rpc('inquiry.send', { path, task, prompt, prior_indices: [], include_transcripts: false });
    } catch (err) {
      reportError(err);
    }
  };
  const turns = info?.history || [];
  return html`<div class="col grow" style=${{ minHeight: 0 }}>
    <div class="scroll grow col gap10" style=${{ padding: '14px 16px' }}>
      ${turns.length ? turns.map((t, i) => html`<div class="bubble-u"><div class="meta">turn ${i + 1} · ${t.task}</div><div class="txt">${t.prompt || t.summary}</div></div>
        <div class=${`bubble-m ${t.unusual ? 'unusual' : ''}`}><div class="meta">${t.model_display || modelShort(t.model)}</div><div class="txt">${t.unusual ? t.raw || t.comment : t.comment || '[no comment]'}</div>
        ${t.tags.length ? html`<div class="sect row" style=${{ flexWrap: 'wrap', gap: '5px' }}>${t.tags.map((x) => html`<span class="tagchip">${x}</span>`)}</div>` : null}</div>`)
        : html`<div class="empty">No inquiry turns for this image with ${modelShort(llama.label)}.</div>`}
      ${running ? html`<div class="bubble-m"><div class="row gap10"><${Dot} kind="run" /><span class="note">${single.stream ? 'streaming…' : 'generating…'}</span></div>
        ${single.stream ? html`<div class="txt" style=${{ marginTop: '8px' }}>${single.stream}</div>` : null}</div>` : null}
    </div>
    <div class="col gap8" style=${{ padding: '12px 16px', borderTop: '1px solid var(--line)', background: 'var(--panel)' }}>
      <div class="row gap6" style=${{ flexWrap: 'wrap', gap: '5px' }}>${tasks.map((t) => html`<button class=${`task ${task === t ? 'on' : ''} ${t === 'audit' ? 'audit' : ''}`} onClick=${() => setTask(t)}>${t}</button>`)}</div>
      <div class="row gap8">
        <div class="grow"><${TextField} value=${prompt} placeholder="Optional question or guidance…" onInput=${setPrompt} onEnter=${send} /></div>
        <${Btn} variant="primary" busy=${running} disabled=${single.running || getState().inquiry.batch.running} onClick=${send}>Send</${Btn}>
      </div>
    </div>
  </div>`;
}

function CommonTagsDialog({ paths }) {
  const [tags, setTags] = useState(null);
  const [checked, setChecked] = useState(new Set());
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    rpc('gallery.common_tags', { paths }).then((r) => { setTags(r.tags); setChecked(new Set(r.tags)); }).catch(reportError);
  }, [paths.join('\u0001')]);
  const removed = (tags || []).filter((t) => !checked.has(t));
  const save = async () => {
    if (!removed.length) { toast('No changes to save.'); return; }
    const ok = await confirmDialog({
      title: 'Save tag changes',
      message: `Apply changes to ${plural(paths.length, 'image')}?\n\nRemove ${plural(removed.length, 'tag')}: ${removed.slice(0, 8).join(', ')}${removed.length > 8 ? '…' : ''}\n\nEach image's unique tags are preserved.`,
      confirmLabel: 'Apply',
    });
    if (!ok) return;
    setSaving(true);
    try {
      const r = await rpc('gallery.save_common', { paths, original: tags, selected: [...checked] });
      toast(r.failed.length ? `Updated ${r.saved} images; ${r.failed.length} failed.` : `Tags updated for all ${r.saved} images.`, r.failed.length ? 'warn' : 'ok');
      setTags([...checked]);
    } catch (err) {
      reportError(err);
    } finally {
      setSaving(false);
    }
  };
  return html`<${Modal} title="Edit Common Tags" subtitle=${`${fmtInt(paths.length)} images selected`} width=${900} onClose=${() => closeModal('inspect')}
    footer=${html`<div class="note">Only tags every selected image shares are listed. Unchecking removes a tag from all of them.</div><div class="spacer"></div>
      <${Btn} onClick=${() => closeModal('inspect')}>Close</${Btn}>
      <${Btn} variant="success" busy=${saving} disabled=${saving || !tags} onClick=${save}>Save changes</${Btn}>`}>
    <div class="row gap14" style=${{ alignItems: 'flex-start' }}>
      <div style=${{ width: '300px', flex: 'none', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
        ${paths.slice(0, 9).map((p) => html`<div class="preview" style=${{ height: '86px' }}><${Img} src=${thumbUrl(p, 200)} /></div>`)}
        ${paths.length > 9 ? html`<div class="note" style=${{ gridColumn: '1 / -1' }}>+${fmtInt(paths.length - 9)} more</div>` : null}
      </div>
      <div class="grow" style=${{ display: 'flex', flexWrap: 'wrap', gap: '5px', alignContent: 'flex-start', minHeight: '120px' }}>
        ${tags == null ? html`<span class="spinner"></span>` : tags.length ? tags.map((t) => html`<span class=${`tagchip toggle ${checked.has(t) ? 'on' : 'off'}`} key=${t}
          onClick=${() => setChecked((prev) => { const n = new Set(prev); if (n.has(t)) n.delete(t); else n.add(t); return n; })}><span class="cb"></span>${t}</span>`)
          : html`<div class="help">These images share no tags.</div>`}
      </div>
    </div>
  </${Modal}>`;
}

// ── Organize by tags (1i) ─────────────────────────────────────────────────────

function OrganizeDialog({ state }) {
  const dir = useStore((s) => s.dir);
  const [scan, setScan] = useState(null);
  const [tags, setTags] = useState([]);
  const [tagInput, setTagInput] = useState('');
  const [matchMode, setMatchMode] = useState('any');
  const [target, setTarget] = useState('organized');
  const [moveText, setMoveText] = useState(true);
  const [recursive, setRecursive] = useState(true);
  const [targetInRoot, setTargetInRoot] = useState(false);
  const [selected, setSelected] = useState(null);
  const [plan, setPlan] = useState(null);
  const [running, setRunning] = useState(false);
  const inputKey = useRef(0);

  useEffect(() => {
    if (!dir.path) return;
    rpc('organize.scan', { target }).then((r) => {
      setScan(r);
      setSelected((prev) => prev || new Set(['.', ...r.subdirs.filter((d) => d.default_selected).map((d) => d.rel)]));
    }).catch(reportError);
  }, [dir.path]);

  const options = { tags, match_mode: matchMode, target, move_text: moveText, recursive, target_in_root: targetInRoot, selected_dirs: selected ? [...selected] : [] };
  const refreshPlan = useMemo(() => debounce((opts) => {
    rpc('organize.plan', opts).then(setPlan).catch((err) => setPlan({ error: err.message }));
  }, 200), []);
  useEffect(() => { if (dir.path && selected) refreshPlan(options); }, [tags.join('\u0001'), matchMode, target, moveText, recursive, targetInRoot, selected && [...selected].join('\u0001')]);

  const close = () => closeModal('organize');
  const addTag = (value) => {
    const tag = (value ?? tagInput).trim();
    if (tag && !tags.includes(tag)) setTags([...tags, tag]);
    setTagInput('');
    inputKey.current += 1;
  };
  const run = async () => {
    setRunning(true);
    try {
      await rpc('organize.run', options);
    } catch (err) {
      setRunning(false);
      reportError(err);
    }
  };
  useEffect(() => { if (state.finished) setRunning(false); }, [state.finished]);

  if (!dir.path) {
    return html`<${Modal} title="Organize Images by Tags" width=${760} onClose=${close}>
      <div class="empty"><div>Select a directory first.</div><${Btn} variant="primary" onClick=${() => { close(); chooseDirectory(); }}>Select Directory…</${Btn}></div></${Modal}>`;
  }
  const suggestions = tagInput.trim() ? (scan?.tags || []).filter((t) => t.toLowerCase().includes(tagInput.trim().toLowerCase()) && !tags.includes(t)).slice(0, 8) : [];
  const count = plan?.count || 0;
  const finished = state.finished;
  const progress = state.progress;

  return html`<${Modal} title="Organize Images by Tags" width=${760} onClose=${running ? null : close}
    footer=${html`<div class="spacer"></div>
      <${Btn} onClick=${close} disabled=${running}>${finished ? 'Close' : 'Cancel'}</${Btn}>
      ${finished ? null : html`<${Btn} variant="destructive" busy=${running} disabled=${running || !count} onClick=${run}>${running ? `Moving… ${progress ? `${progress.current}/${progress.total}` : ''}` : `Move ${plural(count, 'image')}`}</${Btn}>`}`}>
    <div class="banner red" style=${{ marginBottom: '16px' }}>
      <span class="dotc err" style=${{ marginTop: '5px' }}></span>
      <div><div style=${{ font: '600 11.5px/1.4 var(--sans)', marginBottom: '4px' }}>This MOVES files. They leave their current location.</div>
        <div class="mono" style=${{ color: 'var(--red-dim)' }}>No copy is kept. Sidecar .txt files move with their image when enabled below.</div></div>
    </div>
    ${finished ? html`<div class="banner green" style=${{ marginBottom: '14px' }}><div class="mono">Moved ${plural(finished.moved, 'image')}.${finished.errors.length ? ` ${finished.errors.length} errors: ${finished.errors.slice(0, 3).map((e) => basename(e.path)).join(', ')}` : ''}</div></div>` : null}
    <div class="col gap12" style=${{ gap: '13px' }}>
      <div>
        <div class="label" style=${{ marginBottom: '7px' }}>Tags to match</div>
        <div class="row" style=${{ flexWrap: 'wrap', gap: '5px', padding: '7px 9px', borderRadius: '5px', background: 'var(--bg)', border: '1px solid var(--line-strong)', minHeight: '30px' }}>
          ${tags.map((t) => html`<span class="tagchip blue">${t} <span class="x" onClick=${() => setTags(tags.filter((x) => x !== t))}>✕</span></span>`)}
          <input class="input" key=${inputKey.current} style=${{ border: 0, background: 'transparent', height: '20px', flex: 1, minWidth: '120px', padding: 0 }} placeholder="add tag…"
            value=${tagInput} onInput=${(e) => setTagInput(e.currentTarget.value)}
            onKeyDown=${(e) => { if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); addTag(); } if (e.key === 'Backspace' && !tagInput && tags.length) setTags(tags.slice(0, -1)); }} />
        </div>
        ${suggestions.length ? html`<div class="row" style=${{ flexWrap: 'wrap', gap: '5px', marginTop: '6px' }}>${suggestions.map((t) => html`<span class="tagchip toggle off" onClick=${() => addTag(t)}>${t}</span>`)}</div>` : null}
      </div>
      <div class="row gap10" style=${{ gap: '11px', alignItems: 'flex-end' }}>
        <div class="grow"><div class="label" style=${{ marginBottom: '7px' }}>Match mode</div>
          <${Seg} class="pad" value=${matchMode} onChange=${setMatchMode} options=${[{ value: 'any', label: 'any' }, { value: 'all', label: 'all' }]} /></div>
        <div class="grow"><div class="label" style=${{ marginBottom: '7px' }}>Target subdirectory</div>
          <${TextField} large value=${target} onCommit=${(v) => setTarget(v.trim() || 'organized')} /></div>
      </div>
      <div class="col gap8" style=${{ padding: '12px', borderRadius: '6px', background: 'var(--raised)', border: '1px solid var(--line)' }}>
        <${Check} large checked=${moveText} onChange=${setMoveText}>Move .txt files with images</${Check}>
        <${Check} large checked=${recursive} onChange=${setRecursive}>Include subdirectories (recursive)</${Check}>
        <${Check} large checked=${targetInRoot} onChange=${setTargetInRoot}>Create target subdirectory in root instead of each parent</${Check}>
      </div>
      ${recursive ? html`<div>
        <div class="label-row" style=${{ marginBottom: '7px' }}><div class="label">Source directories</div>
          <div class="row gap6"><button class="btn link" onClick=${() => setSelected(new Set(['.', ...(scan?.subdirs || []).map((d) => d.rel)]))}>Select all</button>
          <span class="dim">·</span><button class="btn link" onClick=${() => setSelected(new Set())}>none</button></div></div>
        <div class="listbox" style=${{ maxHeight: '190px' }}>
          ${scan ? [{ rel: '.', label: '(root directory)', count: scan.root_count }, ...scan.subdirs.map((d) => ({ ...d, label: `${d.rel}/` }))].map((d) => {
            const on = selected?.has(d.rel);
            return html`<div class=${`li ${on ? '' : 'off'}`} style=${{ cursor: 'pointer' }} onClick=${() => setSelected((prev) => { const n = new Set(prev); if (n.has(d.rel)) n.delete(d.rel); else n.add(d.rel); return n; })}>
              <span style=${{ width: '12px', height: '12px', borderRadius: '3px', flex: 'none', background: on ? 'var(--blue)' : 'transparent', border: on ? 0 : '1px solid var(--box)' }}></span>
              <span class="grow">${d.label}</span><span class="aside">${plural(d.count, 'image')}${!on && d.rel === target ? ' · excluded' : ''}</span></div>`;
          }) : html`<div class="li off">Scanning…</div>`}
        </div>
        <div class="note" style=${{ marginTop: '7px' }}>deselect folders you have already organized so they are not moved again</div>
      </div>` : null}
      <div style=${{ padding: '12px 14px', borderRadius: '6px', background: 'var(--raised)', border: '1px solid var(--line-strong)' }}>
        <div class="label" style=${{ marginBottom: '8px' }}>Resolved plan</div>
        ${plan?.error ? html`<div class="note err">${plan.error}</div>` : !tags.length ? html`<div class="note">Add at least one tag to match.</div>` : plan ? html`<div class="mono" style=${{ fontSize: '11px', lineHeight: 1.8, color: 'var(--tx2)' }}>
          <b style=${{ color: 'var(--tx1)', fontWeight: 500 }}>${plural(plan.count, 'image')}</b>${plan.move_text ? html` + <b style=${{ color: 'var(--tx1)', fontWeight: 500 }}>${plural(plan.txt_count, '.txt file')}</b>` : null} → <span style=${{ color: 'var(--blue-tx)' }}>${plan.destination}</span><br />
          <span style=${{ color: 'var(--tx6)' }}>from ${plan.sources_used} of ${plan.source_count} source ${plan.source_count === 1 ? 'directory' : 'directories'}${plan.collisions ? ` · ${plural(plan.collisions, 'name collision')} will be suffixed` : ''}</span>
        </div>` : html`<div class="note">Resolving…</div>`}
      </div>
    </div>
  </${Modal}>`;
}

// ── Runtime provisioning ──────────────────────────────────────────────────────

const METHOD_CARDS = [
  { value: 'auto', title: 'Matched release', detail: '~1 min download, source build if none is published' },
  { value: 'source', title: 'Source build only', detail: '~20 min · best for ARM64 / DGX Spark' },
  { value: 'release', title: 'Official release only', detail: 'never compiles' },
];

function ProvisionForm({ env, value, onChange }) {
  const [hint, setHint] = useState(null);
  useEffect(() => {
    if (value.accelerator === 'cuda' || (!value.accelerator && env?.accelerator === 'cuda')) {
      rpc('runtime.cuda_hint', { cuda_arch: value.cuda_arch || '' }).then(setHint).catch(() => setHint(null));
    } else setHint(null);
  }, [value.cuda_arch, value.accelerator, env?.accelerator]);
  if (!env) return html`<div class="note row gap6"><span class="spinner"></span>Detecting this machine…</div>`;
  const accel = value.accelerator || env.accelerator;
  return html`<div class="col gap12">
    <div class="note" style=${{ fontSize: '11px' }}>Detected: ${env.platform}/${env.arch} · backend ${env.detected}</div>
    <div class="row gap8">${METHOD_CARDS.map((m) => html`<button class=${`optcard ${value.method === m.value ? 'on' : ''}`} onClick=${() => onChange({ ...value, method: m.value })}>
      <div class="h">${m.title}</div><div class="d">${m.detail}</div></button>`)}</div>
    <div class="row gap10" style=${{ alignItems: 'flex-end' }}>
      <div class="grow"><div class="label" style=${{ marginBottom: '6px' }}>Accelerator</div>
        <select class="select lg" value=${value.accelerator} onChange=${(e) => onChange({ ...value, accelerator: e.currentTarget.value })}>
          <option value="">Auto-detect (${env.detected})</option>
          ${env.accelerators.map((a) => html`<option value=${a}>${a}</option>`)}
        </select></div>
      <div class="grow"><div class="label" style=${{ marginBottom: '6px' }}>Version</div>
        <${TextField} large value=${value.version} placeholder="latest, or a tag / git ref such as b6000" onCommit=${(v) => onChange({ ...value, version: v || 'latest' })} /></div>
      ${accel === 'cuda' ? html`<div style=${{ width: '140px' }}><div class="label" style=${{ marginBottom: '6px' }}>CUDA arch</div>
        <${TextField} large value=${value.cuda_arch} placeholder=${env.cuda_arch || 'e.g. 121'} onCommit=${(v) => onChange({ ...value, cuda_arch: v })} /></div>` : null}
    </div>
    ${accel === 'cuda' && hint ? html`<div class=${`note ${hint.ok ? '' : 'warn'}`}>${hint.ok ? `CMAKE_CUDA_ARCHITECTURES=${hint.cmake}` : (value.cuda_arch || env.cuda_arch) ? 'Invalid compute capability. Use digits such as 86, 120, or 121.' : 'No GPU detected. Enter a compute capability, e.g. 121 for a DGX Spark.'}</div>` : null}
    <${Check} checked=${value.bypass} onChange=${(v) => onChange({ ...value, bypass: v })}
      title="Preflight catches a missing or too-old CUDA Toolkit in seconds instead of twenty minutes into a compile.">Skip prerequisite checks (compile anyway)</${Check}>
    <div class="note">Falls back to source, then cpu, if a rung fails — you will be told which one stuck.</div>
  </div>`;
}

async function startProvision(value) {
  if (value.method === 'source') {
    const ok = await confirmDialog({ title: 'Start a source build?', message: 'A source build compiles llama.cpp for this machine and can take twenty minutes or more. Start it now?', confirmLabel: 'Start build' });
    if (!ok) return false;
  }
  try {
    await rpc('runtime.provision', { accelerator: value.accelerator, method: value.method, version: value.version, cuda_arch: value.cuda_arch, bypass_checks: value.bypass });
    return true;
  } catch (err) {
    reportError(err);
    return false;
  }
}

function ProvisionProgress({ provision }) {
  const logRef = useRef(null);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [provision.logs.length]);
  const step = provision.step;
  const overall = step ? Math.round((((step.step - 1) + Math.max(0, step.fraction)) / Math.max(1, step.total)) * 100) : 0;
  return html`<div class="col gap8">
    <div class="row gap10"><div class="mono" style=${{ fontSize: '11px', color: 'var(--tx2)' }}>${provision.running ? (step ? `Step ${step.step}/${step.total}: ${step.label}` : 'Starting…') : provision.result?.binary ? 'Done.' : provision.result ? 'Failed.' : ''}</div>
      <div class="spacer"></div><div class="mono" style=${{ fontSize: '10.5px', color: 'var(--blue-tx)' }}>${provision.running ? `${overall}%` : ''}</div></div>
    <${Bar} value=${provision.result?.binary ? 100 : overall} large />
    <pre class="logbox" ref=${logRef} style=${{ height: '230px' }}>${provision.logs.length ? provision.logs.map((l) => html`<span class=${l.stderr ? 'e' : ''}>${l.stderr ? '! ' : ''}${l.line}\n</span>`) : 'Build output appears here.'}</pre>
    ${provision.result?.mismatch ? html`<div class="banner amber"><div class="mono" style=${{ whiteSpace: 'pre-wrap' }}>${provision.result.mismatch}</div></div>` : null}
    ${provision.result?.error ? html`<div class="banner red"><div class="mono" style=${{ whiteSpace: 'pre-wrap' }}>${provision.result.error}${provision.result.log_path ? `\nFull build log: ${provision.result.log_path}` : ''}</div></div>` : null}
    ${provision.result?.binary && !provision.result.mismatch ? html`<div class="banner green"><div class="mono">Installed: ${provision.result.binary}</div></div>` : null}
  </div>`;
}

function ProvisionDialog() {
  const provision = useStore((s) => s.runtime.provision);
  const [env, setEnv] = useState(null);
  const [value, setValue] = useState({ method: 'auto', accelerator: '', version: 'latest', cuda_arch: '', bypass: false });
  const [started, setStarted] = useState(provision.running);
  useEffect(() => { rpc('runtime.environment').then((e) => { setEnv(e); setValue((v) => ({ ...v, cuda_arch: v.cuda_arch || e.cuda_arch })); }).catch(reportError); }, []);
  const close = () => closeModal('provision');
  const cancel = () => rpc('runtime.provision_cancel').catch(reportError);
  return html`<${Modal} title="Provision llama-server" width=${780} onClose=${provision.running ? null : close}
    footer=${html`<div class="note">Installs into cache/llama_cpp; the Inquiry tab resolves it from disk on every load.</div><div class="spacer"></div>
      ${provision.running ? html`<${Btn} variant="danger" onClick=${cancel}>Cancel</${Btn}>` : html`<${Btn} onClick=${close}>Close</${Btn}>`}
      ${!provision.running ? html`<${Btn} variant="primary" disabled=${!env} onClick=${async () => { if (await startProvision(value)) setStarted(true); }}>${provision.result ? 'Run again' : 'Install'}</${Btn}>` : null}`}>
    <div class="col gap14">
      ${!provision.running ? html`<${ProvisionForm} env=${env} value=${value} onChange=${setValue} />` : null}
      ${started || provision.running || provision.result ? html`<${ProvisionProgress} provision=${provision} />` : null}
    </div>
  </${Modal}>`;
}

// ── First run (2d) ────────────────────────────────────────────────────────────

function FirstRunDialog() {
  const { hw, runtime, provision, dir, config } = useStore((s) => ({ hw: s.hw, runtime: s.runtime.summary, provision: s.runtime.provision, dir: s.dir, config: s.inquiry.config }));
  const [step, setStep] = useState(1);
  const [choice, setChoice] = useState('auto');
  const [customPath, setCustomPath] = useState('');
  const [model, setModel] = useState({ model: '', mmproj: '' });
  const [env, setEnv] = useState(null);
  useEffect(() => { rpc('runtime.environment').then(setEnv).catch(() => {}); }, []);
  useEffect(() => { rpc('inquiry.state').then((q) => setModel({ model: q.config.llama_model_path || '', mmproj: q.config.llama_mmproj_path || '' })).catch(() => {}); }, []);

  const finish = async () => {
    await rpc('app.complete_first_run').catch(() => {});
    patch('settings', { first_run_done: true });
    closeModal('firstRun');
  };
  const chooseModel = async () => {
    const picked = await native.pickFile({ title: 'Select multimodal GGUF model', filters: [{ name: 'GGUF', extensions: ['gguf'] }] });
    if (!picked) return;
    const mmproj = await native.pickFile({ title: 'Select multimodal projector (optional — cancel to skip)', filters: [{ name: 'GGUF', extensions: ['gguf'] }] });
    setModel({ model: picked, mmproj: mmproj || '' });
    await rpc('inquiry.save', { config: { llama_model_path: picked, llama_mmproj_path: mmproj || null } }).catch(reportError);
  };
  const installAndContinue = async () => {
    if (choice === 'custom') {
      if (customPath) await rpc('inquiry.save', { config: { llama_runtime_mode: 'custom', llama_binary_path: customPath } }).catch(reportError);
      setStep(2);
      return;
    }
    const ok = await startProvision({ method: choice, accelerator: '', version: 'latest', cuda_arch: env?.cuda_arch || '', bypass: false });
    if (ok) setStep(2);
  };

  const onnxGpu = hw.onnx_cuda;
  const rtInstalled = runtime?.installed;
  return html`<div class="backdrop"><div class="modal" style=${{ width: '960px' }}>
    <div class="modal-head"><img src="/assets/icon.png" alt="" style=${{ width: '16px', height: '16px', borderRadius: '3px' }} />
      <div class="t">Image Interrogator — first run</div><div class="spacer"></div><button class="xbtn" onClick=${finish}>✕</button></div>
    <div class="modal-body fr" style=${{ padding: '36px 52px 40px' }}>
      ${step === 1 ? html`
        <div class="step">Step 1 of 3 · Environment</div>
        <h1>Checking what this machine can actually run.</h1>
        <p class="lead">Everything below is detected, not assumed. Anything amber will still work — just slower — and the fix is one click away.</p>
        <div class="col gap8" style=${{ marginBottom: '24px' }}>
          <div class=${`envrow ${hw.torch_cuda ? '' : 'warn'}`}><${Dot} kind=${hw.torch_cuda ? 'ok' : 'warn'} large /><div class="n">PyTorch (CLIP)</div>
            <div class="d">${hw.torch_cuda ? `CUDA available${hw.gpu_name ? ` · ${hw.gpu_name}` : ''}` : `torch ${hw.torch_version || ''} CPU build — CLIP runs much slower`}</div><div class="r" style=${hw.torch_cuda ? null : { color: 'var(--amber)' }}>${hw.torch_cuda ? 'gpu' : 'cpu'}</div></div>
          <div class=${`envrow ${onnxGpu ? '' : 'warn'}`}><${Dot} kind=${onnxGpu ? 'ok' : 'warn'} large /><div class="n">ONNX Runtime (WD, Camie)</div>
            <div class="d">${onnxGpu ? 'CUDAExecutionProvider available' : `${hw.onnx_version || ''} CPU build — WD Tagger would run 10–50× slower. Run ./setup.sh to install the GPU build.`}</div><div class="r" style=${onnxGpu ? null : { color: 'var(--amber)' }}>${onnxGpu ? 'gpu' : 'cpu'}</div></div>
          <div class="envrow focus" style=${{ flexDirection: 'column', alignItems: 'stretch', padding: '14px 16px' }}>
            <div class="row gap14"><${Dot} kind=${rtInstalled ? (runtime.is_gpu ? 'ok' : 'warn') : 'run'} large /><div class="n">llama.cpp runtime</div>
              <div class="d">${rtInstalled ? `Installed · ${runtime.accelerator} · ${runtime.short_version}${runtime.fallback_from ? ` (fallback — hardware supports ${runtime.fallback_from})` : ''}` : `No runtime installed · detected accelerator: ${runtime?.detected || env?.detected || '…'}`}</div>
              ${rtInstalled ? html`<div class="r" style=${runtime.is_gpu ? null : { color: 'var(--amber)' }}>${runtime.is_gpu ? 'gpu' : 'cpu'}</div>` : null}</div>
            ${!rtInstalled || runtime.fallback_from ? html`<div class="row gap8" style=${{ margin: '12px 0 0 22px' }}>
              ${[['auto', 'Matched release', '~1 min download'], ['source', 'Source build only', '~20 min · best for ARM64 / DGX Spark'], ['custom', 'Custom path', 'use your own llama-server']].map(([id, t, d]) => html`
                <button class=${`optcard ${choice === id ? 'on' : ''}`} onClick=${() => setChoice(id)}><div class="h">${t}</div><div class="d">${d}</div></button>`)}
            </div>
            ${choice === 'custom' ? html`<div class="row gap6" style=${{ margin: '9px 0 0 22px' }}>
              <div class="grow"><${TextField} large value=${customPath} placeholder="/path/to/llama-server" onCommit=${setCustomPath} /></div>
              <button class="iconbtn" style=${{ height: '30px', width: '30px' }} onClick=${async () => { const p = await native.pickFile({ title: 'Select llama-server binary' }); if (p) setCustomPath(p); }}>…</button></div>` : null}
            <div class="note" style=${{ margin: '9px 0 0 22px', fontSize: '10.5px' }}>falls back to source, then cpu, if a rung fails — you will be told which one stuck</div>` : null}
            ${provision.running || provision.result ? html`<div style=${{ margin: '12px 0 0 22px' }}><${ProvisionProgress} provision=${provision} /></div>` : null}
          </div>
          <div class=${`envrow ${model.model ? '' : 'off'}`}><${Dot} kind=${model.model ? 'ok' : ''} large /><div class="n">Multimodal GGUF model</div>
            <div class="d ellipsis" title=${model.model}>${model.model ? `${basename(model.model)}${model.mmproj ? ` + ${basename(model.mmproj)}` : ''}` : 'optional · .gguf + mmproj, HF cache paths resolved automatically'}</div>
            <${Btn} size="sm" onClick=${chooseModel}>Choose files…</${Btn}></div>
        </div>
        <div class="row gap12">
          ${rtInstalled && !runtime.fallback_from
            ? html`<${Btn} size="xtall" variant="primary" onClick=${() => setStep(2)}>Continue — pick an image folder</${Btn}>`
            : html`<${Btn} size="xtall" variant="primary" busy=${provision.running} disabled=${provision.running || (choice === 'custom' && !customPath)} onClick=${installAndContinue}>
                ${choice === 'custom' ? 'Use this path & continue' : 'Install runtime & continue'}</${Btn}>
              <${Btn} size="xtall" onClick=${() => setStep(2)}>Skip — no Inquiry for now</${Btn}>`}
          <div class="spacer"></div><div class="note" style=${{ fontSize: '10.5px', color: 'var(--tx7)' }}>re-runnable from Settings → Hardware</div>
        </div>` : null}
      ${step === 2 ? html`
        <div class="step">Step 2 of 3 · Images</div>
        <h1>Pick the folder you want to tag.</h1>
        <p class="lead">Images in this folder form the shared queue for interrogation, inquiry and the gallery. Subfolders are included when recursive is on.</p>
        <div class="envrow" style=${{ marginBottom: '24px' }}>
          <${Dot} kind=${dir.path ? 'ok' : ''} large /><div class="n">Image directory</div>
          <div class="d ellipsis">${dir.path ? `${dir.path} · ${dir.scanning ? 'scanning…' : `${fmtInt(dir.paths.length)} images`}` : 'none selected'}</div>
          <${Check} checked=${dir.recursive} onChange=${(v) => dir.path ? openDirectory(dir.path, v) : patch('dir', { recursive: v })}>Recursive</${Check}>
          <${Btn} size="sm" onClick=${chooseDirectory}>Choose folder…</${Btn}>
        </div>
        ${provision.running ? html`<div class="banner blue" style=${{ marginBottom: '18px' }}><div class="mono">The llama.cpp runtime is still installing in the background — you can keep going.</div></div>` : null}
        <div class="row gap12">
          <${Btn} size="xtall" variant="primary" onClick=${() => setStep(3)}>Continue</${Btn}>
          <${Btn} size="xtall" onClick=${() => setStep(1)}>Back</${Btn}>
        </div>` : null}
      ${step === 3 ? html`
        <div class="step">Step 3 of 3 · Ready</div>
        <h1>You're set up.</h1>
        <p class="lead">Load a tagger from the MDL rail on the Interrogation tab and press Start Batch. The run bar shows throughput and ETA; every image in the queue carries its own state.</p>
        <div class="col gap8" style=${{ marginBottom: '24px' }}>
          <div class="envrow"><${Dot} kind=${dir.path ? 'ok' : ''} large /><div class="n">Images</div><div class="d">${dir.path ? `${fmtInt(dir.paths.length)} images queued` : 'no folder yet — choose one any time'}</div></div>
          <div class="envrow"><${Dot} kind=${runtime?.installed ? 'ok' : provision.running ? 'run' : ''} large /><div class="n">Inquiry</div><div class="d">${runtime?.installed ? `llama.cpp ${runtime.accelerator} ready${model.model ? ` · ${basename(model.model)}` : ' · choose a GGUF on the Inquiry tab'}` : provision.running ? 'runtime installing…' : 'skipped — install from Settings → llama.cpp runtime'}</div></div>
        </div>
        <div class="row gap12"><${Btn} size="xtall" variant="primary" onClick=${finish}>Start interrogating</${Btn}><${Btn} size="xtall" onClick=${() => setStep(2)}>Back</${Btn}></div>` : null}
    </div>
  </div></div>`;
}

// ── Database busy ─────────────────────────────────────────────────────────────

function BusyDialog({ request }) {
  const reply = async (response) => {
    setState((s) => ({ busy: s.busy.filter((r) => r.id !== request.id) }));
    await rpc('db.busy_reply', { id: request.id, response }).catch(reportError);
  };
  return html`<${Modal} title="Database busy" width=${560} closeOnBackdrop=${false}
    footer=${html`<div class="spacer"></div>
      <${Btn} variant="danger" onClick=${() => reply('abort')}>Abort operation</${Btn}>
      ${request.queueable ? html`<${Btn} onClick=${() => reply('queue')}>Queue & continue</${Btn}>` : null}
      <${Btn} variant="primary" onClick=${() => reply('retry')}>Retry</${Btn}>`}>
    <div class="col gap10">
      <div style=${{ font: '400 12px/1.7 var(--sans)', color: 'var(--tx2)' }}>The database is locked by another process, so <span class="mono">${request.operation}</span> could not complete after ${request.retry_count} retries.</div>
      <div class="help">${request.queueable ? 'Queue the write to finish it later and keep the batch running, retry now, or abort this operation.' : 'This operation cannot be queued. Retry once the other process is done, or abort it.'}</div>
      ${request.queued?.length ? html`<div><div class="label" style=${{ marginBottom: '6px' }}>Already queued</div>
        <div class="listbox" style=${{ maxHeight: '140px' }}>${request.queued.map((op) => html`<div class="li"><span class="grow ellipsis">${op.summary || op.operation}</span></div>`)}</div></div>` : null}
    </div>
  </${Modal}>`;
}

// ── About ─────────────────────────────────────────────────────────────────────

function AboutDialog() {
  const hw = useStore((s) => s.hw);
  return html`<${Modal} title="About Image Interrogator" width=${520} onClose=${() => closeModal('about')}
    footer=${html`<div class="spacer"></div><${Btn} onClick=${() => closeModal('about')}>Close</${Btn}>`}>
    <div class="row gap14" style=${{ alignItems: 'flex-start' }}>
      <img src="/assets/icon.png" alt="" style=${{ width: '64px', height: '64px', borderRadius: '12px' }} />
      <div class="col gap8">
        <div style=${{ font: '500 14px/1.3 var(--sans)', color: 'var(--tx1)' }}>Image Interrogator</div>
        <div class="help">Batch image tagging with CLIP, WD and Camie taggers, cached in SQLite, plus llama.cpp multimodal inquiry. Tag filters shape .txt output only; the database keeps raw results.</div>
        <div class="note">torch ${hw.torch_version || '—'} · onnxruntime ${hw.onnx_version || '—'}${hw.gpu_name ? ` · ${hw.gpu_name}` : ''}</div>
        <div class="note">Double-click a queue row or thumbnail for advanced inspection. Esc closes dialogs.</div>
      </div>
    </div>
  </${Modal}>`;
}
