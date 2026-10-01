// Renderer entry point.
import { html, render, useEffect } from './lib.js';
import { connectEvents, rpc } from './api.js';
import { getState, setState, useStore } from './store.js';
import { bootstrapState, loadAppState, wireEvents } from './state.js';
import { ConfirmHost, reportError, Toasts } from './ui.js';
import { TabStrip, TitleBar } from './views/chrome.js';
import { InterrogationPage } from './views/interrogation.js';

// Tabs other than Interrogation, and the dialogs, load on first use.
const lazyViews = {
  inquiry: () => import('./views/inquiry.js').then((m) => m.InquiryPage),
  gallery: () => import('./views/gallery.js').then((m) => m.GalleryPage),
  settings: () => import('./views/settings.js').then((m) => m.SettingsPage),
  dialogs: () => import('./dialogs.js').then((m) => m.DialogHost),
};
const loadedViews = {};

function Lazy({ name, fallback = true }) {
  useStore((s) => s.viewsVersion || 0);
  const View = loadedViews[name];
  useEffect(() => {
    if (loadedViews[name]) return;
    lazyViews[name]()
      .then((component) => {
        loadedViews[name] = component;
        setState({ viewsVersion: (getState().viewsVersion || 0) + 1 });
      })
      .catch((err) => reportError(err, `Could not load the ${name} view`));
  }, [name]);
  if (!View) return fallback ? html`<div class="page"><div class="empty"><span class="spinner"></span></div></div>` : null;
  return html`<${View} />`;
}

function Dialogs() {
  const { modals, busy } = useStore((s) => ({ modals: s.modals, busy: s.busy }));
  const needed = modals.inspect || modals.organize || modals.firstRun || modals.provision || modals.about || busy.length;
  if (!needed) return null;
  return html`<${Lazy} name="dialogs" fallback=${false} />`;
}

function App() {
  const { tab, connected, loaded } = useStore((s) => ({ tab: s.tab, connected: s.connected, loaded: s.loaded }));
  let page;
  if (!loaded) page = html`<div class="page"><div class="empty"><span class="spinner"></span>Connecting to the backend…</div></div>`;
  else if (tab === 'interrogation') page = html`<${InterrogationPage} />`;
  else page = html`<${Lazy} name=${tab} key=${tab} />`;
  return html`<div class="app">
    <${TitleBar} />
    <${TabStrip} />
    ${page}
    <${Dialogs} />
    <${ConfirmHost} />
    <${Toasts} />
    ${loaded && !connected ? html`<div class="disconnected"><div class="box">
      <div class="t">Backend disconnected</div>
      <div class="help">The Python bridge stopped responding. The window reconnects by itself if it comes back; otherwise close it and run ./run.sh --electron again.</div>
    </div></div>` : null}
  </div>`;
}

async function start() {
  bootstrapState();
  wireEvents();
  connectEvents();
  render(html`<${App} />`, document.getElementById('root'));
  try {
    await loadAppState();
    requestAnimationFrame(() => rpc('app.ready').catch(() => {}));
  } catch (err) {
    reportError(err, 'Could not load the app state');
  }
}

start();
