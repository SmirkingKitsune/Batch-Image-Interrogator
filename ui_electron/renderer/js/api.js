// Talking to the Python bridge (same origin) and to the Electron shell.

const listeners = new Map();
const connectionListeners = new Set();
let source = null;
let connected = false;

export class RpcError extends Error {
  constructor(message, code, data) {
    super(message);
    this.code = code || 'error';
    this.data = data;
  }
}

/** Call a backend method. Resolves with the result or rejects with RpcError. */
export async function rpc(method, params = {}) {
  let response;
  try {
    response = await fetch('/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-II-Request': '1' },
      body: JSON.stringify({ method, params }),
      credentials: 'same-origin',
    });
  } catch (err) {
    throw new RpcError('The backend is not reachable.', 'offline');
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new RpcError(`Unexpected response (${response.status}).`, 'bad_response');
  }
  if (!payload.ok) {
    const error = payload.error || {};
    throw new RpcError(error.message || 'Request failed.', error.code, error.data);
  }
  return payload.result;
}

/** Subscribe to a server-sent event. Returns an unsubscribe function. */
export function on(event, handler) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(handler);
  if (source) attach(event);
  return () => listeners.get(event)?.delete(handler);
}

export function onConnection(handler) {
  connectionListeners.add(handler);
  return () => connectionListeners.delete(handler);
}

export function isConnected() {
  return connected;
}

const attached = new Set();
function attach(event) {
  if (attached.has(event)) return;
  attached.add(event);
  source.addEventListener(event, (message) => {
    let data = null;
    try {
      data = JSON.parse(message.data);
    } catch {
      return;
    }
    for (const handler of listeners.get(event) || []) {
      try {
        handler(data);
      } catch (err) {
        console.error(`handler for ${event} failed`, err);
      }
    }
  });
}

function setConnected(value) {
  if (connected === value) return;
  connected = value;
  for (const handler of connectionListeners) handler(value);
}

/** Open the event stream; EventSource reconnects by itself. */
export function connectEvents() {
  source = new EventSource('/events', { withCredentials: true });
  attached.clear();
  for (const event of listeners.keys()) attach(event);
  source.onopen = () => setConnected(true);
  source.onerror = () => setConnected(false);
}

// ── Native shell (Electron preload), with browser fallbacks for development ──

const shell = typeof window !== 'undefined' ? window.native : undefined;

export const native = {
  available: Boolean(shell),
  platform: shell?.platform || 'browser',
  minimize: () => shell?.minimize(),
  toggleMaximize: () => shell?.toggleMaximize(),
  close: () => (shell ? shell.close() : window.close()),
  isMaximized: () => (shell ? shell.isMaximized() : Promise.resolve(false)),
  onWindowState: (callback) => (shell ? shell.onWindowState(callback) : () => {}),
  async pickDirectory(options = {}) {
    if (shell) return shell.pickDirectory(options);
    const value = window.prompt(options.title || 'Directory path', options.defaultPath || '');
    return value ? value.trim() : null;
  },
  async pickFile(options = {}) {
    if (shell) return shell.pickFile(options);
    const value = window.prompt(options.title || 'File path', options.defaultPath || '');
    return value ? value.trim() : null;
  },
  async pickSavePath(options = {}) {
    if (shell) return shell.pickSavePath(options);
    const value = window.prompt(options.title || 'Save as', options.defaultPath || '');
    return value ? value.trim() : null;
  },
  openFolder: (target) => shell?.openFolder(target),
  showItem: (target) => shell?.showItem(target),
};

export function thumbUrl(path, size = 200) {
  return `/thumb?size=${size}&path=${encodeURIComponent(path)}`;
}

export function imageUrl(path) {
  return `/image?path=${encodeURIComponent(path)}`;
}
