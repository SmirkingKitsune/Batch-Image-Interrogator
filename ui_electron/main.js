'use strict';

// Electron main process for Image Interrogator's opt-in front end.
//
// It is started by the Python bridge (./run.sh --electron), which passes the
// bridge URL and a per-launch token through the environment. The token goes
// into the session as an HttpOnly cookie before the page loads, so page
// script never sees it. The window loads the renderer from the bridge and
// talks to the backend over that same origin.

const { app, BrowserWindow, Menu, dialog, ipcMain, session, shell } = require('electron');
const fs = require('fs');
const path = require('path');

const BRIDGE_URL = process.env.II_BRIDGE_URL || '';
const BRIDGE_TOKEN = process.env.II_BRIDGE_TOKEN || '';
const PROJECT_ROOT = process.env.II_PROJECT_ROOT || path.resolve(__dirname, '..');

if (!BRIDGE_URL || !BRIDGE_TOKEN) {
  console.error('This window is started by ./run.sh --electron. Run that instead of launching Electron directly.');
  app.exit(1);
}

let mainWindow = null;

function isAppFrame(event) {
  const url = (event.senderFrame && event.senderFrame.url) || '';
  return url.startsWith(`${BRIDGE_URL}/`);
}

function windowFor(event) {
  return BrowserWindow.fromWebContents(event.sender);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 940,
    minWidth: 1120,
    minHeight: 700,
    frame: false,
    show: false,
    backgroundColor: '#0f1113',
    title: 'Image Interrogator',
    icon: path.join(PROJECT_ROOT, 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });

  win.once('ready-to-show', () => win.show());

  // Navigation never leaves the bridge; external links open in the browser.
  win.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(`${BRIDGE_URL}/`)) event.preventDefault();
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });

  const sendState = () => {
    if (!win.isDestroyed()) {
      win.webContents.send('window:state', { maximized: win.isMaximized() });
    }
  };
  win.on('maximize', sendState);
  win.on('unmaximize', sendState);

  win.loadURL(`${BRIDGE_URL}/app/`);
  return win;
}

// The bridge owns this process. If it goes away (crash, kill -9) the window
// would otherwise sit there disconnected; reparenting is the tell.
function watchParent() {
  const parent = process.ppid;
  setInterval(() => {
    if (process.ppid !== parent) app.quit();
  }, 2000).unref();
}

app.whenReady().then(async () => {
  Menu.setApplicationMenu(null);
  const ses = session.defaultSession;
  ses.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  await ses.cookies.set({
    url: BRIDGE_URL,
    name: 'ii_token',
    value: BRIDGE_TOKEN,
    httpOnly: true,
    sameSite: 'strict',
  });
  mainWindow = createWindow();
  watchParent();
});

app.on('window-all-closed', () => app.quit());

// -- window controls (custom title bar) --------------------------------------

ipcMain.handle('window:minimize', (event) => {
  if (isAppFrame(event)) windowFor(event)?.minimize();
});

ipcMain.handle('window:toggle-maximize', (event) => {
  if (!isAppFrame(event)) return false;
  const win = windowFor(event);
  if (!win) return false;
  if (win.isMaximized()) win.unmaximize();
  else win.maximize();
  return win.isMaximized();
});

ipcMain.handle('window:close', (event) => {
  if (isAppFrame(event)) windowFor(event)?.close();
});

ipcMain.handle('window:is-maximized', (event) => {
  if (!isAppFrame(event)) return false;
  return Boolean(windowFor(event)?.isMaximized());
});

// -- native dialogs ------------------------------------------------------------

function cleanFilters(filters) {
  if (!Array.isArray(filters)) return undefined;
  return filters
    .filter((f) => f && typeof f.name === 'string' && Array.isArray(f.extensions))
    .map((f) => ({ name: f.name, extensions: f.extensions.map(String) }));
}

ipcMain.handle('dialog:directory', async (event, options = {}) => {
  if (!isAppFrame(event)) return null;
  const result = await dialog.showOpenDialog(windowFor(event), {
    title: String(options.title || 'Select Image Directory'),
    defaultPath: options.defaultPath ? String(options.defaultPath) : undefined,
    properties: ['openDirectory'],
  });
  return result.canceled ? null : result.filePaths[0] || null;
});

ipcMain.handle('dialog:file', async (event, options = {}) => {
  if (!isAppFrame(event)) return null;
  const result = await dialog.showOpenDialog(windowFor(event), {
    title: String(options.title || 'Select File'),
    defaultPath: options.defaultPath ? String(options.defaultPath) : undefined,
    filters: cleanFilters(options.filters),
    properties: ['openFile'],
  });
  return result.canceled ? null : result.filePaths[0] || null;
});

ipcMain.handle('dialog:save', async (event, options = {}) => {
  if (!isAppFrame(event)) return null;
  const result = await dialog.showSaveDialog(windowFor(event), {
    title: String(options.title || 'Save'),
    defaultPath: options.defaultPath ? String(options.defaultPath) : undefined,
    filters: cleanFilters(options.filters),
  });
  return result.canceled ? null : result.filePath || null;
});

// -- file manager ----------------------------------------------------------------

// Only folders are opened: handing an arbitrary file to the desktop's default
// handler could run it.
ipcMain.handle('shell:open-folder', async (event, target) => {
  if (!isAppFrame(event) || typeof target !== 'string') return 'invalid';
  try {
    if (!fs.statSync(target).isDirectory()) return 'not a folder';
  } catch {
    return 'not found';
  }
  return shell.openPath(target);
});

ipcMain.handle('shell:show-item', (event, target) => {
  if (!isAppFrame(event) || typeof target !== 'string') return;
  shell.showItemInFolder(target);
});
