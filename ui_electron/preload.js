'use strict';

// The only native surface the page gets: window controls for the custom title
// bar, file dialogs, and revealing files. Everything else goes to the Python
// bridge over HTTP.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('native', {
  platform: process.platform,
  minimize: () => ipcRenderer.invoke('window:minimize'),
  toggleMaximize: () => ipcRenderer.invoke('window:toggle-maximize'),
  close: () => ipcRenderer.invoke('window:close'),
  isMaximized: () => ipcRenderer.invoke('window:is-maximized'),
  onWindowState: (callback) => {
    const handler = (_event, state) => callback(state);
    ipcRenderer.on('window:state', handler);
    return () => ipcRenderer.removeListener('window:state', handler);
  },
  pickDirectory: (options) => ipcRenderer.invoke('dialog:directory', options || {}),
  pickFile: (options) => ipcRenderer.invoke('dialog:file', options || {}),
  pickSavePath: (options) => ipcRenderer.invoke('dialog:save', options || {}),
  openFolder: (target) => ipcRenderer.invoke('shell:open-folder', target),
  showItem: (target) => ipcRenderer.invoke('shell:show-item', target),
});
