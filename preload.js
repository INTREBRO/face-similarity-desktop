const { contextBridge, ipcRenderer } = require('electron');

// 安全地暴露 API 给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
  // 打开文件选择对话框
  openFileDialog: (options) => ipcRenderer.invoke('dialog:openFile', options),

  // 读取本地文件（返回 Uint8Array）
  readFile: (filePath) => ipcRenderer.invoke('fs:readFile', filePath),

  // 获取文件名
  pathBasename: (filePath) => ipcRenderer.invoke('path:basename', filePath),

  // 调用后端 API
  apiRequest: async (endpoint, options = {}) => {
    const baseURL = 'http://localhost:8000';
    const url = `${baseURL}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });
      return await response.json();
    } catch (error) {
      console.error('API 请求失败:', error);
      throw error;
    }
  },

  // 上传文件到后端（通过主进程转发，绕过渲染进程 fetch 的 FormData 问题）
  uploadFile: async (endpoint, files) => {
    // files: [{ fieldName, filePath }]
    return ipcRenderer.invoke('api:uploadFiles', endpoint, files);
  }
});
