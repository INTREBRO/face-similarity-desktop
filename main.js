const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const net = require('net');
const http = require('http');
const fs = require('fs');

let mainWindow;
let pythonProcess = null;
const PYTHON_PORT = 8000;
// 判断是否为开发模式：npm start 时没有 NODE_ENV，通过判断是否在打包后的资源目录中
const IS_DEV = process.env.NODE_ENV === 'development' || !process.resourcesPath.includes('app.asar');

// 等待端口可用（Python 服务启动完成）
function waitForPort(port, timeout = 30000) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const check = () => {
      const client = net.createConnection({ port }, () => {
        client.end();
        resolve(true);
      });
      client.on('error', () => {
        if (Date.now() - startTime > timeout) {
          reject(new Error('Python 服务启动超时'));
        } else {
          setTimeout(check, 500);
        }
      });
    };
    check();
  });
}

// 查找可用的 Python 可执行文件
function findPythonExe() {
  const candidates = [
    // 优先用绝对路径（Windows 默认安装位置）
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python314', 'python.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python313', 'python.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python312', 'python.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python311', 'python.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python310', 'python.exe'),
    'python3',
    'python',
  ];
  const fs = require('fs');
  for (const c of candidates) {
    try {
      if (c.includes(path.sep) && fs.existsSync(c)) return c;
      if (!c.includes(path.sep)) return c; // 短命令交由 PATH 解析
    } catch (_) {}
  }
  return 'python';
}

// 启动 Python 后端服务
function startPythonServer() {
  return new Promise((resolve, reject) => {
    let pythonExe;

    if (IS_DEV) {
      // 开发模式：直接运行 Python 脚本
      pythonExe = findPythonExe();
      console.log('使用 Python 路径:', pythonExe);
      pythonProcess = spawn(pythonExe, ['python-server/main.py'], {
        cwd: app.getAppPath(),
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      });

      pythonProcess.on('error', (err) => {
        console.error('Python 进程启动失败:', err.message);
        reject(new Error(`Python 启动失败: ${err.message}\n请确认已安装 Python 并在 PATH 中。`));
      });
    } else {
      // 生产模式：运行打包后的 exe
      pythonExe = path.join(process.resourcesPath, 'python-server', 'main-server.exe');
      pythonProcess = spawn(pythonExe, [], {
        cwd: path.dirname(pythonExe),
        detached: true,
        stdio: 'ignore'
      });
      pythonProcess.unref();
    }

    console.log('Python 进程已启动，PID:', pythonProcess.pid);

    pythonProcess.stdout?.on('data', (data) => {
      console.log(`Python: ${data}`);
    });

    pythonProcess.stderr?.on('data', (data) => {
      console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python 进程退出，退出码: ${code}`);
    });

    // 等待服务就绪
    waitForPort(PYTHON_PORT)
      .then(() => resolve())
      .catch((err) => reject(err));
  });
}

// 停止 Python 服务
function stopPythonServer() {
  if (pythonProcess) {
    if (IS_DEV) {
      pythonProcess.kill();
    } else {
      // 生产模式下通过 API 通知 Python 退出
      fetch(`http://localhost:${PYTHON_PORT}/shutdown`)
        .catch(() => {})
        .finally(() => {
          pythonProcess.kill();
        });
    }
    pythonProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  // 加载本地 HTML 文件（开发和生产都用本地文件）
  mainWindow.loadFile(path.join(__dirname, 'src', 'index.html'));
  if (IS_DEV) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  try {
    console.log('正在启动 Python 服务...');
    await startPythonServer();
    console.log('Python 服务启动成功！');
    createWindow();
  } catch (err) {
    console.error('Python 服务启动失败:', err);
    dialog.showErrorBox('启动失败', '无法启动后端服务，请检查 Python 环境。');
    app.quit();
  }
});

app.on('window-all-closed', () => {
  stopPythonServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// IPC 通信：选择文件
ipcMain.handle('dialog:openFile', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

// IPC 通信：读取本地文件
ipcMain.handle('fs:readFile', async (event, filePath) => {
  const buffer = fs.readFileSync(filePath);
  return Array.from(buffer);
});

// IPC 通信：获取文件名
ipcMain.handle('path:basename', async (event, filePath) => {
  return path.basename(filePath);
});

// IPC 通信：上传文件到后端（绕过渲染进程 fetch 的 FormData 问题）
ipcMain.handle('api:uploadFiles', async (event, endpoint, files) => {
  // files: [{ fieldName, filePath }]
  const boundary = '----FormBoundary' + Math.random().toString(36).slice(2);
  const chunks = [];

  for (const { fieldName, filePath } of files) {
    const filename = path.basename(filePath);
    const data = fs.readFileSync(filePath);
    console.log(`Uploading ${fieldName}: ${filePath} (${data.length} bytes)`);
    chunks.push(
      Buffer.from(`--${boundary}\r\n`),
      Buffer.from(`Content-Disposition: form-data; name="${fieldName}"; filename="${filename}"\r\n`),
      Buffer.from(`Content-Type: application/octet-stream\r\n\r\n`),
      data,
      Buffer.from(`\r\n`)
    );
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`));

  const body = Buffer.concat(chunks);
  console.log(`Request body length: ${body.length} bytes`);

  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1',
      port: PYTHON_PORT,
      path: endpoint,
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': body.length
      }
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        console.log(`Response status: ${res.statusCode}, body: ${data.substring(0, 200)}`);
        try {
          resolve(JSON.parse(data));
        } catch {
          resolve({ error: '后端返回非 JSON: ' + data });
        }
      });
    });

    req.on('error', (err) => {
      console.error('Request error:', err);
      reject(err);
    });
    req.write(body);
    req.end();
  });
});
