# 🎯 人脸识别相似度分析 - 桌面版

基于 **InsightFace** 的高精度人脸识别桌面应用，Electron + Python 本地打包，一个 exe 搞定。

## ✨ 功能特性

- ✅ **照片对比** - 上传两张照片，计算人脸相似度
- ✅ **人脸检测** - 检测照片中所有人脸，识别年龄、性别
- ✅ **视频对比** - 在视频中搜索目标人脸，**实时显示进度条**
- ✅ **导出报告** - 一键导出 **HTML 分析报告** 或 **JSON 原始数据**
- ✅ **离线运行** - 所有计算在本地完成，数据不出本机
- ✅ **单文件部署** - 打包后只有一个 exe，发给别人直接用

## 📦 项目结构

```
face-similarity-desktop/
├── main.js                    # Electron 主进程
├── preload.js                 # 前后端桥接
├── package.json               # Electron 配置
├── src/
│   └── index.html             # 前端界面
├── python-server/
│   ├── main.py                # FastAPI 后端服务
│   ├── main.spec              # PyInstaller 配置
│   └── requirements.txt       # Python 依赖
└── assets/                   # 图标等资源
```

## 🚀 快速开始

### 方式一：开发模式运行

**1. 安装 Python 依赖**
```bash
cd python-server
pip install -r requirements.txt
```

**2. 安装 Node 依赖**
```bash
npm install
```

**3. 启动应用**
```bash
npm start
```

---

### 方式二：打包成 exe（推荐）

**1. 打包 Python 后端**
```bash
pip install pyinstaller
cd python-server
pyinstaller main.spec --distpath dist --clean
```
生成的 `main-server.exe` 在 `python-server/dist/` 目录

**2. 打包 Electron 应用**
```bash
npm install
npm run build:win
```

**3. 找到安装包**
打包完成后，exe 安装包在 `dist/` 目录下。

---

## 🎛️ 相似度参考标准

| 分数范围 | 结果 | 说明 |
|---------|------|------|
| ≥ 0.6   | ✅ 极有可能同一人 | 可以确认为同一人 |
| 0.4~0.6 | 🟡 可能同一人 | 需要进一步确认 |
| < 0.4   | ❌ 不是同一人 | 不是同一个人 |

## 🎬 视频对比参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 相似度阈值 | 0.4 | 低于此值的结果不显示 |
| 采样间隔 | 30 | 每隔多少帧检测一次（越大越快但可能漏检） |

**性能优化建议**：
- 短视频（<1分钟）：采样间隔设为 15~30
- 中等视频（1~10分钟）：采样间隔设为 30~60
- 长视频（>10分钟）：采样间隔设为 60~120

## 🔧 技术栈

| 组件 | 技术 |
|------|------|
| 人脸引擎 | InsightFace (ArcFace 模型) |
| 后端框架 | FastAPI |
| 桌面框架 | Electron |
| 前端界面 | 原生 HTML + CSS + JS |

## ⚙️ 配置说明

### Python 后端配置（`python-server/main.py`）

```python
# 使用 GPU 加速（需要 NVIDIA 显卡）
face_app.prepare(ctx_id=0)

# 使用 CPU（兼容所有电脑）
face_app.prepare(ctx_id=-1)
```

### Electron 配置（`package.json`）

```json
{
  "build": {
    "win": {
      "target": "nsis",
      "icon": "assets/icon.ico"
    }
  }
}
```

## 📝 常见问题

**Q: 打包后 exe 太大？**
A: InsightFace 模型约 300MB，这是正常现象。可以考虑使用在线 API 模式减小体积。

**Q: 低配电脑跑得慢？**
A: 修改 `main.py` 中的 `ctx_id=-1` 使用 CPU 模式，或升级硬件。

**Q: 如何隐藏 Python 控制台窗口？**
A: 修改 `python-server/main.spec` 中的 `console=True` 改为 `console=False`

## 📄 许可证

MIT License

---

**开发者**: Zhang  
**创建时间**: 2026-05-13
