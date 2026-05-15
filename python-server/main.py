import logging
import sys
import os
from datetime import datetime

# 配置日志
def setup_logging():
    log_dir = os.path.join(os.path.expanduser("~"), ".face-similarity", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import insightface
import cv2
import numpy as np
import tempfile
import asyncio
import uuid
import json
import time
from collections import defaultdict
from starlette.responses import StreamingResponse
from contextlib import asynccontextmanager

app = FastAPI(
    title="Face Similarity API",
    description="人脸识别相似度分析服务",
    version="1.0.0"
)

# 并发限制
MAX_CONCURRENT_REQUESTS = 3
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# 请求超时（秒）
REQUEST_TIMEOUT = 300  # 5分钟

# 活跃请求追踪
active_requests = defaultdict(lambda: {"start_time": None, "type": None})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型
face_app = None

def get_request_id(request: Request) -> str:
    """从请求头或生成请求ID"""
    return request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 50)
    logger.info("应用启动中...")
    yield
    logger.info("应用关闭中...")

app.router.lifespan_context = lifespan

@app.on_event("startup")
def load_model():
    global face_app
    logger.info("正在加载 InsightFace 模型...")
    model_dir = os.path.join(os.path.expanduser("~"), ".insightface", "models")
    os.makedirs(model_dir, exist_ok=True)
    try:
        face_app = insightface.app.FaceAnalysis(root=model_dir)
        face_app.prepare(ctx_id=0)
        logger.info("模型加载完成 (GPU 模式)")
    except Exception as e:
        logger.warning(f"GPU 模式失败，切换到 CPU 模式: {e}")
        face_app = insightface.app.FaceAnalysis(root=model_dir)
        face_app.prepare(ctx_id=-1)
        logger.info("模型加载完成 (CPU 模式)")

def get_face_embedding(image_bytes, request_id: str = "unknown"):
    """提取人脸特征向量，带错误处理"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"[{request_id}] 无法解码图片")
            return None, None

        faces = face_app.get(img)
        if not faces:
            logger.info(f"[{request_id}] 未检测到人脸")
            return None, None

        logger.info(f"[{request_id}] 检测到 {len(faces)} 个人脸")
        return faces[0].embedding, faces[0].bbox.tolist()
    except Exception as e:
        logger.error(f"[{request_id}] 人脸检测失败: {e}")
        return None, None

def cosine_similarity(emb1, emb2):
    """计算余弦相似度"""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))

def format_time(seconds):
    """格式化时间戳"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ─── 路由 ───

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "人脸识别服务运行中",
        "version": "1.0.0",
        "concurrent_limit": MAX_CONCURRENT_REQUESTS
    }

@app.get("/health")
def health():
    """健康检查"""
    active = sum(1 for r in active_requests.values() if r["start_time"] is not None)
    return {
        "status": "ok",
        "model_loaded": face_app is not None,
        "active_requests": active,
        "concurrent_limit": MAX_CONCURRENT_REQUESTS
    }

@app.get("/stats")
def stats():
    """服务统计"""
    return {
        "active_requests": dict(active_requests),
        "concurrent_limit": MAX_CONCURRENT_REQUESTS
    }

# ─── 照片对比 ───
@app.post("/compare/photos")
async def compare_photos(request: Request, file1: UploadFile = File(None), file2: UploadFile = File(None)):
    req_id = get_request_id(request)
    active_requests[req_id] = {"start_time": time.time(), "type": "compare_photos"}

    logger.info(f"[{req_id}] 照片对比请求: {file1.filename if file1 else None} vs {file2.filename if file2 else None}")

    try:
        async with request_semaphore:
            if not file1 or not file2:
                return JSONResponse(status_code=400, content={"error": "未收到文件", "request_id": req_id})

            try:
                img1_bytes = await asyncio.wait_for(file1.read(), timeout=REQUEST_TIMEOUT)
                img2_bytes = await asyncio.wait_for(file2.read(), timeout=REQUEST_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"[{req_id}] 文件读取超时")
                return JSONResponse(status_code=408, content={"error": "文件读取超时", "request_id": req_id})

            logger.info(f"[{req_id}] 文件大小: {len(img1_bytes)} bytes, {len(img2_bytes)} bytes")

            emb1, bbox1 = get_face_embedding(img1_bytes, req_id)
            emb2, bbox2 = get_face_embedding(img2_bytes, req_id)

            if emb1 is None:
                return JSONResponse(status_code=400, content={"error": "第一张照片未检测到人脸", "request_id": req_id})
            if emb2 is None:
                return JSONResponse(status_code=400, content={"error": "第二张照片未检测到人脸", "request_id": req_id})

            similarity = cosine_similarity(emb1, emb2)

            if similarity >= 0.6:
                result = "极有可能同一人"
            elif similarity >= 0.4:
                result = "可能同一人"
            else:
                result = "不是同一人"

            logger.info(f"[{req_id}] 对比完成: {similarity:.4f} - {result}")

            return {
                "request_id": req_id,
                "similarity": round(similarity, 4),
                "similarity_percent": f"{similarity:.2%}",
                "result": result,
                "face1_bbox": bbox1,
                "face2_bbox": bbox2
            }
    except Exception as e:
        logger.error(f"[{req_id}] 处理失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "request_id": req_id})
    finally:
        active_requests.pop(req_id, None)

# ─── 人脸检测 ───
@app.post("/detect")
async def detect_face(request: Request, file: UploadFile = File(...)):
    req_id = get_request_id(request)
    active_requests[req_id] = {"start_time": time.time(), "type": "detect"}

    logger.info(f"[{req_id}] 人脸检测请求: {file.filename}")

    try:
        async with request_semaphore:
            try:
                img_bytes = await asyncio.wait_for(file.read(), timeout=REQUEST_TIMEOUT)
            except asyncio.TimeoutError:
                return JSONResponse(status_code=408, content={"error": "文件读取超时", "request_id": req_id})

            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return JSONResponse(status_code=400, content={"error": "无法读取图片", "request_id": req_id})

            faces = face_app.get(img)
            results = []
            for face in faces:
                results.append({
                    "bbox": face.bbox.tolist(),
                    "age": int(face.age),
                    "gender": "男" if face.gender == 1 else "女",
                    "det_score": float(face.det_score) if hasattr(face, 'det_score') else None
                })

            logger.info(f"[{req_id}] 检测完成: {len(results)} 个人脸")

            return {"request_id": req_id, "count": len(results), "faces": results}
    except Exception as e:
        logger.error(f"[{req_id}] 处理失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "request_id": req_id})
    finally:
        active_requests.pop(req_id, None)

# ─── 视频对比（SSE 流式进度）───
@app.post("/compare/video-sse")
async def compare_video_sse(
    request: Request,
    background_tasks: BackgroundTasks,
    target: UploadFile = File(...),
    video: UploadFile = File(...),
    threshold: float = Form(0.4),
    sample_interval: int = Form(30)
):
    req_id = get_request_id(request)
    active_requests[req_id] = {"start_time": time.time(), "type": "video_sse"}

    logger.info(f"[{req_id}] 视频对比请求: target={target.filename}, video={video.filename}")

    async def event_generator():
        nonlocal threshold, sample_interval
        temp_video = None

        try:
            async with request_semaphore:
                # Step 1: 读取目标人脸
                yield f"event: progress\ndata: {json.dumps({'step': 'loading_target', 'percent': 5, 'message': '正在加载目标照片...', 'request_id': req_id})}\n\n"

                try:
                    target_bytes = await asyncio.wait_for(target.read(), timeout=REQUEST_TIMEOUT)
                except asyncio.TimeoutError:
                    yield f"event: error\ndata: {json.dumps({'error': '目标照片读取超时', 'request_id': req_id})}\n\n"
                    return

                target_emb, _ = get_face_embedding(target_bytes, req_id)
                if target_emb is None:
                    yield f"event: error\ndata: {json.dumps({'error': '目标照片未检测到人脸', 'request_id': req_id})}\n\n"
                    return

                yield f"event: progress\ndata: {json.dumps({'step': 'loading_video', 'percent': 10, 'message': '正在加载视频文件...', 'request_id': req_id})}\n\n"

                # Step 2: 保存视频到临时文件
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                try:
                    video_bytes = await asyncio.wait_for(video.read(), timeout=REQUEST_TIMEOUT)
                except asyncio.TimeoutError:
                    yield f"event: error\ndata: {json.dumps({'error': '视频文件读取超时', 'request_id': req_id})}\n\n"
                    return

                temp_video.write(video_bytes)
                temp_video.write(target_bytes)
                temp_video.close()

                cap = cv2.VideoCapture(temp_video.name)
                if not cap.isOpened():
                    yield f"event: error\ndata: {json.dumps({'error': '无法打开视频文件', 'request_id': req_id})}\n\n"
                    return

                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0

                logger.info(f"[{req_id}] 视频信息: {total_frames} 帧, {fps} FPS, {duration:.2f}s")

                yield f"event: progress\ndata: {json.dumps({'step': 'analyzing', 'percent': 15, 'message': f'开始分析... 视频共 {total_frames} 帧', 'total_frames': total_frames, 'duration': round(duration, 2), 'request_id': req_id})}\n\n"

                matches = []
                frame_idx = 0
                last_report_percent = 15

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % sample_interval == 0:
                        faces = face_app.get(frame)
                        for face in faces:
                            emb = face.embedding
                            sim = cosine_similarity(target_emb, emb)
                            if sim >= threshold:
                                timestamp = frame_idx / fps if fps > 0 else 0
                                matches.append({
                                    "frame": frame_idx,
                                    "timestamp": round(timestamp, 2),
                                    "time_str": format_time(timestamp),
                                    "similarity": round(sim, 4),
                                    "similarity_percent": f"{sim:.2%}",
                                    "bbox": face.bbox.tolist()
                                })

                        if total_frames > 0:
                            current_percent = 15 + int((frame_idx / total_frames) * 80)
                            if current_percent >= last_report_percent + 5:
                                last_report_percent = current_percent
                                yield f"event: progress\ndata: {json.dumps({'step': 'analyzing', 'percent': min(current_percent, 95), 'message': f'已分析 {frame_idx}/{total_frames} 帧，找到 {len(matches)} 处匹配', 'frame': frame_idx, 'total_frames': total_frames, 'matches': len(matches), 'request_id': req_id})}\n\n"

                    frame_idx += 1

                cap.release()

                logger.info(f"[{req_id}] 分析完成: {len(matches)} 处匹配")

                yield f"event: progress\ndata: {json.dumps({'step': 'done', 'percent': 100, 'message': '分析完成', 'request_id': req_id})}\n\n"

                final_result = {
                    "request_id": req_id,
                    "total_matches": len(matches),
                    "video_duration": round(duration, 2),
                    "video_fps": round(fps, 2),
                    "total_frames": total_frames,
                    "sample_interval": sample_interval,
                    "threshold": threshold,
                    "matches": matches
                }
                yield f"event: result\ndata: {json.dumps(final_result, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"[{req_id}] 处理失败: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e), 'request_id': req_id})}\n\n"
        finally:
            if temp_video and os.path.exists(temp_video.name):
                try:
                    os.unlink(temp_video.name)
                except Exception as e:
                    logger.warning(f"[{req_id}] 清理临时文件失败: {e}")
            active_requests.pop(req_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ─── 导出分析报告 ───
@app.post("/export/report")
async def export_report(
    total_matches: int = Form(...),
    video_duration: float = Form(...),
    video_fps: float = Form(...),
    total_frames: int = Form(...),
    sample_interval: int = Form(...),
    threshold: float = Form(...),
    matches_json: str = Form("[]")
):
    try:
        matches = json.loads(matches_json)
    except:
        matches = []

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_match_frames = len(matches)
    if matches:
        max_sim = max(m['similarity'] for m in matches)
        avg_sim = sum(m['similarity'] for m in matches) / len(matches)
        high_matches = [m for m in matches if m['similarity'] >= 0.6]
    else:
        max_sim = 0
        avg_sim = 0
        high_matches = []

    matches_html = ""
    if matches:
        for i, m in enumerate(matches, 1):
            color = '#0abde3' if m['similarity'] >= 0.6 else '#feca57'
            matches_html += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{m['time_str']}</strong></td>
                <td>{m['frame']}</td>
                <td style="color:{color}; font-weight:bold">{m['similarity_percent']}</td>
                <td style="color:{color}">{'✅ 高置信' if m['similarity'] >= 0.6 else '🟡 中置信'}</td>
            </tr>"""
    else:
        matches_html = "<tr><td colspan='5' style='text-align:center;color:#999;padding:20px;'>未找到匹配人脸</td></tr>"

    report_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>人脸识别分析报告</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; background:#f5f5f5; padding:20px; }}
    .container {{ max-width:900px; margin:0 auto; background:white; border-radius:12px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.1); }}
    .header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:30px; text-align:center; }}
    .header h1 {{ font-size:28px; margin-bottom:8px; }}
    .header p {{ opacity:0.9; font-size:14px; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:#eee; }}
    .summary-item {{ background:white; padding:20px; text-align:center; }}
    .summary-item .value {{ font-size:32px; font-weight:bold; color:#667eea; }}
    .summary-item .label {{ color:#666; font-size:13px; margin-top:5px; }}
    .stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:15px; padding:25px; background:#f9f9f9; }}
    .stat-card {{ background:white; border-radius:8px; padding:15px; text-align:center; border:1px solid #eee; }}
    .stat-card .num {{ font-size:24px; font-weight:bold; }}
    .stat-card .desc {{ font-size:12px; color:#888; margin-top:4px; }}
    table {{ width:100%; border-collapse:collapse; margin:0 25px 25px; }}
    th {{ background:#667eea; color:white; padding:12px 15px; text-align:left; font-size:14px; }}
    td {{ padding:12px 15px; border-bottom:1px solid #eee; font-size:14px; }}
    tr:hover {{ background:#f5f7ff; }}
    .footer {{ text-align:center; padding:20px; color:#999; font-size:12px; border-top:1px solid #eee; }}
    @media print {{ body {{ background:white; }} .container {{ box-shadow:none; }} }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🎯 人脸识别分析报告</h1>
        <p>生成时间：{now}</p>
    </div>

    <div class="summary">
        <div class="summary-item">
            <div class="value">{total_match_frames}</div>
            <div class="label">匹配次数</div>
        </div>
        <div class="summary-item">
            <div class="value">{video_duration:.1f}s</div>
            <div class="label">视频时长</div>
        </div>
        <div class="summary-item">
            <div class="value">{total_frames}</div>
            <div class="label">总帧数</div>
        </div>
        <div class="summary-item">
            <div class="value">{video_fps:.1f}</div>
            <div class="label">帧率 FPS</div>
        </div>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="num" style="color:#0abde3">{max_sim:.2%}</div>
            <div class="desc">最高相似度</div>
        </div>
        <div class="stat-card">
            <div class="num" style="color:#667eea">{avg_sim:.2%}</div>
            <div class="desc">平均相似度</div>
        </div>
        <div class="stat-card">
            <div class="num" style="color:#ff6b6b">{len(high_matches)}</div>
            <div class="desc">高置信匹配 (≥60%)</div>
        </div>
    </div>

    <h3 style="margin:0 25px 15px; color:#333;">📋 详细匹配记录</h3>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>时间戳</th>
                <th>帧号</th>
                <th>相似度</th>
                <th>置信度</th>
            </tr>
        </thead>
        <tbody>{matches_html}</tbody>
    </table>

    <div style="margin:0 25px 25px; padding:15px; background:#f5f7ff; border-radius:8px; font-size:13px; color:#666;">
        <strong>⚙️ 分析参数：</strong>
        相似度阈值={threshold} | 采样间隔={sample_interval}帧 |
        阈值标准: ≥60% 高置信 | 40%-60% 中置信
    </div>

    <div class="footer">
        由人脸识别相似度分析工具生成 | InsightFace ArcFace 模型
    </div>
</div>
</body>
</html>"""

    return HTMLResponse(content=report_html.encode('utf-8'))

@app.post("/shutdown")
def shutdown():
    """安全关闭服务"""
    import threading
    def kill():
        logger.info("收到关闭指令，5秒后退出...")
        time.sleep(5)
        os._exit(0)
    threading.Thread(target=kill, daemon=True).start()
    return {"message": "服务正在关闭", "wait_seconds": 5}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
