from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import insightface
import cv2
import numpy as np
import tempfile
import os
import asyncio
import uuid
import json
from collections import defaultdict
from starlette.responses import StreamingResponse

app = FastAPI(title="Face Similarity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型
face_app = None

@app.on_event("startup")
def load_model():
    global face_app
    print("Loading InsightFace model...")
    # 使用本地模型目录，避免运行时联网下载
    model_dir = os.path.join(os.path.expanduser("~"), ".insightface", "models")
    os.makedirs(model_dir, exist_ok=True)
    face_app = insightface.app.FaceAnalysis(root=model_dir)
    face_app.prepare(ctx_id=0)
    print("Model loaded.")

def get_face_embedding(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    faces = face_app.get(img)
    if not faces:
        return None, None
    return faces[0].embedding, faces[0].bbox.tolist()

def cosine_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

@app.get("/")
def root():
    return {"status": "running", "message": "人脸识别服务运行中"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ─── 照片对比 ───
@app.post("/compare/photos")
async def compare_photos(file1: UploadFile = File(None), file2: UploadFile = File(None)):
    print(f"compare_photos: file1={file1.filename if file1 else None}, file2={file2.filename if file2 else None}")
    if not file1 or not file2:
        return JSONResponse(status_code=400, content={"error": "未收到文件"})
    try:
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()
        print(f"file sizes: file1={len(img1_bytes)} bytes, file2={len(img2_bytes)} bytes")

        emb1, bbox1 = get_face_embedding(img1_bytes)
        emb2, bbox2 = get_face_embedding(img2_bytes)

        if emb1 is None:
            return JSONResponse(status_code=400, content={"error": "第一张照片未检测到人脸"})
        if emb2 is None:
            return JSONResponse(status_code=400, content={"error": "第二张照片未检测到人脸"})

        similarity = cosine_similarity(emb1, emb2)

        if similarity >= 0.6:
            result = "极有可能同一人"
        elif similarity >= 0.4:
            result = "可能同一人"
        else:
            result = "不是同一人"

        return {
            "similarity": round(similarity, 4),
            "similarity_percent": f"{similarity:.2%}",
            "result": result,
            "face1_bbox": bbox1,
            "face2_bbox": bbox2
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ─── 人脸检测 ───
@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "无法读取图片"})

        faces = face_app.get(img)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.tolist(),
                "age": int(face.age),
                "gender": "男" if face.gender == 1 else "女",
                "det_score": float(face.det_score) if hasattr(face, 'det_score') else None
            })

        return {"count": len(results), "faces": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ─── 视频对比（SSE 流式进度）───
@app.post("/compare/video-sse")
async def compare_video_sse(
    target: UploadFile = File(...),
    video: UploadFile = File(...),
    threshold: float = Form(0.4),
    sample_interval: int = Form(30)
):
    """流式视频分析，实时推送进度，返回最终结果"""
    temp_target = None
    temp_video = None

    async def event_generator():
        nonlocal temp_target, temp_video
        try:
            # Step 1: 读取目标人脸
            yield f"event: progress\ndata: {json.dumps({'step': 'loading_target', 'percent': 5, 'message': '正在加载目标照片...'})}\n\n"
            target_bytes = await target.read()
            target_emb, _ = get_face_embedding(target_bytes)
            if target_emb is None:
                yield f"event: error\ndata: {json.dumps({'error': '目标照片未检测到人脸'})}\n\n"
                return

            yield f"event: progress\ndata: {json.dumps({'step': 'loading_video', 'percent': 10, 'message': '正在加载视频文件...'})}\n\n"

            # Step 2: 保存视频到临时文件
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            video_bytes = await video.read()
            temp_video.write(video_bytes)
            temp_video.write(target_bytes)  # 复用同一文件记录原名
            temp_video.close()

            cap = cv2.VideoCapture(temp_video.name)
            if not cap.isOpened():
                yield f"event: error\ndata: {json.dumps({'error': '无法打开视频文件'})}\n\n"
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            yield f"event: progress\ndata: {json.dumps({'step': 'analyzing', 'percent': 15, 'message': f'开始分析... 视频共 {total_frames} 帧', 'total_frames': total_frames, 'duration': round(duration, 2)})}\n\n"

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

                    # 每5%进度推送一次
                    if total_frames > 0:
                        current_percent = 15 + int((frame_idx / total_frames) * 80)
                        if current_percent >= last_report_percent + 5:
                            last_report_percent = current_percent
                            yield f"event: progress\ndata: {json.dumps({'step': 'analyzing', 'percent': min(current_percent, 95), 'message': f'已分析 {frame_idx}/{total_frames} 帧，找到 {len(matches)} 处匹配', 'frame': frame_idx, 'total_frames': total_frames, 'matches': len(matches)})}\n\n"

                frame_idx += 1

            cap.release()

            yield f"event: progress\ndata: {json.dumps({'step': 'done', 'percent': 100, 'message': '分析完成'})}\n\n"

            # 最终结果
            final_result = {
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
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if temp_video and os.path.exists(temp_video.name):
                try:
                    os.unlink(temp_video.name)
                except:
                    pass

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
    """生成 HTML 分析报告"""
    try:
        matches = json.loads(matches_json)
    except:
        matches = []

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 统计
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
    .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; }}
    .badge-high {{ background:#d4edda; color:#155724; }}
    .badge-mid {{ background:#fff3cd; color:#856404; }}
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
    import threading
    def kill():
        os._exit(0)
    threading.Thread(target=kill).start()
    return {"message": "服务正在关闭"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
