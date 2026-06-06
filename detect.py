"""
Gender Detection API Server (InsightFace).

Setup:
    pip install fastapi uvicorn[standard] insightface onnxruntime opencv-python-headless

Run:
    python server.py

API:
    GET /api/detect?path=<image_path>
    GET /api/health
"""

import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from insightface.app import FaceAnalysis

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load InsightFace model once at startup.
# buffalo_l = best accuracy/speed balance (ArcFace + detection + gender/age).
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))


def detect_gender(img_path: str) -> list[dict]:
    """Detect gender for every face in an image. Returns list of dicts."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    faces = face_app.get(img)
    if not faces:
        return []

    # Keep only the largest face (closest person).
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    results = []
    for face in [largest]:
        # InsightFace gender: 0 = Female, 1 = Male
        gender_idx = face.gender  # int: 0 or 1
        gender = "Man" if gender_idx == 1 else "Woman"
        bbox = face.bbox.astype(int)

        results.append({
            "gender": gender,
            "confidence": round(float(face.det_score), 4),
            "scores": {
                "Woman": round(1.0 - gender_idx, 4),
                "Man": round(float(gender_idx), 4),
            },
            "region": {
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "w": int(bbox[2] - bbox[0]),
                "h": int(bbox[3] - bbox[1]),
            },
        })

    return results


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/detect")
def detect(path: str = Query(default=None)):
    if not path:
        return JSONResponse({"error": "Missing 'path' query parameter"}, status_code=400)

    if not os.path.isfile(path):
        return JSONResponse({"error": f"File not found: {path}"}, status_code=400)

    try:
        start = time.perf_counter()
        faces = detect_gender(path)
        elapsed = round(time.perf_counter() - start, 3)
        return {
            "path": path,
            "faces": faces,
            "count": len(faces),
            "detection_time_seconds": elapsed,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    print("Gender Detection API running at http://localhost:5000")
    print("  GET /api/detect?path=<image_path>")
    print("  GET /api/health")
    uvicorn.run(app, host="0.0.0.0", port=5000)
