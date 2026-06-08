"""
Gender Detection API Server (DeepFace + RetinaFace).

Setup:
    pip install fastapi uvicorn[standard] deepface tf-keras opencv-python-headless

Run:
    python detect.py

API:
    GET /api/detect?path=<image_path>
    GET /api/health
"""

import os
import time
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from deepface import DeepFace

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def detect_gender(img_path: str) -> list[dict]:
    """Detect gender for every face in an image. Returns list of dicts."""
    analyses = DeepFace.analyze(
        img_path=img_path,
        actions=["gender"],
        detector_backend="retinaface",
        enforce_detection=False,
        silent=True,
    )

    if not analyses:
        return []

    # Keep only the largest face (closest person).
    largest = max(analyses, key=lambda a: a["region"]["w"] * a["region"]["h"])

    woman_pct = float(largest["gender"]["Woman"])
    man_pct = float(largest["gender"]["Man"])
    dominant = "Man" if man_pct >= woman_pct else "Woman"
    confidence = round(max(woman_pct, man_pct) / 100.0, 4)
    region = largest["region"]

    return [{
        "gender": dominant,
        "confidence": confidence,
        "scores": {
            "Woman": round(woman_pct / 100.0, 4),
            "Man": round(man_pct / 100.0, 4),
        },
        "region": {
            "x": int(region["x"]),
            "y": int(region["y"]),
            "w": int(region["w"]),
            "h": int(region["h"]),
        },
    }]


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
