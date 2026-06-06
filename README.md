# Gender Detection API

A two-tier API for detecting gender in images. A **Python (FastAPI)** backend performs face analysis using InsightFace, and a **Node.js (Express)** frontend proxies requests to it.

## Architecture

```
Client  ──▶  Node.js (:3000)  ──▶  Python (:5000)  ──▶  InsightFace model
              /api/detect              /api/detect
              /api/health              /api/health
```

## Prerequisites

| Tool       | Version           | Check               |
| ---------- | ----------------- | -------------------- |
| **Node.js** | 16 or later       | `node -v`            |
| **npm**     | comes with Node   | `npm -v`             |
| **Python**  | 3.9 - 3.12      | `python --version`   |
| **pip**     | comes with Python | `pip --version`      |

## Setup

### 1. Python backend

```bash
pip install fastapi "uvicorn[standard]" insightface onnxruntime opencv-python-headless
```

> **Note:** On first run InsightFace will download the `buffalo_l` model (~300 MB). An internet connection is required for that initial download.

### 2. Node.js frontend

```bash
npm install express node-fetch@2
```

## Running

Start both servers — **Python first**, then Node.

### Start the Python server

```bash
python server.py
```

Runs on **http://localhost:5000** by default.

### Start the Node.js server

```bash
node server.js
```

Runs on **http://localhost:3000** by default.

## Usage

### Detect gender in an image

```
GET http://localhost:3000/api/detect?path=C:/photos/face.jpg
```

`path` must be an **absolute file path** accessible to the Python server.

Example response:

```json
{
  "path": "C:/photos/face.jpg",
  "faces": [
    {
      "gender": "Woman",
      "confidence": 0.9412,
      "scores": { "Woman": 1.0, "Man": 0.0 },
      "region": { "x": 120, "y": 80, "w": 200, "h": 250 }
    }
  ],
  "count": 1,
  "detection_time_seconds": 0.234
}
```

### Health check

```
GET http://localhost:3000/api/health
```

```json
{ "node": "ok", "python": "ok" }
```

## Environment Variables

| Variable      | Default                  | Description                          |
| ------------- | ------------------------ | ------------------------------------ |
| `PORT`        | `3000`                   | Node.js server port                  |
| `PYTHON_API`  | `http://localhost:5000`  | URL of the Python backend            |
