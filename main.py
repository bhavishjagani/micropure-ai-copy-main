"""
MicroPure AI - Detection backend.

This file is intentionally location-agnostic. It works whether it lives at
the project root or inside a `backend/` folder on the other laptop, because
all paths are resolved relative to this file (and its parent) at startup.
"""

from __future__ import annotations

import io
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Path resolution - works no matter where main.py lives
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SEARCH_ROOTS = [HERE, HERE.parent]  # current dir + parent (covers root + backend/)

# Preference order: a custom-trained model (best.pt) beats the generic ones.
MODEL_CANDIDATES = [
    "best.pt",
    "runs/detect/train/weights/best.pt",
    "runs/detect/train2/weights/best.pt",
    "runs/detect/train3/weights/best.pt",
    "yolov8s.pt",
    "yolov8n.pt",
]

# Upload safety limits (defense in depth - the frontend also enforces these)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024          # 15 MB
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resolve_model_path() -> Path:
    """Walk known locations and return the first existing model file."""
    for root in SEARCH_ROOTS:
        for candidate in MODEL_CANDIDATES:
            p = (root / candidate).resolve()
            if p.is_file():
                return p
    raise FileNotFoundError(
        "No YOLO weights found. Place best.pt / yolov8s.pt / yolov8n.pt "
        "next to main.py (or in the project root)."
    )


# Loaded on startup so `import main` never fails just because weights are
# missing or YOLO is slow — uvicorn can start and /health reports the error.
_model: Any = None
_model_path: Path | None = None
_model_name: str | None = None
_model_load_error: str | None = None


def _frontend_index_path() -> Path | None:
    """Where `frontend/index.html` lives (project root or parent of `backend/`)."""
    for root in (HERE, HERE.parent):
        p = root / "frontend" / "index.html"
        if p.is_file():
            return p
    return None


def _sample_images_dir() -> Path | None:
    for root in (HERE, HERE.parent):
        d = root / "dataset" / "valid" / "images"
        if d.is_dir():
            return d
    return None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _model, _model_path, _model_name, _model_load_error
    _model_load_error = None
    try:
        _model_path = _resolve_model_path()
        _model = YOLO(str(_model_path))
        _model_name = _model_path.name
        print(f"[MicroPure] Loaded model: {_model_path}")
    except Exception as exc:  # noqa: BLE001 — show any load failure in /health
        _model = None
        _model_path = None
        _model_name = None
        _model_load_error = f"{type(exc).__name__}: {exc}"
        print(f"[MicroPure] Model load failed: {_model_load_error}")
    if _frontend_index_path():
        print("[MicroPure] Open the full UI in your browser: /dashboard (same host/port as this server)")
    yield
    _model = None


def _require_model() -> Any:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "YOLO model is not loaded.",
                "error": _model_load_error,
                "fix": (
                    "Put yolov8s.pt, yolov8n.pt, or best.pt in the project folder "
                    "(same folder as main.py) and restart the server."
                ),
            },
        )
    return _model


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MicroPure AI",
    description="Microplastic detection + multi-factor water-risk assessment.",
    version="2.0.0",
    lifespan=_lifespan,
)

# CORS - kept permissive because the frontend is a static file opened locally
# (file://) or served on a random localhost port. Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Dataset thumbnails for the UI when opened from /dashboard (same origin).
_sample_dir = _sample_images_dir()
if _sample_dir is not None:
    app.mount(
        "/sample-images",
        StaticFiles(directory=str(_sample_dir)),
        name="sample_images",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify_size(area_ratio: float) -> str:
    """Bucket detections by their bbox area as a fraction of the whole image."""
    if area_ratio < 0.0005:
        return "nano"      # extremely small - the most concerning fraction
    if area_ratio < 0.002:
        return "micro"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


SIZE_RISK_WEIGHT = {
    "nano": 1.4,
    "micro": 1.2,
    "small": 1.0,
    "medium": 0.85,
    "large": 0.7,
}


def _compute_risk(detections: list[dict[str, Any]], img_w: int, img_h: int) -> dict[str, Any]:
    """
    Multi-factor risk score (0-100). One detection alone never produces 100%.

    Components (each capped):
      - count_score        (0-45)  : log10-scaled particle count
      - confidence_score   (0-25)  : weighted by how confident the model was
      - density_score      (0-20)  : particles per megapixel of image
      - size_score         (0-10)  : penalty for nano/micro fragments
    """
    n = len(detections)

    if n == 0:
        return {
            "score": 0.0,
            "level": "Safe",
            "color": "#22c55e",
            "components": {"count": 0.0, "confidence": 0.0, "density": 0.0, "size": 0.0},
            "summary": "No microplastics detected in this sample.",
            "recommendation": "Water appears clean for this sample. Continue routine monitoring.",
        }

    # Count component - logarithmic so 1 particle is small, 1000+ saturates.
    count_score = min(math.log10(n + 1) * 22.0, 45.0)

    # Confidence component - mean confidence weighted by particle area
    total_area = sum(d["area_ratio"] for d in detections) or 1e-9
    weighted_conf = sum(d["confidence"] * d["area_ratio"] for d in detections) / total_area
    avg_conf = sum(d["confidence"] for d in detections) / n
    blended_conf = (weighted_conf + avg_conf) / 2.0
    confidence_score = blended_conf * 25.0

    # Density component - particles per megapixel of source image
    megapixels = max((img_w * img_h) / 1_000_000.0, 0.1)
    density = n / megapixels
    density_score = min(density / 30.0 * 20.0, 20.0)  # 30 ppm -> full 20 pts

    # Size component - more nano/micro means worse for human health
    weight_sum = sum(SIZE_RISK_WEIGHT.get(d["size_class"], 1.0) for d in detections)
    size_factor = weight_sum / n  # ~0.7 .. 1.4
    size_score = min(max((size_factor - 0.7) / 0.7, 0.0), 1.0) * 10.0

    score = count_score + confidence_score + density_score + size_score
    score = round(min(score, 100.0), 2)

    if score < 20:
        level, color = "Low", "#22c55e"
        recommendation = (
            "Particle load is low. Standard household filtration is generally sufficient."
        )
    elif score < 45:
        level, color = "Moderate", "#eab308"
        recommendation = (
            "Moderate microplastic load. Consider an activated-carbon or "
            "0.2 micron filter before drinking."
        )
    elif score < 70:
        level, color = "High", "#f97316"
        recommendation = (
            "High particle concentration detected. Use multi-stage filtration "
            "(carbon + ultrafiltration) and avoid direct consumption."
        )
    else:
        level, color = "Critical", "#ef4444"
        recommendation = (
            "Critical contamination level. Do not drink without reverse-osmosis "
            "treatment and follow-up laboratory verification."
        )

    summary = (
        f"Detected {n} particle{'s' if n != 1 else ''} "
        f"at {density:.1f} per megapixel "
        f"(avg confidence {avg_conf*100:.1f}%)."
    )

    return {
        "score": score,
        "level": level,
        "color": color,
        "components": {
            "count": round(count_score, 2),
            "confidence": round(confidence_score, 2),
            "density": round(density_score, 2),
            "size": round(size_score, 2),
        },
        "density_per_megapixel": round(density, 2),
        "average_confidence": round(avg_conf, 4),
        "weighted_confidence": round(weighted_conf, 4),
        "summary": summary,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/dashboard", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Full web UI (not JSON — open this in a browser)."""
    index = _frontend_index_path()
    if index is None:
        raise HTTPException(
            status_code=404,
            detail="frontend/index.html not found next to main.py or in the parent folder.",
        )
    return FileResponse(index, media_type="text/html")


@app.get("/")
def root() -> dict[str, Any]:
    ok = _model is not None
    dash = "/dashboard" if _frontend_index_path() else None
    return {
        "name": "MicroPure AI",
        "status": "ok" if ok else "degraded",
        "model": _model_name,
        "model_path": str(_model_path) if _model_path else None,
        "model_error": _model_load_error,
        "endpoints": ["/health", "/predict/", "/dashboard"],
        "dashboard": dash,
        "hint": (
            "JSON is the API. Open http://127.0.0.1:PORT/dashboard in a browser for the full UI."
            if dash
            else None
        ),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    ok = _model is not None
    return {
        "status": "ok" if ok else "degraded",
        "model": _model_name,
        "model_error": _model_load_error,
        "version": app.version,
    }


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.20, ge=0.05, le=0.95, description="Confidence threshold"),
    iou: float = Query(0.30, ge=0.10, le=0.90, description="IOU NMS threshold"),
    imgsz: int = Query(1024, ge=320, le=1536, description="Inference image size"),
) -> JSONResponse:
    # ---- input validation ------------------------------------------------
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported extension: {ext}")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
        )

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    img_w, img_h = img.size

    model = _require_model()

    # ---- inference -------------------------------------------------------
    started = time.perf_counter()
    results = model.predict(
        img,
        verbose=False,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=1000,
        agnostic_nms=True,
    )[0]
    inference_ms = round((time.perf_counter() - started) * 1000.0, 1)

    # ---- shape detections ------------------------------------------------
    detections: list[dict[str, Any]] = []
    boxes = results.boxes
    if boxes is not None and boxes.xyxy is not None:
        for box, cls, conf_t in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            w = max(x2 - x1, 0.0)
            h = max(y2 - y1, 0.0)
            area = w * h
            area_ratio = area / float(img_w * img_h) if img_w and img_h else 0.0
            size_class = _classify_size(area_ratio)
            detections.append({
                "class_id": int(cls),
                "class_name": model.names.get(int(cls), str(int(cls))) if hasattr(model, "names") else str(int(cls)),
                "confidence": float(conf_t),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "width": round(w, 2),
                "height": round(h, 2),
                "area": round(area, 2),
                "area_ratio": round(area_ratio, 6),
                "size_class": size_class,
            })

    # Sort highest-confidence first so the UI can preview the strongest hits
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ---- risk + extras ---------------------------------------------------
    risk = _compute_risk(detections, img_w, img_h)

    size_breakdown: dict[str, int] = {k: 0 for k in SIZE_RISK_WEIGHT}
    for d in detections:
        size_breakdown[d["size_class"]] = size_breakdown.get(d["size_class"], 0) + 1

    confidence_buckets = {"0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for d in detections:
        c = d["confidence"]
        if c < 0.4:
            confidence_buckets["0.2-0.4"] += 1
        elif c < 0.6:
            confidence_buckets["0.4-0.6"] += 1
        elif c < 0.8:
            confidence_buckets["0.6-0.8"] += 1
        else:
            confidence_buckets["0.8-1.0"] += 1

    return JSONResponse(content={
        "model": _model_name,
        "image": {
            "width": img_w,
            "height": img_h,
            "megapixels": round((img_w * img_h) / 1_000_000.0, 3),
        },
        "params": {"conf": conf, "iou": iou, "imgsz": imgsz},
        "inference_ms": inference_ms,
        "predictions": detections,
        "stats": {
            "count": len(detections),
            "size_breakdown": size_breakdown,
            "confidence_buckets": confidence_buckets,
        },
        "risk": risk,
    })


if __name__ == "__main__":
    # Run from the folder that contains main.py — avoids "could not import module main"
    # when the shell's working directory is wrong for `uvicorn main:app`.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
