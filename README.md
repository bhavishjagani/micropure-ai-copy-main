# MicroPure AI

YOLOv8-based microplastic detection in microscope images of water samples,
with a multi-factor water-risk score and a modern web dashboard.

## Project layout

This repo is intentionally **flat** so it works whether `main.py` lives at the
project root or inside a `backend/` folder on another machine — paths are
resolved relative to the script at startup.

```
.
├── main.py                # FastAPI inference server (location-agnostic)
├── train.py               # YOLO training entry-point
├── csv_to_yolo.py         # Convert _annotations.csv -> YOLO labels
├── dataset.yaml           # YOLO dataset descriptor
├── dataset/
│   ├── train/{images,labels,_annotations.csv}
│   └── valid/{images,labels,_annotations.csv}
├── frontend/
│   └── index.html         # Self-contained dashboard (no build step)
├── yolov8n.pt             # Stock weights (fallback)
├── yolov8s.pt             # Stock weights (default fallback)
└── best.pt                # (optional) custom-trained weights — preferred
```

`main.py` searches for weights in this order, in the same directory as the
script and in its parent:

```
best.pt
runs/detect/train/weights/best.pt
runs/detect/train{2,3}/weights/best.pt
yolov8s.pt
yolov8n.pt
```

So if you move `main.py` into `backend/`, copy `yolov8s.pt` there too (or
leave it at the root) and it will still find a model.

## Running the backend

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**If you see `Error loading ASGI app. Could not import module 'main'`:**

1. **Wrong folder** — `uvicorn` must be started from the directory that contains `main.py`, or tell it where to look:
   ```bash
   cd /path/to/micropure-ai-copy-main
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
   Or from anywhere:
   ```bash
   uvicorn main:app --app-dir /path/to/micropure-ai-copy-main --host 127.0.0.1 --port 8000
   ```
2. **Easiest fix** — run the app directly (no `main` import path issue):
   ```bash
   cd /path/to/micropure-ai-copy-main
   python main.py
   ```
3. **See the real error** — from that same folder:
   ```bash
   python -c "import main; print('import ok')"
   ```
   If this fails, read the full traceback (missing `fastapi`, `ultralytics`, etc.).

The API will print which model it loaded, e.g.:

```
[MicroPure] Loaded model: /path/to/yolov8s.pt
```

### Endpoints

| Method | Path        | Description                                       |
| ------ | ----------- | ------------------------------------------------- |
| GET    | `/`         | Service info + chosen model                       |
| GET    | `/health`   | Heartbeat for the frontend                        |
| POST   | `/predict/` | Multipart `file=` upload. Optional query params:  |
|        |             | `conf` (0.05–0.95), `iou` (0.10–0.90), `imgsz`    |

### Response shape

```jsonc
{
  "model": "yolov8s.pt",
  "image": { "width": 640, "height": 640, "megapixels": 0.41 },
  "params": { "conf": 0.2, "iou": 0.3, "imgsz": 1024 },
  "inference_ms": 312.4,
  "predictions": [
    {
      "class_id": 0, "class_name": "microplastic",
      "confidence": 0.74,
      "bbox": [x1, y1, x2, y2],
      "width": 14.2, "height": 12.8,
      "area": 181.7, "area_ratio": 0.000443,
      "size_class": "micro"
    }
  ],
  "stats": {
    "count": 12,
    "size_breakdown": { "nano": 1, "micro": 4, "small": 5, "medium": 2, "large": 0 },
    "confidence_buckets": { "0.2-0.4": 2, "0.4-0.6": 5, "0.6-0.8": 4, "0.8-1.0": 1 }
  },
  "risk": {
    "score": 41.2, "level": "Moderate", "color": "#eab308",
    "components": { "count": 22.0, "confidence": 14.7, "density": 3.5, "size": 1.0 },
    "density_per_megapixel": 29.3,
    "average_confidence": 0.61, "weighted_confidence": 0.59,
    "summary": "Detected 12 particles at 29.3 per megapixel (avg confidence 61.0%).",
    "recommendation": "Moderate microplastic load. ..."
  }
}
```

## Risk model (why one detection ≠ 100%)

Each component is independently capped, so a single low-confidence box can
never push the score to 100. The components are:

| Component  | Range  | What it measures                                     |
| ---------- | ------ | ---------------------------------------------------- |
| Count      | 0–45   | `log10(n+1) * 22`, saturates around 1000 particles   |
| Confidence | 0–25   | Blend of mean + area-weighted mean confidence × 25   |
| Density    | 0–20   | Particles per megapixel (saturates near 30 ppm)      |
| Size       | 0–10   | Penalty for nano/micro fragments                     |

Levels: `<20` Low, `20–45` Moderate, `45–70` High, `70+` Critical.

## Frontend

**Recommended:** with the API already running, open the full dashboard in your browser:

**http://127.0.0.1:8000/dashboard** (use the same host/port as uvicorn)

That page talks to `/predict/` on the same origin, so you do not need to change Settings.

You can also open `frontend/index.html` directly from disk (`file://`) if you prefer; set the API URL in **Settings** to match your server.

`/`, `/health`, and `/docs` are JSON / API docs — not the visual app.

- Drag-drop or paste images (Ctrl/⌘ + V), or pick a sample tile.
- Adjust **Confidence**, **IOU** and **image size** before analysis.
- The Dashboard view shows the bounding-box overlay, a risk gauge,
  size/confidence/risk-component charts, a per-particle log, and exports
  to JSON/CSV/PDF (via Print).
- Past analyses (with thumbnails) are stored in `localStorage`; nothing
  is uploaded anywhere except your local backend.
- Settings panel lets you change the API base URL and the relative path
  the sample-tile loader uses.

## Training (optional)

```bash
python csv_to_yolo.py    # only the first time, to build YOLO labels
python train.py
```

After training, the new weights at `runs/detect/train/weights/best.pt`
will be picked up automatically next time you start the backend.
