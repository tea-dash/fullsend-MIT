# Handstand MotionBERT analysis (Mac/CPU)

Static freestanding floor handstand analysis using 2D MMPose + MotionBERT 3D lift, canonical alignment (pelvis→X, hip-center→neck→Z), angle metrics, baseline vs pros, and GPT‑5 feedback.

## Quick start
1) Create a virtualenv (Mac, CPU or MPS):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

2) Install MotionBERT repo (once), and set `MOTIONBERT_DIR`:
```bash
git clone https://github.com/Walter0807/MotionBERT.git models/motionbert_repo
export MOTIONBERT_DIR="$(pwd)/models/motionbert_repo"
```

3) Download ~100 pro handstand videos (adjust queries as needed):
```bash
python scripts/download_pro.py --max-downloads 120
```

4) Normalize videos (30 FPS, 720p) into `data/*/trimmed`:
```bash
python scripts/preprocess_trim.py --split pro
```

5) Extract 2D keypoints with MMPose (HRNet-W32 + RTMDet):
```bash
python scripts/extract_2d_mmpose.py --split pro
```

6) Convert to AlphaPose-like JSON and lift to 3D with MotionBERT:
```bash
python scripts/lift_3d_motionbert.py --split pro
```

7) Align skeletons and compute metrics:
```bash
python scripts/align_and_metrics.py --split pro
```

8) Build pro baseline:
```bash
python scripts/build_baseline.py
```

9) Analyze your clip:
```bash
python scripts/preprocess_trim.py --split user
python scripts/extract_2d_mmpose.py --split user
python scripts/lift_3d_motionbert.py --split user
python scripts/align_and_metrics.py --split user
python scripts/analyze_user_clip.py
```

10) GPT‑5 feedback (requires OPENAI_API_KEY):
```bash
python scripts/gpt_feedback.py
```

## Web UI + API
Run the API (FastAPI) and the Vite React frontend.

### API
```bash
source .venv/bin/activate
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend (Vite)
```bash
cd web
npm install
npm run dev
```
Open http://127.0.0.1:5173 — upload a video; the UI calls `/api/analyze-sync` and shows outputs.

## Paths
- Data lives under `data/{pro,user}/{raw,trimmed,keypoints2d,poses3d,aligned,metrics}`.
- MotionBERT repo expected at `models/motionbert_repo` (configurable with `MOTIONBERT_DIR`).


