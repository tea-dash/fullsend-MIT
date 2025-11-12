import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
USER_RAW = DATA_DIR / "user" / "raw"
BASELINE_JSON = DATA_DIR / "pro" / "metrics" / "baseline.json"
MB_LITE_CKPT_REL = Path("checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")

app = FastAPI(title="Handstand Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed artifacts for the frontend at /data/*
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


@app.post("/api/analyze-sync")
async def analyze_sync(
    file: UploadFile = File(...),
    llm: str = Form("gemini"),
    gemini_model: str = Form("gemini-2.5-pro"),
) -> JSONResponse:
    job_id = str(uuid.uuid4())[:8]
    USER_RAW.mkdir(parents=True, exist_ok=True)
    raw_path = USER_RAW / f"{job_id}.mp4"

    # Save upload
    with raw_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Derive base name used by scripts
    base = f"{job_id}"

    # Pipeline (reuse scripts)
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = env.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["MOTIONBERT_DIR"] = env.get(
        "MOTIONBERT_DIR", str(ROOT / "models" / "motionbert_repo")
    )

    def run_py(args: list[str]) -> str:
        """
        Run a python script, capture output, and raise a readable error if it fails.
        """
        p = subprocess.run(
            ["python"] + args,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        if p.returncode != 0:
            cmd = "python " + " ".join(args)
            raise RuntimeError(
                f"Command failed ({p.returncode}): {cmd}\n"
                f"stdout:\n{p.stdout}\n"
                f"stderr:\n{p.stderr}\n"
            )
        return p.stdout

    # Pick device for 2D extraction robustly
    device = "cpu"
    try:
        import torch  # type: ignore

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device = "cpu"

    # Preflight checks for clearer errors
    mb_dir = Path(env["MOTIONBERT_DIR"])
    ckpt_path = mb_dir / MB_LITE_CKPT_REL
    if not ckpt_path.exists():
        return JSONResponse(
            {
                "error": (
                    "MotionBERT checkpoint not found.\n"
                    f"Expected at: {ckpt_path}\n"
                    "Download the global-lite 3D weights and place them there."
                )
            },
            status_code=500,
        )

    try:
        # 1) Normalize (process all user files; the new one is picked up)
        run_py(["scripts/preprocess_trim.py", "--split", "user"])
        # 2) 2D
        run_py(["scripts/extract_2d_ultralytics.py", "--split", "user", "--device", device])
        # 3) Convert to Halpe26
        run_py(["scripts/convert_to_halpe26.py", "--split", "user"])
        # 4) 3D lift
        run_py(["scripts/lift_3d_motionbert.py", "--split", "user", "--lite"])
        # 5) Align + metrics
        run_py(["scripts/align_and_metrics.py", "--split", "user"])
        # 6) Analyze vs baseline (build baseline if missing)
        if not BASELINE_JSON.exists():
            try:
                run_py(["scripts/build_baseline.py"])
            except Exception:
                # Baseline build may fail if pro metrics are absent; continue and let analyze raise
                pass
        run_py(["scripts/analyze_user_clip.py"])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # 6.5) Produce videos (3D render + 2D overlay) for this job
    base_30 = f"{base}_30fps"
    aligned_npz = DATA_DIR / "user" / "aligned" / f"{base_30}.npz"
    three_d_path = DATA_DIR / "user" / "aligned" / f"{base_30}_3d.mp4"
    three_d_dbg_path = DATA_DIR / "user" / "aligned" / f"{base_30}_3d_debug.mp4"
    two_d_video = DATA_DIR / "user" / "trimmed" / f"{base_30}.mp4"
    two_d_kpts = DATA_DIR / "user" / "keypoints2d" / f"{base_30}.json"

    if aligned_npz.exists():
        run_py(["scripts/render_3d_video.py", "--aligned_npz", str(aligned_npz), "--out", str(three_d_path)])
        run_py(["scripts/render_3d_video_debug.py", "--aligned_npz", str(aligned_npz), "--out", str(three_d_dbg_path)])
    # Annotated 2D overlay
    ann2d = DATA_DIR / "user" / "trimmed" / f"{base_30}_annotated.mp4"
    if two_d_video.exists() and two_d_kpts.exists():
        run_py(["scripts/annotate_video_2d.py", "--video", str(two_d_video), "--keypoints_json", str(two_d_kpts), "--out", str(ann2d)])

    # 7) LLM feedback
    feedback_path = DATA_DIR / "user" / "metrics" / "gpt_feedback.md"
    if llm == "gemini":
        # Fallback to OpenAI if Google key is not set
        if not env.get("GOOGLE_API_KEY"):
            llm = "openai"
    if llm == "gemini":
        feedback_path = DATA_DIR / "user" / "metrics" / "gemini_feedback.md"
        run_py(["scripts/gemini_feedback.py", "--model", gemini_model])
    else:
        run_py(["scripts/gpt_feedback.py"])

    # Construct output paths for the frontend
    two_d = f"/data/user/trimmed/{base_30}_annotated.mp4"
    three_d = f"/data/user/aligned/{base_30}_3d.mp4"
    three_d_debug = f"/data/user/aligned/{base_30}_3d_debug.mp4"
    metrics_json = "/data/user/metrics/user_vs_baseline.json"
    feedback_md = f"/data/user/metrics/{feedback_path.name}"

    return JSONResponse(
        {
            "job_id": job_id,
            "two_d": two_d,
            "three_d": three_d,
            "three_d_debug": three_d_debug,
            "metrics": metrics_json,
            "feedback": feedback_md,
        }
    )



