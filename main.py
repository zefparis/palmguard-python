import setup_env  # noqa: F401 — must be first import to set GL/GPU env vars

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import biometric
import time

START_TIME = time.time()

app = FastAPI(title="PalmGuard Biometric Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restricted to palmguard origins only in prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(biometric.router, prefix="/biometric")


@app.on_event("startup")
async def startup_event():
    import os
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'softpipe'
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    from services.landmarks import _get_landmarker
    try:
        _get_landmarker()
        print("[palmguard-python] MediaPipe pre-warmed \u2713")
    except Exception as e:
        print(f"[palmguard-python] Pre-warm warning: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "mediapipe": "ready",
        "uptime": round(time.time() - START_TIME, 2),
    }


@app.get("/ping")
def ping():
    return {"pong": True}
