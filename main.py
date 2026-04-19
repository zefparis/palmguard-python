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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "mediapipe": "ready",
        "uptime": round(time.time() - START_TIME, 2),
    }
