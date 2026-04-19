# palmguard-python

Biometric extraction microservice for PalmGuard. Extracts and compares palm biometric vectors from images.

## Architecture

```
Browser (JPEG) → POST /biometric/extract
  → MediaPipe Hands   21 landmarks
  → OpenCV            256×256 palm ROI + Zhang-Suen skeleton
  → Fractal           Box-counting D + lacunarity × 4 lines  [12]
  → Angles            Procrustes-aligned joint angles          [18]
  → Hu Moments        4-quadrant log moments                  [12]
  → TDA               Ripser H0+H1 persistent homology        [32]
  ─────────────────────────────────────────────────────────────────
                      Combined vector Float32                  [74]
```

## API

### POST /biometric/extract
```json
{ "image_b64": "<base64 JPEG/PNG>", "session_id": "optional" }
```
Returns `vector[74]`, sub-vectors, `chirality`, `confidence`, `processing_ms`.

### POST /biometric/compare
```json
{ "vector_a": [...74 floats...], "vector_b": [...74 floats...] }
```
Returns `similarity`, `matched` (threshold 0.97), `cosine_similarity`, `l2_similarity`.

Similarity: `0.6 × cosine + 0.4 × l2_sim`

### GET /health
Returns `{ status, version, mediapipe, uptime }`.

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

## Tests

```bash
pytest tests/ -v
```

## Deploy (Render)

Push to GitHub → Render auto-deploys via `render.yaml`.

- Region: Frankfurt
- Runtime: Python 3.11
- Build: `pip install -r requirements.txt`
- Start: `gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000`

## Integration with palmguard TypeScript API

Set `PALMGUARD_PYTHON_URL=https://palmguard-python.onrender.com` in the TS API env.

Enroll flow:
1. Frontend sends base64 palm image → TS API
2. TS API → `POST /biometric/extract` → vector[74]
3. TS API stores vector in Supabase vault

Verify flow:
1. Frontend sends base64 palm image → TS API
2. TS API → `POST /biometric/extract` → vector[74]
3. TS API retrieves stored vector from Supabase
4. TS API → `POST /biometric/compare` → similarity + matched
