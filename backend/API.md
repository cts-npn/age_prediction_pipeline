# AGE AI Backend API

Base URL (local dev): `http://localhost:5000`
Configure the frontend via `VITE_API_URL` (defaults to the above).

All endpoints are CORS-enabled for the origins listed in the `FRONTEND_ORIGINS`
env var (default: `http://localhost:5173`, `http://127.0.0.1:5173`).

---

## POST /upload

Uploads an image and stores it on the server. Returns a URL the file can be
served back from. Used opportunistically by the frontend before/alongside
prediction; a failure here is non-fatal (the UI keeps working off the local
preview).

**Request:** `multipart/form-data`

| Field   | Type | Required | Notes                          |
|---------|------|----------|---------------------------------|
| `image` | file | yes      | JPEG or PNG, max 5 MB          |

**Success — `200 OK`**
```json
{
  "imageUrl": "/uploads/3f2a9c1e8b7d4f0a9c2e1b6d7a8f9c3e.jpg"
}
```
`imageUrl` is a path relative to the API base URL — the file is served as a
static asset at `{API_BASE_URL}{imageUrl}`.

**Errors**

| Status | Body                                                   | Cause                        |
|--------|---------------------------------------------------------|-------------------------------|
| 400    | `{"detail": "Only JPG and PNG images are supported."}`  | Wrong `Content-Type`          |
| 400    | `{"detail": "Image exceeds the 5 MB size limit."}`      | File too large                |
| 400    | `{"detail": "Could not read the uploaded file as an image."}` | Corrupt/unreadable file |

---

## POST /predict

Runs face detection, alignment, and the 3-model age-estimation ensemble on
the uploaded image, returning an estimated age, a range, and a confidence
score.

**Request:** `multipart/form-data`

| Field       | Type   | Required | Notes                                                        |
|-------------|--------|----------|-----------------------------------------------------------------|
| `image`     | file   | yes      | JPEG or PNG, max 5 MB                                        |
| `image_url` | string | no       | Value returned from `/upload`; accepted but currently unused for inference (prediction always runs on the posted file) |

**Success — `200 OK`**
```json
{
  "age": 27,
  "min_age": 24,
  "max_age": 30,
  "confidence": 82.4
}
```

| Field        | Type   | Meaning                                                              |
|--------------|--------|------------------------------------------------------------------------|
| `age`        | int    | Ensemble-averaged predicted age, in years                            |
| `min_age`    | int    | Lower bound of the estimated range (never below 0)                   |
| `max_age`    | int    | Upper bound of the estimated range                                   |
| `confidence` | float  | 0–100, average top-1 classification confidence across the 3 models   |

**Errors**

| Status | Body                                                                          | Cause                                  |
|--------|--------------------------------------------------------------------------------|------------------------------------------|
| 400    | `{"detail": "Only JPG and PNG images are supported."}`                        | Wrong `Content-Type`                    |
| 400    | `{"detail": "Image exceeds the 5 MB size limit."}`                            | File too large                          |
| 400    | `{"detail": "Could not read the uploaded file as an image."}`                 | Corrupt/unreadable file                 |
| 422    | `{"detail": "No face detected in the image. Try a clearer, front-facing photo."}` | No face found by the landmark detector |
| 503    | `{"detail": "No trained model checkpoints are loaded on the server."}`        | Server started without any `runs/*/best_model.pt` |

---

## GET /health

Simple liveness/readiness check.

**Success — `200 OK`**
```json
{
  "status": "ok",
  "models_loaded": 3
}
```
`models_loaded` is `0` if no checkpoints were found at startup — `/predict`
will return `503` in that state.

---

## Frontend integration notes

- Both `/upload` and `/predict` expect `multipart/form-data`, not JSON —
  build a `FormData` and append the file under the field names above (this
  already matches `uploadPic()` / `predict()` in `App.jsx`).
- Error responses always use FastAPI's default shape: `{"detail": "<message>"}`.
  Read `err.message` after a thrown fetch error, or parse the JSON body's
  `detail` field directly if you want to show the exact server message.
- `/predict` is self-sufficient — it re-reads the posted `image` file, so the
  frontend does not need to wait on `/upload` to succeed before calling
  `/predict` (this matches the current `handlePredict` flow, which calls
  `uploadPic()` then `predict()` back-to-back regardless of the first
  result).
