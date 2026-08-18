# Running the AGE AI Backend

FastAPI service that wraps the 3-model age-estimation ensemble behind
`/upload`, `/predict`, and `/health`. See [API.md](API.md) for endpoint
details.

## 1. Prerequisites

- Python 3.9+
- Trained checkpoints present at:
  - `runs/resnet50/best_model.pt`
  - `runs/efficientnet_b0/best_model.pt`
  - `runs/mobilenet_v2/best_model.pt`

  The server starts without them but logs a warning, and `/predict` returns
  `503` until at least one checkpoint is loaded.

## 2. Install dependencies

From the `age_prediction_pipeline` directory:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r backend/requirements.txt
```

This also installs the pipeline's root `requirements.txt` (via `-r ../requirements.txt`).

## 3. Configure CORS origins (optional)

By default the API accepts requests from `http://localhost:5173` and
`http://127.0.0.1:5173` (the Vite dev server). Override with:

```bash
set FRONTEND_ORIGINS=http://localhost:5173,http://your-frontend-host   # Windows
# export FRONTEND_ORIGINS=...                                          # macOS/Linux
```

## 4. Run the server

From the `age_prediction_pipeline` directory, with the venv active:

```bash
uvicorn backend.main:app --reload --port 5000
```

- API base URL: `http://localhost:5000`
- Uploaded images are stored in `backend/uploads/` and served at `/uploads/<file>`
- GPU is used automatically if `torch.cuda.is_available()` is `True`, otherwise CPU

## 5. Verify it's up

```bash
curl http://localhost:5000/health
```

```json
{"status": "ok", "models_loaded": 3}
```

If `models_loaded` is `0`, check that the checkpoint files listed in step 1 exist.

## 6. Connect the frontend

Point the frontend at this server via `VITE_API_URL` (defaults to
`http://localhost:5000` if unset).
