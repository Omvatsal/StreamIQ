# StreamIQ Deployment Guide

## Scope

The production prototype consists of the FastAPI backend and React/nginx
frontend. Mininet and Open vSwitch remain separate development-only network
experimentation infrastructure and are not placed in the production
containers.

## Local host deployment

Install Python dependencies and create configuration:

```bash
cp .env.example .env
python -m pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

In another shell, install and build the frontend:

```bash
cd frontend
npm install
npm run build
npm run dev
```

Put only authorized source videos in `video_assets/originals/` and process them
with `VideoProcessor`. Processed DASH files belong in
`video_assets/processed/<video-id>/`.

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The frontend is available at `http://localhost:5173`; the API is available at
`http://localhost:8000`. Compose persists processed videos and streaming
session results in named volumes.

The default image intentionally excludes original videos, checkpoints, and
experiment outputs. Mount or provision these assets separately. For a
production deployment, set `STREAMIQ_DQN_CHECKPOINT` to a mounted checkpoint
path and use a private image/volume policy for model files.

## Configuration

All backend settings use the `STREAMIQ_` prefix:

- `STREAMIQ_ENVIRONMENT`
- `STREAMIQ_HOST`
- `STREAMIQ_PORT`
- `STREAMIQ_CORS_ORIGINS`
- `STREAMIQ_PROCESSED_VIDEO_ROOT`
- `STREAMIQ_EXPERIMENT_ROOT`
- `STREAMIQ_DQN_CHECKPOINT`
- `STREAMIQ_LOG_LEVEL`

Never commit `.env`. Use a secret manager or deployment environment variables
for production configuration.

## Health and readiness

- `GET /health` checks process liveness.
- `GET /ready` checks whether the configured DQN checkpoint can be loaded.

Readiness is intentionally degraded when the model is unavailable. The
decision endpoint still returns an explicit 240p fallback response.

## Operations and limitations

The backend emits one JSON log object per line. Session JSON and CSV files are
written atomically/through a temporary file for the JSON record. The DQN is
loaded lazily once and guarded by a lock so concurrent first requests cannot
load duplicate models.

Browser playback, container startup, resource limits, TLS, authentication,
object storage, and horizontal model-serving have not been verified in this
workspace. Add those controls before exposing the service publicly.
