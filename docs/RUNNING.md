# Running StreamIQ

This guide explains how to run the simulation, DQN experiments, and the
MPEG-DASH demonstration. Run commands from the repository root.

## 1. Prerequisites

Required for the Python pipeline:

- Python 3.11 or newer
- `pip`
- FFmpeg for video processing

Required for the browser demonstration:

- Node.js and npm
- A modern desktop or mobile browser

Optional network-emulation tools:

- Linux
- Mininet
- Open vSwitch
- `iperf3`, `ping`, and `tc`

Mininet is optional and is not required for trace generation, ABR simulation,
training, or backend development.

## 2. Install Python dependencies

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

Use `PYTHONPATH=.` when running tests or modules if the repository is not
installed as a package.

## 3. Run validation tests

```bash
PYTHONPATH=. pytest -q
```

The suite covers telemetry, trace generation, the ABR environment, DQN
components, streaming services, and deployment/session behavior.

## 4. Generate network traces

Generate a deterministic trace without Mininet:

```bash
python -m network_lab.experiments.run \
  --profile recovery \
  --duration 60 \
  --interval 1 \
  --seed 42
```

Supported profiles:

`stable`, `random`, `oscillating`, `sudden_drop`, `recovery`, and `congestion`.

Trace files are written under
`experiments/network_traces/`; plots and other generated outputs are written
under `experiments/results/`.

## 5. Run the ABR environment

```bash
python -m rl_engine.environment.sample_experiment \
  --profile sudden_drop \
  --duration 60 \
  --seed 42
```

The environment uses five actions representing 240p, 360p, 480p, 720p, and
1080p. It models segment downloads, delay, packet loss, buffer consumption,
rebuffering, and reward calculation.

## 6. Train and evaluate the DQN

Run a configurable training job:

```bash
python -m rl_engine.training.run \
  --episodes 100 \
  --duration 60 \
  --seed 42 \
  --output-dir experiments/results/dqn
```

Resume from a checkpoint:

```bash
python -m rl_engine.training.run \
  --episodes 100 \
  --resume experiments/results/dqn/checkpoint_latest.pt \
  --output-dir experiments/results/dqn
```

Run the research-validation workflow with disjoint datasets and ablations:

```bash
python -m experiments.research_validation \
  --episodes 100 \
  --seeds 11 12 13 \
  --output experiments/results
```

Do not use unseen test traces for training or hyperparameter selection.

## 7. Process a video into MPEG-DASH

Only process videos that you own or are authorized to stream. Create the
source directory and place a video there:

```bash
mkdir -p video_assets/originals
```

Process a source video:

```bash
python - <<'PY'
from backend.app.services.video_processor import VideoProcessor

VideoProcessor().process(
    "video_assets/originals/example.mp4",
    "example",
)
PY
```

The command creates five aligned representations and an MPD manifest under:

```text
video_assets/processed/example/
```

FFmpeg must be installed and available on `PATH`.

## 8. Start the backend

Create local configuration:

```bash
cp .env.example .env
```

Start FastAPI from the repository root:

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Check the service:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/api/videos
```

Important endpoints:

- `GET /health` — process liveness.
- `GET /ready` — model readiness.
- `GET /api/videos` — available processed videos.
- `GET /api/videos/{video_id}/manifest` — DASH manifest.
- `POST /api/abr/decision` — DQN bitrate decision.
- `POST /api/sessions` — create a playback session.
- `GET /api/sessions/{session_id}/export.csv` — export session metrics.

If the checkpoint is unavailable, the controller returns an explicit 240p
fallback and `/ready` reports degraded readiness.

## 9. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally:

```text
http://localhost:5173
```

The player supports:

- DQN-controlled ABR
- dash.js default ABR
- manual fixed-bitrate selection

The frontend expects the backend to be available at the same origin or through
the configured development proxy. For the containerized setup, use the Docker
instructions below.

## 10. Run with Docker Compose

Docker must be installed and available as `docker`.

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

Stop the services:

```bash
docker compose down
```

Named volumes preserve processed videos and streaming session results. Model
checkpoints and source videos are intentionally not bundled into the default
images; provide them through the configured mount or deployment storage.

## 11. Run Mininet telemetry

This requires a compatible Linux host with Mininet, Open vSwitch, `iperf3`,
`ping`, and `tc`. Run as root:

```bash
sudo PYTHONPATH=. python -m network_lab.experiments.run \
  --profile recovery \
  --duration 60 \
  --interval 1 \
  --seed 42 \
  --replay
```

The legacy telemetry runner is also available:

```bash
sudo PYTHONPATH=. python -m network_lab.experiments.run_telemetry_experiment
```

Live Mininet shaping and measurement are environment-dependent and should be
validated on the target Linux kernel.

## 12. Typical end-to-end order

For a local streaming demonstration:

1. Install Python and frontend dependencies.
2. Copy `.env.example` to `.env`.
3. Place an authorized source video in `video_assets/originals/`.
4. Process the video with `VideoProcessor`.
5. Confirm the DQN checkpoint exists at the configured path.
6. Start the backend on port `8000`.
7. Start the frontend on port `5173`.
8. Select a video and ABR mode in the browser.
9. Check `/health`, `/ready`, and the playback session exports.

Browser playback, Docker startup, and production deployment must be tested on
the target machine before claiming an end-to-end deployment is verified.
