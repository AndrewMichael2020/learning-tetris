
# RL Tetris Web App in Python for Cloud Run
_Last updated: 2025-08-30T04:50:50Z

**Goal**: Implement a reinforcement learning Tetris web app in Python with a FastAPI backend, static UI, training harness, and CI/CD to Cloud Run. Integrate ideas from the CMU.16899.HW1 Java project where appropriate: feature-based afterstate evaluation, Cross-Entropy Method (CEM), and REINFORCE with a baseline.

---

## 1) Tech stack and constraints
- **Language**: Python 3.11
- **Web**: FastAPI + Uvicorn
- **RL core**: NumPy only for model math (no TF/PyTorch)
- **Realtime**: WebSocket stream of frames from the agent
- **Tests**: pytest, pytest-asyncio, httpx, websockets
- **Container**: Docker (python:3.11-slim)
- **Cloud**: Cloud Run, gcloud CLI, GitHub Actions with Workload Identity Federation
- **Determinism**: Seeded `numpy.random.Generator` everywhere
- **Performance**: Keep training route bounded to short runs for Cloud Run

---

## 2) High-level architecture
- **rl/**: Environment, features, agents (CEM and REINFORCE), training loop, policy I/O
- **app/**: FastAPI app, schemas, config, static UI (HTML+JS+CSS), WebSocket stream
- **tests/**: Unit tests for env, features, agents, and API
- **ci/**: GitHub Actions workflow for build and deploy
- **container**: Dockerfile optimized for Cloud Run

```
rl-tetris/
  pyproject.toml
  README.md
  .gitignore
  .dockerignore
  Dockerfile
  cloudrun.yaml
  app/
    __init__.py
    main.py
    config.py
    schemas.py
    static/
      index.html
      app.js
      styles.css
  rl/
    __init__.py
    tetris_env.py
    afterstate.py
    features.py
    cem_agent.py
    reinforce_agent.py
    policy_store.py
    train.py
    eval.py
  tests/
    test_env.py
    test_features.py
    test_cem.py
    test_reinforce.py
    test_api.py
  .github/workflows/deploy_cloud_run.yaml
```

---

## 3) Algorithmic choices (from CMU.16899.HW1 ideas)
Use a **feature-based policy** on Tetris afterstates. Borrow these ideas:
- **Afterstate evaluation**: evaluate the board after placing and locking the piece. This simplifies credit assignment.
- **Feature vector**: aggregate height, number of holes, bumpiness, wells, completed lines, max height. Add row and column transitions if desired.
- **Policy types**:
  - **CEM**: population search over linear weights or small MLP weights. Elite fraction 0.2 by default. Add Gaussian noise with variance decay.
  - **REINFORCE**: stochastic policy with a baseline to reduce variance. Use an exponential moving average baseline or value-function baseline over features.
- Rewards: +10 per cleared line, small negative step cost, small positive for lower bumpiness or fewer holes via reward shaping.

---

## 4) Copilot-ready file-by-file prompts

### 4.1 `pyproject.toml`
Ask Copilot to create a Hatch-based project file:
```toml
[project]
name = "rl-tetris"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi==0.112.2",
  "uvicorn[standard]==0.30.6",
  "numpy==1.26.4",
  "pydantic==2.8.2",
  "jinja2==3.1.4",
  "httpx==0.27.2",
  "pytest==8.3.2",
  "pytest-asyncio==0.23.8",
  "websockets==12.0"
]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

`.gitignore` should include: `.venv`, `__pycache__`, `.pytest_cache`, `.idea`, `.vscode`, `dist`, `build`, `node_modules`, `.DS_Store`.
`.dockerignore` should mirror `.gitignore` and also exclude `.git`, `tests`, `*.md` when building images.

---

### 4.2 `rl/tetris_env.py`
Copilot prompt:
- Implement a 10x20 Tetris environment.
- API: `reset(seed: int | None) -> np.ndarray`, `step(action: int) -> tuple[np.ndarray, float, bool, dict]`, `legal_actions() -> list[int]`, `render(mode="rgb_array") -> np.ndarray` (uint8 HxWx3).
- Actions: `0=left, 1=right, 2=rotate, 3=drop, 4=soft_drop, 5=noop`.
- Use a 7-bag distribution or simple RNG piece sampling. Deterministic seeding.
- Rewards: base + shaping. Include in `info`: `lines_cleared`, `score`, `holes`, `height`.
- Invariants to satisfy: no overlaps at lock time, pieces stay in bounds, line clears correct.

---

### 4.3 `rl/afterstate.py`
- Implement helper to enumerate all legal placements and rotations for the current piece and return the resulting afterstates and actions. This will be used by CEM and REINFORCE to score options.

---

### 4.4 `rl/features.py`
- Implement `board_to_features(board: np.ndarray) -> np.ndarray` that returns a fixed-length vector with fields: per-column height (10), holes, aggregate height, bumpiness, wells, max height, completed lines potential.
- Normalize to stable ranges. Provide helper `holes(board)`, `bumpiness(board)`, `wells(board)` for testing.
- Provide `reward_shaping(prev_board, next_board)` that adds a small positive term when holes and bumpiness decrease.

---

### 4.5 `rl/cem_agent.py`
- Implement a policy over features using either linear weights or a tiny MLP (64->32). Start with linear for speed.
- Evolution loop:
  - Population size: 50 (configurable).
  - Elite fraction: 0.2.
  - Gaussian noise with variance decay per generation.
  - Fitness: average score plus 0.1 * lines cleared across 1–3 episodes per candidate.
- Method signatures:
  - `class CEMPolicy:` with `predict(board, legal_actions) -> int` and `weights`.
  - `def evolve(env_factory, generations: int, seed: int, out_path: str, episodes_per_candidate: int = 1) -> dict`.
- Use `policy_store.py` for `.npz` save/load of weights.
- Ensure deterministic behavior with seed.

---

### 4.6 `rl/reinforce_agent.py`
- Stochastic policy with softmax over linear feature preferences.
- Learning: REINFORCE with baseline (EMA of returns or separate value head). Entropy bonus optional.
- Expose `train(env_factory, episodes, seed, out_path)` and `predict(board, legal_actions)`.
- Keep hyperparameters small for the CI smoke test.

---

### 4.7 `rl/policy_store.py`
- `save_policy(weights: dict[str, np.ndarray], path: str)`
- `load_policy(path: str) -> dict[str, np.ndarray]`
- Default location: `policies/best.npz`. Ensure the folder exists.

---

### 4.8 `rl/train.py`
- CLI using `argparse`:
  - `--algo [cem|reinforce]`
  - `--generations` or `--episodes`
  - `--episodes-per-candidate`
  - `--seed`
  - `--out policies/best.npz`
- Logs per generation or per 100 episodes. Save metrics JSON next to policy.

---

### 4.9 `rl/eval.py`
- CLI to evaluate a policy for N episodes and print summary stats. Optionally write PNG frames for the best episode to `out/`.

---

### 4.10 `app/main.py`, `app/config.py`, `app/schemas.py`
- **Routes**:
  - `GET /api/health` -> `{"status":"ok"}`
  - `POST /api/play` body: `{"episodes":1, "seed":null, "algo":"cem"|"reinforce"}` -> returns `{"total_lines":int, "avg_score":float}` and optional reduced frames.
  - `POST /api/train` available only if `TRAIN_ENABLED=true`. Limit to small runs inside Cloud Run (for example, generations<=5 or episodes<=200).
- **WebSocket**: `/ws/stream` streams `{"frame":"<base64-png>","lines":int,"score":float,"step":int}` at 10–20 fps.
- **Static UI**: serve `index.html`, `app.js`, `styles.css` at `/`.
- Load policy on startup. If missing, use a random policy. Use `numpy.random.default_rng` seeded from env for demo repeatability.
- Config via env: `PORT`, `TRAIN_ENABLED` (default false).

---

### 4.11 `app/static/*`
- A simple canvas-based viewer showing the board on the left and stats on the right that echoes the reference screenshot.
- Buttons: **Stream Agent**, **Play Once**, **Quick Train** (disabled when `TRAIN_ENABLED=false`).

---

## 5) Tests (must run under 60 seconds)
Create these with pytest:

### 5.1 `tests/test_env.py`
- `reset()` yields 10x20 board and seeded resets match.
- `step()` with a forced line-clear setup clears at least one line.
- No overlaps at lock. Game eventually terminates.

### 5.2 `tests/test_features.py`
- Validate feature vector shape and numeric bounds.
- Monotonicity: creating a hole increases `holes()`. Increasing surface unevenness increases `bumpiness()`.

### 5.3 `tests/test_cem.py`
- Use a tiny-board patch (for example, 6x8) or cap steps to keep runtime short.
- Verify that the median score after 2–3 generations improves over random baseline.

### 5.4 `tests/test_reinforce.py`
- Run a small number of episodes with a learning rate and baseline. Expect improvement vs. random on the toy board.

### 5.5 `tests/test_api.py`
- Async tests with httpx and websockets.
- `/api/health` returns 200.
- `/api/play` returns valid stats, episodes respected.
- WebSocket `/ws/stream` yields at least 5 frames before close (cap steps).

---

## 6) Local run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## 7) Dockerfile
```Dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install -e .
COPY app ./app
COPY rl ./rl
COPY policies ./policies || true
ENV PORT=8080 TRAIN_ENABLED=false
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

`.dockerignore` should exclude `.git`, `tests`, caches, and local artifacts.

---

## 8) Cloud Run deployment

### 8.1 Manual
```bash
PROJECT_ID=your-project
REGION=northamerica-northeast1
gcloud auth login
gcloud config set project $PROJECT_ID

gcloud builds submit --tag gcr.io/$PROJECT_ID/rl-tetris
gcloud run deploy rl-tetris   --image gcr.io/$PROJECT_ID/rl-tetris   --region $REGION   --platform managed   --allow-unauthenticated
```

### 8.2 `cloudrun.yaml` (optional)
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rl-tetris
spec:
  template:
    spec:
      containers:
        - image: gcr.io/PROJECT_ID/rl-tetris:latest
          env:
            - name: TRAIN_ENABLED
              value: "false"
          ports:
            - containerPort: 8080
      containerConcurrency: 80
      timeoutSeconds: 300
```

---

## 9) GitHub Actions (build and deploy with WIF)
`.github/workflows/deploy_cloud_run.yaml`:
```yaml
name: Build & Deploy to Cloud Run
on:
  push: { branches: [main] }
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions: { contents: read, id-token: write }
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.SERVICE_ACCOUNT_EMAIL }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud auth configure-docker --quiet
      - name: Build & Push
        run: |
          IMAGE="gcr.io/${{ secrets.GCP_PROJECT_ID }}/rl-tetris:${{ github.sha }}"
          docker build -t "$IMAGE" .
          docker push "$IMAGE"
          echo "IMAGE=$IMAGE" >> $GITHUB_ENV
      - name: Deploy
        run: |
          gcloud run deploy "rl-tetris"             --image "$IMAGE"             --project "${{ secrets.GCP_PROJECT_ID }}"             --region "${{ secrets.GCP_REGION }}"             --platform managed             --allow-unauthenticated
```

Required repo secrets: `WORKLOAD_IDENTITY_PROVIDER`, `SERVICE_ACCOUNT_EMAIL`, `GCP_PROJECT_ID`, `GCP_REGION`.

---

## 10) Reproducibility checklist
- One RNG created by `np.random.default_rng(seed)` and passed to env, agents, and training loops.
- All training configs recorded in `metrics.json` and embedded in policy file metadata.
- CI runs a micro-suite with fixed seeds and small workloads.

---

## 11) Performance notes
- Use afterstate enumeration to reduce branching factor on each step.
- Cache feature vectors for identical afterstates within a move.
- Keep Cloud Run training minimal; long training should run locally or on a VM.

---

## 12) Optional enhancements
- Replace linear policy with a 2-layer MLP while keeping NumPy implementation.
- Add Prometheus metrics endpoint and basic Grafana dashboard.
- Export episode replay to MP4 using Pillow + imageio (optional dependency).

---

## 13) Deliverables
- Working web app at Cloud Run URL with streaming agent demo.
- Policies saved under `policies/` with metadata.
- Tests passing in CI in under 60 seconds.
- README covering quickstart, training, deploy, and API examples.

---

### Short Copilot mega-prompt (paste at repo root)
Implement the Python project exactly as specified in this document. Create all files under the proposed tree. Populate each module with working code following the API and test contracts. Ensure `pytest -q` passes on Python 3.11. Generate a FastAPI app with `/api/health`, `/api/play`, `/api/train` (guarded), and `/ws/stream`. Use NumPy-only policies for CEM and REINFORCE with baseline. Prepare Dockerfile and GitHub Actions workflow for Cloud Run.
