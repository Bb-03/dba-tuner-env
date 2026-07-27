---
title: DBA Tuner Env — SQL Query Rewrite Arena
emoji: 🗄️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - database
  - sql
  - duckdb
  - reinforcement-learning
---

# DBA Tuner Env 🗄️

An OpenEnv environment where an AI agent acts as a Database Administrator,
rewriting inefficient analytical SQL queries into optimised equivalents.
The agent is rewarded based on the reduction of physical execution-tree
complexity — measured deterministically via DuckDB's `EXPLAIN` operator count —
making the reward signal fully reproducible without wall-clock timing.

## Quick Start

```python
from server.dba_tuner_env_environment import DbaTunerEnvironment
from models import DbaTunerAction

env = DbaTunerEnvironment()
obs = env.reset(seed=42, level=1)
print(obs.scenario_description)

# Step 1 — examine the query plan (earns one-time +0.1 reasoning bonus)
obs = env.step(DbaTunerAction(action_type="explain"))
print(obs.query_plan)

# Step 2 — submit a rewritten query
obs = env.step(DbaTunerAction(
    action_type="rewrite",
    sql=(
        "SELECT CASE WHEN total_amount > 500 THEN 'high' "
        "WHEN total_amount >= 100 THEN 'medium' ELSE 'low' END AS segment, "
        "COUNT(*) AS cnt FROM orders GROUP BY segment"
    )
))
print(f"Reward: {obs.reward:.2f}  Reduction: {obs.metadata['cost_reduction_ratio']:.2%}")

# Step 3 — lock in the terminal score
obs = env.step(DbaTunerAction(action_type="done"))
print(f"Final score: {obs.reward:.2f}")

env.close()
```

## Building the Docker Image

```bash
# From the dba-tuner-env directory
docker build -t dba_tuner_env:latest .
```

## Deploying to Hugging Face Spaces

```bash
# Push to your HF namespace
openenv push

# Custom repo
openenv push --repo-id my-org/dba-tuner-env

# Private space
openenv push --private
```

After deployment your space exposes:

- **`/web`** — Interactive UI for manual testing
- **`/docs`** — Swagger / OpenAPI interface
- **`/health`** — Health check endpoint
- **`/ws`** — WebSocket endpoint for persistent low-latency sessions

## Environment Details

### Database Schema

```
users      (100 000 rows)  user_id*, username, email, signup_date, country
products   ( 10 000 rows)  product_id*, name, category, price, created_at
orders     (100 000 rows)  order_id, user_id*, order_date, status, total_amount
line_items (300 000 rows)  line_item_id, order_id, product_id*, quantity, unit_price

* Pareto-skewed (alpha=1.1): ~20% of IDs account for ~80% of order volume.
  Use get_stats to identify hot values before rewriting joins.
```

Data is generated fresh on every `reset()` call using DuckDB's vectorised
`generate_series`. Pass `seed=N` for a fully deterministic episode.

### Episode Structure

Each episode runs one of three task levels:

1. `reset(level=N)` — initialises schema, generates data, sets the initial SQL
2. `step(action)` — agent examines the plan, inspects stats, or submits a rewrite
3. Terminal — agent calls `done` to lock in the score, or max steps is reached

### Action Space

| `action_type` | Extra fields | Effect |
|---|---|---|
| `explain` | — | Run EXPLAIN on current SQL. Returns plan text. One-time +0.1 bonus. |
| `get_stats` | `table` | Return row count and column cardinalities for a table. One-time +0.1 bonus. |
| `rewrite` | `sql` | Submit new SQL. Correctness-checked against ground truth, then scored. |
| `done` | — | End episode. Terminal reward is locked in. |

Valid tables for `get_stats`: `users`, `products`, `orders`, `line_items`.

### Observation Space

| Field | Type | Description |
|---|---|---|
| `current_sql` | `str` | The SQL query currently being tracked |
| `query_plan` | `str` | EXPLAIN output or stats text from last action |
| `scenario_description` | `str` | Task description with schema hints |
| `scenario_level` | `int` | Task level 1–3 |
| `latency_ms` | `float` | Wall-clock execution time of current SQL (ms) |
| `total_cost` | `float` | Physical plan operator count (deterministic cost proxy) |
| `is_correct` | `bool` | False if rewrite produced wrong results — episode ends |
| `error_message` | `str` | Non-empty if last action failed |
| `done` | `bool` | True when episode has ended |
| `reward` | `float` | Step reward (may be negative mid-episode) |
| `metadata` | `dict` | `step_count`, `episode_id`, `cost_reduction_ratio`, `reasoning_bonus_paid`, `task_solved` |

### Reward Function

```
step_reward  = cost_reduction_ratio - (0.01 x step_count)
             + 0.1  [one-time, first explain or get_stats only]

terminal_reward (done=True,  task_solved=True)  = clamp(cost_reduction_ratio, 0.0, 1.0)
terminal_reward (done=True,  task_solved=False) = 0.0
terminal_reward (is_correct=False)              = 0.0

cost_reduction_ratio = (baseline_operators - current_operators) / baseline_operators

  where operators = count of _SCAN + _JOIN + GROUP + FILTER + PROJECTION + WINDOW
                    nodes in the DuckDB EXPLAIN output.
```

- Intermediate step rewards **may be negative** — this is intentional RL signal.
- The baseline operator count is frozen at `reset()` from the initial SQL.
- `task_solved` requires `cost_reduction_ratio > 0.3` — trivial changes do not score.
- Submitting an identical action twice in a row terminates the episode with a penalty.

## 3-Task Curriculum

### Task 1 (Easy) — Redundant Multi-Scan Aggregation

The initial query segments orders by revenue tier using three separate table
scans joined by `UNION ALL`. The agent must consolidate these into a single
scan that preserves the exact grouped output.

- **Baseline plan complexity:** ~6 operators (3x scan + 3x aggregation)
- **Achievable reduction:** ~67% (down to ~2 operators)
- **Max steps:** 10

### Task 2 (Medium) — Per-Row Correlated Aggregation

A correlated subquery computes the lifetime spend of each order's user by
re-scanning the orders table once per row — an O(n^2) pattern on 100k rows.
The agent must restructure the query so the per-user aggregation is computed
only once.

- **Baseline plan complexity:** ~5 operators
- **Achievable reduction:** ~40–60% depending on rewrite strategy
- **Max steps:** 15

### Task 3 (Hard) — Heavy On-Demand Aggregation

A full users-orders join with COUNT DISTINCT and SUM grouped by every user is
expensive to re-run on demand. The agent must restructure the workload so that
repeated reads of this result are served cheaply, without altering the output
seen by the caller.

- **Baseline plan complexity:** ~5 operators
- **Achievable reduction:** ~60–80% (a simple scan replaces the join + aggregation)
- **Max steps:** 15

## Running Locally

```bash
# Install dependencies
uv sync

# Start the HTTP server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run inference across all 3 tasks
export HF_TOKEN="hf_..."
python inference.py

# Smoke-test the environment directly (no server needed)
python server/dba_tuner_env_environment.py
```

## Running Inference

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"   # default
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # default

python inference.py
```

### Inference Log Format

```
[START] task=union_consolidation env=dba_tuner_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=explain() reward=0.09 done=false error=null
[STEP]  step=2 action=rewrite(...) reward=0.63 done=false error=null
[STEP]  step=3 action=done() reward=0.63 done=true error=null
[END]   success=true steps=3 score=0.63 rewards=0.09,0.63,0.63
```

Only `[START]`, `[STEP]`, and `[END]` lines are written to stdout.
All debug output goes to stderr.

## Connecting to an Existing Server

```python
from client import DbaTunerEnv
from models import DbaTunerAction

env = DbaTunerEnv(base_url="http://localhost:8000")
obs = env.reset(level=2)
obs = env.step(DbaTunerAction(action_type="explain"))
obs = env.step(DbaTunerAction(action_type="rewrite", sql="SELECT ..."))
obs = env.step(DbaTunerAction(action_type="done"))
env.close()
```

## Project Structure

```
dba-tuner-env/
├── README.md                            # This file
├── openenv.yaml                         # OpenEnv manifest
├── pyproject.toml                       # Package metadata and dependencies
├── models.py                            # Action + Observation Pydantic models
├── inference.py                         # LLM inference script (all 3 tasks)
├── client.py                            # HTTP/WebSocket client
├── Dockerfile                           # Container image definition
└── server/
    ├── app.py                           # FastAPI application (HTTP + WebSocket)
    ├── dba_tuner_env_environment.py     # Core environment logic
    └── requirements.txt                 # Server-side dependencies
```

## Use Cases

- **LLM Evaluation** — benchmark SQL reasoning and query optimisation skills across difficulty levels
- **Agent Training** — train RL agents on a real DBA workload with a continuous, non-sparse reward
- **Curriculum Learning** — three difficulty levels provide natural progressive training signal
- **Research** — fully deterministic via `reset(seed=N)`, no wall-clock timing involved

## Learn More

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [DuckDB EXPLAIN Reference](https://duckdb.org/docs/guides/meta/explain.html)
- [OpenEnv Environment Design Guide](https://github.com/meta-pytorch/OpenEnv/blob/main/README.md)

Contributors- Akshara Gupta & Bhavya Jain
