---
title: DBA Tuner Env — SQL Performance Optimisation Arena
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

**An AI agent acts as a Database Administrator (DBA) to optimise DuckDB query
performance across 3 difficulty levels — from simple index creation to
join optimisation and selective range scans.**

## Overview

| Attribute | Value |
|-----------|-------|
| **Domain** | Database performance, SQL optimisation |
| **Backend** | DuckDB (in-memory, `:memory:`) |
| **Dataset** | 100k-row Pareto-skewed (α=1.1) e-commerce data |
| **Tasks** | 3 (Easy → Hard) |
| **Reward** | Continuous [0.0 → 1.0], deterministic plan-cost-based |
| **Reproducible** | `reset(seed=N)` produces identical datasets every time |
| **Hardware** | 8 GB RAM / 2 vCPU |

---

## Quick Start

```python
from server.dba_tuner_env_environment import DbaTunerEnvironment
from models import DbaTunerAction

env = DbaTunerEnvironment()
obs = env.reset(seed=42, level=1)     # deterministic dataset from seed
print(obs.scenario_description)

# Thinking trajectory: explain → get_stats → add_index
obs = env.step(DbaTunerAction(action_type="explain"))
print(obs.query_plan)             # EXPLAIN ANALYZE output

obs = env.step(DbaTunerAction(action_type="get_stats", table="orders"))
print(obs.query_plan)             # column stats with est_index_size

obs = env.step(DbaTunerAction(
    action_type="add_index", table="orders", column="user_id"
))
print(f"Reward: {obs.reward:.2f}   Cost-reduction: {obs.metadata['cost_reduction_ratio']:.2%}")

env.close()
```

---

## Database Schema

```
users      (100 000 rows)  user_id★, username, email, signup_date, country
products   ( 10 000 rows)  product_id★, name, category, price, created_at
orders     (100 000 rows)  order_id, user_id★, order_date, status, total_amount
line_items (300 000 rows)  line_item_id, order_id, product_id★, quantity, unit_price

★ Pareto-skewed (α=1.1): ~20% of IDs account for ~80% of rows.
  Run get_stats to identify hot values before choosing which columns to index.
```

---

## 3-Task Curriculum

| Task | Name | Goal |
|------|------|------|
| 1 | Point Lookup | Add index on `orders.user_id` |
| 2 | Join Optimisation | Add indexes on `orders.user_id` + `line_items.order_id` |
| 3 | Range Scan | Add index on `orders.order_date` for BETWEEN queries |

---

## Action Space

```json
{"action_type": "explain"}
{"action_type": "get_stats",  "table": "orders"}
{"action_type": "add_index",  "table": "orders",  "column": "user_id"}
{"action_type": "drop_index", "index_name": "idx_orders_user_id"}
{"action_type": "done"}
```

---

## Reward Function

```
Reward = CostReductionRatio                     # (baseline_cost - current_cost) / baseline_cost
       + 0.1  (one-time)                        # first explain or get_stats call
       - 0.02 × index_count                     # over-indexing penalty
       - 0.005 × storage_used_mb                # storage waste penalty
       - 0.005 × step_count                     # efficiency penalty

Cost is measured via EXPLAIN plan estimated rows (deterministic, no timing).

Terminal score (done=True):
  task_solved=True  → max(0.0, min(1.0, computed_reward))
  task_solved=False → 0.0  (explain/get_stats bonuses alone cannot win)
  is_correct=False  → 0.0  (wrong results = zero)

task_solved requires cost_reduction_ratio > 0.3 (meaningful optimisation).
```

Step rewards during an episode may be **negative** (RL signal).
The final episode score is always in **[0.0, 1.0]**.

---

## Reward Model

A typed `DbaTunerReward` Pydantic model is available for structured reward metadata:

```python
from models import DbaTunerReward

reward = DbaTunerReward(
    step_reward=0.35,
    terminal_reward=0.35,
    task_solved=True,
    cost_reduction_ratio=0.42,
    index_penalty=-0.02,
    storage_penalty=-0.004,
    step_penalty=-0.015,
    reasoning_bonus=0.1,
)
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Run the HTTP server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run inference against all 3 tasks
export HF_TOKEN="hf_..."
python inference.py

# Direct environment smoke-test (no server needed)
python server/dba_tuner_env_environment.py
```

---

## Inference Log Format

```
[START] task=simple_index env=dba_tuner_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=explain() reward=0.09 done=false error=null
[STEP] step=2 action=get_stats(orders) reward=0.04 done=false error=null
[STEP] step=3 action=add_index(orders,user_id) reward=0.29 done=false error=null
[END] success=true steps=3 score=0.29 rewards=0.09,0.04,0.29
```

Only `[START]`, `[STEP]`, and `[END]` lines are emitted on stdout.
The `[END]` score is the final terminal reward from the environment.

---

## Project Structure

```
dba-tuner-env/
├── models.py                          # Action + Observation + Reward Pydantic models
├── inference.py                       # LLM inference script (all 3 tasks)
├── client.py                          # WebSocket client (DbaTunerEnv)
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Package metadata
├── Dockerfile                         # Container build
└── server/
    ├── app.py                         # FastAPI server (HTTP + WS)
    ├── dba_tuner_env_environment.py   # Core environment logic
    └── requirements.txt               # Server dependencies
```

---

## Deployment to Hugging Face Spaces

```bash
openenv push                          # push to your HF namespace
openenv push --repo-id org/dba-env   # custom repo
openenv push --private                # private space
```

The deployed space exposes:
- **`/web`** — Interactive UI
- **`/docs`** — Swagger / OpenAPI
- **`/health`** — Health check
- **`/ws`** — WebSocket (persistent sessions)
