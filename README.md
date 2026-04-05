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
performance across 7 difficulty levels — from simple index creation to
materialised views, N+1 pattern fixes, and multi-query budget challenges.**

## Overview

| Attribute | Value |
|-----------|-------|
| **Domain** | Database performance, SQL optimisation |
| **Backend** | DuckDB (in-memory, `:memory:`) |
| **Dataset** | 100k-row Pareto-skewed (α=1.1) e-commerce data |
| **Levels** | 7 (Easy → Expert) |
| **Reward** | Continuous [0.0 → 1.0], multi-faceted |
| **Hardware** | 8 GB RAM / 2 vCPU |

---

## Quick Start

```python
from server.dba_tuner_env_environment import DbaTunerEnvironment
from models import DbaTunerAction

env = DbaTunerEnvironment()
obs = env.reset(level=1)          # or omit level for random pick
print(obs.scenario_description)

# Thinking trajectory: explain → get_stats → add_index
obs = env.step(DbaTunerAction(action_type="explain"))
print(obs.query_plan)             # EXPLAIN ANALYZE output

obs = env.step(DbaTunerAction(action_type="get_stats", table="orders"))
print(obs.query_plan)             # column stats with est_index_size

obs = env.step(DbaTunerAction(
    action_type="add_index", table="orders", column="user_id"
))
print(f"Reward: {obs.reward:.4f}   Cost-reduction: {obs.metadata['cost_reduction_ratio']:.2%}")

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

## 7-Level Curriculum

| Level | Name | Goal |
|-------|------|------|
| 1 | Sequential Scan | Add index on `orders.user_id` |
| 2 | FK Join Index | Index `line_items.order_id` for Hash→Nested-Loop |
| 3 | Subquery Refactor | Replace correlated subquery with window function |
| 4 | Budget Challenge | Optimise 5 queries within 50 MB index limit |
| 5 | N+1 Fix | Collapse per-row SELECTs into a single batch JOIN |
| 6 | Range Scan | Index `orders.order_date` for BETWEEN queries |
| 7 | Materialised View | CREATE TABLE AS → SELECT from it (5-table join) |

---

## Action Space

```json
{"action_type": "explain"}
{"action_type": "get_stats",  "table": "orders"}
{"action_type": "add_index",  "table": "orders",  "column": "user_id"}
{"action_type": "drop_index", "index_name": "idx_orders_user_id"}
{"action_type": "rewrite",    "new_sql": "SELECT ..."}
```

---

## Reward Function

```
Reward = CostReductionRatio                     # (baseline_latency - current_latency) / baseline
       + 0.1  (one-time)                        # first explain or get_stats call
       - 0.02 × index_count                     # over-indexing penalty
       - 0.005 × storage_used_mb                # storage waste penalty
       - 0.01 × step_count                      # efficiency penalty

Terminal score (done=True):
  is_correct=True  → max(0.0, min(1.0, best_step_reward))
  is_correct=False → 0.0  (wrong results = zero)
```

Step rewards during an episode may be **negative** (RL signal).
The final episode score is always in **[0.0, 1.0]**.

---

## Running Locally

```bash
# Install dependencies
uv sync

# Run the HTTP server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run inference against all 7 tasks
export HF_TOKEN="hf_..."
python inference.py

# Direct environment smoke-test (no server needed)
python server/dba_tuner_env_environment.py
```

---

## Inference Log Format

```
[START] task=simple_index env=dba_tuner_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=explain() reward=0.0900 done=false error=null
[STEP]  step=2 action=get_stats(orders) reward=0.0389 done=false error=null
[STEP]  step=3 action=add_index(orders,user_id) reward=0.2901 done=false error=null
[END]   success=true steps=3 score=0.2901 avg_reward=0.1396 rewards=0.0900,0.0389,0.2901
```

---

## Project Structure

```
dba-tuner-env/
├── models.py                          # Action + Observation Pydantic models
├── inference.py                       # LLM inference script (all 7 tasks)
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
