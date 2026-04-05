"""
DBA Tuner Env - Inference Script
===================================
Demonstrates a full "Thinking Trajectory" across all 7 task levels:
    explain → get_stats → add_index / rewrite → verify reward

MANDATORY environment variables:
    HF_TOKEN        Your Hugging Face API key (or any OpenAI-compatible key).
    API_BASE_URL    LLM endpoint  (default: HuggingFace router).
    MODEL_NAME      Model identifier (default:meta-llama/Llama-3.1-8B-Instruct).

STDOUT FORMAT  (one line type per event, in order):
    [START] task=<name> env=dba_tuner_env model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> avg_reward=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
import traceback

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from server.dba_tuner_env_environment import DbaTunerEnvironment
from models import DbaTunerAction

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

BENCHMARK    = "dba_tuner_env"
MAX_STEPS    = 25  # must cover the hardest scenario (Level 4: max_steps=25)

# All 7 tasks (easy → expert)
TASKS = [
    {"name": "simple_index",      "level": 1, "difficulty": "easy"},
    {"name": "join_optimization", "level": 2, "difficulty": "medium"},
    {"name": "subquery_rewrite",  "level": 3, "difficulty": "medium"},
    {"name": "budget_challenge",  "level": 4, "difficulty": "hard"},
    {"name": "n_plus_one_fix",    "level": 5, "difficulty": "hard"},
    {"name": "range_scan",        "level": 6, "difficulty": "hard"},
    {"name": "materialized_view", "level": 7, "difficulty": "hard"},
]

# ── Full database schema (injected into every LLM prompt) ─────────────────────
DATABASE_SCHEMA = textwrap.dedent("""\
    DATABASE SCHEMA (DuckDB, generated fresh each episode):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ users      (100 000 rows)                                               │
    │   user_id INTEGER PK, username VARCHAR, email VARCHAR,                  │
    │   signup_date DATE, country VARCHAR                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ products   ( 10 000 rows)                                               │
    │   product_id INTEGER PK, name VARCHAR, category VARCHAR,               │
    │   price DOUBLE, created_at DATE                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ orders     (100 000 rows)  ★ Pareto-skewed on user_id (α=1.1)          │
    │   order_id INTEGER PK, user_id INTEGER, order_date DATE,               │
    │   status VARCHAR, total_amount DOUBLE                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ line_items (300 000 rows)  ★ Pareto-skewed on product_id (α=1.1)       │
    │   line_item_id INTEGER PK, order_id INTEGER, product_id INTEGER,       │
    │   quantity INTEGER, unit_price DOUBLE                                   │
    └─────────────────────────────────────────────────────────────────────────┘
    ★ Pareto skew: ~20% of user/product IDs account for ~80% of rows.
      Use get_stats to identify hot columns before indexing.
      Index budget: 50 MB total. Each index on orders/line_items ≈ 0.76 MB.
""")

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(f"""\
You are an expert Database Administrator (DBA) optimising a DuckDB database for performance.
Your goal is to reduce query latency by adding indexes and/or rewriting queries.

{DATABASE_SCHEMA}

AVAILABLE ACTIONS (respond with ONLY a valid JSON object — no markdown, no explanation):

1. Examine the query execution plan (earns a ONE-TIME +0.1 reasoning bonus on first call):
   {{"action_type": "explain"}}

2. Get full column stats including estimated index sizes (earns the same one-time +0.1 bonus):
   {{"action_type": "get_stats", "table": "TABLE_NAME"}}
   Valid tables: users, products, orders, line_items

3. Add an index on a column (costs storage budget ≈ 0.76 MB each):
   {{"action_type": "add_index", "table": "TABLE_NAME", "column": "COLUMN_NAME"}}

4. Drop an existing index (reclaims storage):
   {{"action_type": "drop_index", "index_name": "INDEX_NAME"}}

5. Rewrite the SQL query (must produce IDENTICAL results to the original):
   {{"action_type": "rewrite", "new_sql": "SELECT ..."}}

   For Level 7 (Materialised View): first submit a CREATE TABLE AS SELECT to build
   the view, then submit a SELECT FROM that table — only the SELECT is graded.

REWARD SIGNAL:
  +CostReductionRatio   Proportional to latency reduction (0.0 → 1.0).
  +0.1                  One-time bonus for your FIRST explain or get_stats call.
  −0.02 × indexes       Penalty per index created (over-indexing is penalised).
  −0.005 × MB_used      Penalty per MB of index storage.
  −0.01 × step          Efficiency penalty per step — solve it quickly!
  Terminal score        Clamped to [0.0, 1.0] when done=true.

OPTIMAL STRATEGY:
  1. ALWAYS start with explain (earns reasoning bonus + reveals scan type).
  2. Use get_stats on the WHERE / JOIN columns to confirm high selectivity.
  3. Add at most 2-3 targeted indexes — more is penalised.
  4. Only rewrite SQL when the scenario explicitly requires it (subqueries, N+1).
  5. For N+1 (Level 5): rewrite using WHERE user_id IN (...) batch join.
  6. For Budget Challenge (Level 4): index orders.user_id and line_items.order_id first.
  7. Stop early — the environment will end successfully when cost drops > 95%.

IMPORTANT: First, think step-by-step in a <thought>...</thought> block about your observations and strategy. Then, output exactly ONE JSON action object.
""")


# ── Helper functions ───────────────────────────────────────────────────────────

def build_user_message(obs, step_num: int, prev_actions: list) -> str:
    """Construct the per-step observation message for the LLM."""
    history = ""
    if prev_actions:
        history = "\nPrevious actions this episode:\n" + "\n".join(
            f"  Step {i + 1}: {a}" for i, a in enumerate(prev_actions)
        )

    cost_ratio = obs.metadata.get("cost_reduction_ratio", 0.0) if obs.metadata else 0.0
    bonus_paid = obs.metadata.get("reasoning_bonus_paid", False) if obs.metadata else False

    return textwrap.dedent(f"""\
Scenario (Level {obs.scenario_level}):
{obs.scenario_description}

Current SQL:
{obs.current_sql}

Performance Metrics:
  Latency        : {obs.latency_ms:.2f} ms
  Cost-Reduction : {cost_ratio:.2%}
  Storage used   : {obs.storage_used_mb:.2f} MB  /  remaining: {obs.storage_remaining_mb:.2f} MB
  Active indexes : {obs.active_indexes if obs.active_indexes else 'none'}
  Reasoning bonus: {'already earned — no more bonus for explain/get_stats' if bonus_paid else 'NOT YET EARNED — call explain or get_stats to earn +0.1'}

Query Plan / Info:
{obs.query_plan if obs.query_plan else '(not yet examined — use explain action)'}

Error: {obs.error_message if obs.error_message else 'none'}
{history}

Step {step_num} of {MAX_STEPS}. Respond with your next action as JSON:\
""")


def parse_llm_action(response_text: str) -> DbaTunerAction:
    """Parse LLM text response into a DbaTunerAction, stripping out thoughts."""
    text = response_text.strip()

    # Try JSON from markdown code block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # Fallback: extract from first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            text = text[start : end + 1]

    data = json.loads(text)
    return DbaTunerAction(**data)


def format_action_str(action: DbaTunerAction) -> str:
    """Human-readable summary for [STEP] log lines."""
    if action.action_type == "explain":
        return "explain()"
    if action.action_type == "get_stats":
        return f"get_stats({action.table})"
    if action.action_type == "add_index":
        return f"add_index({action.table},{action.column})"
    if action.action_type == "drop_index":
        return f"drop_index({action.index_name})"
    if action.action_type == "rewrite":
        preview = (action.new_sql or "")[:60].replace("\n", " ")
        return f"rewrite({preview})"
    return f"{action.action_type}()"


# ── Task runner ────────────────────────────────────────────────────────────────

def run_task(
    env: DbaTunerEnvironment,
    task_name: str,
    level: int,
    llm_client: OpenAI,
    model_name: str,
) -> float:
    """Run one task episode, emit [START]/[STEP]/[END] log lines, return score."""
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)

    rewards: list[float] = []
    steps = 0
    success = False
    prev_actions: list[str] = []

    try:
        obs = env.reset(level=level)

        for step_num in range(1, MAX_STEPS + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_message(obs, step_num, prev_actions)},
            ]

            # Call LLM (with retry logic for 429/402 errors)
            max_retries = 5
            llm_text = ""
            for attempt in range(max_retries):
                try:
                    resp = llm_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=800,  # increased to allow thoughts
                    )
                    llm_text = resp.choices[0].message.content or ""
                    break  # Success
                except Exception as e:
                    print(f"  [API ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}", file=sys.stderr)
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        llm_text = '{"action_type": "explain"}'  # safe fallback after all retries fail

            # Parse action
            try:
                action = parse_llm_action(llm_text)
            except Exception as e:
                print(f"  [PARSING ERROR] {e}\n  LLM Text: {llm_text[:150]}...", file=sys.stderr)
                action = DbaTunerAction(action_type="explain")

            action_str = format_action_str(action)
            prev_actions.append(action_str)

            # Execute in environment
            try:
                obs = env.step(action)
                reward = obs.reward
                done   = obs.done
                error  = obs.error_message if obs.error_message else "null"
            except Exception as e:
                reward = 0.0
                done   = True
                error  = str(e)

            rewards.append(reward)
            steps = step_num

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.4f} done={str(done).lower()} error={error}",
                flush=True,
            )

            if done:
                break

        # Episode score = max reward seen (terminal reward is already [0,1])
        score = max(rewards) if rewards else 0.0
        avg_r = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.0

    except Exception:
        traceback.print_exc(file=sys.stderr)
        score = 0.0
        avg_r = 0.0
        if not rewards:
            rewards = [0.0]
            steps = 1

    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} avg_reward={avg_r:.4f} rewards={rewards_str}",
        flush=True,
    )
    print()
    return score


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Run inference across all 7 tasks and print the overall average score."""
    if not API_KEY:
        print("ERROR: No API key found.", file=sys.stderr)
        print("  Set HF_TOKEN (or API_KEY) in your environment:", file=sys.stderr)
        print('    PowerShell : $env:HF_TOKEN = "hf_..."', file=sys.stderr)
        print('    Bash/Zsh   : export HF_TOKEN="hf_..."', file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = DbaTunerEnvironment()

    total_score = 0.0
    results: list[dict] = []

    for task in TASKS:
        score = run_task(
            env=env,
            task_name=task["name"],
            level=task["level"],
            llm_client=llm_client,
            model_name=MODEL_NAME,
        )
        total_score += score
        results.append({"task": task["name"], "level": task["level"], "score": score})

    env.close()

    avg_score = total_score / len(TASKS)

    # Pretty summary table
    print("=" * 55)
    print(f"{'Task':<22} {'Level':>5}  {'Score':>7}")
    print("-" * 55)
    for r in results:
        print(f"  {r['task']:<20} {r['level']:>5}  {r['score']:>7.4f}")
    print("-" * 55)
    print(f"  {'AVERAGE':<20} {'—':>5}  {avg_score:>7.4f}")
    print("=" * 55)
    print(f"\n[OVERALL] avg_score={avg_score:.4f} across {len(TASKS)} tasks")


if __name__ == "__main__":
    main()
