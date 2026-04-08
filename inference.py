"""
DBA Tuner Env - Inference Script
===================================
Demonstrates a full "Thinking Trajectory" across all 3 tasks:
    explain → get_stats → add_index → done

MANDATORY environment variables:
    HF_TOKEN        Your Hugging Face API key (or any OpenAI-compatible key).
    API_BASE_URL    LLM endpoint  (default: HuggingFace router).
    MODEL_NAME      Model identifier (default:Qwen/Qwen2.5-72B-Instruct).

STDOUT FORMAT  (one line type per event, in order):
    [START] task=<name> env=dba_tuner_env model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

No extra lines on stdout.  Debug/error output goes to stderr only.
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

from server.dba_tuner_env_environment import DbaTunerEnvironment  # type: ignore
from models import DbaTunerAction  # type: ignore

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK    = "dba_tuner_env"
MAX_STEPS    = 15  # covers all 3 tasks

# All 3 tasks (easy → hard)
TASKS = [
    {"name": "simple_index",      "level": 1, "difficulty": "easy"},
    {"name": "join_optimization", "level": 2, "difficulty": "medium"},
    {"name": "range_scan",        "level": 3, "difficulty": "hard"},
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
Your goal is to reduce query cost by adding targeted indexes.

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

5. Signal that the task is solved (terminates the episode):
   {{"action_type": "done"}}

REWARD SIGNAL:
  +CostReductionRatio   Proportional to EXPLAIN plan cost reduction (0.0 -> 1.0).
  +0.1                  One-time bonus for your FIRST explain or get_stats call.
  -0.02 x indexes       Penalty per index created (over-indexing is penalised).
  -0.005 x MB_used      Penalty per MB of index storage.
  -0.005 x step         Efficiency penalty per step -- solve it quickly!
  Terminal score        Clamped to [0.0, 1.0] when done=true.
  IMPORTANT: You must achieve real cost reduction (>30%) for a non-zero terminal score.
             Explain/get_stats bonuses alone will NOT earn a passing score.

OPTIMAL STRATEGY:
  1. ALWAYS start with explain (earns reasoning bonus + reveals scan type).
  2. Use get_stats on the WHERE / JOIN columns to confirm selectivity.
  3. Add at most 1-2 targeted indexes — more is penalised.
  4. Call done when cost reduction is high to lock in the terminal reward.

IMPORTANT: Respond with exactly ONE valid JSON action object in your reply. Do NOT output any other text or markdown code blocks.
DANGER: Do NOT repeat any action (like explain or get_stats) twice in a row with identical parameters. Each step costs efficiency bonus, and IDENTICAL CONSECUTIVE actions will kill the episode with a -0.1 penalty.
""")


# ── Helper functions ───────────────────────────────────────────────────────────

def build_user_message(obs, step_num: int, prev_actions: list) -> str:
    """Construct the per-step observation message for the LLM."""
    history = ""
    if prev_actions:
        recent = prev_actions[-3:]
        offset = len(prev_actions) - len(recent)
        history = f"\nRecent actions (last {len(recent)} of {len(prev_actions)} total):\n" + "\n".join(
            f"  Step {offset + i + 1}: {a}" for i, a in enumerate(recent)
        )

    cost_ratio = obs.metadata.get("cost_reduction_ratio", 0.0) if obs.metadata else 0.0
    bonus_paid = obs.metadata.get("reasoning_bonus_paid", False) if obs.metadata else False

    return textwrap.dedent(f"""\
Scenario (Task {obs.scenario_level}):
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

Step {step_num} of {MAX_STEPS}. 
HINT: If you have already achieved a high Cost-Reduction (>50%) and cannot find further improvements, call {{"action_type": "done"}} to finalize your score and earn the maximum terminal reward.

Respond with your next action as JSON:\
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

            # Call LLM (with retry logic for transient errors)
            max_retries = 5
            llm_text = ""
            for attempt in range(max_retries):
                try:
                    resp = llm_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=800,
                    )
                    llm_text = resp.choices[0].message.content or ""
                    break  # Success
                except Exception as e:
                    # Fail-fast on quota/auth errors (401, 402)
                    err_str = str(e).lower()
                    if "402" in err_str or "payment" in err_str:
                        print(f"\n[FATAL] API Quota Exhausted (402): {e}", file=sys.stderr)
                        sys.exit(1)
                    if "401" in err_str or "unauthorized" in err_str:
                        print(f"\n[FATAL] Invalid API Key (401): {e}", file=sys.stderr)
                        sys.exit(1)

                    print(f"  [API ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}", file=sys.stderr)
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        llm_text = '{"action_type": "done"}'

            # Parse action
            try:
                action = parse_llm_action(llm_text)
            except Exception as e:
                print(f"  [PARSING ERROR] {e}\n  LLM Text: {llm_text[:150]}...", file=sys.stderr)
                action = DbaTunerAction(action_type="done")

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
                f"reward={reward:.2f} done={str(done).lower()} error={error}",
                flush=True,
            )

            if done:
                break

        # Episode score = final terminal reward (the last reward when done=True)
        score = rewards[-1] if rewards else 0.0
        success = score > 0.0

    except Exception:
        traceback.print_exc(file=sys.stderr)
        score = 0.0
        if not rewards:
            rewards = [0.0]
            steps = 1

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return score


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Run inference across all 3 tasks."""
    if not API_KEY:
        print("ERROR: No API key found.", file=sys.stderr)
        print("  Set HF_TOKEN (or API_KEY) in your environment:", file=sys.stderr)
        print('    PowerShell : $env:HF_TOKEN = "hf_..."', file=sys.stderr)
        print('    Bash/Zsh   : export HF_TOKEN="hf_..."', file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = DbaTunerEnvironment()

    for task in TASKS:
        run_task(
            env=env,
            task_name=task["name"],
            level=task["level"],
            llm_client=llm_client,
            model_name=MODEL_NAME,
        )

    env.close()


if __name__ == "__main__":
    main()
