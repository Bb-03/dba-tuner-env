"""
DBA Tuner Env - Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import textwrap
import traceback

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from server.dba_tuner_env_environment import DbaTunerEnvironment
from models import DbaTunerAction

# ── Configuration ──────────────────────────────────────────────────────
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "dba_tuner_env"
MAX_STEPS = 15

# All 7 tasks (easy -> medium -> hard)
TASKS = [
    {"name": "simple_index",       "level": 1, "difficulty": "easy"},
    {"name": "join_optimization",  "level": 2, "difficulty": "easy"},
    {"name": "subquery_rewrite",   "level": 3, "difficulty": "medium"},
    {"name": "budget_challenge",   "level": 4, "difficulty": "medium"},
    {"name": "n_plus_one_fix",     "level": 5, "difficulty": "medium"},
    {"name": "range_scan",         "level": 6, "difficulty": "hard"},
    {"name": "materialized_view",  "level": 7, "difficulty": "hard"},
]


# ── System Prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert Database Administrator (DBA) tuning a DuckDB database for performance.
You will be given a scenario description, the current SQL query, and performance metrics.
Your goal is to improve query performance through indexing and query rewriting.

Available actions (respond with ONLY a valid JSON object, no markdown, no explanation):

1. Examine the query execution plan:
   {"action_type": "explain"}

2. Get statistics about a table:
   {"action_type": "get_stats", "table": "TABLE_NAME"}
   Valid tables: users, products, orders, line_items

3. Add an index on a column:
   {"action_type": "add_index", "table": "TABLE_NAME", "column": "COLUMN_NAME"}

4. Drop an existing index:
   {"action_type": "drop_index", "index_name": "INDEX_NAME"}

5. Rewrite the SQL query:
   {"action_type": "rewrite", "new_sql": "SELECT ..."}

Strategy tips:
- ALWAYS start with "explain" to see the query plan, then "get_stats" on relevant tables
- Look for sequential scans that could benefit from indexes
- For join queries, index the join columns
- For WHERE clauses, index the filtered columns
- Only rewrite SQL if the scenario specifically requires it (e.g. correlated subqueries, N+1 patterns)
- The reward is based on latency reduction minus a small penalty for storage used

IMPORTANT: Respond with ONLY the JSON action. No explanation, no markdown blocks, no extra text.
""")


def build_user_message(obs, step_num, prev_actions):
    """Build the observation message for the LLM."""
    history = ""
    if prev_actions:
        history = "\n\nPrevious actions this episode:\n" + "\n".join(
            f"  Step {i+1}: {a}" for i, a in enumerate(prev_actions)
        )

    return textwrap.dedent(f"""\
Scenario (Level {obs.scenario_level}): {obs.scenario_description}

Current SQL: {obs.current_sql}

Current metrics:
- Latency: {obs.latency_ms:.2f} ms
- Storage used: {obs.storage_used_mb:.2f} MB / remaining: {obs.storage_remaining_mb:.2f} MB
- Active indexes: {obs.active_indexes}
- Index count: {obs.index_count}

Query plan / info:
{obs.query_plan if obs.query_plan else '(not yet examined - use explain action)'}

Error (if any): {obs.error_message if obs.error_message else 'none'}
{history}

Step {step_num} of {MAX_STEPS}. Respond with your next action as JSON:""")


def parse_llm_action(response_text):
    """Parse LLM response into a DbaTunerAction."""
    text = response_text.strip()

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            text = json_match.group(0)

    data = json.loads(text)
    return DbaTunerAction(**data)


def format_action_str(action):
    """Format action for [STEP] log line."""
    if action.action_type == "explain":
        return "explain()"
    elif action.action_type == "get_stats":
        return f"get_stats({action.table})"
    elif action.action_type == "add_index":
        return f"add_index({action.table},{action.column})"
    elif action.action_type == "drop_index":
        return f"drop_index({action.index_name})"
    elif action.action_type == "rewrite":
        sql_preview = (action.new_sql or "")[:50].replace("\n", " ")
        return f"rewrite({sql_preview})"
    return f"{action.action_type}()"


def run_task(env, task_name, level, llm_client, model_name):
    """Run a single task and emit [START]/[STEP]/[END] logs."""

    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}")

    rewards = []
    steps = 0
    success = False
    prev_actions = []

    try:
        # Reset environment with specific level
        obs = env.reset(level=level)

        for step_num in range(1, MAX_STEPS + 1):
            # Build messages for LLM
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(obs, step_num, prev_actions)},
            ]

            # Call LLM
            try:
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                )
                llm_text = response.choices[0].message.content or ""
            except Exception as e:
                # LLM call failed - use a fallback action
                llm_text = '{"action_type": "explain"}'

            # Parse action from LLM response
            try:
                action = parse_llm_action(llm_text)
            except Exception:
                # If parsing fails, default to explain
                action = DbaTunerAction(action_type="explain")

            action_str = format_action_str(action)
            prev_actions.append(action_str)

            # Execute action in environment
            try:
                obs = env.step(action)
                reward = obs.reward
                done = obs.done
                error = obs.error_message if obs.error_message else "null"
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps = step_num

            # Emit [STEP] log
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error}"
            )

            if done:
                break

        # Score = best reward achieved during the episode
        score = max(rewards) if rewards else 0.0
        success = score > 0.0

    except Exception as e:
        score = 0.0
        success = False
        if not rewards:
            rewards = [0.0]
            steps = 1

    # Emit [END] log
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}"
    )

    return score


def main():
    """Run inference across all 7 tasks."""

    # Initialize OpenAI client (uses HF router)
    if not API_KEY:
        print("ERROR: No API key found. Set your HuggingFace token:")
        print('  PowerShell:  $env:HF_TOKEN = "hf_your_token_here"')
        print('  CMD:         set HF_TOKEN=hf_your_token_here')
        print('  Linux/Mac:   export HF_TOKEN=hf_your_token_here')
        sys.exit(1)

    llm_client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    # Create environment directly (in-process, no server needed)
    env = DbaTunerEnvironment()

    total_score = 0.0

    for task in TASKS:
        score = run_task(
            env=env,
            task_name=task["name"],
            level=task["level"],
            llm_client=llm_client,
            model_name=MODEL_NAME,
        )
        total_score += score

    env.close()

    avg_score = total_score / len(TASKS)
    print(f"\n=== OVERALL: avg_score={avg_score:.2f} across {len(TASKS)} tasks ===")


if __name__ == "__main__":
    main()
