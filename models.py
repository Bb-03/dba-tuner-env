"""
Data models for the DBA Tuner Env Environment.

Defines a discriminated-union action space for database performance tuning,
a rich observation model carrying query plans, metrics, and correctness info,
and a typed Reward model for structured reward metadata.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reward – typed breakdown of reward components
# ---------------------------------------------------------------------------


class DbaTunerReward(BaseModel):
    """Typed reward breakdown for a single step.

    This model provides structured access to individual reward components.
    The scalar ``step_reward`` is also available on the Observation's
    ``reward`` field for backward compatibility.
    """

    step_reward: float = Field(
        default=0.0,
        description="Raw step reward (may be negative — intentional RL signal)",
    )
    terminal_reward: Optional[float] = Field(
        default=None,
        description=(
            "Final clamped reward [0.0, 1.0] when done=True. "
            "None for non-terminal steps."
        ),
    )
    task_solved: bool = Field(
        default=False,
        description=(
            "Whether the task success condition was met "
            "(meaningful complexity reduction achieved). Explain/get_stats bonuses "
            "alone cannot set this to True."
        ),
    )
    cost_reduction_ratio: float = Field(
        default=0.0,
        description="(baseline_plan_complexity - current_plan_complexity) / baseline_plan_complexity",
    )
    step_penalty: float = Field(
        default=0.0,
        description="-0.01 * step_count",
    )
    reasoning_bonus: float = Field(
        default=0.0,
        description="+0.1 one-time bonus (if earned this step)",
    )


# ---------------------------------------------------------------------------
# Action – single model with action_type discriminator
# ---------------------------------------------------------------------------


class DbaTunerAction(Action):
    """Action for the DBA Tuner environment.

    The *action_type* field selects which operation to perform:

    * ``explain``   – Run EXPLAIN on the current query.
                      Earns a one-time +0.1 reasoning bonus.
    * ``get_stats`` – Retrieve cardinality / distribution stats for *table*.
                      Earns a one-time +0.1 reasoning bonus.
    * ``rewrite``   – Submit a rewritten SQL query to replace the current one.
    * ``done``      – Terminate the episode explicitly.
    """

    action_type: Literal[
        "explain", "get_stats", "rewrite", "done"
    ] = Field(..., description="Which action to perform")

    # --- get_stats ---
    table: Optional[str] = Field(
        default=None,
        description="Target table name. Valid: users, products, orders, line_items",
    )

    # --- rewrite ---
    sql: Optional[str] = Field(
        default=None,
        description="The rewritten SQL query to test against the environment.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class DbaTunerObservation(Observation):
    """Observation returned by the DBA Tuner environment after every step.

    Database Schema (generated fresh each episode with Pareto α=1.1 skew):

        users      (100 000 rows)  user_id, username, email, signup_date, country
        products   (  10 000 rows) product_id, name, category, price, created_at
        orders     (100 000 rows)  order_id, user_id, order_date, status, total_amount
        line_items (300 000 rows)  line_item_id, order_id, product_id, quantity, unit_price

    Reward structure:
        Reward = CostReductionRatio - 0.01 * step_count
        Cost is measured via 0-compute tree node counting.
        Step rewards are raw (may be negative — this is intentional RL signal).
        Terminal reward (done=True, task_solved=True) is clamped to [0.0, 1.0].
        task_solved requires cost_reduction_ratio > 0.15.
    """

    # ── Query plan / stats output ────────────────────────────────────────
    query_plan: str = Field(
        default="",
        description="EXPLAIN output or get_stats text",
    )

    # ── Performance metrics ──────────────────────────────────────────────
    latency_ms: float = Field(
        default=0.0,
        description="Wall-clock execution time of the current query (milliseconds)",
    )
    total_cost: float = Field(
        default=0.0,
        description="Deterministic plan cost (execution tree node count)",
    )

    # ── Correctness ──────────────────────────────────────────────────────
    is_correct: bool = Field(
        default=True,
        description=(
            "True if the current SQL produces a result set matching the gold standard. "
            "False triggers episode termination with reward=0.0."
        ),
    )

    # ── Active query ─────────────────────────────────────────────────────
    current_sql: str = Field(
        default="",
        description="The SQL query currently being tracked / optimised",
    )

    # ── Scenario metadata ────────────────────────────────────────────────
    scenario_level: int = Field(
        default=1,
        description="Task level 1–3",
    )
    scenario_description: str = Field(
        default="",
        description=(
            "Human-readable task description including schema hints, "
            "and optimisation goal."
        ),
    )

    # ── Errors ───────────────────────────────────────────────────────────
    error_message: str = Field(
        default="",
        description="Error details if the action failed (empty string = no error)",
    )

    # ── Episode control ──────────────────────────────────────────────────
    done: bool = Field(
        default=False,
        description=(
            "True when the episode has ended (max steps, correctness failure)."
        ),
    )
    reward: float = Field(
        default=0.0,
        description=(
            "Step reward. Intra-episode values may be negative (efficiency/penalty signal). "
            "Terminal reward (done=True, task_solved=True) is clamped to [0.0, 1.0]. "
        ),
    )

    # ── Extra metadata ───────────────────────────────────────────────────
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra step metadata: step_count (int), episode_id (str), "
            "cost_reduction_ratio (float), reasoning_bonus_paid (bool), "
            "task_solved (bool)."
        ),
    )
