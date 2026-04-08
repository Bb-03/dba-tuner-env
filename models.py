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
            "(cost_reduction_ratio > 0.3). Explain/get_stats bonuses alone "
            "cannot set this to True."
        ),
    )
    cost_reduction_ratio: float = Field(
        default=0.0,
        description="(baseline_plan_cost - current_plan_cost) / baseline_plan_cost",
    )
    index_penalty: float = Field(
        default=0.0,
        description="-0.02 × index_count",
    )
    storage_penalty: float = Field(
        default=0.0,
        description="-0.005 × storage_used_mb",
    )
    step_penalty: float = Field(
        default=0.0,
        description="-0.005 × step_count",
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

    * ``explain``   – Run EXPLAIN ANALYZE on the current query.
                      Earns a one-time +0.1 reasoning bonus (first call per episode).
    * ``add_index`` – CREATE INDEX on *table*.*column* (costs storage budget).
    * ``drop_index``– DROP INDEX by *index_name* (frees budget).
    * ``rewrite``   – Replace the current SQL with *new_sql*.
                      Not available on Level 4 (index-budget-only).
    * ``create_materialized_view`` – For Level 7: create a materialized table
                      using *view_name* and *sql* fields. The *sql* may be a
                      bare SELECT (auto-wrapped) or a CREATE TABLE AS SELECT.
                      Then submit a rewrite() with a SELECT from that table.
    * ``get_stats`` – Retrieve cardinality / distribution stats for *table*.
                      Earns a one-time +0.1 reasoning bonus (first call per episode).
    * ``done``      – Terminate the episode explicitly when you believe the
                      task is fully solved.

    Reward Signal:
        +CostReductionRatio      Proportional to (baseline_cost − current_cost) / baseline_cost.
                                 Based on EXPLAIN plan estimated rows (deterministic).
        +0.1                     One-time bonus for the first explain or get_stats call.
        −0.02 × index_count      Penalty per index created (over-indexing discouraged).
        −0.005 × storage_mb      Penalty per MB of index storage consumed.
        −0.005 × step_count      Efficiency penalty for each step taken.

        Final episode score (done=True, task_solved=True): clamped to [0.0, 1.0].
        Final episode score (done=True, task_solved=False): forced to 0.0.
        Intra-episode step rewards may be negative (RL signal).

    Task Solved Condition:
        cost_reduction_ratio > 0.3 (meaningful optimisation achieved).
        Explain/get_stats bonuses alone cannot satisfy this condition.
    """

    action_type: Literal[
        "explain", "add_index", "drop_index", "rewrite", "get_stats", "create_materialized_view", "done"
    ] = Field(..., description="Which action to perform")

    # --- add_index / get_stats ---
    table: Optional[str] = Field(
        default=None,
        description="Target table name. Valid: users, products, orders, line_items",
    )
    column: Optional[str] = Field(
        default=None, description="Target column name (add_index only)"
    )

    # --- drop_index ---
    index_name: Optional[str] = Field(
        default=None, description="Exact index name to drop (drop_index only)"
    )

    # --- rewrite ---
    new_sql: Optional[str] = Field(
        default=None,
        description=(
            "Full replacement SQL query (rewrite only). "
            "May be a SELECT, CTE, window function, or CREATE TABLE AS SELECT. "
            "Destructive DDL (DELETE, DROP TABLE, TRUNCATE, ALTER TABLE, UPDATE, INSERT) is blocked. "
            "Not available on Level 4 (index-budget-only task)."
        ),
    )

    # --- create_materialized_view ---
    view_name: Optional[str] = Field(
        default=None,
        description="Name of the materialized table to create (e.g. 'top_users_revenue')",
    )
    sql: Optional[str] = Field(
        default=None,
        description=(
            "SQL for the materialized view. Can be a bare SELECT (auto-wrapped as "
            "CREATE TABLE <view_name> AS <sql>) or a full CREATE TABLE AS SELECT."
        ),
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class DbaTunerObservation(Observation):
    """Observation returned by the DBA Tuner environment after every step.

    Database Schema (generated fresh each episode with Pareto α=1.1 skew):

        users      (100 000 rows)  user_id, username, email, signup_date, country
        products   (  10 000 rows) product_id, name, category, price, created_at
        orders     (100 000 rows)  order_id, user_id★, order_date, status, total_amount
        line_items (300 000 rows)  line_item_id, order_id, product_id★, quantity, unit_price

        ★ = Pareto-skewed column: ~20% of values account for ~80% of rows.
            Use get_stats to identify the hot values before indexing.

    7-Level Curriculum:
        L1 (Easy)   – Sequential scan on high-cardinality non-indexed column.
        L2 (Medium) – Large table point lookup (missing index on line_items.product_id).
        L3 (Medium) – Redundant UNION ALL scans → single CASE WHEN GROUP BY.
        L4 (Hard)   – Budget Challenge: 5 slow queries, 50 MB index limit (index-only, no rewrite).
        L5 (Hard)   – N+1 query pattern → single batch JOIN.
        L6 (Hard)   – Range scan optimisation on order_date.
        L7 (Hard)   – Materialised view for a 5-table aggregation join.

    Reward structure:
        Reward = CostReductionRatio - 0.02×indexes - 0.005×storage_mb - 0.005×steps
        Cost is measured via EXPLAIN plan estimated rows (deterministic, no timing).
        Step rewards are raw (may be negative — this is intentional RL signal).
        Terminal reward (done=True, task_solved=True) is clamped to [0.0, 1.0].
        Terminal reward (done=True, task_solved=False) is forced to 0.0.
        task_solved requires cost_reduction_ratio > 0.3 (explain/get_stats alone insufficient).
    """

    # ── Query plan / stats output ────────────────────────────────────────
    query_plan: str = Field(
        default="",
        description="EXPLAIN ANALYZE output or get_stats text",
    )

    # ── Performance metrics ──────────────────────────────────────────────
    latency_ms: float = Field(
        default=0.0,
        description="Wall-clock execution time of the current query (milliseconds)",
    )
    total_cost: float = Field(
        default=0.0,
        description="Deterministic plan cost (estimated rows from EXPLAIN)",
    )

    # ── Storage budget ───────────────────────────────────────────────────
    storage_used_mb: float = Field(
        default=0.0,
        description="Total index storage consumed this episode (MB)",
    )
    storage_remaining_mb: float = Field(
        default=50.0,
        description="Remaining index storage budget (MB)",
    )

    # ── Index tracking ───────────────────────────────────────────────────
    index_count: int = Field(
        default=0,
        description="Number of agent-created indexes currently active",
    )
    active_indexes: List[str] = Field(
        default_factory=list,
        description="List of active index names with their estimated sizes (e.g. 'idx_orders_user_id: 0.76MB')",
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
        description="Difficulty level 1–7",
    )
    scenario_description: str = Field(
        default="",
        description=(
            "Human-readable task description including schema hints, "
            "available index budget, and optimisation goal."
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
            "True when the episode has ended (max steps, correctness failure, "
            "or early success via CostReductionRatio > 0.95)."
        ),
    )
    reward: float = Field(
        default=0.0,
        description=(
            "Step reward. Intra-episode values may be negative (efficiency/penalty signal). "
            "Terminal reward (done=True, task_solved=True) is clamped to [0.0, 1.0]. "
            "Terminal reward (done=True, task_solved=False) is always 0.0."
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
