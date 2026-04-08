"""
DBA Tuner Environment Implementation.

High-fidelity OpenEnv environment for database performance tuning using DuckDB.

Key features
------------
* 100k-row e-commerce dataset with Pareto (α=1.1) skewed distributions
  – ~20% of users/products drive ~80% of activity, forcing the agent to use
    GetStats before blindly adding indexes.
* Fully deterministic: reset(seed=N) reproduces identical datasets and baselines.
* 7-level scenario registry (easy → expert).
* Correctness checking via pandas DataFrame comparison (robust to column aliases).
* Storage budget tracking (default 50 MB per episode).
* 5-second SQL timeout to prevent cross-join hangs.
* One-shot reasoning bonus: +0.1 awarded for the FIRST explain or get_stats call.
* Deterministic reward: uses EXPLAIN plan estimated rows, not wall-clock timing.
* Early termination when CostReductionRatio > 0.95 (efficiency signal).
* Duplicate-rewrite detection (whitespace-normalised comparison → done + penalty).
* _task_solved gating: explain/get_stats bonuses alone cannot win an episode.
  The agent must achieve cost_reduction_ratio > 0.3 for a non-zero terminal reward.
* Step rewards are unclamped (may be negative — real RL signal).
* Terminal reward (done=True) is clamped to [0.0, 1.0] for hackathon scoring.
  If _task_solved is False or correctness fails, terminal reward is forced to 0.0.
* Level 4 is index-budget-only; rewrite actions are explicitly rejected.
* create_materialized_view is a first-class action using view_name + sql fields.
"""

from __future__ import annotations

import re
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import duckdb
import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DbaTunerAction, DbaTunerObservation
except (ImportError, SystemError):
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import DbaTunerAction, DbaTunerObservation


# ──────────────────────────────────────────────────────────────────────────────
# Database schema hint (injected into every scenario description)
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA_HINT = (
    "SCHEMA: "
    "users(user_id PK, username, email, signup_date, country) 100k rows | "
    "products(product_id PK, name, category, price, created_at) 10k rows | "
    "orders(order_id PK, user_id★, order_date, status, total_amount) 100k rows | "
    "line_items(line_item_id PK, order_id, product_id★, quantity, unit_price) 300k rows. "
    "★=Pareto-skewed (α=1.1): run get_stats to find hot values. "
    "Index budget: 50 MB total. Each index ≈ 0.76 MB on orders/line_items. "
    "Use explain → get_stats → add_index for a +0.1 reasoning bonus."
)


# ──────────────────────────────────────────────────────────────────────────────
# Scenario registry
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS: List[Dict[str, Any]] = [
    # ── Level 1: Sequential scan → index on high-cardinality column ───────
    {
        "level": 1,
        "description": (
            "Level 1 (Easy) — Sequential Scan Optimisation.\n"
            "A SELECT filters orders by user_id on a non-indexed column, causing a full "
            "table scan on 100k rows. Add an index on orders.user_id to enable an "
            "index seek.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT order_id, user_id, order_date, status, total_amount "
            "FROM orders WHERE user_id = 42"
        ),
        "initial_sql": (
            "SELECT order_id, user_id, order_date, status, total_amount "
            "FROM orders WHERE user_id = 42"
        ),
        "max_steps": 10,
        "storage_budget_mb": 50.0,
    },
    # ── Level 2: Large-table point lookup → index on line_items ────────────
    {
        "level": 2,
        "description": (
            "Level 2 (Medium) — Large Table Point Lookup.\n"
            "A filter on line_items.product_id scans all 300k rows to find orders for "
            "a single product. Without an index, DuckDB does a full sequential scan. "
            "Add an index on line_items.product_id to enable an ART index seek that "
            "fetches only matching rows from 300k.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT product_id, order_id, quantity, unit_price, "
            "quantity * unit_price AS line_total "
            "FROM line_items "
            "WHERE product_id = 42 "
            "ORDER BY line_total DESC"
        ),
        "initial_sql": (
            "SELECT product_id, order_id, quantity, unit_price, "
            "quantity * unit_price AS line_total "
            "FROM line_items "
            "WHERE product_id = 42 "
            "ORDER BY line_total DESC"
        ),
        "max_steps": 15,
        "storage_budget_mb": 50.0,
    },
    # ── Level 3: Multi-scan UNION ALL → single-scan CASE consolidation ────
    {
        "level": 3,
        "description": (
            "Level 3 (Medium) — Redundant Table Scan Consolidation.\n"
            "The initial query scans the orders table THREE separate times (via UNION "
            "ALL) to compute segment counts. DuckDB cannot auto-merge UNION ALL scans. "
            "Rewrite into a single scan using CASE WHEN expressions inside GROUP BY to "
            "compute all three segments in one pass — this should yield ~3x speedup.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT "
            "CASE "
            "WHEN total_amount > 500 THEN 'high' "
            "WHEN total_amount >= 100 THEN 'medium' "
            "ELSE 'low' "
            "END AS segment, "
            "COUNT(*) AS cnt, "
            "ROUND(SUM(total_amount), 2) AS total "
            "FROM orders WHERE status = 'completed' "
            "GROUP BY segment"
        ),
        "initial_sql": (
            "SELECT 'high' AS segment, COUNT(*) AS cnt, ROUND(SUM(total_amount), 2) AS total "
            "FROM orders WHERE status = 'completed' AND total_amount > 500 "
            "UNION ALL "
            "SELECT 'medium', COUNT(*), ROUND(SUM(total_amount), 2) "
            "FROM orders WHERE status = 'completed' AND total_amount BETWEEN 100 AND 500 "
            "UNION ALL "
            "SELECT 'low', COUNT(*), ROUND(SUM(total_amount), 2) "
            "FROM orders WHERE status = 'completed' AND total_amount < 100"
        ),
        "max_steps": 15,
        "storage_budget_mb": 50.0,
    },
    # ── Level 4: Budget Challenge (5 queries, 50 MB) ──────────────────────
    {
        "level": 4,
        "description": (
            "Level 4 (Hard — Long Horizon) — The Budget Challenge.\n"
            "You have 5 slow queries and only 50 MB of index storage. Each index on "
            "orders or line_items consumes ~0.76 MB. Prioritise which indexes give the "
            "best aggregate latency reduction across ALL queries — over-indexing wastes "
            "budget and earns penalty (−0.005 × MB used).\n"
            "Queries target: orders.user_id, orders.user_id + line_items.order_id, "
            "line_items.product_id, orders.user_id (GROUP BY), products.product_id + "
            "line_items.product_id.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": "MULTI_QUERY",
        "initial_sql": "MULTI_QUERY",
        "multi_queries": [
            "SELECT * FROM orders WHERE user_id = 100",
            (
                "SELECT o.order_id, li.product_id, li.quantity "
                "FROM orders o JOIN line_items li ON o.order_id = li.order_id "
                "WHERE o.user_id = 200"
            ),
            "SELECT * FROM line_items WHERE product_id = 50",
            (
                "SELECT user_id, COUNT(*) AS cnt FROM orders "
                "GROUP BY user_id ORDER BY cnt DESC LIMIT 20"
            ),
            (
                "SELECT p.name, SUM(li.quantity) AS total_sold "
                "FROM products p JOIN line_items li ON p.product_id = li.product_id "
                "GROUP BY p.name ORDER BY total_sold DESC LIMIT 10"
            ),
        ],
        "max_steps": 25,
        "storage_budget_mb": 50.0,
    },
    # ── Level 5: N+1 pattern → single batch JOIN ──────────────────────────
    {
        "level": 5,
        "description": (
            "Level 5 (Hard — Algorithmic) — N+1 Query Pattern Fix.\n"
            "The initial pattern issues 10 separate SELECT queries (one per user_id "
            "in [1..10]), causing 10 round-trips. Rewrite into a single batch JOIN that "
            "uses WHERE user_id IN (1,2,...,10) to fetch all results in one pass.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT o.order_id, o.user_id, o.total_amount, "
            "li.line_item_id, li.product_id, li.quantity, li.unit_price "
            "FROM orders o "
            "JOIN line_items li ON o.order_id = li.order_id "
            "WHERE o.user_id IN (1,2,3,4,5,6,7,8,9,10) "
            "ORDER BY o.order_id, li.line_item_id"
        ),
        "initial_sql": "N_PLUS_ONE",
        "n_plus_one_user_ids": list(range(1, 11)),
        "max_steps": 20,
        "storage_budget_mb": 50.0,
    },
    # ── Level 6: Narrow range scan → ART index on date column ─────────────
    {
        "level": 6,
        "description": (
            "Level 6 (Hard — Architectural) — Selective Range Scan Optimisation.\n"
            "A BETWEEN filter on order_date selects only ~2% of rows (1 week out of a "
            "full year), but DuckDB still performs a full sequential scan on 100k rows. "
            "Add an ART index on orders.order_date to enable an efficient range seek "
            "that skips 98% of the table. Optionally, CREATE a clustered copy sorted "
            "by order_date for zone-map pruning.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT order_id, user_id, order_date, total_amount "
            "FROM orders "
            "WHERE order_date BETWEEN '2023-06-15' AND '2023-06-21' "
            "ORDER BY order_date"
        ),
        "initial_sql": (
            "SELECT order_id, user_id, order_date, total_amount "
            "FROM orders "
            "WHERE order_date BETWEEN '2023-06-15' AND '2023-06-21' "
            "ORDER BY order_date"
        ),
        "max_steps": 15,
        "storage_budget_mb": 50.0,
    },
    # ── Level 7: Materialised view for 5-table aggregation ───────────────
    {
        "level": 7,
        "description": (
            "Level 7 (Hard — Trade-offs) — Materialised View Implementation.\n"
            "A complex 5-table aggregation (users ⋈ orders ⋈ line_items ⋈ products) "
            "is executed repeatedly. Step 1: issue a CREATE TABLE <name> AS SELECT … "
            "to materialise the result. Step 2: issue a SELECT … FROM <name> … that "
            "reads from your materialised table — this SELECT is graded against the "
            "gold standard result set.\n"
            "Gold query: top-50 users by revenue with order_count, total_items, "
            "total_revenue, unique_products (status='completed' only).\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT u.user_id, u.username, u.country, "
            "COUNT(DISTINCT o.order_id) AS order_count, "
            "SUM(li.quantity) AS total_items, "
            "SUM(li.quantity * li.unit_price) AS total_revenue, "
            "COUNT(DISTINCT li.product_id) AS unique_products "
            "FROM users u "
            "JOIN orders o ON u.user_id = o.user_id "
            "JOIN line_items li ON o.order_id = li.order_id "
            "JOIN products p ON li.product_id = p.product_id "
            "WHERE o.status = 'completed' "
            "GROUP BY u.user_id, u.username, u.country "
            "ORDER BY total_revenue DESC LIMIT 50"
        ),
        "initial_sql": (
            "SELECT u.user_id, u.username, u.country, "
            "COUNT(DISTINCT o.order_id) AS order_count, "
            "SUM(li.quantity) AS total_items, "
            "SUM(li.quantity * li.unit_price) AS total_revenue, "
            "COUNT(DISTINCT li.product_id) AS unique_products "
            "FROM users u "
            "JOIN orders o ON u.user_id = o.user_id "
            "JOIN line_items li ON o.order_id = li.order_id "
            "JOIN products p ON li.product_id = p.product_id "
            "WHERE o.status = 'completed' "
            "GROUP BY u.user_id, u.username, u.country "
            "ORDER BY total_revenue DESC LIMIT 50"
        ),
        "max_steps": 20,
        "storage_budget_mb": 50.0,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _execute_with_timeout(
    conn: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = 5.0
) -> Tuple[list, float]:
    """Execute *sql* with a wall-clock timeout.

    Returns (rows, elapsed_ms).  Raises TimeoutError on timeout.
    """
    result_container: Dict[str, Any] = {"rows": None, "error": None}

    def _run() -> None:
        try:
            result_container["rows"] = conn.execute(sql).fetchall()
        except Exception as exc:
            result_container["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    start = time.perf_counter()
    t.start()
    t.join(timeout=timeout_s)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if t.is_alive():
        try:
            conn.interrupt()
        except Exception:
            pass
        raise TimeoutError(f"SQL execution exceeded {timeout_s}s timeout")

    if result_container["error"] is not None:
        raise result_container["error"]

    return result_container["rows"] or [], elapsed_ms


def _estimate_index_size_mb(
    conn: duckdb.DuckDBPyConnection, table: str, column: str
) -> float:
    """Estimate the storage cost of an index on *table*.*column*.

    Heuristic: 8 bytes × num_rows / 1 MiB, minimum 0.01 MB.
    """
    try:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return max(0.01, (row_count * 8) / (1024 * 1024))
    except Exception:
        return 0.5  # safe fallback


def _parse_explain_cost(explain_text: str) -> float:
    r"""Extract total execution time from EXPLAIN ANALYZE timing values only.

    Uses regex (\d+\.\d+)s to capture only seconds-denominated values,
    ignoring row counts, cardinality estimates, and other numeric noise.
    NOTE: Used only for display purposes; reward uses _get_plan_cost instead.
    """
    total = 0.0
    for line in explain_text.split("\n"):
        for n in re.findall(r"(\d+\.\d+)s", line):
            total += float(n)
    return total if total > 0 else 0.0001


def _get_plan_cost(conn: duckdb.DuckDBPyConnection, sql: str) -> float:
    """Deterministic cost proxy from EXPLAIN (no execution).

    Parses the query plan and sums estimated row counts across all operator
    nodes.  This gives a stable, reproducible cost signal that does not
    depend on wall-clock timing or CPU load.

    Returns a positive float (minimum 1.0).
    """
    try:
        rows = conn.execute(f"EXPLAIN {sql}").fetchall()
        plan_text = "\n".join(str(r) for r in rows)

        # Extract estimated row counts from plan nodes
        # DuckDB EXPLAIN output contains lines like "~100000 Rows" or "EC: 100000"
        total_rows = 0.0
        for line in plan_text.split("\n"):
            # Match patterns like "~12345" or "EC: 12345" or "12345 Rows"
            for m in re.findall(r"(?:~|EC:\s*)(\d+)", line):
                total_rows += float(m)
            for m in re.findall(r"(\d+)\s*[Rr]ows?", line):
                total_rows += float(m)

        return max(1.0, total_rows)
    except Exception:
        return 1.0


def _shrink_plan(raw_plan: str, top_n: int = 5) -> str:
    """Compress DuckDB EXPLAIN ANALYZE output to the top-N costliest nodes.

    Strips box-drawing characters and tuple artifacts, extracts operator nodes
    with timing data, and returns a compact summary sorted by execution time.
    This keeps LLM payloads small and avoids 413 errors.
    """
    # Clean tuple wrappers from str(row) and box-drawing chars
    text = raw_plan
    text = re.sub(r"\('", "", text)
    text = re.sub(r"',\)", "", text)
    text = re.sub(r"[─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬┃┏┓┗┛]", "", text)

    # Known DuckDB operator keywords
    _OPS = re.compile(
        r"(SEQ_SCAN|INDEX_SCAN|HASH_JOIN|NESTED_LOOP_JOIN|MERGE_JOIN|"
        r"HASH_GROUP_BY|PERFECT_HASH_GROUP_BY|FILTER|PROJECTION|ORDER_BY|"
        r"TOP_N|LIMIT|UNGROUPED_AGGREGATE|STREAMING_WINDOW|"
        r"CROSS_PRODUCT|DELIM_SCAN|COLUMN_DATA_SCAN|PIECEWISE_MERGE_JOIN)",
        re.IGNORECASE,
    )
    _TIME = re.compile(r"(\d+\.?\d*)\s*s(?:ec)?")
    _ROWS = re.compile(r"~?(\d[\d,]*)\s*(?:rows|tuples)?")

    nodes: List[Tuple[float, str]] = []
    lines = text.split("\n")

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        m = _OPS.search(line)
        if not m:
            continue
        op_name = m.group(1).upper()
        time_val = 0.0
        row_info = ""
        # Scan this line + next 3 for timing / row info
        for j in range(i, min(i + 4, len(lines))):
            ctx = lines[j].strip()
            tm = _TIME.search(ctx)
            if tm:
                time_val = max(time_val, float(tm.group(1)))
            rm = _ROWS.search(ctx)
            if rm and not row_info:
                row_info = rm.group(0)
        summary = op_name
        if row_info:
            summary += f" ({row_info})"
        summary += f" [{time_val:.4f}s]"
        nodes.append((time_val, summary))

    if not nodes:
        # Fallback: return cleaned text truncated to 15 lines
        clean_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 2]
        return "\n".join(clean_lines[:15])

    # Sort by time descending, take top N
    nodes.sort(key=lambda x: x[0], reverse=True)
    top = nodes[:top_n]

    result_lines = [f"Top-{len(top)} costliest operators:"]
    for _, summary in top:
        result_lines.append(f"  -> {summary}")
    total_time = sum(t for t, _ in nodes)
    result_lines.append(f"  Total plan time: {total_time:.4f}s across {len(nodes)} operators")
    return "\n".join(result_lines)


# NOTE: _get_explain_cost has been removed. Reward computation now uses
# _get_plan_cost (deterministic EXPLAIN-based, no timing) instead.


def _extract_create_table_name(sql: str) -> Optional[str]:
    """Return the table name from a CREATE TABLE [IF NOT EXISTS] <name> AS … statement."""
    m = re.match(
        r"\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
        sql,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────


class DbaTunerEnvironment(Environment):
    """DBA Tuner environment — OpenEnv compliant.

    Uses an in-memory DuckDB database with Pareto-skewed (α=1.1) e-commerce
    data to present 7 levels of database performance tuning challenges.

    Reward design
    -------------
    * Cost measured via EXPLAIN plan estimated rows (deterministic, no timing).
    * Step rewards are raw / unclamped (may be negative — provides RL signal).
    * Terminal reward (done=True, task_solved=True):
          max(0.0, min(1.0, best_reward_seen_this_episode))
    * Terminal reward (done=True, task_solved=False): forced to 0.0.
    * task_solved requires cost_reduction_ratio > 0.3 (meaningful improvement).
    * One-shot reasoning bonus (+0.1) on the first explain or get_stats call.
    * Explain/get_stats bonuses alone CANNOT set task_solved.
    * Duplicate rewrite detection (whitespace-normalised): done=True, reward=-0.1.
    * Early success termination when CostReductionRatio > 0.95.
    * Level 4 is index-budget-only; rewrite actions are rejected.
    * Fully deterministic: reset(seed=N) reproduces identical datasets.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Init ─────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # DuckDB connection
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

        # Episode state
        self._scenario: Dict[str, Any] = {}
        self._current_sql: str = ""
        self._gold_sql: str = ""
        self._baseline_latency: float = 1.0
        self._baseline_cost: float = 1.0
        self._active_indexes: Dict[str, float] = {}  # idx_name → size_mb
        self._storage_budget_mb: float = 50.0
        self._storage_used_mb: float = 0.0
        self._max_steps: int = 15
        self._done: bool = False
        self._last_rewrite_sql: str = ""

        # Reward tracking
        self._best_reward: float = 0.0           # best intra-episode step reward
        self._cost_reduction_ratio: float = 0.0  # last computed ratio (for metadata)
        self._episode_failed: bool = False        # True when episode ends due to incorrect SQL
        self._task_solved: bool = False           # True only when real task success condition met

        # Deterministic cost baseline (from EXPLAIN estimated rows, not timing)
        self._baseline_plan_cost: float = 1.0

        # NumPy RNG for deterministic data generation
        self._rng: Optional[np.random.Generator] = None

        # Reasoning bonus (one-shot: fires on the first explain or get_stats call)
        self._reasoning_bonus_paid: bool = False

        # Materialised view grader state
        self._mv_created: bool = False
        self._mv_table_name: str = ""

        # Level 4 multi-query state
        self._multi_queries: List[str] = []
        self._multi_baselines: List[float] = []

        # N+1 state (Level 5)
        self._n_plus_one_user_ids: List[int] = []

        # Duplicate action detection (global)
        self._last_action_json: str = ""

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        level: Optional[int] = None,
        **kwargs: Any,
    ) -> DbaTunerObservation:
        """Reset the environment: generate data, pick scenario, measure baseline.

        Args:
            seed:       Random seed for reproducibility.
            episode_id: Optional episode identifier.
            level:      Specific scenario level (1–7).  None → random pick.
        """
        # Initialize RNG — use new-style Generator for full reproducibility
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if seed is not None:
            random.seed(seed)

        # Close previous connection
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

        self._conn = duckdb.connect(":memory:")
        self._conn.execute("PRAGMA memory_limit='2GB'")

        # Fresh episode state
        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)
        self._active_indexes = {}
        self._storage_used_mb = 0.0
        self._done = False
        self._last_rewrite_sql = ""
        self._last_action_json = ""
        self._best_reward = 0.0
        self._cost_reduction_ratio = 0.0
        self._episode_failed = False
        self._task_solved = False
        self._reasoning_bonus_paid = False
        self._baseline_plan_cost = 1.0
        self._mv_created = False
        self._mv_table_name = ""

        # Generate fresh dataset deterministically from seed
        self._generate_data()

        # Pick scenario
        if level is not None and 1 <= level <= len(SCENARIOS):
            self._scenario = SCENARIOS[level - 1]
        else:
            self._scenario = SCENARIOS[self._rng.integers(0, len(SCENARIOS))]

        self._storage_budget_mb = self._scenario["storage_budget_mb"]
        self._max_steps = self._scenario["max_steps"]

        lv = self._scenario["level"]

        if lv == 4:
            self._multi_queries = list(self._scenario["multi_queries"])
            self._gold_sql = "MULTI_QUERY"
            self._current_sql = "MULTI_QUERY"
            self._multi_baselines = []
            for q in self._multi_queries:
                _, lat = _execute_with_timeout(self._conn, q)
                self._multi_baselines.append(lat)
            self._baseline_latency = sum(self._multi_baselines)
            self._baseline_plan_cost = sum(
                _get_plan_cost(self._conn, q) for q in self._multi_queries
            )

        elif lv == 5:
            self._n_plus_one_user_ids = self._scenario["n_plus_one_user_ids"]
            self._gold_sql = self._scenario["gold_sql"]
            self._current_sql = "N_PLUS_ONE"
            total_lat = 0.0
            for uid in self._n_plus_one_user_ids:
                q = self._n_plus_one_query(uid)
                _, lat = _execute_with_timeout(self._conn, q)
                total_lat += lat
            self._baseline_latency = total_lat
            self._baseline_plan_cost = sum(
                _get_plan_cost(self._conn, self._n_plus_one_query(uid))
                for uid in self._n_plus_one_user_ids
            )

        else:
            self._gold_sql = self._scenario["gold_sql"]
            self._current_sql = self._scenario["initial_sql"]
            try:
                _, lat = _execute_with_timeout(self._conn, self._current_sql)
                self._baseline_latency = lat
                self._baseline_plan_cost = _get_plan_cost(self._conn, self._current_sql)
            except Exception:
                self._baseline_latency = 100.0
                self._baseline_plan_cost = 100.0

        return DbaTunerObservation(
            query_plan="Environment reset. Use 'explain' to see the query plan, 'get_stats' to analyse table statistics.",
            latency_ms=self._baseline_latency,
            total_cost=self._baseline_cost,
            storage_used_mb=0.0,
            storage_remaining_mb=self._storage_budget_mb,
            index_count=0,
            active_indexes=[],
            is_correct=True,
            current_sql=self._display_sql(),
            scenario_level=self._scenario["level"],
            scenario_description=self._scenario["description"],
            error_message="",
            done=False,
            reward=0.0,
            metadata={
                "step_count": 0,
                "episode_id": eid,
                "cost_reduction_ratio": 0.0,
                "reasoning_bonus_paid": False,
            },
        )

    # ── Step ──────────────────────────────────────────────────────────────

    def step(
        self,
        action: DbaTunerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DbaTunerObservation:
        """Execute one action in the environment and return a new observation."""
        if self._done:
            # Return consistent terminal reward: 0.0 for failed episodes, clamped best for success
            return self._make_obs(
                error_message="Episode is already done. Call reset() to start a new episode.",
                reward=0.0,
                done=True,
                is_correct=not self._episode_failed,
            )

        self._state.step_count += 1

        # ── Duplicate action detection (global) ───────────────────────────
        # Normalize action to detect identical repeats (including parameters)
        try:
            import json  # noqa: PLC0415
            current_action_json = json.dumps(action.dict(), sort_keys=True)
            if current_action_json == self._last_action_json and action.action_type != "done":
                self._done = True
                self._episode_failed = True
                return self._make_obs(
                    error_message=(
                        f"Repeated action '{action.action_type}' with identical parameters detected. "
                        "Episode terminated for efficiency — solve it, don't loop."
                    ),
                    reward=-0.1,
                    is_correct=False,
                    done=True,
                )
            self._last_action_json = current_action_json
        except Exception:
            pass

        if self._state.step_count > self._max_steps:
            self._done = True
            # Return terminal reward on max-steps termination
            terminal_r = max(0.0, min(1.0, self._best_reward)) if self._task_solved else 0.0
            return self._make_obs(
                error_message="Max steps reached.",
                reward=terminal_r,
                done=True,
            )

        try:
            atype = action.action_type
            if atype == "explain":
                obs = self._handle_explain()
            elif atype == "add_index":
                obs = self._handle_add_index(action.table, action.column)
            elif atype == "drop_index":
                obs = self._handle_drop_index(action.index_name)
            elif atype == "rewrite":
                obs = self._handle_rewrite(action.new_sql)
            elif atype == "get_stats":
                obs = self._handle_get_stats(action.table)
            elif atype == "create_materialized_view":
                obs = self._handle_create_materialized_view(action.view_name, action.sql)
            elif atype == "done":
                self._done = True
                terminal_r = max(0.0, min(1.0, self._best_reward)) if self._task_solved else 0.0
                return self._make_obs(
                    query_plan="Agent explicitly called done.",
                    reward=terminal_r,
                    is_correct=not self._episode_failed,
                    done=True,
                )
            else:
                return self._make_obs(
                    error_message=f"Unknown action_type: {atype!r}",
                    reward=0.0,
                )
        except TimeoutError as e:
            obs = self._make_obs(error_message=f"Timeout: {e}", reward=-0.05)
        except Exception as e:
            obs = self._make_obs(error_message=f"Error: {e}", reward=0.0)

        # ── Post-action max-step check (agent gets full max_steps actions) ─
        if self._state.step_count >= self._max_steps and not self._done:
            self._done = True
            terminal_r = max(0.0, min(1.0, self._best_reward)) if self._task_solved else 0.0
            obs.done = True
            obs.reward = round(terminal_r, 6)

        return obs

    # ── Action handlers ───────────────────────────────────────────────────

    def _handle_explain(self) -> DbaTunerObservation:
        """Run EXPLAIN ANALYZE on the current query."""
        # One-shot reasoning bonus
        bonus = 0.0
        if not self._reasoning_bonus_paid:
            bonus = 0.1
            self._reasoning_bonus_paid = True

        lv = self._scenario["level"]

        if lv == 4:
            plans = []
            for i, q in enumerate(self._multi_queries):
                try:
                    rows = self._conn.execute(f"EXPLAIN ANALYZE {q}").fetchall()
                    plans.append(f"--- Query {i+1} ---\n" + "\n".join(str(r) for r in rows))
                except Exception as e:
                    plans.append(f"--- Query {i+1} --- ERROR: {e}")
            plan_text = "\n\n".join(plans)

        elif lv == 5 and self._current_sql == "N_PLUS_ONE":
            uid = self._n_plus_one_user_ids[0]
            rows = self._conn.execute(
                f"EXPLAIN ANALYZE {self._n_plus_one_query(uid)}"
            ).fetchall()
            plan_text = (
                f"N+1 PATTERN: This query runs {len(self._n_plus_one_user_ids)} times in a loop "
                f"(once per user_id). Plan for user_id={uid}:\n"
                + "\n".join(str(r) for r in rows)
            )

        else:
            rows = self._conn.execute(
                f"EXPLAIN ANALYZE {self._current_sql}"
            ).fetchall()
            plan_text = "\n".join(str(r) for r in rows)

        plan_text = _shrink_plan(plan_text)
        reward = self._calculate_reward() + bonus
        return self._make_obs(query_plan=plan_text, reward=reward)

    def _handle_add_index(
        self, table: Optional[str], column: Optional[str]
    ) -> DbaTunerObservation:
        """Create an index on *table*.*column*."""
        if not table or not column:
            return self._make_obs(
                error_message="add_index requires 'table' and 'column' fields.",
                reward=0.0,
            )

        table = self._sanitise_identifier(table)
        column = self._sanitise_identifier(column)
        idx_name = f"idx_{table}_{column}"

        if idx_name in self._active_indexes:
            return self._make_obs(
                error_message=f"Index {idx_name} already exists.",
                reward=0.0,
            )

        size_mb = _estimate_index_size_mb(self._conn, table, column)
        if self._storage_used_mb + size_mb > self._storage_budget_mb:
            return self._make_obs(
                error_message=(
                    f"Storage budget exceeded. Need {size_mb:.2f} MB, "
                    f"only {self._storage_budget_mb - self._storage_used_mb:.2f} MB remaining."
                ),
                reward=-0.05,
            )

        try:
            self._conn.execute(f"CREATE INDEX {idx_name} ON {table}({column})")
        except Exception as e:
            return self._make_obs(
                error_message=f"Failed to create index: {e}", reward=0.0
            )

        self._active_indexes[idx_name] = size_mb
        self._storage_used_mb += size_mb

        reward = self._calculate_reward()
        return self._make_obs(
            query_plan=f"Index {idx_name} created ({size_mb:.2f} MB used).",
            reward=reward,
        )

    def _handle_drop_index(self, index_name: Optional[str]) -> DbaTunerObservation:
        """Drop a named index."""
        if not index_name:
            return self._make_obs(
                error_message="drop_index requires 'index_name' field.", reward=0.0
            )

        index_name = self._sanitise_identifier(index_name)

        if index_name not in self._active_indexes:
            return self._make_obs(
                error_message=f"Index {index_name} does not exist.", reward=0.0
            )

        try:
            self._conn.execute(f"DROP INDEX {index_name}")
        except Exception as e:
            return self._make_obs(
                error_message=f"Failed to drop index: {e}", reward=0.0
            )

        freed = self._active_indexes.pop(index_name)
        self._storage_used_mb -= freed

        reward = self._calculate_reward()
        return self._make_obs(
            query_plan=f"Index {index_name} dropped ({freed:.2f} MB freed).",
            reward=reward,
        )

    def _handle_rewrite(self, new_sql: Optional[str]) -> DbaTunerObservation:
        """Replace the current SQL with *new_sql*."""
        if not new_sql or not new_sql.strip():
            return self._make_obs(
                error_message="rewrite requires a non-empty 'new_sql' field.", reward=0.0
            )

        # ── Duplicate-rewrite detection (whitespace-normalised) ───────────
        normalised = " ".join(new_sql.split()).upper()
        normalised_last = " ".join(self._last_rewrite_sql.split()).upper()
        if normalised == normalised_last and normalised_last:
            self._done = True
            self._episode_failed = True
            return self._make_obs(
                error_message=(
                    "Repeated rewrite with identical SQL detected. "
                    "Episode terminated for efficiency — solve it, don't loop."
                ),
                reward=-0.1,
                is_correct=False,
                done=True,
            )
        self._last_rewrite_sql = new_sql

        # ── Sandbox: block destructive statements ─────────────────────────
        sql_upper = new_sql.strip().upper()
        forbidden = ["DELETE ", "DROP TABLE", "TRUNCATE", "ALTER TABLE", "UPDATE ", "INSERT "]
        for kw in forbidden:
            if kw in sql_upper and "DROP INDEX" not in sql_upper:
                self._done = True
                return self._make_obs(
                    error_message=(
                        f"Forbidden SQL keyword detected: {kw.strip()}. "
                        "Only SELECT, CREATE INDEX, DROP INDEX, and CREATE TABLE AS SELECT are allowed."
                    ),
                    reward=0.0,
                    done=True,
                )

        lv = self._scenario["level"]

        # ── Level 4: index-budget-only task, rewrite not applicable ────────
        if lv == 4:
            return self._make_obs(
                error_message=(
                    "Level 4 is an index-budget-only task. "
                    "Rewrite is not applicable — use add_index to optimise queries."
                ),
                reward=0.0,
            )

        # ── Materialised-view DDL path (CREATE TABLE AS) ──────────────────
        is_create_table_as = (
            sql_upper.lstrip().startswith("CREATE TABLE")
            and "AS" in sql_upper
        )

        if is_create_table_as:
            mv_name = _extract_create_table_name(new_sql)
            try:
                _execute_with_timeout(self._conn, new_sql)
            except Exception as e:
                return self._make_obs(
                    error_message=f"CREATE TABLE AS failed: {e}", reward=0.0
                )

            # Track MV state; do NOT update _current_sql (keep the original SELECT)
            self._mv_created = True
            self._mv_table_name = mv_name or ""

            # One-shot reasoning bonus (unlikely they've already triggered it here,
            # but honour the flag consistently)
            bonus = 0.0
            if not self._reasoning_bonus_paid:
                bonus = 0.1
                self._reasoning_bonus_paid = True

            # Small positive reward for the DDL step itself
            ddl_reward = 0.15 + bonus
            self._best_reward = max(self._best_reward, ddl_reward)

            return self._make_obs(
                query_plan=(
                    f"Materialised table '{self._mv_table_name}' created successfully. "
                    f"Now issue a rewrite with a SELECT query that reads from '{self._mv_table_name}' "
                    f"— that SELECT will be graded against the gold standard result set."
                ),
                reward=ddl_reward,
                is_correct=True,
            )

        # ── Normal SELECT / CTE rewrite path ─────────────────────────────
        self._current_sql = new_sql

        # ── Correctness check ─────────────────────────────────────────────
        is_correct = True

        if lv == 4:
            # Multi-query level: correctness not checked on rewrite
            is_correct = True
        else:
            try:
                import pandas as pd  # noqa: PLC0415

                df_agent = self._conn.execute(self._current_sql).df()
                df_gold = self._conn.execute(self._gold_sql).df()

                if df_agent.shape != df_gold.shape:
                    is_correct = False
                else:
                    # Rename agent columns to gold columns to be alias-tolerant
                    df_agent.columns = df_gold.columns
                    # Round float columns to 4 decimal places for tolerance
                    for col in df_gold.columns:
                        if df_gold[col].dtype in ('float64', 'float32'):
                            df_gold[col] = df_gold[col].round(4)
                            df_agent[col] = df_agent[col].round(4)
                    sort_cols = list(df_gold.columns)
                    df_agent = df_agent.sort_values(by=sort_cols, na_position='last').reset_index(drop=True)
                    df_gold = df_gold.sort_values(by=sort_cols, na_position='last').reset_index(drop=True)
                    is_correct = df_agent.equals(df_gold)

            except Exception as e:
                self._done = True
                self._episode_failed = True
                return self._make_obs(
                    error_message=f"Correctness check failed with invalid SQL error: {e}. Episode terminated.",
                    reward=0.0,
                    is_correct=False,
                    done=True,
                )

        if not is_correct:
            self._done = True
            self._episode_failed = True
            return self._make_obs(
                error_message=(
                    "Query results do not match the gold standard. "
                    "Episode terminated — reward = 0.0."
                ),
                reward=0.0,
                is_correct=False,
                done=True,
            )

        # ── Compute reward ────────────────────────────────────────────────
        reward = self._calculate_reward()
        cost_pct = f"{self._cost_reduction_ratio:.1%}"
        if self._cost_reduction_ratio > 0:
            feedback = f"Query rewritten successfully. Cost reduction: {cost_pct}."
        else:
            feedback = (
                "Query rewritten — no cost improvement detected. "
                "Try adding indexes on columns used in WHERE/JOIN clauses first."
            )
        return self._make_obs(
            query_plan=feedback,
            reward=reward,
            is_correct=True,
        )

    def _handle_get_stats(self, table: Optional[str]) -> DbaTunerObservation:
        """Return full cardinality / distribution statistics for *table*."""
        # One-shot reasoning bonus
        bonus = 0.0
        if not self._reasoning_bonus_paid:
            bonus = 0.1
            self._reasoning_bonus_paid = True

        if not table:
            return self._make_obs(
                error_message="get_stats requires 'table' field.", reward=0.0
            )

        table = self._sanitise_identifier(table)

        try:
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            cols = self._conn.execute(
                f"PRAGMA table_info('{table}')"
            ).fetchall()

            lines = [
                f"=== Table: {table} ===",
                f"Row count : {count:,}",
                f"Columns   : {len(cols)}",
                f"Budget    : {self._storage_budget_mb:.0f} MB total  |  "
                f"{self._storage_budget_mb - self._storage_used_mb:.2f} MB remaining",
                "",
            ]

            for ci in cols:
                col_name = ci[1]
                col_type = ci[2]

                distinct = self._conn.execute(
                    f"SELECT COUNT(DISTINCT {col_name}) FROM {table}"
                ).fetchone()[0]
                nulls = self._conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL"
                ).fetchone()[0]
                est_size = _estimate_index_size_mb(self._conn, table, col_name)

                lines.append(
                    f"  {col_name} ({col_type}): "
                    f"distinct={distinct:,}, nulls={nulls:,}, "
                    f"selectivity={distinct/max(count,1):.4f}, "
                    f"est_index_size={est_size:.2f} MB"
                )

                # Top-5 most frequent values (skew detection)
                if 0 < distinct < count:
                    try:
                        top5 = self._conn.execute(
                            f"SELECT {col_name}, COUNT(*) AS cnt "
                            f"FROM {table} GROUP BY {col_name} "
                            f"ORDER BY cnt DESC LIMIT 5"
                        ).fetchall()
                        top_str = ", ".join(f"{r[0]}({r[1]:,})" for r in top5)
                        lines.append(f"    top-5: {top_str}")
                    except Exception:
                        pass

            stats_text = "\n".join(lines)
            reward = self._calculate_reward() + bonus
            return self._make_obs(query_plan=stats_text, reward=reward)

        except Exception as e:
            return self._make_obs(
                error_message=f"Failed to get stats: {e}", reward=0.0
            )

    def _handle_create_materialized_view(
        self, view_name: Optional[str], sql: Optional[str]
    ) -> DbaTunerObservation:
        """Create a materialized view (CREATE TABLE AS SELECT).

        This is a first-class action for Level 7.  The agent provides a
        view_name and a SQL statement.  If the SQL is a bare SELECT, it is
        wrapped as CREATE TABLE {view_name} AS {sql}.
        """
        if not view_name or not sql or not sql.strip():
            return self._make_obs(
                error_message=(
                    "create_materialized_view requires 'view_name' and 'sql' fields."
                ),
                reward=0.0,
            )

        view_name = self._sanitise_identifier(view_name)
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()

        # If agent provided a bare SELECT, wrap it
        if sql_upper.startswith("SELECT"):
            create_sql = f"CREATE TABLE {view_name} AS {sql_stripped}"
        elif sql_upper.startswith("CREATE TABLE"):
            create_sql = sql_stripped
        else:
            return self._make_obs(
                error_message=(
                    "sql must be a SELECT or CREATE TABLE AS SELECT statement."
                ),
                reward=0.0,
            )

        try:
            _execute_with_timeout(self._conn, create_sql)
        except Exception as e:
            return self._make_obs(
                error_message=f"CREATE TABLE AS failed: {e}", reward=0.0
            )

        self._mv_created = True
        self._mv_table_name = view_name

        bonus = 0.0
        if not self._reasoning_bonus_paid:
            bonus = 0.1
            self._reasoning_bonus_paid = True

        ddl_reward = 0.15 + bonus
        self._best_reward = max(self._best_reward, ddl_reward)

        return self._make_obs(
            query_plan=(
                f"Materialised table '{view_name}' created successfully. "
                f"Now issue a rewrite with a SELECT query that reads from "
                f"'{view_name}' — that SELECT will be graded against the "
                f"gold standard result set."
            ),
            reward=ddl_reward,
            is_correct=True,
        )

    # ── Reward calculation ────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        """Compute the raw (unclamped) step reward using deterministic plan cost.

        Formula:
            CostReductionRatio  (from EXPLAIN estimated rows, not timing)
            - 0.02  x index_count
            - 0.005 x storage_used_mb
            - 0.005 x step_count

        The ratio is clamped to [0, 1] before penalty subtraction so that an
        *increase* in cost does not produce a strongly negative base.
        Penalties may still push the final value negative -- this is intentional
        RL signal.  Terminal clamping to [0.0, 1.0] is applied in _make_obs.

        Sets _task_solved = True when cost_reduction_ratio > 0.3 (meaningful
        improvement beyond what explain/get_stats bonuses can provide).
        """
        lv = self._scenario["level"]
        self._cost_reduction_ratio = 0.0

        try:
            if lv == 4:
                current_cost = sum(
                    _get_plan_cost(self._conn, q)
                    for q in self._multi_queries
                )
                ratio = 1.0 - (current_cost / max(self._baseline_plan_cost, 1.0))
            elif lv == 5 and self._current_sql == "N_PLUS_ONE":
                ratio = 0.0  # no improvement yet
            else:
                if (
                    not self._current_sql
                    or self._current_sql.strip().upper().startswith("CREATE TABLE")
                    or self._current_sql in ("N_PLUS_ONE", "MULTI_QUERY")
                ):
                    ratio = 0.0
                else:
                    current_cost = _get_plan_cost(self._conn, self._current_sql)
                    ratio = 1.0 - (current_cost / max(self._baseline_plan_cost, 1.0))
        except Exception:
            ratio = 0.0

        # Clamp ratio to [0.0, 1.0] for base reward
        ratio = max(0.0, min(1.0, ratio))
        self._cost_reduction_ratio = ratio

        # Mark task as solved when meaningful improvement is achieved
        if ratio > 0.3:
            self._task_solved = True

        # Early success termination
        if ratio > 0.95:
            self._done = True
            self._task_solved = True

        idx_count = len(self._active_indexes)
        mb_used = self._storage_used_mb

        reward = ratio - (0.02 * idx_count) - (0.005 * mb_used) - (0.005 * self._state.step_count)
        # NOTE: intentionally NOT clamped here -- negatives are valid RL signal
        return reward

    # ── Data generation ───────────────────────────────────────────────────

    def _generate_data(self) -> None:
        """Generate the 100k-row e-commerce dataset with Pareto (α=1.1) skew.

        All random data is generated via self._rng (numpy Generator) for full
        seed-based reproducibility.  No DuckDB random() calls are used.
        """
        conn = self._conn
        rng = self._rng
        ALPHA = 1.1  # Aggressive skew: ~20% of keys → ~80% of rows

        # ── Users (100k) ─────────────────────────────────────────────────
        num_users = 100_000
        countries = [
            "US", "UK", "DE", "FR", "JP", "IN", "BR", "CA", "AU", "MX",
            "IT", "ES", "KR", "NL", "SE", "NO", "PL", "RU", "CN", "ZA",
        ]
        conn.execute("""
            CREATE TABLE users (
                user_id     INTEGER PRIMARY KEY,
                username    VARCHAR,
                email       VARCHAR,
                signup_date DATE,
                country     VARCHAR
            )
        """)
        conn.execute(f"""
            INSERT INTO users
            SELECT
                i AS user_id,
                'user_' || i AS username,
                'user_' || i || '@example.com' AS email,
                DATE '2020-01-01' + INTERVAL (i % 1095) DAY AS signup_date,
                CASE (i % {len(countries)})
                    {" ".join(f"WHEN {j} THEN '{c}'" for j, c in enumerate(countries))}
                END AS country
            FROM generate_series(1, {num_users}) t(i)
        """)

        # ── Products (10k) ────────────────────────────────────────────────
        num_products = 10_000
        categories = [
            "Electronics", "Clothing", "Books", "Home", "Sports",
            "Toys", "Food", "Beauty", "Auto", "Garden",
        ]
        conn.execute("""
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                name       VARCHAR,
                category   VARCHAR,
                price      DOUBLE,
                created_at DATE
            )
        """)
        conn.execute(f"""
            INSERT INTO products
            SELECT
                i AS product_id,
                'product_' || i AS name,
                CASE (i % {len(categories)})
                    {" ".join(f"WHEN {j} THEN '{c}'" for j, c in enumerate(categories))}
                END AS category,
                ROUND(5.0 + (i % 500) * 0.5, 2) AS price,
                DATE '2020-01-01' + INTERVAL (i % 730) DAY AS created_at
            FROM generate_series(1, {num_products}) t(i)
        """)

        # ── Orders (100k) with Pareto-skewed user_id ─────────────────────
        num_orders = 100_000
        statuses = ["pending", "completed", "cancelled", "shipped", "returned"]

        raw_u = rng.pareto(a=ALPHA, size=num_orders) + 1.0
        user_ids = np.clip(
            (raw_u / raw_u.max() * num_users).astype(int), 1, num_users
        )

        # Generate all columns deterministically in Python
        order_day_offsets = rng.integers(0, 365, size=num_orders)
        order_status_idx = rng.integers(0, len(statuses), size=num_orders)
        order_amounts = np.round(10.0 + rng.random(num_orders) * 990.0, 2)

        conn.execute("""
            CREATE TABLE orders (
                order_id     INTEGER PRIMARY KEY,
                user_id      INTEGER,
                order_date   DATE,
                status       VARCHAR,
                total_amount DOUBLE
            )
        """)

        # Build orders data as list of tuples and bulk insert
        orders_data = []
        for i in range(num_orders):
            orders_data.append((
                i + 1,
                int(user_ids[i]),
                f"2023-01-01",  # placeholder, offset applied below
                statuses[int(order_status_idx[i])],
                float(order_amounts[i]),
            ))

        # Use batch insert via temp table for speed
        conn.execute("CREATE TEMPORARY TABLE _tmp_orders (oid INTEGER, uid INTEGER, day_offset INTEGER, status VARCHAR, amount DOUBLE)")
        batch_size = 10_000
        for start in range(0, num_orders, batch_size):
            end = min(start + batch_size, num_orders)
            vals = ", ".join(
                f"({i+1}, {int(user_ids[i])}, {int(order_day_offsets[i])}, '{statuses[int(order_status_idx[i])]}', {float(order_amounts[i])})"
                for i in range(start, end)
            )
            conn.execute(f"INSERT INTO _tmp_orders SELECT * FROM (VALUES {vals})")

        conn.execute("""
            INSERT INTO orders
            SELECT oid AS order_id, uid AS user_id,
                   DATE '2023-01-01' + INTERVAL (day_offset) DAY AS order_date,
                   status, amount AS total_amount
            FROM _tmp_orders
        """)
        conn.execute("DROP TABLE _tmp_orders")

        # ── LineItems (300k) with Pareto-skewed product_id ────────────────
        num_line_items = 300_000

        raw_p = rng.pareto(a=ALPHA, size=num_line_items) + 1.0
        product_ids = np.clip(
            (raw_p / raw_p.max() * num_products).astype(int), 1, num_products
        )

        # Generate all columns deterministically
        li_order_ids = rng.integers(1, num_orders + 1, size=num_line_items)
        li_quantities = rng.integers(1, 11, size=num_line_items)
        li_unit_prices = np.round(5.0 + rng.random(num_line_items) * 200.0, 2)

        conn.execute("""
            CREATE TABLE line_items (
                line_item_id INTEGER PRIMARY KEY,
                order_id     INTEGER,
                product_id   INTEGER,
                quantity     INTEGER,
                unit_price   DOUBLE
            )
        """)
        conn.execute("CREATE TEMPORARY TABLE _tmp_li (lid INTEGER, oid INTEGER, pid INTEGER, qty INTEGER, price DOUBLE)")
        for start in range(0, num_line_items, batch_size):
            end = min(start + batch_size, num_line_items)
            vals = ", ".join(
                f"({i+1}, {int(li_order_ids[i])}, {int(product_ids[i])}, {int(li_quantities[i])}, {float(li_unit_prices[i])})"
                for i in range(start, end)
            )
            conn.execute(f"INSERT INTO _tmp_li SELECT * FROM (VALUES {vals})")

        conn.execute("""
            INSERT INTO line_items
            SELECT lid AS line_item_id, oid AS order_id, pid AS product_id,
                   qty AS quantity, price AS unit_price
            FROM _tmp_li
        """)
        conn.execute("DROP TABLE _tmp_li")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _n_plus_one_query(self, user_id: int) -> str:
        return (
            f"SELECT o.order_id, o.user_id, o.total_amount, "
            f"li.line_item_id, li.product_id, li.quantity, li.unit_price "
            f"FROM orders o "
            f"JOIN line_items li ON o.order_id = li.order_id "
            f"WHERE o.user_id = {user_id} "
            f"ORDER BY o.order_id, li.line_item_id"
        )

    def _display_sql(self) -> str:
        if self._current_sql == "N_PLUS_ONE":
            return f"-- N+1 loop over user_ids: {self._n_plus_one_user_ids}"
        if self._current_sql == "MULTI_QUERY":
            return "\n---\n".join(self._multi_queries)
        return self._current_sql

    def _make_obs(
        self,
        query_plan: str = "",
        reward: float = 0.0,
        done: bool | None = None,
        is_correct: bool = True,
        error_message: str = "",
    ) -> DbaTunerObservation:
        """Build a DbaTunerObservation from current state.

        Reward clamping policy:
        - Non-terminal steps: raw reward passed through (may be negative).
        - Terminal steps (done=True):
            * is_correct=False or _episode_failed → reward = 0.0
            * _task_solved=True  → reward = max(0.0, min(1.0, best_reward_seen))
            * _task_solved=False → reward = 0.0  (explain/get_stats alone cannot win)
        """
        if done is None:
            done = self._done

        # Only track best reward while episode is still active
        if not self._done or not done:
            self._best_reward = max(self._best_reward, reward)

        # Terminal reward clamping
        if done:
            if not is_correct or self._episode_failed:
                reward = 0.0
            elif self._task_solved:
                reward = max(0.0, min(1.0, self._best_reward))
            else:
                reward = 0.0

        # Measure current latency/cost for the observation
        latency_ms = 0.0
        total_cost = 0.0
        lv = self._scenario.get("level", 1)

        if not error_message and not done:
            try:
                if lv == 4:
                    for q in self._multi_queries:
                        _, lat = _execute_with_timeout(self._conn, q)
                        latency_ms += lat
                    total_cost = sum(
                        _get_plan_cost(self._conn, q) for q in self._multi_queries
                    )
                elif lv == 5 and self._current_sql == "N_PLUS_ONE":
                    for uid in self._n_plus_one_user_ids:
                        _, lat = _execute_with_timeout(self._conn, self._n_plus_one_query(uid))
                        latency_ms += lat
                    total_cost = sum(
                        _get_plan_cost(self._conn, self._n_plus_one_query(uid))
                        for uid in self._n_plus_one_user_ids
                    )
                elif self._current_sql and self._current_sql not in ("N_PLUS_ONE", "MULTI_QUERY"):
                    if self._current_sql.strip().upper().startswith("CREATE TABLE"):
                        latency_ms = 0.0
                        total_cost = self._baseline_plan_cost
                    else:
                        _, latency_ms = _execute_with_timeout(self._conn, self._current_sql)
                        total_cost = _get_plan_cost(self._conn, self._current_sql)
            except Exception:
                pass

        return DbaTunerObservation(
            query_plan=query_plan,
            latency_ms=latency_ms,
            total_cost=total_cost,
            storage_used_mb=self._storage_used_mb,
            storage_remaining_mb=self._storage_budget_mb - self._storage_used_mb,
            index_count=len(self._active_indexes),
            active_indexes=[
                f"{name}: {size:.2f} MB" for name, size in self._active_indexes.items()
            ],
            is_correct=is_correct,
            current_sql=self._display_sql(),
            scenario_level=self._scenario.get("level", 1),
            scenario_description=self._scenario.get("description", ""),
            error_message=error_message,
            done=done,
            reward=round(reward, 6),
            metadata={
                "step_count": self._state.step_count,
                "episode_id": self._state.episode_id,
                "cost_reduction_ratio": round(self._cost_reduction_ratio, 4),
                "reasoning_bonus_paid": self._reasoning_bonus_paid,
                "task_solved": self._task_solved,
                "mv_created": self._mv_created,
                "mv_table_name": self._mv_table_name,
            },
        )

    @staticmethod
    def _sanitise_identifier(name: str) -> str:
        """Allow only alphanumeric + underscore in SQL identifiers."""
        sanitised = re.sub(r"[^a-zA-Z0-9_]", "", name)
        if not sanitised:
            raise ValueError(f"Invalid SQL identifier: {name!r}")
        return sanitised

    @property
    def state(self) -> State:
        """Current environment state (episode_id + step_count)."""
        return self._state

    def close(self) -> None:
        """Clean up the DuckDB connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ──────────────────────────────────────────────────────────────────────────────
# Direct smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = DbaTunerEnvironment()
    obs = env.reset(seed=42, level=1)
    print(f"Level {obs.scenario_level}: {obs.scenario_description[:80]}…")
    print(f"Baseline latency: {obs.latency_ms:.2f} ms")
    print()

    # Step 1: explain (should earn +0.1 bonus)
    obs = env.step(DbaTunerAction(action_type="explain"))
    print(f"[explain]  reward={obs.reward:.4f}  bonus_paid={obs.metadata['reasoning_bonus_paid']}")

    # Step 2: explain again (no bonus — already paid)
    obs = env.step(DbaTunerAction(action_type="explain"))
    print(f"[explain2] reward={obs.reward:.4f}  bonus_paid={obs.metadata['reasoning_bonus_paid']}")

    # Step 3: get_stats (no bonus — already paid)
    obs = env.step(DbaTunerAction(action_type="get_stats", table="orders"))
    print(f"[get_stats] reward={obs.reward:.4f}")
    print(obs.query_plan[:600])
    print()

    # Step 4: add_index
    obs = env.step(DbaTunerAction(action_type="add_index", table="orders", column="user_id"))
    print(
        f"[add_index] reward={obs.reward:.4f}  "
        f"indexes={obs.active_indexes}  "
        f"cost_ratio={obs.metadata['cost_reduction_ratio']}"
    )
    print(f"done={obs.done}")
    env.close()
    print("\nSmoke-test passed ✓")
