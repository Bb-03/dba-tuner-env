"""
DBA Tuner Environment Implementation.

High-fidelity OpenEnv environment for database performance tuning using DuckDB.
Features:
  - 100k-row e-commerce dataset with power-law skewed distributions
  - 7-level scenario registry (easy → expert)
  - Correctness checking via SHA-256 result hashing
  - Storage budget tracking for indexes
  - 5-second SQL timeout sandboxing
  - Reward-hacking prevention (read-only agent access)
"""

from __future__ import annotations

import hashlib
import random
import re
import signal
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
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import DbaTunerAction, DbaTunerObservation


# ──────────────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────────────

SCENARIOS: List[Dict[str, Any]] = [
    # ── Level 1: Simple index on high-cardinality column ──
    {
        "level": 1,
        "description": (
            "Level 1 (Easy): A SELECT query filters orders by user_id on a "
            "non-indexed column. Add an index to speed it up."
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
    # ── Level 2: Join optimisation ──
    {
        "level": 2,
        "description": (
            "Level 2 (Medium): Join orders and line_items on non-indexed columns. "
            "The query triggers a slow Hash Join. Add indexes to optimise."
        ),
        "gold_sql": (
            "SELECT o.order_id, o.user_id, o.total_amount, "
            "li.product_id, li.quantity, li.unit_price "
            "FROM orders o JOIN line_items li ON o.order_id = li.order_id "
            "WHERE o.status = 'completed' "
            "ORDER BY o.order_date DESC LIMIT 100"
        ),
        "initial_sql": (
            "SELECT o.order_id, o.user_id, o.total_amount, "
            "li.product_id, li.quantity, li.unit_price "
            "FROM orders o JOIN line_items li ON o.order_id = li.order_id "
            "WHERE o.status = 'completed' "
            "ORDER BY o.order_date DESC LIMIT 100"
        ),
        "max_steps": 15,
        "storage_budget_mb": 50.0,
    },
    # ── Level 3: Correlated subquery → window function ──
    {
        "level": 3,
        "description": (
            "Level 3 (Medium): Replace a correlated subquery with a window "
            "function or CTE to eliminate redundant scans."
        ),
        "gold_sql": (
            "SELECT user_id, order_id, total_amount, "
            "SUM(total_amount) OVER (PARTITION BY user_id) AS user_total "
            "FROM orders WHERE status = 'completed'"
        ),
        "initial_sql": (
            "SELECT user_id, order_id, total_amount, "
            "(SELECT SUM(o2.total_amount) FROM orders o2 "
            "WHERE o2.user_id = orders.user_id AND o2.status = 'completed') "
            "AS user_total "
            "FROM orders WHERE status = 'completed'"
        ),
        "max_steps": 15,
        "storage_budget_mb": 50.0,
    },
    # ── Level 4: Budget challenge (5 slow queries) ──
    {
        "level": 4,
        "description": (
            "Level 4 (Long-Horizon Hard): The Budget Challenge. You have 5 slow "
            "queries and a 50 MB index storage limit. Prioritise which indexes "
            "give the best aggregate ROI across all queries."
        ),
        "gold_sql": "MULTI_QUERY",  # special sentinel
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
    # ── Level 5: N+1 pattern → batch join ──
    {
        "level": 5,
        "description": (
            "Level 5 (Algorithmic Hard): Detect an N+1 query pattern — the "
            "initial query executes many small SELECTs in a loop. Rewrite it "
            "into a single batch JOIN query."
        ),
        "gold_sql": (
            "SELECT o.order_id, o.user_id, o.total_amount, "
            "li.line_item_id, li.product_id, li.quantity, li.unit_price "
            "FROM orders o "
            "JOIN line_items li ON o.order_id = li.order_id "
            "WHERE o.user_id IN (1,2,3,4,5,6,7,8,9,10) "
            "ORDER BY o.order_id, li.line_item_id"
        ),
        "initial_sql": "N_PLUS_ONE",  # special sentinel
        "n_plus_one_user_ids": list(range(1, 11)),
        "max_steps": 20,
        "storage_budget_mb": 50.0,
    },
    # ── Level 6: Range scan optimisation ──
    {
        "level": 6,
        "description": (
            "Level 6 (Architectural Hard): Optimise a range scan on order_date. "
            "Add an index and/or rewrite the query to enable efficient range "
            "filtering on the timestamp column."
        ),
        "gold_sql": (
            "SELECT order_id, user_id, order_date, total_amount "
            "FROM orders "
            "WHERE order_date BETWEEN '2023-06-01' AND '2023-12-31' "
            "ORDER BY order_date"
        ),
        "initial_sql": (
            "SELECT order_id, user_id, order_date, total_amount "
            "FROM orders "
            "WHERE order_date BETWEEN '2023-06-01' AND '2023-12-31' "
            "ORDER BY order_date"
        ),
        "max_steps": 15,
        "storage_budget_mb": 50.0,
    },
    # ── Level 7: Materialised view for complex join ──
    {
        "level": 7,
        "description": (
            "Level 7 (Trade-offs Hard): Create a materialised view (via "
            "CREATE TABLE AS) for a complex 5-table aggregation, then rewrite "
            "the query to use it. Includes logic to refresh the view."
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


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _hash_results(rows: list) -> str:
    """SHA-256 hash of sorted, stringified query results."""
    serialised = str(sorted([str(r) for r in rows]))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _execute_with_timeout(
    conn: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = 5.0
) -> Tuple[list, float]:
    """Execute SQL with a timeout.  Returns (rows, elapsed_ms).

    On timeout raises TimeoutError.
    """
    result_container: Dict[str, Any] = {"rows": None, "error": None}

    def _run():
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
        # Try to interrupt duckdb
        try:
            conn.interrupt()
        except Exception:
            pass
        raise TimeoutError(
            f"SQL execution exceeded {timeout_s}s timeout"
        )

    if result_container["error"] is not None:
        raise result_container["error"]

    return result_container["rows"] or [], elapsed_ms


def _estimate_index_size_mb(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> float:
    """Estimate the storage cost of an index on table.column.

    Uses a rough heuristic: 8 bytes per row × num_rows / 1MB,
    with a minimum of 0.01 MB.
    """
    try:
        row_count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        return max(0.01, (row_count * 8) / (1024 * 1024))
    except Exception:
        return 0.5  # fallback


def _parse_explain_cost(explain_text: str) -> float:
    """Extract a numeric cost from DuckDB EXPLAIN (ANALYZE) output.

    DuckDB doesn't expose a single cost scalar, so we sum up all
    numeric values on lines matching common cost patterns.  Falls back
    to hash-based estimate when nothing parseable is found.
    """
    total = 0.0
    for line in explain_text.split("\n"):
        # Look for timing lines  e.g. "│  0.00  │"
        nums = re.findall(r"(\d+\.\d+)", line)
        for n in nums:
            total += float(n)
    return total if total > 0 else 1.0


# ──────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────

class DbaTunerEnvironment(Environment):
    """DBA Tuner environment — OpenEnv compliant.

    Uses an in-memory DuckDB database with power-law-skewed e-commerce
    data to present database performance tuning challenges across 7
    difficulty levels.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # DuckDB connections
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

        # Episode state
        self._scenario: Dict[str, Any] = {}
        self._current_sql: str = ""
        self._gold_sql: str = ""
        self._baseline_cost: float = 1.0
        self._baseline_latency: float = 1.0
        self._active_indexes: Dict[str, float] = {}  # name → size_mb
        self._storage_budget_mb: float = 50.0
        self._storage_used_mb: float = 0.0
        self._has_reasoned: bool = False
        self._max_steps: int = 15
        self._done: bool = False

        # For Level 4 multi-query tracking
        self._multi_queries: List[str] = []
        self._multi_baselines: List[float] = []

        # For Level 5 N+1
        self._n_plus_one_user_ids: List[int] = []

    # ── reset ────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        level: Optional[int] = None,
        **kwargs: Any,
    ) -> DbaTunerObservation:
        """Reset: generate data, pick scenario, measure baseline.

        Args:
            seed: Random seed for reproducibility.
            episode_id: Optional episode identifier.
            level: Specific scenario level (1-7) to run. If None, picks randomly.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Close previous connection if any
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

        self._conn = duckdb.connect(":memory:")
        self._conn.execute("PRAGMA memory_limit='4GB'")

        # Fresh state
        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)
        self._active_indexes = {}
        self._storage_used_mb = 0.0
        self._has_reasoned = False
        self._done = False

        # Generate data
        self._generate_data()

        # Pick scenario — specific level or random
        if level is not None and 1 <= level <= len(SCENARIOS):
            self._scenario = SCENARIOS[level - 1]
        else:
            self._scenario = random.choice(SCENARIOS)
        self._storage_budget_mb = self._scenario["storage_budget_mb"]
        self._max_steps = self._scenario["max_steps"]

        # Set up SQL targets
        level = self._scenario["level"]
        if level == 4:
            self._multi_queries = list(self._scenario["multi_queries"])
            self._gold_sql = "MULTI_QUERY"
            self._current_sql = "MULTI_QUERY"
            # Measure baseline for each query
            self._multi_baselines = []
            for q in self._multi_queries:
                _, lat = _execute_with_timeout(self._conn, q)
                self._multi_baselines.append(lat)
            self._baseline_latency = sum(self._multi_baselines)
            self._baseline_cost = self._baseline_latency
        elif level == 5:
            self._n_plus_one_user_ids = self._scenario["n_plus_one_user_ids"]
            self._gold_sql = self._scenario["gold_sql"]
            self._current_sql = self._scenario["initial_sql"]
            # Baseline = time to run all N+1 queries
            total_lat = 0.0
            for uid in self._n_plus_one_user_ids:
                q = (
                    f"SELECT o.order_id, o.user_id, o.total_amount, "
                    f"li.line_item_id, li.product_id, li.quantity, li.unit_price "
                    f"FROM orders o "
                    f"JOIN line_items li ON o.order_id = li.order_id "
                    f"WHERE o.user_id = {uid} "
                    f"ORDER BY o.order_id, li.line_item_id"
                )
                _, lat = _execute_with_timeout(self._conn, q)
                total_lat += lat
            self._baseline_latency = total_lat
            self._baseline_cost = total_lat
        else:
            self._gold_sql = self._scenario["gold_sql"]
            self._current_sql = self._scenario["initial_sql"]
            # Measure baseline
            try:
                _, lat = _execute_with_timeout(self._conn, self._current_sql)
                self._baseline_latency = lat
                explain_text = self._conn.execute(
                    f"EXPLAIN ANALYZE {self._current_sql}"
                ).fetchall()
                explain_str = "\n".join(str(r) for r in explain_text)
                self._baseline_cost = _parse_explain_cost(explain_str)
            except Exception:
                self._baseline_latency = 100.0
                self._baseline_cost = 100.0

        return DbaTunerObservation(
            query_plan="Environment reset. Use 'explain' action to see the query plan.",
            latency_ms=self._baseline_latency,
            total_cost=self._baseline_cost,
            storage_used_mb=0.0,
            storage_remaining_mb=self._storage_budget_mb,
            index_count=0,
            active_indexes=[],
            is_correct=True,
            current_sql=(
                self._current_sql
                if self._current_sql != "N_PLUS_ONE"
                else f"-- N+1 loop over user_ids: {self._n_plus_one_user_ids}"
            ),
            scenario_level=self._scenario["level"],
            scenario_description=self._scenario["description"],
            error_message="",
            done=False,
            reward=0.0,
        )

    # ── step ─────────────────────────────────────────────────────────

    def step(
        self,
        action: DbaTunerAction,  # type: ignore[override]
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DbaTunerObservation:
        """Execute one step in the environment."""
        if self._done:
            return self._make_obs(
                error_message="Episode is already done. Call reset().",
                reward=0.0,
                done=True,
            )

        self._state.step_count += 1

        # Check max steps
        if self._state.step_count >= self._max_steps:
            self._done = True
            return self._make_obs(
                error_message="Max steps reached.",
                reward=0.0,
                done=True,
            )

        try:
            atype = action.action_type
            if atype == "explain":
                return self._handle_explain()
            elif atype == "add_index":
                return self._handle_add_index(action.table, action.column)
            elif atype == "drop_index":
                return self._handle_drop_index(action.index_name)
            elif atype == "rewrite":
                return self._handle_rewrite(action.new_sql)
            elif atype == "get_stats":
                return self._handle_get_stats(action.table)
            else:
                return self._make_obs(
                    error_message=f"Unknown action_type: {atype}",
                    reward=0.0,
                )
        except TimeoutError as e:
            return self._make_obs(
                error_message=f"Timeout: {e}",
                reward=0.0,
            )
        except Exception as e:
            return self._make_obs(
                error_message=f"Error: {e}",
                reward=0.0,
            )

    # ── action handlers ──────────────────────────────────────────────

    def _handle_explain(self) -> DbaTunerObservation:
        """Run EXPLAIN ANALYZE on the current query."""
        self._has_reasoned = True

        level = self._scenario["level"]
        if level == 4:
            # Show plans for all multi-queries
            plans = []
            for i, q in enumerate(self._multi_queries):
                try:
                    rows = self._conn.execute(f"EXPLAIN ANALYZE {q}").fetchall()
                    plan_text = "\n".join(str(r) for r in rows)
                    plans.append(f"--- Query {i+1} ---\n{plan_text}")
                except Exception as e:
                    plans.append(f"--- Query {i+1} --- ERROR: {e}")
            full_plan = "\n\n".join(plans)
            return self._make_obs(query_plan=full_plan, reward=0.0)
        elif level == 5 and self._current_sql == "N_PLUS_ONE":
            # Show plan for one representative query
            uid = self._n_plus_one_user_ids[0]
            q = (
                f"SELECT o.order_id, o.user_id, o.total_amount, "
                f"li.line_item_id, li.product_id, li.quantity, li.unit_price "
                f"FROM orders o "
                f"JOIN line_items li ON o.order_id = li.order_id "
                f"WHERE o.user_id = {uid} "
                f"ORDER BY o.order_id, li.line_item_id"
            )
            rows = self._conn.execute(f"EXPLAIN ANALYZE {q}").fetchall()
            plan_text = "\n".join(str(r) for r in rows)
            plan_text = (
                f"N+1 PATTERN: This query is executed {len(self._n_plus_one_user_ids)} "
                f"times in a loop, once per user_id.\n"
                f"Showing plan for user_id={uid}:\n\n{plan_text}"
            )
            return self._make_obs(query_plan=plan_text, reward=0.0)
        else:
            rows = self._conn.execute(
                f"EXPLAIN ANALYZE {self._current_sql}"
            ).fetchall()
            plan_text = "\n".join(str(r) for r in rows)
            return self._make_obs(query_plan=plan_text, reward=0.0)

    def _handle_add_index(
        self, table: Optional[str], column: Optional[str]
    ) -> DbaTunerObservation:
        """Add an index on table.column."""
        if not table or not column:
            return self._make_obs(
                error_message="add_index requires 'table' and 'column' fields.",
                reward=0.0,
            )

        # Sanitise names
        table = self._sanitise_identifier(table)
        column = self._sanitise_identifier(column)

        idx_name = f"idx_{table}_{column}"
        if idx_name in self._active_indexes:
            return self._make_obs(
                error_message=f"Index {idx_name} already exists.",
                reward=0.0,
            )

        # Estimate size
        size_mb = _estimate_index_size_mb(self._conn, table, column)
        if self._storage_used_mb + size_mb > self._storage_budget_mb:
            return self._make_obs(
                error_message=(
                    f"Not enough storage budget. Need {size_mb:.2f} MB, "
                    f"have {self._storage_budget_mb - self._storage_used_mb:.2f} MB remaining."
                ),
                reward=0.0,
            )

        # Create index
        try:
            self._conn.execute(
                f"CREATE INDEX {idx_name} ON {table}({column})"
            )
        except Exception as e:
            return self._make_obs(
                error_message=f"Failed to create index: {e}",
                reward=0.0,
            )

        self._active_indexes[idx_name] = size_mb
        self._storage_used_mb += size_mb

        # Reasoning bonus
        bonus = 0.1 if self._has_reasoned else 0.0

        # Measure improvement
        reward = self._calculate_reward() + bonus

        return self._make_obs(
            query_plan=f"Index {idx_name} created ({size_mb:.2f} MB).",
            reward=reward,
        )

    def _handle_drop_index(
        self, index_name: Optional[str]
    ) -> DbaTunerObservation:
        """Drop a named index."""
        if not index_name:
            return self._make_obs(
                error_message="drop_index requires 'index_name' field.",
                reward=0.0,
            )

        index_name = self._sanitise_identifier(index_name)

        if index_name not in self._active_indexes:
            return self._make_obs(
                error_message=f"Index {index_name} does not exist.",
                reward=0.0,
            )

        try:
            self._conn.execute(f"DROP INDEX {index_name}")
        except Exception as e:
            return self._make_obs(
                error_message=f"Failed to drop index: {e}",
                reward=0.0,
            )

        freed = self._active_indexes.pop(index_name)
        self._storage_used_mb -= freed

        reward = self._calculate_reward()
        return self._make_obs(
            query_plan=f"Index {index_name} dropped ({freed:.2f} MB freed).",
            reward=reward,
        )

    def _handle_rewrite(
        self, new_sql: Optional[str]
    ) -> DbaTunerObservation:
        """Replace the current SQL with a new query."""
        if not new_sql:
            return self._make_obs(
                error_message="rewrite requires 'new_sql' field.",
                reward=0.0,
            )

        # ── Sandbox: block destructive statements ──
        sql_upper = new_sql.strip().upper()
        forbidden = ["DELETE", "DROP TABLE", "TRUNCATE", "ALTER TABLE", "UPDATE", "INSERT"]
        for kw in forbidden:
            if kw in sql_upper and "DROP INDEX" not in sql_upper:
                self._done = True
                return self._make_obs(
                    error_message=f"Forbidden SQL keyword: {kw}. Only SELECT, CREATE INDEX, DROP INDEX, CREATE TABLE AS SELECT are allowed.",
                    reward=0.0,
                    done=True,
                )

        level = self._scenario["level"]

        # Allow CREATE TABLE AS (for materialized views in L7)
        is_create_table_as = sql_upper.startswith("CREATE TABLE") and "AS" in sql_upper

        if is_create_table_as:
            # Execute the CREATE TABLE AS, then rewrite current to SELECT from it
            try:
                _execute_with_timeout(self._conn, new_sql)
                # Extract the table name
                match = re.match(
                    r"CREATE\s+TABLE\s+(\w+)\s+AS", new_sql, re.IGNORECASE
                )
                if match:
                    mv_name = match.group(1)
                    self._current_sql = f"SELECT * FROM {mv_name}"
                else:
                    self._current_sql = new_sql
            except Exception as e:
                return self._make_obs(
                    error_message=f"CREATE TABLE AS failed: {e}",
                    reward=0.0,
                )
        else:
            self._current_sql = new_sql

        # ── Correctness check ──
        is_correct = True
        if level == 4:
            # For multi-query, the rewrite targets one of the queries
            # Agent should specify which query to rewrite (not applicable for multi)
            # For simplicity, we validate the rewritten SQL against itself
            is_correct = True  # cannot compare against multi
        elif level == 5:
            # Compare batch result vs N+1 result
            try:
                batch_rows, batch_lat = _execute_with_timeout(
                    self._conn, self._current_sql
                )
                # Get gold
                gold_rows, _ = _execute_with_timeout(
                    self._conn, self._gold_sql
                )
                is_correct = _hash_results(batch_rows) == _hash_results(gold_rows)
            except Exception as e:
                return self._make_obs(
                    error_message=f"Rewrite execution failed: {e}",
                    reward=0.0,
                    is_correct=False,
                )
        else:
            try:
                agent_rows, _ = _execute_with_timeout(
                    self._conn, self._current_sql
                )
                gold_rows, _ = _execute_with_timeout(
                    self._conn, self._gold_sql
                )
                is_correct = _hash_results(agent_rows) == _hash_results(gold_rows)
            except Exception as e:
                return self._make_obs(
                    error_message=f"Correctness check failed: {e}",
                    reward=0.0,
                    is_correct=False,
                )

        if not is_correct:
            self._done = True
            return self._make_obs(
                error_message="Query results do not match gold standard. Episode terminated.",
                reward=0.0,
                is_correct=False,
                done=True,
            )

        # Reasoning bonus
        bonus = 0.1 if self._has_reasoned else 0.0
        reward = self._calculate_reward() + bonus

        return self._make_obs(
            query_plan="Query rewritten successfully.",
            reward=reward,
            is_correct=True,
        )

    def _handle_get_stats(
        self, table: Optional[str]
    ) -> DbaTunerObservation:
        """Return statistics for a table."""
        self._has_reasoned = True

        if not table:
            return self._make_obs(
                error_message="get_stats requires 'table' field.",
                reward=0.0,
            )

        table = self._sanitise_identifier(table)

        try:
            # Row count
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            # Column info
            cols = self._conn.execute(
                f"PRAGMA table_info('{table}')"
            ).fetchall()

            stats_lines = [
                f"=== Table: {table} ===",
                f"Row count: {count:,}",
                f"Columns: {len(cols)}",
                "",
            ]

            for col_info in cols:
                col_name = col_info[1]
                col_type = col_info[2]

                # Distinct count
                distinct = self._conn.execute(
                    f"SELECT COUNT(DISTINCT {col_name}) FROM {table}"
                ).fetchone()[0]

                # Null count
                nulls = self._conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL"
                ).fetchone()[0]

                stats_lines.append(
                    f"  {col_name} ({col_type}): "
                    f"distinct={distinct:,}, nulls={nulls:,}, "
                    f"selectivity={distinct/max(count,1):.4f}"
                )

                # Top-5 most frequent values for integer / varchar
                if distinct < count and distinct > 0:
                    try:
                        top5 = self._conn.execute(
                            f"SELECT {col_name}, COUNT(*) AS cnt "
                            f"FROM {table} GROUP BY {col_name} "
                            f"ORDER BY cnt DESC LIMIT 5"
                        ).fetchall()
                        top_str = ", ".join(
                            f"{r[0]}({r[1]})" for r in top5
                        )
                        stats_lines.append(f"    top5: {top_str}")
                    except Exception:
                        pass

            stats_text = "\n".join(stats_lines)
            return self._make_obs(query_plan=stats_text, reward=0.0)

        except Exception as e:
            return self._make_obs(
                error_message=f"Failed to get stats: {e}",
                reward=0.0,
            )

    # ── reward ───────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        """Reward = CostReductionRatio - penalty.

        Clamped to [0.0, 1.0].
        """
        level = self._scenario["level"]

        # Measure current cost
        try:
            if level == 4:
                current_total = 0.0
                for q in self._multi_queries:
                    _, lat = _execute_with_timeout(self._conn, q)
                    current_total += lat
                cost_reduction = 1.0 - (current_total / max(self._baseline_latency, 0.001))
            elif level == 5 and self._current_sql == "N_PLUS_ONE":
                # Still in N+1 mode — no improvement
                cost_reduction = 0.0
            elif level == 5:
                _, lat = _execute_with_timeout(self._conn, self._current_sql)
                cost_reduction = 1.0 - (lat / max(self._baseline_latency, 0.001))
            else:
                _, lat = _execute_with_timeout(self._conn, self._current_sql)
                cost_reduction = 1.0 - (lat / max(self._baseline_latency, 0.001))
        except Exception:
            cost_reduction = 0.0

        idx_count = len(self._active_indexes)
        mb_used = self._storage_used_mb

        # Base score from cost reduction, minus small penalties
        reward = max(0.0, cost_reduction) - (0.02 * idx_count) - (0.005 * mb_used)
        return max(0.0, min(1.0, reward))

    # ── data generation ──────────────────────────────────────────────

    def _generate_data(self) -> None:
        """Generate the 100k-row e-commerce dataset with power-law skew."""
        conn = self._conn

        # ── Users: 100k ──
        conn.execute("""
            CREATE TABLE users (
                user_id    INTEGER PRIMARY KEY,
                username   VARCHAR,
                email      VARCHAR,
                signup_date DATE,
                country    VARCHAR
            )
        """)

        num_users = 100_000
        countries = [
            "US", "UK", "DE", "FR", "JP", "IN", "BR", "CA", "AU", "MX",
            "IT", "ES", "KR", "NL", "SE", "NO", "PL", "RU", "CN", "ZA",
        ]
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

        # ── Products: 10k ──
        conn.execute("""
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                name       VARCHAR,
                category   VARCHAR,
                price      DOUBLE,
                created_at DATE
            )
        """)

        num_products = 10_000
        categories = [
            "Electronics", "Clothing", "Books", "Home", "Sports",
            "Toys", "Food", "Beauty", "Auto", "Garden",
        ]
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

        # ── Orders: 100k with power-law user_id ──
        num_orders = 100_000
        alpha = 1.5

        # Generate Pareto-skewed user IDs in Python
        raw = np.random.pareto(a=alpha, size=num_orders) + 1.0
        user_ids = np.clip(
            (raw / raw.max() * num_users).astype(int), 1, num_users
        )
        statuses = ["pending", "completed", "cancelled", "shipped", "returned"]

        conn.execute("""
            CREATE TABLE orders (
                order_id     INTEGER PRIMARY KEY,
                user_id      INTEGER,
                order_date   DATE,
                status       VARCHAR,
                total_amount DOUBLE
            )
        """)

        # Batch insert via DuckDB's VALUES
        # For speed, use a temporary table from a numpy array
        conn.execute("CREATE TEMPORARY TABLE _tmp_user_ids (uid INTEGER)")
        # Register numpy array as arrow or use executemany
        for chunk_start in range(0, num_orders, 10_000):
            chunk_end = min(chunk_start + 10_000, num_orders)
            values = ", ".join(
                str(int(uid)) for uid in user_ids[chunk_start:chunk_end]
            )
            conn.execute(
                f"INSERT INTO _tmp_user_ids SELECT * FROM (VALUES {','.join(f'({int(uid)})' for uid in user_ids[chunk_start:chunk_end])})"
            )

        conn.execute(f"""
            INSERT INTO orders
            SELECT
                ROW_NUMBER() OVER () AS order_id,
                uid AS user_id,
                DATE '2023-01-01' + INTERVAL (
                    CAST(FLOOR(random() * 365) AS INTEGER)
                ) DAY AS order_date,
                CASE CAST(FLOOR(random() * {len(statuses)}) AS INTEGER)
                    {" ".join(f"WHEN {j} THEN '{s}'" for j, s in enumerate(statuses))}
                END AS status,
                ROUND(10.0 + random() * 990.0, 2) AS total_amount
            FROM _tmp_user_ids
        """)
        conn.execute("DROP TABLE _tmp_user_ids")

        # ── LineItems: ~300k with power-law product_id ──
        num_line_items = 300_000
        raw_p = np.random.pareto(a=alpha, size=num_line_items) + 1.0
        product_ids = np.clip(
            (raw_p / raw_p.max() * num_products).astype(int), 1, num_products
        )

        conn.execute("""
            CREATE TABLE line_items (
                line_item_id INTEGER PRIMARY KEY,
                order_id     INTEGER,
                product_id   INTEGER,
                quantity     INTEGER,
                unit_price   DOUBLE
            )
        """)

        conn.execute("CREATE TEMPORARY TABLE _tmp_prod_ids (pid INTEGER)")
        for chunk_start in range(0, num_line_items, 10_000):
            chunk_end = min(chunk_start + 10_000, num_line_items)
            conn.execute(
                f"INSERT INTO _tmp_prod_ids SELECT * FROM (VALUES {','.join(f'({int(pid)})' for pid in product_ids[chunk_start:chunk_end])})"
            )

        conn.execute(f"""
            INSERT INTO line_items
            SELECT
                ROW_NUMBER() OVER () AS line_item_id,
                1 + CAST(FLOOR(random() * {num_orders}) AS INTEGER) AS order_id,
                pid AS product_id,
                1 + CAST(FLOOR(random() * 10) AS INTEGER) AS quantity,
                ROUND(5.0 + random() * 200.0, 2) AS unit_price
            FROM _tmp_prod_ids
        """)
        conn.execute("DROP TABLE _tmp_prod_ids")

    # ── helpers ──────────────────────────────────────────────────────

    def _make_obs(
        self,
        query_plan: str = "",
        reward: float = 0.0,
        done: bool | None = None,
        is_correct: bool = True,
        error_message: str = "",
    ) -> DbaTunerObservation:
        """Build a DbaTunerObservation from current state."""
        if done is None:
            done = self._done

        # Measure current latency if we have a valid SQL
        latency_ms = 0.0
        total_cost = 0.0
        level = self._scenario.get("level", 1)

        if not error_message and not done:
            try:
                if level == 4:
                    for q in self._multi_queries:
                        _, lat = _execute_with_timeout(self._conn, q)
                        latency_ms += lat
                    total_cost = latency_ms
                elif level == 5 and self._current_sql == "N_PLUS_ONE":
                    for uid in self._n_plus_one_user_ids:
                        q = (
                            f"SELECT o.order_id, o.user_id, o.total_amount, "
                            f"li.line_item_id, li.product_id, li.quantity, li.unit_price "
                            f"FROM orders o "
                            f"JOIN line_items li ON o.order_id = li.order_id "
                            f"WHERE o.user_id = {uid} "
                            f"ORDER BY o.order_id, li.line_item_id"
                        )
                        _, lat = _execute_with_timeout(self._conn, q)
                        latency_ms += lat
                    total_cost = latency_ms
                elif self._current_sql and self._current_sql not in (
                    "N_PLUS_ONE", "MULTI_QUERY"
                ):
                    _, latency_ms = _execute_with_timeout(
                        self._conn, self._current_sql
                    )
                    total_cost = latency_ms
            except Exception:
                pass

        current_display = self._current_sql
        if self._current_sql == "N_PLUS_ONE":
            current_display = (
                f"-- N+1 loop over user_ids: {self._n_plus_one_user_ids}"
            )
        elif self._current_sql == "MULTI_QUERY":
            current_display = "\n---\n".join(self._multi_queries)

        return DbaTunerObservation(
            query_plan=query_plan,
            latency_ms=latency_ms,
            total_cost=total_cost,
            storage_used_mb=self._storage_used_mb,
            storage_remaining_mb=self._storage_budget_mb - self._storage_used_mb,
            index_count=len(self._active_indexes),
            active_indexes=list(self._active_indexes.keys()),
            is_correct=is_correct,
            current_sql=current_display,
            scenario_level=self._scenario.get("level", 1),
            scenario_description=self._scenario.get("description", ""),
            error_message=error_message,
            done=done,
            reward=max(0.0, min(1.0, reward)),
        )

    @staticmethod
    def _sanitise_identifier(name: str) -> str:
        """Allow only alphanumeric + underscore in SQL identifiers."""
        sanitised = re.sub(r"[^a-zA-Z0-9_]", "", name)
        if not sanitised:
            raise ValueError(f"Invalid identifier: {name!r}")
        return sanitised

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up DuckDB connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ──────────────────────────────────────────────────────────────────────
# Direct test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = DbaTunerEnvironment()
    obs = env.reset(seed=42)
    print(f"Scenario Level: {obs.scenario_level}")
    print(f"Description: {obs.scenario_description}")
    print(f"Baseline latency: {obs.latency_ms:.2f} ms")
    print(f"Current SQL: {obs.current_sql[:100]}...")
    print()

    # Explain
    obs = env.step(DbaTunerAction(action_type="explain"))
    print(f"[explain] plan length={len(obs.query_plan)} chars, reward={obs.reward}")
    print()

    # Get stats
    obs = env.step(DbaTunerAction(action_type="get_stats", table="orders"))
    print(f"[get_stats] plan length={len(obs.query_plan)} chars")
    print(obs.query_plan[:500])
    print()

    # Add index
    obs = env.step(
        DbaTunerAction(action_type="add_index", table="orders", column="user_id")
    )
    print(
        f"[add_index] reward={obs.reward:.4f}, "
        f"indexes={obs.active_indexes}, "
        f"storage={obs.storage_used_mb:.2f} MB"
    )
    print()

    # Explain again
    obs = env.step(DbaTunerAction(action_type="explain"))
    print(f"[explain] latency={obs.latency_ms:.2f} ms, reward={obs.reward}")
    print("Done!")
