"""
DBA Tuner Environment Implementation.

High-fidelity OpenEnv environment for database query rewrite optimisation.

Key features
------------
* 100k-row e-commerce dataset (Pareto skewed).
* Fully deterministic.
* 3-task analytical curriculum: UNION ALL consolidation, Window function decorrelation, Summary tables.
* 0-compute deterministic cost grading based purely on exact physical operator counting from DuckDB EXPLAIN.
* One-shot reasoning bonus (+0.1) for first explain/get_stats.
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


_SCHEMA_HINT = (
    "SCHEMA: "
    "users(user_id PK, username, email, signup_date, country) 100k rows | "
    "products(product_id PK, name, category, price, created_at) 10k rows | "
    "orders(order_id PK, user_id, order_date, status, total_amount) 100k rows | "
    "line_items(line_item_id PK, order_id, product_id, quantity, unit_price) 300k rows."
)


SCENARIOS: List[Dict[str, Any]] = [
    {
        "level": 1,
        "description": (
            "Task 1 (Easy) — UNION ALL Consolidation.\n"
            "Three separate UNION ALL queries are scanning the orders table redundantly. "
            "Rewrite this into a single query using a CASE WHEN statement inside the aggregation.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT CASE WHEN total_amount > 500 THEN 'high' WHEN total_amount >= 100 THEN 'medium' ELSE 'low' END AS segment, "
            "COUNT(*) AS cnt FROM orders GROUP BY segment"
        ),
        "verification_sql": (
            "SELECT CASE WHEN total_amount > 500 THEN 'high' WHEN total_amount >= 100 THEN 'medium' ELSE 'low' END AS segment, "
            "COUNT(*) AS cnt FROM orders GROUP BY segment"
        ),
        "initial_sql": (
            "SELECT 'high' AS segment, COUNT(*) AS cnt FROM orders WHERE total_amount > 500 "
            "UNION ALL SELECT 'medium', COUNT(*) FROM orders WHERE total_amount BETWEEN 100 AND 500 "
            "UNION ALL SELECT 'low', COUNT(*) FROM orders WHERE total_amount < 100"
        ),
        "max_steps": 10,
    },
    {
        "level": 2,
        "description": (
            "Task 2 (Medium) — Correlated Subquery Decorrelation.\n"
            "A correlated subquery is fetching the total amount spent by each user for every order row. "
            "Rewrite this to use an efficient Window Function (SUM(...) OVER(...)) or an INNER JOIN with a CTE.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT order_id, user_id, total_amount, SUM(total_amount) OVER(PARTITION BY user_id) AS user_lifetime_value "
            "FROM orders WHERE status = 'completed'"
        ),
        "verification_sql": (
            "SELECT order_id, user_id, total_amount, SUM(total_amount) OVER(PARTITION BY user_id) AS user_lifetime_value "
            "FROM orders WHERE status = 'completed'"
        ),
        "initial_sql": (
            "SELECT order_id, user_id, total_amount, "
            "(SELECT SUM(total_amount) FROM orders o2 WHERE o2.user_id = orders.user_id) AS user_lifetime_value "
            "FROM orders WHERE status = 'completed'"
        ),
        "max_steps": 15,
    },
    {
        "level": 3,
        "description": (
            "Task 3 (Hard) — The Materialized View (Summary Table).\n"
            "A heavy aggregation is counting distinct orders and summing revenue per user. "
            "Use CREATE TABLE mv_summary AS SELECT ... to pre-compute this, then submit a second "
            "rewrite action simply doing SELECT * FROM mv_summary.\n"
            + _SCHEMA_HINT
        ),
        "gold_sql": (
            "SELECT * FROM mv_summary"
        ),
        "verification_sql": (
            "SELECT u.user_id, u.country, COUNT(DISTINCT o.order_id) AS order_count, SUM(o.total_amount) AS total_revenue "
            "FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.user_id, u.country"
        ),
        "initial_sql": (
            "SELECT u.user_id, u.country, COUNT(DISTINCT o.order_id) AS order_count, SUM(o.total_amount) AS total_revenue "
            "FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.user_id, u.country"
        ),
        "max_steps": 15,
    },
]


def _execute_with_timeout(
    conn: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = 5.0
) -> Tuple[list, float]:
    """Execute *sql* with a wall-clock timeout."""
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


def _shrink_plan(raw_plan: str, top_n: int = 5) -> str:
    """Compress DuckDB EXPLAIN output to make it compact."""
    text = raw_plan
    text = re.sub(r"\('", "", text)
    text = re.sub(r"',\)", "", text)
    text = re.sub(r"[─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬┃┏┓┗┛]", "", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    if len(lines) > 20:
        return "\n".join(lines[:20]) + "\n... (truncated)"
    return "\n".join(lines)


class DbaTunerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._scenario: Dict[str, Any] = {}
        self._current_sql: str = ""
        self._gold_sql: str = ""
        self._verification_sql: str = ""
        self._baseline_latency: float = 1.0
        self._max_steps: int = 15
        self._done: bool = False

        self._cost_reduction_ratio: float = 0.0
        self._episode_failed: bool = False
        self._task_solved: bool = False
        self._baseline_plan_cost: float = 100.0
        self._rng: Optional[np.random.Generator] = None
        self._reasoning_bonus_paid: bool = False
        self._last_action_json: str = ""

    def _get_plan_complexity(self, sql: str) -> int:
        """0-compute cost based purely on physical execution tree size."""
        try:
            rows = self._conn.execute(f"EXPLAIN {sql}").fetchall()
            plan_text = "\n".join(str(r).upper() for r in rows)
            
            operator_count = (
                plan_text.count("_SCAN") + 
                plan_text.count("_JOIN") + 
                plan_text.count("GROUP") + 
                plan_text.count("FILTER") + 
                plan_text.count("PROJECTION") +
                plan_text.count("WINDOW")
            )
            return max(1, operator_count)
        except Exception:
            return 100

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        level: Optional[int] = None,
        **kwargs: Any,
    ) -> DbaTunerObservation:
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if seed is not None:
            random.seed(seed)

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

        self._conn = duckdb.connect(":memory:")
        self._conn.execute("PRAGMA memory_limit='2GB'")

        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)
        self._done = False
        self._last_action_json = ""
        self._cost_reduction_ratio = 0.0
        self._episode_failed = False
        self._task_solved = False
        self._reasoning_bonus_paid = False

        self._generate_data()

        if level is not None and 1 <= level <= len(SCENARIOS):
            self._scenario = SCENARIOS[level - 1]
        else:
            self._scenario = SCENARIOS[int(self._rng.integers(0, len(SCENARIOS)))]

        self._max_steps = self._scenario["max_steps"]
        self._gold_sql = self._scenario["gold_sql"]
        self._verification_sql = self._scenario["verification_sql"]
        self._current_sql = self._scenario["initial_sql"]
        
        try:
            _, lat = _execute_with_timeout(self._conn, self._current_sql)
            self._baseline_latency = lat
        except Exception:
            self._baseline_latency = 100.0
            
        self._baseline_plan_cost = float(self._get_plan_complexity(self._current_sql))

        return DbaTunerObservation(
            query_plan="Environment reset. Use 'explain' to see the query plan.",
            latency_ms=self._baseline_latency,
            total_cost=self._baseline_plan_cost,
            is_correct=True,
            current_sql=self._current_sql,
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
                "task_solved": False,
            },
        )

    def step(
        self,
        action: DbaTunerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DbaTunerObservation:
        if self._done:
            return self._make_obs(
                error_message="Episode is already done. Call reset() to start a new episode.",
                reward=0.0,
                done=True,
                is_correct=not self._episode_failed,
            )

        self._state.step_count += 1

        try:
            import json as _json  # noqa: PLC0415
            current_action_json = _json.dumps(action.dict(), sort_keys=True)
            if current_action_json == self._last_action_json and action.action_type != "done":
                self._done = True
                self._episode_failed = True
                return self._make_obs(
                    error_message="Repeated identical action detected. Episode terminated.",
                    reward=-0.1,
                    is_correct=False,
                    done=True,
                )
            self._last_action_json = current_action_json
        except Exception:
            pass

        if self._state.step_count > self._max_steps:
            self._done = True
            return self._make_obs(
                error_message="Max steps reached.",
                reward=self._compute_terminal_reward(),
                done=True,
            )

        try:
            atype = action.action_type
            if atype == "explain":
                obs = self._handle_explain()
            elif atype == "get_stats":
                obs = self._handle_get_stats(action.table)
            elif atype == "rewrite":
                obs = self._handle_rewrite(action.sql)
            elif atype == "done":
                self._done = True
                return self._make_obs(
                    query_plan="Agent explicitly called done.",
                    reward=self._compute_terminal_reward(),
                    is_correct=not self._episode_failed,
                    done=True,
                )
            else:
                return self._make_obs(error_message=f"Unknown action: {atype!r}", reward=0.0)
        except TimeoutError as e:
            obs = self._make_obs(error_message=f"Timeout: {e}", reward=-0.05)
        except Exception as e:
            obs = self._make_obs(error_message=f"Error: {e}", reward=0.0)

        if self._state.step_count >= self._max_steps and not self._done:
            self._done = True
            obs.done = True
            obs.reward = round(self._compute_terminal_reward(), 6)

        return obs

    def _handle_explain(self) -> DbaTunerObservation:
        bonus = 0.0
        if not self._reasoning_bonus_paid:
            bonus = 0.1
            self._reasoning_bonus_paid = True

        rows = self._conn.execute(f"EXPLAIN {self._current_sql}").fetchall()
        plan_text = "\n".join(str(r) for r in rows)
        plan_text = _shrink_plan(plan_text)
        
        return self._make_obs(query_plan=plan_text, reward=self._calculate_reward() + bonus)

    def _handle_get_stats(self, table: Optional[str]) -> DbaTunerObservation:
        bonus = 0.0
        if not self._reasoning_bonus_paid:
            bonus = 0.1
            self._reasoning_bonus_paid = True

        if not table:
            return self._make_obs(error_message="get_stats requires 'table' field.", reward=0.0)

        table = self._sanitise_identifier(table)
        try:
            count = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = self._conn.execute(f"PRAGMA table_info('{table}')").fetchall()

            lines = [f"=== Table: {table} ===", f"Row count : {count:,}", f"Columns   : {len(cols)}", ""]
            for ci in cols:
                col_name = ci[1]
                col_type = ci[2]
                distinct = self._conn.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table}").fetchone()[0]
                lines.append(f"  {col_name} ({col_type}): distinct={distinct:,}")

            stats_text = "\n".join(lines)
            return self._make_obs(query_plan=stats_text, reward=self._calculate_reward() + bonus)
        except Exception as e:
            return self._make_obs(error_message=f"Failed to get stats: {e}", reward=0.0)

    def _handle_rewrite(self, new_sql: Optional[str]) -> DbaTunerObservation:
        if not new_sql:
            return self._make_obs(error_message="rewrite requires 'sql' field.", reward=0.0)

        is_create = new_sql.strip().upper().startswith("CREATE")
        
        if not is_create:
            try:
                # 1. Execute both queries
                new_df = self._conn.execute(new_sql).df()
                gold_df = self._conn.execute(self._verification_sql).df()
                
                # 2. Strict Dataset Correctness Check
                if len(new_df.columns) != len(gold_df.columns) or len(new_df) != len(gold_df):
                    self._done = True
                    self._episode_failed = True
                    return self._make_obs(error_message="Result row/column count mismatch.", is_correct=False, reward=0.0, done=True)
                
                # 3. Align column order
                new_df = new_df.reindex(sorted(new_df.columns), axis=1)
                gold_df = gold_df.reindex(sorted(gold_df.columns), axis=1)
                
                # 4. Round floats and sort to prevent micro-variance failures
                for col in gold_df.columns:
                    if gold_df[col].dtype in ('float64', 'float32'):
                        gold_df[col] = gold_df[col].round(4)
                        if col in new_df.columns:
                            new_df[col] = new_df[col].round(4)
                            
                # Sort rows to ensure order doesn't trigger false failure
                sort_cols = list(gold_df.columns)
                new_df = new_df.sort_values(by=list(new_df.columns), na_position='last').reset_index(drop=True)
                gold_df = gold_df.sort_values(by=sort_cols, na_position='last').reset_index(drop=True)
                
                # Compare underlying values to bypass column alias mismatches
                if not np.array_equal(new_df.values, gold_df.values):
                    self._done = True
                    self._episode_failed = True
                    return self._make_obs(error_message="Query results do not match gold standard values.", is_correct=False, reward=0.0, done=True)

            except Exception as e:
                self._done = True
                self._episode_failed = True
                return self._make_obs(error_message=f"SQL Execution Error: {str(e)}", is_correct=False, reward=0.0, done=True)
        else:
            # Bypass correctness for CREATE TABLE (Task 3 Materialized View)
            try:
                self._conn.execute(new_sql)
            except Exception as e:
                return self._make_obs(error_message=f"Failed to create table: {e}", reward=-0.05)

        self._current_sql = new_sql
        return self._make_obs(query_plan="Query successfully rewritten.", reward=self._calculate_reward())
        
    def _calculate_reward(self) -> float:
        self._cost_reduction_ratio = 0.0
        try:
            if not self._current_sql or self._current_sql.strip().upper().startswith("CREATE"):
                ratio = 0.0
            else:
                baseline = self._baseline_plan_cost
                current_cost = float(self._get_plan_complexity(self._current_sql))
                ratio = 1.0 - (current_cost / baseline)
        except Exception:
            ratio = 0.0
            
        ratio = max(0.0, min(1.0, ratio))
        self._cost_reduction_ratio = ratio

        if ratio > 0.15:
            self._task_solved = True
        
        return ratio - (0.01 * self._state.step_count)

    def _compute_terminal_reward(self) -> float:
        if self._episode_failed:
            return 0.0
        if not self._task_solved:
            return 0.0
        return max(0.0, min(1.0, self._cost_reduction_ratio))

    def _generate_data(self) -> None:
        conn = self._conn
        rng = self._rng
        ALPHA = 1.1

        num_users = 100_000
        countries = ["US", "UK", "DE", "FR", "JP", "IN", "BR", "CA", "AU", "MX"]
        conn.execute("CREATE TABLE users (user_id INTEGER PRIMARY KEY, username VARCHAR, email VARCHAR, signup_date DATE, country VARCHAR)")
        conn.execute(f"INSERT INTO users SELECT i, 'user_'||i, 'user_'||i||'@example.com', DATE '2020-01-01' + INTERVAL (i % 1095) DAY, CASE (i % {len(countries)}) " + " ".join(f"WHEN {j} THEN '{c}'" for j, c in enumerate(countries)) + f" END FROM generate_series(1, {num_users}) t(i)")

        num_products = 10_000
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
        conn.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR, category VARCHAR, price DOUBLE, created_at DATE)")
        conn.execute(f"INSERT INTO products SELECT i, 'product_'||i, CASE (i % {len(categories)}) " + " ".join(f"WHEN {j} THEN '{c}'" for j, c in enumerate(categories)) + f" END, ROUND(5.0 + (i % 500) * 0.5, 2), DATE '2020-01-01' + INTERVAL (i % 730) DAY FROM generate_series(1, {num_products}) t(i)")

        num_orders = 100_000
        statuses = ["pending", "completed", "cancelled", "shipped", "returned"]
        raw_u = rng.pareto(a=ALPHA, size=num_orders) + 1.0
        user_ids = np.clip((raw_u / raw_u.max() * num_users).astype(int), 1, num_users)
        order_day_offsets = rng.integers(0, 365, size=num_orders)
        order_status_idx = rng.integers(0, len(statuses), size=num_orders)
        order_amounts = np.round(10.0 + rng.random(num_orders) * 990.0, 2)

        conn.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER, order_date DATE, status VARCHAR, total_amount DOUBLE)")
        conn.execute("CREATE TEMPORARY TABLE _tmp_orders (oid INTEGER, uid INTEGER, day_offset INTEGER, status VARCHAR, amount DOUBLE)")
        for start in range(0, num_orders, 10_000):
            end = min(start + 10_000, num_orders)
            vals = ", ".join(f"({i+1}, {int(user_ids[i])}, {int(order_day_offsets[i])}, '{statuses[int(order_status_idx[i])]}', {float(order_amounts[i])})" for i in range(start, end))
            conn.execute(f"INSERT INTO _tmp_orders SELECT * FROM (VALUES {vals})")
        conn.execute("INSERT INTO orders SELECT oid, uid, DATE '2023-01-01' + INTERVAL (day_offset) DAY, status, amount FROM _tmp_orders")
        conn.execute("DROP TABLE _tmp_orders")

        num_line_items = 300_000
        raw_p = rng.pareto(a=ALPHA, size=num_line_items) + 1.0
        product_ids = np.clip((raw_p / raw_p.max() * num_products).astype(int), 1, num_products)
        li_order_ids = rng.integers(1, num_orders + 1, size=num_line_items)
        li_quantities = rng.integers(1, 11, size=num_line_items)
        li_unit_prices = np.round(5.0 + rng.random(num_line_items) * 200.0, 2)

        conn.execute("CREATE TABLE line_items (line_item_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price DOUBLE)")
        conn.execute("CREATE TEMPORARY TABLE _tmp_li (lid INTEGER, oid INTEGER, pid INTEGER, qty INTEGER, price DOUBLE)")
        for start in range(0, num_line_items, 10_000):
            end = min(start + 10_000, num_line_items)
            vals = ", ".join(f"({i+1}, {int(li_order_ids[i])}, {int(product_ids[i])}, {int(li_quantities[i])}, {float(li_unit_prices[i])})" for i in range(start, end))
            conn.execute(f"INSERT INTO _tmp_li SELECT * FROM (VALUES {vals})")
        conn.execute("INSERT INTO line_items SELECT lid, oid, pid, qty, price FROM _tmp_li")
        conn.execute("DROP TABLE _tmp_li")

    def _make_obs(
        self,
        query_plan: str = "",
        reward: float = 0.0,
        done: bool | None = None,
        is_correct: bool = True,
        error_message: str = "",
    ) -> DbaTunerObservation:
        if done is None:
            done = self._done

        if done:
            if not is_correct or self._episode_failed:
                reward = 0.0
            elif self._task_solved:
                reward = max(0.0, min(1.0, reward))
            else:
                reward = 0.0

        latency_ms = 0.0
        total_cost = 0.0

        if not error_message and not done:
            try:
                if self._current_sql:
                    _, latency_ms = _execute_with_timeout(self._conn, self._current_sql)
                    total_cost = float(self._get_plan_complexity(self._current_sql))
            except Exception:
                pass

        return DbaTunerObservation(
            query_plan=query_plan,
            latency_ms=latency_ms,
            total_cost=total_cost,
            is_correct=is_correct,
            current_sql=self._current_sql,
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
            },
        )

    @staticmethod
    def _sanitise_identifier(name: str) -> str:
        sanitised = re.sub(r"[^a-zA-Z0-9_]", "", name)
        if not sanitised:
            raise ValueError(f"Invalid SQL identifier: {name!r}")
        return sanitised

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


if __name__ == "__main__":
    env = DbaTunerEnvironment()
    obs = env.reset(seed=42, level=1)
    print(f"Level {obs.scenario_level}: {obs.scenario_description[:80]}…")
    
    obs = env.step(DbaTunerAction(action_type="explain"))
    print(f"[explain] reward={obs.reward:.4f} cost={obs.total_cost}")
    
    # Task 1 Rewrite
    rewrite_sql = "SELECT CASE WHEN total_amount > 500 THEN 'high' WHEN total_amount >= 100 THEN 'medium' ELSE 'low' END AS segment, COUNT(*) AS cnt FROM orders GROUP BY segment"
    obs = env.step(DbaTunerAction(action_type="rewrite", sql=rewrite_sql))
    print(f"[rewrite] reward={obs.reward:.4f} cost={obs.total_cost} ratio={obs.metadata['cost_reduction_ratio']}")
    
    obs = env.step(DbaTunerAction(action_type="done"))
    print(f"[done] final_score={obs.reward:.4f}")
    env.close()