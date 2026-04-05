"""
Data models for the DBA Tuner Env Environment.

Defines a discriminated-union action space for database performance tuning
and a rich observation model carrying query plans, metrics, and correctness info.
"""

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action – single model with action_type discriminator
# ---------------------------------------------------------------------------

class DbaTunerAction(Action):
    """Action for the DBA Tuner environment.

    The *action_type* field selects which operation to perform:

    * ``explain``   – Run EXPLAIN ANALYZE on the current query.
    * ``add_index`` – CREATE INDEX on *table*.*column* (costs storage budget).
    * ``drop_index``– DROP INDEX by *index_name* (frees budget).
    * ``rewrite``   – Replace the current SQL with *new_sql*.
    * ``get_stats`` – Retrieve cardinality / distribution stats for *table*.
    """

    action_type: Literal[
        "explain", "add_index", "drop_index", "rewrite", "get_stats"
    ] = Field(..., description="Which action to perform")

    # --- add_index ---
    table: Optional[str] = Field(
        default=None, description="Target table (add_index / get_stats)"
    )
    column: Optional[str] = Field(
        default=None, description="Target column (add_index)"
    )

    # --- drop_index ---
    index_name: Optional[str] = Field(
        default=None, description="Index name to drop (drop_index)"
    )

    # --- rewrite ---
    new_sql: Optional[str] = Field(
        default=None, description="Replacement SQL query (rewrite)"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class DbaTunerObservation(Observation):
    """Observation returned by the DBA Tuner environment after every step."""

    # Query plan / stats text
    query_plan: str = Field(
        default="", description="EXPLAIN ANALYZE output or stats text"
    )

    # Performance metrics
    latency_ms: float = Field(
        default=0.0, description="Execution time of the current query (ms)"
    )
    total_cost: float = Field(
        default=0.0, description="Estimated cost from EXPLAIN"
    )

    # Storage budget
    storage_used_mb: float = Field(
        default=0.0, description="Index storage consumed (MB)"
    )
    storage_remaining_mb: float = Field(
        default=50.0, description="Remaining storage budget (MB)"
    )

    # Index tracking
    index_count: int = Field(
        default=0, description="Number of active agent-created indexes"
    )
    active_indexes: List[str] = Field(
        default_factory=list, description="Names of active indexes"
    )

    # Correctness
    is_correct: bool = Field(
        default=True,
        description="Whether the current SQL produces correct results",
    )

    # Current state
    current_sql: str = Field(
        default="", description="The current SQL query string"
    )

    # Scenario information
    scenario_level: int = Field(
        default=1, description="Difficulty level (1-7)"
    )
    scenario_description: str = Field(
        default="", description="Human-readable task description"
    )

    # Errors
    error_message: str = Field(
        default="", description="Error details if any action failed"
    )
