"""DBA Tuner Env Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import DbaTunerAction, DbaTunerObservation
except (ImportError, SystemError):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import DbaTunerAction, DbaTunerObservation


class DbaTunerEnv(
    EnvClient[DbaTunerAction, DbaTunerObservation, State]
):
    """
    Client for the DBA Tuner Env Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with DbaTunerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.scenario_description)
        ...
        ...     result = client.step(DbaTunerAction(action_type="explain"))
        ...     print(result.observation.query_plan)
    """

    def _step_payload(self, action: DbaTunerAction) -> Dict[str, Any]:
        """Convert DbaTunerAction to JSON payload for step message."""
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
        }
        if action.table is not None:
            payload["table"] = action.table
        if action.column is not None:
            payload["column"] = action.column
        if action.index_name is not None:
            payload["index_name"] = action.index_name
        if action.new_sql is not None:
            payload["new_sql"] = action.new_sql
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DbaTunerObservation]:
        """Parse server response into StepResult[DbaTunerObservation]."""
        obs_data = payload.get("observation", {})
        observation = DbaTunerObservation(
            query_plan=obs_data.get("query_plan", ""),
            latency_ms=obs_data.get("latency_ms", 0.0),
            total_cost=obs_data.get("total_cost", 0.0),
            storage_used_mb=obs_data.get("storage_used_mb", 0.0),
            storage_remaining_mb=obs_data.get("storage_remaining_mb", 50.0),
            index_count=obs_data.get("index_count", 0),
            active_indexes=obs_data.get("active_indexes", []),
            is_correct=obs_data.get("is_correct", True),
            current_sql=obs_data.get("current_sql", ""),
            scenario_level=obs_data.get("scenario_level", 1),
            scenario_description=obs_data.get("scenario_description", ""),
            error_message=obs_data.get("error_message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
