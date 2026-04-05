"""
FastAPI application for the DBA Tuner Env Environment.

This module creates an HTTP server that exposes the DbaTunerEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DbaTunerAction, DbaTunerObservation
    from .dba_tuner_env_environment import DbaTunerEnvironment
except (ImportError, SystemError):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import DbaTunerAction, DbaTunerObservation
    from server.dba_tuner_env_environment import DbaTunerEnvironment

import gradio as gr
import os

def my_gradio_builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    with gr.Blocks() as custom_blocks:
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        if os.path.exists(readme_path):
            gr.Markdown(open(readme_path, encoding='utf-8').read())
        else:
            gr.Markdown("# README.md not found")
    return custom_blocks

# Create the app with web interface and README integration
app = create_app(
    DbaTunerEnvironment,
    DbaTunerAction,
    DbaTunerObservation,
    env_name="dba_tuner_env",
    max_concurrent_envs=4,  # allow concurrent WebSocket sessions
    gradio_builder=my_gradio_builder,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
