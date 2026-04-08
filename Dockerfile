# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build arguments
ARG BUILD_MODE=in-repo
ARG ENV_NAME=dba_tuner_env

# Copy environment code (exclude list handled by .dockerignore)
COPY . /app/env
WORKDIR /app/env

# Ensure uv is installed
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
# Sync dependencies (creates /app/env/.venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app
COPY --from=builder /app/env /app/env

# Environment configuration
ENV ENABLE_WEB_INTERFACE=true
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose port for documentation and cloud platform compatibility
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
