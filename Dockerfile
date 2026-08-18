# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS frontend
WORKDIR /build/ui
COPY src/microgrid_simulator/ui/web/package*.json ./
RUN npm ci
COPY src/microgrid_simulator/ui/web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/app/.venv/bin:$PATH

RUN useradd --create-home --uid 10001 simulator && mkdir /app && chown simulator:simulator /app
USER simulator
WORKDIR /app

COPY --chown=simulator:simulator pyproject.toml README.md ./
COPY --chown=simulator:simulator src/ ./src/
COPY --chown=simulator:simulator configs/ ./configs/
COPY --chown=simulator:simulator data/forecasts.jsonl data/forecast_manifest.json ./data/
COPY --chown=simulator:simulator --from=frontend /build/ui/dist/ ./src/microgrid_simulator/ui/web/dist/

# Shared services use only the active pandapower/dashboard path. The EMS-only
# cpu-rl target below adds inference dependencies without making telemetry,
# simulator, or dashboard carry Torch in memory or on disk.
RUN uv venv && uv pip install \
      "pandapower>=2.14" \
      "gymnasium>=0.29" \
      "pydantic>=2.6" \
      "pydantic-settings>=2.2" \
      "pyyaml>=6.0" \
      "typer>=0.12" \
      "openpyxl>=3.1" \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.29" \
      "pypsa>=0.35.2" \
      "python-multipart>=0.0.9" && \
    uv pip install "nats-py>=2.9,<3" && \
    uv pip install --no-deps .

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/api/defaults', timeout=4)"]

ENTRYPOINT ["microgrid-sim"]
CMD ["dashboard", "--host", "0.0.0.0", "--port", "8501"]

FROM runtime AS cpu-rl

# Install the CPU wheel explicitly before SB3 so Docker never resolves a CUDA
# training stack. This image performs inference only; training stays host-side.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3" && \
    uv pip install "stable-baselines3>=2.3"
