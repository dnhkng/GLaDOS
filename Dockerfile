# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12-slim

# espeak-ng phonemizes non-English speech; PortAudio and libGL back sounddevice and OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
  libportaudio2 \
  libgl1 \
  libglib2.0-0 \
  espeak-ng \
  && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Create a non-root user for security
RUN useradd -m -u 1000 glados

WORKDIR /app

# Set ownership of /app so glados can create .venv
RUN chown glados:glados /app

USER glados

# Dependencies first, from the lockfile, so source edits do not reinstall them.
COPY --chown=glados:glados pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra api --extra cpu --no-dev --no-install-project

COPY --chown=glados:glados models/ ./models/
COPY --chown=glados:glados src/ ./src/
RUN uv sync --locked --extra api --extra cpu --no-dev

# Keep downloaded models in a BuildKit cache so rebuilding after a code change does not fetch 2 GB again.
RUN --mount=type=cache,target=/home/glados/.cache/glados-models,uid=1000,gid=1000 \
  cp -a /home/glados/.cache/glados-models/. models/ \
  && uv run --no-sync glados download \
  && cp -a models/. /home/glados/.cache/glados-models/

EXPOSE 5050
CMD ["uv", "run", "--no-sync", "litestar", "--app", "glados.api.app:create_app", "run", "--host", "0.0.0.0", "--port", "5050"]
