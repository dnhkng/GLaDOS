FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
  libportaudio2 \
  portaudio19-dev \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Create a non-root user for security
RUN useradd -m -u 1000 glados

WORKDIR /app

# Set ownership of /app so glados can create .venv
RUN chown glados:glados /app

COPY --chown=glados:glados pyproject.toml README.md ./
COPY --chown=glados:glados models/ ./models/
COPY --chown=glados:glados src/ ./src/

USER glados

RUN uv sync --extra api --extra cpu --no-dev \
  && uv run glados download

EXPOSE 5050
CMD ["uv", "run", "litestar", "--app", "glados.api.app:app", "run", "--host", "0.0.0.0", "--port", "5050"]
