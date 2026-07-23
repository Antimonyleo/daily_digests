# DailyDigest — self-contained local web app.
# Small image: the default embedder is fastembed (ONNX, CPU), so no PyTorch.
FROM python:3.12-slim

# uv: fast, reproducible installs from the committed uv.lock.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install dependencies first (cached layer) from the lockfile only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App source.
COPY . .
RUN uv sync --frozen --no-dev

# Bake the embedding model into the image so the first brew is instant and works
# fully offline (no HuggingFace download at runtime).
RUN uv run python -c "from dailydigest.rank.embed import embed_texts; embed_texts(['warm up the embedding model'])"

# All persistent state (SQLite DB, profile.yaml, learned models, calibrator)
# lives under ./data. docker-compose maps a host folder here so it survives
# container rebuilds.
EXPOSE 8765
VOLUME ["/app/data"]

# --no-browser: the container cannot open the host browser; the user opens the
# mapped port themselves. --host 0.0.0.0 so the mapped port is reachable.
CMD ["uv", "run", "dd", "start", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
