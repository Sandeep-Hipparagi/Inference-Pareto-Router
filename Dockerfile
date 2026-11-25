# ==========================================
# Build Stage
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip build uv hatchling

# Copy project files
COPY pyproject.toml README.md LICENSE benchmark_results.json ./
COPY pareto_router ./pareto_router

# Build wheel
RUN python -m build --wheel --no-isolation

# ==========================================
# Runtime Stage
# ==========================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root user
RUN groupadd -r paretorouter && useradd -r -g paretorouter -d /app -s /sbin/nologin paretorouter && \
    chown -R paretorouter:paretorouter /app

# Copy wheel from builder and install
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Switch to non-root user
USER paretorouter

# OCI labels for GitHub Packages repository linking
LABEL org.opencontainers.image.source="https://github.com/Sandeep-Hipparagi/Inference-Pareto-Router" \
      org.opencontainers.image.description="Sub-millisecond 2D model-tier and reasoning-effort routing for LLM inference" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="Pareto Router" \
      org.opencontainers.image.version="0.1.2"

ENTRYPOINT ["pareto-router"]
CMD ["--help"]