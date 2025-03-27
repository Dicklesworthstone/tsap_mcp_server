# Multi-stage build for TSAP MCP Server
# Stage 1: Build dependencies
FROM python:3.13-slim AS builder

# Set working directory
WORKDIR /app

# Install build essentials and dependency tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ripgrep \
    jq \
    gawk \
    sqlite3 \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -sSf https://astral.sh/uv/install.sh | sh

# Copy pyproject.toml and other files needed for installation
COPY pyproject.toml README.md ./
COPY src ./src/

# Create venv and install dependencies with uv
ENV PATH="/root/.cargo/bin:${PATH}"
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
RUN uv pip install --no-cache-dir -e ".[performance]"

# Stage 2: Runtime image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    TSAP_ENV="production"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ripgrep \
    jq \
    gawk \
    sqlite3 \
    libxml2 \
    libxslt1.1 \
    libpoppler-cpp-dev \
    poppler-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 tsap

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=tsap:tsap . /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/cache \
    && chown -R tsap:tsap /app/data /app/logs /app/cache

# Switch to non-root user
USER tsap

# Expose ports
EXPOSE 8000

# Create volume mount points
VOLUME ["/app/data", "/app/logs", "/app/cache"]

# Set up entrypoint and default command
ENTRYPOINT ["python", "-m", "tsap"]
CMD ["server", "--host", "0.0.0.0", "--port", "8000"]