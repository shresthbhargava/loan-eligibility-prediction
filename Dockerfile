# Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Builds the FastAPI backend into a portable container.
#
# WHY Docker?
# "Works on my machine" is not a deployment strategy.
# Docker packages your code, dependencies, and runtime into one artifact
# that runs identically on your laptop, Render, AWS, or anywhere else.
#
# Multi-stage thinking (simplified here for portfolio use):
#   Base image  → Python 3.11 slim (smaller than full Python image)
#   Copy src/   → model code and utilities
#   Copy api/   → FastAPI application
#   Copy models/ → trained model artifact
#   EXPOSE 8000 → documents which port the app listens on
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# If requirements don't change, pip install is skipped on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/     ./src/
COPY api/     ./api/
COPY models/  ./models/
COPY data/raw/ ./data/raw/

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Start command
# --host 0.0.0.0 required — without it, container only listens internally
# --workers 1 for free tier (limited RAM)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]