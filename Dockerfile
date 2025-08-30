FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -e .

# Copy application code
COPY app ./app
COPY rl ./rl

# Copy policies if they exist (optional)
COPY policies ./policies 2>/dev/null || true

# Environment variables
ENV PORT=8080
ENV TRAIN_ENABLED=false

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]