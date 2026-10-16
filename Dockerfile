# 1. Use an official lightweight Python base image
FROM python:3.13.7-slim

# Set environment variables to optimize Python for containerized environments
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 2. Set working directory inside the container
WORKDIR /app

# Install system-level dependencies required for building C extensions or ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy requirements file and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy application source code, configuration files, and saved model artifacts
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY model/ ./model/

# 5. Expose server port 8000 for FastAPI
EXPOSE 8000

# 6. Define container startup entrypoint using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]