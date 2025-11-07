# Dockerfile for Aegis IDS

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY model.py .
COPY utils.py .
COPY train_model.py .
COPY streamlit_app.py .
COPY setup.py .
COPY README.md .
COPY .gitignore .

# Create necessary directories
RUN mkdir -p data models logs

# Expose Streamlit port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "streamlit_app.py"]

# Build: docker build -t aegis-ids:latest .
# Run: docker run -p 8501:8501 -v $(pwd)/data:/app/data aegis-ids:latest
