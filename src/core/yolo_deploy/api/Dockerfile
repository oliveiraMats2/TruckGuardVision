# Use Python 3.9 as base image
FROM python:3.9-slim

# Create a non-root user to run the application
RUN groupadd -r user && useradd -r -g user user

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY Src/app/yolo_deploy/api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY Src/app/yolo_deploy/api/main.py .

# Expose the port for the FastAPI application
EXPOSE 8000

# Set environment variables with defaults that can be overridden
# ENV TRITON_URL="localhost:8001" \
#     MODEL_WIDTH=640 \
#     MODEL_HEIGHT=640 \
#     MODEL_NAME="yolov7"

USER user
# Command to run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]