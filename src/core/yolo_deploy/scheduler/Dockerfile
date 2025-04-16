# Use an official Python runtime as a parent image
# Using the 'slim' variant for a smaller image size
FROM python:3.11-slim

# Create a non-root user "appuser"
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker build cache
COPY Src/app/yolo_deploy/scheduler/requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
# --upgrade pip ensures the latest pip is used
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the custom module directory 'Src' into the container's working directory
# Ensure the 'Src' directory exists in the same context as the Dockerfile
COPY Src/ ./Src/

# Copy the main application script into the container's working directory
COPY Src/app/yolo_deploy/scheduler/scheduler.py .

# Create directories for input and output within the container image
# Note: It's generally better practice to mount volumes for input/output
# when running the container, but this ensures the directories exist.
RUN mkdir input output

USER appuser

# Command to run the application when the container launches
# The script will start the scheduler and run indefinitely (or until stopped)
CMD ["python", "scheduler.py"]
