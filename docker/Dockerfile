# Dockerfile for running the ocr service
FROM python:3.11.5-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY transcription/requirements-doc-int.txt transcription/
RUN pip install --no-cache-dir -r transcription/requirements-doc-int.txt
COPY backend/requirements-backend.txt backend/
RUN pip install --no-cache-dir -r backend/requirements-backend.txt

# Copy code into the container
COPY . .

# Expose port 8080 for FastAPI
EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]