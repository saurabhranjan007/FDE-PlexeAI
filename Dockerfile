# Slim Python image for FDE assignment API
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (exclude notebooks/data if large; add .dockerignore as needed)
COPY src/ src/
COPY experiments/ experiments/

# Serve on port 8000
EXPOSE 8000

# Run FastAPI with uvicorn (run from project root: docker run -p 8000:8000 <image>)
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
