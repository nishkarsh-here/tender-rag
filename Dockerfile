# For hosts that take a container: Google Cloud Run, Fly.io, Koyeb, Railway.
# Render reads render.yaml instead and does not need this file.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The vector store is committed, so the container does not re-index on boot.
ENV PORT=8080
EXPOSE 8080
CMD ["python", "app.py"]
