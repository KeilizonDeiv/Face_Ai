FROM python:3.12-slim

# opencv needs these system libs even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python download_models.py

# Bind to all interfaces inside the container; expose the port to the host
# via `docker run -p`, and terminate TLS at a reverse proxy in front of it.
ENV FACE_AI_HOST=0.0.0.0
ENV FACE_AI_PORT=5000
EXPOSE 5000

CMD ["python", "serve.py"]
