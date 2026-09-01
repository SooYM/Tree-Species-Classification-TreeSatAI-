# Deployment & Release Manual

This document outlines best practices for containerizing, deploying, and releasing the PureForest Model Verification platform in production environments.

---

## 1. Production Architecture Overview

In production, the Python application server operates behind a reverse proxy (e.g., Nginx or Traefik) providing TLS termination, static asset caching, and request rate limiting.

```
[ Internet / Clients ]
          │
          ▼
   [ Nginx Reverse Proxy ] (Port 80 / 443 with HTTPS)
          │
          ▼
   [ PureForest Web Server ] (Port 8080 - Threaded Python API)
          │
     ┌────┴───────────────┐
     ▼                    ▼
[ PyTorch Engine ]  [ Local Datasets ]
```

---

## 2. Docker Containerization

### 2.1 Dockerfile Reference
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and static assets
COPY app.py .
COPY train_classifier.py .
COPY train_efficientnet.py .
COPY public/ public/
COPY Metadata/ Metadata/

# Expose HTTP port
EXPOSE 8080
ENV PORT=8080

# Launch server
CMD ["python3", "app.py"]
```

### 2.2 Building and Running the Image
```bash
# Build Docker image
docker build -t pureforest-verification:latest .

# Run container mounting model weights
docker run -d -p 8080:8080 \
  -v $(pwd)/efficientnet_v2_forest.pth:/app/efficientnet_v2_forest.pth \
  --name pureforest-app pureforest-verification:latest
```

---

## 3. Systemd Service Configuration (Linux Servers)

Create `/etc/systemd/system/pureforest.service`:

```ini
[Unit]
Description=PureForest Verification Web Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/pureforest
ExecStart=/opt/pureforest/venv/bin/python3 app.py
Restart=always
RestartSec=5
Environment=PORT=8080

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pureforest
sudo systemctl start pureforest
sudo systemctl status pureforest
```

---

## 4. Model Weights & Large File Management

> [!IMPORTANT]
> **Source Code & Documentation Only in Git**:
> Large model checkpoints (`checkpoint.pth` [243MB], `rf_model.joblib` [89MB], `efficientnet_v2_forest.pth` [81MB]) and multi-gigabyte aerial imagery datasets must **never** be committed directly into Git repositories.
>
> **Recommended Storage Strategies**:
> 1. **GitHub Releases**: Attach trained `.pth` and `.joblib` model binaries as release artifacts on tagged versions (e.g. `v1.0.0`).
> 2. **Cloud Storage**: Store datasets and weights in Amazon S3, Google Cloud Storage, or Hugging Face Hub, downloading them during CI/CD provisioning.

---

## 5. Release Workflow & Versioning

Follow Semantic Versioning (`MAJOR.MINOR.PATCH`):
1. **MAJOR**: Breaking API changes or architectural model redesigns.
2. **MINOR**: New dashboard features, analytics subtabs, or additional spectral indices.
3. **PATCH**: Bug fixes, performance enhancements, or documentation updates.
