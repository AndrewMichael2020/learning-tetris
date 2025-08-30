# 🚀 Google Cloud Run Deployment Guide

## Quick Setup

### For Codespaces/Development Environment:

1. **Setup development environment** (installs gcloud CLI):
   ```bash
   ./setup-codespaces.sh
   source ~/.bashrc  # Reload shell to get gcloud in PATH
   ```

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   ```

3. **Run the GCP setup script**:
   ```bash
   ./setup-gcp-deployment.sh
   ```

4. **Add GitHub Secrets** (from the script output):
   - Go to your GitHub repo: Settings → Secrets and variables → Actions
   - Add the two secrets shown in the script output

5. **Deploy**: Push to `main` branch or trigger workflow manually

## Manual Deployment (Testing)

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project hospigen

# Deploy directly from source
gcloud run deploy rl-tetris \
  --source . \
  --region northamerica-northeast1 \
  --allow-unauthenticated \
  --set-env-vars "TRAIN_ENABLED=false"
```

## Configuration

- **Project**: `hospigen`
- **Region**: `northamerica-northeast1` (Montreal)
- **Service**: `rl-tetris`
- **Training**: Disabled in production (memory optimization)

## Environment Variables

- `PORT`: 8080 (Cloud Run standard)
- `TRAIN_ENABLED`: false (reduces memory usage)

## Resource Limits

- **Memory**: 1GB
- **CPU**: 1 vCPU
- **Concurrency**: 80 requests
- **Timeout**: 300 seconds
- **Max Instances**: 10

Your deployed app will be available at: `https://rl-tetris-[hash]-nn.a.run.app`
