#!/bin/bash

# Codespaces setup script for Google Cloud deployment
# This installs gcloud CLI and sets up the deployment environment

set -e

echo "🚀 Setting up development environment for Tetris AI deployment..."

# Install Google Cloud CLI
echo "1️⃣ Installing Google Cloud CLI..."
if ! command -v gcloud &> /dev/null; then
    echo "📦 Downloading and installing gcloud CLI..."
    curl -sSL https://sdk.cloud.google.com | bash
    source ~/.bashrc
    echo "export PATH=\$PATH:\$HOME/google-cloud-sdk/bin" >> ~/.bashrc
    export PATH=$PATH:$HOME/google-cloud-sdk/bin
else
    echo "✅ gcloud CLI already installed"
fi

# Install Python dependencies
echo "2️⃣ Installing Python dependencies..."
pip install -r requirements.txt

# Install Docker (if not available)
echo "3️⃣ Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "📦 Docker not found, but should be available in Codespaces"
else
    echo "✅ Docker is available"
fi

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Authenticate with Google Cloud:"
echo "   source ~/.bashrc  # Reload to get gcloud in PATH"
echo "   gcloud auth login"
echo ""
echo "2. Run the GCP deployment setup:"
echo "   ./setup-gcp-deployment.sh"
echo ""
echo "3. Or deploy manually for testing:"
echo "   gcloud run deploy rl-tetris --source . --region northamerica-northeast1 --allow-unauthenticated"
