#!/bin/bash

# Setup script for Google Cloud Run deployment
# Run this script after authenticating with: gcloud auth login

set -e

echo "🚀 Setting up Google Cloud deployment for Tetris AI..."

# Configuration
PROJECT_ID="hospigen"
REGION="northamerica-northeast1"
SERVICE_NAME="rl-tetris"
REPO_NAME="AndrewMichael2020/learning-tetris"

echo "📋 Project: $PROJECT_ID"
echo "📍 Region: $REGION"
echo "🎯 Service: $SERVICE_NAME"
echo "📦 Repository: $REPO_NAME"
echo ""

# Set project
echo "1️⃣ Setting GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "2️⃣ Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable iamcredentials.googleapis.com

# Create service account
echo "3️⃣ Creating service account for GitHub Actions..."
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Service Account" \
    --description="Service account for GitHub Actions deployments" || echo "Service account already exists"

# Grant necessary roles
echo "4️⃣ Granting IAM roles..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudbuild.builds.builder"

# Create Workload Identity Pool
echo "5️⃣ Creating Workload Identity Pool..."
gcloud iam workload-identity-pools create "github-pool" \
    --location="global" \
    --display-name="GitHub Actions Pool" \
    --description="Pool for GitHub Actions authentication" || echo "Pool already exists"

# Create Workload Identity Provider
echo "6️⃣ Creating Workload Identity Provider..."
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
    --location="global" \
    --workload-identity-pool="github-pool" \
    --display-name="GitHub Actions Provider" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
    --issuer-uri="https://token.actions.githubusercontent.com" || echo "Provider already exists"

# Get project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Allow GitHub Actions to impersonate the service account
echo "7️⃣ Setting up Workload Identity..."
gcloud iam service-accounts add-iam-policy-binding \
    github-actions@$PROJECT_ID.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/$REPO_NAME"

echo ""
echo "✅ Setup complete! Now add these secrets to your GitHub repository:"
echo ""
echo "🔑 GitHub Repository Secrets (Settings → Secrets and variables → Actions):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Secret Name: WORKLOAD_IDENTITY_PROVIDER"
echo "Secret Value: projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
echo ""
echo "Secret Name: SERVICE_ACCOUNT_EMAIL"
echo "Secret Value: github-actions@$PROJECT_ID.iam.gserviceaccount.com"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 After adding the secrets, push to main branch to trigger deployment!"
echo "📱 Your app will be available at: https://rl-tetris-[hash]-nn.a.run.app"
echo ""
echo "🔧 To deploy manually for testing:"
echo "   gcloud run deploy rl-tetris --source . --region $REGION --allow-unauthenticated"
