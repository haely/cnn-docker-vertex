#!/bin/bash

# Ensure Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Environment variables
export VAULT_URL="http://your-vault-url"
export VAULT_TOKEN="your-vault-token"
export VAULT_SECRET_PATH="secret/data/gcp/credentials"

# Fetch GCP credentials from Vault and save to a temporary file
curl --header "X-Vault-Token: $VAULT_TOKEN" \
     $VAULT_URL/v1/$VAULT_SECRET_PATH \
     | jq -r '.data' > /tmp/gcs_credentials.json

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcs_credentials.json

# Docker image name
DOCKER_IMAGE="gcr.io/your-project-id/my-docker-image:latest"

# Run Docker container
docker run \
    --rm \
    -v $(pwd):/app \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcs_credentials.json \
    $DOCKER_IMAGE

echo "Training complete."

# haely todo add vaultpath
