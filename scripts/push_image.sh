#!/bin/bash

# Ensure Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Google Cloud authentication
echo "Authenticating with Google Cloud..."
echo "$GCP_SERVICE_ACCOUNT_KEY" | docker login -u _json_key --password-stdin https://gcr.io

# Docker image name
DOCKER_IMAGE="my-docker-image"
DOCKER_TAG="latest"
GCR_IMAGE="gcr.io/your-project-id/$DOCKER_IMAGE:$DOCKER_TAG"

# Build Docker image
docker build -t $DOCKER_IMAGE:$DOCKER_TAG .

# Tag Docker image
docker tag $DOCKER_IMAGE:$DOCKER_TAG $GCR_IMAGE

# Push Docker image to GCR
docker push $GCR_IMAGE

echo "Docker image pushed to $GCR_IMAGE"

# haely todo confirm
