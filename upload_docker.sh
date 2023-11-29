#!/bin/bash

# Set your Docker Hub username
DOCKER_HUB_USERNAME="omarkhalil10"

# Set the version/tag for your container
VERSION="v1"

# Build the Docker image
docker build -t $DOCKER_HUB_USERNAME/camil:$VERSION -f Dockerfile .

# Log in to Docker Hub (you will be prompted for your Docker Hub credentials)
docker login

# Push the Docker image to Docker Hub
docker push $DOCKER_HUB_USERNAME/camil:$VERSION
