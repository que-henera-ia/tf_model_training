#!/bin/bash

# Default value for build argument
BUILD=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build) BUILD=true ;;
        --no-build) BUILD=false ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Perform actions based on the value of BUILD
if [ "$BUILD" = true ]; then
    echo "Build is set to true. Building Docker Containers..."
    # Place your build actions here
    sudo docker build -t my_tensorflow_image .
else
    echo "Build is set to false. Not building Docker Containers..."
    # Place your non-build actions here
fi
sudo docker run --gpus all -v $(pwd):/app -v $(pwd)/main.py:/app/main.py my_tensorflow_image