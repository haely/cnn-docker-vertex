# Use the official PyTorch base image with CUDA support
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set a working directory inside the container
WORKDIR /app

# Copy your local files to the working directory in the container
COPY . /app

# Install additional dependencies (if needed)
RUN pip install --no-cache-dir torchvision Pillow numpy

# Specify the command to run your training script
CMD ["python", "main.py", "--arch", "resnet18", "--train", "--data", "/input", "--epochs", "10"]
