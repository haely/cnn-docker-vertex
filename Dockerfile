# Use an official PyTorch base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Install Vault CLI (to confirm)
RUN apt-get update && apt-get install -y curl unzip && \
    curl -o vault.zip https://releases.hashicorp.com/vault/1.12.0/vault_1.12.0_linux_amd64.zip && \
    unzip vault.zip && mv vault /usr/local/bin/ && \
    rm vault.zip


# Expose a port if needed (e.g., for TensorBoard or Flask API)
# EXPOSE 6006

# Set the default command to run the training script
CMD ["python", "src/train.py"]

