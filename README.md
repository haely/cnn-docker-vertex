# cnn-docker-vertex


## Build the Docker image
docker build -t gcr.io/your-project/imagenet-training .

## Push the docker image to GCR
gcloud auth configure-docker
docker push gcr.io/your-project/imagenet-training

## Run this from Vertex AI
'''
from google.cloud import aiplatform

aiplatform.CustomJob.from_local_script(
    display_name="imagenet-training",
    script_path="main.py",
    container_uri="gcr.io/your-project/imagenet-training",
    args=[
        "--arch", "resnet18", "--train", "--data", "/gcs/path-to-data", "--epochs", "10"
    ],
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=1,
).run()
'''
