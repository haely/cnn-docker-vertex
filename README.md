# CNN Docker Vertex
This repository contains the code and setup for a deep learning-based image segmentation project, built using PyTorch and Docker. The project includes data downloading, model training, inference, and utilities for handling datasets and models.
To Do - Fix GCP auth access + ci/cd with Vault

## Project Structure

```
cnn-docker-vertex/
│
├── Dockerfile                # Docker instructions for building the image
├── .dockerignore             # Ignore files you don’t want copied to the container
├── requirements.txt          # Python dependencies for your project
├── config.yaml               # Configuration file for hyperparameters, paths, etc.
│
├── data/
│   └── download_data.py      # Script to download dataset from GCS
│
├── src/
│   ├── __init__.py           # Marks directory as a Python package
│   ├── train.py              # Training script
│   ├── dataset.py            # Dataset loader (e.g., for GCS or local)
│   ├── inference.py          # Script for performing inference on images
│   ├── utils.py              # Utility functions (e.g., for data pre-processing)
│   └── model/                # Separate folder for model components
│       ├── __init__.py       # Marks model directory as a Python package
│       ├── backbone.py       # Defines the backbone (e.g., ResNet, VGG, etc.)
│       ├── module.py         # Specific modules like attention blocks, etc.
│       ├── keypoint.py       # Keypoint detection implementation
│       ├── segmentation.py   # Image segmentation logic (e.g., Unet, Mask R-CNN, etc.)
│       └── model.py          # Combines backbone, modules, keypoint detection, and segmentation
│
├── notebooks/
│   ├── exploration.ipynb     # Jupyter notebooks for data exploration and testing
│
├── scripts/
│   ├── run_training.sh       # Shell script to run training with Docker
│   ├── push_image.sh         # Shell script to push the Docker image to GCR
│
├── outputs/
│   ├── checkpoints/          # Model checkpoints during training
│   └── logs/                 # Training logs (TensorBoard or plain text)
│
└── tests/
    ├── test_backbone.py      # Unit tests for backbone
    ├── test_module.py        # Unit tests for modules
    ├── test_keypoint.py      # Unit tests for keypoint detection
    └── test_model.py         # Unit tests for the final model
```

## Setup

### Requirements
* Docker
* Python 3.12
* PyTorch (specified in `requirements.txt`)

### Install Dependencies
1. Clone the repository:
```
git clone <add link>
cd automation/cd-docker-vertex/
```
2. Install Python dependencies:
```
pip install -r requirements.txt
```
3. Configure the project using the config.yaml file. Adjust paths, hyperparameters, and any other necessary settings.

### Docker Setup
1. Build the Docker image:
```docker build -t cnn-docker-vertex . ```

2. Push the Docker image to Google Container Registry (GCR):
``` ./scripts/push_image.sh```

## Dataset
To download the dataset from GCS, run the following:
```
python data/download_data.py
```

## Training
You can run training either locally or using Docker.

### Locally:
```
python src/train.py --config config.yaml
```

### With Docker:
```
./scripts/run_training.sh
```

## Inference
Run inference on new images using the inference.py script:
```
python src/inference.py --config config.yaml --image_path path/to/image
```

## Outputs
* Model checkpoints will be saved in `outputs/checkpoints/`
* Training logs (e.g., TensorBoard logs) will be in `outputs/logs/`

## Testing
To run unit tests for different components, execute:
```
pytest tests/
```

## Notebooks
Explore the dataset and experiment with the model using the Jupyter notebooks in the `notebooks/` directory:
```
jupyter notebook notebooks/exploration.ipynb
```

## Contributing
Feel free to open issues or submit pull requests to improve the project.
