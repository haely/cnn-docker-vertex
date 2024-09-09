import torch
from torchvision import transforms
from PIL import Image
import gcsfs

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, gcs_path):
        # Initialize the dataset (e.g., load file paths from GCS)
        self.gcs_path = gcs_path
        self.fs = gcsfs.GCSFileSystem()
        self.file_list = self._load_file_list()

    def _load_file_list(self):
        # Load file paths from GCS
        # Placeholder logic
        return ["path/to/file1.jpg", "path/to/file2.jpg"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # Read image from GCS
        with self.fs.open(file_path, 'rb') as f:
            img = Image.open(f)
            img = transforms.ToTensor()(img)
        # Return image and label (placeholder label)
        return img, torch.tensor(0)  # Replace with actual label loading logic
# haely todo - update file path
