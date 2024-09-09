# if needed, add paths to
import torch
from model.model import FullModel
from dataset import CustomDataset
from torchvision import transforms
from PIL import Image

def load_model(checkpoint_path):
    model = FullModel(num_classes=21, num_keypoints=17)  # Update with actual parameters
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def infer_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs, keypoints = model(img)
    return outputs, keypoints

if __name__ == "__main__":
    model = load_model("path/to/checkpoint.pth")
    outputs, keypoints = infer_image("path/to/image.jpg", model)
    print(outputs)
    print(keypoints)

