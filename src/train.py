import os
import torch
import yaml
from model.model import FullModel
from model.utils import fetch_gcp_credentials_from_vault, write_gcp_credentials
from dataset import CustomDataset
from torch.utils.data import DataLoader

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Fetch GCP credentials from Vault
    vault_url = os.getenv("VAULT_URL")
    vault_token = os.getenv("VAULT_TOKEN")
    vault_secret_path = os.getenv("VAULT_SECRET_PATH")
    
    credentials = fetch_gcp_credentials_from_vault(vault_url, vault_token, vault_secret_path)
    write_gcp_credentials(credentials)

    # Initialize the model
    model = FullModel(num_classes=config['model']['num_classes'], num_keypoints=config['model']['num_keypoints'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_dataset = CustomDataset(config['data']['train_data_gcs'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            
            # Forward pass
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {loss.item()}")

if __name__ == "__main__":
    main()

