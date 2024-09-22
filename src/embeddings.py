import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from .dataset import FaceDataset
from .model import load_model

def generate_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    image_paths = []

    with torch.no_grad():
        for batch_images, batch_labels, batch_paths in tqdm(dataloader, desc="Generating Embeddings"):
            batch_images = batch_images.to(device)
            batch_embeddings = model(batch_images)
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.extend(batch_labels.numpy())
            image_paths.extend(batch_paths)

    embeddings = np.vstack(embeddings)
    return embeddings, labels, image_paths

def save_embeddings(embeddings, labels, image_paths, output_file):
    np.savez(output_file, embeddings=embeddings, labels=labels, image_paths=image_paths)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(project_root, "data", "processed", "preprocessed_faces")
    output_file = os.path.join(project_root, "data", "embeddings", "embeddings.npz")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FaceDataset(input_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = False, num_workers = 4)

    print("Loading model...")
    model = load_model(device)

    print("Generating embeddings...")
    embeddings, labels, image_paths = generate_embeddings(model, dataloader, device)

    print("Saving embeddings...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_embeddings(embeddings, labels, image_paths, output_file)

    print("Embeddings generated and saved successfully")