import os
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for person_id, person_name in enumerate(os.listdir(root_dir)):
            person_dir = os.path.join(root_dir, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(person_dir, image_name))
                        self.labels.append(person_id)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path