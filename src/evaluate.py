import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib
from .embeddings import load_model
import face_recognition

def load_classifier_and_scaler(models_folder):
    classifier_path = os.path.join(models_folder, "svm_classifier.joblib")
    scaler_path = os.path.join(models_folder, "scaler.joblib")

    classifier = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)

    return classifier, scaler

def preprocess_image(image_path, target_size = (224, 224)):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        print(f"No face detected in {image_path}")
        return None
    
    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]
    face_image = Image.fromarray(face)
    face_image = face_image.resize(target_size, Image.LANCZOS)

    return face_image

def generate_embedding(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor).cpu().numpy()

    return embedding

def predict_identity(embedding, classifier, scaler):
    scaled_embedding = scaler.transform(embedding)
    identity = classifier.predict(scaled_embedding)
    return identity[0]

def evaluate_image(image_path, model, device, classifier, scaler):
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image is None:
        return None
    
    embedding = generate_embedding(model, preprocessed_image, device)
    predicted_identity = predict_identity(embedding, classifier, scaler)

    return predicted_identity

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_folder = os.path.join(project_root, "models")
    test_images_folder = os.path.join(project_root, "data", "test_images")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(device)
    classifier, scaler = load_classifier_and_scaler(models_folder)

    for image_name in os.listdir(test_images_folder):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(test_images_folder, image_name)
            predicted_identity = evaluate_image(image_path, model, device, classifier, scaler)

            if predicted_identity is not None:
                print(f"Predicted Identity for {image_name}: {predicted_identity}")
            else:
                print(f"No face detected in {image_name}")

    print("Evaluation done!")