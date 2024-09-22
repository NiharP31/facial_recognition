import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_embeddings(embeddings_file):
    data = np.load(embeddings_file)
    labels = data['labels']
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    return embeddings, labels, image_paths

def train_svm_classifier(X_train, y_train, writer):
    print("Training SVM Classifier...")
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train, y_train)

    train_accuracy = svm.score(X_train, y_train)
    writer.add_scalar("Accuracy/Train", train_accuracy, 0)

    return svm

def evaluate_classifier(classifier, X_test, y_test, writer):
    print("Evaluating Classifier...")
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    writer.add_scalar("Accuracy/Test", accuracy, 0)

    report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                writer.add_scalar(f"Metrics/{label}/{metric}", value, 0)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    writer.add_figure("Confusion Matrix", fig, 0)

def save_classifier(classifier, scaler, output_folder):
    os.makedirs(output_folder, exist_ok = True)
    classifier_file = os.path.join(output_folder, "svm_classifier.joblib")
    scaler_path = os.path.join(output_folder, "scaler.joblib")

    joblib.dump(classifier, classifier_file)
    joblib.dump(scaler, scaler_path)
    print(f"Classifier saved to {classifier_file}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_file = os.path.join(project_root, "data", "embeddings", "embeddings.npz")
    output_folder = os.path.join(project_root, "models")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(project_root, "logs", current_time)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    print("Loading embeddings...")
    embeddings, labels, image_paths = load_embeddings(embeddings_file)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_classifier = train_svm_classifier(X_train_scaled, y_train, writer)

    evaluate_classifier(svm_classifier, X_test_scaled, y_test, writer)

    save_classifier(svm_classifier, scaler, output_folder)

    print("Training completed.")