import os
import argparse
from src import preprocess, embeddings, train, evaluate, download_dataset

def main(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Download and organize dataset
    if args.download:
        raw_folder = os.path.join(project_root, 'data', 'raw')
        processed_folder = os.path.join(project_root, 'data', 'processed', 'aligned_faces')
        url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
        
        download_dataset.download_dataset(url, raw_folder)
        download_dataset.organize_dataset(raw_folder, processed_folder)
    
    # Step 2: Preprocess images
    if args.preprocess:
        input_folder = os.path.join(project_root, 'data', 'processed', 'aligned_faces')
        output_folder = os.path.join(project_root, 'data', 'processed', 'preprocessed_faces')
        preprocess.preprocess_dataset(input_folder, output_folder)
    
    # Step 3: Generate embeddings
    if args.embeddings:
        input_folder = os.path.join(project_root, 'data', 'processed', 'preprocessed_faces')
        output_file = os.path.join(project_root, 'data', 'embeddings', 'embeddings.npz')
        embeddings.generate_and_save_embeddings(input_folder, output_file)
    
    # Step 4: Train classifier
    if args.train:
        embeddings_file = os.path.join(project_root, 'data', 'embeddings', 'embeddings.npz')
        models_folder = os.path.join(project_root, 'models')
        train.train_and_save_classifier(embeddings_file, models_folder)
    
    # Step 5: Evaluate model
    if args.evaluate:
        models_folder = os.path.join(project_root, 'models')
        test_images_folder = os.path.join(project_root, 'data', 'test_images')
        evaluate.evaluate_model(models_folder, test_images_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Recognition Pipeline")
    parser.add_argument("--download", action="store_true", help="Download and organize the dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the images")
    parser.add_argument("--embeddings", action="store_true", help="Generate face embeddings")
    parser.add_argument("--train", action="store_true", help="Train the SVM classifier")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model on test images")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    
    args = parser.parse_args()
    
    if args.all:
        args.download = args.preprocess = args.embeddings = args.train = args.evaluate = True
    
    main(args)