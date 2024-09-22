# Facial Recognition Project

This project implements a facial recognition system using deep learning (Torch, Resnet50) and machine learning (SVM). It processes the Labeled Faces in the Wild (LFW) dataset, generates face embeddings using a pre-trained ResNet50 model, and trains an SVM classifier for face recognition.

## Project Structure

facial_recognition_project/  
│  
├── data/  
│ ├── raw/  
│ ├── processed/  
│ └── embeddings/  
│  
├── src/  
| |── __init__.py
│ ├── config.py  
│ ├── preprocess.py  
│ ├── dataset.py  
| |── download_dataset.py
│ ├── model.py  
│ ├── embeddings.py  
│ ├── train.py  
│ └── evaluate.py   
│  
|── main.py
├── requirements.txt  
├── README.md  
└── .gitignore  


## Setup

1. Clone the repository:

git clone https://github.com/yourusername/facial_recognition_project.git  
cd facial_recognition_project

2. Create a virtual environment and activate it:  

conda activate env  

3. Install the required dependencies:  

pip install -r requirements.txt OR Conda install -r requirements.txt  

## Usage

1. Download and prepare the dataset:

python scripts/download_dataset.py  

2. Run the entire pipeline:  

python scripts/run_pipeline.py

This script will preprocess the images, generate embeddings, train the SVM classifier, and evaluate the model.

3. To evaluate the model on new images, use:  

python src/evaluate.py  

## Components

- `preprocess.py`: Handles face detection, alignment, and cropping.
- `dataset.py`: Implements a custom PyTorch Dataset for the face images.
- `model.py`: Defines the face embedding model (ResNet50).
- `embeddings.py`: Generates face embeddings using the pre-trained model.
- `train.py`: Trains the SVM classifier on the generated embeddings.
- `evaluate.py`: Evaluates the trained model on test images.

## Testing

Run the unit tests using:  

python -m unittest discover tests  

## Logging and Visualization

This project uses TensorBoard for logging and visualization. To view the logs, run:  

tensorboard --logdir=logs  

Then open a web browser and go to `http://localhost:6006`.

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Labeled Faces in the Wild dataset
- PyTorch and TorchVision libraries
- scikit-learn library
- face_recognition library

