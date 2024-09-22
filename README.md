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

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/facial_recognition_project.git  
    cd facial_recognition_project
    ```

2. **Create a virtual environment and activate it**:  
    ```bash
    conda create -n env_name
    conda activate env_name  
    ```

3. **Install the required dependencies**:  
    ```bash
    pip install -r requirements.txt   
    ```
    or
    ```bash
    Conda install -r requirements.txt
    ```

## Usage

The project includes a `main.py` file that allows you to run the entire pipeline or specific parts of it using command-line arguments.

1. **To run the entire pipeline**:
    ```bash
    python main.py --all
    ```
    This command will download the dataset, preprocess the images, generate embeddings, train the SVM classifier, and evaluate the model.

2. **To run specific parts of the pipeline, you can use the following arguments**:

   - Download and prepare the dataset:
     ```bash
     python main.py --download
     ```

   - Preprocess the images:
     ```bash
     python main.py --preprocess
     ```

   - Generate embeddings:
     ```bash
     python main.py --embeddings
     ```

   - Train the SVM classifier:
     ```bash
     python main.py --train
     ```

   - Evaluate the model on test images:
     ```bash
     python main.py --evaluate
     ```

3. **You can also combine multiple steps**:
    ```bash
    python main.py --download --preprocess --embeddings
    ```

4. **For help and to see all available options**:
    ```bash
    python main.py --help
    ```

**Note**: Make sure you're in the project's root directory when running these commands.

## Components

- `download_dataset.py`: Downloads and organizes the dataset.
- `preprocess.py`: Handles face detection, alignment, and cropping.
- `dataset.py`: Implements a custom PyTorch Dataset for the face images.
- `model.py`: Defines the face embedding model (ResNet50).
- `embeddings.py`: Generates face embeddings using the pre-trained model.
- `train.py`: Trains the SVM classifier on the generated embeddings.
- `evaluate.py`: Evaluates the trained model on test images.

## Logging and Visualization

This project uses TensorBoard for logging and visualization. To view the logs, run:  
    ```bash
    tensorboard --logdir=logs
    ```  

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

- Building a Facial Recognition Pipeline with Deep Learning in Tensorflow [LINK](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8)
- Labeled Faces in the Wild dataset

