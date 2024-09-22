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
│ ├── config.py  
│ ├── preprocess.py  
│ ├── dataset.py  
│ ├── model.py  
│ ├── embeddings.py  
│ ├── train.py  
│ └── evaluate.py  
│  
├── scripts/  
│ ├── download_dataset.py  
│ └── run_pipeline.py  
│  
├── tests/  
│ ├── test_preprocess.py  
│ ├── test_dataset.py  
│ └── test_model.py  
│  
├── requirements.txt  
├── README.md  
└── .gitignore  

