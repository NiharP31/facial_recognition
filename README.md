facial_recognition_project/
│
├── data/
│   ├── raw/
│   │   └── lfw/  # Original LFW dataset
│   ├── processed/
│   │   └── aligned_faces/  # Preprocessed and aligned faces
│   └── embeddings/  # Stored face embeddings
│
├── src/
│   ├── __init__.py
│   ├── config.py  # Configuration settings
│   ├── preprocess.py  # Face detection and alignment
│   ├── dataset.py  # Custom PyTorch Dataset
│   ├── model.py  # PyTorch model definition
│   ├── embeddings.py  # Embedding generation
│   ├── train.py  # SVM classifier training
│   └── evaluate.py  # Model evaluation
│
├── notebooks/
│   └── exploration.ipynb  # Jupyter notebook for data exploration
│
├── scripts/
│   ├── download_dataset.sh  # Script to download LFW dataset
│   └── run_pipeline.py  # Main script to run the entire pipeline
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_dataset.py
│   └── test_model.py
│
├── requirements.txt  # Project dependencies
├── README.md  # Project documentation
└── .gitignore  # Git ignore file