# Animal Classification Project

This project implements a deep learning model for classifying different types of animals using transfer learning with EfficientNetB0.

## Project Structure

```
animalsclassification/
├── Training Data/
│   ├── Butterfly/
│   ├── Cat/
│   └── ...
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── callbacks.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Organize your data:
   - Place your training data in the `Training Data` directory
   - Split your data into train, validation, and test sets
   - Each class should have its own subdirectory

## Usage

1. Train the model:
```bash
python src/train.py
```

2. The training process will:
   - Load and preprocess the data
   - Create and compile the model
   - Train the model with early stopping and learning rate scheduling
   - Evaluate the model on the test set
   - Save the best model weights

## Model Architecture

- Base model: EfficientNetB0 (pre-trained on ImageNet)
- Additional layers:
  - Batch Normalization
  - Dense layer (256 units) with L1/L2 regularization
  - Dropout (0.45)
  - Output layer (softmax activation)

## Training Features

- Data augmentation (horizontal flip)
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Custom training callback for monitoring and control

## Results

The model's performance can be monitored through:
- Training history plots
- Confusion matrix
- Classification report
- Sample predictions visualization

## Requirements

- Python 3.7+
- TensorFlow 2.9.1
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn 