# Toxic Comment Classification Notebook

## Overview
This project focuses on the classification of toxic comments using deep learning models. It was developed as part of the Deep Learning course in the Master of Science in Data Science program at CentraleSupélec. The LSTM and BERT classifiers are employed to identify various forms of toxicity in online comments. The dataset is sourced from the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).

## Installation
To run this notebook, ensure you have the following packages installed:
- Python 3.10
- torch
- torchmetrics
- numpy
- pandas
- matplotlib
- seaborn
- spacy
- nltk
- scikit-learn
- torchtext
- torchvision
- transformers

## Data
The dataset includes comments labeled with multiple forms of toxicity like insults, threats, and identity attacks. The data is pre-processed and augmented to address class imbalance issues.

## Models
Two models are implemented:
1. **LSTM Classifier**: Utilizes a Long Short-Term Memory layer followed by a fully connected layer.
2. **BERT Classifier**: Employs a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model for classification.

## Training and Evaluation
The notebook includes detailed steps for training and evaluating both models. Evaluation metrics include accuracy and F1 Score. A specific evaluation function `worst_group_accuracy` is implemented to identify the model's performance across different demographic groups.

## Results
The BERT model demonstrates superior performance in validation and test compared to the LSTM model, indicating its effectiveness in handling complex text data.

## Usage
To run the project:
1. Install required packages.
2. Download the dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place it in the appropriate directory.
3. Execute the notebook cells sequentially.

## Note
- The notebook requires GPU acceleration for efficient training of models.
- The data augmentation process is crucial to handle class imbalance in the dataset.
- The project includes custom dataset classes and loaders for both LSTM and BERT models.
