"""mlp.py

Runs the MLP model either for testing or producing the Kaggle results upload
file
"""
import pandas as pd
import numpy as np

import prep
from sklearn.neural_network import MLPClassifier

def test_model(training_data, layers, subset_pct=.8, iterations=1):
    """Tests the model on the set of training data

    Extracts a random subset of the training data and trains the model on that
    subset.  Then, the model predicts the remainder of the training data.  The
    ratio of correct to incorrect predictions is printed.

    `training_data`: filepath to the training data
    `layers`: a tuple containing the sizes of the hidden layers
    `subset_pct`: percentage of training data to use for training the model
    `iterations`: number of test iterations performed
    """
    pass

def run_model(training_data, prediction_data, output_file, layers):
    """Runs the model and creates an output file with survival predictions

    `training_data`: filepath to the training dataset
    `prediction_data`: filepath to the dataset we will use to predict values
    `output_file`: filepath of the .csv file to be uploaded to Kaggle
    `layers`: tuple containing the sizes of the hidden layers
    """
    pass
