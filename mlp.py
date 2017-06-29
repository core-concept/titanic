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
    print('Parsing data......', end='')
    trainingData = prep.parse_data(training_data)
    trainingData = trainingData.fillna(value=0) # Placeholder - must assign NaNs
    print('DONE')
    correct = 0
    total = 0
    for i in range(iterations):
        print('Iteration {}...'.format(i+1), end='')
        sample = trainingData.sample(frac=subset_pct)
        complement = trainingData[~trainingData.index.isin(sample.index)]
        X = sample[[1, 2, 'male', 'n(Age)', 'n(SibSp)', 'n(Parch)', 'n(Fare)']]
        y = sample[['Survived']]
        Z = complement[[1, 2, 'male', 'n(Age)', 'n(SibSp)', 'n(Parch)',
                       'n(Fare)']]
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=layers, random_state=1)
        mlp.fit(X, y)
        prediction = mlp.predict(Z)
        total += len(prediction)
        correct += len(prediction) - sum(abs(complement['Survived'] - prediction))
        print('DONE')
    print('Testing complete: {} of {} survivors correctly predicted accurately.'
          .format(correct, total))
    print('Accuracy: {0:.2f}%'.format(correct/total))

def run_model(training_data, prediction_data, output_file, layers):
    """Runs the model and creates an output file with survival predictions

    `training_data`: filepath to the training dataset
    `prediction_data`: filepath to the dataset we will use to predict values
    `output_file`: filepath of the .csv file to be uploaded to Kaggle
    `layers`: tuple containing the sizes of the hidden layers
    """
    pass
