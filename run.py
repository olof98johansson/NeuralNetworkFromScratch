import numpy as np
import pandas as pd
import os

import plotly.graph_objects as go
import plotly.express as px
import time

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import NeuralNetwork

def run2d():
    # Training data
    training_df = pd.read_csv('./datasets/2D/training_set.csv', header=None)
    training_list = training_df.values.tolist()
    training_inputs = np.array(training_list)[:, :-1]
    training_labels = np.array(training_list)[:, -1]

    # Validation data
    validation_df = pd.read_csv('datasets/2D/validation_set.csv', header=None)
    validation_list = validation_df.values.tolist()
    validation_inputs = np.array(validation_list)[:, :-1]
    validation_labels = np.array(validation_list)[:, -1]

    # Initialize network
    net = NeuralNetwork.Network([5, 5], input_size=2, epochs=300, learning_rate=0.02, init_type='glorot')

    # Train the network
    training_errors, validation_errors = net.train(training_inputs,
                                                   training_labels,
                                                   validation_inputs,
                                                   validation_labels,
                                                   validate=True)
    # Validate on validation data and plot
    net.validate(validation_inputs, validation_labels)
    net.plot_errors(training_errors, validation_errors, True)
    net.plot_errors(training_errors)


def run3d():
    # Training data
    training_df = pd.read_csv('./datasets/3D/training_set_3d.csv', header=None)
    training_list = training_df.values.tolist()
    training_inputs = np.array(training_list)[:, :-1]
    training_labels = np.array(training_list)[:, -1]

    # Validation data
    validation_df = pd.read_csv('datasets/3D/validation_set_3d.csv', header=None)
    validation_list = validation_df.values.tolist()
    validation_inputs = np.array(validation_list)[:, :-1]
    validation_labels = np.array(validation_list)[:, -1]

    # Initialize network
    net = NeuralNetwork.Network([3, 3, 3], input_size=3, epochs=30, learning_rate=0.009, init_type='glorot')

    # Train the network
    training_errors, validation_errors = net.train(training_inputs,
                                                   training_labels,
                                                   validation_inputs,
                                                   validation_labels,
                                                   validate=True)
    # Validation and plot results
    net.validate(validation_inputs, validation_labels)
    net.plot_errors(training_errors, validation_errors, True)
    net.plot_errors(training_errors)

if __name__=='__main__':
    run2d()
    run3d()
