import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os

# Plotting function to visualize the loss
def plot_loss(losses, type: str = "Training"):
    '''
    This function plots the loss over the epochs 

    Args:
    losses: list of losses
    type: string indicating the type of loss (Training or Validation)
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label=f'{type} Loss')
    plt.title(f'{type} Loss with Acceleration')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_accuracy(acc_list):
    '''
    This function plots the accuracy over the epochs

    Args: acc_list: list of accuracies
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(acc_list, label='Accuracy')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_basic(output, title):
    '''
    This function plots the distribution of the model

    Args: output: list of distributions
    title: string indicating the title of the plot
    '''
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for dist in output:  # Loop through each distribution in the output list
        dist = dist[0, 0]
        x = np.linspace(0, 200, 100)  # Generate a list of numbers between 0 and 1
        y = [dist.prob(i) for i in x]  # Calculate the probability for each value in x
        axs.plot(x, y)
    axs.set_title(title)
    plt.tight_layout()
    plt.show(block=False)

