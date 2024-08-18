import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import preprocess 
from preprocess import jsonl_to_data
from model_experimental import Recurrent
import visualization

def train(model, times, magnitudes, accels, start_time, end_time, sequence_length, has_accel):
    '''
    This function trains the model on the given data and returns the loss and the prediction

    Args: 
    model: The model to train
    times: The times of the events
    magnitudes: The magnitudes of the events
    accels: The accelerations of the events
    start_time: The start time of the events
    end_time: The end time of the events
    sequence_length: The length of the sequence
    has_accel: Boolean indicating if the acceleration is being used

    Returns:
    losses: The loss of the model
    pred: The distributions predicted by the model
    '''
    # magnitudes: [batchsize(1) x sequence size x 1]
    # times: [batchsize(1) x sequence size + 1 x 1]
    # accels: [batchsize(1) x sequence size x 96]
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    losses = []
    times = np.expand_dims(np.array(times).T, axis=[0,2])
    magnitudes = np.expand_dims(np.array(magnitudes).T, axis=[0,2])
    accels = np.expand_dims(np.array(accels), axis=[0])
    with tf.GradientTape() as tape:
        pred = model(times, magnitudes, accels, has_accel=has_accel, training=True)
        loss = model.loss_function(pred, times[:, 1:, :], start_time, end_time)
        losses.append(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return losses, pred

def validate(model, times, magnitudes, accels, start_time, end_time, sequence_length, has_accel):
    '''
    This function validates the model on the given data and returns the loss and the prediction

    Args: model: The model to validate
    times: The times of the events
    magnitudes: The magnitudes of the events
    accels: The accelerations of the events
    start_time: The start time of the events
    end_time: The end time of the events
    sequence_length: The length of the sequence
    has_accel: Boolean indicating if the acceleration is being used

    Returns:
    losses: The loss of the model
    pred: The distributions predicted by the model
    '''
    losses = []
    times = np.expand_dims(np.array(times).T, axis=[0,2])
    magnitudes = np.expand_dims(np.array(magnitudes).T, axis=[0,2])
    accels = np.expand_dims(np.array(accels), axis=[0])
    pred = model(times, magnitudes, accels, has_accel=has_accel, training=False)
    loss = model.loss_function(pred, times[:, 1:, :], start_time, end_time)
    losses.append(loss)
    return losses, pred

def main():

    start_time, end_time = preprocess.get_year_unix_times(2018)
    epochs = 150
    model = Recurrent()
    training_losses = []
    validation_losses = []
    test_losses = []
    acc_list = []
    preprocess.shuffle_files('data')

    # Here is some code to loop through all of our training data and then run it against our validation data
    if os.path.exists('data/training'):
        val_dists_list = []
        # We now loop through all of our training data for the specified number of epochs
        for epoch in tqdm(range(epochs), "Training Progress"):
            epoch_training_losses = []
            epoch_validation_losses = []
            for filename in os.listdir('data/training'):
                file_path = os.path.join('data/training', filename)
                if os.path.isfile(file_path):
                    times, magnitudes, accels = preprocess.jsonl_to_data(file_path, start_time, end_time)
                    losses, train_pred = train(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=False)
                    epoch_training_losses.append(losses)
            training_losses.append(tf.math.reduce_mean(epoch_training_losses))
            # We now loop through all of our validation data
            for filename in os.listdir('data/validation'):
                file_path = os.path.join('data/validation', filename)
                if os.path.isfile(file_path):
                    times, magnitudes, accels = preprocess.jsonl_to_data(file_path, start_time, end_time)
                    losses, valid_pred = validate(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=False)
                    epoch_validation_losses.append(losses)
            validation_losses.append(tf.math.reduce_mean(epoch_validation_losses))
            acc = accuracy(model, start_time, end_time)
            acc_list.append(acc)
            print(f"Epoch {epoch}, Training Loss: {training_losses[-1]}, Validation Loss: {validation_losses[-1]}, Accuracy: {acc}")
            val_dists_list+=[valid_pred]
    #visuaization.plot_weibull_mixture(train_pred)
    visualization.plot_loss(training_losses)
    visualization.plot_loss(validation_losses, "Validation")
    visualization.plot_accuracy(acc_list)

    # Now we test our model on the test data
    for filename in os.listdir('data/testing'):
        file_path = os.path.join('data/testing', filename)
        if os.path.isfile(file_path):
            times, magnitudes, accels = preprocess.jsonl_to_data(file_path, start_time, end_time)
            losses, test_pred = validate(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=False)
            test_losses.append(tf.math.reduce_mean(losses))
    print(f"Test Loss: {tf.math.reduce_mean(test_losses)}")
    visualization.plot_basic(val_dists_list, "Distribution of one event over epochs")
    

def accuracy(model, start_time, end_time):
    '''
    This function calculates the accuracy of the model on the test data and returns the accuracy

    Args: model: The model to validate
    start_time: The start time of the events
    end_time: The end time of the events

    Returns: accuracy: The accuracy of the model
    '''
    correct = 0
    total=0
    for filename in os.listdir('data/testing'):
        file_path = os.path.join('data/testing', filename)
        if os.path.isfile(file_path):
            times, magnitudes, accels = preprocess.jsonl_to_data(file_path, start_time, end_time)
            losses, test_pred = validate(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=True)
            samples = test_pred.sample(1000)
            samples = tf.transpose(samples, perm=[1, 2, 0]) #1 by sequence by 1000

            quantiles = tfp.stats.percentile(samples, q=95, axis=-1)

            for i, quantile in enumerate(quantiles[0]):
                if quantile>times[i+1]:
                    correct+=1
                total+=1
    
    accuracy = correct/total
    print("Accuracy", accuracy)
    return accuracy
    

    
if __name__ == "__main__":
    main()
    plt.show()