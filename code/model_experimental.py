import tensorflow as tf
import tensorflow_probability as tfp
from keras import Sequential
#from keras_layers import Dense, Flatten, Reshape, Concatenate
from math import exp, sqrt
import numpy as np

class Recurrent(tf.keras.Model):

    '''
    Create a recurrent TPP model w/ recurrent encoder

    Args:
        input_magnitude: if magnitude be used as model input ----------------- YES
        predict_magnitude: if model predict the magnitude? ------------------- YES
        num_extra_features: Number of extra features to use as input. 
        hidden_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
    '''
    
    def __init__(self, input_magnitude: bool = True, # to use magnitude as input
                 #predict_magnitude: bool = False, # output distribution, NOT magnitude
                 hidden_size: int = 32, # hidden state size
                 num_components: int = 32, # WHAT SHOULD THIS BE ????? I THOUGHT 3 ???
                 rnn_type: str = "LSTM",
                 dropout_proba: float = 0.5,
                 tau_mean: float = 1.0, # mean inter-event times in data
                 mag_mean: float = 0.0, # mean earthquake magnitude in data
                 richter_b: float = 1.0, # fixed b value of Gutenberg-Richter distribution
                 mag_completeness: float = 2.0, # magnitude completeness
                 learning_rate: float = .001):
        
        # initialize model
        super().__init__()

        # set parameters
        self.input_magnitude = input_magnitude
        self.hidden_size = hidden_size
        self.num_components = num_components
 
        # set untrainables
        self.tau_mean = tf.constant(tau_mean, dtype=tf.float32)
        self.log_tau_mean = tf.math.log(self.tau_mean)
        self.mag_mean = tf.constant(mag_mean, dtype=tf.float32)
        self.richter_b = tf.constant(richter_b, dtype=tf.float32)
        self.mag_completeness = tf.constant(mag_completeness, dtype=tf.float32)
        
        # set learning rate
        self.learning_rate = learning_rate

        # RNN input features
        self.num_time_params = 3 * self.num_components
        self.hypernet_time = tf.keras.layers.Dense(self.num_time_params)
        
        # RNN defining
        self.num_rnn_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # magnitude features
        )

        # input size is num_rnn_inputs
        self.rnn = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, recurrent_dropout=0.5)
        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout_proba)

    # call function
    def call(self, times, magnitudes, accelaration=None, has_accel=True, training=False):
        '''
            inputs: 
                magnitudes: array containing the magnitudes of events
                times: array containing the times of events
                accels: array containing the accelerations of events
                has_accel: boolean indicating if the acceleration is being used
        '''
        # encode time
        times = self.encode_time(times)
        # concatenate all features
        if has_accel:
            features = tf.concat((times[:, :-1, :], magnitudes, accelaration), axis=-1)
        else:
            features = tf.concat((times[:, :-1, :], magnitudes), axis=-1)
        # pass features into RNN
        rnn_output = self.rnn(features, training=training)
        ## SHAPE OF OUTPUT (BATCH_SIZE, SEQUENCE_LENGTH, 32)

        context = self.dropout(rnn_output, training=training)
        # Time distribution parameters
        time_params = self.hypernet_time(context)
        ## SHAPE OF OUTPUT (BATCH_SIZE, SEQUENCE_LENGTH, 32)
        scale, shape, weight_logits = tf.split(
        time_params,
        [self.num_components, self.num_components, self.num_components],
        axis=-1,
        )
        scale = tf.math.softplus(tf.clip_by_value(scale, -5.0, float('inf')))
        shape = tf.math.softplus(tf.clip_by_value(shape, -5.0, float('inf')))
        weight_logits = tf.math.log_softmax(weight_logits, axis=-1)
        # print(shape, "\n", scale, "\n", weight_logits)
        component_dists = tfp.distributions.Weibull(shape, scale)
        mixture_dist = tfp.distributions.Categorical(logits=weight_logits)
        
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=component_dists,
            )

    def encode_time(self, inter_times):
        '''
        Encode the time intervals using the log transformation

        Args: inter_times: The time intervals to encode

        Returns: The encoded time intervals
        '''
        encoded_time = tf.math.log(tf.maximum(inter_times, 1e-15))
        
        return encoded_time

    def loss_function(self, distributions, intervals, start_time, end_time):
        '''
        Compute the negative log likelihood loss.

        Args:
            distributions: A batch of sequences of TensorFlow distributions.
            intervals: Shape (B, S, 1) interval[]

        Returns:
            The negative log likelihood loss.
        '''
       
        log_like = distributions.log_prob(tf.squeeze(tf.cast(tf.maximum(intervals, 1e-15), dtype=tf.float32), axis=-1)) #(B, S,)
        log_likelihood = tf.reduce_sum(log_like, -1) # (B,)

        surv = distributions.survival_function(
            tf.cast(tf.maximum(intervals[:, -1, :], 1e-15), dtype=tf.float32) #index into one after the last distribution, since we have num_distributions+1 time intervals
        )[:, -1]
        try:
            tf.debugging.check_numerics(surv, "Tensor has NaN values")
            pass
        except Exception as e:
            print("surv:", surv)
            print("distributions:", distributions)
            print(tf.cast(tf.maximum(intervals[:, -1, :], 1e-15), dtype=tf.float32))
        log_surv = tf.math.log(tf.maximum(surv, 1e-15) + 1e-10)
        try:
            tf.debugging.check_numerics(log_surv, "Tensor has NaN values")
        except Exception as e:
            print("log_surv:", log_surv)
        log_likelihood = log_likelihood + tf.reduce_sum(log_surv,-1)
        len_sequence = tf.cast(tf.shape(intervals)[1], dtype=tf.float32)
        return -log_likelihood/(len_sequence)