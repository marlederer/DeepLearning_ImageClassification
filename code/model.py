import os

import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import random
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Flatten
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from sklearn.cluster import KMeans
import cv2
from feature_extraction import evaluate_model, create_bag_of_visual_words, create_visual_vocabulary, extract_sift_descriptors, extract_histograms
import world
import pickle

matplotlib.use('TkAgg')


x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="x")
y = tf.placeholder(tf.int32, (None), name="y")

keep_prob = tf.placeholder(tf.float32, name="keep_prob")       # For fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32, name="keep_prob_conv")  # For convolutional layers

# BOVW & Histogram


class Histogram:
    def __init__(self, dataset = 0, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        # Train and test variables
        y_train = dataset.y_train
        y_test = dataset.y_test

        start = time.time()

        X_train_hist = extract_histograms(dataset.X_train)
        X_test_hist = extract_histograms(dataset.X_test)
        X_train_hist = np.squeeze(X_train_hist)
        X_test_hist = np.squeeze(X_test_hist)

        end = time.time()
        print("Preprocessing of", world.model_name, "with dataset:", world.dataset, "took", end-start, "s")
        print()

        rf_classifier_hist = 0

        print(os.getcwd())

        if world.LOAD == 0:
            print('Training models...')
            print('Training Random Forest Classifier for histogram-based approach...')
            start = time.time()
            rf_classifier_hist = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier_hist.fit(X_train_hist, y_train)
            end = time.time()
            print("Training of", world.model_name, "with dataset:", world.dataset, "took", end - start, "s")
            with open(os.path.dirname(os.getcwd()) + '\\Saved_Models\\' + world.model_name + '-' + world.dataset + '.pkl', 'wb') as f:
                pickle.dump(rf_classifier_hist, f)
        else:
            with open(os.path.dirname(os.getcwd()) + '\\Saved_Models\\' + world.model_name + '-' + world.dataset + '.pkl', 'rb') as f:
                rf_classifier_hist = pickle.load(f)

        start = time.time()
        acc_hist, cm_hist = evaluate_model(rf_classifier_hist, X_test_hist, y_test, dataset.label_names)

        end = time.time()
        print("Prediction of dataset:", world.dataset, "with model:", world.model_name, "took", end-start, "s")
        print()

        print("Accuracy per class (Histogram-based approach):")
        print(acc_hist)
        print("Confusion Matrix (Histogram-based approach):")
        print(cm_hist)



class BOVW:
    def __init__(self, dataset=0, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        # Train and test variables
        X_train = dataset.X_train
        y_train = dataset.y_train
        X_test = dataset.X_test
        y_test = dataset.y_test


        print("Num classes: ", dataset.n_classes)
        num_clusters = 10 * dataset.n_classes

        start = time.time()
        X_train_sift = extract_sift_descriptors(X_train)

        kmeans, visual_vocabulary = create_visual_vocabulary(X_train_sift, num_clusters)

        X_train_bovw = create_bag_of_visual_words(X_train, kmeans, num_clusters)
        X_test_bovw = create_bag_of_visual_words(X_test, kmeans, num_clusters)

        end = time.time()
        print("Preprocessing of", world.model_name, "with dataset:", world.dataset, "took", end-start, "s")
        print()

        rf_classifier_bow = 0

        if world.LOAD == 0:
            print('Training Random Forest Classifier for SIFT + Bag of Words approach...')
            start = time.time()
            rf_classifier_bow = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier_bow.fit(X_train_bovw, y_train)
            end = time.time()
            print("Training of", world.model_name, "with dataset:", world.dataset, "took", end - start, "s")
            with open(os.path.dirname(os.getcwd()) + '\\Saved_Models\\' + world.model_name + '-' + world.dataset + '.pkl', 'wb') as f:
                pickle.dump(rf_classifier_bow, f)
        else:
            with open(os.path.dirname(os.getcwd()) + '\\Saved_Models\\' + world.model_name + '-' + world.dataset + '.pkl', 'rb') as f:
                rf_classifier_bow = pickle.load(f)

        start = time.time()
        acc_bow, cm_bow = evaluate_model(rf_classifier_bow, X_test_bovw, y_test, dataset.label_names)

        end = time.time()
        print("Prediction of dataset:", world.dataset, "with model:", world.model_name, "took", end-start, "s")
        print()

        print("Accuracy per class (SIFT + Bag of Words approach):")
        print(acc_bow)
        print("Confusion Matrix (SIFT + Bag of Words approach):")
        print(cm_bow)


class LeNet:

    def __init__(self, n_out=10, mu=0, sigma=0.1, learning_rate=0.001):
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma

        self.x = x
        self.y = y

        self.keep_prob = keep_prob
        self.keep_prob_conv = keep_prob_conv

        # Layer 1 (Convolutional): Input = 32x32x1. Output = 28x28x6.
        self.filter1_width = 5
        self.filter1_height = 5
        self.input1_channels = 1
        self.conv1_output = 6
        # Weight and bias
        self.conv1_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter1_width, self.filter1_height, self.input1_channels, self.conv1_output),
            mean=self.mu, stddev=self.sigma))
        self.conv1_bias = tf.Variable(tf.zeros(self.conv1_output))
        # Apply Convolution
        self.conv1 = tf.nn.conv2d(self.x, self.conv1_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_bias

        # Activation:
        self.conv1 = tf.nn.relu(self.conv1)

        # Pooling: Input = 28x28x6. Output = 14x14x6.
        self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2 (Convolutional): Output = 10x10x16.
        self.filter2_width = 5
        self.filter2_height = 5
        self.input2_channels = 6
        self.conv2_output = 16
        # Weight and bias
        self.conv2_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter2_width, self.filter2_height, self.input2_channels, self.conv2_output),
            mean=self.mu, stddev=self.sigma))
        self.conv2_bias = tf.Variable(tf.zeros(self.conv2_output))
        # Apply Convolution
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_weight, strides=[1, 1, 1, 1],
                                  padding='VALID') + self.conv2_bias

        # Activation:
        self.conv2 = tf.nn.relu(self.conv2)

        # Pooling: Input = 10x10x16. Output = 5x5x16.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flattening: Input = 5x5x16. Output = 400.
        latten_layer = Flatten()
        self.fully_connected0 = latten_layer(self.conv2)

        # Layer 3 (Fully Connected): Input = 400. Output = 120.
        self.connected1_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
        self.connected1_bias = tf.Variable(tf.zeros(120))
        self.fully_connected1 = tf.add((tf.matmul(self.fully_connected0, self.connected1_weights)),
                                       self.connected1_bias)

        # Activation:
        self.fully_connected1 = tf.nn.relu(self.fully_connected1)

        # Layer 4 (Fully Connected): Input = 120. Output = 84.
        self.connected2_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
        self.connected2_bias = tf.Variable(tf.zeros(84))
        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.connected2_weights)),
                                       self.connected2_bias)

        # Activation.
        self.fully_connected2 = tf.nn.relu(self.fully_connected2)

        # Layer 5 (Fully Connected): Input = 84. Output = 43.
        self.output_weights = tf.Variable(tf.truncated_normal(shape=(84, n_out), mean=self.mu, stddev=self.sigma))
        self.output_bias = tf.Variable(tf.zeros(n_out))
        self.logits = tf.add((tf.matmul(self.fully_connected2, self.output_weights)), self.output_bias)

        # Training operation
        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Saving all variables
        self.saver = tf.train.Saver()

    def y_predict(self, X_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset:offset + BATCH_SIZE]
            y_pred[offset:offset + BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1),
                                                          feed_dict={x: batch_x, keep_prob: 1, keep_prob_conv: 1})
        return y_pred

    def evaluate(self, X_data, y_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation,
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples


class VGGnet:

    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma

        self.x = x
        self.y = y

        self.keep_prob = keep_prob
        self.keep_prob_conv = keep_prob_conv

        # Layer 1 (Convolutional): Input = 32x32x1. Output = 32x32x32.
        self.conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean=self.mu, stddev=self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(32))
        self.conv1 = tf.nn.conv2d(x, self.conv1_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b

        # ReLu Activation.
        self.conv1 = tf.nn.relu(self.conv1)

        # Layer 2 (Convolutional): Input = 32x32x32. Output = 32x32x32.
        self.conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean=self.mu, stddev=self.sigma))
        self.conv2_b = tf.Variable(tf.zeros(32))
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b

        # ReLu Activation.
        self.conv2 = tf.nn.relu(self.conv2)

        # Layer 3 (Pooling): Input = 32x32x32. Output = 16x16x32.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv2 = tf.nn.dropout(self.conv2, keep_prob_conv)

        # Layer 4 (Convolutional): Input = 16x16x32. Output = 16x16x64.
        self.conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean=self.mu, stddev=self.sigma))
        self.conv3_b = tf.Variable(tf.zeros(64))
        self.conv3 = tf.nn.conv2d(self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b

        # ReLu Activation.
        self.conv3 = tf.nn.relu(self.conv3)

        # Layer 5 (Convolutional): Input = 16x16x64. Output = 16x16x64.
        self.conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean=self.mu, stddev=self.sigma))
        self.conv4_b = tf.Variable(tf.zeros(64))
        self.conv4 = tf.nn.conv2d(self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b

        # ReLu Activation.
        self.conv4 = tf.nn.relu(self.conv4)

        # Layer 6 (Pooling): Input = 16x16x64. Output = 8x8x64.
        self.conv4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv4 = tf.nn.dropout(self.conv4, keep_prob_conv)  # dropout

        # Layer 7 (Convolutional): Input = 8x8x64. Output = 8x8x128.
        self.conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean=self.mu, stddev=self.sigma))
        self.conv5_b = tf.Variable(tf.zeros(128))
        self.conv5 = tf.nn.conv2d(self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b

        # ReLu Activation.
        self.conv5 = tf.nn.relu(self.conv5)

        # Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
        self.conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean=self.mu, stddev=self.sigma))
        self.conv6_b = tf.Variable(tf.zeros(128))
        self.conv6 = tf.nn.conv2d(self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv6_b

        # ReLu Activation.
        self.conv6 = tf.nn.relu(self.conv6)

        # Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
        self.conv6 = tf.nn.max_pool(self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv6 = tf.nn.dropout(self.conv6, keep_prob_conv)  # dropout

        # Flatten. Input = 4x4x128. Output = 2048.
        flatten_layer = Flatten()
        self.fc0 = flatten_layer(self.conv6)

        # Layer 10 (Fully Connected): Input = 2048. Output = 128.
        self.fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean=self.mu, stddev=self.sigma))
        self.fc1_b = tf.Variable(tf.zeros(128))
        self.fc1 = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b

        # ReLu Activation.
        self.fc1 = tf.nn.relu(self.fc1)
        self.fc1 = tf.nn.dropout(self.fc1, keep_prob)  # dropout

        # Layer 11 (Fully Connected): Input = 128. Output = 128.
        self.fc2_W = tf.Variable(tf.truncated_normal(shape=(128, 128), mean=self.mu, stddev=self.sigma))
        self.fc2_b = tf.Variable(tf.zeros(128))
        self.fc2 = tf.matmul(self.fc1, self.fc2_W) + self.fc2_b

        # ReLu Activation.
        self.fc2 = tf.nn.relu(self.fc2)
        self.fc2 = tf.nn.dropout(self.fc2, keep_prob)  # dropout

        # Layer 12 (Fully Connected): Input = 128. Output = n_out.
        self.fc3_W = tf.Variable(tf.truncated_normal(shape=(128, n_out), mean=self.mu, stddev=self.sigma))
        self.fc3_b = tf.Variable(tf.zeros(n_out))
        self.logits = tf.matmul(self.fc2, self.fc3_W) + self.fc3_b

        # Training operation
        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Saving all variables
        self.saver = tf.train.Saver()

    def y_predict(self, X_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset:offset + BATCH_SIZE]
            y_pred[offset:offset + BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1),
                                                          feed_dict={x: batch_x, keep_prob: 1, keep_prob_conv: 1})
        return y_pred

    def evaluate(self, X_data, y_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation,
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples