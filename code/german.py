# Importing Python libraries
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils

import random
import cv2
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Flatten
from sklearn.metrics import confusion_matrix
import model

matplotlib.use('TkAgg')

# is it using the GPU?
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Show current TensorFlow version
print(tf.__version__)

os.chdir("D:\TU\\1_Semster\ML\Exercise_3\DeepLearning_ImageClassification")
os.getcwd()

training_file = "data\gtsrb\\traffic-signs-data\\train.p"
validation_file= "data\gtsrb\\traffic-signs-data\\valid.p"
testing_file = "data\gtsrb\\traffic-signs-data\\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# Mapping ClassID to traffic sign names
signs = []
with open('data\gtsrb\\signnames.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Number of training examples
n_train = X_train.shape[0]

# Number of testing examples
n_test = X_test.shape[0]

# Number of validation examples.
n_validation = X_valid.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples: ", n_train)
print("Number of testing examples: ", n_test)
print("Number of validation examples: ", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

X_train, y_train = shuffle(X_train, y_train)

keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers

EPOCHS = 30
BATCH_SIZE = 64
DIR = 'Saved_Models'

#LeNet_Model = LeNet(n_out=n_classes)
LeNet_Model = model.VGGnet(n_out=n_classes)
model_name = "LeNet"

normalized_images = utils.preprocess(X_train)

VGGNet_Model = model.VGGnet(n_out = n_classes)
model_name = "VGGNet"

# Validation set preprocessing
X_valid_preprocessed = utils.preprocess(X_valid)
one_hot_y_valid = tf.one_hot(y_valid, 43)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    print("Training...")
    print()
    try:
        for i in range(EPOCHS):
            normalized_images, y_train = shuffle(normalized_images, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]
                sess.run(VGGNet_Model.training_operation,
                feed_dict={VGGNet_Model.x: batch_x, VGGNet_Model.y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})

            validation_accuracy = VGGNet_Model.evaluate(X_valid_preprocessed, y_valid)
            print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))
        VGGNet_Model.saver.save(sess, os.path.join(DIR, model_name))
        print("Model saved")
    except Exception as e:
        print(e)

# Test set preprocessing
X_test_preprocessed = util.preprocess(X_test)

with tf.Session() as sess:
    VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
    y_pred = VGGNet_Model.y_predict(X_test_preprocessed)
    test_accuracy = sum(y_test == y_pred)/len(y_test)
    print("Test Accuracy = {:.1f}%".format(test_accuracy*100))

cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.log(.0001 + cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Log of normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()