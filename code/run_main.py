import world
import utils
from world import cprint
import numpy as np
import time, datetime
import Procedure
from os.path import join
import os
import model
import utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import random
import numpy as np
from sklearn.utils import shuffle

import register
from register import dataset
seed = 2024

#Classifcation_Model = register.MODELS[world.model_name](world.config, dataset, world.args)

############
# Validation set preprocessing
X_valid_preprocessed = utils.preprocess(dataset.X_valid)
normalized_images = utils.preprocess(dataset.X_train)

DIR = 'Saved_Models'

model = register.MODELS[world.model_name]
if world.model_name in ['LeNet', 'VGGnet']:
    keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
    keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(dataset.y_train)
        print("Training ...")
        print()
        for i in range(world.TRAIN_epochs):
            normalized_images, y_train = shuffle(normalized_images, dataset.y_train)
            for offset in range(0, num_examples, world.BATCH_SIZE):
                end = offset + world.BATCH_SIZE
                batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]
                sess.run(model.training_operation,
                        feed_dict={model.x: batch_x,
                                    model.y: batch_y,
                                    keep_prob: 0.5,
                                    keep_prob_conv: 0.7})

            validation_accuracy = model.evaluate(X_valid_preprocessed, dataset.y_valid, world.BATCH_SIZE)
            print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i + 1, (validation_accuracy * 100)))
        model.saver.save(sess, os.path.join(DIR, world.model_name))
        print("Model saved")
