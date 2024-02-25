import world
import utils
from world import cprint
import numpy as np
import time, datetime
import Procedure
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import os
import model
import utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import register
from register import dataset
from keras.preprocessing.image import ImageDataGenerator

seed = 2024

############
# Validation set preprocessing

DIR = 'Saved_Models'

model = register.MODELS[world.model_name]

if world.model_name in ['LeNet', 'VGGnet']:
    if world.LOAD == 0:
        start = time.time()

        X_valid_preprocessed = utils.preprocess(dataset.X_valid)
        normalized_images = utils.preprocess(dataset.X_train)

        end = time.time()
        print("Preprocessing of", world.model_name, "with dataset:", world.dataset, "took", end-start, "s")
        print()

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(dataset.y_train)
            print("Training ...")
            print()
            start = time.time()
            for i in range(world.TRAIN_epochs):
                normalized_images, dataset.y_train = shuffle(normalized_images, dataset.y_train)

                # Step 3: Use the flag to determine whether to apply augmentation
                if world.AUGMENTATION:
                    # Apply augmentation using ImageDataGenerator
                    datagen.fit(normalized_images)
                    augmented_data = datagen.flow(normalized_images, dataset.y_train, batch_size=world.BATCH_SIZE)
                else:
                    augmented_data = zip(normalized_images, dataset.y_train)

                for offset in range(0, num_examples, world.BATCH_SIZE):
                    end = offset + world.BATCH_SIZE
                    if world.AUGMENTATION:
                        batch_x, batch_y = augmented_data.next()
                    else:
                        batch_x, batch_y = normalized_images[offset:end], dataset.y_train[offset:end]

                    sess.run(model.training_operation,
                             feed_dict={model.x: batch_x,
                                        model.y: batch_y,
                                        model.keep_prob: 0.5,
                                        model.keep_prob_conv: 0.7})

                validation_accuracy = model.evaluate(X_valid_preprocessed, dataset.y_valid, world.BATCH_SIZE)
                print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i + 1, (validation_accuracy * 100)))
            end = time.time()
            model.saver.save(sess, os.path.join("..//", DIR, world.model_name + "-" + world.dataset + "-augmention_" + str(world.AUGMENTATION)))
            print("Model saved")
            print("Training of", world.model_name, "with dataset:", world.dataset, "took", end-start, "s")
            print()

    X_test_preprocessed = utils.preprocess(dataset.X_test)

    with tf.Session() as sess:
        model.saver.restore(sess, os.path.join("..//", DIR, world.model_name + "-" + world.dataset + "-augmention_" + str(world.AUGMENTATION)))
        start = time.time()
        y_pred = model.y_predict(X_test_preprocessed)
        end = time.time()
        print("Prediction of dataset:", world.dataset, "with model:", world.model_name, "took", end-start, "s")
        print()

        test_accuracy = sum(dataset.y_test == y_pred) / len(dataset.y_test)
        print("Test Accuracy = {:.1f}%".format(test_accuracy * 100))

    cm = confusion_matrix(dataset.y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.log(.0001 + cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Log of normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
