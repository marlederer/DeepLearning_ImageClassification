from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2

matplotlib.use('TkAgg')


from feature_extraction import create_three_channel_histograms, extract_SIFT_features, extract_ORB_features


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_batch(file):
    batch_data = unpickle(file)
    X = batch_data[b'data']
    Y = batch_data[b'labels']
    X = X.reshape(len(Y), 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y = np.array(Y)
    return X, Y


def show_images(images, labels, num_images=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.title(labels[i])
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


def show_image_and_histogram(images, labels, histograms, bins=256, num_images=5):
    fig, axes = plt.subplots(num_images, 4, figsize=(15, 4 * num_images))

    axes[0, 1].set_title('Red Histogram', color='red')
    axes[0, 2].set_title('Green Histogram', color='green')
    axes[0, 3].set_title('Blue Histogram', color='blue')

    for i in range(num_images):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(labels[i])
        axes[i, 0].axis('off')
        for j, color in enumerate(['red', 'green', 'blue']):
            axes[i, j + 1].bar(np.arange(len(histograms[i, j, :])), histograms[i, j, :], color=color, alpha=0.5)
    plt.show()


def get_labels_from_meta_file(file_path):
    meta_dict = unpickle(file_path)
    label_names = meta_dict[b'label_names']
    label_names = [name.decode('utf-8') for name in label_names]
    return label_names




# LOAD DATA
num_training_batches = 5
X_train_list = []
Y_train_list = []

for i in range(1, num_training_batches + 1):
    filename = f"../data/cifar_10/cifar-10-python/cifar-10-batches-py/data_batch_{i}"
    X_batch, Y_batch = load_cifar_batch(filename)
    X_train_list.append(X_batch)
    Y_train_list.append(Y_batch)

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(Y_train_list)

# Load test data
X_test, y_test = load_cifar_batch("../data/cifar_10/cifar-10-python/cifar-10-batches-py/test_batch")

path = os.path.join('..', 'data', 'cifar_10', 'cifar-10-python', 'cifar-10-batches-py')
img_path = os.path.join(path, 'data_batch_1')
meta_path = os.path.join(path, 'batches.meta')

label_names = get_labels_from_meta_file(meta_path)

# FEATURE EXTRACTION
"""
img_histograms_train = create_three_channel_histograms(X_train)
img_histograms_test = create_three_channel_histograms(X_test)

img_histograms_train = img_histograms_train.reshape(len(img_histograms_train), -1)
img_histograms_test = img_histograms_test.reshape(len(img_histograms_test), -1)

# Display some random images and their 3-channel histograms
random_indices = np.random.randint(0, len(X_train), 5)
labels = [label_names[label] for label in Y_train[random_indices]]
show_image_and_histogram(X_train[random_indices], labels, img_histograms[random_indices])
"""
sift_keypoints_train, sift_descriptors_train = extract_SIFT_features(X_train)
sift_keypoints_test, sift_descriptors_test = extract_SIFT_features(X_test)

# CREATE MODEL
#X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = KNeighborsClassifier()
clf.fit(img_histograms_train, y_train)
predict = clf.predict(img_histograms_test)

accuracy = accuracy_score(y_test, predict)
print("Accuracy for historgram:", accuracy)
###
