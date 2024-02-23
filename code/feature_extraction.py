import numpy as np
import cv2
from sklearn.cluster import KMeans
import skimage.morphology as morp
from skimage.filters import rank
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to extract 3-channel histograms from images
def extract_histograms(images, bins=256):
    print('Extracting 3-channel histograms for {} images...'.format(len(images)))
    histograms = []
    for image in images:
        hist = []
        for channel in range(3):
            hist.append(cv2.calcHist([image], [channel], None, [bins], [0, 256]))
        histograms.append(np.concatenate(hist))

    print('Extracted {} histograms'.format(len(histograms)))
    return np.array(histograms)


# Function to extract SIFT descriptors
def extract_sift_descriptors(images):
    print('Extracting SIFT descriptors for {} images...'.format(len(images)))
    sift = cv2.SIFT_create()
    descriptors = []
    for image in images:
        keypoints, descriptor = sift.detectAndCompute(image, None)
        if descriptor is not None:
            descriptors.extend(descriptor)

    print('Extracted {} SIFT descriptors for {} images'.format(len(descriptors), len(images)))
    return np.array(descriptors)


def create_visual_vocabulary(X_train_sift, num_clusters):
    print('Creating visual vocabulary with {} clusters...'.format(num_clusters))
    # Create KMeans instance
    kmeans = KMeans(n_clusters=num_clusters)
    # Fit KMeans to the SIFT descriptors
    kmeans.fit(X_train_sift)
    # Get cluster centers
    visual_vocabulary = kmeans.cluster_centers_
    print('Visual vocabulary created with {} clusters'.format(num_clusters))
    return kmeans, visual_vocabulary


def create_bag_of_visual_words(images, kmeans, num_clusters):
    print('Creating Bag of Visual Words for {} images...'.format(len(images)))
    histograms = []
    sift = cv2.SIFT_create()
    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            # Assign each descriptor to the closest cluster center
            labels = kmeans.predict(descriptors)
            # Count occurrences of each cluster center
            hist, _ = np.histogram(labels, bins=num_clusters, range=(0, num_clusters - 1))
            histograms.append(hist)
        else:
            # If no descriptors found, append zeros
            histograms.append(np.zeros(num_clusters))
    print('Created Bag of Visual Words for {} images'.format(len(histograms)))
    return np.array(histograms)

# Evaluate models
def evaluate_model(model, X_val, y_val, label_names):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    if label_names is not None:
        plt.xticks(ticks=np.arange(len(label_names)), labels=label_names, rotation=45)
        plt.yticks(ticks=np.arange(len(label_names)), labels=label_names, rotation=0)

    plt.show()
    return acc, cm

def gray_scale(image):
    """
    Convert images to gray scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, footprint=kernel)
    return img_local

def image_normalize(image):
    """
    Normalize images to [0, 1] scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    image = np.divide(image, 255)
    return image