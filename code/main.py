import numpy as np
import matplotlib.pyplot as plt
import os

from feature_extraction import create_three_channel_histograms

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_images(images, labels, num_images=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.title(labels[i])
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

def show_image_and_histogram(images, labels, histograms, bins=256, num_images=5):
    fig, axes = plt.subplots(num_images, 4, figsize=(15, 4*num_images))

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


path = os.path.join('..', 'data', 'cifar_10', 'cifar-10-python', 'cifar-10-batches-py')
img_path = os.path.join(path, 'data_batch_1')
meta_path = os.path.join(path, 'batches.meta')

img_dict = unpickle(img_path)
label_names = get_labels_from_meta_file(meta_path)

X = img_dict[b'data']
Y = img_dict[b'labels']

X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
Y = np.array(Y)

img_histograms = create_three_channel_histograms(X)

# Display some random images and their 3-channel histograms
random_indices = np.random.randint(0, len(X), 5)
labels = [label_names[label] for label in Y[random_indices]]
show_image_and_histogram(X[random_indices], labels, img_histograms[random_indices])

