import numpy as np
import matplotlib.pyplot as plt
import os

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


def get_labels_from_meta_file(file_path):
    meta_dict = unpickle(file_path)
    label_names = meta_dict[b'label_names']
    label_names = [name.decode('utf-8') for name in label_names]
    return label_names

path = os.path.join(os.getcwd(), 'data', 'cifar_10', 'cifar-10-python', 'cifar-10-batches-py')
img_path = os.path.join(path, 'data_batch_1')
meta_path = os.path.join(path, 'batches.meta')

img_dict = unpickle(img_path)
label_names = get_labels_from_meta_file(meta_path)

X = img_dict[b'data']
Y = img_dict[b'labels']

X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
Y = np.array(Y)

# Display some random images
random_indices = np.random.randint(0, len(X), 5)
labels = [label_names[label] for label in Y[random_indices]]
show_images(X[random_indices],labels)
