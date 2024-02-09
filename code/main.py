import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

img_dict = unpickle("../data/cifar_10/cifar-10-python/cifar-10-batches-py/data_batch_1")

X = img_dict[b'data']
Y = img_dict[b'labels']

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

def show_images(images, labels, num_images=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

# Display some random images
random_indices = np.random.randint(0, len(X), 5)
show_images(X[random_indices], Y[random_indices])