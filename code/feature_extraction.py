import numpy as np

# creates a 3-channel histogram for each image which is used as a feature
def create_three_channel_histograms(images, bins=256):
    histograms = []
    for image in images:
        histograms.append([np.histogram(image[:, :, i], bins=bins, range=(0, 256))[0] for i in range(3)])
    return np.array(histograms)
