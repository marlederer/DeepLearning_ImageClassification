import numpy as np
import cv2


def create_three_channel_histograms(images, bins=256):
    """
    creates the 3-channel histograms (red, green, blue) for all images

    Parameters
    ----------
    images : list
        List of images which are 3D-arrays in shape (32, 32, 3)
    bins : int
        number of desired bins. default value of 256 means that now pixel values are grouped together

    Returns
    -------
    histograms: np.array
        Array of all histograms, has shape of [len(images), 3, bins]
    """
    histograms = []
    for image in images:
        histograms.append([np.histogram(image[:, :, i], bins=bins, range=(0, 256))[0] for i in range(3)])
    return np.array(histograms)

def extract_SIFT_features(images, debug=False):
    """
    converts the images to grayscale and extracts SIFT features

    Parameters
    ----------
    images : list
        List of images which are 3D-arrays in shape (32, 32, 3)
    debug : bool
        displays the original and graysccale image if set to True

    Returns
    -------
    keypoints_list: list
        List of SIFT-keypoints for each image
    descriptors_list: list
        List of SIFT-descriptors for each image
    """

    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

        if debug:
            gray_with_channels = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            concatenated_image = np.concatenate((image, gray_with_channels), axis=1)

            cv2.imshow('Original and Grayscale Image', concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return keypoints_list, descriptors_list
