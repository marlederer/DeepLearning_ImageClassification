import numpy as np
import cv2
from sklearn.cluster import KMeans


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

def extract_ORB_features(images, debug=False):
    """
    converts the images to grayscale and extracts ORB features

    Parameters
    ----------
    images : list
        List of images which are 3D-arrays in shape (32, 32, 3)
    debug : bool
        displays the original and graysccale image if set to True

    Returns
    -------
    keypoints_list: list
        List of ORB-keypoints for each image
    descriptors_list: list
        List of ORB-descriptors for each image
    """
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptors_list = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

        if debug:
            gray_with_channels = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            concatenated_image = np.concatenate((image, gray_with_channels), axis=1)

            cv2.imshow('Original and Grayscale Image', concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return keypoints_list, descriptors_list

def create_bovw(descriptors_list, num_clusters, keypoints_list):
    """
    Creates Bag of Visual Words (BOVW) from a list of SIFT descriptors

    Parameters
    ----------
    descriptors_list : list
        List of SIFT descriptors for each image
    num_clusters : int
        Number of clusters for KMeans clustering

    Returns
    -------
    bovw_features : numpy array
        Bag of Visual Words (BOVW) features for all images
    """

    # Concatenate all descriptors into a single array
    all_descriptors = np.concatenate(descriptors_list, axis=0)

    print(all_descriptors.shape)
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters*10)
    kmeans.fit(all_descriptors)

    # Assign descriptors to visual words
    """"
    bovw_features = []
    for descriptors in descriptors_list:
        visual_words = kmeans.predict(descriptors)
        histogram, _ = np.histogram(visual_words, bins=range(num_clusters + 1), density=True)
        bovw_features.append(histogram)

    return np.array(bovw_features)
    """
    histo_list = []
    for keypoint in keypoints_list:
        nkp = np.size(len(keypoint))

        for d in all_descriptors:
            #d = np.array(d, dtype=np.float64)
            idx = kmeans.predict([d])
            histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

            histo_list.append(histo)

    return histo_list
