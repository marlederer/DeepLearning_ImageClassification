import numpy as np
import world
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import os
import pickle
import csv
from sklearn.utils import shuffle
from zipfile import ZipFile
import requests

#Set Dir
current_dir = os.path.abspath(__file__)
# Navigate to the parent directory (one level up)
parent_dir = os.path.dirname(current_dir)
# Set the working directory to the parent directory
os.chdir(parent_dir)


def download_and_extract(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

    with ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(save_path))


def load_data(file_path):
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    return data


class Loader:
    """
    Dataset type for pytorch \n
    Incldue graph information
    cifar-10 dataset,['cifar-10', 'gtsrb']:
    """

    def __init__(self, config=world.config, path="../data/cifar-10", flag_test=0):
        # train or test
        world.cprint(f'loading [{path}]')
        if "cifar-10" in path:
            (X_train, y_train), (self.X_test, y_test) = cifar10.load_data()
            self.X_train, self.X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

            self.y_test = y_test.reshape(len(y_test))
            self.y_train = y_train.reshape(len(y_train))
            self.y_valid = y_valid.reshape(len(y_valid))

            self.n_classes = len(np.unique(self.y_train))

            self.label_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

            print(f"{world.dataset} is ready to go")
            return

        if "gtsrb" in path:
            training_file = "..\\data\gtsrb\\traffic-signs-data\\train.p"
            validation_file = "..\\data\gtsrb\\traffic-signs-data\\valid.p"
            testing_file = "..\\data\gtsrb\\traffic-signs-data\\test.p"

            # Check if files exist
            if all(os.path.exists(file) for file in [training_file, validation_file, testing_file]):
                # Load data
                train = load_data(training_file)
                valid = load_data(validation_file)
                test = load_data(testing_file)
            else:
                # Download and extract the zip file
                zip_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
                zip_save_path = "..\\data\\gtsrb\\traffic-signs-data\\traffic-signs-data.zip"
                download_and_extract(zip_url, zip_save_path)

                # Load data after extraction
                train = load_data(training_file)
                valid = load_data(validation_file)
                test = load_data(testing_file)


            # Mapping ClassID to traffic sign names
            self.label_names = []
            with open('..\\data\gtsrb\\signnames.csv', 'r') as csvfile:
                signnames = csv.reader(csvfile, delimiter=',')
                next(signnames, None)
                for row in signnames:
                    self.label_names.append(row[1])
                csvfile.close()

            self.X_train, self.y_train = train['features'], train['labels']
            self.X_valid, self.y_valid = valid['features'], valid['labels']
            self.X_test, self.y_test = test['features'], test['labels']

            self.n_classes = len(np.unique(self.y_train))

            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

            print(f"{world.dataset} is ready to go")
            return;