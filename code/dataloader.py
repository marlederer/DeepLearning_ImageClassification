import numpy as np
import world
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import os
import pickle
import csv

#Set Dir
current_dir = os.path.abspath(__file__)
# Navigate to the parent directory (one level up)
parent_dir = os.path.dirname(current_dir)
# Set the working directory to the parent directory
os.chdir(parent_dir)

class Loader():
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
            return;

        if "gtsrb" in path:
            training_file = "..\\data\gtsrb\\traffic-signs-data\\train.p"
            validation_file = "..\\data\gtsrb\\traffic-signs-data\\valid.p"
            testing_file = "..\\data\gtsrb\\traffic-signs-data\\test.p"

            with open(training_file, mode='rb') as f:
                train = pickle.load(f)
            with open(validation_file, mode='rb') as f:
                valid = pickle.load(f)
            with open(testing_file, mode='rb') as f:
                test = pickle.load(f)

            # Mapping ClassID to traffic sign names
            signs = []
            with open('..\\data\gtsrb\\signnames.csv', 'r') as csvfile:
                signnames = csv.reader(csvfile, delimiter=',')
                next(signnames, None)
                for row in signnames:
                    signs.append(row[1])
                csvfile.close()

            self.X_train, self.y_train = train['features'], train['labels']
            self.X_valid, self.y_valid = valid['features'], valid['labels']
            self.X_test, self.y_test = test['features'], test['labels']

            self.n_classes = len(np.unique(self.y_train))

            self.label_names = None

            print(f"{world.dataset} is ready to go")
            return;