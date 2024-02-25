import os
from os.path import join
from enum import Enum
from parse import parse_args
import multiprocessing

import sys

import random
import numpy as np

from warnings import simplefilter

seed = 2020

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')


sys.path.append(join(CODE_PATH, 'sources'))

config = {}
all_dataset = ['cifar-10', 'gtsrb']
all_models = ['hist', 'bovw', 'LeNet', 'VGGnet']
# config['batch_size'] = 4096
config['batch_size'] = args.batch
config['keep_prob'] = args.keepprob
config['keep_prob_conv'] = args.keepprobconv
config['lr'] = args.lr
config['args'] = args
config['dataset'] = args.dataset
config['epochs'] = args.epochs
config['augmentation'] = args.augmentation
config['mu'] = args.mu
config['sigma'] = args.sigma

seed = args.seed

dataset = args.dataset
model_name = args.model

LR = args.lr
MU = args.mu
SIGMA = args.sigma
AUGMENTATION = args.augmentation
BATCH_SIZE = args.batch
TRAIN_epochs = args.epochs
LOAD = args.load
# let pandas shut up


simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")