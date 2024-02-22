import world
import dataloader
import model
import utils
from pprint import pprint

seed = 2020
import random
import numpy as np

if world.dataset in ['cifar-10', 'gtsrb']:
    dataset = dataloader.Loader(path="../data/" + world.dataset)
else:
    print("Dataset: ", world.dataset, " not found. Check dataset parameter!")

print('===========config================')
pprint(world.config)
print('===========end===================')


MODELS = {
    'histogram': model.Histogram,
    'bag-of-visual-words' : model.BOVW,
    'LeNet' : model.LeNet(n_out=dataset.n_classes),
    'VGGnet' : model.VGGnet(n_out=dataset.n_classes)
}