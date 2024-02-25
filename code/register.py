import world
import dataloader
import model
import utils
from pprint import pprint

import random
import numpy as np

seed = 2020


if world.dataset in ['cifar-10', 'gtsrb']:
    dataset = dataloader.Loader(path="../data/" + world.dataset)
else:
    print("Dataset: ", world.dataset, " not found. Check dataset parameter! (PLEASE USE EITHER 'cifar-10' or 'gtsrb')")

print('===========config================')
pprint(world.config)
print('===========end===================')

# MODELS Dictionary in order to chose the model with a cmd-argument

MODELS = {}

if world.model_name == 'hist':
    MODELS['hist'] = model.Histogram(dataset=dataset)
elif world.model_name == 'bovw':
    MODELS['bovw'] = model.BOVW(dataset=dataset)
elif world.model_name == 'LeNet':
    MODELS['LeNet'] = model.LeNet(n_out=dataset.n_classes, mu=world.mu, sigma=world.sigma)
elif world.model_name == 'VGGnet':
    MODELS['VGGnet'] = model.VGGnet(n_out=dataset.n_classes, mu=world.mu, sigma=world.sigma)
else:
    print("Unknown model name:", world.model_name)
