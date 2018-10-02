#!/bin/env python3
#
# convert lasagne model to pytorch model
#
import numpy as np
import torch
import torch.nn as nn
import lasagne
from lasagne.layers import Conv2DLayer, BatchNormLayer

import models
import SonoNet

lasagne_model = models.SonoNet64(None, (None, None), num_labels=14)

with np.load('SonoNet64.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(lasagne_model['last_activation'], param_values)
layers = lasagne.layers.get_all_layers(lasagne_model['last_activation'])
pytorch_model = SonoNet.SonoNet64()

idx = 0
for layer in layers[1:]:
    if isinstance(layer, Conv2DLayer):
        pytorch_model[idx].weight = nn.Parameter(torch.from_numpy(layer.W.get_value()))
    elif isinstance(layer, BatchNormLayer):
        pytorch_model[idx].bias = nn.Parameter(torch.from_numpy(layer.beta.get_value()))
        pytorch_model[idx].weight = nn.Parameter(torch.from_numpy(layer.gamma.get_value()))
        pytorch_model[idx].running_mean = nn.Parameter(torch.from_numpy(layer.mean.get_value()))
        # Note that lasagne stores inv_std = 1 / \sqrt{\sigma^2 + \epsilon} with epsilon = 1e-4
        # while pytorch stores var = \sigma^2 with epsilon = 1e-5
        # and lasagne
        epsilon = 1e-4
        inv_std = torch.from_numpy(layer.inv_std.get_value())
        var = torch.add(torch.pow(torch.reciprocal(inv_std), 2), -1e-4)
        pytorch_model[idx].running_var = nn.Parameter(var)
    idx += 1

print('Saving SonoNet64 model to SonoNet64.pytorch.pth...')
torch.save(pytorch_model.state_dict(), 'SonoNet64.pytorch.pth')
