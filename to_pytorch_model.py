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

# SonoNet16
lasagne_model = models.SonoNet16(None, (None, None), num_labels=14)

with np.load('SonoNet16.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(lasagne_model['last_activation'], param_values)
layers = lasagne.layers.get_all_layers(lasagne_model['last_activation'])
pytorch_model = SonoNet.SonoNet16()

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

print('Saving SonoNet16 model to SonoNet16.pytorch.pth...')
torch.save(pytorch_model.state_dict(), 'SonoNet16.pytorch.pth')

# SonoNet32
lasagne_model = models.SonoNet32(None, (None, None), num_labels=14)

with np.load('SonoNet32.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(lasagne_model['last_activation'], param_values)
layers = lasagne.layers.get_all_layers(lasagne_model['last_activation'])
pytorch_model = SonoNet.SonoNet32()

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

print('Saving SonoNet32 model to SonoNet32.pytorch.pth...')
torch.save(pytorch_model.state_dict(), 'SonoNet32.pytorch.pth')

# SonoNet64
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

# SmallNet
lasagne_model = models.SmallNet(None, (None, None), num_labels=14)

with np.load('SmallNet.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(lasagne_model['last_activation'], param_values)
layers = lasagne.layers.get_all_layers(lasagne_model['last_activation'])
pytorch_model = SonoNet.SmallNet()

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

print('Saving SmallNet model to SmallNet.pytorch.pth...')
torch.save(pytorch_model.state_dict(), 'SmallNet.pytorch.pth')
