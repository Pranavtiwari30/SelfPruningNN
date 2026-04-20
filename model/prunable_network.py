# model/prunable_network.py

import torch
import torch.nn as nn
from model.prunable_layer import PrunableLinear
import config


class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()

        layer_sizes = (
            [config.INPUT_SIZE]
            + config.HIDDEN_SIZES
            + [config.OUTPUT_SIZE]
        )

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(PrunableLinear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

    def get_all_gates(self):
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().view(-1))
        return torch.cat(all_gates)