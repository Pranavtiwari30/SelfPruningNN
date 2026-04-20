# model/prunable_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight, also learnable
        # Initialized to 0.0 so sigmoid(0) = 0.5, gates start at 50% open
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialize weight the same way nn.Linear does (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Step 1: squash gate_scores into (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: mask the weights — dead gates zero out their weights
        pruned_weights = self.weight * gates

        # Step 3: standard linear operation with masked weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        # Convenience method — returns gate values detached from graph
        # Used later in evaluator and visualizer
        return torch.sigmoid(self.gate_scores).detach()