# utils/visualize.py

import os
import torch
import matplotlib.pyplot as plt
from model.prunable_layer import PrunableLinear
import config


def plot_gate_distribution(model, lambda_val):
    """
    Plots histogram of all gate values for a trained model.
    A successful result shows a large spike near 0 and a
    smaller cluster away from 0.
    """
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates().cpu().numpy().flatten()
            all_gates.extend(gates)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.hist(all_gates, bins=100, color='steelblue', edgecolor='none')
    plt.title(f'Gate Value Distribution  (λ = {lambda_val})')
    plt.xlabel('Gate Value (0 = pruned, 1 = fully active)')
    plt.ylabel('Count')
    plt.axvline(x=config.GATE_THRESHOLD, color='red',
                linestyle='--', label=f'Prune threshold ({config.GATE_THRESHOLD})')
    plt.legend()
    plt.tight_layout()

    filename = f"gate_dist_lambda_{str(lambda_val).replace('.', '_')}.png"
    filepath = os.path.join(config.RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"Plot saved → {filepath}")
    return filepath