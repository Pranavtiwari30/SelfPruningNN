# evaluate/evaluator.py

import torch
from model.prunable_layer import PrunableLinear
import config


def evaluate(model, test_loader, device):
    """Test accuracy on CIFAR-10 test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def compute_sparsity(model):
    """
    Percentage of gates below GATE_THRESHOLD across all PrunableLinear layers.
    This is your sparsity level metric for the report table.
    """
    total_gates  = 0
    pruned_gates = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates         = module.get_gates()          # detached, (0,1)
            total_gates  += gates.numel()
            pruned_gates += (gates < config.GATE_THRESHOLD).sum().item()

    sparsity_pct = 100.0 * pruned_gates / total_gates
    return sparsity_pct, pruned_gates, total_gates


def report(model, test_loader, lambda_val, device):
    test_acc            = evaluate(model, test_loader, device)
    sparsity, pruned, total = compute_sparsity(model)

    print(f"\n── Results for λ = {lambda_val} ──")
    print(f"Test Accuracy : {test_acc:.2f}%")
    print(f"Sparsity      : {sparsity:.2f}%  ({pruned:,} / {total:,} gates pruned)")

    return {
        "lambda"    : lambda_val,
        "accuracy"  : round(test_acc, 2),
        "sparsity"  : round(sparsity, 2)
    }