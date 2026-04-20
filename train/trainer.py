# train/trainer.py

import torch
import torch.nn as nn
from model.prunable_layer import PrunableLinear
import config


def compute_sparsity_loss(model):
    """
    L1 norm of all gate values across every PrunableLinear layer.
    Gates are already in (0,1) via sigmoid, so sum = L1 norm.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total = total + gates.sum()
    return total


def train_one_epoch(model, loader, optimizer, criterion, lambda_val, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Total Loss = CrossEntropy + λ * SparsityLoss
        cls_loss      = criterion(outputs, labels)
        sparsity_loss = compute_sparsity_loss(model)
        loss          = cls_loss + lambda_val * sparsity_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(model, train_loader, lambda_val, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Training with λ = {lambda_val} ---")

    for epoch in range(1, config.EPOCHS + 1):
        loss, acc = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_val, device
        )
        print(f"Epoch [{epoch:02d}/{config.EPOCHS}] "
              f"Loss: {loss:.4f} | Train Acc: {acc:.2f}%")