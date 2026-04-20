# Self-Pruning Neural Network

A feedforward neural network that learns to prune its own weights during 
training via learnable gates — built from scratch in PyTorch for CIFAR-10 
image classification.

## Core Idea

Each weight has a learnable `gate_score` parameter. During the forward pass:

```
gate         = sigmoid(gate_score)        # squash to (0, 1)
pruned_weight = weight * gate             # mask the weight
output        = input @ pruned_weight.T + bias
```

A custom L1 sparsity loss penalises active gates, forcing the network to 
zero out weights it doesn't need:

```
Total Loss = CrossEntropyLoss + λ * Σ(all gate values)
```

The higher λ is, the more aggressively the network prunes itself.

## Project Structure

```
self-pruning-nn/
├── model/
│   ├── prunable_layer.py       # PrunableLinear — built from scratch
│   └── prunable_network.py     # SelfPruningNetwork using prunable layers
├── train/
│   └── trainer.py              # training loop + sparsity loss
├── evaluate/
│   └── evaluator.py            # test accuracy + sparsity metrics
├── utils/
│   ├── data_loader.py          # CIFAR-10 pipeline
│   └── visualize.py            # gate distribution plots
├── results/                    # plots + logs saved here
├── config.py                   # all hyperparameters
├── main.py                     # entry point — runs λ sweep
└── report.md                   # analysis and results
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Trains 3 models with λ = [0.0001, 0.01, 0.1] for 30 epochs each.
Results table and gate distribution plots saved to `./results/`.

Requires CUDA GPU recommended (tested on RTX 4050).

## Results

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|--------------|-------------------|
| 0.0001 | 55.14%       | 70.17%            |
| 0.01   | 44.98%       | 99.98%            |
| 0.1    | 21.65%       | 100.00%           |

**Best model:** λ = 0.0001 — 70% of weights pruned with 55.14% test accuracy.

## Key Implementation Detail

`PrunableLinear` does not use `torch.nn.Linear`. Weights, bias, and 
gate_scores are all declared as raw `nn.Parameter` tensors. Gradients 
flow through both the weight and gate_scores simultaneously — no custom 
backward pass needed.

## Why MLP Accuracy is Moderate

CIFAR-10 is a spatial task. MLPs have no spatial awareness — pixels are 
treated as flat independent numbers. 55% is expected for this architecture. 
The goal of this project is demonstrating learned sparsity, not SOTA accuracy.
