# Self-Pruning Neural Network — Report

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

Each weight in the network has a learnable `gate_score` parameter. During the forward pass, this score is passed through a sigmoid function to produce a gate value bounded in (0, 1):

```
gate          = sigmoid(gate_score)
pruned_weight = weight * gate
output        = input @ pruned_weight.T + bias
```

The total training loss is:

```
Total Loss = CrossEntropyLoss + λ * Σ(gate values)
```

The L1 penalty — the sum of all gate values — creates a **constant gradient of magnitude λ** pushing every gate toward zero, regardless of how small the gate already is. This is the fundamental difference from L2 regularization, which produces a gradient proportional to the weight's current magnitude and therefore never fully drives values to exactly zero.

Because the sigmoid function saturates near 0 and 1, gates naturally "snap" to 0 when the sparsity pressure outweighs the classification benefit of keeping a connection alive. The result: connections that genuinely contribute to classification survive, while redundant connections get zeroed out. The network learns to prune itself — no post-training step required.

---

## 2. Results

All models trained for **30 epochs** on CIFAR-10 (50,000 training images, 10,000 test images) using the Adam optimizer (lr = 1e-3). Architecture: feedforward MLP with hidden layers [1024, 512, 256, 128]. Total gate parameters: **3,835,136**.

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|:------------:|:-----------------:|
| 0.0001 | 55.14%       | 70.17%            |
| 0.01   | 44.98%       | 99.98%            |
| 0.1    | 21.65%       | 100.00%           |

---

## 3. Analysis of the λ Trade-off

**λ = 0.0001 — Low Pressure (Best Model)**

The sparsity penalty is weak relative to the classification loss. The network retains approximately 30% of its connections and achieves the best test accuracy of **55.14%**. The gate distribution shows a clear bimodal shape — a large spike near 0 (pruned connections) and a distinct cluster of surviving gates away from 0. This is the hallmark of successful learned sparsity. The network found a meaningful sparse substructure that still performs well on the task.

**λ = 0.01 — Medium Pressure**

The sparsity term begins to dominate the classification objective. Nearly every gate (99.98%) is driven to zero and accuracy drops to **44.98%**. The network is over-pruned — too many connections that were contributing to classification are being zeroed out by the aggressive penalty. The gate distribution collapses almost entirely to a spike at 0.

**λ = 0.1 — Aggressive Pressure**

The sparsity penalty completely overwhelms the classification objective from the very first epoch. 100% of gates are pruned and accuracy falls to **21.65%** — close to random guessing on a 10-class problem (~10% would be pure random). The network loses all capacity to learn meaningful representations. The gate distribution is a single spike at exactly 0.

**Key Takeaway**

λ = 0.0001 is the optimal operating point for this architecture. It achieves meaningful compression — **70% of all 3.8M weights effectively removed** — while retaining reasonable classification performance. This demonstrates that the self-pruning mechanism works correctly: it identifies and removes redundant connections without destroying the network's ability to generalise.

---

## 4. Gate Distribution Plot

Gate distribution plots for all three λ values are saved in `./results/` as PNG files.

For the best model (λ = 0.0001), the histogram displays:
- A **large spike near 0** — the majority of gates the network determined were unnecessary
- A **smaller cluster of values away from 0** — surviving connections that carry genuine classification signal

This bimodal distribution confirms the L1 + sigmoid mechanism is working as intended. The network is not uniformly shrinking all gates (as L2 would do) — it is making hard binary-like decisions: keep or prune.

---

## 5. Implementation Note

`PrunableLinear` does not use `torch.nn.Linear`. Weights, bias, and gate_scores are declared as raw `nn.Parameter` tensors, with Kaiming uniform initialisation applied manually to the weight tensor — matching the initialisation strategy of `nn.Linear`. Gradients flow through both `weight` and `gate_scores` simultaneously via standard autograd. No custom backward pass was required.

```python
self.weight      = nn.Parameter(torch.empty(out_features, in_features))
self.bias        = nn.Parameter(torch.zeros(out_features))
self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
```

The gate_scores are initialised to zero so that `sigmoid(0) = 0.5` — every gate starts half-open, giving the optimiser a balanced starting point with gradient signal in both directions.