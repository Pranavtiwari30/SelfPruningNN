# main.py

import torch
import config
from model.prunable_network import SelfPruningNetwork
from utils.data_loader import get_dataloaders
from train.trainer import train
from evaluate.evaluator import report
from utils.visualize import plot_gate_distribution
import os


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    train_loader, test_loader = get_dataloaders()

    all_results = []

    for lambda_val in config.LAMBDA_VALUES:
        # Fresh model for each λ — no shared state between runs
        model = SelfPruningNetwork().to(device)

        # Train
        train(model, train_loader, lambda_val, device)

        # Evaluate
        result = report(model, test_loader, lambda_val, device)
        all_results.append(result)

        # Plot gate distribution for this λ
        plot_gate_distribution(model, lambda_val)

    # Print final comparison table
    print("\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print("="*55)
    for r in all_results:
        print(f"{r['lambda']:<12} {r['accuracy']:>14.2f}% {r['sparsity']:>14.2f}%")
    print("="*55)

    # Save results to txt log
    log_path = os.path.join(config.RESULTS_DIR, "results_log.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}\n")
        f.write("="*55 + "\n")
        for r in all_results:
            f.write(f"{r['lambda']:<12} {r['accuracy']:>14.2f}% {r['sparsity']:>14.2f}%\n")
    print(f"\nResults log saved → {log_path}")


if __name__ == "__main__":
    main()