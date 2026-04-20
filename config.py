# config.py

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "./data"          # CIFAR-10 downloads here automatically
NUM_CLASSES    = 10                # CIFAR-10 has 10 classes, fixed
BATCH_SIZE     = 128               # standard for CIFAR-10, fits most GPUs/CPUs

# ── Architecture ──────────────────────────────────────────────────────────────
INPUT_SIZE     = 3 * 32 * 32       # CIFAR-10: 3 channels, 32x32 pixels = 3072
HIDDEN_SIZES   = [1024, 512, 256, 128]   # 4 hidden layers, funneling down
OUTPUT_SIZE    = NUM_CLASSES       # final layer outputs 10 class scores

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS         = 30
LEARNING_RATE  = 1e-3              # Adam default, works well here
DEVICE         = "cuda"            # will fallback to cpu in main.py if no GPU

# ── Sparsity ──────────────────────────────────────────────────────────────────
LAMBDA_VALUES  = [0.0001, 0.01, 0.1]     # your λ sweep
GATE_THRESHOLD = 1e-2              # gates below this are counted as "pruned"

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR    = "./results"       # plots and logs saved here