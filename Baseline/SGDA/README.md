# Differentially Private SGDA for Imbalanced AUC Maximization

Implementation of the DP-SGDA baseline described in *Improved Rates of Differentially Private Nonconvex-Strongly-Concave Minimax* (AAAI 2025). The project mirrors the `Diff` setup but executes Algorithm 1 (DP-SGDA) with the same data pipeline and evaluation.

## Dependencies

- Python 3.8+
- PyTorch
- TorchVision
- LibAUC
- NumPy
- Pillow

Install them in a virtual environment:

```bash
pip install torch torchvision libauc numpy pillow
```

## Usage

```bash
python main.py \
  --epsilon 0.5 \
  --delta 1e-4 \
  --lr_w 0.2 \
  --lr_v 0.2 \
  --clip_w 1.0 \
  --clip_v 1.0 \
  --total_epochs 80
```

The script will:

1. Construct the same imbalanced MNIST dataset as `Baseline/Diff`.
2. Train the LibAUC AUC surrogate with DP-SGDA.
3. Print the test AUC after each epoch and report the best score at the end.

Noise scales are computed from the privacy budget, dataset size, and total optimisation steps following the paper's description.
