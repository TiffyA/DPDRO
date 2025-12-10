# DRO Baselines and MIA Experiments

This repo bundles three related pieces:

- `dro1_new`: DP Double-SPIDER implementation for a DRO objective on an imbalanced CIFAR-10 (ResNet20 baseline + optional DP run).
- `dro2_new`: Recursive Spider DRO (RS-DRO) with DP noise for CIFAR10-ST, including a convenience launcher.
- `MIA`: end-to-end membership inference attacks (confidence/loss + LiRA) targeting the RS-DRO checkpoints.

All paths below are relative to the repo root (`/home/Arnold/Code`).

## Requirements

- Python 3.10+, PyTorch, torchvision.
- Extra packages for MIA: `numpy`, `scikit-learn`, `matplotlib`.
- CIFAR-10 is downloaded automatically into `./data` on first use.

## One-command runner: `run_all_projects.sh`

Located at the repo root. It sequentially runs Baseline/SGDA, Baseline/Diff, `dro1_new`, and `dro2_new`. By default it sweeps `epsilon` over `5 10`, runs each setting 5 times (random seeds auto-generated), and writes outputs under `multi_results/`.

### Usage

```bash
bash run_all_projects.sh
```

Optional environment variables:

- `EPS_LIST="0.2 0.5 1"` – override the epsilon list.
- `REPEAT_COUNT=3` – number of repeats per epsilon.
- `CIFAR_TORCH_ROOT=/path/to/data` – data root (default `Baseline/SGDA/data`).
- `RESULTS_DIR=/path/to/out` – results root (default `multi_results/`).

Example:

```bash
EPS_LIST="0.5 1 5" REPEAT_COUNT=3 RESULTS_DIR=/tmp/multi_out bash run_all_projects.sh
```

Outputs:

- Per-algorithm/epsilon/run folders and logs, e.g., `multi_results/dro2_new_eps5_run1/train.log`.
- Aggregated `multi_results/summary.txt` with average accuracies per epsilon (parsed from each algorithm’s logs/summary).

## `dro1_new` – DP Double-SPIDER baseline

Implements the Double-SPIDER algorithm with DP noise on an imbalanced CIFAR-10 split (classes 0–4 keep only the last 100 samples; classes 5–9 keep all 5k). Uses a CIFAR-style ResNet20.

### Run standard (non-DP) baseline

```bash
python dro1_new/algorithm.py \
  --data-root ./data \
  --train-batch-size 128 \
  --test-batch-size 100 \
  --baseline-epochs 20
```

This trains with SGD + cosine schedule and reports test accuracy each epoch.

### Run DP Double-SPIDER

```bash
python dro1_new/algorithm.py \
  --data-root ./data \
  --run-dp \
  --skip-baseline \
  --epsilon 4.0 \
  --delta-exponent 1.1 \
  --T 100
```

Key flags:

- `--run-dp`: enable the DP Double-SPIDER loop (otherwise only the baseline runs).
- `--skip-baseline`: skip the non-DP warm-up if you only want the DP run.
- `--epsilon`, `--delta`/`--delta-exponent`: privacy parameters (default δ = n^-1.1).
- `--T`, `--q`, `--C1..C4`, `--N1..N4`: control iteration count, refresh period, clipping, and batch sizes (defaults follow the closed-form in the script).
- Data/config: `--data-root`, `--train-batch-size`, `--test-batch-size`, `--num-workers`.

The script prints the computed hyperparameters, trains, and evaluates on the CIFAR-10 test split.

## `dro2_new` – RS-DRO baseline (DP)

Implements Recursive Spider DRO with DP noise. The training set is CIFAR10-ST (n=25,500, same imbalance pattern as above). A short ERM pretraining stage is enabled by default to stabilise training.

### Quick start (recommended)

```bash
bash dro2_new/new.sh
```

Environment variables override defaults, e.g.:

```bash
DEVICE=cpu WIDTH_FACTOR=0.5 PRETRAIN_EPOCHS=20 MAX_T=10 RESULTS_DIR=./runs/rsdro bash dro2_new/new.sh
```

Notable env vars (all optional): `DATA_ROOT`, `RESULTS_DIR`, `DEVICE`, `RHO`, `EPSILON`, `G_CONST`, `L_CONST`, `C_CONST`, `LAMBDA0`, `ETA_T_SQUARED`, `WIDTH_FACTOR`, `PRETRAIN_*`, `MAX_T`, `MAX_B1`, `MAX_B2`, `MAX_Q`.

### Manual invocation

```bash
python dro2_new/train_rsdro.py \
  --data-root ./data \
  --results-dir ./runs/rsdro \
  --device cuda \
  --rho 0.1 --epsilon 4.0 --G 1.0 --L 1.0 --c 10.0 \
  --eta-t-squared 1e-4 \
  --width-factor 0.5 \
  --pretrain-epochs 12 --pretrain-balanced \
  --max-T 5 --max-b1 4096 --max-b2 64 --max-q 5 \
  --save-model
```

What happens:

1. Builds CIFAR10-ST with/without augmentation and estimates Ψ₀ via a warm-up.
2. Checks the sample-size condition for the provided constants.
3. Computes `(T, q, b1, b2, η, β_t, σ₁, σ₂, ĥσ₂, σ_{s_t})` and prints them.
4. Runs RS-DRO, returns a random iterate `(w_τ, λ_τ)`, evaluates on CIFAR-10 test set.
5. Writes `runs/rsdro/summary.json` and, when `--save-model` is set, `runs/rsdro/rsdro_resnet20.pt` (state_dict under the `model` key plus `lambda`).

If you change `--width-factor` during training, reuse the same value for evaluation and MIA.

## `MIA` – membership inference baselines on RS-DRO

This folder reuses the `dro2_new` ResNet20 to run confidence/loss attacks and LiRA against an RS-DRO checkpoint.

### Dependencies

```bash
pip install numpy scikit-learn matplotlib
```

### Prepare splits (optional; auto-run by the scripts)

```bash
python MIA/data_prep.py --data-root ./data --split-root MIA/splits
```

Outputs `train_st.npz`, `public.npz`, `eval_members.npz`, `eval_nonmembers.npz`, `calibration_nonmembers.npz`, and metadata under `MIA/splits/`.

### Run confidence/loss attacks

Assuming you trained and saved `runs/rsdro/rsdro_resnet20.pt`:

```bash
bash MIA/run_mia_rsdro.sh runs/rsdro/rsdro_resnet20.pt 0.5
```

Or directly:

```bash
python MIA/run_attacks.py \
  --checkpoint runs/rsdro/rsdro_resnet20.pt \
  --width-factor 0.5 \
  --data-root ./data \
  --split-root MIA/splits \
  --output-dir MIA/outputs/rsdro \
  --member-count 10000 \
  --calibration-count 5000 \
  --per-class-thresholds
```

Outputs: `attack_metrics.json` plus ROC plots (`roc_confidence.png`, `roc_loss.png`) in the chosen output dir. Thresholds are calibrated on non-member scores to hit FPR targets (1% / 0.1%); per-class thresholds are optional.

### LiRA (shadow-model attack, optional)

1) Sample shadow splits:

```bash
python MIA/make_shadow_manifest.py \
  --public-npz MIA/splits/public.npz \
  --output MIA/shadows/manifest.json \
  --num-shadows 16 \
  --train-fraction 0.5 \
  --mode proportional \
  --checkpoint-template MIA/shadows/shadow_{i:03d}/checkpoint.pt
```

2) Train shadow models (non-DP ResNet20):

```bash
python MIA/train_shadows.py --manifest MIA/shadows/manifest.json --epochs 160 --batch-size 128
```

3) Score with LiRA:

```bash
python MIA/run_lira.py \
  --target-checkpoint runs/rsdro/rsdro_resnet20.pt \
  --shadow-manifest MIA/shadows/manifest.json \
  --output-dir MIA/outputs/lira_dp
```

Outputs: `lira_metrics.json` and `lira_scores.npz` in the chosen directory.

### Notes

- The default shadow pool (`public.npz`) covers classes 0–4 only; use a different pool if you need all classes in shadows.
- Always match `--width-factor` between training and attack scripts so the model loads correctly.
