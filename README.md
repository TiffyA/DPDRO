# Differentially Private ASCDRO Experiments

This workspace hosts two closely related implementations of the ASCDRO
(Double Spider DRO) training algorithm together with experiment-specific entry
points.

## Code organisations

- `Baseline/SGDA` & `Baseline/Diff`: baseline methods; pipelines and usage stay
  the same as before.
- `dro1_new/`: refreshed version of the original `dro1` project with
  script-configurable hyper-parameters. Use this in place of the deprecated
  `dro1/` directory, which is kept only for archival purposes.
- `dro_alg2/`: current ASCDRO stack (formerly `dro_new`). Contains the
  production trainer, datasets, models, and training scripts used for the latest
  experiments.
- `dro_alg1/`: legacy ASCDRO codebase kept for reference and comparison with
  the updated implementation.
- `imagenet/`: ImageNet-LT manifests and the ASCDRO training script that builds
  upon `dro_alg2`.

## Repository layout

| Path | Description |
|------|-------------|
| `dro_alg2/ascdro/algorithms.py` | ASCDRO trainer with Private SpiderBoost (DSDRO). |
| `dro_alg2/ascdro/datasets.py` | Dataset utilities for CIFAR10-ST and ImageNet. |
| `dro_alg2/ascdro/models.py` | ResNet builders for CIFAR and ImageNet. |
| `dro_alg2/training/train_cifar10_st_ascdro.py` | CIFAR10-ST ASCDRO training entry point. |
| `dro_alg2/scripts/run_cifar10_st_ascdro.sh` | Bash helper for CIFAR10-ST training. |
| `dro_alg1/` | Legacy implementation mirroring the structure of `dro_alg2`. |
| `imagenet/train_imagenet_lt.py` | ImageNet-LT ASCDRO/DSDRO training script. |
| `imagenet/ImageNet_LT/*.txt` | Long-tailed train/val manifests. |

## Running training

### ImageNet-LT (ASCDRO)

```bash
python imagenet/train_imagenet_lt.py \
  --epsilon 8 \
  --delta 0.1 \
  --batch-size 192 \
  --num-workers 8 \
  --eta 0.05 \
  --rho 0.5 \
  --lambda0 0.1 \
  --beta 0.5
```

Artifacts are written to `imagenet/runs/<experiment>/`. Use `--help` for the
complete list of arguments (image size, debug samples, warm-up controls, etc.).

### CIFAR10-ST (ASCDRO)

```bash
python dro_alg2/training/train_cifar10_st_ascdro.py \
  --data-root /path/to/cifar10_st \
  --output-dir dro_alg2/runs/cifar10_st_demo \
  --epsilon 1
```

or run the helper wrapper:

```bash
./dro_alg2/scripts/run_cifar10_st_ascdro.sh 1
```

## Environment

Create the required Conda environment with:

```
conda env create --name atnew --file envs/atnew-environment.yml
```

After activation (`conda activate atnew`), the training scripts above are ready
to use.
