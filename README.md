# Black-Box Data Reconstruction via List Decoding

Code for the experiments in "From Memorization to Extraction: Provable Training-Data Recovery from Black-Box Models".

## Setup

```bash
pip install -r requirements.txt
```

Both experiments pick the device automatically (CUDA, Apple MPS, or CPU) and are run from the repository root.

## Next-Token Prediction

```bash
python run_ntp.py
```

Trains a one-layer causal Transformer on the clustered next-token prediction task (5 trials) and runs the black-box reconstruction attack on singleton clusters. Saves `ntp2.pdf`: validation accuracy and reconstruction accuracy vs. epoch.

## Hypercube Cluster Labeling

```bash
python run_clustring.py --activation relu --hidden 1000 --center_input --lr 1e-3 --weight_decay 1e-3 \
    --epochs 1000 --eval_every 50 --early_until 100 --out clustering_regularized
```

Trains an MLP on the hypercube clustering task (20 runs; about 15 minutes on an Apple M3 Pro). At each evaluation it runs the black-box correlation attack on every singleton cluster. Saves:
- `<out>.pdf` and `<out>.png`: validation accuracy and reconstruction accuracy vs. epoch, mean ± standard error over runs.
- `<out>_results.json`: the raw per-run numbers.

With no flags, `python run_clustring.py` trains a one-hidden-layer sigmoid MLP (500 units, Adam with lr 5e-4, 200 epochs). `python run_clustring.py --help` lists all options.

## Layout

```
src/ntp/        next-token prediction: data, model, training, attack, experiment
src/clustring/  hypercube clustering: data, model, attack, experiment
src/utils/      plotting
```
