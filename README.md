# From Memorization to Extraction

Code for the experiments in "From Memorization to Extraction: Provable Training-Data Recovery from Black-Box Models".

## Setup

```bash
pip install -r requirements.txt
```

Scripts are run from the repository root and pick the device automatically (CUDA, Apple MPS, or CPU).

## Running the experiments

Each task has one script. Run with no arguments, it reproduces the figure in the paper. It trains a model several times from scratch and, at regular intervals during training, runs the black-box reconstruction attack on every singleton cluster. It writes:

- `<out>.pdf` and `<out>.png`: validation accuracy and reconstruction accuracy vs. epoch, mean ± standard error over runs
- `<out>_results.json`: the raw per-run numbers

Every script accepts `--n_runs` (number of runs) and `--out` (output prefix); `--help` lists all options.

### Next-token prediction

```bash
python run_ntp.py
```

Model: one-layer causal Transformer. 5 runs of 1500 epochs, evaluated every 150 epochs. Output: `ntp.pdf`.

### Hypercube cluster labeling

```bash
python run_clustring.py
```

Model: one-hidden-layer ReLU MLP with weight decay. 20 runs of 1000 epochs, evaluated every 50 epochs (every 10 during the first 100). Output: `clustering.pdf`. Takes about 15 minutes on an Apple M3 Pro.

## Layout

```
run_ntp.py        next-token prediction experiment
run_clustring.py  hypercube cluster labeling experiment
src/ntp/          data, model, training, attack, experiment
src/clustring/    data, model, attack, experiment
src/utils/        plotting
```
