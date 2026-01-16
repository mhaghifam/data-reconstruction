This repo contains the empirical results presented in "Data Reconstruction through the Lens of List Decoding" (PDF Coming Soon).


# Next Token Prediction

This experiment demonstrates that learning implies memorization in the next token prediction setting. We show that a black-box attacker can reconstruct training data from singleton clusters by querying a trained model based on the following paper: [[https://arxiv.org/abs/2012.06421](https://arxiv.org/abs/2012.06421)]

## Problem Setup

We consider a clustered binary sequence prediction task:

1. **Data Generation**: 
   - Sample N cluster centers uniformly from {0,1}^d
   - Each training sample: pick a random cluster, apply BSC(δ/2) noise, and truncate to a random length T
   - The model sees prefix Z = (z_1, ..., z_{T-1}) and predicts the next bit Y = z_T

2. **Singleton Clusters**: When n = N training samples are drawn, roughly 1/e fraction of clusters contain exactly one sample. These are vulnerable to reconstruction.

3. **Learning Objective**: Train a model to minimize cross-entropy loss for next token prediction.

## Attack Strategy

The attacker has black-box query access to the trained model and knows which cluster to attack (but not the training sample itself).

For each singleton cluster j:
1. Query the model at every prefix length t ∈ {1, ..., d-1}
2. Average predictions over multiple noisy queries from cluster j
3. Reconstruct bits by thresholding: predict 1 if average logit > 0, else 0

## Model Architecture

We use a causal Transformer decoder:

| Component | Details |
|-----------|---------|
| Token Embedding | 3 tokens: {0, 1, pad} → embed_dim |
| Positional Encoding | Sinusoidal |
| Transformer Layers | 1 layer, 4 heads, causal masking |
| Output Head | Linear → 1 (binary logit) |

Default hyperparameters: embed_dim=256, hidden_dim=800, trained for 2000 epochs.

## Output Plot

The experiment produces a dual-axis plot showing **as the model learns to predict well, it simultaneously memorizes singleton training samples**, enabling reconstruction.

## Usage

```bash
python run_ntp.py
```

## File Structure

```
src/ntp/
├── data_generation.py   # DataGeneration class, NextTokenDataset
├── model.py             # TransformerNextToken architecture
├── training.py          # Training and evaluation loops
├── attacker.py          # Attack functions
└── ntp_experiment.py    # Multi-trial experiment and plotting
```


# Hypercube Cluster Labeling

This experiment demonstrates that learning implies memorization in the hypercube cluster labeling setting. We show that a black-box attacker can reconstruct training data from singleton clusters by querying a trained model based on the following paper: [https://arxiv.org/abs/2012.06421](https://arxiv.org/abs/2012.06421)

## Problem Setup

We consider a multiclass classification task over binary hypercube clusters:

1. **Data Generation**: 
   - Sample N cluster centers: for each cluster j, independently mark each bit as "fixed" with probability ρ, then assign random values to fixed bits
   - Each training sample: pick a random cluster j, copy fixed bits, fill unfixed bits uniformly at random
   - Label is the cluster index j ∈ {1, ..., N}



2. **Learning Objective**: Train a model to minimize cross-entropy loss for multiclass classification.

## Attack Strategy

The attacker has black-box query access to the trained model and knows the fixed bit locations/values for each cluster (but not the unfixed bits of the training sample).


## Model Architecture

We use a 3-layer MLP:

| Component | Details |
|-----------|---------|
| Input | d-dimensional binary vector |
| Hidden Layers | 2 layers, 1500 units each, ReLU |
| Output | N-way softmax classification |

Default hyperparameters: d=500, N=50, trained for 1000 epochs with Adam optimizer.



## File Structure
```
src/clustring/
├── data_generation.py       # data_generation class, HypercubeDataset
├── model.py                 # MLP architecture
├── train.py                 # Training and evaluation loops
├── attack.py                # Attack functions
└── clustring_expriment.py   # Multi-trial experiment and plotting
```

