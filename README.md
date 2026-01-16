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


### Hypercube Cluster Labeling

**Data Generation:**
- N clusters, each defined by ~ρd fixed bit positions with random values
- Training: n samples from uniform mixture over clusters
- Singletons: clusters with exactly one training sample (~n/e for uniform mixture)

**Attack:**
- Attacker knows: fixed bit locations and values for each cluster
- Goal: recover unfixed bits of singleton training examples
- Method: For each unfixed bit i, generate probes with bit i=0 and i=1, compare model logits

**Key Insight:** 
- Model achieves 100% train accuracy → must memorize singleton examples
- Each unfixed bit contributes ~0.016 to logit
- Need sufficient probes to overcome noise from other random bits


