"""
experiment.py - Run multiple trials and plot learning vs memorization
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

from src.ntp.data_generation import DataGeneration, NextTokenDataset
from src.ntp.model import TransformerNextToken
from src.ntp.training import train_epoch, evaluate_last_position
from src.ntp.attacker import attack_singletons


def compute_median_reconstruction_accuracy(model, dg, X_train, singleton_clusters, num_queries=50,
                                           length_threshold=None, threshold_sweep=(), device='cpu'):
    """Run the attack on every singleton cluster.

    Returns the median reconstruction accuracy at `length_threshold`, and a dict with the median
    accuracy at every threshold in `threshold_sweep` and with the true length ('known_length'),
    all computed from the same queries.
    """
    thresholds = [length_threshold] + [t for t in threshold_sweep if t != length_threshold]
    accuracies = attack_singletons(model, dg, X_train, singleton_clusters,
                                   num_queries=num_queries, thresholds=thresholds,
                                   include_known_length=True, device=device)
    medians = {t: float(np.median(a)) if a else 0.0 for t, a in accuracies.items()}
    sweep = {t: medians[t] for t in list(threshold_sweep) + ['known_length']}
    return medians[length_threshold], sweep


def train_with_tracking(model, train_loader, val_loader, dg, X_train, singleton_clusters,
                        num_epochs, device, eval_every=50, num_attack_queries=50, lr=1e-3,
                        weight_decay=0.0, schedule='step', warmup_steps=0, length_threshold=None,
                        threshold_sweep=(), unknown_half=False):
    """Train model while tracking val loss and reconstruction accuracy.

    schedule='step' multiplies the learning rate by 0.3 after each third of training;
    schedule='cosine' warms up linearly for warmup_steps and then decays with a cosine to zero.
    """
    total_steps = num_epochs * len(train_loader)
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    if schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=total_steps // 3,
            gamma=0.3
        )
    elif schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
            * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
        )
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    
    iterations = []
    val_acces = []
    median_accuracies = []
    sweep_accuracies = []
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler,
                                            unknown_half=unknown_half)
        
        # Evaluate at intervals
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            _, val_acc = evaluate_last_position(model, val_loader, device)
            median_acc, sweep_acc = compute_median_reconstruction_accuracy(
                model, dg, X_train, singleton_clusters,
                num_queries=num_attack_queries, length_threshold=length_threshold,
                threshold_sweep=threshold_sweep, device=device
            )
            
            iterations.append(epoch + 1)
            val_acces.append(val_acc)
            median_accuracies.append(median_acc)
            sweep_accuracies.append(sweep_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Val acc: {val_acc:.4f} | Recon Acc: {median_acc:.4f}")
    
    return model, iterations, val_acces, median_accuracies, sweep_accuracies


def run_multiple_trials(num_trials, N, d, delta, n_train, n_val, batch_size,
                        num_epochs, device, eval_every=50, num_attack_queries=50,
                        num_layers=1, dropout=0.0, lr=1e-3, weight_decay=0.0,
                        schedule='step', warmup_steps=0, length_threshold=None,
                        threshold_sweep=(), unknown_half=False):
    """Run experiment multiple times with different seeds."""
    
    all_val_acces = []
    all_median_accs = []
    all_sweeps = []
    iterations = None
    
    for trial in range(num_trials):
        print(f"\n{'='*50}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*50}")
        torch.manual_seed(trial)
        np.random.seed(trial)

        # Generate data
        dg = DataGeneration(N=N, d=d, delta=delta)
        X_train, lengths_train, singleton = dg.generate_samples(n=n_train)
        train_cluster_ids = dg.train_cluster_ids.clone()
        X_val, lengths_val, _ = dg.generate_samples(n=n_val)
        dg.train_cluster_ids = train_cluster_ids
        
        # Create dataloaders
        train_dataset = NextTokenDataset(X_train, lengths_train)
        val_dataset = NextTokenDataset(X_val, lengths_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
        
        # Create model
        model = TransformerNextToken(
            embed_dim=256,
            hidden_dim=512,
            num_layers=num_layers,
            num_heads=4,
            max_len=d + 10,
            pad_value=-1,
            dropout=dropout
        ).to(device)
        
        # Train
        _, iters, val_acces, median_accs, sweeps = train_with_tracking(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            dg=dg,
            X_train=X_train,
            singleton_clusters=singleton,
            num_epochs=num_epochs,
            device=device,
            eval_every=eval_every,
            num_attack_queries=num_attack_queries,
            lr=lr,
            weight_decay=weight_decay,
            schedule=schedule,
            warmup_steps=warmup_steps,
            length_threshold=length_threshold,
            threshold_sweep=threshold_sweep,
            unknown_half=unknown_half
        )
        
        iterations = iters
        all_val_acces.append(val_acces)
        all_median_accs.append(median_accs)
        all_sweeps.append(sweeps)

    return iterations, np.array(all_val_acces), np.array(all_median_accs), all_sweeps


def cross_validated_accuracy(acc_by_threshold):
    """Pick the attack's length threshold without tuning it on the run it is reported for.

    acc_by_threshold maps each candidate threshold to an array [run, checkpoint] of median
    accuracies. For every run and checkpoint, the threshold that maximizes the mean accuracy over
    the other runs is selected, and this run's accuracy at that threshold is reported.
    Returns (accuracies [run, checkpoint], chosen thresholds [run, checkpoint]).
    """
    candidates = list(acc_by_threshold)
    A = np.stack([np.asarray(acc_by_threshold[t], dtype=float) for t in candidates])
    num_runs, num_checkpoints = A.shape[1], A.shape[2]
    accuracies = np.zeros((num_runs, num_checkpoints))
    chosen = [[None] * num_checkpoints for _ in range(num_runs)]
    for k in range(num_runs):
        best = np.delete(A, k, axis=1).mean(axis=1).argmax(axis=0)
        accuracies[k] = A[best, k, np.arange(num_checkpoints)]
        chosen[k] = [candidates[b] for b in best]
    return accuracies, chosen


def plot_learning_vs_memorization(iterations, all_val_acces, all_median_accs, save_path=None):
    """Plot mean ± std of val acc and reconstruction accuracy."""
    num_trials = all_val_acces.shape[0]
    val_acc_mean = np.mean(all_val_acces, axis=0)
    val_acc_std = np.std(all_val_acces, axis=0)/np.sqrt(num_trials)
    acc_mean = np.mean(all_median_accs, axis=0)
    acc_std = np.std(all_median_accs, axis=0)/np.sqrt(num_trials)
    iterations = np.array(iterations)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    tick_labelsize = 14

    # Left axis: validation loss
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Validation Accuracy', color=color1, fontsize=16)
    line1, = ax1.plot(
        iterations,
        val_acc_mean,
        color=color1,
        linewidth=2,
        marker='*',
        markersize=8,
        label='Validation Accuracy'
    )
    ax1.fill_between(iterations, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                     color=color1, alpha=0.2)
    ax1.tick_params(axis='x', labelsize=tick_labelsize)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=tick_labelsize)
    ax1.set_ylim(0.5, 1)
    
    # Right axis: reconstruction accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Reconstruction Accuracy', color=color2, fontsize=16)
    line2, = ax2.plot(
        iterations,
        acc_mean,
        color=color2,
        linewidth=2,
        marker='o',
        markersize=5,
        label='Reconstruction Accuracy'
    )
    ax2.fill_between(iterations, acc_mean - acc_std, acc_mean + acc_std,
                     color=color2, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=tick_labelsize)
    ax2.set_ylim(min(0.5, np.floor((acc_mean - acc_std).min() / 0.05) * 0.05), 1)
    
    # Random baseline
    # ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=16)
    
    
    plt.title(f'Next Token Prediction Task', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    return fig
