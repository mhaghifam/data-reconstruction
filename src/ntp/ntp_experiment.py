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


def compute_median_reconstruction_accuracy(model, dg, X_train, singleton_clusters, num_queries=50, device='cpu'):
    """Run attack on singletons and return median reconstruction accuracy."""
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for cluster_id in singleton_clusters.tolist():
            train_idx = (dg.train_cluster_ids == cluster_id).nonzero(as_tuple=True)[0].item()
            true_sample = X_train[train_idx]
            
            correct = 0
            total = 0
            
            for prefix_len in range(1, dg.dim):
                true_bit = true_sample[prefix_len].item()
                if true_bit == -1:  # padding
                    break
                    
                X, Y = dg.generate_fixed_length_samples(
                    n=num_queries,
                    cluster_idx=cluster_id,
                    prefix_len=prefix_len
                )
                input_seq = X[:, :-1].to(device)
                logits = model(input_seq)
                last_logits = logits[:, prefix_len - 1]
                p_bar = torch.sigmoid(last_logits).mean().item()
                p_bar = max(min(p_bar, 1-1e-9), 1e-9)
                avg_logit = math.log(p_bar / (1 - p_bar))
                
                pred = 1 if avg_logit > 0 else 0
                
                if pred == true_bit:
                    correct += 1
                total += 1
            
            if total > 0:
                accuracies.append(correct / total)
    
    return np.median(accuracies) if accuracies else 0.0


def train_with_tracking(model, train_loader, val_loader, dg, X_train, singleton_clusters,
                        num_epochs, device, eval_every=50, num_attack_queries=50):
    """Train model while tracking val loss and reconstruction accuracy."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=num_epochs * len(train_loader) // 3,
        gamma=0.3
    )
    
    iterations = []
    val_acces = []
    median_accuracies = []
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler)
        
        # Evaluate at intervals
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            _, val_acc = evaluate_last_position(model, val_loader, device)
            median_acc = compute_median_reconstruction_accuracy(
                model, dg, X_train, singleton_clusters,
                num_queries=num_attack_queries, device=device
            )
            
            iterations.append(epoch + 1)
            val_acces.append(val_acc)
            median_accuracies.append(median_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Val acc: {val_acc:.4f} | Recon Acc: {median_acc:.4f}")
    
    return model, iterations, val_acces, median_accuracies


def run_multiple_trials(num_trials, N, d, delta, n_train, n_val, batch_size,
                        num_epochs, device, eval_every=50, num_attack_queries=50):
    """Run experiment multiple times with different seeds."""
    
    all_val_acces = []
    all_median_accs = []
    iterations = None
    
    for trial in range(num_trials):
        print(f"\n{'='*50}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*50}")
        
    
        
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
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = TransformerNextToken(
            embed_dim=256,
            hidden_dim=512,
            num_layers=1,
            num_heads=4,
            max_len=d + 10,
            pad_value=-1
        ).to(device)
        
        # Train
        _, iters, val_acces, median_accs = train_with_tracking(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            dg=dg,
            X_train=X_train,
            singleton_clusters=singleton,
            num_epochs=num_epochs,
            device=device,
            eval_every=eval_every,
            num_attack_queries=num_attack_queries
        )
        
        iterations = iters
        all_val_acces.append(val_acces)
        all_median_accs.append(median_accs)
    
    return iterations, np.array(all_val_acces), np.array(all_median_accs)


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
    ax2.set_ylim(0.5, 1)
    
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
