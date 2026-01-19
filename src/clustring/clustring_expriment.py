import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_generation import data_generation, HypercubeDataset
from .model import MLP
from .attack import attack_singletons


def run_experiment(d, N, rho, epochs=1000, prob_num=500, bits_per_batch=50, device='cuda'):
    report_freq = epochs//20 + 1
    
    data_gen = data_generation(d)
    train_dataset = HypercubeDataset(data_gen, n=N, rho=rho, fixed_instance=False)
    val_dataset = HypercubeDataset(data_gen, n=500, fixed_instance=True)
    train_loader = DataLoader(train_dataset, batch_size=N, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    print(f"Number of singletons: {len(data_gen.singletons)} / {N}")
    
    model = MLP(d=d, n_classes=N).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    results = {'epochs': [], 'val_acc': [], 'median_acc': [], 'mean_acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch-1)%report_freq ==0:
            print('saving info at epcho',epoch)
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += y_batch.size(0)
            val_acc = total_correct / total_samples if total_samples else 0.0
            num_singletons = len(data_gen.singletons)
            subset_size = num_singletons // 2
            subset_idx = torch.randperm(num_singletons)[:subset_size].tolist()
            subset_singletons = [data_gen.singletons[i] for i in subset_idx]
            accuracies, median_acc = attack_singletons(
                model, data_gen, subset_singletons,
                train_dataset.X, train_dataset.y,
                prob_num=prob_num,
                device=device
            )
            
            results['epochs'].append(epoch)
            results['val_acc'].append(val_acc)
            results['median_acc'].append(median_acc)
            results['mean_acc'].append(np.mean(accuracies))
            print(f"Epoch {epoch}: val_acc={val_acc:.4f}, median_acc={median_acc:.4f}")
    
    return results


def run_multiple_experiments(n_runs=5, d=500, N=50, rho=None, epochs=1000, prob_num=500, device='cuda'):
    
    print(f"Parameters: d={d}, N={N}, rho={rho:.4f}, epochs={epochs}, prob_num={prob_num}")
    
    all_results = []
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*50}")
        results = run_experiment(d, N, rho, epochs, prob_num, device=device)
        all_results.append(results)
    
    return all_results