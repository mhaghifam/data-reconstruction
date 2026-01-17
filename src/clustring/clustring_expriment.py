import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_generation import data_generation, HypercubeDataset
from .model import MLP
from .attack import attack_singletons


def run_experiment(d, N, rho, epochs=1000, prob_num=500, bits_per_batch=50, device='cuda', checkpoints=None):
    if checkpoints is None:
        checkpoints = [10, 50, 100, 200, 500, 1000]
    
    data_gen = data_generation(d)
    train_dataset = HypercubeDataset(data_gen, n=N, rho=rho, fixed_instance=False)
    val_dataset = HypercubeDataset(data_gen, n=2500, fixed_instance=True)
    train_loader = DataLoader(train_dataset, batch_size=N, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    print(f"Number of singletons: {len(data_gen.singletons)} / {N}")
    
    model = MLP(d=d, n_classes=N).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    results = {'epochs': [], 'val_loss': [], 'median_acc': [], 'mean_acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        
        if epoch in checkpoints:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    total_loss += criterion(model(X_batch), y_batch).item()
            val_loss = total_loss / len(val_loader)
            
            accuracies, median_acc = attack_singletons(
                model, data_gen, data_gen.singletons,
                train_dataset.X, train_dataset.y,
                prob_num=prob_num
            )
            
            results['epochs'].append(epoch)
            results['val_loss'].append(val_loss)
            results['median_acc'].append(median_acc)
            results['mean_acc'].append(np.mean(accuracies))
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}, median_acc={median_acc:.4f}")
    
    return results


def run_multiple_experiments(n_runs=5, d=500, N=50, rho=None, epochs=1000, prob_num=500, device='cuda'):
    if rho is None:
        rho = np.sqrt((np.log(N) - np.log(np.log(N))) / d)
    
    print(f"Parameters: d={d}, N={N}, rho={rho:.4f}, epochs={epochs}, prob_num={prob_num}")
    
    all_results = []
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*50}")
        results = run_experiment(d, N, rho, epochs, prob_num, device=device)
        all_results.append(results)
    
    return all_results