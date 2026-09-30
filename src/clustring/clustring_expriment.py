import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_generation import data_generation, HypercubeDataset
from .model import MLP
from .attack import attack_singletons


def evaluate(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = torch.argmax(model(X_batch), dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)
    model.train()
    return total_correct / total_samples if total_samples else 0.0


def run_experiment(d, N, rho, epochs=200, prob_num=50000, eval_every=10, n_val=10000,
                   seed=None, model_kwargs=None, lr=5e-4, weight_decay=0.0, early_until=0,
                   eval_every_early=10, device='cuda'):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    data_gen = data_generation(d)
    train_dataset = HypercubeDataset(data_gen, n=N, rho=rho, fixed_instance=False)
    val_dataset = HypercubeDataset(data_gen, n=n_val, fixed_instance=True)
    train_loader = DataLoader(train_dataset, batch_size=N, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5000, shuffle=False)

    # Only singleton clusters are attacked: they are the ones whose training point is identifiable.
    singletons = data_gen.singletons
    print(f"Number of singletons: {len(singletons)} / {N}")

    model = MLP(d=d, n_classes=N, **(model_kwargs or {})).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    results = {'epochs': [], 'val_acc': [], 'median_acc': [], 'mean_acc': [],
               'per_singleton_acc': [], 'n_singletons': len(singletons)}

    def checkpoint(epoch):
        val_acc = evaluate(model, val_loader, device)
        accuracies, median_acc = attack_singletons(
            model, data_gen, singletons,
            train_dataset.X, train_dataset.y,
            prob_num=prob_num,
            device=device
        )
        results['epochs'].append(epoch)
        results['val_acc'].append(val_acc)
        results['median_acc'].append(float(median_acc))
        results['mean_acc'].append(float(np.mean(accuracies)))
        results['per_singleton_acc'].append(accuracies)
        print(f"Epoch {epoch}: val_acc={val_acc:.4f}, "
              f"recon mean={np.mean(accuracies):.4f}, median={median_acc:.4f}")

    checkpoint(0)
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # Optionally evaluate more often early on, where training moves fastest.
        if epoch % eval_every == 0 or (epoch <= early_until and epoch % eval_every_early == 0):
            checkpoint(epoch)

    return results


def run_multiple_experiments(n_runs=5, d=500, N=50, rho=None, epochs=200, prob_num=50000,
                             eval_every=10, n_val=10000, model_kwargs=None, lr=5e-4,
                             weight_decay=0.0, early_until=0, eval_every_early=10,
                             device='cuda'):

    print(f"Parameters: d={d}, N={N}, rho={rho:.4f}, epochs={epochs}, prob_num={prob_num}")

    all_results = []
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*50}")
        start = time.time()
        results = run_experiment(d, N, rho, epochs, prob_num, eval_every=eval_every,
                                 n_val=n_val, seed=run, model_kwargs=model_kwargs, lr=lr,
                                 weight_decay=weight_decay, early_until=early_until,
                                 eval_every_early=eval_every_early, device=device)
        print(f"Run {run + 1} took {time.time() - start:.1f}s")
        all_results.append(results)

    return all_results
