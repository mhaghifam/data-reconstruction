import torch
import numpy as np
from tqdm import tqdm

def attack_singletons_fast(model, data_gen, singletons, X_train, y_train, prob_num=500, device='cuda', bits_per_batch=50):
    model.eval()
    model = model.to(device)

    accuracies = []

    for cluster_idx in tqdm(singletons, desc="Attacking singletons"):
        d = data_gen.dim

        unfixed = torch.where(data_gen.fixed_loc[cluster_idx] == 0)[0]
        n_unfixed = len(unfixed)
        
        mask = data_gen.fixed_loc[cluster_idx] == 1
        fixed_vals = data_gen.fixed_vals[cluster_idx]
        
        predicted_bits = torch.zeros(n_unfixed)
        
        for batch_start in range(0, n_unfixed, bits_per_batch):
            batch_end = min(batch_start + bits_per_batch, n_unfixed)
            batch_unfixed = unfixed[batch_start:batch_end]
            batch_len = len(batch_unfixed)
            
            X = torch.distributions.Bernoulli(probs=0.5).sample((batch_len * 2 * prob_num, d))
            X[:, mask] = fixed_vals
            
            for j, i in enumerate(batch_unfixed):
                i = i.item()
                start_0 = j * 2 * prob_num
                start_1 = start_0 + prob_num
                X[start_0:start_0 + prob_num, i] = 0
                X[start_1:start_1 + prob_num, i] = 1
            
            with torch.no_grad():
                scores = model(X.float().to(device))[:, cluster_idx]
            
            for j in range(batch_len):
                start_0 = j * 2 * prob_num
                start_1 = start_0 + prob_num
                logit_0 = scores[start_0:start_0 + prob_num].mean()
                logit_1 = scores[start_1:start_1 + prob_num].mean()
                predicted_bits[batch_start + j] = 1 if logit_1 > logit_0 else 0
        
        idx = (y_train == cluster_idx).nonzero(as_tuple=True)[0][0]
        ground_truth = X_train[idx, unfixed]

        acc_c = (predicted_bits == ground_truth).float().mean().item()
        accuracies.append(acc_c)

    return accuracies, np.median(accuracies)