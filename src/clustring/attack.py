import torch
import numpy as np
from tqdm import tqdm

def attack_singletons(model, data_gen, singletons, X_train, y_train, prob_num=50, device='cuda'):
    model.eval()
    model = model.to(device)

    accuracies = []

    for cluster_idx in tqdm(singletons, desc="Attacking singletons"):
        d = data_gen.dim

        unfixed = torch.where(data_gen.fixed_loc[cluster_idx] == 0)[0]
        predicted_bits = torch.zeros(len(unfixed))

        for j, i in enumerate(unfixed):
            i = i.item()

            X = torch.distributions.Bernoulli(probs=0.5).sample((2 * prob_num, d))

            mask = data_gen.fixed_loc[cluster_idx] == 1
            X[:, mask] = data_gen.fixed_vals[cluster_idx]

            X[:prob_num, i] = 0
            X[prob_num:, i] = 1

            with torch.no_grad():
                logits = model(X.float().to(device))
                lj = logits[:, cluster_idx]
                other = torch.cat([logits[:, :cluster_idx], logits[:, cluster_idx+1:]], dim=1)
                scores = lj - torch.logsumexp(other, dim=1)

                logit_0 = scores[:prob_num].median()
                logit_1 = scores[prob_num:].median()
                logit_diff = logit_1 - logit_0

                predicted_bits[j] = 1 if logit_diff.item() > 0 else 0

        idx = (y_train == cluster_idx).nonzero(as_tuple=True)[0][0]
        ground_truth = X_train[idx, unfixed]

        acc_c = (predicted_bits == ground_truth).float().mean().item()
        accuracies.append(acc_c)
        print(accuracies)

    return accuracies, np.median(accuracies)
