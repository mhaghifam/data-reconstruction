import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn



def attack_singletons(model, dg, X_train, lengths_train, singleton_clusters, num_queries=20, device='cpu'):
    """
    Attack singleton clusters by querying the model at each prefix length.

    Returns:
        results: dict {cluster_id: {'avg_logits': [...], 'true_train_sample': tensor}}
    """
    model.eval()
    results = {}

    # Take first 3 singletons
    clusters_to_attack = singleton_clusters.tolist()

    with torch.no_grad():
        for cluster_id in clusters_to_attack:
            # Find the training sample for this singleton cluster
            train_idx = (dg.train_cluster_ids == cluster_id).nonzero(as_tuple=True)[0].item()
            true_train_sample = X_train[train_idx]

            # Store average logits for each prefix length
            avg_logits = []

            for prefix_len in range(1, dg.dim):
                # Generate noisy samples with fixed length
                X, Y = dg.generate_fixed_length_samples(
                    n=num_queries,
                    cluster_idx=cluster_id,
                    prefix_len=prefix_len
                )

                # Prepare input (remove last column)
                input_seq = X[:, :-1].to(device)

                # Query model
                logits = model(input_seq)

                # Get logit at position prefix_len - 1
                last_logits = logits[:, prefix_len - 1]

                # Average over queries
                avg_logit = last_logits.mean().item()
                avg_logits.append(avg_logit)

            results[cluster_id] = {
                'avg_logits': avg_logits,
                'true_train_sample': true_train_sample
            }

    return results




def compute_reconstruction(results):
    acc_total = []
    for cluster_id, data in results.items():
        print(cluster_id)
        logits = data['avg_logits']
        true_sample = data['true_train_sample']

        correct = 0
        total = 0

        for i, logit in enumerate(logits):
            true_pos = i + 1  # logits[i] predicts position i+1
            if true_pos >= len(true_sample):
                break
            true_bit = true_sample[true_pos].item()
            if true_bit == -1:  # padding
                break

            pred = 1 if logit > 0 else 0
            if pred == true_bit:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        acc_total.append(acc)
        print(f"Cluster {cluster_id}: {correct}/{total} = {acc:.2%}")
    print(np.median(acc_total))