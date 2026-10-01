import torch

UNKNOWN = -1  # '?' symbol: coordinates the attacker declares outside the training string


def binary_kl(p, q):
    """kl(p || q) between Bernoulli(p) and Bernoulli(q), elementwise."""
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


@torch.no_grad()
def estimate_q_hat(model, dg, cluster_id, num_queries=100, device='cpu'):
    """q_hat[s - 1] = clip(E_Z[Q(next bit = 1 | Z)]) for positions s = 1, ..., d - 1.

    Z is a fresh prefix from the cluster. The model is causal, so its prediction at position s - 1
    depends only on tokens 0, ..., s - 1: one batch of full-length samples from the cluster yields
    the queries for every prefix length at once, with the same output as padding each sample after
    position s - 1. Position 0 has an empty prefix and is never predicted.
    """
    model.eval()
    X, _ = dg.generate_fixed_length_samples(n=num_queries, cluster_idx=cluster_id,
                                            prefix_len=dg.dim - 1)
    probs = torch.sigmoid(model(X[:, :-1].to(device))).mean(dim=0).cpu().double()
    return probs.clamp(dg.delta / 2, 1 - dg.delta / 2)


def length_scores(q_hat, delta):
    """F(L) for L = 0, ..., len(q_hat): the first L positions are scored against the two
    probabilities a confident predictor assigns to an observed bit, the rest against 1/2."""
    observed = torch.tensor([1 - delta + delta ** 2 / 2, delta * (1 - delta / 2)], dtype=q_hat.dtype)
    fit_observed = binary_kl(observed[:, None], q_hat[None, :]).min(dim=0).values
    fit_unknown = binary_kl(torch.tensor(0.5, dtype=q_hat.dtype), q_hat)
    zero = torch.zeros(1, dtype=q_hat.dtype)
    head = torch.cat([zero, fit_observed.cumsum(0)])                       # sum over positions <= L
    tail = torch.cat([fit_unknown.flip(0).cumsum(0).flip(0), zero])       # sum over positions > L
    return head + tail


def estimate_length(q_hat, delta, threshold=None, scores=None):
    """Largest L with F(L) <= threshold. threshold=None uses min_L F(L), i.e. the largest minimizer.

    `scores` can pass precomputed length_scores(q_hat, delta) to evaluate many thresholds cheaply.
    """
    F = length_scores(q_hat, delta) if scores is None else scores
    if threshold is None:
        threshold = F.min().item() + 1e-9
    feasible = (F <= threshold).nonzero(as_tuple=True)[0]
    return int(feasible.max()) if len(feasible) else 0


def reconstruct(q_hat, length):
    """Threshold q_hat at 1/2 on the first `length` positions; UNKNOWN afterwards."""
    W_hat = (q_hat >= 0.5).long()
    W_hat[length:] = UNKNOWN
    return W_hat


def attack_singletons(model, dg, X_train, singleton_clusters, num_queries=100, thresholds=(None,),
                      include_known_length=False, device='cpu'):
    """Reconstruct the training string of every singleton cluster.

    Returns {threshold: [accuracy per singleton]} for each length threshold in `thresholds`
    (None = smallest feasible threshold). The model is queried once per singleton; all thresholds
    reuse the same q_hat. Accuracy is the fraction of positions 1, ..., d - 1 on which the
    reconstruction matches the padded training string over {0, 1, ?}, so a wrong length counts
    as errors too. With include_known_length=True, the key 'known_length' holds the accuracy
    when the true length is given (a reference for the bit-recovery step alone).
    """
    accuracies = {t: [] for t in thresholds}
    if include_known_length:
        accuracies['known_length'] = []
    for cluster_id in singleton_clusters.tolist():
        train_idx = (dg.train_cluster_ids == cluster_id).nonzero(as_tuple=True)[0].item()
        W = X_train[train_idx, 1:].long()          # padding (-1) plays the role of '?'
        q_hat = estimate_q_hat(model, dg, cluster_id, num_queries, device)
        scores = length_scores(q_hat, dg.delta)
        for t in thresholds:
            W_hat = reconstruct(q_hat, estimate_length(q_hat, dg.delta, t, scores=scores))
            accuracies[t].append((W_hat == W).double().mean().item())
        if include_known_length:
            W_hat = reconstruct(q_hat, int((W != UNKNOWN).sum()))
            accuracies['known_length'].append((W_hat == W).double().mean().item())
    return accuracies
