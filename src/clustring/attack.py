import torch
import numpy as np


@torch.no_grad()
def estimate_correlation(model, data_gen, cluster_idx, prob_num=50000, chunk_size=25000,
                         center=True, generator=None, device='cuda'):
    """Monte-Carlo estimate of Gamma_i(t) = E_{Z ~ P_{theta,i}}[Q(i | Z) Z(t)] for every t in [d].

    theta (the cluster signatures in data_gen) and the model are held fixed; the expectation is
    replaced by an average over prob_num fresh draws Z ~ P_{theta,i}. Features are stored in
    {0,1}; the correlation uses Z(t) = 2 X(t) - 1 in {-1,+1}.

    center=True uses E[(Q - E[Q]) Z(t)], which equals Gamma_i(t) on the unfixed coordinates
    (E[Z(t)] = 0 there) but has lower Monte-Carlo variance.
    """
    sum_qz = torch.zeros(data_gen.dim, device=device)
    sum_z = torch.zeros(data_gen.dim, device=device)
    sum_q = torch.zeros((), device=device)
    for start in range(0, prob_num, chunk_size):
        m = min(chunk_size, prob_num - start)
        X = data_gen.sample_cluster(cluster_idx, m, generator=generator, device=device)
        q = torch.softmax(model(X), dim=1)[:, cluster_idx]
        Z = 2 * X - 1
        sum_qz += Z.T @ q
        sum_z += Z.sum(dim=0)
        sum_q += q.sum()

    gamma = sum_qz / prob_num
    if center:
        gamma -= (sum_q / prob_num) * (sum_z / prob_num)
    return gamma.cpu()


def reconstruct_cluster(model, data_gen, cluster_idx, prob_num=50000, chunk_size=25000,
                        center=True, generator=None, device='cuda'):
    """Correlation attack: W_hat(t) = sign(Gamma_i(t)) on unfixed coordinates, b_i(t) on fixed ones.

    Returns W_hat in {0,1}^d (the encoding used by data_generation).
    """
    gamma = estimate_correlation(model, data_gen, cluster_idx, prob_num, chunk_size,
                                 center=center, generator=generator, device=device)
    W_hat = (gamma > 0).float()
    fixed = data_gen.fixed_loc[cluster_idx] == 1
    W_hat[fixed] = data_gen.fixed_vals[cluster_idx]
    return W_hat


def attack_singletons(model, data_gen, singletons, X_train, y_train, prob_num=50000,
                      chunk_size=25000, center=True, seed=None, device='cuda'):
    """Run the correlation attack on each singleton cluster and score it against its unique training point.

    Accuracy is measured on the unfixed coordinates U_i only (the fixed ones are copied from theta
    and are always correct), so 0.5 is the no-information baseline.
    """
    was_training = model.training
    model.eval()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    accuracies = []
    for cluster_idx in singletons:
        W_hat = reconstruct_cluster(model, data_gen, cluster_idx, prob_num, chunk_size,
                                    center=center, generator=generator, device=device)
        idx = (y_train == cluster_idx).nonzero(as_tuple=True)[0][0]
        unfixed = data_gen.fixed_loc[cluster_idx] == 0
        acc_c = (W_hat[unfixed] == X_train[idx, unfixed].cpu()).float().mean().item()
        accuracies.append(acc_c)

    model.train(was_training)
    return accuracies, np.median(accuracies)
