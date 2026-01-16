import torch
from torch.utils.data import Dataset

class data_generation:
    def __init__(self, d):
        self.dim = d
        self.fixed_loc = None
        self.fixed_vals = None
        self.n_clusters = None
        self.singletons = None

    def generate_centers(self, rho, N):
        self.n_clusters = N
        dist = torch.distributions.Bernoulli(probs=rho)
        self.fixed_loc = dist.sample(sample_shape=(N, self.dim))
        
        self.fixed_vals = []
        dist = torch.distributions.Bernoulli(probs=0.5)
        for i in range(N):
            m = int(self.fixed_loc[i].sum().item())
            self.fixed_vals.append(dist.sample(sample_shape=(m,)))

    def generate_samples(self, n, rho=None, label=None, fixed_instance=False):
        if fixed_instance is False:
            self.generate_centers(rho, n)
            y = torch.randint(0, self.n_clusters, (n,))
            X = torch.distributions.Bernoulli(probs=0.5).sample((n, self.dim))
            for i in range(n):
                cluster = y[i].item()
                mask = self.fixed_loc[cluster] == 1
                X[i, mask] = self.fixed_vals[cluster]
            cluster_counts = torch.bincount(y, minlength=self.n_clusters)
            self.singletons = torch.where(cluster_counts == 1)[0].tolist()
            return X, y
        else:
            if label is not None:
                X = torch.distributions.Bernoulli(probs=0.5).sample((n, self.dim))
                mask = self.fixed_loc[label] == 1
                for i in range(n):
                    X[i, mask] = self.fixed_vals[label]
                y = torch.full((n,), label)
                return X, y
            else:
                y = torch.randint(0, self.n_clusters, (n,))
                X = torch.distributions.Bernoulli(probs=0.5).sample((n, self.dim))
                for i in range(n):
                    cluster = y[i].item()
                    mask = self.fixed_loc[cluster] == 1
                    X[i, mask] = self.fixed_vals[cluster]
                return X, y


class HypercubeDataset(Dataset):
    def __init__(self, data_gen, n, rho=None, fixed_instance=True):
        if not fixed_instance:
            self.X, self.y = data_gen.generate_samples(n, rho=rho, fixed_instance=False)
        else:
            self.X, self.y = data_gen.generate_samples(n, fixed_instance=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx]