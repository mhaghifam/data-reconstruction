import argparse
import sys
import torch
import numpy as np
sys.path.append('src')

from clustring.clustring_expriment import run_multiple_experiments
from utils.plotting import plot_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypercube Clustering Reconstruction Attack')
    parser.add_argument('--d', type=int, default=800, help='Dimension of hypercube')
    parser.add_argument('--N', type=int, default=100, help='Number of clusters')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--prob_num', type=int, default=1000, help='Number of probes per bit')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")

    print(f"Using device: {device}")
    args = parser.parse_args()
    d = args.d
    n = args.N
    rho = np.sqrt((2*np.log(2*n)-np.log(np.log(n)))/d)
    all_results = run_multiple_experiments(
        n_runs=args.n_runs,
        d=args.d,
        N=args.N,
        rho=rho,
        epochs=args.epochs,
        prob_num=args.prob_num,
        device=device
    )
    
    plot_results(all_results, save_path='clustering2.pdf')