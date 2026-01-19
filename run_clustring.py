import argparse
import sys
sys.path.append('src')

from clustring.clustring_expriment import run_multiple_experiments
from utils.plotting import plot_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypercube Clustering Reconstruction Attack')
    parser.add_argument('--d', type=int, default=500, help='Dimension of hypercube')
    parser.add_argument('--N', type=int, default=50, help='Number of clusters')
    parser.add_argument('--rho', type=float, default=None, help='Probability of fixing each bit')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--prob_num', type=int, default=500, help='Number of probes per bit')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_path', type=str, default='clustering_results.pdf', help='Save path')
    
    args = parser.parse_args()
    
    all_results = run_multiple_experiments(
        n_runs=args.n_runs,
        d=args.d,
        N=args.N,
        rho=args.rho,
        epochs=args.epochs,
        prob_num=args.prob_num,
        device=args.device
    )
    
    plot_results(all_results, save_path=args.save_path)