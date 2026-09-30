import argparse
import json
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
    parser.add_argument('--prob_num', type=int, default=50000,
                        help='Fresh draws per singleton cluster used by the attack')
    parser.add_argument('--eval_every', type=int, default=10, help='Epochs between evaluations')
    parser.add_argument('--early_until', type=int, default=0,
                        help='Also evaluate every --eval_every_early epochs up to this epoch')
    parser.add_argument('--eval_every_early', type=int, default=10)
    parser.add_argument('--n_val', type=int, default=10000, help='Validation set size')
    parser.add_argument('--n_runs', type=int, default=20, help='Number of runs')
    parser.add_argument('--hidden', type=int, default=500, help='Hidden layer width')
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['sigmoid', 'relu'])
    parser.add_argument('--center_input', action='store_true',
                        help='Feed the model {-1,+1} features instead of {0,1}')
    parser.add_argument('--lr', type=float, default=5e-4, help='Adam learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam L2 weight decay')
    parser.add_argument('--out', type=str, default='clustering', help='Output file prefix')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
        eval_every=args.eval_every,
        n_val=args.n_val,
        model_kwargs={'h1': args.hidden, 'activation': args.activation,
                      'center_input': args.center_input},
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_until=args.early_until,
        eval_every_early=args.eval_every_early,
        device=device
    )

    with open(f'{args.out}_results.json', 'w') as f:
        json.dump({'args': vars(args), 'rho': float(rho), 'runs': all_results}, f)

    plot_results(all_results, save_path=f'{args.out}.pdf')
    plot_results(all_results, save_path=f'{args.out}.png')
