import argparse
import json
import torch
from src.ntp.ntp_experiment import run_multiple_trials, plot_learning_vs_memorization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Next-Token Prediction Reconstruction Attack')
    parser.add_argument('--d', type=int, default=500, help='Sequence length')
    parser.add_argument('--N', type=int, default=100, help='Number of clusters')
    parser.add_argument('--delta', type=float, default=0.05, help='Noise level (bits flipped w.p. delta/2)')
    parser.add_argument('--n_train', type=int, default=100, help='Training set size')
    parser.add_argument('--n_val', type=int, default=500, help='Validation set size')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--eval_every', type=int, default=150, help='Epochs between evaluations')
    parser.add_argument('--prob_num', type=int, default=100,
                        help='Fresh draws per prefix length used by the attack')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--out', type=str, default='ntp', help='Output file prefix')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    iterations, all_val_acces, all_median_accs = run_multiple_trials(
        num_trials=args.n_runs,
        N=args.N,
        d=args.d,
        delta=args.delta,
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=device,
        eval_every=args.eval_every,
        num_attack_queries=args.prob_num
    )

    with open(f'{args.out}_results.json', 'w') as f:
        json.dump({'args': vars(args), 'epochs': iterations, 'val_acc': all_val_acces.tolist(),
                   'median_acc': all_median_accs.tolist()}, f)

    fig = plot_learning_vs_memorization(
        iterations, all_val_acces, all_median_accs,
        save_path=f'{args.out}.pdf'
    )
    fig.savefig(f'{args.out}.png', bbox_inches='tight')
