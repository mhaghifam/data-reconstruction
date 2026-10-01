import argparse
import json
import numpy as np
import torch
from src.ntp.ntp_experiment import (run_multiple_trials, plot_learning_vs_memorization,
                                    cross_validated_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Next-Token Prediction Reconstruction Attack')
    parser.add_argument('--d', type=int, default=500, help='Sequence length')
    parser.add_argument('--N', type=int, default=100, help='Number of clusters')
    parser.add_argument('--delta', type=float, default=0.05, help='Noise level (bits flipped w.p. delta/2)')
    parser.add_argument('--n_train', type=int, default=100, help='Training set size')
    parser.add_argument('--n_val', type=int, default=5000, help='Validation set size')
    parser.add_argument('--layers', type=int, default=2, help='Number of Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout in the Transformer layers')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--schedule', type=str, default='cosine', choices=['cosine', 'step'],
                        help='cosine: linear warmup then cosine decay; step: x0.3 after each third')
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--unknown_half', action=argparse.BooleanOptionalAction, default=True,
                        help='Also train positions past each sequence end: random input bits, target 1/2')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--eval_every', type=int, default=50, help='Epochs between evaluations')
    parser.add_argument('--prob_num', type=int, default=100,
                        help='Fresh draws per singleton cluster used by the attack')
    parser.add_argument('--length_threshold', type=float, default=None,
                        help='Attack: largest length whose score is at most this value (nats); '
                             'default: the smallest feasible threshold (largest minimizer)')
    parser.add_argument('--length_threshold_sweep', type=float, nargs='*',
                        default=[5, 10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 300],
                        help='Extra length thresholds evaluated from the same queries (saved to json)')
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

    iterations, all_val_acces, all_median_accs, all_sweeps = run_multiple_trials(
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
        num_attack_queries=args.prob_num,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        length_threshold=args.length_threshold,
        threshold_sweep=args.length_threshold_sweep,
        unknown_half=args.unknown_half
    )

    sweep = {str(t): [[checkpoint[t] for checkpoint in run] for run in all_sweeps]
             for t in list(args.length_threshold_sweep) + ['known_length']}

    # Plotted reconstruction accuracy: with several runs, each run's length threshold is chosen on
    # the other runs (never on itself); candidates are the sweep plus --length_threshold.
    plotted = all_median_accs
    chosen = None
    if args.n_runs > 1:
        primary = 'smallest_feasible' if args.length_threshold is None else str(args.length_threshold)
        candidates = {primary: all_median_accs.tolist(),
                      **{k: v for k, v in sweep.items() if k != 'known_length'}}
        plotted, chosen = cross_validated_accuracy(candidates)

    with open(f'{args.out}_results.json', 'w') as f:
        json.dump({'args': vars(args), 'epochs': iterations, 'val_acc': all_val_acces.tolist(),
                   'median_acc': all_median_accs.tolist(), 'median_acc_by_threshold': sweep,
                   'median_acc_cross_validated': np.asarray(plotted).tolist(),
                   'chosen_threshold': chosen}, f)

    fig = plot_learning_vs_memorization(
        iterations, all_val_acces, np.asarray(plotted),
        save_path=f'{args.out}.pdf'
    )
    fig.savefig(f'{args.out}.png', bbox_inches='tight')
