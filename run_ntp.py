import torch
from src.ntp.ntp_experiment import run_multiple_trials, plot_learning_vs_memorization

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")

    print(f"Using device: {device}")
    
    iterations, all_val_acces, all_median_accs = run_multiple_trials(
        num_trials=5,
        N=100,
        d=500,
        delta=0.05,
        n_train=100,
        n_val=500,
        batch_size=100,
        num_epochs=3000,
        device=device,
        eval_every=300,
        num_attack_queries=50
    )
    
    plot_learning_vs_memorization(
        iterations, all_val_acces, all_median_accs,
        save_path='ntp.pdf'
    )