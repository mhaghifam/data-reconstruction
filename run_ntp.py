import torch
from src.ntp.ntp_experiment import run_multiple_trials, plot_learning_vs_memorization

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    iterations, all_val_losses, all_median_accs = run_multiple_trials(
        num_trials=5,
        N=100,
        d=400,
        delta=0.1,
        n_train=100,
        n_val=1000,
        batch_size=100,
        num_epochs=2000,
        device=device,
        eval_every=100,
        num_attack_queries=50
    )
    
    plot_learning_vs_memorization(
        iterations, all_val_losses, all_median_accs,
        save_path='ntp.pdf'
    )