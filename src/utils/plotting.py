import numpy as np
import matplotlib.pyplot as plt

def plot_results(all_results, save_path=None):
    epochs = all_results[0]['epochs']
    
    val_losses = np.array([r['val_loss'] for r in all_results])
    median_accs = np.array([r['median_acc'] for r in all_results])
    mean_accs = np.array([r['mean_acc'] for r in all_results])
    
    val_loss_mean = val_losses.mean(axis=0)
    val_loss_std = val_losses.std(axis=0)
    median_acc_mean = median_accs.mean(axis=0)
    median_acc_std = median_accs.std(axis=0)
    mean_acc_mean = mean_accs.mean(axis=0)
    mean_acc_std = mean_accs.std(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: vs epoch
    ax1.errorbar(epochs, val_loss_mean, yerr=val_loss_std, label='Val Loss', marker='o', capsize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_xscale('log')
    ax1.legend(loc='upper right')
    ax1.set_title('Validation Loss vs Epoch')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.errorbar(epochs, median_acc_mean, yerr=median_acc_std, label='Median Recon Acc', 
                       marker='s', color='orange', capsize=3)
    ax1_twin.set_ylabel('Reconstruction Accuracy')
    ax1_twin.legend(loc='right')
    
    # Plot 2: val loss vs accuracy
    ax2.errorbar(val_loss_mean, median_acc_mean, xerr=val_loss_std, yerr=median_acc_std,
                  label='Median Acc', marker='o', capsize=3)
    ax2.errorbar(val_loss_mean, mean_acc_mean, xerr=val_loss_std, yerr=mean_acc_std,
                  label='Mean Acc', marker='s', capsize=3)
    ax2.set_xlabel('Validation Loss')
    ax2.set_ylabel('Reconstruction Accuracy')
    ax2.set_title('Reconstruction Accuracy vs Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()