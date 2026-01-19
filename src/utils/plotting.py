import numpy as np
import matplotlib.pyplot as plt

def plot_results(all_results, save_path=None):
    epochs = np.array(all_results[0]['epochs'])

    val_acces = np.array([r['val_acc'] for r in all_results])
    median_accs = np.array([r['median_acc'] for r in all_results])

    num_trials = val_acces.shape[0]
    val_acc_mean = np.mean(val_acces, axis=0)
    val_acc_std = np.std(val_acces, axis=0) / np.sqrt(num_trials)
    median_acc_mean = np.mean(median_accs, axis=0)
    median_acc_std = np.std(median_accs, axis=0) / np.sqrt(num_trials)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: validation accuracy
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Validation Accuracy', color=color1, fontsize=16)
    line1, = ax1.plot(
        epochs,
        val_acc_mean,
        color=color1,
        linewidth=2,
        marker='*',
        markersize=8,
        label='Validation Accuracy'
    )
    ax1.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                     color=color1, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)

    # Right axis: reconstruction accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Reconstruction Accuracy', color=color2, fontsize=16)
    line2, = ax2.plot(
        epochs,
        median_acc_mean,
        color=color2,
        linewidth=2,
        marker='o',
        markersize=5,
        label='Reconstruction Accuracy'
    )
    ax2.fill_between(epochs, median_acc_mean - median_acc_std, median_acc_mean + median_acc_std,
                     color=color2, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=16)

    plt.title(f'Hypercube Clustring Task', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
