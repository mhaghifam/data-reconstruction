import numpy as np
import matplotlib.pyplot as plt


def plot_results(all_results, save_path=None, recon_key='median_acc',
                 title='Hypercube Clustering Task', val_ylim=None, recon_ylim=(0.5, 1)):
    """Same layout as plot_learning_vs_memorization in src/ntp/ntp_experiment.py.

    Curves are the mean over runs; shaded bands are +/- one standard error over runs.
    recon_key='median_acc' is, per run, the median over singleton clusters of the fraction of
    unfixed coordinates the attack recovers (the statistic the NTP plot uses);
    'mean_acc' uses the mean over singleton clusters instead.
    """
    epochs = np.array(all_results[0]['epochs'])
    val_accs = np.array([r['val_acc'] for r in all_results])
    recon_accs = np.array([r[recon_key] for r in all_results])

    num_trials = val_accs.shape[0]
    ddof = 1 if num_trials > 1 else 0
    val_acc_mean = np.mean(val_accs, axis=0)
    val_acc_se = np.std(val_accs, axis=0, ddof=ddof) / np.sqrt(num_trials)
    recon_mean = np.mean(recon_accs, axis=0)
    recon_se = np.std(recon_accs, axis=0, ddof=ddof) / np.sqrt(num_trials)

    if val_ylim is None:
        top = np.max(val_acc_mean + val_acc_se)
        val_ylim = (0, np.ceil(top * 1.1 / 0.05) * 0.05)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    tick_labelsize = 14

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
    ax1.fill_between(epochs, val_acc_mean - val_acc_se, val_acc_mean + val_acc_se,
                     color=color1, alpha=0.2)
    ax1.tick_params(axis='x', labelsize=tick_labelsize)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=tick_labelsize)
    ax1.set_ylim(*val_ylim)

    # Right axis: reconstruction accuracy on singleton clusters
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Reconstruction Accuracy', color=color2, fontsize=16)
    line2, = ax2.plot(
        epochs,
        recon_mean,
        color=color2,
        linewidth=2,
        marker='o',
        markersize=5,
        label='Reconstruction Accuracy'
    )
    ax2.fill_between(epochs, recon_mean - recon_se, recon_mean + recon_se,
                     color=color2, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=tick_labelsize)
    ax2.set_ylim(*recon_ylim)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=16)

    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig
