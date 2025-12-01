import numpy as np
import matplotlib.pyplot as plt


def plot_force_learning(time, x_hist, z_hist, w_norm_hist, target_hist,title="Force learning",T_total=3000):
    learning_start = 500
    learning_end = T_total - learning_start

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1 = axes[0]
    ax1.plot(time, target_hist, 'g--', label='Target', alpha=0.6)
    ax1.plot(time, z_hist, 'r', label='Output z(t)')
    ax1.set_title(title)
    ax1.set_ylabel("Output")
    ax1.axvline(learning_start, color='k', linestyle='--')
    ax1.axvline(learning_end, color='k', linestyle='--')
    ax1.legend(loc='upper right')
    ax1.text(learning_start / 2, 1.5, 'Spontaneous', ha='center')
    ax1.text((learning_start + learning_end) / 2, 1.5, 'Learning', ha='center')
    ax1.text((learning_end + T_total) / 2, 1.5, 'Test', ha='center')

    ax3 = axes[1]
    ax3.plot(time, w_norm_hist, 'orange', label='|dw/dt|')
    ax3.set_ylabel("Weight change")
    ax3.set_xlabel("Time (ms)")
    ax3.axvline(learning_start, color='k', linestyle='--')
    ax3.axvline(learning_end, color='k', linestyle='--')
    ax3.legend()

    plt.tight_layout()
    plt.show()

def plot_force_learning_original(time, x_hist, z_hist, w_norm_hist, target_hist,T_total=3000):
    learning_start = 500
    learning_end = T_total - learning_start

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(time, target_hist, 'g--', label='Target', alpha=0.6)
    ax1.plot(time, z_hist, 'r', label='Output z(t)')
    ax1.set_title("Figure 2: FORCE Learning")
    ax1.set_ylabel("Output")
    ax1.axvline(learning_start, color='k', linestyle='--')
    ax1.axvline(learning_end, color='k', linestyle='--')
    ax1.legend(loc='upper right')
    ax1.text(learning_start / 2, 1.5, 'Spontaneous', ha='center')
    ax1.text((learning_start + learning_end) / 2, 1.5, 'Learning', ha='center')
    ax1.text((learning_end + T_total) / 2, 1.5, 'Test', ha='center')

    ax2 = axes[1]
    for k in range(5):
        ax2.plot(time, x_hist[:, k], linewidth=0.8)
    ax2.set_ylabel("Activity r(t)")
    ax2.axvline(learning_start, color='k', linestyle='--')
    ax2.axvline(learning_end, color='k', linestyle='--')

    ax3 = axes[2]
    ax3.plot(time, w_norm_hist, 'orange', label='|dw/dt|')
    ax3.set_ylabel("Weight change")
    ax3.set_xlabel("Time (ms)")
    ax3.axvline(learning_start, color='k', linestyle='--')
    ax3.axvline(learning_end, color='k', linestyle='--')
    ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_feedback_term_exploration(ft_range,differences,k = 7):
    output = np.zeros(len(differences))
    for i in range(len(differences)):
        min_index = max(0,i - k)
        max_index = min(len(differences),i + k)
        avg_diff = np.mean(differences[min_index:max_index+1])
        output[i] = avg_diff

    plt.figure(figsize=(8,6))
    plt.plot(ft_range,output,linestyle ='',marker = 'x')
    plt.xlabel('Feedback term coefficient')
    plt.ylabel('Max difference between target and output')
    plt.title('Effect of feedback term coefficient on learning performance')
    plt.grid()
    plt.show()

def plot_feedback_analysis_multiple_runs(file_name, treatment_function, ylabel='Metric', title='Analysis vs Feedback Coefficient', figsize=(10, 6)):
    """
    Load experiment data and plot treated results as a function of feedback coefficient.
    
    Parameters:
    - file_name: Path to .npy file with experiment results
    - treatment_function: Function that takes (time, x_hist, z_hist, w_norm_hist, target_hist) 
                         and returns a single real number
    - ylabel: Label for the y-axis
    - title: Plot title
    - figsize: Figure size tuple
    
    Returns:
    - fig, ax: Matplotlib figure and axis objects
    - ft_range: Array of feedback coefficients
    - means: Array of mean values for each coefficient
    - stds: Array of standard deviations for each coefficient
    """
    # Load the data
    data = np.load(file_name, allow_pickle=True).item()
    ft_range = data['ft_range']
    num_trials = data['num_trials']
    results = data['results']
    
    # Process each feedback coefficient
    means = []
    stds = []
    
    for ft in ft_range:
        if ft not in results:
            print(f'Warning: No results found for ft={ft}')
            means.append(np.nan)
            stds.append(np.nan)
            continue
        
        trials = results[ft]
        treated_values = []
        
        for trial_data in trials:
            time, x_hist, z_hist, w_norm_hist, target_hist = trial_data
            value = treatment_function(time, x_hist, z_hist, w_norm_hist, target_hist)
            treated_values.append(value)
        print(f'ft={ft},samples: {len(treated_values)}') 
        means.append(np.mean(treated_values))
        stds.append(np.std(treated_values))
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean line
    ax.plot(ft_range, means, 'o-', linewidth=2, markersize=6, color='blue', label='Mean')
    
    # Plot standard deviation as shaded region
    ax.fill_between(ft_range, means - stds, means + stds, alpha=0.3, color='blue', label='± 1 std')
    
    ax.set_xlabel('Feedback Coefficient', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax, ft_range, means, stds


def plot_force_learning_averaged(file_name, ft_value, T_total=3000,save_file = None):
    """
    Plot FORCE learning results averaged across multiple trials for a specific feedback coefficient.
    Shows mean with standard deviation bands.
    
    Parameters:
    - file_name: Path to .npy file with experiment results
    - ft_value: Feedback coefficient value to plot
    - T_total: Total time for the experiment (for phase markers)
    """
    # Load the data
    data = np.load(file_name, allow_pickle=True).item()
    results = data['results']
    
    # Find the closest ft value in the data
    ft_range = data['ft_range']
    closest_ft = min(ft_range, key=lambda x: abs(x - ft_value))
    
    if abs(closest_ft - ft_value) > 1e-6:
        print(f'Warning: Requested ft={ft_value}, using closest available ft={closest_ft}')
    
    if closest_ft not in results:
        print(f'Error: No results found for ft={closest_ft}')
        return
    
    trials = results[closest_ft]
    print(f'Plotting averaged results for ft={closest_ft} with {len(trials)} trials')
    
    # Extract data from all trials
    time = trials[0][0]  # Time should be the same for all trials
    
    # Stack all trials' data
    z_hists = np.array([trial[2] for trial in trials])
    x_hists = np.array([trial[1] for trial in trials])
    w_norm_hists = np.array([trial[3] for trial in trials])
    target_hist = trials[0][4]  # Target is the same for all trials
    
    # Calculate means and stds
    z_mean = np.mean(z_hists, axis=0)
    z_std = np.std(z_hists, axis=0)
    
    x_mean = np.mean(x_hists, axis=0)
    x_std = np.std(x_hists, axis=0)
    
    w_norm_mean = np.mean(w_norm_hists, axis=0)
    w_norm_std = np.std(w_norm_hists, axis=0)
    
    # Create the plot
    learning_start = 500
    learning_end = T_total - learning_start

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Output z(t) vs Target
    ax1 = axes[0]
    ax1.plot(time, target_hist, 'g--', label='Target', alpha=0.6, linewidth=2)
    ax1.plot(time, z_mean, 'r', label=f'Output z(t) (mean, n={len(trials)})', linewidth=2)
    ax1.fill_between(time, z_mean - z_std, z_mean + z_std, color='red', alpha=0.2, label='± 1 std')
    ax1.set_title(f"FORCE Learning (Averaged, ft={closest_ft:.4f})")
    ax1.set_ylabel("Output")
    ax1.axvline(learning_start, color='k', linestyle='--')
    ax1.axvline(learning_end, color='k', linestyle='--')
    ax1.legend(loc='upper right')
    ax1.text(learning_start / 2, ax1.get_ylim()[1] * 0.9, 'Spontaneous', ha='center')
    ax1.text((learning_start + learning_end) / 2, ax1.get_ylim()[1] * 0.9, 'Learning', ha='center')
    ax1.text((learning_end + T_total) / 2, ax1.get_ylim()[1] * 0.9, 'Test', ha='center')

    # Plot 2: Weight change
    ax2 = axes[1]
    ax2.plot(time, w_norm_mean, 'orange', label='|dw/dt| (mean)', linewidth=2)
    ax2.fill_between(time, w_norm_mean - w_norm_std, w_norm_mean + w_norm_std, 
                    color='orange', alpha=0.2, label='± 1 std')
    ax2.set_ylabel("Weight change")
    ax2.set_xlabel("Time (ms)")
    ax2.axvline(learning_start, color='k', linestyle='--')
    ax2.axvline(learning_end, color='k', linestyle='--')
    ax2.legend()

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
    
    return fig, axes
