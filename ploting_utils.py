import numpy as np
import matplotlib.pyplot as plt


def plot_force_learning(time, x_hist, z_hist, w_norm_hist, target_hist):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(time, target_hist, 'g--', label='Target', alpha=0.6)
    ax1.plot(time, z_hist, 'r', label='Output z(t)')
    ax1.set_title("Figure 2: FORCE Learning")
    ax1.set_ylabel("Output")
    ax1.axvline(500, color='k', linestyle='--')
    ax1.axvline(2500, color='k', linestyle='--')
    ax1.legend(loc='upper right')
    ax1.text(250, 1.5, 'Spontaneous', ha='center')
    ax1.text(1500, 1.5, 'Learning', ha='center')
    ax1.text(2750, 1.5, 'Test', ha='center')

    ax2 = axes[1]
    for k in range(5):
        ax2.plot(time, x_hist[:, k], linewidth=0.8)
    ax2.set_ylabel("Activity r(t)")
    ax2.axvline(500, color='k', linestyle='--')
    ax2.axvline(2500, color='k', linestyle='--')

    ax3 = axes[2]
    ax3.plot(time, w_norm_hist, 'orange', label='|dw/dt|')
    ax3.set_ylabel("Weight change")
    ax3.set_xlabel("Time (ms)")
    ax3.axvline(500, color='k', linestyle='--')
    ax3.axvline(2500, color='k', linestyle='--')
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
