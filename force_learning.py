import numpy as np
import numpy.random as rnd

N = 1000
tau = 1.0
dt = 1.0  
g = 1.5   
p = 0.1   

np.random.seed(42)

scale = g / np.sqrt(p * N)
J = rnd.normal(0, 1.0/np.sqrt(p*N), (N, N)) * (rnd.rand(N, N) < p)

w_fb = rnd.uniform(-1, 1, N)

w = rnd.normal(0, 1.0/np.sqrt(N), N) 

alpha = 1.0
P = (1.0/alpha) * np.eye(N)

def triangle_wave(t, period=600.0):
    phase = (t % period) / period
    return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0


def run_force_learning(T_total=3000, learning_start=500, feedback_amplitude = 1.0,update_output = 0.0):
    learning_end = T_total - learning_start
    time_points = np.arange(0, T_total, dt)
    T_steps = len(time_points)
    
    x_hist = np.zeros((T_steps, N))
    z_hist = np.zeros(T_steps)
    w_norm_hist = np.zeros(T_steps)
    target_hist = np.zeros(T_steps)
    dx_history = np.zeros(T_steps)
    
    x = rnd.normal(0, 0.5, N)
    r = np.tanh(x)
    
    w_current = w.copy()
    P_current = P.copy()
    
    w_prev = w_current.copy() 
    
    for i, t in enumerate(time_points):
        target = triangle_wave(t)
        target_hist[i] = target
        
        z_current = np.dot(w_current, r)
        z_hist[i] = z_current
        
        if learning_start <= t < learning_end:
            e_minus = z_current - target
            
            k = np.dot(P_current, r)
            rPr = np.dot(r, k)
            c = 1.0 / (1.0 + rPr)
            P_current = P_current - c * np.outer(k, k)
            
            dw = -e_minus * c * k
            w_current = w_current + dw
            
            w_norm_hist[i] = np.linalg.norm(dw)
            
        else:
            w_norm_hist[i] = 0.0

        z_new = np.dot(w_current, r)

        feedback = z_current * (1-update_output) + z_new * update_output
        feedback_term = w_fb * feedback
        
        recurrence = g * np.dot(J, r)
        circuit_input = recurrence + feedback_term
        dx = recurrence + feedback_term - x #you want x + dx = recurrence + feedback term
        x += dx * feedback_amplitude
        #dx = feedback_coefficent * dx
        #r = np.tanh(x)
        r = np.tanh(x)
        x_hist[i] = x
    
    return time_points, x_hist, z_hist, w_norm_hist, target_hist

def feedback_term_exploration(max_feedback_term = 1.0, number_of_runs = 10, save_file = 'feedback_exploration_results.npy', print_frequency=10):
    dft = max_feedback_term/number_of_runs
    ft_range = np.arange(0, max_feedback_term, dft)
    
    # Load existing results if file exists
    import os
    if os.path.exists(save_file):
        saved_data = np.load(save_file, allow_pickle=True).item()
        results = saved_data.get('results', [])
        start_idx = len(results)
        print(f'Loaded {start_idx} existing results from {save_file}')
    else:
        results = []
        start_idx = 0
    
    for i in range(start_idx, len(ft_range)):
        if (i+1) % print_frequency == 0:
            print(f'Run {i+1} out of {len(ft_range)}')
            # Save progress every 10 runs
            save_data = {'ft_range': ft_range, 'results': results}
            np.save(save_file, save_data)
            print(f'Progress saved to {save_file}')
    
        ft = ft_range[i]
        time, x_hist, z_hist, w_norm_hist, target_hist = force_learning.run_force_learning(feedback_coefficent = ft)
        results.append((time, x_hist, z_hist, w_norm_hist, target_hist))
    
    # Final save
    save_data = {'ft_range': ft_range, 'results': results}
    np.save(save_file, save_data)
    print(f'Final results saved to {save_file}')
    
    return results

def feedback_term_multiple_runs(max_feedback_term=1.0, num_coefficients=10, num_trials=5, save_file='feedback_multiple_runs.npy', checkpoint_frequency=5):
    """
    Run multiple experiments across different feedback coefficients, with multiple trials per coefficient.
    
    Parameters:
    - max_feedback_term: Maximum feedback coefficient value
    - num_coefficients: Number of different feedback coefficient values to test (from 0 to max)
    - num_trials: Number of trials to run for each coefficient value
    - save_file: File to save/load results
    - checkpoint_frequency: Save progress every N coefficient values
    
    Returns:
    - Dictionary with structure: {
        'ft_range': array of feedback coefficient values tested,
        'num_trials': number of trials per coefficient,
        'results': {
            ft_value: [list of num_trials results, each being (time, x_hist, z_hist, w_norm_hist, target_hist)]
        }
      }
    """
    import os
    
    # Generate range of feedback coefficients
    dft = max_feedback_term / num_coefficients
    ft_range = np.arange(0, max_feedback_term, dft)
    
    # Load existing results if file exists and preserve relevant data
    results = {}
    if os.path.exists(save_file):
        saved_data = np.load(save_file, allow_pickle=True).item()
        saved_results = saved_data.get('results', {})
        
        # Keep existing results for any ft values that match current ft_range
        preserved_count = 0
        for ft in ft_range:
            # Check if this ft value exists in saved results (with small tolerance for floating point)
            for saved_ft, saved_trials in saved_results.items():
                if abs(saved_ft - ft) < 1e-9:  # Close enough
                    results[ft] = saved_trials[:num_trials]  # Keep up to num_trials
                    preserved_count += len(results[ft])
                    break
        
        if preserved_count > 0:
            print(f'Preserved {preserved_count} existing trials from {save_file}')
    
    # Calculate how many experiments need to be run
    total_needed = len(ft_range) * num_trials
    total_existing = sum(len(trials) for trials in results.values())
    total_to_run = total_needed - total_existing
    
    print(f'Running experiments for {len(ft_range)} feedback coefficients with {num_trials} trials each')
    print(f'Feedback coefficient range: {ft_range[0]:.3f} to {ft_range[-1]:.3f}')
    print(f'Total experiments: {total_existing} existing + {total_to_run} new = {total_needed} total')
    
    experiments_run = 0
    
    for i, ft in enumerate(ft_range):
        # Check how many trials we already have for this ft
        existing_trials = results.get(ft, [])
        trials_needed = num_trials - len(existing_trials)
        
        if trials_needed == 0:
            continue  # Already have all trials for this ft
        
        print(f'\n--- Coefficient {i+1}/{len(ft_range)}: ft={ft:.4f} ---')
        if len(existing_trials) > 0:
            print(f'  (Found {len(existing_trials)} existing trials, running {trials_needed} more)')
        
        trials_for_this_ft = list(existing_trials)  # Start with existing trials
        
        for trial in range(len(existing_trials), num_trials):
            print(f'  Trial {trial+1}/{num_trials}', end='')
            
            time, x_hist, z_hist, w_norm_hist, target_hist = run_force_learning(
                feedback_coefficent=ft
            )
            trials_for_this_ft.append((time, x_hist, z_hist, w_norm_hist, target_hist))
            experiments_run += 1
            print(' ✓')
        
        # Store all trials for this coefficient
        results[ft] = trials_for_this_ft
        
        # Save checkpoint periodically
        if experiments_run % (checkpoint_frequency * num_trials) == 0 or (i + 1) == len(ft_range):
            save_data = {
                'ft_range': ft_range,
                'num_trials': num_trials,
                'results': results
            }
            np.save(save_file, save_data)
            completed_fts = sum(1 for ft in ft_range if len(results.get(ft, [])) == num_trials)
            print(f'Checkpoint saved to {save_file} ({completed_fts}/{len(ft_range)} coefficients completed)')
    
    final_data = {
        'ft_range': ft_range,
        'num_trials': num_trials,
        'results': results
    }
    
    print(f'\n✓ All experiments completed! {experiments_run} new experiments run.')
    return final_data
