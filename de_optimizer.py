import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from pathlib import Path
import shutil
import time
from datetime import datetime
import warnings
import traceback
from brian2 import start_scope, set_device, device
warnings.filterwarnings("ignore")

import step_1_SFVs as st1
import step_2_learning as st2
import step_seq_generator as stseq
import step_3_testing as st3
import step_4_mean_firing_rate as st4
import step_7_Zscore as st7


# Configuration

# Fixed parameters used across all optimization evaluations.
seed = 42
np.random.seed(seed)
fixed_decimals = 3  # Number of decimal places used for reproducible parameter rounding.
dt = 0.1
n_motif = 10
sequence_length = 5
initial_noise = 100  # ms
n_neurons = 768
n_time = 16000
n_time_each = int((n_time - (initial_noise / dt)) / sequence_length)

# Differential Evolution parameters.
pop_size = 10
max_iter = 20 
strategy = 'rand1bin'  # Alternative option: 'best1bin'.
f_mutation = 0.8
cr_crossover = 0.9
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # alpha, beta, epsilon

# Output directories for the optimization run.
main_dir = Path("Results_DE_optimization")
progress_dir = main_dir / "progress"
best_dir = main_dir / "best_results"
main_dir.mkdir(exist_ok=True)
progress_dir.mkdir(exist_ok=True)
best_dir.mkdir(exist_ok=True)


# Utility functions

# Round optimization parameters to a fixed number of decimals for reproducibility.
def round_params(params):

    return [round(float(p), fixed_decimals) for p in params]


# Prepare a clean Brian2 standalone simulation environment before running tests.
def prepare_simulation():

    output_dir = Path('standalone_code')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_scope()
    set_device('cpp_standalone', directory=str(output_dir), build_on_run=False)
    device.reinit()
    device.activate()


# Remove large intermediate folders to reduce disk usage after each evaluation.
def cleanup_heavy_data(results_dir):

    folders_to_delete = [
        results_dir / "step_0",
        results_dir / "step_1",
        results_dir / "step_2",
        results_dir / "step_3",
        results_dir / "step_5",
        results_dir / "step_6",
        results_dir / "standalone_code"
    ]

    for folder in folders_to_delete:
        if folder.exists():
            try:
                shutil.rmtree(folder)
                print(f"    Cleaned: {folder.name}")
            except:
                pass


# Evaluate one Differential Evolution candidate by running the full simulation and analysis pipeline.
def objective_function(params, iteration_counter, results_history):

    # Round parameters before using them in filenames and simulations.
    params_rounded = round_params(params)
    alpha, beta, epsilon = params_rounded

    # Assign a unique evaluation index for tracking progress.
    iteration = next(iteration_counter)
    print(f"\n{'='*70}")
    print(f"EVALUATION {iteration:04d}")
    print(f"Parameters: alpha={alpha}, beta={beta}, epsilon={epsilon}")
    print(f"{'='*70}")

    # Create the output directory for this parameter combination.
    file_suffix = f"alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}"
    base_path = Path(f"/home/dilettabartolini/neural-adaptation-adex-triplet-stdp/Results_DE_optimization/{file_suffix}_{seed}")

    results_dir = main_dir / f"{file_suffix}_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create one subdirectory per pipeline step.
    step_dirs = []
    for i in range(8):
        step_dir = results_dir / f"step_{i}"
        step_dir.mkdir(exist_ok=True)
        step_dirs.append(step_dir)

    try:
        # Step 1: Input pattern generation
        print("  Step 1: Generating input pattern...")
        x = st1.generate_block_structured_input(alpha, beta, epsilon, n_neurons, seed)
        np.save(step_dirs[1] / f"X_{file_suffix}.npy", x)

        # Step 2: Network training
        print("  Step 2: Training network...")
        set_device("runtime")
        st2.learn_and_save_network(file_suffix, n_motif, seed, base_path, step_dirs[2])

        # Step 3: Network testing
        print("  Step 3: Testing network...")

        stseq.generate_sequences(n_motif, sequence_length, seed, step_dirs[3])

        test_sequences_csv = step_dirs[3] / "combined_dataset_matrix.csv"

        # Reset and configure Brian2 before running the test simulation.
        prepare_simulation()

        # Run the AdEx test phase using the generated sequences.
        st3.AdEx_test(file_suffix, seed, base_path, step_dirs[3], test_sequences_csv)

        # Matrix preparation for steps 4 and 7
        print("  Loading activation matrices...")

        # Load activation metadata saved during the testing phase.
        activations_dir = step_dirs[3] / "Activations_test" / file_suffix
        matrices_dict = st3.load_matrices_with_metadata(activations_dir)

        output_dir_st3 = step_dirs[3] / "Processed_matrices"
        output_dir_st3.mkdir(exist_ok=True)

        # Build the Primed and Control matrices used by downstream analyses.
        results_st3 = st3.process_matrices(sequence_length, matrices_dict, n_neurons, n_time_each, output_dir_st3)

        p_matrix = np.load(output_dir_st3 / "P_extended.npy")
        c_matrix = np.load(output_dir_st3 / "C_extended.npy")

        p_index = np.load(output_dir_st3 / "P_extended_indices.npy")
        c_index = np.load(output_dir_st3 / "C_extended_indices.npy")

        n_pres = p_index.shape[0]

        # Step 4: Firing-rate analysis
        print("  Step 4: Analyzing firing rates...")
        results_st4 = st4.population_level_firing_rate_dynamics(p_matrix, c_matrix, step_dirs[4], n_pres, dt, seed, n_permutations=10000, alpha=0.02, min_cluster_size=5, t_threshold=None)

        c_g_p = results_st4['total_primed_lt_control']
        p_g_c = results_st4['total_primed_gt_control']

        print(f"    Results: primed < control = {c_g_p}, primed > control = {p_g_c}")

        # Step 7: Z-score analysis
        print("  Step 7: Running Z-score analysis...")

        results_st7 = st7.main_analysis_optimization(activations_dir, step_dirs[7],
                n_neurons=n_neurons, n_timepoints=3000,
                baseline_window=3000, response_window=3000, max_rank=10)


        # Cleanup
        cleanup_heavy_data(results_dir)

        # Save evaluation results
        result_entry = {
            'iteration': iteration,
            'alpha': alpha,
            'beta': beta,
            'epsilon': epsilon,
            'c_g_p': c_g_p,
            'p_g_c': p_g_c,
            'score': -c_g_p,  # Differential Evolution minimizes, so the target score is negated.
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'directory': str(results_dir)
        }

        results_history.append(result_entry)

        # Save progress to disk after every successful evaluation.
        save_progress(results_history)

        print("  Evaluation completed successfully")
        print(f"  Result: c_g_p = {c_g_p}, DE score = {-c_g_p}")

        # Return the negated objective because Differential Evolution minimizes.
        return -c_g_p

    except Exception as e:
        print(f"  ERROR during evaluation: {str(e)}")
        print("  Traceback:")
        traceback.print_exc()

        # Clean intermediate data even when the evaluation fails.
        try:
            cleanup_heavy_data(results_dir)
        except:
            pass

        # Store failed evaluations so the optimization history remains complete.
        result_entry = {
            'iteration': iteration,
            'alpha': alpha,
            'beta': beta,
            'epsilon': epsilon,
            'c_g_p': -1,
            'p_g_c': -1,
            'score': 1e6,  # Large penalty score for Differential Evolution minimization.
            'success': False,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'directory': str(results_dir),
            'error': str(e)
        }

        results_history.append(result_entry)
        save_progress(results_history)

        # Return a large value so Differential Evolution rejects this candidate.
        return 1e6


# Differential Evolution support functions

# Save the full optimization history and the best successful evaluations to disk.
def save_progress(results_history):

    if not results_history:
        return

    df = pd.DataFrame(results_history)

    # Save the complete history for every attempted evaluation.
    df.to_csv(progress_dir / "optimization_history.csv", index=False)

    # Save a filtered table containing only successful evaluations.
    df_success = df[df['success']].copy()
    if not df_success.empty:

        if len(df_success) > 1:
            c_g_p_values = df_success['c_g_p'].values
            min_val = c_g_p_values.min()
            max_val = c_g_p_values.max()

            if max_val > min_val:
                df_success['c_g_p_normalized'] = (c_g_p_values - min_val) / (max_val - min_val)
            else:
                df_success['c_g_p_normalized'] = 0.5

            df_success.to_csv(progress_dir / "successful_results_with_normalization.csv", index=False)

        # Save the current top 10 evaluations ranked by c_g_p.
        top_10 = df_success.nlargest(10, 'c_g_p')
        top_10.to_csv(progress_dir / "top_10_results.csv", index=False)


# Create the final optimization report and copy the most relevant best-result outputs.
def create_final_report(results_history, de_result, optimization_time):

    df = pd.DataFrame(results_history)
    df_success = df[df['success']].copy()

    if df_success.empty:
        print("No successful evaluations for final report")
        return

    # Identify the best successful parameter set based on c_g_p.
    best_idx = df_success['c_g_p'].idxmax()
    best_result = df_success.loc[best_idx]

    # Create the final report directory.
    report_dir = main_dir / "final_report"
    report_dir.mkdir(exist_ok=True)

    # Store best-parameter metadata for possible downstream use.
    best_params = {
        'alpha': best_result['alpha'],
        'beta': best_result['beta'],
        'epsilon': best_result['epsilon'],
        'c_g_p': best_result['c_g_p'],
        'directory': best_result['directory']
    }

    # Copy the most important output directories for the best evaluation.
    try:
        source_dir = Path(best_result['directory'])
        if source_dir.exists():
            import shutil
            best_copy = best_dir / f"best_alpha{best_result['alpha']:.3f}_beta{best_result['beta']:.3f}_eps{best_result['epsilon']:.3f}"

            # Copy only the directories needed for inspection and reporting.
            important_dirs = ['step_1', 'step_4', 'step_7']
            for item in important_dirs:
                src = source_dir / item
                dst = best_copy / item
                if src.exists():
                    if src.is_file():
                        shutil.copy2(src, dst)
                    else:
                        shutil.copytree(src, dst, dirs_exist_ok=True)

            print(f"\nBest results copied to: {best_copy}")
    except Exception as e:
        print(f"Warning: Could not copy best results: {e}")

    # Save all evaluations in a single CSV report.
    df.to_csv(report_dir / "all_evaluations.csv", index=False)


# Main Differential Evolution function

# Run Differential Evolution optimization over alpha, beta, and epsilon.
def run_differential_evolution():

    print("\n" + "="*80)
    print("DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("="*80)
    print("Objective: Maximize c_g_p (primed < control)")
    print(f"Parameters: alpha, beta, epsilon in [0, 1] (rounded to {fixed_decimals} decimals)")
    print(f"DE settings: {strategy}, pop={pop_size}, maxiter={max_iter}")
    print(f"F={f_mutation}, CR={cr_crossover}, seed={seed}")

    start_time = time.time()

    # Track the number of objective-function evaluations.
    class IterationCounter:

        # Initialize the evaluation counter.
        def __init__(self):
            self.count = 0

        # Return the next evaluation index.
        def __next__(self):
            self.count += 1
            return self.count

    iteration_counter = IterationCounter()
    results_history = []

    # Report the current best candidates after each Differential Evolution iteration.
    def de_callback(xk, convergence):

        print(f"\nDE callback: current best params = [{xk[0]:.3f}, {xk[1]:.3f}, {xk[2]:.3f}]")
        print(f"Convergence: {convergence:.6f}")

        # Print the current top 5 successful candidates.
        df = pd.DataFrame(results_history)
        df_success = df[df['success']].copy()
        if not df_success.empty:
            top_5 = df_success.nlargest(5, 'c_g_p')
            print("\nCurrent Top 5:")
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"  {i}. alpha={row['alpha']:.3f}, beta={row['beta']:.3f}, epsilon={row['epsilon']:.3f} "
                      f"-> c_g_p={row['c_g_p']}")

    # Wrap the objective function to match SciPy's expected signature.
    def objective_wrapper(params):
        return objective_function(params, iteration_counter, results_history)

    try:
        # Run Differential Evolution with serial execution.
        result = differential_evolution(
            func=objective_wrapper,
            bounds=bounds,
            strategy=strategy,
            maxiter=max_iter,
            popsize=pop_size,
            mutation=f_mutation,
            recombination=cr_crossover,
            seed=seed,
            callback=de_callback,
            disp=True,
            polish=False,  # Do not apply the final local minimization step.
            updating='immediate',
            workers=1  # Serial execution.
        )

        optimization_time = time.time() - start_time

        print("\nDE optimization completed!")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Optimal params: alpha={result.x[0]:.3f}, beta={result.x[1]:.3f}, epsilon={result.x[2]:.3f}")
        print(f"Optimal value: {-result.fun:.2f} (c_g_p = {-result.fun})")
        print(f"Function evaluations: {result.nfev}")
        print(f"Optimization time: {optimization_time/60:.1f} minutes")

        # Find the successful evaluation closest to the optimizer's final rounded parameters.
        df = pd.DataFrame(results_history)
        df_success = df[df['success']].copy()

        if not df_success.empty:
            # Round Differential Evolution parameters for comparison with saved evaluations.
            de_params_rounded = round_params(result.x)

            # Identify the successful result with the smallest parameter distance.
            df_success['param_distance'] = df_success.apply(
                lambda row: sum((row[['alpha', 'beta', 'epsilon']] - de_params_rounded)**2),
                axis=1
            )

            closest_idx = df_success['param_distance'].idxmin()
            closest_result = df_success.loc[closest_idx]

            print("\nClosest successful evaluation:")
            print(f"  alpha={closest_result['alpha']:.3f}, beta={closest_result['beta']:.3f}, "
                  f"epsilon={closest_result['epsilon']:.3f}")
            print(f"  c_g_p: {closest_result['c_g_p']}")
            print(f"  Directory: {closest_result['directory']}")

        # Generate the final optimization report.
        create_final_report(results_history, result, optimization_time)

        return result, results_history

    except KeyboardInterrupt:
        optimization_time = time.time() - start_time
        print(f"\n\nOptimization interrupted by user after {optimization_time/60:.1f} minutes")
        print(f"Partial results saved in: {main_dir}")

        # Save a final report using the partial results collected so far.
        if results_history:
            create_final_report(results_history, None, optimization_time)

        return None, results_history

    except Exception as e:
        optimization_time = time.time() - start_time
        print(f"\n\nERROR during optimization: {str(e)}")
        traceback.print_exc()

        return None, results_history


# ============================================================================
# Main execution

if __name__ == "__main__":
    de_result, history = run_differential_evolution()
