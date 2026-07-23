import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
import warnings
from scipy.stats import sem
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

# Global plot settings.
plt.rcParams.update({'font.size': 14})

fontsize_title = 18
fontsize_axis_label = 16
fontsize_tick = 16

blue_pastel = (0.4, 0.6, 1.0, 0.7)  # Blue for Primed.
red_pastel = (1.0, 0.4, 0.4, 0.7)   # Red for Control.
gray_pastel = (0.6, 0.6, 0.6, 0.7)  # Gray for the unclassified first window.


# Return the first spike latency in milliseconds from a binary activity signal.
def find_first_spike(binary_signal, dt, min_spikes=1):

    # Skip signals that do not contain the minimum number of active samples.
    if np.sum(binary_signal) < min_spikes:
        return np.nan

    # Identify all active time bins in the binary signal.
    spike_indices = np.where(binary_signal >= 1)[0]

    if len(spike_indices) == 0:
        return np.nan

    # Convert the first active bin into milliseconds using the sampling step.
    first_spike_idx = spike_indices[0]
    onset_time_ms = first_spike_idx * dt

    return onset_time_ms


# Extract valid paired onset latencies for Primed and Control conditions across neurons.
def filter_neurons_for_onset(p_response_matrix, c_response_matrix, p_baseline_matrix, c_baseline_matrix,
                             n_pres, n_neurons_total, dt, min_onset, max_onset, min_active_presentations):

    neuron_onset_primed = []
    neuron_onset_control = []
    neuron_indices = []

    # Compute onset latency independently for each neuron and presentation.
    for neuron_idx in range(n_neurons_total):
        onset_times_per_pres_primed = []
        onset_times_per_pres_control = []

        for pres_idx in range(n_pres):
            row_idx = pres_idx * n_neurons_total + neuron_idx

            baseline_primed = p_baseline_matrix[row_idx, :].astype(float)
            baseline_control = c_baseline_matrix[row_idx, :].astype(float)
            signal_primed = p_response_matrix[row_idx, :].astype(float)
            signal_control = c_response_matrix[row_idx, :].astype(float)

            # Accept response onsets only when the preceding baseline segment is silent.
            onset_pr = np.nan
            onset_ctrl = np.nan
            if not np.any(baseline_primed >= 1):
                onset_pr = find_first_spike(signal_primed, dt, min_spikes=1)
            if not np.any(baseline_control >= 1):
                onset_ctrl = find_first_spike(signal_control, dt, min_spikes=1)

            if not np.isnan(onset_pr):
                onset_times_per_pres_primed.append(onset_pr)
            if not np.isnan(onset_ctrl):
                onset_times_per_pres_control.append(onset_ctrl)

        # Use the median onset only when enough presentations are active.
        if len(onset_times_per_pres_primed) >= min_active_presentations:
            onset_pr = np.median(onset_times_per_pres_primed)
            if not (min_onset <= onset_pr <= max_onset):
                onset_pr = np.nan
        else:
            onset_pr = np.nan

        if len(onset_times_per_pres_control) >= min_active_presentations:
            onset_ctrl = np.median(onset_times_per_pres_control)
            if not (min_onset <= onset_ctrl <= max_onset):
                onset_ctrl = np.nan
        else:
            onset_ctrl = np.nan

        # Keep only neurons with valid paired onsets in both conditions.
        if not np.isnan(onset_pr) and not np.isnan(onset_ctrl):
            neuron_onset_primed.append(onset_pr)
            neuron_onset_control.append(onset_ctrl)
            neuron_indices.append(neuron_idx)

    return np.array(neuron_onset_primed), np.array(neuron_onset_control), neuron_indices


# Run a paired sign-swap permutation test on onset latencies.
def permutation_test_paired_onset(onset_primed, onset_control, n_permutations, random_seed):

    if random_seed is not None:
        np.random.seed(random_seed)

    # Return empty statistics when there are not enough paired observations.
    if len(onset_primed) < 2:
        return {
            'p_value': np.nan,
            'median_diff': np.nan,
            'median_primed': np.nan,
            'median_control': np.nan,
            'n_pairs': len(onset_primed),
            'mean_primed': np.nan,
            'mean_control': np.nan,
            'iqr_primed': np.nan,
            'iqr_control': np.nan,
            'range_primed': (np.nan, np.nan),
            'range_control': (np.nan, np.nan),
            'statistic': np.nan
        }

    n_pairs = len(onset_primed)
    observed_median_diff = np.median(onset_primed - onset_control)

    # Generate all random condition swaps for the permutation test.
    swap = np.random.rand(n_permutations, n_pairs) < 0.5

    # Swap paired labels while preserving the paired structure of the data.
    perm_primed = np.where(swap, onset_control, onset_primed)
    perm_control = np.where(swap, onset_primed, onset_control)

    # Compute the median paired difference for each permutation.
    perm_diffs = np.median(perm_primed - perm_control, axis=1)

    # Compute a two-sided permutation p-value.
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_median_diff)) / n_permutations

    # Compute descriptive statistics for the paired onset distributions.
    median_primed = np.median(onset_primed)
    median_control = np.median(onset_control)
    q75_primed, q25_primed = np.percentile(onset_primed, [75, 25])
    q75_control, q25_control = np.percentile(onset_control, [75, 25])
    iqr_primed = q75_primed - q25_primed
    iqr_control = q75_control - q25_control

    return {
        'p_value': p_value,
        'median_diff': observed_median_diff,
        'median_primed': median_primed,
        'median_control': median_control,
        'n_pairs': n_pairs,
        'mean_primed': np.mean(onset_primed),
        'mean_control': np.mean(onset_control),
        'iqr_primed': iqr_primed,
        'iqr_control': iqr_control,
        'range_primed': (np.min(onset_primed), np.max(onset_primed)),
        'range_control': (np.min(onset_control), np.max(onset_control)),
        'statistic': observed_median_diff
    }


# Apply multiple-comparison correction to onset p-values across windows.
def apply_multiple_comparisons_correction_onset(results, alpha, method):

    # Keep the original p-value when there is only one comparison.
    if len(results) <= 1:
        for res in results:
            res['p_value_corrected'] = res['p_value']
            res['correction_method'] = 'none (single window)'
        return results

    p_values = []
    valid_indices = []

    # Collect valid p-values while tracking their original result indices.
    for i, res in enumerate(results):
        if not np.isnan(res['p_value']):
            p_values.append(res['p_value'])
            valid_indices.append(i)

    # Propagate missing values when no valid p-values are available.
    if not p_values:
        for res in results:
            res['p_value_corrected'] = np.nan
            res['correction_method'] = method
        return results

    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)

    # Write corrected p-values back into the corresponding result entries.
    for idx, res_idx in enumerate(valid_indices):
        results[res_idx]['p_value_corrected'] = pvals_corrected[idx]
        results[res_idx]['correction_method'] = method

    # Preserve missing corrected p-values for invalid comparisons.
    for i, res in enumerate(results):
        if i not in valid_indices:
            res['p_value_corrected'] = np.nan
            res['correction_method'] = method

    return results



# Plot paired onset-latency boxplots for all analysis windows in a single figure.
def plot_combined_onset_boxplots(results, output_dir, alpha, random_seed):

    n_windows = len(results)

    fig, axes = plt.subplots(1, n_windows, figsize=(12, 6), sharey=True)

    if n_windows == 1:
        axes = [axes]

    np.random.seed(random_seed)

    # Compute one global y-position so all significance labels are aligned.
    global_max = 0
    for res in results:
        onset_primed = res['raw_onset_primed']
        onset_control = res['raw_onset_control']

        # Estimate the top whisker for each condition and keep the largest value.
        for peaks in [onset_primed, onset_control]:
            if len(peaks) > 0:
                q75 = np.percentile(peaks, 75)
                q25 = np.percentile(peaks, 25)
                iqr = q75 - q25
                upper_whisker_limit = q75 + 1.5 * iqr

                # Use the highest observed value that still falls within the whisker limit.
                data_within_whisker = peaks[peaks <= upper_whisker_limit]
                if len(data_within_whisker) > 0:
                    actual_whisker_top = np.max(data_within_whisker)
                else:
                    actual_whisker_top = q75

                global_max = max(global_max, actual_whisker_top)

    # Add a small margin above the highest whisker for significance labels.
    y_max_global = global_max * 1.07

    # Draw each window using the shared y-position for label placement.
    for i, (res, ax) in enumerate(zip(results, axes)):
        onset_primed = res['raw_onset_primed']
        onset_control = res['raw_onset_control']
        median_pr = res['median_onset_primed_ms']
        median_ctrl = res['median_onset_control_ms']
        p_value_corrected = res.get('p_value_corrected', res['p_value'])
        window_num = res['window']

        # Use gray for the first unclassified window and condition colors otherwise.
        if window_num == 1:
            col1 = gray_pastel
            col2 = gray_pastel
        else:
            col1 = blue_pastel
            col2 = red_pastel

        # Add horizontal jitter to make overlapping points visible.
        jitter_primed = np.random.normal(0, 0.04, len(onset_primed)) if len(onset_primed) > 0 else []
        jitter_control = np.random.normal(0, 0.04, len(onset_control)) if len(onset_control) > 0 else []

        # Plot individual onset latencies on top of the boxplots.
        if len(onset_primed) > 0:
            ax.scatter(1 + jitter_primed, onset_primed, alpha=0.5, color=col1,
                      s=20, edgecolors='black', linewidths=0.5)
        if len(onset_control) > 0:
            ax.scatter(2 + jitter_control, onset_control, alpha=0.5, color=col2,
                      s=20, edgecolors='black', linewidths=0.5)

        # Draw one boxplot per condition without changing the original plot layout.
        data_to_plot = [onset_primed, onset_control]
        bp = ax.boxplot(data_to_plot, labels=['Primed', 'Control'], patch_artist=True,
                        widths=0.6, showmeans=False,
                        boxprops={'edgecolor': 'black'},
                        whiskerprops={'color': 'black'},
                        capprops={'color': 'black'},
                        medianprops={'color': 'black', 'linewidth': 1},
                        flierprops={'markerfacecolor': 'gray', 'markeredgecolor': 'none',
                                   'marker': 'o', 'markersize': 4, 'alpha': 0.5})
        bp['boxes'][0].set_facecolor(col1)
        bp['boxes'][1].set_facecolor(col2)

        # Convert the corrected p-value into a compact significance marker.
        if not np.isnan(p_value_corrected):
            if p_value_corrected < 0.001:
                p_text = '***'
                fontsize = 14
                fontweight='bold'
            elif p_value_corrected < 0.01:
                p_text = '**'
                fontweight='bold'
            elif p_value_corrected < alpha:
                p_text = '*'
                fontweight='bold'
            else:
                p_text = 'n.s.'
                fontweight='normal'

            # Use the same vertical position for all windows.
            y_position = y_max_global

            ax.text(1.5, y_position, p_text, ha='center', va='bottom', fontsize=14, fontweight=fontweight, color='black')

        # Rotate x-axis tick labels for readability.
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

        if window_num == 1:
            ax.set_xticklabels(['Unclassified', 'Unclassified'], fontsize=fontsize_tick)
        else:
            ax.set_xticklabels(['Primed', 'Control'], fontsize=fontsize_tick)

        ax.set_ylabel('Onset latency (ms)' if i == 0 else '', fontsize=fontsize_axis_label)

        if i != 0:
            ax.tick_params(axis='y', labelleft=False)

        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(bottom=0, top=y_max_global * 1.07)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_onset_boxplots.pdf", bbox_inches='tight', dpi=300)
    plt.close()

    return fig


# Format p-values for permutation tests using the permutation-test resolution.
def format_p_value_permutation(p_val, n_permutations, min_decimals=1):

    if pd.isna(p_val):
        return 'NaN'

    # Resolution of the permutation test: the smallest non-zero p-value it can produce.
    resolution = 1 / (n_permutations + 1)

    if p_val <= resolution:
        mantissa, exponent = f"{resolution:.1e}".split('e')
        return f"< {mantissa}e{int(exponent)}"

    # Use enough decimal places to show the resolution without scientific notation.
    decimals_from_resolution = int(np.ceil(-np.log10(resolution)))
    decimals = max(decimals_from_resolution, min_decimals)

    return f"{p_val:.{decimals}f}"


# Export a compact Excel table summarizing onset statistics for each window.
def export_onset_table(results, output_dir, alpha, n_permutations):

    table_data = []

    # Format each window as one row in the output table.
    for res in results:
        p_corr = res.get('p_value_corrected', res['p_value'])
        statistic = res.get('statistic', np.nan)

        q1_primed = res['median_onset_primed_ms'] - res['iqr_primed'] / 2
        q3_primed = res['median_onset_primed_ms'] + res['iqr_primed'] / 2
        q1_control = res['median_onset_control_ms'] - res['iqr_control'] / 2
        q3_control = res['median_onset_control_ms'] + res['iqr_control'] / 2

        # Values at the permutation-test floor are reported as below the test
        # resolution; larger values are printed as fixed decimals.
        if not np.isnan(p_corr):
            p_permutation = format_p_value_permutation(p_corr, n_permutations)
            p_formatted = f"{p_permutation}*" if p_corr < alpha else p_permutation
        else:
            p_formatted = "NaN"

        row = {
            'Window': res['window'],
            'Time (ms)': res['time_range'],
            'Baseline (ms)': res['baseline_time_range'],
            'Baseline length (ms)': res['baseline_window'],
            'n (pairs)': res['n_neurons_valid'],
            'Primed (ms)': f"{res['median_onset_primed_ms']:.1f} [{q1_primed:.1f}, {q3_primed:.1f}]",
            'Control (ms)': f"{res['median_onset_control_ms']:.1f} [{q1_control:.1f}, {q3_control:.1f}]",
            'Median delta (ms)': f"{statistic:.1f}" if not np.isnan(statistic) else "NaN",
            'p-value': p_formatted
        }
        table_data.append(row)

    df_table = pd.DataFrame(table_data)
    df_table.to_excel(f"{output_dir}/combinated_onset_table.xlsx", index=False)

    return df_table


# Main onset lateny: compute onset-latency statistics, generate plots, and export summary tables.
def onset_latency(p_matrix, c_matrix, n_pres, n_neurons_total, n_windows, dt, output_dir, min_onset, max_onset, n_permutations, alpha, random_seed, correction_method, min_active_presentations, baseline_window):

    points_per_window = p_matrix.shape[1] // n_windows
    baseline_points = int(round(baseline_window / dt))

    if baseline_points < 1:
        raise ValueError("baseline_window must correspond to at least one time bin.")
    if baseline_points > points_per_window:
        raise ValueError("baseline_window cannot be longer than one temporal window.")

    all_results = []

    # Analyze windows 2..n only; window 1 is used as the baseline for window 2.
    for window_idx in range(1, n_windows):
        window_start_time = window_idx * points_per_window * dt
        window_duration_ms = points_per_window * dt
        col_start = window_idx * points_per_window
        col_end = (window_idx + 1) * points_per_window
        baseline_col_start = col_start - baseline_points
        baseline_col_end = col_start

        p_baseline = p_matrix[:, baseline_col_start:baseline_col_end]
        c_baseline = c_matrix[:, baseline_col_start:baseline_col_end]
        p_window = p_matrix[:, col_start:col_end]
        c_window = c_matrix[:, col_start:col_end]

        # Collect paired onset latencies for neurons that are valid in both conditions.
        onset_primed, onset_control, neuron_indices = filter_neurons_for_onset(
            p_window, c_window, p_baseline, c_baseline, n_pres, n_neurons_total, dt,
            min_onset, max_onset, min_active_presentations)

        stats = permutation_test_paired_onset(onset_primed, onset_control, n_permutations, random_seed)

        # Compute SEM only when valid onset values are available.
        if len(onset_primed) > 0:
            sem_primed = sem(onset_primed)
            sem_control = sem(onset_control)
        else:
            sem_primed = np.nan
            sem_control = np.nan

        # Store both summary statistics and raw onset values for plotting/export.
        all_results.append({
            'window': window_idx + 1,
            'window_start_ms': window_start_time,
            'window_end_ms': window_start_time + window_duration_ms,
            'time_range': f"{int(window_start_time)}-{int(window_start_time + window_duration_ms)} ms",
            'baseline_window': baseline_window,
            'baseline_time_range': f"{int((baseline_col_start) * dt)}-{int((baseline_col_end) * dt)} ms",
            'n_neurons_total': n_neurons_total,
            'n_neurons_valid': stats['n_pairs'],
            'median_onset_primed_ms': stats['median_primed'],
            'median_onset_control_ms': stats['median_control'],
            'median_onset_diff_ms': stats['median_diff'],
            'p_value': stats['p_value'],
            'statistic': stats['statistic'],
            'mean_onset_primed_ms': stats['mean_primed'],
            'mean_onset_control_ms': stats['mean_control'],
            'mean_onset_diff_ms': stats['mean_primed'] - stats['mean_control'],
            'sem_primed_ms': sem_primed,
            'sem_control_ms': sem_control,
            'range_primed_min': stats['range_primed'][0],
            'range_primed_max': stats['range_primed'][1],
            'range_control_min': stats['range_control'][0],
            'range_control_max': stats['range_control'][1],
            'iqr_primed': stats['iqr_primed'],
            'iqr_control': stats['iqr_control'],
            'neuron_indices': neuron_indices,
            'raw_onset_primed': onset_primed,
            'raw_onset_control': onset_control,
        })

    # Correct p-values across windows and generate the requested outputs.
    all_results = apply_multiple_comparisons_correction_onset(all_results, alpha, correction_method)

    plot_combined_onset_boxplots(all_results, output_dir, alpha, random_seed)

    export_onset_table(all_results, output_dir, alpha, n_permutations)

    return all_results
