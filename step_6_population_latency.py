import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

# Colors used to compare Primed, Control, and Unclassified segments in plots.
blue_pastel = (0.4, 0.6, 1.0, 0.7)  # RGBA for the Primed condition.
red_pastel = (1.0, 0.4, 0.4, 0.7)   # RGBA for the Control condition.
gray_pastel = (0.6, 0.6, 0.6, 0.7)  # RGBA for the Unclassified segment.

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14
})

# Propagate the first activation to the right across each row.
def fill_right(matrix):
    filled = matrix.copy()

    # Once a neuron becomes active, keep it "active" for all following time points.
    for row in range(filled.shape[0]):
        activations = np.where(filled[row] == 1)[0]
        if len(activations) > 0:
            filled[row, activations[0]:] = 1

    return filled


# Split a vertically stacked activity matrix into individual trial matrices.
def split_stacked_matrix(matrix, neurons_per_trial):

    n_total_neurons = matrix.shape[0]
    n_trials = n_total_neurons // neurons_per_trial

    trials = []
    for i in range(n_trials):
        start_row = i * neurons_per_trial
        end_row = (i + 1) * neurons_per_trial
        trial_matrix = matrix[start_row:end_row, :]
        trials.append(trial_matrix)

    return trials


# Split a matrix into consecutive time segments along the column dimension.
def split_matrix_into_segments(matrix, n_segments):

    n_cols = matrix.shape[1]
    segment_size = n_cols // n_segments

    segments = []
    for i in range(n_segments):
        start_col = i * segment_size

        # Include all remaining columns in the last segment.
        if i == n_segments - 1:
            end_col = n_cols
        else:
            end_col = (i + 1) * segment_size

        segment = matrix[:, start_col:end_col]
        segments.append(segment)

    return segments


# Convert one segment into a normalized cumulative activation curve.
def process_segment_matrix(matrix_segment):

    filled = fill_right(matrix_segment)
    n_neurons = filled.shape[0]

    # Express cumulative activity as the percentage of active neurons over time.
    if n_neurons > 0 and filled.shape[1] > 0:
        normalized_curve = (np.sum(filled, axis=0) / n_neurons) * 100
    else:
        normalized_curve = np.zeros(filled.shape[1])

    return normalized_curve


# Analyze stacked Primed and Control matrices by extracting threshold-crossing latencies per segment.
def analyze_stacked_matrices(p_matrix, c_matrix, dt, threshold, n_segments, neurons_per_trial):

    # Split the stacked matrices into individual trials.
    p_trials = split_stacked_matrix(p_matrix, neurons_per_trial)
    c_trials = split_stacked_matrix(c_matrix, neurons_per_trial)

    n_trials = len(p_trials)

    # Initialize the nested result structure used by downstream analysis functions.
    analyses = {
        'priming': {seg: {'filtered': {'all_times': []}} for seg in range(n_segments)},
        'not_priming': {seg: {'filtered': {'all_times': []}} for seg in range(n_segments)}
    }

    # Process all Primed trials.
    for trial_idx, trial_matrix in enumerate(p_trials):
        trial_segments = split_matrix_into_segments(trial_matrix, n_segments)

        for seg in range(n_segments):
            activation_curve = process_segment_matrix(trial_segments[seg])

            # Store the first time point where the activation curve crosses the threshold.
            above_threshold = np.where(activation_curve >= threshold)[0]
            activation_time = above_threshold[0] * dt if len(above_threshold) > 0 else np.nan

            if not np.isnan(activation_time):
                analyses['priming'][seg]['filtered']['all_times'].append(activation_time)

    # Process all Control trials.
    for trial_idx, trial_matrix in enumerate(c_trials):
        trial_segments = split_matrix_into_segments(trial_matrix, n_segments)

        for seg in range(n_segments):
            activation_curve = process_segment_matrix(trial_segments[seg])

            # Store the first time point where the activation curve crosses the threshold.
            above_threshold = np.where(activation_curve >= threshold)[0]
            activation_time = above_threshold[0] * dt if len(above_threshold) > 0 else np.nan

            if not np.isnan(activation_time):
                analyses['not_priming'][seg]['filtered']['all_times'].append(activation_time)

    # Compute descriptive statistics for each segment and condition.
    for seg in range(n_segments):
        p_times = analyses['priming'][seg]['filtered']['all_times']
        if p_times:
            analyses['priming'][seg]['filtered']['stats'] = {
                'mean': np.mean(p_times),
                'median': np.median(p_times),
                'std': np.std(p_times),
                'n_valid': len(p_times)
            }
        else:
            analyses['priming'][seg]['filtered']['stats'] = {
                'mean': np.nan, 'median': np.nan, 'std': np.nan, 'n_valid': 0
            }

        c_times = analyses['not_priming'][seg]['filtered']['all_times']
        if c_times:
            analyses['not_priming'][seg]['filtered']['stats'] = {
                'mean': np.mean(c_times),
                'median': np.median(c_times),
                'std': np.std(c_times),
                'n_valid': len(c_times)
            }
        else:
            analyses['not_priming'][seg]['filtered']['stats'] = {
                'mean': np.nan, 'median': np.nan, 'std': np.nan, 'n_valid': 0
            }

    return analyses


# Format numeric values for compact result tables.
def fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NaN"
    if abs(value) < 0.0005 and value != 0:
        return f"{value:.3e}"
    return f"{value:.3f}"


# Format Mann-Whitney p-values in standard scientific notation.
def format_p_value_scientific(p_val, decimals=2):

    if p_val is None or (isinstance(p_val, float) and np.isnan(p_val)):
        return 'NaN'

    return f"{p_val:.{decimals}e}"


# Compare Primed and Control latency distributions with Mann-Whitney U tests for each segment.
def perform_mann_whitney_tests(directory, analyses, threshold, n_segments):
    results = []
    p_values = []

    for segment in range(n_segments):
        cond1_times = analyses['priming'][segment]['filtered']['all_times']
        cond2_times = analyses['not_priming'][segment]['filtered']['all_times']

        # Remove missing values before running the statistical test.
        cond1_clean = [x for x in cond1_times if not np.isnan(x)]
        cond2_clean = [x for x in cond2_times if not np.isnan(x)]

        # Skip segments that do not contain enough observations in both conditions.
        if len(cond1_clean) < 5 or len(cond2_clean) < 5:
            print(f"Segment {segment}: insufficient data (n1={len(cond1_clean)}, n2={len(cond2_clean)})")
            continue

        try:
            # Use a non-parametric test because latency distributions are not assumed to be normal.
            test_result = pg.mwu(cond1_clean, cond2_clean)

            # Compute the common language effect size for the non-parametric comparison.
            cohens_d = pg.compute_effsize(cond1_clean, cond2_clean, eftype='cles')

            # Compute quartiles and interquartile ranges for the compact summary table.
            q1_cond1 = np.percentile(cond1_clean, 25)
            q3_cond1 = np.percentile(cond1_clean, 75)
            q1_cond2 = np.percentile(cond2_clean, 25)
            q3_cond2 = np.percentile(cond2_clean, 75)
            iqr_cond1 = q3_cond1 - q1_cond1
            iqr_cond2 = q3_cond2 - q1_cond2

            result = {
                'segment': segment,
                'statistic': test_result['U-val'].values[0],
                'p_value': test_result['p-val'].values[0],
                'cohens_d': cohens_d,
                'mean_cond1': np.mean(cond1_clean),
                'mean_cond2': np.mean(cond2_clean),
                'median_cond1': np.median(cond1_clean),
                'median_cond2': np.median(cond2_clean),
                'std_cond1': np.std(cond1_clean),
                'std_cond2': np.std(cond2_clean),
                'iqr_cond1': iqr_cond1,
                'iqr_cond2': iqr_cond2,
                'q1_cond1': q1_cond1,
                'q3_cond1': q3_cond1,
                'q1_cond2': q1_cond2,
                'q3_cond2': q3_cond2,
            }

            results.append(result)
            p_values.append(result['p_value'])

        except Exception as e:
            print(f"Error in segment {segment}: {str(e)}")

    # Apply Holm correction across segment-wise tests.
    if p_values:
        rejected, p_values_corrected = pg.multicomp(p_values, alpha=0.05, method='holm')[:2]

        for i in range(len(results)):
            results[i]['p_value_corrected'] = p_values_corrected[i]
            results[i]['significant'] = rejected[i]

    # Convert results to a DataFrame with a stable column order.
    results_df = pd.DataFrame(results)

    column_order = ['segment', 'statistic', 'p_value',
        'p_value_corrected', 'significant', 'cohens_d', 'mean_cond1', 'mean_cond2', 'median_cond1', 'median_cond2',
        'std_cond1', 'std_cond2', 'iqr_cond1', 'iqr_cond2', 'q1_cond1', 'q3_cond1', 'q1_cond2', 'q3_cond2']

    # Keep only columns that exist in the current results.
    existing_columns = [col for col in column_order if col in results_df.columns]

    comparison_df = results_df[existing_columns]

    # Export p-values as text in scientific notation while keeping the returned
    # DataFrame numeric for plotting and threshold comparisons.
    comparison_export = comparison_df.copy()
    for p_col in ['p_value', 'p_value_corrected']:
        if p_col in comparison_export.columns:
            comparison_export[p_col] = comparison_export[p_col].apply(format_p_value_scientific)

    comparison_export.to_excel(directory / f"condition_comparisons_mann_whitney_thr{threshold}.xlsx", index=False)

    return comparison_df


# Create and save a compact Mann-Whitney summary table.
def create_compact_df(results_df, directory, threshold):
    compact_results = []

    for _, row in results_df.iterrows():
        summary_cond1 = f"{fmt(row['median_cond1'])} [{fmt(row['q1_cond1'])}, {fmt(row['q3_cond1'])}]"
        summary_cond2 = f"{fmt(row['median_cond2'])} [{fmt(row['q1_cond2'])}, {fmt(row['q3_cond2'])}]"

        compact_results.append({
            'segment': row['segment'],
            'Primed_median_[Q1,Q3]': summary_cond1,
            'Control_median_[Q1,Q3]': summary_cond2,
            'U_value': fmt(row['statistic']),
            'p_value_corrected': format_p_value_scientific(row['p_value_corrected']),
        })

    compact_df = pd.DataFrame(compact_results)
    compact_df.to_excel(directory / f"compact_mann_whitney_thr{threshold}.xlsx", index=False)

    return compact_df


# Plot latency distributions for two activation thresholds side by side.
def plot_segment_comparison_two_thresholds(results_df1, analyses1, results_df2, analyses2, directory, threshold1, threshold2, n_segments):

    # Create one panel per threshold without sharing the y-axis across panels.
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    # Compute the maximum whisker height for one threshold-specific dataset.
    def compute_max_for_dataset(analyses, n_segments):
        max_val = 0

        for segment in range(n_segments):
            for condition in ['priming', 'not_priming']:
                peaks_list = [t for t in analyses[condition][segment]['filtered']['all_times'] if not np.isnan(t)]
                if len(peaks_list) > 0:
                    peaks = np.array(peaks_list)

                    q75 = np.percentile(peaks, 75)
                    q25 = np.percentile(peaks, 25)
                    iqr = q75 - q25
                    upper_whisker_limit = q75 + 1.5 * iqr

                    data_within_whisker = peaks[peaks <= upper_whisker_limit]
                    if len(data_within_whisker) > 0:
                        actual_whisker_top = np.max(data_within_whisker)
                    else:
                        actual_whisker_top = q75

                    max_val = max(max_val, actual_whisker_top)

        return max_val

    # Compute independent y-axis reference values for the two thresholds.
    global_max_thr1 = compute_max_for_dataset(analyses1, n_segments)
    global_max_thr2 = compute_max_for_dataset(analyses2, n_segments)

    # Use a default value if one threshold has no valid observations.
    if global_max_thr1 == 0:
        global_max_thr1 = 1.0
    if global_max_thr2 == 0:
        global_max_thr2 = 1.0

    # Add vertical space for the significance label in each panel.
    significance_y_thr1 = global_max_thr1 * 1.07
    significance_y_thr2 = global_max_thr2 * 1.07

    # Draw one threshold-specific plot inside an existing panel axis.
    def plot_on_axis(ax, results_df, analyses, threshold, title_label, significance_y, global_max):
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs = GridSpecFromSubplotSpec(1, n_segments, subplot_spec=ax.get_subplotspec(), wspace=0.4, width_ratios=[1]*n_segments)

        axes_segments = [fig.add_subplot(gs[0, i]) for i in range(n_segments)]

        # Share the y-axis only between segments belonging to the same threshold panel.
        if n_segments > 1:
            for seg_idx in range(1, n_segments):
                axes_segments[seg_idx].sharey(axes_segments[0])

        for segment, ax_seg in enumerate(axes_segments):
            priming_data = [t for t in analyses['priming'][segment]['filtered']['all_times'] if not np.isnan(t)]
            not_priming_data = [t for t in analyses['not_priming'][segment]['filtered']['all_times'] if not np.isnan(t)]

            # The first segment is treated as unclassified and plotted in gray.
            is_gray_segment = (segment == 0)
            primed_color = gray_pastel if is_gray_segment else blue_pastel
            control_color = gray_pastel if is_gray_segment else red_pastel
            labels = ['Unclassified', 'Unclassified'] if is_gray_segment else ['Primed', 'Control']

            # Add jittered scatter points to show individual latency values.
            if priming_data:
                x_priming = np.random.normal(1, 0.04, len(priming_data))
                ax_seg.scatter(x_priming, priming_data, alpha=0.5, color=primed_color, s=20, edgecolors='black', linewidths=0.5)
            if not_priming_data:
                x_not_priming = np.random.normal(2, 0.04, len(not_priming_data))
                ax_seg.scatter(x_not_priming, not_priming_data, alpha=0.5, color=control_color, s=20, edgecolors='black', linewidths=0.5)

            # Draw boxplots for Primed and Control latency distributions.
            data_to_plot = [priming_data if priming_data else [np.nan], not_priming_data if not_priming_data else [np.nan]]

            bp = ax_seg.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6,
                                showmeans=False, boxprops={'facecolor': primed_color, 'edgecolor': 'black'},
                                whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                                medianprops={'color': 'black', 'linewidth': 1},
                                flierprops={'markerfacecolor': 'gray', 'markeredgecolor': 'none'})

            if not is_gray_segment:
                bp['boxes'][1].set_facecolor(control_color)

            ax_seg.tick_params(axis='x', rotation=45)
            for label in ax_seg.get_xticklabels():
                label.set_horizontalalignment('right')

            if segment == 0:
                ax_seg.set_ylabel('Time (ms)')
            else:
                ax_seg.set_ylabel('')
                ax_seg.tick_params(axis='y', labelleft=False)

            ax_seg.grid(True, linestyle=':', alpha=0.6)
            ax_seg.set_ylim(0, significance_y * 1.07)

            # Add significance markers using corrected p-values when available.
            if not results_df.empty:
                segment_result = results_df[(results_df['segment'] == segment)]
                if not segment_result.empty:
                    p_val = segment_result['p_value_corrected'].values[0]
                    
                    if p_val < 0.001:
                        significance = '***'
                        fontweight='bold'
                        
                    elif p_val < 0.01:
                        significance = '**'
                        fontweight='bold'
                        
                    elif p_val < 0.05:
                        significance = '*'
                        fontweight='bold'
                        
                    else:
                        significance = 'n.s.'
                        fontweight='normal'

                    y_position = significance_y

                    ax_seg.text(1.5, y_position, significance, ha='center', va='bottom',
                               fontsize=14, fontweight=fontweight, color='black')

        # Hide the container axis and use it only as a panel title holder.
        ax.axis('off')
        ax.set_title(title_label, fontsize=18, pad=20)

    # Draw the two threshold-specific panels.
    plot_on_axis(ax_left, results_df1, analyses1, threshold1,
                f'(a) Latency at {threshold1}% neuronal activation',
                significance_y_thr1, global_max_thr1)
    plot_on_axis(ax_right, results_df2, analyses2, threshold2,
                f'(b) Latency at {threshold2}% neuronal activation',
                significance_y_thr2, global_max_thr2)

    plt.tight_layout()
    plt.savefig(directory / f"comparison_thr{threshold1}_thr{threshold2}.pdf", bbox_inches='tight')
    plt.close()


# Run the full population latency analysis for two activation thresholds.
def population_latency(p_matrix, c_matrix, dt, threshold1, threshold2, n_segments, n_pres, output_dir):

    neurons_per_trial = p_matrix.shape[0] // n_pres

    # Split the matrices into trials only once and reuse them for both thresholds.
    p_trials = split_stacked_matrix(p_matrix, neurons_per_trial)
    c_trials = split_stacked_matrix(c_matrix, neurons_per_trial)

    # Initialize result containers for the two activation thresholds.
    analyses1 = {
        'priming': {seg: {'filtered': {'all_times': []}} for seg in range(n_segments)},
        'not_priming': {seg: {'filtered': {'all_times': []}} for seg in range(n_segments)}
    }
    analyses2 = {
        'priming': {seg: {'filtered': {'all_times': []}} for seg in range(n_segments)},
        'not_priming': {seg: {'filtered': {'all_times': []}} for seg in range(n_segments)}
    }

    # Extract threshold-crossing latencies for both thresholds and both conditions.
    for cond_name, trials in [('priming', p_trials), ('not_priming', c_trials)]:
        for trial_matrix in trials:
            trial_segments = split_matrix_into_segments(trial_matrix, n_segments)

            for seg in range(n_segments):
                activation_curve = process_segment_matrix(trial_segments[seg])

                # Apply the first activation threshold.
                above_threshold1 = np.where(activation_curve >= threshold1)[0]
                activation_time1 = above_threshold1[0] * dt if len(above_threshold1) > 0 else np.nan
                if not np.isnan(activation_time1):
                    analyses1[cond_name][seg]['filtered']['all_times'].append(activation_time1)

                # Apply the second activation threshold.
                above_threshold2 = np.where(activation_curve >= threshold2)[0]
                activation_time2 = above_threshold2[0] * dt if len(above_threshold2) > 0 else np.nan
                if not np.isnan(activation_time2):
                    analyses2[cond_name][seg]['filtered']['all_times'].append(activation_time2)

    # Compute descriptive statistics for every segment, condition, and threshold.
    for analyses in [analyses1, analyses2]:
        for seg in range(n_segments):
            for cond in ['priming', 'not_priming']:
                times = analyses[cond][seg]['filtered']['all_times']
                if times:
                    analyses[cond][seg]['filtered']['stats'] = {
                        'mean': np.mean(times),
                        'median': np.median(times),
                        'std': np.std(times),
                        'n_valid': len(times)
                    }
                else:
                    analyses[cond][seg]['filtered']['stats'] = {
                        'mean': np.nan, 'median': np.nan, 'std': np.nan, 'n_valid': 0
                    }

    # Run statistical analysis and export tables for the first threshold.
    results_df1 = perform_mann_whitney_tests(output_dir, analyses1, threshold1, n_segments)

    if results_df1.empty:
        print("\nWARNING: No valid statistical comparisons could be performed for threshold 1.")
        compact_df1 = pd.DataFrame()
    else:
        compact_df1 = create_compact_df(results_df1, output_dir, threshold1)

    # Run statistical analysis and export tables for the second threshold.
    results_df2 = perform_mann_whitney_tests(output_dir, analyses2, threshold2, n_segments)

    if results_df2.empty:
        print("\nWARNING: No valid statistical comparisons could be performed for threshold 2.")
        compact_df2 = pd.DataFrame()
    else:
        compact_df2 = create_compact_df(results_df2, output_dir, threshold2)

    plot_segment_comparison_two_thresholds(results_df1, analyses1, results_df2, analyses2, output_dir, threshold1, threshold2, n_segments)

    return {'threshold1': {'analyses': analyses1, 'results': results_df1, 'compact': compact_df1},
            'threshold2': {'analyses': analyses2, 'results': results_df2, 'compact': compact_df2}}
