import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
import scipy.stats as stats

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14
})

# Colors used across the firing-rate plots.
col_pr = (0.4, 0.6, 1.0, 0.7)       # Blue for Primed.
col_not_pr = (1.0, 0.4, 0.4, 0.7)   # Red for Control.
col_uncl = (0.6, 0.6, 0.6, 0.7)     # Gray for the unclassified window.


# Compute mean activity traces for each presentation by averaging contiguous neuron blocks.
def calculate_presentation_means(data, n_presentations, neurons_per_pres):

    presentation_means = []

    # Average the rows belonging to each presentation while preserving the time axis.
    for i in range(n_presentations):
        start = i * neurons_per_pres
        end = (i + 1) * neurons_per_pres
        pres_mean = np.mean(data[start:end, :], axis=0)
        presentation_means.append(pres_mean)

    return np.array(presentation_means)


# Identify contiguous clusters where the absolute statistic exceeds a threshold.
def find_clusters(stat_vector, threshold, min_size):

    above_thresh = np.abs(stat_vector) > threshold
    clusters = []

    # Scan the statistic vector and store only clusters that meet the minimum size.
    i = 0
    while i < len(stat_vector):
        if above_thresh[i]:
            start = i
            while i < len(stat_vector) and above_thresh[i]:
                i += 1
            end = i - 1

            if (end - start + 1) >= min_size:
                clusters.append((start, end))
        else:
            i += 1

    return clusters


# Estimate a dynamic cluster-forming threshold from sign-flip permutations.
def calculate_dynamic_threshold(p_means, c_means, n_permutations, alpha, random_seed):

    np.random.seed(random_seed)
    n_presentations, n_timepoints = p_means.shape
    diff_per_pres = p_means - c_means

    max_abs_ts = []

    # Build the null distribution using the maximum absolute t-statistic per permutation.
    for perm in range(n_permutations):
        signs = np.random.choice([1, -1], size=n_presentations)
        perm_diff = diff_per_pres * signs[:, np.newaxis]

        perm_mean = np.mean(perm_diff, axis=0)
        perm_std = np.std(perm_diff, axis=0, ddof=1)
        perm_t = perm_mean / (perm_std / np.sqrt(n_presentations)) # + 1e-10)
        perm_t = np.where(np.isnan(perm_t) | np.isinf(perm_t), 0, perm_t)

        max_abs_ts.append(np.max(np.abs(perm_t)))

    # Compute the cluster-forming threshold from the permutation distribution.
    threshold = np.percentile(max_abs_ts, (1 - alpha) * 100)

    return threshold, max_abs_ts


# Compute descriptive statistics for each timepoint across presentations.
def calculate_comprehensive_statistics(data_means):

    conf_level=0.95
    n_presentations, n_timepoints = data_means.shape

    # Compute the mean firing rate as a percentage.
    mean_vals = np.mean(data_means, axis=0) 

    # Compute the standard error of the mean as a percentage.
    sem_vals = np.std(data_means, axis=0, ddof=1) / np.sqrt(n_presentations)

    # Compute the 95% confidence interval using the t distribution.
    ci_multiplier = stats.t.ppf((1 + conf_level) / 2, n_presentations - 1)
    ci_lower = mean_vals - ci_multiplier * sem_vals
    ci_upper = mean_vals + ci_multiplier * sem_vals

    # Compute the interquartile range.
    q25 = np.percentile(data_means, 25, axis=0)
    q75 = np.percentile(data_means, 75, axis=0)
    iqr_vals = q75 - q25

    # Compute the median firing rate as a percentage.
    median_vals = np.median(data_means, axis=0)

    return {
        'mean': mean_vals,
        'sem': sem_vals,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'iqr': iqr_vals,
        'median': median_vals,
        'q25': q25,
        'q75': q75
    }


# Run a paired sign-flip permutation cluster test between Primed and Control traces.
def run_permutation_cluster_test(p_means, c_means, params, random_seed):

    np.random.seed(random_seed)
    n_presentations, n_timepoints = p_means.shape

    print(f"  Presentations: {n_presentations}, Timepoints: {n_timepoints}")

    # Compute the observed paired t-statistic at each timepoint.
    diff_per_pres = p_means - c_means
    mean_diff = np.mean(diff_per_pres, axis=0)
    std_diff = np.std(diff_per_pres, axis=0, ddof=1)
    t_obs = mean_diff / (std_diff / np.sqrt(n_presentations)) # + 1e-10)
    t_obs = np.where(np.isnan(t_obs) | np.isinf(t_obs), 0, t_obs)

    # Compute a dynamic threshold when no fixed threshold is provided.
    if params['t_threshold'] is None or params['t_threshold'] == 0:

        t_threshold, threshold_dist = calculate_dynamic_threshold(
            p_means, c_means,
            n_permutations=params['n_permutations'],
            alpha=params['alpha'],
            random_seed=random_seed
        )
        params['t_threshold'] = t_threshold

    # Detect observed clusters that exceed the selected threshold.
    clusters = find_clusters(t_obs, params['t_threshold'], params['min_cluster_size'])
    print(f"  Clusters found: {len(clusters)}")

    if not clusters:
        return t_obs, [], mean_diff, params

    # Run sign-flip permutations to create a null distribution of cluster statistics.
    max_cluster_stats = []

    for perm in range(params['n_permutations']):
        signs = np.random.choice([1, -1], size=n_presentations)
        perm_diff = diff_per_pres * signs[:, np.newaxis]

        perm_mean = np.mean(perm_diff, axis=0)
        perm_std = np.std(perm_diff, axis=0, ddof=1)
        perm_t = perm_mean / (perm_std / np.sqrt(n_presentations)) # + 1e-10)
        perm_t = np.where(np.isnan(perm_t) | np.isinf(perm_t), 0, perm_t)

        above_perm = np.abs(perm_t) > params['t_threshold']
        max_sum = 0
        current_sum = 0

        # Store the largest cluster statistic observed in this permutation.
        for t in range(n_timepoints):
            if above_perm[t]:
                current_sum += np.abs(perm_t[t])
                max_sum = max(max_sum, current_sum)
            else:
                current_sum = 0

        max_cluster_stats.append(max_sum)

    # Compute corrected p-values and descriptive information for each observed cluster.
    cluster_info = []

    for start, end in clusters:
        cluster_t = t_obs[start:end+1]
        cluster_sum_abs = np.sum(np.abs(cluster_t))
        cluster_sum = np.sum(cluster_t)

        p_val = (np.sum(np.array(max_cluster_stats) >= cluster_sum_abs) + 1) / (params['n_permutations'] + 1)

        direction = 'primed > control' if cluster_sum > 0 else 'primed < control'

        mean_p_primed = np.mean(np.mean(p_means[:, start:end+1], axis=1))
        mean_p_control = np.mean(np.mean(c_means[:, start:end+1], axis=1))
        mean_diff_cluster = mean_p_primed - mean_p_control

        cluster_info.append({
            'start_index': start,
            'end_index': end,
            'cluster_size': end - start + 1,
            't_sum': float(cluster_sum),
            't_abs_sum': float(cluster_sum_abs),
            'mean_t': float(np.mean(cluster_t)),
            'p_value': float(p_val),
            'direction': direction,
            'mean_p_primed': float(mean_p_primed),
            'mean_p_control': float(mean_p_control),
            'mean_diff': float(mean_diff_cluster),
            'significant': p_val < params['alpha']
        })

    return t_obs, cluster_info, mean_diff, params


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


# Build and save compact summary tables for descriptive statistics and significant clusters.
def tables(results, output_dir, primed_stats, control_stats, n_timepoints, dt):

    times = np.arange(n_timepoints) * dt

    # Number of permutations used by the cluster test, needed to scale p-value precision.
    n_permutations = results['params']['n_permutations']

    # Build the compact descriptive statistics table.
    compact_descriptive = pd.DataFrame()

    # Add the time axis.
    compact_descriptive['time_ms'] = times.round(3)

    # Add the t-statistic trace.
    compact_descriptive['t_statistic'] = results['t_stats'].round(3)

    # Add Primed mean and SEM values.
    primed_formatted = [f"{m:.3f} ({s:.3f})" for m, s in zip(
        primed_stats['mean'].round(3), primed_stats['sem'].round(3)
    )]
    compact_descriptive['primed_mean (sem)'] = primed_formatted

    # Add Control mean and SEM values.
    control_formatted = [f"{m:.3f} ({s:.3f})" for m, s in zip(
        control_stats['mean'].round(3), control_stats['sem'].round(3)
    )]

    compact_descriptive['control_mean (sem)'] = control_formatted

    # Add Primed median and quartiles.
    primed_median_formatted = [f"{med:.3f} [{q25:.3f}, {q75:.3f}]" for med, q25, q75 in zip(
        primed_stats['median'].round(3),
        primed_stats['q25'].round(3),
        primed_stats['q75'].round(3)
    )]
    compact_descriptive['primed_median (q25, q75)'] = primed_median_formatted

    # Add Control median and quartiles.
    control_median_formatted = [f"{med:.3f} [{q25:.3f}, {q75:.3f}]" for med, q25, q75 in zip(
        control_stats['median'].round(3),
        control_stats['q25'].round(3),
        control_stats['q75'].round(3)
    )]
    compact_descriptive['control_median (q25, q75)'] = control_median_formatted

    # Save the compact descriptive statistics table.
    compact_descriptive.to_excel(output_dir / f"compact_descriptive_statistics.xlsx", index=False)

    # Build a compact table for significant clusters when any are present.
    if results['significant_clusters']:
        compact_clusters = []

        for i, cluster in enumerate(results['significant_clusters'], 1):
            start_ms = cluster['start_index'] * dt
            end_ms = cluster['end_index'] * dt
            duration_ms = (cluster['end_index'] - cluster['start_index'] + 1) * dt

            # Round cluster-level values for readable reporting.
            start_ms_rounded = round(start_ms, 1)
            end_ms_rounded = round(end_ms, 1)
            duration_ms_rounded = round(duration_ms, 1)
            t_sum_rounded = round(cluster['t_sum'], 1)
            mean_t_rounded = round(cluster['mean_t'], 1)

            # Values at the permutation-test floor are reported as below the
            # test resolution; larger values are printed as fixed decimals.
            p_val_formatted = format_p_value_permutation(cluster['p_value'], n_permutations)

            # Compute descriptive statistics within the cluster time window.
            cluster_primed_mean = round(np.mean(primed_stats['mean'][cluster['start_index']:cluster['end_index']+1]), 1)
            cluster_primed_sem = round(np.mean(primed_stats['sem'][cluster['start_index']:cluster['end_index']+1]), 1)
            cluster_control_mean = round(np.mean(control_stats['mean'][cluster['start_index']:cluster['end_index']+1]), 1)
            cluster_control_sem = round(np.mean(control_stats['sem'][cluster['start_index']:cluster['end_index']+1]), 1)

            compact_clusters.append({
                'cluster_id': i,
                'time_window_ms': f'[{start_ms_rounded}, {end_ms_rounded}]',
                'duration_ms': duration_ms_rounded,
                'direction': cluster['direction'],
                'p_value': p_val_formatted,
                't_sum': t_sum_rounded,
                'mean_t': mean_t_rounded,
                'primed_mean (sem)': f'{cluster_primed_mean} ({cluster_primed_sem})',
                'control_mean (sem)': f'{cluster_control_mean} ({cluster_control_sem})'
            })

        compact_clusters_df = pd.DataFrame(compact_clusters)

        # Save the compact significant-cluster table.
        compact_clusters_df.to_excel(output_dir / f"compact_significant_clusters.xlsx", index=False)

    return {
        'descriptive': compact_descriptive,
        'clusters': compact_clusters if results['significant_clusters'] else None,
    }


# Analyze population-level firing-rate dynamics and save the corresponding plot and tables.
def population_level_firing_rate_dynamics(p_data, c_data, output_dir, n_presentations, dt, random_seed, n_permutations, alpha, min_cluster_size, t_threshold=None):

    # Check the input dimensions and infer the number of neurons per presentation.
    n_total_neurons, n_timepoints = p_data.shape

    if n_total_neurons % n_presentations != 0:
        print(f"{n_total_neurons} neurons are not divisible by {n_presentations} presentations!")
        n_presentations = n_total_neurons // (n_total_neurons // n_presentations)
        print(f" Using {n_presentations} presentations...")

    neurons_per_pres = n_total_neurons // n_presentations

    # Compute the mean activity trace for each presentation.
    p_pres_means = calculate_presentation_means(p_data, n_presentations, neurons_per_pres)
    c_pres_means = calculate_presentation_means(c_data, n_presentations, neurons_per_pres)

    # Compute descriptive statistics for both conditions.
    primed_stats = calculate_comprehensive_statistics(p_pres_means)
    control_stats = calculate_comprehensive_statistics(c_pres_means)

    # Keep these arrays for compatibility with the existing plotting code.
    mean_primed = primed_stats['mean']
    mean_control = control_stats['mean']
    sem_primed = primed_stats['sem']
    sem_control = control_stats['sem']

    # Store the statistical-test parameters in a single dictionary.
    params = {
        'n_total_neurons': n_total_neurons,
        'n_presentations': n_presentations,
        'neurons_per_pres': neurons_per_pres,
        'n_permutations': n_permutations,
        'alpha': alpha,
        't_threshold': t_threshold,
        'min_cluster_size': min_cluster_size,
        'random_seed': random_seed
    }

    # Run the permutation cluster test with a fixed random seed.
    t_stats, all_clusters, mean_diff, params = run_permutation_cluster_test(p_pres_means, c_pres_means, params, random_seed)

    # Keep only statistically significant clusters for reporting and plotting.
    significant_clusters = [c for c in all_clusters if c['significant']]

    # Compute the total significant duration for each effect direction.
    total_primed_gt = sum((c['end_index']-c['start_index']+1)*dt
                         for c in significant_clusters
                         if c['direction'] == 'primed > control')
    total_primed_lt = sum((c['end_index']-c['start_index']+1)*dt
                         for c in significant_clusters
                         if c['direction'] == 'primed < control')

    # Plot only significant clusters on top of the population-level dynamics.
    times = np.arange(n_timepoints) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6), gridspec_kw={'height_ratios': [2,1]}, sharex=True)

    # Top panel: mean firing-rate activity with SEM shading.
    # The first 300 ms are shown as unclassified for both conditions.
    first_window_end = 300  # ms
    mask_first = times <= first_window_end
    mask_rest = times > first_window_end

    ax1.plot(times[mask_first], mean_primed[mask_first], color=col_uncl, linewidth=1.5, label='Unclassified')
    ax1.plot(times[mask_first], mean_control[mask_first], color=col_uncl, linewidth=1.5)

    ax1.fill_between(times[mask_first], mean_primed[mask_first] - sem_primed[mask_first], mean_primed[mask_first] + sem_primed[mask_first], color=col_uncl, alpha=0.2, edgecolor='none')
    ax1.fill_between(times[mask_first], mean_control[mask_first] - sem_control[mask_first], mean_control[mask_first] + sem_control[mask_first], color=col_uncl, alpha=0.2, edgecolor='none')

    # Plot the remaining time window using the condition-specific colors.
    ax1.plot(times[mask_rest], mean_primed[mask_rest], label='Primed', color=col_pr, linewidth=1.5)
    ax1.plot(times[mask_rest], mean_control[mask_rest], label='Control', color=col_not_pr, linewidth=1.5)

    ax1.fill_between(times[mask_rest], mean_primed[mask_rest] - sem_primed[mask_rest], mean_primed[mask_rest] + sem_primed[mask_rest], color=col_pr, alpha=0.2, edgecolor='none')
    ax1.fill_between(times[mask_rest], mean_control[mask_rest] - sem_control[mask_rest], mean_control[mask_rest] + sem_control[mask_rest], color=col_not_pr, alpha=0.2, edgecolor='none')


    ax1.set_ylabel('Mean firing rate (Hz)', fontsize=16)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    ax1.grid(True, alpha=0.3)

    # Add vertical reference lines every 300 ms.
    for stim_onset in np.arange(0, times[-1] + 1, 300):
        if stim_onset <= times[-1]:
            ax1.axvline(x=stim_onset, color='black', linestyle='--', linewidth=0.5, alpha=0.4)
            ax2.axvline(x=stim_onset, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

    # Bottom panel: t-statistics with significant clusters highlighted.
    ax2.plot(times, t_stats, color='black', linewidth=0.8, alpha=0.9, label='t-statistic')
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)

    # Highlight significant clusters using the corresponding condition color.
    legend_patches = []

    for cluster in significant_clusters:
        start_time = cluster['start_index'] * dt
        end_time = cluster['end_index'] * dt

        if cluster['direction'] == 'primed > control':
            color = col_pr
            label = 'Primed > Control'
        else:
            color = col_not_pr
            label = 'Primed < Control'

        ax2.axvspan(start_time, end_time, alpha=0.3, color=color)

        # Add each cluster direction to the legend only once.
        if label not in [p.get_label() for p in legend_patches]:
            legend_patches.append(Patch(facecolor=color, alpha=0.3, label=label))

    ax2.set_xlabel('Time (ms)', fontsize=16)
    ax2.set_ylabel('t-statistic', fontsize=16)
    ax2.grid(True, alpha=0.3)

    if legend_patches:
        #ax2.legend(handles=legend_patches, loc='upper left', fontsize=14)
        ax2.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    xticks = np.linspace(0, times[-1] + dt, 6)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{int(x)}" for x in xticks])
    ax2.set_xlim(0, times[-1] + dt)

    plt.tight_layout()
    plot_path = output_dir / "firingrate.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Store the full analysis output for downstream use.
    results = {
        't_stats': t_stats,
        'all_clusters': all_clusters,
        'significant_clusters': significant_clusters,
        'total_primed_gt_control': total_primed_gt,
        'total_primed_lt_control': total_primed_lt,
        'mean_primed': mean_primed,
        'mean_control': mean_control,
        'mean_diff': mean_diff,
        'times': times,
        'P_pres_means': p_pres_means,
        'C_pres_means': c_pres_means,
        'params': params
    }

    compact_tables = tables(results, output_dir, primed_stats, control_stats, n_timepoints, dt)

    return results
