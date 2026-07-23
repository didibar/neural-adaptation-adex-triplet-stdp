import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import pingouin as pg


warnings.filterwarnings("ignore")


type_names_display = {
    "sharpening": "Sharpening",
    "fatiguing": "Fatiguing",
    "None": "None",
    "none": "None",
    "tie": "Tie",
    "unknown": "Unknown",
}


def _table_label(value):
    # Use lowercase none only in exported tables, without changing the analysis labels.
    if pd.isna(value):
        return "none"
    if value == "None":
        return "none"
    return value


def _lowercase_none_labels(df):
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace({"None": "none"})

    return df


def _safe_round(value, decimals=3):
    if pd.isna(value):
        return np.nan
    return round(value, decimals)


def _safe_count(value):
    if pd.isna(value):
        return 0
    return int(value)


def _format_mean_std(mean_value, std_value):
    if pd.isna(mean_value):
        return "N/A"

    if pd.isna(std_value):
        return f"{mean_value:.3f} (N/A)"

    return f"{mean_value:.3f} ({std_value:.3f})"


def _format_median_iqr(median_value, q1_value, q3_value):
    if pd.isna(median_value) or pd.isna(q1_value) or pd.isna(q3_value):
        return "N/A"

    return f"{median_value:.3f} [{q1_value:.3f}, {q3_value:.3f}]"

# part 2: cluster analysis and classification

# Create a summary table from step_4 results for all configuration folders.
def create_cluster_summary(main_results_dir):

    main_path = Path(main_results_dir)

    # Check that the main results folder exists.
    if not main_path.exists():
        raise FileNotFoundError(f"folder {main_path} not found")

    # Pattern used to identify configuration folders containing alpha, beta, and eps.
    param_pattern = r"alpha([0-9.]+)_beta([0-9.]+)_eps([0-9.]+)"                           
    config_dirs = []

    # Find all configuration folders matching the parameter pattern.
    for directory in main_path.iterdir():
        if directory.is_dir() and re.search(param_pattern, directory.name):
            config_dirs.append(directory)

    all_rows = []

    for config_dir in config_dirs:
        config_name = config_dir.name

        # Extract alpha, beta, and eps from the folder name.
        match = re.search(param_pattern, config_name)
        if match:
            alpha = float(match.group(1))
            beta = float(match.group(2))
            eps = float(match.group(3))
        else:
            alpha = np.nan
            beta = np.nan
            eps = np.nan

        # Initialize the row for the current configuration.
        row = {
            "config_name": config_name,
            "alpha": alpha,
            "beta": beta,
            "eps": eps,
            "duration_primed_gt_control": 0,
            "duration_primed_lt_control": 0,
        }

        # Read the step_4 file containing significant clusters, if available.
        step4_file = config_dir / "step_4" / "compact_significant_clusters.xlsx"

        if step4_file.exists():
            try:
                #df_clusters = pd.read_excel(step4_file)
                df_clusters = pd.read_excel(step4_file)

                # Sum cluster durations for each direction.
                primed_gt = df_clusters[
                    df_clusters["direction"] == "primed > control"
                ]["duration_ms"].sum()

                primed_lt = df_clusters[
                    df_clusters["direction"] == "primed < control"
                ]["duration_ms"].sum()

                row["duration_primed_gt_control"] = primed_gt
                row["duration_primed_lt_control"] = primed_lt

            except Exception as exc:
                # Continue with the next configuration if this file cannot be read.
                print(
                    f"  error reading compact_significant_clusters.xlsx "
                    f"in {config_name}: {exc}"
                )

        # Add the current configuration row to the final table.
        all_rows.append(row)

    # Build the final summary dataframe.
    df_summary = pd.DataFrame(all_rows)

    return df_summary


# Merge cluster and delta dataframes using config_name as the common key.
def merge_data(df_cluster, df_delta):

    # Keep only configurations that are present in both dataframes.
    df_merged = pd.merge(
        df_cluster,
        df_delta,
        on="config_name",
        how="inner",
    )

    # Return early if no common configuration is found.
    if len(df_merged) == 0:
        print("no common configurations found")
        return None

    # If duplicated parameter columns exist, keep the values from df_cluster.
    if "alpha_x" in df_merged.columns:
        df_merged["alpha"] = df_merged["alpha_x"]
        df_merged["beta"] = df_merged["beta_x"]
        df_merged["eps"] = df_merged["eps_x"]

        # Drop duplicated parameter columns created by the merge.
        df_merged = df_merged.drop(
            columns=[
                "alpha_x",
                "alpha_y",
                "beta_x",
                "beta_y",
                "eps_x",
                "eps_y",
            ]
        )

    return df_merged


# Classify each configuration according to the most numerous neuron type.
def classify_configurations_by_count(df, type_names):

    df = df.copy()

    count_columns = []
    type_list = []

    # Add the left class if its count column is available.
    left_name = type_names["left"]
    n_left_col = f"n_{left_name}"

    if n_left_col in df.columns:
        count_columns.append(n_left_col)
        type_list.append(left_name)

    # Add the middle class if it is defined and available.
    if type_names.get("middle"):
        middle_name = type_names["middle"]
        n_middle_col = f"n_{middle_name}"

        if n_middle_col in df.columns:
            count_columns.append(n_middle_col)
            type_list.append(middle_name)

    # Add the right class if its count column is available.
    right_name = type_names["right"]
    n_right_col = f"n_{right_name}"

    if n_right_col in df.columns:
        count_columns.append(n_right_col)
        type_list.append(right_name)

    # If no count columns are available, assign unknown.
    if len(count_columns) == 0:
        print("warning: no count columns found for configuration classification")
        df["type"] = "unknown"
        return df

    # Find the index of the maximum count for each row.
    max_idx = df[count_columns].values.argmax(axis=1)

    # Convert the maximum-count index into the corresponding class name.
    df["type"] = [type_list[idx] for idx in max_idx]

    # Always keep ties as ties; they are not reclassified later.
    max_values = df[count_columns].max(axis=1)
    tie_mask = df[count_columns].eq(max_values, axis=0).sum(axis=1) > 1
    df.loc[tie_mask, "type"] = "tie"

    return df


# Prepare configuration-level classification from neuron counts.
def prepare_data_for_analysis(df, type_names):

    # Use the default three-class scheme if no class dictionary is provided.
    if type_names is None:
        type_names = {
            "left": "sharpening",
            "middle": "fatiguing",
            "right": "None",
        }

    # Remove configurations without duration or total-neuron information.
    df = df.dropna(
        subset=[
            "duration_primed_lt_control",
            "total_neurons",
        ]
    )

    return classify_configurations_by_count(df, type_names)


# Create a two-panel 3d figure:
    # left panel: configuration distribution by type;
    # right panel: significant-window duration in parameter space.
def plot_3d_comparison(df, type_col, save_dir, method_name):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 14})

    fontsize_title = 18
    fontsize_axis_label = 16
    fontsize_tick = 16
    fontsize_legend = 16

    colors_custom = {
        "sharpening": (1.0, 0.6, 0.2, 0.7),
        "fatiguing": (0.7, 0.5, 0.9, 0.7),
    }

    # Keep only the two informative classes for this comparison figure.
    df_filtered = df[df[type_col].isin(["sharpening", "fatiguing"])].copy()

    if len(df_filtered) == 0:
        print(f"no data after filtering for {method_name}")
        return None

    # Sort classes to show sharpening and fatiguing in a stable order.
    unique_types = [
        type_name
        for type_name in df_filtered[type_col].unique()
        if type_name not in ["tie", "unknown"]
    ]

    unique_types_sorted = sorted(
        unique_types,
        key=lambda type_name: (
            type_name != "sharpening",
            type_name != "fatiguing",
        ),
    )

    type_names_plot = [
        type_names_display.get(type_name, type_name.capitalize())
        for type_name in unique_types_sorted
    ]

    point_size = 80

    # Create the figure with two side-by-side 3d panels.
    fig = plt.figure(figsize=(12, 6))
    grid = gridspec.GridSpec(
        1,
        2,
        width_ratios=[5, 5],
        wspace=0.15,
        figure=fig,
    )

    # left panel: type distribution in parameter space

    ax1 = fig.add_subplot(grid[0, 0], projection="3d")

    for type_name, type_display in zip(unique_types_sorted, type_names_plot):
        mask = df_filtered[type_col] == type_name
        data = df_filtered[mask]

        if len(data) > 0:
            color = colors_custom.get(type_name, "gray")

            ax1.scatter(
                data["alpha"],
                data["beta"],
                data["eps"],
                c=[color],
                label=f"{type_display}",
                s=point_size,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

    ax1.set_xlabel(r"$\alpha$", fontsize=fontsize_axis_label, labelpad=10)
    ax1.set_ylabel(r"$\beta$", fontsize=fontsize_axis_label, labelpad=10)
    ax1.set_zlabel(
        r"$\epsilon$",
        fontsize=fontsize_axis_label,
        labelpad=10,
        rotation=0,
    )

    ax1.legend(loc="upper left", fontsize=fontsize_legend)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(True, alpha=0.3)
    ax1.set_title("(a) Input parameters", fontsize=fontsize_title)

    # right panel: significant-window duration in parameter space

    ax2 = fig.add_subplot(grid[0, 1], projection="3d")

    vmin_dur = df_filtered["duration_primed_lt_control"].min()
    vmax_dur = df_filtered["duration_primed_lt_control"].max()

    scatter_dur = ax2.scatter(
        df_filtered["alpha"],
        df_filtered["beta"],
        df_filtered["eps"],
        c=df_filtered["duration_primed_lt_control"],
        cmap="viridis",
        s=point_size,
        alpha=0.8,
        vmin=vmin_dur,
        vmax=vmax_dur,
        edgecolors="black",
        linewidth=0.5,
    )

    ax2.set_xlabel(r"$\alpha$", fontsize=fontsize_axis_label, labelpad=10)
    ax2.set_ylabel(r"$\beta$", fontsize=fontsize_axis_label, labelpad=10)
    ax2.set_zlabel(
        r"$\epsilon$",
        fontsize=fontsize_axis_label,
        labelpad=10,
        rotation=0,
    )

    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.grid(True, alpha=0.3)
    ax2.set_title("(b) Significant window length", fontsize=fontsize_title)

    # Add a small horizontal colorbar in the upper-right area of the figure.
    cbar_ax = fig.add_axes([0.75, 0.8, 0.2, 0.03])
    cbar_dur = plt.colorbar(
        scatter_dur,
        cax=cbar_ax,
        orientation="horizontal",
    )

    cbar_dur.ax.tick_params(labelsize=fontsize_tick)
    cbar_dur.set_label(
        "Primed < Control (ms)",
        fontsize=fontsize_axis_label,
        labelpad=5,
    )

    # Use the same camera angle for both 3d panels.
    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=25, azim=-60)

    plt.tight_layout()

    filename_dur = f"3d_comparison_{method_name.replace(' ', '_').lower()}.pdf"
    plt.savefig(save_dir / filename_dur, bbox_inches="tight")
    plt.close()

    return fig


# Generate the tables used for paper reporting: configuration summary and statistical tests.
def generate_tables(df, type_col, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Remove configurations labeled as None.
    df_filtered = df[~df[type_col].isin(["None", "none"])].copy()

    # table 1: per-configuration summary

    table1_rows = []

    for _, row in df_filtered.iterrows():
        # Extract configuration parameters.
        alpha = row["alpha"]
        beta = row["beta"]
        eps = row["eps"]
        total_neurons = row["total_neurons"]

        # Read delta summary statistics.
        mean_delta = row["mean_delta"]
        std_delta = row["std_delta"] if "std_delta" in row.index else np.nan

        # Read median and quartiles if available.
        if "median_delta" in row.index:
            median_delta = row["median_delta"]
            q1_delta = row.get("q1_delta", np.nan)
            q3_delta = row.get("q3_delta", np.nan)
        else:
            median_delta = np.nan
            q1_delta = np.nan
            q3_delta = np.nan

        # Format the median and interquartile range.
        median_iqr_str = _format_median_iqr(median_delta, q1_delta, q3_delta)

        duration_lt = row["duration_primed_lt_control"]
        duration_gt = row["duration_primed_gt_control"]
        n_sharp = row["n_sharpening"]
        n_fat = row["n_fatiguing"]
        n_none = row["n_None"] if "n_None" in row.index else 0
        type_maj = row[type_col]

        # Add the formatted configuration row to table 1.
        table1_rows.append(
            {
                "alpha": _safe_round(alpha),
                "beta": _safe_round(beta),
                "eps": _safe_round(eps),
                "total_neurons": total_neurons,
                "mean_delta_(std)": _format_mean_std(mean_delta, std_delta),
                "median_delta_[Q1_Q3]": median_iqr_str,
                "duration_primed_lt_control": _safe_round(duration_lt),
                "duration_primed_gt_control": _safe_round(duration_gt),
                "n_sharpening": _safe_count(n_sharp),
                "n_fatiguing": _safe_count(n_fat),
                "n_None": _safe_count(n_none),
                "type": _table_label(type_maj),
            }
        )

    table1 = _lowercase_none_labels(pd.DataFrame(table1_rows))

    # Save table 1.
    table1_path = save_dir / "Table1_Summary.xlsx"
    table1.to_excel(table1_path, index=False)

    # table 2: statistical tests for parameters

    # Keep only sharpening and fatiguing configurations for statistical tests.
    df_comp = df_filtered[
        df_filtered[type_col].isin(["sharpening", "fatiguing"])
    ].copy()

    table2_data = []

    for param, param_label in zip(
        ["alpha", "beta", "eps"],
        ["alpha", "beta", "epsilon"],
    ):
        # Extract parameter values for each class.
        sharp_vals = df_comp[df_comp[type_col] == "sharpening"][param].dropna()
        fat_vals = df_comp[df_comp[type_col] == "fatiguing"][param].dropna()

        if len(sharp_vals) > 0 and len(fat_vals) > 0:
            # Compute median and interquartile range for sharpening.
            sharp_median = sharp_vals.median()
            sharp_q1 = sharp_vals.quantile(0.25)
            sharp_q3 = sharp_vals.quantile(0.75)
            sharp_str = f"{sharp_median:.3f} [{sharp_q1:.3f}, {sharp_q3:.3f}]"

            # Compute median and interquartile range for fatiguing.
            fat_median = fat_vals.median()
            fat_q1 = fat_vals.quantile(0.25)
            fat_q3 = fat_vals.quantile(0.75)
            fat_str = f"{fat_median:.3f} [{fat_q1:.3f}, {fat_q3:.3f}]"

            # Run a two-sided Mann-Whitney U test.
            u_stat, p_val = mannwhitneyu(
                sharp_vals,
                fat_vals,
                alternative="two-sided",
            )

            u_stat = round(u_stat, 2)

            # Format the p-value with significance markers.
            if p_val < 0.001:
                p_formatted = f"{p_val:.2e}***"
            elif p_val < 0.01:
                p_formatted = f"{p_val:.4f}**"
            elif p_val < 0.05:
                p_formatted = f"{p_val:.4f}*"
            else:
                p_formatted = f"{p_val:.4f}"

        else:
            sharp_str = "N/A"
            fat_str = "N/A"
            u_stat = "N/A"
            p_formatted = "N/A"

        # Add the current parameter result to table 2.
        table2_data.append(
            {
                "parameter": param_label,
                "median_sharpening_[Q1_Q3]": sharp_str,
                "median_fatiguing_[Q1_Q3]": fat_str,
                "U": u_stat,
                "p-value": p_formatted,
            }
        )

    # Add a final row with the number of configurations in each class.
    n_sharp = (df_comp[type_col] == "sharpening").sum()
    n_fat = (df_comp[type_col] == "fatiguing").sum()

    table2_data.append(
        {
            "parameter": "n_configurations",
            "median_sharpening_[Q1_Q3]": str(n_sharp),
            "median_fatiguing_[Q1_Q3]": str(n_fat),
            "U": "",
            "p-value": "",
        }
    )

    table2 = _lowercase_none_labels(pd.DataFrame(table2_data))
    table2.to_excel(
        save_dir / "Table2_Parameters_Statistical_Tests.xlsx",
        index=False,
    )

    # table 3: statistical test for duration

    duration_col = "duration_primed_lt_control"

    if duration_col in df_comp.columns:
        sharp_dur = df_comp[df_comp[type_col] == "sharpening"][duration_col].dropna()
        fat_dur = df_comp[df_comp[type_col] == "fatiguing"][duration_col].dropna()

        if len(sharp_dur) > 0 and len(fat_dur) > 0:
            # Compute median and interquartile range for sharpening.
            sharp_median = sharp_dur.median()
            sharp_q1 = sharp_dur.quantile(0.25)
            sharp_q3 = sharp_dur.quantile(0.75)
            sharp_str = f"{sharp_median:.3f} [{sharp_q1:.3f}, {sharp_q3:.3f}]"

            # Compute median and interquartile range for fatiguing.
            fat_median = fat_dur.median()
            fat_q1 = fat_dur.quantile(0.25)
            fat_q3 = fat_dur.quantile(0.75)
            fat_str = f"{fat_median:.3f} [{fat_q1:.3f}, {fat_q3:.3f}]"

            # Run a two-sided Mann-Whitney U test on duration values.
            u_stat, p_val = mannwhitneyu(
                sharp_dur,
                fat_dur,
                alternative="two-sided",
            )

            u_stat = round(u_stat, 2)

            # Format the p-value with significance markers.
            if p_val < 0.001:
                p_formatted = f"{p_val:.2e}***"
            elif p_val < 0.01:
                p_formatted = f"{p_val:.4f}**"
            elif p_val < 0.05:
                p_formatted = f"{p_val:.4f}*"
            else:
                p_formatted = f"{p_val:.4f}"

        else:
            sharp_str = "N/A"
            fat_str = "N/A"
            u_stat = "N/A"
            p_formatted = "N/A"

        table3_data = [
            {
                "condition": "Sharpening vs Fatiguing",
                "median_sharpening_[Q1_Q3]_(ms)": sharp_str,
                "median_fatiguing_[Q1_Q3]_(ms)": fat_str,
                "U": u_stat,
                "p-value": p_formatted,
            }
        ]

        # Add a final row with the number of configurations in each class.
        table3_data.append(
            {
                "condition": "n_configurations",
                "median_sharpening_[Q1_Q3]_(ms)": str(len(sharp_dur)),
                "median_fatiguing_[Q1_Q3]_(ms)": str(len(fat_dur)),
                "U": "",
                "p-value": "",
            }
        )

        table3 = _lowercase_none_labels(pd.DataFrame(table3_data))
        table3.to_excel(
            save_dir / "Table3_Duration_Statistical_Test.xlsx",
            index=False,
        )

    else:
        table3 = None

    return {
        "table1": table1,
        "table2": table2,
        "table3": table3,
    }


def export_table(df, save_dir, filename, phenotype_col="type"):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_excel = df.copy()
    phenotype_source = phenotype_col if phenotype_col in df_excel.columns else "type"

    table_rows = []

    for _, row in df_excel.iterrows():
        mean_delta = row.get("mean_delta", np.nan)
        std_delta = row.get("std_delta", np.nan)
        median_delta = row.get("median_delta", np.nan)
        q1_delta = row.get("q1_delta", np.nan)
        q3_delta = row.get("q3_delta", np.nan)

        table_rows.append(
            {
                "alpha": _safe_round(row.get("alpha", np.nan)),
                "beta": _safe_round(row.get("beta", np.nan)),
                "eps": _safe_round(row.get("eps", np.nan)),
                "mean_delta_(std)": _format_mean_std(mean_delta, std_delta),
                "median_delta_[Q1_Q3]": _format_median_iqr(
                    median_delta,
                    q1_delta,
                    q3_delta,
                ),
                "duration_primed_lt_control": _safe_round(
                    row.get("duration_primed_lt_control", np.nan)
                ),
                "duration_primed_gt_control": _safe_round(
                    row.get("duration_primed_gt_control", np.nan)
                ),
                "n_sharpening": _safe_count(row.get("n_sharpening", 0)),
                "n_fatiguing": _safe_count(row.get("n_fatiguing", 0)),
                "n_None": _safe_count(row.get("n_None", 0)),
                "type": _table_label(row.get(phenotype_source, "none")),
            }
        )

    df_excel = _lowercase_none_labels(pd.DataFrame(table_rows))

    excel_path = save_dir / filename
    df_excel.to_excel(excel_path, index=False)

    return df_excel, excel_path


# Create a side-by-side figure with boxplots:
    # the first three panels compare alpha, beta, and eps;
    # the fourth panel compares primed < control duration.
def create_side_by_side_figure(df_filtered, save_dir, method_name, type_col):

    # Keep only the two informative classes for this comparison figure.
    df_plot = df_filtered[df_filtered[type_col].isin(["sharpening", "fatiguing"])].copy()

    if len(df_plot) == 0:
        print("no data after filtering")
        return None

    # Keep only available classes among sharpening and fatiguing.
    available_types = [
        type_name
        for type_name in ["sharpening", "fatiguing"]
        if type_name in df_plot[type_col].unique()
    ]

    type_display_names = [
        type_names_display.get(type_name, type_name.capitalize())
        for type_name in available_types
    ]

    colors_custom = {
        "sharpening": (1.0, 0.6, 0.2, 0.7),
        "fatiguing": (0.7, 0.5, 0.9, 0.7),
    }

    type_colors = [
        colors_custom.get(type_name, "gray")
        for type_name in available_types
    ]

    # Define the parameters shown in the first three panels.
    params = ["alpha", "beta", "eps"]
    param_labels = [r"$\alpha$", r"$\beta$", r"$\epsilon$"]
    duration_col = "duration_primed_lt_control"

    # calculate global y-limits for alpha, beta, and eps

    all_param_values = []

    for param in params:
        for type_name in available_types:
            mask = df_plot[type_col] == type_name
            values = df_plot.loc[mask, param].dropna().values
            all_param_values.extend(values)

    if len(all_param_values) > 0:
        param_ymin = np.min(all_param_values)
        param_ymax = np.max(all_param_values)
        param_padding = (param_ymax - param_ymin) * 0.15
        param_ylim = (
            param_ymin - param_padding,
            param_ymax + param_padding,
        )
    else:
        param_ylim = (0, 1)

    # calculate global y-limits for duration

    all_duration_values = []

    for type_name in available_types:
        mask = df_plot[type_col] == type_name
        values = df_plot.loc[mask, duration_col].dropna().values
        all_duration_values.extend(values)

    if len(all_duration_values) > 0:
        dur_ymin = np.min(all_duration_values)
        dur_ymax = np.max(all_duration_values)
        dur_padding = (dur_ymax - dur_ymin) * 0.15
        dur_ylim = (
            dur_ymin - dur_padding,
            dur_ymax + dur_padding,
        )
    else:
        dur_ylim = (0, 10)

    fontsize_label = 16
    fontsize_tick = 16
    fontsize_sig = 16
    fontsize_title = 18

    # Create a figure with four visible panels and a spacer column.
    fig = plt.figure(figsize=(12, 4))
    grid = gridspec.GridSpec(
        1,
        5,
        width_ratios=[1, 1, 1, 0.3, 1],
        wspace=0.4,
    )

    axes = [
        plt.subplot(grid[0]),
        plt.subplot(grid[1]),
        plt.subplot(grid[2]),
        plt.subplot(grid[4]),
    ]

    # panels 1, 2, and 3: alpha, beta, and eps

    for idx, (param, param_label) in enumerate(zip(params, param_labels)):
        ax = axes[idx]

        data_by_type = []
        positions = []

        # Collect parameter values for each available class.
        for type_index, type_name in enumerate(available_types):
            mask = df_plot[type_col] == type_name
            data = df_plot.loc[mask, param].dropna().values

            data_by_type.append(data)
            positions.append(type_index + 1)

        # Draw individual points with horizontal jitter.
        for color_index, (position, data) in enumerate(zip(positions, data_by_type)):
            if len(data) > 0:
                x_jitter = np.random.normal(
                    position,
                    0.04,
                    size=len(data),
                )

                ax.scatter(
                    x_jitter,
                    data,
                    alpha=0.5,
                    color=type_colors[color_index],
                    s=15,
                    edgecolors="black",
                    linewidths=0.3,
                )

        # Draw the boxplot without showing fliers.
        boxplot = ax.boxplot(
            data_by_type,
            positions=positions,
            patch_artist=True,
            widths=0.6,
            showmeans=False,
            showfliers=False,
        )

        # Apply class colors to the boxes.
        for color_index, patch in enumerate(boxplot["boxes"]):
            patch.set_facecolor(type_colors[color_index])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        # Standardize boxplot line appearance.
        for element in ["whiskers", "caps", "medians"]:
            if element in boxplot:
                for line in boxplot[element]:
                    line.set_color("black")
                    line.set_linewidth(1.0)

        # Make the median line slightly thicker.
        for median in boxplot["medians"]:
            median.set_linewidth(1.5)

        # Set labels, limits, and grid.
        ax.set_xticks(positions)
        ax.set_xticklabels(
            type_display_names,
            rotation=45,
            ha="right",
            fontsize=fontsize_tick,
        )

        ax.set_ylabel(
            param_label,
            rotation=0,
            fontsize=fontsize_label,
            labelpad=15,
        )

        if idx != 0:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.tick_params(axis="y", labelsize=fontsize_tick)

        ax.set_ylim(param_ylim)
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")

        # Run a statistical test between the first two available groups.
        y_pos_base = 1.02

        if len(data_by_type) >= 2:
            data1 = data_by_type[0]
            data2 = data_by_type[1]

            if len(data1) >= 3 and len(data2) >= 3:
                _, p_val = mannwhitneyu(
                    data1,
                    data2,
                    alternative="two-sided",
                )

                x1 = positions[0]
                x2 = positions[1]

                if p_val < 0.05:
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    else:
                        stars = "*"

                    fontweight = "bold"
                else:
                    stars = "n.s."
                    fontweight = "normal"

                ax.text(
                    (x1 + x2) / 2,
                    y_pos_base,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize_sig,
                    fontweight=fontweight,
                )

    # panel 4: duration boxplot

    ax_dur = axes[3]
    ax_dur.tick_params(axis="y", labelsize=fontsize_tick)

    duration_by_type = []
    duration_positions = []
    duration_display_names = []
    duration_colors = []

    # Collect duration values for each available class.
    for type_index, type_name in enumerate(available_types):
        mask = df_plot[type_col] == type_name
        duration_data = df_plot.loc[mask, duration_col].dropna().values

        if len(duration_data) > 0:
            duration_by_type.append(duration_data)
            duration_positions.append(type_index + 1)
            duration_display_names.append(
                type_names_display.get(type_name, type_name.capitalize())
            )
            duration_colors.append(type_colors[type_index])

    if len(duration_by_type) > 0:
        # Draw individual points with horizontal jitter.
        for color_index, (position, data) in enumerate(
            zip(duration_positions, duration_by_type)
        ):
            if len(data) > 0:
                x_jitter = np.random.normal(
                    position,
                    0.04,
                    size=len(data),
                )

                ax_dur.scatter(
                    x_jitter,
                    data,
                    alpha=0.5,
                    color=duration_colors[color_index],
                    s=15,
                    edgecolors="black",
                    linewidths=0.3,
                )

        # Draw the duration boxplot without showing fliers.
        boxplot_dur = ax_dur.boxplot(
            duration_by_type,
            positions=duration_positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )

        # Apply class colors to the duration boxes.
        for color_index, patch in enumerate(boxplot_dur["boxes"]):
            patch.set_facecolor(duration_colors[color_index])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        # Standardize boxplot line appearance.
        for element in ["whiskers", "caps", "medians"]:
            if element in boxplot_dur:
                for line in boxplot_dur[element]:
                    line.set_color("black")
                    line.set_linewidth(1.0)

        # Make the median line slightly thicker.
        for median in boxplot_dur["medians"]:
            median.set_linewidth(1.5)

        # Set labels, limits, and grid.
        ax_dur.set_xticks(duration_positions)
        ax_dur.set_xticklabels(
            duration_display_names,
            rotation=45,
            ha="right",
            fontsize=fontsize_tick,
        )

        ax_dur.set_ylabel(
            "Primed < Control (ms)",
            fontsize=fontsize_label,
        )

        ax_dur.set_ylim(dur_ylim)
        ax_dur.grid(True, linestyle=":", alpha=0.6, axis="y")

        # Run a statistical test between the first two available groups.
        y_pos_base = 347

        if len(duration_by_type) >= 2:
            data1 = duration_by_type[0]
            data2 = duration_by_type[1]

            if len(data1) >= 3 and len(data2) >= 3:
                _, p_val = mannwhitneyu(
                    data1,
                    data2,
                    alternative="two-sided",
                )

                x1 = duration_positions[0]
                x2 = duration_positions[1]

                if p_val < 0.05:
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    else:
                        stars = "*"

                    fontweight = "bold"
                else:
                    stars = "n.s."
                    fontweight = "normal"

                ax_dur.text(
                    (x1 + x2) / 2,
                    y_pos_base,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize_sig,
                    fontweight=fontweight,
                )

    # add titles above plot groups

    fig.canvas.draw()

    # Center title over the first three panels.
    bbox_alpha = axes[0].get_position()
    bbox_eps = axes[2].get_position()
    center_x = (bbox_alpha.x0 + bbox_eps.x1) / 2

    fig.text(
        center_x,
        0.91,
        "(c) Zoom - input parameters",
        ha="center",
        va="bottom",
        fontsize=fontsize_title,
        fontweight="normal",
    )

    # Center title over the duration panel.
    bbox_dur = axes[3].get_position()
    center_x_dur = (bbox_dur.x0 + bbox_dur.x1) / 2

    fig.text(
        center_x_dur - 0.02,
        0.91,
        "(d) Zoom - sig. window length",
        ha="center",
        va="bottom",
        fontsize=fontsize_title,
        fontweight="normal",
    )

    # Save and close the figure.
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(
        save_dir / f"combined_parameter_duration_{method_name}.pdf",
        bbox_inches="tight",
    )
    plt.close()

    return fig

# Test normality of a given set of values using the Shapiro-Wilk test and save the results to an Excel file.
def test_normality(values, save_dir, name, alpha=0.05):
   
    values = pd.Series(values).dropna()

    if len(values) < 3:
        print(f"Not enough values to test normality for {name}.")
        return None

    # Pingouin Shapiro-Wilk normality test
    normality_result = pg.normality(
        values,
        method="shapiro",
        alpha=alpha,
    )

    n_total = len(values)
    n_negative = int((values < 0).sum())
    n_positive = int((values > 0).sum())
    n_zero = int((values == 0).sum())

    percent_negative = 100 * n_negative / n_total
    percent_positive = 100 * n_positive / n_total
    percent_zero = 100 * n_zero / n_total

    result_table = pd.DataFrame(
        [
            {
                "variable": name,
                "n": n_total,
                "W": normality_result["W"].iloc[0],
                "pval": normality_result["pval"].iloc[0],
                "normal": normality_result["normal"].iloc[0],
                "alpha": alpha,
                "n_negative": n_negative,
                "n_positive": n_positive,
                "n_zero": n_zero,
                "percent_negative": percent_negative,
                "percent_positive": percent_positive,
                "percent_zero": percent_zero,
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "skewness": values.skew(),
            }
        ]
    )

    output_file = save_dir / f"{name}_normality_and_direction.xlsx"
    result_table.to_excel(output_file, index=False)

    return result_table
