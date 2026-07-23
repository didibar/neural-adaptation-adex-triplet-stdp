import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


# Main paths
base_dir = Path("/home/dilettabartolini/neural-adaptation-adex-triplet-stdp")

results_root = base_dir / "Results_DE_optimization"

active_neurons_file = (base_dir / "Z_score/Thr2_thr3_1/active_neurons_per_config.json")

# Keep the output folder name requested for this analysis.
output_root = base_dir / "Z-score_rank"
output_root.mkdir(exist_ok=True, parents=True)

# Analysis configuration

# Stop each plot when both conditions remain close to zero for two consecutive stimuli.
threshold_low = 0.92

# Keep only the first ranked stimuli in each neuron-level plot.
# Set to None to restore the original near-zero stopping rule.
max_rank_to_plot = 2

# Optional side-by-side comparison saved inside Z-score_rank.
sharpening_neuron_id = 542
sharpening_config = "alpha0.22_beta0.565_eps0.492_seed42_42"
fatiguing_neuron_id = 175
fatiguing_config = "alpha0.016_beta0.617_eps0.273_seed42_42"

comparison_output_dir = output_root / "sharpening_fatiguing_comparison"

# Plot colors for the Primed and Control conditions.
blue_pastel = (0.4, 0.6, 1.0, 0.7)
red_pastel = (1.0, 0.4, 0.4, 0.7)

condition_colors = {
    "priming": blue_pastel,
    "not_priming": red_pastel
}

# Configuration-name parsing

# Match folders with names like: alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}_{seed}
config_pattern = re.compile(
    r"alpha(?P<alpha>[^_]+)"
    r"_beta(?P<beta>[^_]+)"
    r"_eps(?P<eps>[^_]+)"
    r"_seed(?P<seed_1>\d+)_(?P<seed_2>\d+)$"
)

# Check whether a folder name matches the expected configuration format.
def is_valid_config_name(config_name):
    match = config_pattern.match(config_name)

    if match is None:
        return False

    # Keep only configurations where the two seed fields are identical.
    return match.group("seed_1") == match.group("seed_2")


# Load the active-neuron dictionary and convert neuron IDs to integers.
def load_active_neurons(active_neurons_file):
    if not active_neurons_file.exists():
        raise FileNotFoundError(
            f"Active-neuron JSON file not found: {active_neurons_file}"
        )

    with open(active_neurons_file, "r") as file:
        active_neurons_per_config = json.load(file)

    cleaned_active_neurons = {}

    for config_key, neurons in active_neurons_per_config.items():
        cleaned_active_neurons[config_key] = [int(neuron) for neuron in neurons]

    return cleaned_active_neurons


# Apply the rank filter used by the plots.
def get_filtered_neuron_data(neuron_data, threshold_low, max_rank_to_plot=None):
    neuron_data = neuron_data.sort_values("Rank")

    if len(neuron_data) == 0:
        return neuron_data.copy(), 0

    if max_rank_to_plot is not None:
        neuron_data_plot = neuron_data[
            neuron_data["Rank"] <= max_rank_to_plot
        ].copy()
        return neuron_data_plot, max_rank_to_plot

    # By default, keep all ranks unless the stopping condition is met.
    stop_rank = int(neuron_data["Rank"].max())
    count_near_zero = 0

    for _, row in neuron_data.iterrows():
        is_near_zero_in_both_conditions = (
            abs(row["Z_Priming"]) <= threshold_low
            and abs(row["Z_Control"]) <= threshold_low
        )

        if is_near_zero_in_both_conditions:
            count_near_zero += 1

            # Stop before the second consecutive near-zero stimulus.
            if count_near_zero >= 2:
                stop_rank = int(row["Rank"]) - 1
                break
        else:
            count_near_zero = 0

    neuron_data_plot = neuron_data[neuron_data["Rank"] <= stop_rank].copy()

    return neuron_data_plot, stop_rank


# Extract the already-ranked data for one neuron from the responsive CSV table.
def prepare_ranked_data(rank_df, neuron_id):
    neuron_data = rank_df[rank_df["Neuron_ID"] == neuron_id].copy()

    if len(neuron_data) == 0:
        return None

    # The Rank column is already available in rank_analysis_data_responsive.csv.
    neuron_data = neuron_data.sort_values("Rank")

    return neuron_data


# Create and save the ranked Z-score plot for a single neuron.
def plot_neuron_z_scores(
    neuron_data,
    neuron_id,
    output_path,
    condition_colors,
    threshold_low,
    max_rank_to_plot=None
):
    neuron_data_plot, _ = get_filtered_neuron_data(
        neuron_data,
        threshold_low,
        max_rank_to_plot=max_rank_to_plot
    )

    if len(neuron_data_plot) == 0:
        print(f"    - Neuron {neuron_id}: no data to plot after filtering")
        return

    fig, ax = plt.subplots(figsize=(4, 4))

    # Use stimulus IDs as y-axis labels, while Rank defines their order.
    stimulus_names = neuron_data_plot["Stimulus_ID"].astype(str).tolist()

    ax.plot(
        neuron_data_plot["Z_Priming"],
        neuron_data_plot["Rank"],
        color=condition_colors["priming"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="Primed"
    )

    ax.plot(
        neuron_data_plot["Z_Control"],
        neuron_data_plot["Rank"],
        color=condition_colors["not_priming"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="Control"
    )

    # Configure labels, ticks, limits, legend, and grid.
    ax.set_xlabel("Z-score", fontsize=14)
    ax.set_ylabel("Stimulus-ID", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)

    ax.set_yticks(neuron_data_plot["Rank"].tolist())
    ax.set_yticklabels(stimulus_names, fontsize=14)

    ax.set_ylim(0.5, max(neuron_data_plot["Rank"]) + 0.5)

    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    output_path.mkdir(exist_ok=True, parents=True)

    plot_filename = output_path / f"neuron_{neuron_id}_z_scores.pdf"

    plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


# Create compact CSV and Excel tables with only the rows included in the plots.
def create_filtered_table(neuron_data_dict, output_path, threshold_low, max_rank_to_plot=None):
    filtered_data = []

    for neuron_id, neuron_data in neuron_data_dict.items():
        neuron_data_sorted = neuron_data.sort_values("Rank")

        # Reuse the plotting filter so the table and plots contain the same rows.
        neuron_data_filtered, _ = get_filtered_neuron_data(
            neuron_data_sorted,
            threshold_low,
            max_rank_to_plot=max_rank_to_plot
        )

        if len(neuron_data_filtered) == 0:
            continue

        for _, row in neuron_data_filtered.iterrows():
            output_row = {
                "Neuron_ID": int(neuron_id),
                "Stimulus_ID": int(row["Stimulus_ID"]),
                "Rank": int(row["Rank"]),
                "Z_Priming": round(float(row["Z_Priming"]), 3),
                "Z_Control": round(float(row["Z_Control"]), 3),
            }

            # Use the CSV column if available; otherwise compute the difference.
            if "Z_Difference" in row.index:
                output_row["Z_Difference"] = round(
                    float(row["Z_Difference"]),
                    3
                )
            else:
                output_row["Z_Difference"] = round(
                    float(row["Z_Priming"]) - float(row["Z_Control"]),
                    3
                )

            filtered_data.append(output_row)

    if not filtered_data:
        print("    - No data to save in the filtered table")
        return None

    filtered_df = pd.DataFrame(filtered_data)

    # Sort the exported table by neuron and stimulus rank.
    filtered_df = filtered_df.sort_values(["Neuron_ID", "Rank"])

    output_path.mkdir(exist_ok=True, parents=True)

    if max_rank_to_plot is None:
        excel_filename = output_path / f"filtered_neurons_stimuli_thr{threshold_low}.xlsx"
    else:
        excel_filename = output_path / f"filtered_neurons_stimuli_rank_1_{max_rank_to_plot}.xlsx"

    filtered_df.to_excel(excel_filename, index=False, float_format="%.3f")

    return filtered_df


# Run the full workflow for one configuration folder.
def process_single_config(config_dir, active_neurons_per_config):
    config_key = config_dir.name

    rank_analysis_file = (
        config_dir
        / "step_7"
        / "rank_analysis_data_responsive.csv"
    )

    if not rank_analysis_file.exists():
        print(f"[SKIP] Missing CSV for {config_key}")
        return {
            "config": config_key,
            "status": "missing_csv",
            "n_active_json": 0,
            "n_plotted": 0
        }

    if config_key not in active_neurons_per_config:
        print(f"[SKIP] Configuration not found in JSON: {config_key}")
        return {
            "config": config_key,
            "status": "missing_json_key",
            "n_active_json": 0,
            "n_plotted": 0
        }

    active_neurons = active_neurons_per_config[config_key]

    print(f"\n[CONFIG] {config_key}")
    print(f"    Active neurons in JSON: {len(active_neurons)}")

    df = pd.read_csv(rank_analysis_file)

    required_cols = [
        "Neuron_ID",
        "Stimulus_ID",
        "Rank",
        "Z_Priming",
        "Z_Control"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {rank_analysis_file}"
            )

    # Normalize key columns to integer type before matching neuron IDs.
    df["Neuron_ID"] = df["Neuron_ID"].astype(int)
    df["Stimulus_ID"] = df["Stimulus_ID"].astype(int)
    df["Rank"] = df["Rank"].astype(int)

    available_neurons = set(df["Neuron_ID"].unique().tolist())

    # Keep only active neurons that are also present in the responsive CSV.
    neurons_to_plot = [
        neuron_id
        for neuron_id in active_neurons
        if neuron_id in available_neurons
    ]

    missing_neurons = [
        neuron_id
        for neuron_id in active_neurons
        if neuron_id not in available_neurons
    ]

    if missing_neurons:
        print(
            "    Warning: these neurons are listed in JSON "
            "but are missing from the responsive CSV:"
        )
        print(f"    {missing_neurons}")

    if len(neurons_to_plot) == 0:
        print("    No plottable neurons for this configuration")
        return {
            "config": config_key,
            "status": "no_matching_neurons",
            "n_active_json": len(active_neurons),
            "n_plotted": 0
        }

    output_config_dir = output_root / config_key
    plots_dir = output_config_dir / "plot_neuron_z_scores"
    tables_dir = output_config_dir / "create_filtered_table"

    output_config_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    neuron_data_dict = {}

    # Prepare one filtered DataFrame per neuron.
    for neuron_id in neurons_to_plot:
        neuron_data = prepare_ranked_data(df, neuron_id)

        if neuron_data is not None:
            neuron_data_dict[neuron_id] = neuron_data

    # Save one PDF plot per neuron.
    for neuron_id, neuron_data in neuron_data_dict.items():
        plot_neuron_z_scores(
            neuron_data=neuron_data,
            neuron_id=neuron_id,
            output_path=plots_dir,
            condition_colors=condition_colors,
            threshold_low=threshold_low,
            max_rank_to_plot=max_rank_to_plot
        )

    # Save the compact table for this configuration.
    create_filtered_table(
        neuron_data_dict=neuron_data_dict,
        output_path=tables_dir,
        threshold_low=threshold_low,
        max_rank_to_plot=max_rank_to_plot
    )

    return {
        "config": config_key,
        "status": "ok",
        "n_active_json": len(active_neurons),
        "n_plotted": len(neuron_data_dict)
    }


def load_validated_neuron_for_comparison(
    config_key,
    neuron_id,
    active_neurons_per_config,
    max_rank_to_plot
):
    if config_key not in active_neurons_per_config:
        raise KeyError(
            f"Configuration '{config_key}' not found in {active_neurons_file}"
        )

    active_neurons = active_neurons_per_config[config_key]

    if neuron_id not in active_neurons:
        raise ValueError(
            f"Neuron {neuron_id} is not listed as active for "
            f"configuration '{config_key}' in {active_neurons_file}"
        )

    rank_analysis_file = (
        results_root
        / config_key
        / "step_7"
        / "rank_analysis_data_responsive.csv"
    )

    if not rank_analysis_file.exists():
        raise FileNotFoundError(f"File not found: {rank_analysis_file}")

    df = pd.read_csv(rank_analysis_file)

    required_cols = [
        "Neuron_ID",
        "Stimulus_ID",
        "Rank",
        "Z_Priming",
        "Z_Control"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {rank_analysis_file}"
            )

    df["Neuron_ID"] = df["Neuron_ID"].astype(int)
    df["Stimulus_ID"] = df["Stimulus_ID"].astype(int)
    df["Rank"] = df["Rank"].astype(int)

    neuron_data = df[df["Neuron_ID"] == neuron_id].copy()

    if len(neuron_data) == 0:
        raise ValueError(
            f"Neuron {neuron_id} is active in JSON but missing from "
            f"{rank_analysis_file}"
        )

    neuron_data_plot, _ = get_filtered_neuron_data(
        neuron_data,
        threshold_low,
        max_rank_to_plot=max_rank_to_plot
    )

    if len(neuron_data_plot) == 0:
        raise ValueError(
            f"Neuron {neuron_id} has no data after rank filtering."
        )

    return neuron_data_plot


def plot_comparison_neuron_on_ax(ax, neuron_data, subplot_label, hide_ylabel=False):
    stimulus_names = neuron_data["Stimulus_ID"].astype(str).tolist()

    ax.plot(
        neuron_data["Z_Priming"],
        neuron_data["Rank"],
        color=condition_colors["priming"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="Primed"
    )

    ax.plot(
        neuron_data["Z_Control"],
        neuron_data["Rank"],
        color=condition_colors["not_priming"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="Control"
    )

    ax.set_xlabel("Z-score", fontsize=16)
    ax.set_ylabel("" if hide_ylabel else "Stimulus-ID", fontsize=16)
    ax.set_title(subplot_label, fontsize=18, pad=15)
    ax.tick_params(axis="x", labelsize=16)

    ax.set_yticks(neuron_data["Rank"].tolist())
    ax.set_yticklabels(stimulus_names, fontsize=16)
    ax.set_ylim(0.5, max(neuron_data["Rank"]) + 0.5)

    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_sharpening_fatiguing_comparison(
    sharpening_data,
    fatiguing_data,
    output_path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    plot_comparison_neuron_on_ax(
        ax1,
        sharpening_data,
        subplot_label="(a) Sharpening",
        hide_ylabel=False
    )

    plot_comparison_neuron_on_ax(
        ax2,
        fatiguing_data,
        subplot_label="(b) Fatiguing",
        hide_ylabel=True
    )

    plt.tight_layout()

    output_path.mkdir(exist_ok=True, parents=True)
    plot_filename = (
        output_path
        / f"comparison_neuron_{sharpening_neuron_id}_vs_{fatiguing_neuron_id}.pdf"
    )

    plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return plot_filename


def create_comparison_table(sharpening_data, fatiguing_data, output_path):
    table_rows = []

    for label, neuron_id, neuron_data in [
        ("Sharpening", sharpening_neuron_id, sharpening_data),
        ("Fatiguing", fatiguing_neuron_id, fatiguing_data),
    ]:
        for _, row in neuron_data.iterrows():
            table_rows.append({
                "Profile": label,
                "Neuron_ID": int(neuron_id),
                "Stimulus_ID": int(row["Stimulus_ID"]),
                "Rank": int(row["Rank"]),
                "Z_Priming": round(float(row["Z_Priming"]), 3),
                "Z_Control": round(float(row["Z_Control"]), 3),
                "Z_Difference": round(
                    float(row["Z_Priming"]) - float(row["Z_Control"]),
                    3
                ),
            })

    summary_df = pd.DataFrame(table_rows)
    summary_df = summary_df.sort_values(["Profile", "Neuron_ID", "Rank"])

    output_path.mkdir(exist_ok=True, parents=True)
    table_filename = (
        output_path
        / f"comparison_neuron_{sharpening_neuron_id}_vs_{fatiguing_neuron_id}.xlsx"
    )

    summary_df.to_excel(table_filename, index=False, float_format="%.3f")

    return table_filename


def create_sharpening_fatiguing_comparison(active_neurons_per_config):
    sharpening_data = load_validated_neuron_for_comparison(
        config_key=sharpening_config,
        neuron_id=sharpening_neuron_id,
        active_neurons_per_config=active_neurons_per_config,
        max_rank_to_plot=max_rank_to_plot
    )

    fatiguing_data = load_validated_neuron_for_comparison(
        config_key=fatiguing_config,
        neuron_id=fatiguing_neuron_id,
        active_neurons_per_config=active_neurons_per_config,
        max_rank_to_plot=max_rank_to_plot
    )

    plot_filename = plot_sharpening_fatiguing_comparison(
        sharpening_data=sharpening_data,
        fatiguing_data=fatiguing_data,
        output_path=comparison_output_dir
    )

    table_filename = create_comparison_table(
        sharpening_data=sharpening_data,
        fatiguing_data=fatiguing_data,
        output_path=comparison_output_dir
    )

    return plot_filename, table_filename


# Find all valid configuration folders and process them one by one.
def main():
    if not results_root.exists():
        raise FileNotFoundError(
            f"Results folder not found: {results_root}"
        )

    active_neurons_per_config = load_active_neurons(active_neurons_file)

    config_dirs = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and is_valid_config_name(path.name)
    ]

    config_dirs = sorted(config_dirs, key=lambda path: path.name)

    if len(config_dirs) == 0:
        raise ValueError(
            f"No configuration folders found in {results_root}"
        )

    summary = []

    for config_dir in config_dirs:
        result = process_single_config(
            config_dir=config_dir,
            active_neurons_per_config=active_neurons_per_config
        )
        summary.append(result)

    summary_df = pd.DataFrame(summary)

    summary_file_xlsx = output_root / "summary_z_score_rank.xlsx"

    summary_df.to_excel(summary_file_xlsx, index=False)

    comparison_plot, comparison_table = create_sharpening_fatiguing_comparison(
        active_neurons_per_config
    )

if __name__ == "__main__":
    main()
