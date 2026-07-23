import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


fixed_type_names = {
    "left": "sharpening",
    "middle": "fatiguing",
    "right": "None",
}


# Load all rank_analysis_data_responsive.csv files from the expected results folder structure and collect their raw data.
def load_raw_data(results_path, pattern_folder):

    results_path = Path(results_path)

    # Check that the main results path exists before searching for files.
    if not results_path.exists():
        raise FileNotFoundError(f"results_path does not exist: {results_path}")

    total_raw_neurons = 0
    all_raw_data = []

    # Search files with this structure:
    # results_path / parameter_folder / step_7 / rank_analysis_data_responsive.csv
    csv_files = sorted(
        results_path.glob("*/step_7/rank_analysis_data_responsive.csv")
    )

    print(f"found candidate csv files: {len(csv_files)}")

    for source_file in csv_files:
        # Get the parameter folder from the source file path.
        param_folder = source_file.parents[1]

        # Keep only folders matching the expected parameter pattern.
        if not pattern_folder.fullmatch(param_folder.name):
            print(f"  skipping non-matching folder: {param_folder.name}")
            continue

        try:
            # Read the current csv file into a dataframe.
            df = pd.read_csv(source_file)

            # Check that the neuron identifier column is available.
            if "Neuron_ID" not in df.columns:
                print(f"  error {param_folder.name} - Neuron_ID column not found")
                continue

            # Count unique neurons in the raw file.
            n_neurons_raw = df["Neuron_ID"].nunique()
            total_raw_neurons += n_neurons_raw

            print(f"  {param_folder.name:<50} : {n_neurons_raw:4d} unique neurons")

            # Store all useful information for later processing.
            all_raw_data.append(
                {
                    "folder": param_folder.name,
                    "source_file": source_file,
                    "df": df,
                    "n_neurons": n_neurons_raw,
                }
            )

        except Exception as exc:
            # Continue with the next file if the current one cannot be read.
            print(f"  error reading {param_folder.name}: {exc}")

    return all_raw_data, total_raw_neurons


 # Apply the first filter: keep neurons with absolute z-score greater than threshold_high in both control and priming conditions.
def apply_first_filter(df, threshold_high):

    high_threshold_mask = (
        (abs(df["Z_Control"]) > threshold_high)
        & (abs(df["Z_Priming"]) > threshold_high)
    )

    # Return the unique neuron ids that pass the first filter.
    return df[high_threshold_mask]["Neuron_ID"].unique()


# Apply the second filter: keep neurons that respond to at least min_stimuli stimuli with absolute z-score greater than threshold_low in both conditions.
def apply_second_filter(df, valid_neurons_step1, threshold_low, min_stimuli):

    valid_neurons_step2 = []

    for neuron_id in valid_neurons_step1:
        # Extract all rows associated with the current neuron.
        neuron_data = df[df["Neuron_ID"] == neuron_id]
        responsive_stimuli = 0

        # Check each stimulus separately for the current neuron.
        for stimulus_id in neuron_data["Stimulus_ID"].unique():
            stimulus_data = neuron_data[
                neuron_data["Stimulus_ID"] == stimulus_id
            ].iloc[0]

            # Count the stimulus only if both conditions pass the low threshold.
            if (
                abs(stimulus_data["Z_Priming"]) > threshold_low
                and abs(stimulus_data["Z_Control"]) > threshold_low
            ):
                responsive_stimuli += 1

        # Keep the neuron only if enough responsive stimuli were found.
        if responsive_stimuli >= min_stimuli:
            valid_neurons_step2.append(neuron_id)

    return np.array(valid_neurons_step2)


 # Count the number of responsive stimuli for one neuron using an and condition: a stimulus is responsive only if both priming and control pass threshold_low.
def count_responsive_stimuli_and(df, neuron_id, threshold_low):

    neuron_full_data = df[df["Neuron_ID"] == neuron_id]
    n_responsive_stimuli = 0

    # Loop through all stimuli associated with the selected neuron.
    for stimulus_id in neuron_full_data["Stimulus_ID"].unique():
        stimulus_data = neuron_full_data[
            neuron_full_data["Stimulus_ID"] == stimulus_id
        ].iloc[0]

        # Increase the counter only when both conditions pass the threshold.
        if (
            abs(stimulus_data["Z_Priming"]) > threshold_low
            and abs(stimulus_data["Z_Control"]) > threshold_low
        ):
            n_responsive_stimuli += 1

    return n_responsive_stimuli


# Calculate the delta metric for one neuron by comparing the control-priming difference between rank 1 and rank 2.
def calculate_delta_metric(df_filtered, df_full, neuron_id, threshold_low):

    neuron_data = df_filtered[df_filtered["Neuron_ID"] == neuron_id]

    # Extract rank 1 and rank 2 rows for the current neuron.
    rank1_data = neuron_data[neuron_data["Rank"] == 1]
    rank2_data = neuron_data[neuron_data["Rank"] == 2]

    # Delta cannot be calculated if either rank is missing.
    if rank1_data.empty or rank2_data.empty:
        return None

    # Use the first available row for each rank.
    rank1_data = rank1_data.iloc[0]
    rank2_data = rank2_data.iloc[0]

    # Compute control-minus-priming differences for both ranks.
    diff_rank1 = rank1_data["Z_Control"] - rank1_data["Z_Priming"]
    diff_rank2 = rank2_data["Z_Control"] - rank2_data["Z_Priming"]

    # Delta is the difference between the two rank-specific differences.
    delta = diff_rank1 - diff_rank2

    return {
        "delta": delta,
        "z_priming_rank1": rank1_data["Z_Priming"],
        "z_control_rank1": rank1_data["Z_Control"],
        "z_priming_rank2": rank2_data["Z_Priming"],
        "z_control_rank2": rank2_data["Z_Control"],
        "n_responsive_stimuli": count_responsive_stimuli_and(
            df_full,
            neuron_id,
            threshold_low,
        ),
    }


# Apply the two z-score filters, save raw and filtered files, and collect delta values for active neurons.
def process_data(
    all_raw_data,
    raw_score_path,
    filtered_score_path,
    threshold_high,
    threshold_low,
    min_stimuli,
):

    raw_score_path = Path(raw_score_path)
    filtered_score_path = Path(filtered_score_path)

    # Create output folders if they do not already exist.
    raw_score_path.mkdir(parents=True, exist_ok=True)
    filtered_score_path.mkdir(parents=True, exist_ok=True)

    file_count = 0
    filtered_count = 0
    total_filtered_neurons = 0
    delta_data = []
    active_neurons_dict = {}

    # Define the columns required by the filtering and delta calculation steps.
    required_columns = {
        "Z_Priming",
        "Z_Control",
        "Rank",
        "Neuron_ID",
        "Stimulus_ID",
    }

    for raw_data in all_raw_data:
        param_folder = raw_data["folder"]
        source_file = raw_data["source_file"]
        df = raw_data["df"]

        print(f"\nprocessing {param_folder}...")

        # Copy the raw source file into the raw-score output folder.
        destination_file = raw_score_path / f"{param_folder}.csv"
        shutil.copy2(source_file, destination_file)
        file_count += 1

        try:
            # Skip the current file if any required column is missing.
            if not required_columns.issubset(df.columns):
                missing = sorted(required_columns - set(df.columns))
                print(f"  missing required columns: {missing}")
                continue

            # First filter: high absolute z-score in both conditions.
            valid_neurons_step1 = apply_first_filter(df, threshold_high)
            if len(valid_neurons_step1) == 0:
                continue

            # Second filter: enough responsive stimuli per neuron.
            valid_neurons = apply_second_filter(
                df,
                valid_neurons_step1,
                threshold_low,
                min_stimuli,
            )
            if len(valid_neurons) == 0:
                continue

            # Keep only rows associated with valid neurons.
            df_filtered = df[df["Neuron_ID"].isin(valid_neurons)]

            # Save the filtered dataframe.
            filtered_file = filtered_score_path / f"{param_folder}.csv"
            df_filtered.to_csv(filtered_file, index=False)
            filtered_count += 1

            # Update counters and store the active-neuron ids.
            total_filtered_neurons += len(valid_neurons)
            active_neurons_dict[param_folder] = valid_neurons.tolist()

            # Calculate delta for each valid neuron.
            for neuron_id in valid_neurons:
                delta_result = calculate_delta_metric(
                    df_filtered,
                    df,
                    neuron_id,
                    threshold_low,
                )

                # Store only valid delta results.
                if delta_result:
                    delta_data.append(
                        {
                            "folder": param_folder,
                            "neuron_id": neuron_id,
                            **delta_result,
                        }
                    )

        except Exception as exc:
            # Continue with the next configuration if this one fails.
            print(f"  error: {exc}")

    return (
        file_count,
        filtered_count,
        total_filtered_neurons,
        delta_data,
        active_neurons_dict,
    )


 # Calculate fixed mad-based classification thresholds and return the labels used for sharpening, fatiguing, and None classes.
def calculate_classification_params(delta_df):

    # Extract delta values from the dataframe.
    delta_values = delta_df["delta"].values

    # Compute the median and median absolute deviation of delta values.
    median_delta = np.median(delta_values)
    mad_delta = np.median(np.abs(delta_values - median_delta))

    # Use half of the mad as a symmetric tolerance around zero.
    tol_delta = 0.5 * mad_delta

    # Define left and right thresholds.
    left_threshold = -tol_delta
    right_threshold = tol_delta

    mode_name = "MAD-based tolerance"
    type_names = fixed_type_names.copy()

    print(f"calculated tolerance: tol_delta = {tol_delta:.6f} (mad = {mad_delta:.6f})")

    return (
        left_threshold,
        right_threshold,
        mad_delta,
        tol_delta,
        mode_name,
        type_names,
    )


# Classify one delta value using the fixed mad-based thresholds.
def classify_delta(delta, left_threshold, right_threshold, type_names):

    # Values below the left threshold are classified as sharpening.
    if delta < left_threshold:
        return type_names["left"]

    # Values above the right threshold are classified as None.
    if delta > right_threshold:
        return type_names["right"]

    # Values between the two thresholds are classified as fatiguing.
    return type_names["middle"]
    