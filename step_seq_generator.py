import numpy as np
import random
from collections import defaultdict, Counter
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# Global parameters used to define the image pool and the generated dataset size.
n_images = 100
n_groups = 10
n_sequences = 20

# Map an image index to its group using integer division.
def get_group(img_idx):
    return img_idx // n_groups

# Required transition patterns, where C indicates control and P indicates primed.
fixed_patterns = [
    ['C', 'C', 'P', 'P'],  # CCPP
    ['C', 'P', 'C', 'P'],  # CPCP
    ['C', 'P', 'P', 'C'],  # CPPC
    ['P', 'C', 'C', 'P'],  # PCCP
    ['P', 'C', 'P', 'C'],  # PCPC
    ['P', 'P', 'C', 'C'],  # PPCC
] * 3
additional_patterns_fixed = [['C', 'C', 'P', 'P'], ['P', 'P', 'C', 'C']]


# Select two random image indices per group to be used as possible first positions.
def select_first_images_per_group(seed):
    if seed is not None:
        random.seed(seed)

    first_images_by_group = {}
    all_first_images = []

    # Sample the first-position candidates independently within each image group.
    for group in range(n_groups):
        group_start = group * n_groups
        group_images = list(range(group_start, group_start + n_groups))
        selected = random.sample(group_images, 2)
        first_images_by_group[group] = selected
        all_first_images.extend(selected)

    return first_images_by_group, all_first_images


# Check whether a candidate image satisfies the current transition and usage constraints.
def is_valid_image(img_idx, prev_img, condition, used_in_current_sequence, index_usage_count, current_first_image):

    # Prevent repeated images within the same sequence.
    if img_idx in used_in_current_sequence: return False

    # Prevent the current first image from being reused as a repeated item.
    if index_usage_count.get(img_idx, 0) >= 1:
        if img_idx == current_first_image: return False

    # Enforce the global maximum usage count for each image.
    if index_usage_count.get(img_idx, 0) >= 2: return False

    # The first image has no previous image, so it is always valid at this stage.
    if prev_img is None: return True

    # Compare the candidate group with the previous image group.
    same_group = get_group(img_idx) == get_group(prev_img)

    # Primed transitions require the same group, while control transitions require different groups.
    if condition == 'P' and same_group: return True
    elif condition == 'C' and not same_group: return True

    return False


# Build one sequence using the required pattern and a priority-based image selection system.
def build_sequence(pattern, available_images, first_images_pool, used_first_images_indices, index_usage_count, first_image_reuse_tracker, seq_idx, sequence_length):

    sequence = [None] * sequence_length
    used_in_current_sequence = set()
    current_available = available_images.copy()

    # Select the first image from the predefined first-image pool.
    unused_first_images = [img for img in first_images_pool if img not in used_first_images_indices]

    if unused_first_images:
        # Prefer first images that have not been used yet.
        sequence[0] = random.choice(unused_first_images)
    else:
        candidates = [img for img in first_images_pool
                     if index_usage_count.get(img, 0) == 1]
        if candidates:
            sequence[0] = random.choice(candidates)
        else:
            # Fallback to the first available non-first image when the first-image pool is exhausted.
            if current_available:
                sequence[0] = current_available.pop(0)
            else:
                return None, None, None, None

    current_first_image = sequence[0]
    used_in_current_sequence.add(current_first_image)
    index_usage_count[current_first_image] = index_usage_count.get(current_first_image, 0) + 1
    used_first_images_indices.add(current_first_image)

    if current_first_image in current_available:
        current_available.remove(current_first_image)

    # Fill the remaining sequence positions according to the requested transition pattern.
    for i in range(1, sequence_length):
        condition = pattern[i-1]
        prev_img = sequence[i-1]

        all_candidates = []

        # Priority 1: use images that have never been used before.
        for img_idx in current_available:
            if is_valid_image(img_idx, prev_img, condition, used_in_current_sequence,
                            index_usage_count, current_first_image):
                all_candidates.append((img_idx, 0, 'new'))

        # Priority 2: reuse first images that have been used exactly once.
        first_images_used_once = [img for img in first_images_pool
                                 if img != current_first_image and
                                 index_usage_count.get(img, 0) == 1 and
                                 img not in used_in_current_sequence]

        for img_idx in first_images_used_once:
            if is_valid_image(img_idx, prev_img, condition, used_in_current_sequence,
                            index_usage_count, current_first_image):
                all_candidates.append((img_idx, 1, 'first_reuse'))

        if not all_candidates:
            return None, None, None, None

        # Choose the highest-priority valid candidate.
        all_candidates.sort(key=lambda x: x[1])
        chosen_img, priority, img_type = all_candidates[0]
        sequence[i] = chosen_img
        used_in_current_sequence.add(chosen_img)
        index_usage_count[chosen_img] = index_usage_count.get(chosen_img, 0) + 1

        if img_type == 'first_reuse':
            first_image_reuse_tracker[chosen_img] = first_image_reuse_tracker.get(chosen_img, 0) + 1

        # Remove selected images from the available pool.
        if chosen_img in current_available:
            current_available.remove(chosen_img)

    return sequence, current_available, used_first_images_indices, index_usage_count


# Generate one dataset of image sequences while enforcing transition and usage constraints.
def generate_dataset(sequence_length, seed, max_attempts, dataset_name):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Select two first-image candidates per group.
    first_images_by_group, all_first_images = select_first_images_per_group(seed)

    for group in sorted(first_images_by_group.keys()):
        images = first_images_by_group[group]

    # Initialize the available image pool and usage trackers.
    available_images = list(range(n_images))
    random.shuffle(available_images)
    available_images = [img for img in available_images if img not in all_first_images]

    used_first_images_indices = set()
    index_usage_count = defaultdict(int)
    first_image_reuse_tracker = defaultdict(int)
    sequences = []

    # Build the complete list of transition patterns to use.
    all_patterns = fixed_patterns + additional_patterns_fixed

    # Retry generation until all required sequences satisfy the constraints.
    for attempt in range(max_attempts):
        current_available = available_images.copy()
        current_used_first = set()
        current_usage_count = defaultdict(int)
        current_reuse_tracker = defaultdict(int)
        current_sequences = []

        success = True

        for pattern_idx, pattern in enumerate(all_patterns[:n_sequences]):
            sequence, current_available, current_used_first, current_usage_count = build_sequence(
                pattern, current_available, all_first_images, current_used_first,
                current_usage_count, current_reuse_tracker, pattern_idx, sequence_length
            )

            if sequence is None:
                success = False
                break

            current_sequences.append(sequence)

        # Retry when the current attempt did not generate all required sequences.
        if not success or len(current_sequences) != n_sequences:
            if attempt % 50 == 0 and attempt > 0:
                print(f" Attempt {attempt}: failed, retrying...")
            continue

        # Check the quality of the generated solution.
        all_indices = []
        for seq in current_sequences:
            all_indices.extend(seq)

        index_counts = Counter(all_indices)
        unique_count = len(index_counts)

        # Count how often selected first images were used.
        first_images_usage = {img: index_counts.get(img, 0) for img in all_first_images}
        first_images_used_twice = sum(1 for count in first_images_usage.values() if count == 2)
        first_images_used_once = sum(1 for count in first_images_usage.values() if count == 1)
        first_images_unused = sum(1 for count in first_images_usage.values() if count == 0)

        # Check for violations of the global usage constraints.
        violations = []
        for img, count in index_counts.items():
            if count > 2:
                violations.append((img, count))
            elif count == 2 and img in all_first_images:
                seqs_with_img = []
                for seq_idx, seq in enumerate(current_sequences):
                    if img in seq:
                        seqs_with_img.append(seq_idx)

                first_seq = None
                reuse_seq = None

                for seq_idx in seqs_with_img:
                    if current_sequences[seq_idx][0] == img:
                        first_seq = seq_idx
                    else:
                        reuse_seq = seq_idx

                if first_seq is None or reuse_seq is None:
                    violations.append((img, count))

        if violations:
            continue

        return current_sequences, all_patterns[:n_sequences], seed, all_first_images

    print(f"Failed to generate {dataset_name} after {max_attempts} attempts")
    return None, None, None, None, None


# Analyze a generated dataset and report duplicated, missing, and invalid indices.
def analyze_dataset_with_duplicates_missing(sequences, patterns_used, all_first_images, dataset_name, sequence_length):

    all_used_indices = []
    for seq in sequences:
        all_used_indices.extend(seq)

    index_counter = Counter(all_used_indices)

    # Categorize first-position images by their final usage count.
    first_images_used_once = [img for img in all_first_images if index_counter.get(img, 0) == 1]
    first_images_used_twice = [img for img in all_first_images if index_counter.get(img, 0) == 2]
    first_images_unused = [img for img in all_first_images if index_counter.get(img, 0) == 0]

    # Find all duplicated indices.
    all_duplicate_indices = [(img, count) for img, count in index_counter.items() if count > 1]
    all_duplicate_indices.sort(key=lambda x: x[1], reverse=True)

    # Find all indices that were never used.
    used_indices_set = set(all_used_indices)
    all_missing_indices = sorted([idx for idx in range(n_images) if idx not in used_indices_set])

    # Group missing indices by image group.
    missing_by_group = defaultdict(list)
    for idx in all_missing_indices:
        missing_by_group[get_group(idx)].append(idx)

    # Verify that each transition matches the requested pattern.
    errors = 0
    for i, (seq, pattern) in enumerate(zip(sequences, patterns_used)):
        for j in range(1, sequence_length):
            prev_group = get_group(seq[j-1])
            curr_group = get_group(seq[j])
            condition = 'P' if prev_group == curr_group else 'C'
            expected = pattern[j-1]

            if condition != expected:
                errors += 1

    return {
        'sequences': sequences,
        'patterns': patterns_used,
        'first_images': all_first_images,
        'first_used_twice': first_images_used_twice,
        'first_used_once': first_images_used_once,
        'first_unused': first_images_unused,
        'all_duplicates': all_duplicate_indices,
        'all_missing': all_missing_indices,
        'missing_by_group': missing_by_group,
        'unique_count': len(used_indices_set),
        'errors': errors,
        'index_counter': index_counter
    }


# Create one combined DataFrame containing all generated datasets and sequence metadata.
def create_combined_dataframe(all_datasets, sequence_length):
    all_rows = []
    all_row_names = []

    for dataset in all_datasets:
        dataset_name = dataset['name']
        sequences = dataset['sequences']
        patterns = dataset['patterns']

        for seq_idx, (seq, pattern) in enumerate(zip(sequences, patterns)):
            # Create the sequence name using the dataset name, sequence number, and pattern string.
            pattern_str = ''.join(pattern)  # Example: ['C', 'C', 'P', 'P'] -> "CCPP"
            row_name = f"{dataset_name}_seq_{seq_idx+1}_{pattern_str}"
            all_row_names.append(row_name)

            # Add the sequence as one row in the combined table.
            all_rows.append(seq)

    df = pd.DataFrame(all_rows, columns=[f'Pos_{i+1}' for i in range(sequence_length)])
    df.index = all_row_names
    df.index.name = 'Sequence_ID'

    return df


# Generate multiple datasets and save the combined sequence matrix as a CSV file.
def generate_sequences(rep, sequence_length, seed, directory):

    random.seed(seed)
    seeds = [random.randint(1, 10000) for _ in range(rep)]
    dataset_names = [f"dataset_{i+1}" for i in range(rep)]

    all_datasets = []
    individual_figures = []

    for seed, name in zip(seeds, dataset_names):

        sequences, patterns, dataset_seed, first_images = generate_dataset(sequence_length, seed, max_attempts=200, dataset_name=name)

        if sequences is None:
            print(f"Retrying {name} with a different seed...")
            sequences, patterns, dataset_seed, first_images = generate_dataset(sequence_length, seed=seed + 1000, max_attempts=250, dataset_name=name)

        if sequences is None:
            continue

        # Analyze the generated dataset before adding it to the combined output.
        analysis = analyze_dataset_with_duplicates_missing(sequences, patterns, first_images, name, sequence_length)

        dataset_info = {
            'name': name,
            'sequences': sequences,
            'patterns': patterns,
            'first_images': first_images,
            'analysis': analysis,
            'seed': dataset_seed,
        }

        all_datasets.append(dataset_info)

    if not all_datasets:
        print("\nNo datasets were successfully generated!")
        return

    # Create and save the combined DataFrame.
    combined_df = create_combined_dataframe(all_datasets, sequence_length)
    csv_filename = directory / "combined_dataset_matrix.csv"
    combined_df.to_csv(csv_filename)
