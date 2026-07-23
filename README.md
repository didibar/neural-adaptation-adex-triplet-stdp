# Neural Adaptation in an AdEx Network with t-STDP

# Neural Adaptation in an AdEx Network with t-STDP

This repository contains a simulation and analysis pipeline for recurrent spiking neural networks based on Adaptive Exponential Integrate-and-Fire (AdEx) neurons with triplet Spike-Timing-Dependent Plasticity (t-STDP).

The study investigates whether structured input statistics induce neural adaptation and identifies the single-neuron response profiles that give rise to the resulting population-level adaptation.

Neural responses are compared under two experimental conditions:

- **Primed**: transitions between stimuli belonging to the same structured group.
- **Control**: transitions between stimuli belonging to different structured groups.

Adaptation is quantified as the duration of the time interval during which the population firing rate is significantly lower in the **Primed** condition than in the **Control** condition.

The workflow follows a progressive analysis strategy. First, adaptation is characterized at the **population level** through firing-rate dynamics, onset-latency analyses, and population-latency measures. This stage establishes whether a reliable adaptation effect emerges at the network level.

Once population-level adaptation has been established, the analysis proceeds to the **single-neuron level**. Z-score rank analyses and post-hoc neuron-level metrics are used to identify responsive neurons and classify them into two principal response phenotypes: **sharpening** and **fatiguing**.

---

## Repository overview

The codebase is organized around a step-wise pipeline. The main simulation workflow is executed by `step_main.py`, which calls the core steps from input generation to population-level analyses. The Z-score step is intentionally kept separate from the main pipeline and is used later during optimization and post-hoc single-neuron analyses.

---

## Scientific workflow

### 1. Main simulation pipeline

Run:

```bash
python step_main.py
```

`step_main.py` executes the main workflow up to step 6:

```text
step_0  DINO feature-vector extraction from real images
step_1  Synthetic feature-vector generation
step_2  AdEx network learning with triplet STDP
step_3  Network testing with primed/control sequences
step_4  Population firing-rate analysis
step_5  Onset-latency analysis
step_6  Population-latency analysis
```

The script can work with either:

1. **Real image-derived feature vectors** extracted with DINO, using:

```python
file_suffix = "features_normalized"
```

2. **Synthetic feature vectors**, using a parameterized input structure:

```python
file_suffix = f"alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}"
```

The default parameters used in the current scripts are:

```python
alpha = 0.798
beta = 0.727
epsilon = 0.151
seed = 42
```

---

## Input representations

### Real dataset: DINO feature vectors

`step_0_DINO_FVs.py` uses a pre-trained DINO Vision Transformer model to convert `.jpg` images into normalized feature vectors. The expected image categories are:

```text
wild_animals, fruit, flowers, insects, birds,
manmade_food, clothes, furniture, instruments, computer
```

The resulting normalized matrix is saved as:

```text
Results/features_normalized_{seed}/step_0/X_features_normalized.npy
```

### Synthetic dataset: synthetic feature vectors

`step_1_SFVs.py` creates synthetic feature vectors with a block-structured organization. The synthetic input space is controlled by three parameters:

- `alpha`: controls within-block similarity.
- `beta`: controls between-block dissimilarity.
- `epsilon`: controls the higher-level structure by moving samples toward macro-group centroids.

The generated matrix is saved as:

```text
Results/alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}_{seed}/step_1/X_alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}.npy
```

---

## Network model

The network is trained in `step_2_learning.py`. It uses:

- AdEx neuron dynamics.
- Excitatory and inhibitory neuron populations.
- Triplet STDP for synaptic plasticity.
- Brian2 for simulation.

The learned network state is saved as a `.pkl` file in:

```text
Results/<configuration>/step_2/
```

This saved state is then reused during the testing phase.

---

## Primed/Control testing

`step_seq_generator.py` creates sequences of length 5. Each transition between consecutive stimuli is labeled as:

- `P`: primed transition.
- `C`: control transition.

The testing phase in `step_3_testing.py` runs the trained network on these generated sequences.  

---

## Population-level analyses

### Step 4: firing-rate dynamics

`step_4_mean_firing_rate.py` compares Primed and Control population activity over time using a cluster-based permutation approach.

The most important summary values are:

- `total_primed_gt_control`: total duration where Primed activity is greater than Control.
- `total_primed_lt_control`: total duration where Primed activity is lower than Control.

The second quantity, `total_primed_lt_control`, is used as the optimization target because it reflects stronger neural adaptation in the Primed condition.

### Step 5: onset latency

`step_5_onset_latency.py` estimates onset latencies for neurons across analysis windows and compares Primed and Control conditions using paired permutation tests with multiple-comparison correction.

### Step 6: population latency

`step_6_population_latency.py` analyzes the timing of population activation using threshold-crossing latency measures. In the current workflow, two thresholds are used:

```python
threshold1 = 10
threshold2 = 15
```

---

## Z-score analysis

`step_7_Zscore.py` is not part of the standard `step_main.py` workflow. It is used later during the optimization workflow and for post-hoc single-neuron analysis.

This step:

1. Loads activation `.pkl` files.
2. Builds neuron-wise activity dictionaries for Primed and Control transitions.
3. Identifies responsive neurons with a Wilcoxon signed-rank test.
4. Computes stimulus-level Z-scores.
5. Ranks stimuli according to Control response strength.
6. Saves the responsive rank-analysis table:

```text
rank_analysis_data_responsive.csv
```

---

## Differential Evolution optimization

Run:

```bash
python de_optimizer.py
```

`de_optimizer.py` searches for the triplet of synthetic input parameters that maximizes neural adaptation:

```text
alpha, beta, epsilon
```

The optimization uses Differential Evolution over the interval `[0, 1]` for each parameter. Each candidate parameter set runs the simulation and analysis pipeline, including the Z-score step.

The objective is:

```text
maximize total_primed_lt_control
```

Each evaluated configuration is saved using the naming convention:

```text
alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}_{seed}
```

---

## Single-neuron analysis: sharpening vs fatiguing

Run:

```bash
python single_neuron_main.py
```

The single-neuron analysis uses the Z-score rank outputs from the optimization runs to identify and classify responsive neurons.

The analysis applies two filtering stages:

1. Keep neurons with high absolute Z-score responses in both Primed and Control conditions.
2. Keep neurons that respond to at least a minimum number of stimuli.

The current threshold configuration is:

```python
threshold_high = 2
threshold_low = 1
min_stimuli = 3
```

This produces a dedicated output folder:

```text
Z_score/Thr2_thr3_1/
```

The file:

```text
active_neurons_per_config.json
```

stores the active neurons for each optimized configuration and is later used by `Z-score_rank.py`.

`single_neuron_2.py` combines the single-neuron delta summaries with population-level cluster durations from step 4, then generates classification summaries and parameter-space plots.

---

## Ranked Z-score plots

Run:

```bash
python Z-score_rank.py
```

`Z-score_rank.py` creates per-neuron plots for the active neurons listed in:

```text
Z_score/Thr2_thr3_1/active_neurons_per_config.json
```

These plots are useful for visually distinguishing sharpening and fatiguing response profiles at the single-neuron level.

---

## Input-space comparison

Run:

```bash
python input_analysis.py
```

`input_analysis.py` compares the real image-derived DINO feature vectors with the synthetic feature vectors.

It generates comparative plots for:

- Cosine similarity.
- Euclidean distance.
- t-SNE embeddings.

Outputs are saved in:

```text
Input_analysis/
```

---

## Suggested execution order

The execution order depends on whether the goal is to search for the best synthetic input parameters or to rerun the full analysis for an already selected configuration.

### 1. Parameter-optimization workflow

Use this workflow to search for the synthetic feature-vector parameters that maximize neural adaptation.

```bash
# 1. Run parameter optimization over the synthetic feature-vector structure
python de_optimizer.py

# 2. Run the standard simulation pipeline 
python step_main.py 

# 3. Classify optimized configurations and neurons as sharpening/fatiguing
python single_neuron_main.py

# 4. Generate ranked Z-score plots for active neurons
python Z-score_rank.py
```

---


## Requirements

The code uses Python and the following main packages:

```text
numpy
pandas
scipy
matplotlib
brian2
torch
torchvision
Pillow
scikit-learn
statsmodels
pingouin
tqdm
openpyxl
```

A minimal installation command is:

```bash
pip install numpy pandas scipy matplotlib brian2 torch torchvision pillow scikit-learn statsmodels pingouin tqdm openpyxl
```

---

## Path configuration

Several scripts currently use absolute paths.

If the repository is moved to another machine or folder, update the path variables at the top of the relevant scripts, especially:

```text
step_main.py
input_analysis.py
single_neuron_main.py
Z-score_rank.py
de_optimizer.py
```

---

## Citation

If you use this code in a publication, please cite the associated manuscript.

```bibtex
@misc{neural_adaptation_adex_triplet_stdp,
  title  = {Emergent sharpening and fatiguing in a spiking associative memory model of visual priming},
  author = {D. Bartolini, T.P. Reber, T. Tchumatchenko, M. Voigt},
  year   = {2026}
  }
```
