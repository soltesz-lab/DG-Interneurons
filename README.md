# Reduced Dentate Gyrus Circuit Model

A PyTorch-based reduced computational model of the dentate gyrus
microcircuit, designed to study paradoxical excitation, which the
counterintuitive increase in principal cell activity following
optogenetic stimulation of inhibitory interneurons. The model
implements anatomically constrained connectivity, biophysically
realistic dendritic-somatic transfer functions, and virtual
optogenetic stimulation protocols.

---

## Table of Contents

- [Model Architecture](#model-architecture)
- [Dendritic Transfer Function](#dendritic-transfer-function)
- [Optogenetic Protocol](#optogenetic-protocol)
- [Nested Experiment Design](#nested-experiment-design)
- [Running Experiments: `DG_protocol.py`](#running-experiments-dg_protocolpy)
- [Analysis: `DG_analysis.py`](#analysis-dg_analysispy)
- [Output Structure](#output-structure)

---

## Model Architecture

The model represents five interconnected cell populations within a 1D spatial layout:

| Population | Symbol | Role |
|---|---|---|
| Granule cells | GC | Principal cells; sparse activation (~8%) |
| Mossy cells | MC | Excitatory hilar cells; local + distant projections |
| Parvalbumin interneurons | PV | Fast perisomatic inhibition |
| Somatostatin interneurons | SST | Dendritic inhibition |
| Medial entorhinal cortex inputs | MEC | External excitatory drive |

Default population sizes reflect experimental ratios (e.g., 1000 GC :
30 MC : 30 PV : 20 SST : 60 MEC). Population sizes are configurable
via `CircuitParams`.

### Connectivity

Connectivity is probabilistic and distance-dependent. Granule cell
(GC) to interneuron (IN) and GC to mossy cell (MC) projections are
locally constrained; MC to GC projections span both local and distant
targets, reflecting the known bimodal connectivity of mossy cells. MEC
provides direct drive to GC, PV, MC, and SST, but **not** SST,
creating an asymmetry that shapes feedforward inhibition dynamics.

Per-connection conductances are drawn from configurable distributions
(`PerConnectionSynapticParams`), supporting both AMPA and GABA synapse
types with cell-type-specific mean, standard deviation, and
coefficient of variation.

### Computational Framework

- **Backend**: PyTorch (supports CPU and CUDA)
- **Time integration**: Fixed or gradient-adaptive time stepping (`GradientAdaptiveStepper`)
- **State variables**: Per-cell firing rates, dendritic voltages, somatic voltages, and adaptation currents
- **Optimization**: Particle swarm optimization (PSO) with opposition-based learning for parameter fitting

---

## Dendritic Transfer Function

`dendritic_somatic_transfer.py`

Each cell's current-to-rate conversion uses a two-stage biophysically
realistic transfer function, replacing a simple sigmoid-exponential
with a more mechanistic model. The transfer function is cell-type
specific.

### Dendritic Integration

- **NMDA voltage dependence**: Mg2+ block removal at depolarized potentials scales effective excitatory current nonlinearly.
- **Dendritic Ca2+ spike nonlinearities**: Threshold-dependent boosting of inputs in apical dendrites (active in GC and MC).
- **Cell-type-specific membrane properties**: Input resistance, NMDA weighting, and active dendritic gain differ by population.

### Somatic Firing Rate Conversion

- **Dendrite-soma coupling**: Linear transfer with a cell-type-specific coupling coefficient.
- **Spike frequency adaptation**: Exponential decay of firing rate with a population-specific time constant.
- **Exponential I-F relationship**: Firing rate as a function of somatic current above threshold.

### Cell-Type Parameters

| Cell type | Input resistance | NMDA weight | Active dendrites | Adaptation |
|---|---|---|---|---|
| GC | 300 MOhm (high) | 0.40 (strong) | Yes | Strong |
| MC | 180 MOhm | 0.30 | Yes | Moderate |
| PV | 120 MOhm (low) | 0.15 (minimal) | No | None |
| SST | 200 MOhm | - | Yes | Moderate |

## Optogenetic Protocol

`optogenetic_experiment.py`

Virtual optogenetics is implemented via `OptogeneticExperiment`, which
wraps `DentateCircuit` and manages opsin expression, light delivery,
and trial averaging.

### Opsin Expression

Expression levels are drawn per-cell from a truncated normal
distribution (`OpsinParams.expression_mean`, `expression_std`). A
stochastic failure rate (`OpsinParams.failure_rate`) controls the
fraction of activation events that fail on any given time step,
modeling incomplete ChR2 coupling. These parameters can be varied
systematically (see [Running
Experiments](#running-experiments-dg_protocolpy)).

### Stimulation

`experiment.simulate_stimulation(target_population, light_intensity,
...)` delivers a rectangular light pulse to the specified
population. The light decays spatially following a Hill function with
configurable half-saturation and exponent. Key arguments:

| Argument | Description | Default |
|---|---|---|
| `target_population` | `'pv'` or `'sst'` | — |
| `light_intensity` | Normalized intensity (0–2+) | — |
| `stim_start` | Pulse onset (ms) | 1500.0 |
| `stim_duration` | Pulse duration (ms) | 1000.0 |
| `mec_current` | External MEC drive (pA) | 40.0 |
| `opsin_current` | Max optogenetic current (pA) | 200.0 |
| `n_trials` | Trials to average | 1 |

### Time-Varying MEC Input

The MEC drive can be replaced with a structured temporal pattern to
study interactions between oscillatory input and optogenetic
perturbations:

```bash
python DG_protocol.py --time-varying-mec \
    --mec-pattern-type oscillatory \
    --mec-theta-freq 5.0 \
    --mec-gamma-freq 20.0
```

Pattern types: `oscillatory` (theta-nested gamma), `drift`, `noisy`, `constant`.

### Output

Each `simulate_stimulation` call returns a dictionary including:
- `time`: time vector
- `activity_trace_mean` / `activity_trace_std`: per-population firing rates (cells × time), averaged across trials
- `opsin_expression_mean`: mean expression vector for the target population
- `trial_results`: list of per-trial dicts (when `save_full_activity=True`)
- `adaptive_stats`: time-stepping diagnostics (when adaptive stepping enabled)
- `recorded_currents`: per-source, per-type synaptic currents (when `record_currents=True`)

### Analyzing Non-Expressing Cells

For the stimulated population, results are automatically split into
opsin-expressing and non-expressing (opsin-neg) subpopulations. The
opsin-neg fraction reflects **network-mediated** responses only,
providing a clean readout of paradoxical excitation independent of
direct optogenetic drive.

Metrics stored with `{pop}_nonexpr_*` keys include excited/inhibited
fraction, mean rate change, and trial-to-trial variability.

---

## Nested Experiment Design

`nested_experiment.py`

The nested experiment design decomposes the variance in paradoxical
excitation responses across two hierarchical levels:

```
Level 1:  Connectivity instances (different random circuit realizations)
Level 2:  MEC input patterns (different external drive patterns per circuit)
```

This structure enables **variance decomposition** and **intraclass
correlation coefficient (ICC)** analysis to determine whether
paradoxical excitation is primarily driven by specific synaptic weight
configurations (weight-driven regime, high ICC_connectivity) or by
network-state dynamics (population-dynamics regime, high ICC_mec).

### Configuration

```python
from nested_experiment import NestedExperimentConfig

config = NestedExperimentConfig(
    n_connectivity_instances=10,       # Independent circuit realizations
    n_mec_patterns_per_connectivity=5, # Input patterns per circuit
    base_seed=42,
    save_nested_trials=True,
    save_full_activity=True
)
```

Seed hierarchy is deterministic: connectivity seeds are spaced by
`nested_seeds_offset` (default 10000); MEC pattern seeds are offset by
100 within each connectivity.

### Running a Nested Experiment

```python
from nested_experiment import run_nested_comparative_experiment

nested_results = run_nested_comparative_experiment(
    optimization_json_file='optimized_params.json',
    intensities=[1.0, 1.5],
    mec_current=40.0,
    opsin_current=200.0,
    nested_config=config,
    save_results_file='nested_experiment_results.h5',
    device=device
)
```

Results are saved to HDF5 format for efficient access to large nested datasets.

### Variance Decomposition

```python
from nested_experiment import compute_variance_decomposition, classify_mechanism_regime

variance_analysis = compute_variance_decomposition(nested_results)
regime = classify_mechanism_regime(variance_analysis)
# regime: 'weight_driven', 'population_dynamics', or 'mixed'
```

| ICC_connectivity | ICC_mec | Interpretation |
|---|---|---|
| > 0.4 | < 0.2 | Weight-driven: synaptic structure dominates |
| < 0.2 | > 0.4 | Population-dynamics: network state dominates |
| Both moderate | Both moderate | Mixed regime |

> **Note**: Observed ICC values near 1.0 indicate that cell responses
    are effectively deterministic within each connectivity
    instance. In this case, connectivity instances are the appropriate
    statistical unit, and analysis should use connectivity-level
    resampling rather than cell-level tests.

### Nested Weights Analysis

`nested_weights_analysis.py`

After identifying the dominant regime, the weights analysis examines
whether cells that show paradoxical excitation (excited) differ
systematically in their synaptic input from cells that are suppressed.

The analysis automatically separates **opsin+ cells** (directly
stimulated, potentially confounded) from **opsin-neg cells**
(network-mediated responses only) when the source population matches
the stimulated population, enabling four distinct mechanistic
comparisons:

1. Opsin+ -> excited vs. suppressed (direct pathway)
2. Opsin- -> excited vs. suppressed (indirect/disinhibitory pathway)
3. Opsin+/opsin− difference -> excited cells
4. Opsin+/opsin− difference -> suppressed cells

```python
from nested_weights_analysis import analyze_weights_by_average_response_nested

analysis = analyze_weights_by_average_response_nested(
    nested_results=pv_trials,
    circuit=circuit,
    target_population='pv',
    post_population='gc',
    source_populations=['pv', 'sst', 'mc', 'mec'],
    stim_start=1500.0,
    stim_duration=1000.0,
    warmup=500.0,
    n_bootstrap=10000,
    random_seed=42
)
```

Bootstrap confidence intervals are computed by resampling at the
connectivity level (respecting the hierarchical structure), and effect
sizes are reported as Cohen's *d*.

---

## Running Experiments: `DG_protocol.py`

`DG_protocol.py` is the main entry point for running simulations from
the command line. All experiment types write results to `--output-dir`
(default: `protocol/`).

```
python DG_protocol.py [experiment flags] [options]
```

### Experiment Selection

| Flag | Description |
|---|---|
| `--comparative` | Compare PV vs. SST stimulation at specified intensities |
| `--nested` | Run nested connectivity * MEC pattern experiment |
| `--ablations` | Run circuit ablation tests (block interneuron interactions, excitation to interneurons, or recurrent excitation) |
| `--expression` | Sweep opsin expression levels at fixed failure rate |
| `--failure-rates` | Sweep opsin failure rates at fixed expression level |
| `--all` | Run all of the above |

### Common Options

```
--optimization-file PATH   Load optimized synaptic parameters from JSON
--mec-current FLOAT        MEC drive amplitude in pA (default: 40.0)
--opsin-current FLOAT      Max optogenetic current in pA (default: 200.0)
--stim-start FLOAT         Stimulus onset in ms (default: 1500.0)
--stim-duration FLOAT      Stimulus duration in ms (default: 1000.0)
--n-trials INT             Trials to average per condition (default: 3)
--base-seed INT            Random seed (default: 42)
--device {cpu,cuda}        Compute device (default: auto-detect)
--output-dir DIR           Output directory (default: protocol/)
```

### Nested Experiment Options

```
--n-connectivity INT        Number of circuit instantiations (default: 5)
--n-mec-patterns INT        MEC patterns per connectivity (default: 3)
```

### Opsin Sweep Options

```
# Expression sweep
--expression-levels FLOAT [...]     Expression levels to test (default: 0.1-1.0)

# Failure rate sweep
--failure-rate-values FLOAT [...]   Failure rates to test (default: 0.0-0.9)
--expression-mean FLOAT             Fixed expression level for failure rate tests (default: 0.8)
```

### Paired Mode

Ablation, expression, and failure-rate tests can be run in **paired
mode** by providing a nested experiment HDF5 file. In this mode,
connectivity seeds are loaded from the existing nested experiment,
ensuring that each experimental condition is tested on exactly the
same circuit realizations:

```bash
python DG_protocol.py --ablations \
    --nested-file protocol/nested_experiment_results.h5
```

### Time-Varying MEC Input

```
--time-varying-mec             Enable structured MEC temporal patterns
--mec-pattern-type TYPE        oscillatory | drift | noisy | constant
--mec-theta-freq FLOAT         Theta frequency in Hz (default: 5.0)
--mec-theta-amplitude FLOAT    Theta modulation depth 0–1 (default: 0.3)
--mec-gamma-freq FLOAT         Gamma frequency in Hz (default: 20.0)
--mec-gamma-amplitude FLOAT    Gamma modulation depth 0–1 (default: 0.15)
--mec-gamma-coupling FLOAT     Theta-gamma coupling strength 0–1 (default: 0.8)
--mec-gamma-phase FLOAT        Preferred theta phase for gamma (rad, default: 0.0)
--mec-rotation-groups INT      Spatial rotation groups (default: 3)
```

### Adaptive Time Stepping

```
--adaptive-step                Enable gradient-driven adaptive time stepping
--adaptive-dt-min FLOAT        Minimum time step in ms (default: 0.05)
--adaptive-dt-max FLOAT        Maximum time step in ms (default: 0.25)
--adaptive-gradient-low FLOAT  Low gradient threshold Hz/ms (default: 0.5)
--adaptive-gradient-high FLOAT High gradient threshold Hz/ms (default: 10.0)
```

Adaptive stepping typically reduces the number of integration steps by 20–40% during quiescent periods, at the cost of finer resolution during high-activity epochs (e.g., stimulation onset).

### Data Saving

```
--save-full-activity     Save complete activity traces per trial (larger files)
--record-currents        Record per-source synaptic currents
--no-auto-save           Suppress automatic result saving
--save-results-file PATH Explicit output file path
```

### Usage Examples

```bash
# Basic comparative experiment
python DG_protocol.py --comparative --n-trials 5

# Nested experiment (10 circuits × 5 MEC patterns)
python DG_protocol.py --nested --n-connectivity 10 --n-mec-patterns 5

# Nested experiment with oscillatory MEC drive and adaptive stepping
python DG_protocol.py --nested \
    --n-connectivity 10 --n-mec-patterns 5 \
    --time-varying-mec --mec-pattern-type oscillatory \
    --adaptive-step

# Opsin failure rate sweep (paired with existing nested results)
python DG_protocol.py --failure-rates \
    --nested-file protocol/nested_experiment_results.h5 \
    --failure-rate-values 0.0 0.2 0.4 0.6 0.8

# Full suite with GPU and custom parameters
python DG_protocol.py --all \
    --optimization-file optimized_params.json \
    --device cuda \
    --mec-current 60.0 \
    --opsin-current 200.0 \
    --n-trials 3 \
    --n-connectivity 10 --n-mec-patterns 5 \
    --output-dir results/run_01
```

---

## Analysis: `DG_analysis.py`

Post-hoc analysis and visualization of saved results are handled by
`DG_analysis.py`. Run `python DG_analysis.py --help` for the full list
of subcommands. Key workflows include:

```bash
# Plot comparative results
python DG_analysis.py plot-comparative protocol/DG_experiment_*.pkl

# Bootstrap effect sizes from nested results
python DG_analysis.py bootstrap-analysis protocol/nested_experiment_results.h5

# Nested weights analysis with forest plots
python DG_analysis.py nested-weights-analysis protocol/nested_experiment_results.h5 \
    --target-populations pv \
    --post-populations gc mc \
    --n-bootstrap 10000
```

---

## Output Structure

```
protocol/
├── DG_experiment_*_seed{NN}_*.pkl        # Comparative experiment results
├── nested_experiment_results.h5         # Nested experiment (HDF5)
├── nested_experiment_summary/           # Variance decomposition plots
├── ablation_tests/
│   └── all_ablation_tests.pkl
├── expression_tests/
│   └── expression_results.pkl
├── failure_rate_tests/
│   └── failure_rate_results.pkl
└── analysis/
    ├── bootstrap_analysis_results.pkl
    ├── effect_size_decision_summary.txt
    └── {target}_intensity_{I}/{post_pop}/
        ├── effect_sizes_forest.pdf
        ├── bootstrap_dist_{source}.pdf
        └── weight_distributions_{source}_conn0.pdf
```

---

## Key References

- Hainmueller et al. (2026): paradoxical excitation in dentate gyrus
- Pofahl et al. (2025): excitation-inhibition balance measurements
- Savanthrapadian et al. (2014): interneuron connectivity parameters
- GoodSmith et al. (2025): mossy cell and granule cell functional roles
