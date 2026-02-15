#!/usr/bin/env python3
"""DG Analysis Script - Offline analysis of dentate gyrus experiments

This script provides analysis and visualization capabilities for
results from DG optogenetic experiments.

Available Commands:
    plot-comparative         Plot PV vs SST comparative results
    plot-ablations           Plot ablation test results
    plot-weights             Plot synaptic weight distributions
    plot-expression          Plot expression level dependence
    plot-failure             Plot failure rate dependence
    plot-combined            Generate combined ablation + expression figure
    plot-currents            Plot synaptic currents
    plot-connectivity-activity  Plot aggregated activity for specific connectivity instance
    analyze-weights-by-response  Statistical weight analysis by response
    plot-variance            Plot variance decomposition
    bootstrap-analysis       Bootstrap effect size analysis
    nested-weights-analysis  Analyze weights by response in nested experiments
    decision-analysis        Effect size decision framework
    export-per-cell-csv      Export per-cell weights and firing rates to CSV

Usage Examples:
    # Basic plotting
    python DG_analysis.py plot-comparative results.pkl
    python DG_analysis.py plot-ablations ablation_results.pkl
    python DG_analysis.py plot-weights results.pkl

    # Connectivity-specific plotting
    python DG_analysis.py plot-connectivity-activity nested_results.h5 \\
        --connectivity-idx 0 --target-population pv --intensity 1.0

    # Advanced analyses
    python DG_analysis.py analyze-weights-by-response results.pkl --n-bootstrap 10000
    python DG_analysis.py bootstrap-analysis nested_results.pkl --n-samples 10000
    python DG_analysis.py decision-analysis nested_results.pkl --target-power 0.8
    python DG_analysis.py nested-weights-analysis nested_results.pkl

    # Combined figures
    python DG_analysis.py plot-combined results.pkl ablation.pkl expression.pkl

    # Export per-cell data to CSV
    python DG_analysis.py export-per-cell-csv nested_results.h5
    python DG_analysis.py export-per-cell-csv nested_results.h5 \\
        --csv-filename my_data.csv \\
        --target-populations pv \\
        --intensities 1.0 1.5
    
    # Show help for specific command
    python DG_analysis.py plot-comparative --help
"""

import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import logging
import h5py
from hdf5_storage import load_nested_trials_from_hdf5, load_metadata_from_hdf5
import torch
import matplotlib.pyplot as plt



from DG_circuit_dendritic_somatic_transfer import (
    CircuitParams, PerConnectionSynapticParams, OpsinParams
)
from DG_visualization import DGCircuitVisualization
from optogenetic_experiment import (aggregate_trial_results,
                                    aggregate_adaptive_stats)
from ablation_tests import (plot_ablation_results_violin,
                            export_ablation_results_csv)
# Import functions from DG_protocol module
from DG_protocol import (
    OptogeneticExperiment, set_random_seed, get_default_device,
    
    # Loading
    load_experiment_results,
    
    # Plotting - Weights
    plot_synaptic_weights_from_results,
    plot_synaptic_distributions,
    
    # Plotting - Currents
    plot_currents_from_results,
    
    # Analysis - Weights
    analyze_synaptic_weights_by_response_from_results,
    
    # Nested experiment analysis
    plot_variance_decomposition,
    plot_connectivity_instance_variance,
    plot_connectivity_instance_variance_detailed,
    print_nested_experiment_summary,
    
    # Bootstrap and decision framework
    run_nested_effect_size_analysis,
    run_nested_effect_size_decision_analysis,
)
from nested_experiment import (
    export_nested_experiment_csv,
    export_nested_experiment_csv_from_hdf5,
    export_per_cell_weights_from_hdf5
)
from nested_weights_analysis import (
    analyze_weights_by_average_response_nested,
    plot_weights_by_average_response_nested,
    plot_connectivity_weight_comparison,
    plot_summary_forest_plot_all_targets,
    plot_pca_summary_all_targets
)
from nested_weights_dist_analysis import (
    analyze_weights_distributional_nested,
    plot_distributional_analysis_nested,
    plot_distributional_analysis_nested_opsin_aware,
    plot_distributional_violin_grid_across_populations_with_boxplot,
    plot_quantile_summary_across_populations
)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

plt.rcParams['axes.grid'] = False

# ============================================================================
# Helper Functions
# ============================================================================

def load_results_with_validation(filepath: str, result_type: str) -> Dict:
    """
    Load and validate results file (supports both pickle and HDF5)
    
    Args:
        filepath: Path to results file (*.pkl or *.h5)
        result_type: Expected type ('comparative', 'ablation', 'expression', 'nested')
        
    Returns:
        Loaded results dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or unexpected type
    """
    filepath = Path(filepath)
    
    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    # Check if HDF5 or pickle
    is_hdf5 = filepath.suffix == '.h5'
    
    if result_type == 'nested':
        # Nested results can be either HDF5 or pickle
        if is_hdf5:
            logger.info(f"Loading nested results from HDF5: {filepath}")
            
            # Load metadata and seed structure
            from hdf5_storage import load_metadata_from_hdf5
            import h5py
            
            metadata = load_metadata_from_hdf5(str(filepath))
            
            required_keys = ['n_connectivity_instances', 'n_mec_patterns_per_connectivity']
            missing_keys = [k for k in required_keys if k not in metadata]
            if missing_keys:
                raise ValueError(f"HDF5 file missing required metadata: {missing_keys}")
            
            # Load seed_structure from HDF5
            seed_structure = {}
            with h5py.File(str(filepath), 'r') as f:
                if 'metadata/seed_structure' in f:
                    seed_group = f['metadata/seed_structure']
                    if 'connectivity_seeds' in seed_group:
                        seed_structure['connectivity_seeds'] = seed_group['connectivity_seeds'][()].tolist()
                    if 'mec_pattern_seeds' in seed_group:
                        seed_structure['mec_pattern_seeds'] = seed_group['mec_pattern_seeds'][()].tolist()
            
            logger.info(f"  Format: HDF5")
            logger.info(f"  Connectivity instances: {metadata['n_connectivity_instances']}")
            logger.info(f"  MEC patterns: {metadata['n_mec_patterns_per_connectivity']}")
            if seed_structure:
                logger.info(f"  Loaded {len(seed_structure.get('connectivity_seeds', []))} connectivity seeds")
            
            # Return metadata with HDF5 file path and seed_structure
            return {
                'metadata': metadata,
                'hdf5_file': str(filepath),
                'nested_results': None,  # Not loaded into memory
                'seed_structure': seed_structure,  # Add this!
                'file_format': 'hdf5'
            }
        else:
            # Pickle format
            logger.info(f"Loading nested results from pickle: {filepath}")
            
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load pickle file: {e}")
            
            # Validate pickle format
            if not isinstance(data, dict):
                raise ValueError("Invalid nested results format. Expected dictionary")
            
            required_keys = ['nested_results', 'metadata']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                raise ValueError(f"Nested results missing required keys: {missing_keys}")
            
            logger.info(f"  Format: Pickle")
            if 'metadata' in data:
                metadata = data['metadata']
                if 'n_connectivity' in metadata:
                    logger.info(f"  Connectivity instances: {metadata['n_connectivity']}")
                if 'n_mec_patterns' in metadata:
                    logger.info(f"  MEC patterns: {metadata['n_mec_patterns']}")
            
            data['file_format'] = 'pickle'
            return data
    
    # For non-nested types, use original pickle loading
    if filepath.suffix != '.pkl':
        logger.warning(f"Unexpected file extension: {filepath.suffix} (expected .pkl)")
    
    # Load pickle file
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load results file: {e}")
    
    # Validate based on expected type
    if result_type == 'comparative':

        # Check if it's the new dict format or legacy tuple format
        if isinstance(data, dict):
            # New format: extract from dict
            if 'results' not in data:
                raise ValueError("Comparative results missing 'results' key")

            results = data['results']
            conn_analysis = data.get('connectivity_analysis', {})
            cond_analysis = data.get('conductance_analysis', {})
            metadata = data.get('metadata', {})

            # Validate results structure
            if 'pv' not in results and 'sst' not in results:
                raise ValueError("Comparative results missing 'pv' and 'sst' populations")

            logger.info(f"Loaded comparative results from {filepath}")
            logger.info(f"  Populations: {list(results.keys())}")
            if metadata and 'n_trials' in metadata:
                logger.info(f"  Trials: {metadata['n_trials']}")

            # Return as tuple for consistency
            return (results, conn_analysis, cond_analysis, metadata)

        elif isinstance(data, tuple) and len(data) == 4:
            # Legacy format: already a tuple
            results, conn_analysis, cond_analysis, metadata = data

            if 'pv' not in results and 'sst' not in results:
                raise ValueError("Comparative results missing 'pv' and 'sst' populations")

            logger.info(f"Loaded comparative results from {filepath}")
            logger.info(f"  Populations: {list(results.keys())}")
            if metadata and 'n_trials' in metadata:
                logger.info(f"  Trials: {metadata['n_trials']}")

            return data

        else:
            raise ValueError(
                "Invalid comparative results format. Expected either:\n"
                "  - Dictionary with 'results', 'connectivity_analysis', "
                "'conductance_analysis', 'metadata' keys\n"
                "  - Tuple of (results, connectivity_analysis, conductance_analysis, metadata)"
            )
        
        
    elif result_type == 'ablation':
        if not isinstance(data, dict):
            raise ValueError("Invalid ablation results format. Expected dictionary")
        
        expected_keys = ['interneuron_interactions', 'excitation_to_interneurons', 
                         'recurrent_excitation']
        if not any(k in data for k in expected_keys):
            raise ValueError(
                f"Ablation results missing expected keys. Found: {list(data.keys())}"
            )
        
        logger.info(f"Loaded ablation results from {filepath}")
        logger.info(f"  Test types: {list(data.keys())}")
        
        return data
        
    elif result_type == 'expression':
        if not isinstance(data, dict):
            raise ValueError("Invalid expression results format. Expected dictionary")
        
        if 'full_network' not in data:
            raise ValueError("Expression results missing 'full_network' key")
        
        logger.info(f"Loaded expression results from {filepath}")
        logger.info(f"  Conditions: {list(data.keys())}")
        
        return data
        
    else:
        raise ValueError(f"Unknown result type: {result_type}")


def setup_output_directory(output_path: str, create: bool = True) -> Path:
    """
    Create and validate output directory
    
    Args:
        output_path: Path to output directory
        create: Whether to create directory if it doesn't exist
        
    Returns:
        Path object for output directory
    """
    output_dir = Path(output_path)
    
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
    elif not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    
    return output_dir


def validate_intensities(intensities: Optional[list], available: list) -> list:
    """
    Validate and filter intensity values
    
    Args:
        intensities: Requested intensities (None = use all available)
        available: Available intensities in results
        
    Returns:
        List of validated intensities to use
    """
    if intensities is None:
        return sorted(available)
    
    # Check that requested intensities are available
    invalid = [i for i in intensities if i not in available]
    if invalid:
        logger.warning(f"Requested intensities not in results: {invalid}")
        logger.warning(f"Available intensities: {sorted(available)}")
    
    # Filter to valid intensities
    valid = [i for i in intensities if i in available]
    if not valid:
        raise ValueError("No valid intensities specified")
    
    return sorted(valid)


def filter_trials_for_connectivity(nested_data: Dict,
                                   target_population: str,
                                   intensity: float,
                                   connectivity_idx: int) -> List:
    """
    Filter trials for a specific connectivity instance
    
    Args:
        nested_data: Loaded nested experiment data
        target_population: Target population ('pv', 'sst')
        intensity: Light intensity
        connectivity_idx: Connectivity instance index
        
    Returns:
        List of trial results for this connectivity/target/intensity combination
    """
    file_format = nested_data.get('file_format', 'pickle')
    
    if file_format == 'hdf5':
        # Load trials from HDF5
        hdf5_file = nested_data['hdf5_file']
        metadata = nested_data['metadata']
        
        n_connectivity = metadata['n_connectivity_instances']
        n_mec_patterns = metadata['n_mec_patterns_per_connectivity']
        
        with h5py.File(hdf5_file, 'r') as f:
            all_trials = load_nested_trials_from_hdf5(
                f, target_population, intensity, 
                n_connectivity, n_mec_patterns,
                require_full_activity=True,
                conn_idx_filter=[connectivity_idx]
            )
        
        # Filter for specific connectivity
        filtered_trials = [
            trial for trial in all_trials 
            if trial.connectivity_idx == connectivity_idx
        ]
        
    else:
        # Pickle format
        nested_results = nested_data['nested_results']
        
        if target_population not in nested_results:
            return []
        
        if intensity not in nested_results[target_population]:
            return []
        
        all_trials = nested_results[target_population][intensity]
        
        # Filter for specific connectivity
        filtered_trials = [
            trial for trial in all_trials 
            if trial.connectivity_idx == connectivity_idx
        ]
    
    return filtered_trials

def convert_trial_results_to_torch(trial_results: List[Dict],
                                   device: Optional[torch.device] = None) -> List[Dict]:
    """
    Convert trial results from numpy arrays to torch tensors
    
    Handles data loaded from HDF5/pickle files where arrays are stored as numpy.
    
    Args:
        trial_results: List of trial result dicts (may contain numpy arrays)
        device: Device to create tensors on
        
    Returns:
        List of trial result dicts with torch tensors
    """
    if device is None:
        device = get_default_device()
    
    converted_results = []
    
    for trial_result in trial_results:
        converted = {}
        
        for key, value in trial_result.items():
            if isinstance(value, np.ndarray):
                # Convert numpy to torch tensor
                converted[key] = torch.from_numpy(value).to(device)
            elif isinstance(value, dict):
                # Recursively convert nested dicts (e.g., activity_trace)
                converted[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        converted[key][subkey] = torch.from_numpy(subvalue).to(device)
                    else:
                        converted[key][subkey] = subvalue
            else:
                # Keep other types as-is
                converted[key] = value
        
        converted_results.append(converted)
    
    return converted_results


def plot_comparative_experiment_results(results: Dict, conn_analysis: Dict,
                                        stimulation_level: float = 1.0,
                                        metadata: Optional[Dict] = None,
                                        save_path: Optional[str] = None) -> None:
    """
    Create visualizations from comparative experiment results
    
    Now handles multi-trial statistics with error bars
    """

    # Helper function to handle both numpy arrays and torch tensors
    def to_numpy(arr):
        """Convert torch tensor to numpy if needed"""
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return np.array(arr)
    
    stim_start = None
    stim_duration = None
    warmup = None
    if metadata is not None:
        stim_start = metadata.get('stim_start', 1500.0)
        stim_duration = metadata.get('stim_duration', 1000.0)
        warmup = metadata.get('warmup', 500.0)

    # Check if we have multi-trial data
    n_trials = results['pv'][stimulation_level].get('n_trials', 1)
    has_multitrial = n_trials > 1
        
    # Define colors matching the paper
    colors = {
        'pv': '#FF6B9D', # Pink 
        'sst': '#45B7D1', # Blue
        'gc': '#96CEB4', # Green 
        'mc': '#FFEAA7', # Yellow
    }
    
    # Create summary figure
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Firing ratio bar plots
    ax1 = plt.subplot(3, 4, (1, 2))

    targets = ['pv', 'sst']
    populations = ['gc', 'mc', 'pv', 'sst']

    bar_data = []
    bar_errors = []
    bar_labels = []
    bar_colors = []
    individual_points = []  # Store individual trial data points
    point_colors = []  # Store colors for each point based on cell responses

    # Get time array and create masks (needed for individual trial analysis)
    # Extract from first available trial_results if present
    time = None
    for target in ['pv', 'sst']:
        if target in results and stimulation_level in results[target]:
            if 'trial_results' in results[target][stimulation_level]:
                trial_results = results[target][stimulation_level]['trial_results']
                if len(trial_results) > 0:
                    time = trial_results[0]['time']
                    break
                
    for target in targets:

        # Create time masks for baseline and stimulation periods
        if (metadata is not None) and (time is not None):
            baseline_mask = (time >= warmup) & (time < stim_start)
            stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
        else:
            # Fallback: create empty masks (won't be used if no trial_results)
            baseline_mask = None
            stim_mask = None
        
        for pop in populations:
            if pop != target and f'{pop}_mean_change' in results[target][stimulation_level]:
                baseline_rate = results[target][stimulation_level][f'{pop}_mean_baseline_rate']
                stim_rate = results[target][stimulation_level][f'{pop}_mean_stim_rate']

                if baseline_rate > 0:
                    ratio = np.log2(stim_rate / baseline_rate)
                    bar_data.append(ratio)
                    bar_labels.append(f'{target.upper()}->{pop.upper()}')
                    bar_colors.append(colors[pop])

                    # Extract individual trial data points and classify responses
                    trial_points = []
                    trial_colors = []
                    if has_multitrial and 'trial_results' in results[target][stimulation_level]:
                        for trial_result in results[target][stimulation_level]['trial_results']:
                            trial_activity = to_numpy(trial_result['activity_trace'][pop])

                            # Convert masks to numpy boolean arrays
                            baseline_mask_np = to_numpy(baseline_mask)
                            stim_mask_np = to_numpy(stim_mask)
                            
                            # Calculate per-cell baseline and stim rates
                            trial_baseline = np.mean(trial_activity[:, baseline_mask_np], axis=1)
                            trial_stim = np.mean(trial_activity[:, stim_mask_np], axis=1)
                            trial_change = trial_stim - trial_baseline
                            trial_baseline_std = np.std(trial_baseline)

                            # Classify cells as excited/suppressed/unchanged
                            excited_cells = (trial_change > trial_baseline_std)
                            suppressed_cells = (trial_change < -trial_baseline_std)
                            unchanged_cells = 1.0 - excited_cells - suppressed_cells

                            frac_excited = np.mean(excited_cells)
                            frac_suppressed = np.mean(suppressed_cells)
                            frac_unchanged = np.mean(unchanged_cells)

                            # Determine point color based on dominant response
                            if frac_excited > max(frac_suppressed, frac_unchanged):
                                point_color = 'red'  # Majority excited
                            elif frac_suppressed > max(frac_excited, frac_unchanged):
                                point_color = 'blue'  # Majority suppressed
                            else:
                                point_color = 'gray'  # Balanced or unchanged

                            # Calculate trial mean ratio
                            trial_baseline_mean = np.mean(trial_baseline)
                            trial_stim_mean = np.mean(trial_stim)

                            if trial_baseline_mean > 0:
                                trial_ratio = np.log2(trial_stim_mean / trial_baseline_mean)
                                trial_points.append(trial_ratio)
                                trial_colors.append(point_color)

                    individual_points.append(trial_points)
                    point_colors.append(trial_colors)

                    # Add error bars if multi-trial
                    if has_multitrial and f'{pop}_mean_change_std' in results[target][stimulation_level]:
                        std = results[target][stimulation_level][f'{pop}_mean_change_std']
                        error = std / (baseline_rate * np.log(2))
                        bar_errors.append(error)
                    else:
                        bar_errors.append(0)

    if bar_data:
        x_pos = np.arange(len(bar_labels))
        bars = ax1.bar(x_pos, bar_data, color=bar_colors, alpha=0.7, 
                       edgecolor='black', linewidth=1.5,
                       yerr=bar_errors if has_multitrial else None,
                       capsize=5)

        # Overlay individual data points with color coding
        if has_multitrial:
            for i, (points, colors_for_points) in enumerate(zip(individual_points, point_colors)):
                if len(points) > 0:
                    # Add jitter to x-position for visibility
                    x_jittered = np.random.normal(x_pos[i], 0.05, len(points))
                    ax1.scatter(x_jittered, points, c=colors_for_points, s=40, 
                                alpha=0.7, zorder=10, edgecolors='white', linewidth=0.8)

        # Value labels on bars
        for i, (bar, value) in enumerate(zip(bars, bar_data)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')

        # Add legend for point colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Excited'),
            Patch(facecolor='blue', alpha=0.7, label='Suppressed'),
            Patch(facecolor='gray', alpha=0.7, label='Balanced/Unchanged')
        ]
        ax1.legend(handles=legend_elements, loc='lower left', fontsize=9, 
                  framealpha=0.9, title='Trial Response')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel(r'Modulation Ratio ($\log_2$)', fontsize=12)
    title_suffix = f' (n={n_trials} trials)' if has_multitrial else ' (Single Trial)'
    ax1.set_title(f'Firing Rate Modulation{title_suffix}', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Panel B: Network effects summary with error bars
    ax2 = plt.subplot(3, 4, (3, 4))
    
    effect_data = []
    effect_errors = []
    effect_labels = []
    
    for target in targets:
        for pop in populations:
            if pop != target:
                excited_key = f'{pop}_excited'
                if excited_key in results[target][stimulation_level]:
                    excited_frac = results[target][stimulation_level][excited_key]
                    inhibited_frac = results[target][stimulation_level][f'{pop}_inhibited']
                    
                    effect_data.append([excited_frac, inhibited_frac])
                    effect_labels.append(f'{target.upper()}->{pop.upper()}')
                    
                    # Add error bars if available
                    if has_multitrial and f'{pop}_excited_std' in results[target][stimulation_level]:
                        excited_std = results[target][stimulation_level][f'{pop}_excited_std']
                        effect_errors.append([excited_std, excited_std])  # Same for both
                    else:
                        effect_errors.append([0, 0])
    
    if effect_data:
        effect_array = np.array(effect_data)
        effect_errors_array = np.array(effect_errors)
        x_pos = np.arange(len(effect_labels))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, effect_array[:, 0], width, 
               label='Excited', color='red', alpha=0.7, edgecolor='black',
               yerr=effect_errors_array[:, 0] if has_multitrial else None,
               capsize=5)
        bars2 = ax2.bar(x_pos + width/2, effect_array[:, 1], width, 
               label='Inhibited', color='blue', alpha=0.7, edgecolor='black',
               yerr=effect_errors_array[:, 1] if has_multitrial else None,
               capsize=5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom',
                        fontsize=10)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(effect_labels, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Fraction of Cells', fontsize=12)
        title_suffix = f' (n={n_trials} trials)' if has_multitrial else ' (Single Trial)'
        ax2.set_title(f'Network Effects{title_suffix}', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_axisbelow(True)
    
    # Panel C: Firing rate changes bar plot with error bars
    ax4 = plt.subplot(3, 4, (5, 6))
    
    change_data = []
    change_errors = []
    change_labels = []
    change_colors = []
    
    for target in targets:
        for pop in ['gc', 'mc']:
            if f'{pop}_mean_change' in results[target][stimulation_level]:
                mean_change = results[target][stimulation_level][f'{pop}_mean_change']
                
                change_data.append(mean_change)
                change_labels.append(f'{target.upper()}->{pop.upper()}')
                change_colors.append(colors['pv'] if target == 'pv' else colors['sst'])
                
                # Add error bars if available
                if has_multitrial and f'{pop}_mean_change_std' in results[target][stimulation_level]:
                    std = results[target][stimulation_level][f'{pop}_mean_change_std']
                    change_errors.append(std)
                else:
                    change_errors.append(0)
    
    if change_data:
        bars = ax4.bar(range(len(change_labels)), change_data, 
                      color=change_colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                      yerr=change_errors if has_multitrial else None,
                      capsize=5)
        
        # Add value labels
        for bar, value in zip(bars, change_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')
        
        ax4.set_xticks(range(len(change_labels)))
        ax4.set_xticklabels(change_labels, fontsize=10)
        ax4.set_ylabel(r'$\Delta$ Firing Rate (Hz)', fontsize=12)
        title_suffix = f' (n={n_trials} trials)' if has_multitrial else ' (Single Trial)'
        ax4.set_title(f'Mean Rate Changes{title_suffix}', fontsize=12, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_axisbelow(True)

    # Panel D: Scatter plots showing correlation
    for i, target in enumerate(targets):
        ax = plt.subplot(3, 4, 7 + i)
        
        opsin_expression = results[target][stimulation_level]['opsin_expression_mean']

        if has_multitrial or (f'{target}_stim_rates_mean' in results[target][stimulation_level]):
            stim_rates = results[target][stimulation_level][f'{target}_stim_rates_mean'][opsin_expression <= 0.2]
            baseline_rates = results[target][stimulation_level][f'{target}_baseline_rates_mean'][opsin_expression <= 0.2]
        else:
            stim_rates = results[target][stimulation_level][f'{target}_stim_rates'][opsin_expression <= 0.2]
            baseline_rates = results[target][stimulation_level][f'{target}_baseline_rates'][opsin_expression <= 0.2]

        ax.scatter(baseline_rates, stim_rates, c=colors[target], alpha=0.6, s=30, 
                  edgecolors='black', linewidth=0.5)
        
        # Add correlation line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(baseline_rates, stim_rates)
        
        line = slope * baseline_rates + intercept
        ax.plot(baseline_rates, line, 'r--', alpha=0.8, linewidth=2, 
               label=f'Fit (R={r_value:.2f})')

        if hasattr(baseline_rates, 'numpy'):
            baseline_rates = baseline_rates.numpy()
        if hasattr(stim_rates, 'numpy'):
            stim_rates = stim_rates.numpy()
            
        # Identity line
        max_rate = max(np.max(baseline_rates), np.max(stim_rates))
        ax.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, linewidth=1.5, 
               label='Identity')
        
        ax.set_xlabel('Baseline Rate (Hz)', fontsize=10)
        ax.set_ylabel('Stimulation Rate (Hz)', fontsize=10)
        title_suffix = f'\n(n={n_trials} trials)' if has_multitrial else '\n(Single Trial)'
        ax.set_title(f'{target.upper()} Stimulation{title_suffix}\n(Non-expressing cells)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_axisbelow(True)

    # Panel E: Non-expressing cells in target populations
    for i, target in enumerate(targets):
        ax = plt.subplot(3, 4, 9 + i)

        # Check if non-expressing cell data exists
        if f'{target}_nonexpr_excited' in results[target][stimulation_level]:
            nonexpr_data = results[target][stimulation_level]

            # Get statistics
            excited_nonexpr = nonexpr_data[f'{target}_nonexpr_excited']
            n_nonexpr = nonexpr_data.get(f'{target}_nonexpr_count', 0)

            if has_multitrial and f'{target}_nonexpr_excited_std' in nonexpr_data:
                excited_std = nonexpr_data[f'{target}_nonexpr_excited_std']
            else:
                excited_std = 0

            # Create bar plot
            x_pos = [0]
            bars = ax.bar(x_pos, [excited_nonexpr * 100], 
                         yerr=[excited_std * 100] if has_multitrial else None,
                         color='purple', alpha=0.7, edgecolor='black', linewidth=1.5,
                         capsize=5)

            # Add value label
            height = bars[0].get_height()
            ax.text(0, height, f'{excited_nonexpr*100:.1f}%\n(n={n_nonexpr})',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_ylabel('% Paradoxically Excited', fontsize=10)
            title_suffix = f' (n={n_trials} trials)' if has_multitrial else ' (Single Trial)'
            ax.set_title(f'{target.upper()} Non-Expressing Cells{title_suffix}',
                        fontsize=11, fontweight='bold', color='purple')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(100, excited_nonexpr * 120))
        else:
            # No data available
            ax.text(0.5, 0.5, 'No non-expressing\ncell data available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic')
            ax.axis('off')
        
    # Panel F: Summary text
    ax6 = plt.subplot(3, 4, 12)
    ax6.axis('off')
    
    trial_text = f'{n_trials} Trial Average' if has_multitrial else 'Single Trial'
    summary_text = f"{trial_text} Summary\n"
    summary_text += "=" * 30 + "\n\n"
    
    # Network effects
    summary_text += "Optogenetic Effects:\n"
    for target in targets:
        summary_text += f"{target.upper()} stimulation:\n"
        for pop in ['gc', 'mc']:
            if f'{pop}_excited' in results[target][stimulation_level]:
                excited = results[target][stimulation_level][f'{pop}_excited']
                if has_multitrial and f'{pop}_excited_std' in results[target][stimulation_level]:
                    std = results[target][stimulation_level][f'{pop}_excited_std']
                    summary_text += f"  {pop.upper()}: {excited:.1%} ± {std:.1%} excited\n"
                else:
                    summary_text += f"  {pop.upper()}: {excited:.1%} excited\n"
        summary_text += "\n"

    summary_text += "\nNon-Expressing Cells:\n"
    for target in targets:
        if f'{target}_nonexpr_excited' in results[target][stimulation_level]:
            excited_nonexpr = results[target][stimulation_level][f'{target}_nonexpr_excited']
            n_nonexpr = results[target][stimulation_level].get(f'{target}_nonexpr_count', 0)

            if has_multitrial and f'{target}_nonexpr_excited_std' in results[target][stimulation_level]:
                std = results[target][stimulation_level][f'{target}_nonexpr_excited_std']
                summary_text += f"{target.upper()}: {excited_nonexpr:.1%} ± {std:.1%} (n={n_nonexpr})\n"
            else:
                summary_text += f"{target.upper()}: {excited_nonexpr:.1%} (n={n_nonexpr})\n"
        else:
            summary_text += f"{target.upper()}: No data\n"
        
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, va='top', ha='left',
            fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
    main_title = 'Dentate Gyrus Interneuron Stimulation Effects'
    if has_multitrial:
        main_title += f' (Average of {n_trials} Trials)'
    else:
        main_title += ' (Representative Single Trial)'
        
    plt.suptitle(main_title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        trial_suffix = f'_n{n_trials}trials' if has_multitrial else '_single_trial'
        plt.savefig(f"{save_path}/DG_comparative_experiment_stim_{stimulation_level}{trial_suffix}.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/DG_comparative_experiment_stim_{stimulation_level}{trial_suffix}.png", 
                   dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_failure_rate_overlay(
    results: Dict,
    save_path: Optional[str] = None,
    error_type: str = 'sem',
    figsize: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
) -> plt.Figure:
    """
    Plot failure rate results with all cell types overlaid on the same axes.
    
    Shows fraction excited (%) vs fraction of opsin-expressing cells with 
    functional opsins (%).  The x-axis represents the effective expression
    accounting for failure rate: (1 - failure_rate) * expression_mean * 100.

    For each target population (PV, SST), creates one panel per ablation
    config showing fraction excited vs effective expression for all cell 
    types as colored lines.

    Args:
        results: Dictionary from test_opsin_failure_rates().
            Structure: results[config_name][target][failure_rate] -> analysis dict
        save_path: Optional directory path to save figure (PDF + PNG).
        error_type: 'std' for standard deviation, 'sem' for standard error,
            or 'none' to suppress error bands.
        figsize: Override default figure size (width, height) in inches.
        show_legend: Whether to display a legend on the first panel.

    Returns:
        matplotlib Figure object.
    """
    # --- extract structure ---
    config_names = list(results.keys())[:1]
    targets = list(results[config_names[0]].keys())
    failure_rates = sorted(results[config_names[0]][targets[0]].keys())

    n_configs = len(config_names)
    n_targets = len(targets)

    # --- colors matching the reference figure ---
    pop_colors = {
        'pv': '#CC00CC',   # magenta
        'sst': '#00AA44',  # green
        'gc': '#DDBB00',   # gold / yellow
        'mc': '#AA0000',   # dark red
    }
    pop_markers = {'pv': 's', 'sst': '^', 'gc': 'o', 'mc': 'D'}
    pop_labels = {'pv': 'PV', 'sst': 'SST', 'gc': 'GC', 'mc': 'MC'}

    config_labels = {
        'full_network': 'Full Network',
        'blocked_exc_to_int': 'Block Exc$\\to$Int',
    }

    # --- populations to plot per target ---
    pop_order = {
        'pv': ['pv_nonexpr', 'sst', 'gc', 'mc'],
        'sst': ['pv', 'sst_nonexpr', 'gc', 'mc'],
    }

    # --- layout ---
    if figsize is None:
        figsize = (4.0 * n_configs + 0.5, 3.5 * n_targets + 0.5)
    fig, axes = plt.subplots(
        n_targets, n_configs, figsize=figsize,
        squeeze=False, sharex=True, sharey=True,
    )

    # Convert failure rates to fraction with functional opsins
    # Assume constant expression_mean across failure rates
    # (extract from first available result)
    expression_mean = None
    for config_name in config_names:
        for target in targets:
            for failure_rate in failure_rates:
                # Try to get expression_mean from metadata if stored
                # Otherwise infer from the experimental design (typically 0.5)
                data = results[config_name][target].get(failure_rate, {})
                if 'expression_mean' in data:
                    expression_mean = data['expression_mean']
                    break
            if expression_mean is not None:
                break
        if expression_mean is not None:
            break
    
    if expression_mean is None:
        expression_mean = 0.8
        logger.warning(f"Could not determine expression_mean from results, "
                      f"assuming {expression_mean}")

    # Calculate effective expression: (1 - failure_rate)
    # This represents the fraction of cells with functional opsins
    x_vals = np.array([(1.0 - fr) * 100.0 for fr in failure_rates])

    for row_idx, target in enumerate(targets):
        populations = pop_order.get(target, ['gc', 'mc', 'pv', 'sst'])

        for col_idx, config_name in enumerate(config_names):
            ax = axes[row_idx, col_idx]

            for pop in populations:
                # --- resolve data keys ---
                is_nonexpr = pop.endswith('_nonexpr')
                if is_nonexpr:
                    base_pop = pop.replace('_nonexpr', '')
                    excited_key = f'{base_pop}_nonexpr_excited'
                    std_key = f'{base_pop}_nonexpr_excited_std'
                    sem_key = f'{base_pop}_nonexpr_excited_sem'
                    display_pop = base_pop
                else:
                    excited_key = f'{pop}_excited'
                    std_key = f'{pop}_excited_std'
                    sem_key = f'{pop}_excited_sem'
                    display_pop = pop

                # --- collect values across failure rates ---
                y_vals = []
                y_errs = []
                for failure_rate in failure_rates:
                    data = results[config_name][target].get(failure_rate, {})
                    y_vals.append(data.get(excited_key, 0.0) * 100.0)

                    if error_type == 'sem' and sem_key in data:
                        y_errs.append(data[sem_key] * 100.0)
                    elif error_type == 'std' and std_key in data:
                        y_errs.append(data[std_key] * 100.0)
                    else:
                        y_errs.append(0.0)

                y_vals = np.array(y_vals)
                y_errs = np.array(y_errs)

                color = pop_colors.get(display_pop, 'gray')
                marker = pop_markers.get(display_pop, 'o')
                label_suffix = ' (non-expr)' if is_nonexpr else ''
                label = pop_labels.get(display_pop, display_pop.upper()) + label_suffix

                # --- plot line + optional error band ---
                ax.plot(
                    x_vals, y_vals,
                    color=color, marker=marker, markersize=5,
                    linewidth=1.8, label=label, alpha=0.9,
                )
                if error_type != 'none' and np.any(y_errs > 0):
                    ax.fill_between(
                        x_vals, y_vals - y_errs, y_vals + y_errs,
                        color=color, alpha=0.15,
                    )

            # --- formatting ---
            ax.set_xlabel(r'Functional opsin expression (%)', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(r'Fraction excited (%)', fontsize=10)

            config_title = config_labels.get(config_name, config_name)
            ax.set_title(
                f'{target.upper()} stim - {config_title}',
                fontsize=10, fontweight='bold',
            )
            ax.set_axisbelow(True)

            if show_legend and row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='best', framealpha=0.8)

    fig.suptitle(
        f'Fraction Excited vs. Opsin Expression\n'
        f'(Expression mean = {expression_mean:.1%}, varying failure rate)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            save_dir / 'failure_rate_overlay.pdf',
            dpi=300, bbox_inches='tight',
        )
        fig.savefig(
            save_dir / 'failure_rate_overlay.png',
            dpi=300, bbox_inches='tight',
        )
        logger.info(f"Saved failure rate overlay plots to {save_dir}")

    plt.show()
    return fig

    
def plot_expression_level_overlay(
    results: Dict,
    save_path: Optional[str] = None,
    error_type: str = 'sem',
    figsize: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
    x_as_percent: bool = True,
) -> plt.Figure:
    """
    Plot expression level results with all cell types overlaid on the same axes.

    For each target population (PV, SST), creates one panel per ablation
    config showing fraction excited (%) vs opsin expression (%) for all
    cell types as colored lines.  When multiple configs exist, panels are
    arranged in columns.

    Args:
        results: Dictionary from test_opsin_expression_levels().
            Structure: results[config_name][target][expr_level] -> analysis dict
        save_path: Optional directory path to save figure (PDF + PNG).
        error_type: 'std' for standard deviation, 'sem' for standard error,
            or 'none' to suppress error bands.
        figsize: Override default figure size (width, height) in inches.
        show_legend: Whether to display a legend on the first panel.
        x_as_percent: If True, multiply expression levels by 100 for the
            x-axis (matching "Opsin expression (%)" label).

    Returns:
        matplotlib Figure object.
    """
    # --- extract structure ---
    config_names = list(results.keys())
    targets = list(results[config_names[0]].keys())
    expression_levels = sorted(results[config_names[0]][targets[0]].keys())

    n_configs = len(config_names)
    n_targets = len(targets)

    # --- colours matching the reference figure ---
    pop_colors = {
        'pv': '#CC00CC',   # magenta
        'sst': '#00AA44',  # green
        'gc': '#DDBB00',   # gold / yellow
        'mc': '#AA0000',   # dark red
    }
    pop_markers = {'pv': 's', 'sst': '^', 'gc': 'o', 'mc': 'D'}
    pop_labels = {'pv': 'PV', 'sst': 'SST', 'gc': 'GC', 'mc': 'MC'}

    config_labels = {
        'full_network': 'Full Network',
        'blocked_exc_to_int': 'Block Exc$\\to$Int',
        'blocked_int_int': 'Block Int-Int',
        'blocked_recurrent': 'Block Recurrent',
    }

    # --- populations to plot per target ---
    #   Include both non-target pops and non-expressing target cells.
    pop_order = {
        'pv': ['pv_nonexpr', 'sst', 'gc', 'mc'],
        'sst': ['pv', 'sst_nonexpr', 'gc', 'mc'],
    }

    # --- layout ---
    if figsize is None:
        figsize = (4.0 * n_configs + 0.5, 3.5 * n_targets + 0.5)
    fig, axes = plt.subplots(
        n_targets, n_configs, figsize=figsize,
        squeeze=False, sharex=True, sharey=True,
    )

    x_vals = np.array(expression_levels)
    if x_as_percent:
        x_vals = x_vals * 100.0

    for row_idx, target in enumerate(targets):
        populations = pop_order.get(target, ['gc', 'mc', 'pv', 'sst'])

        for col_idx, config_name in enumerate(config_names):
            ax = axes[row_idx, col_idx]

            for pop in populations:
                # --- resolve data keys ---
                is_nonexpr = pop.endswith('_nonexpr')
                if is_nonexpr:
                    base_pop = pop.replace('_nonexpr', '')
                    excited_key = f'{base_pop}_nonexpr_excited'
                    std_key = f'{base_pop}_nonexpr_excited_std'
                    sem_key = f'{base_pop}_nonexpr_excited_sem'
                    display_pop = base_pop  # colour / label lookup
                else:
                    excited_key = f'{pop}_excited'
                    std_key = f'{pop}_excited_std'
                    sem_key = f'{pop}_excited_sem'
                    display_pop = pop

                # --- collect values across expression levels ---
                y_vals = []
                y_errs = []
                for expr_level in expression_levels:
                    data = results[config_name][target].get(expr_level, {})
                    y_vals.append(data.get(excited_key, 0.0) * 100.0)

                    if error_type == 'sem' and sem_key in data:
                        y_errs.append(data[sem_key] * 100.0)
                    elif error_type == 'std' and std_key in data:
                        y_errs.append(data[std_key] * 100.0)
                    else:
                        y_errs.append(0.0)

                y_vals = np.array(y_vals)
                y_errs = np.array(y_errs)

                color = pop_colors.get(display_pop, 'gray')
                marker = pop_markers.get(display_pop, 'o')
                label_suffix = ' (non-expr)' if is_nonexpr else ''
                label = pop_labels.get(display_pop, display_pop.upper()) + label_suffix

                # --- plot line + optional error band ---
                ax.plot(
                    x_vals, y_vals,
                    color=color, marker=marker, markersize=5,
                    linewidth=1.8, label=label, alpha=0.9,
                )
                if error_type != 'none' and np.any(y_errs > 0):
                    ax.fill_between(
                        x_vals, y_vals - y_errs, y_vals + y_errs,
                        color=color, alpha=0.15,
                    )

            # --- formatting ---
            ax.set_xlabel(r'Opsin expression (%)', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(r'Fraction excited (%)', fontsize=10)

            config_title = config_labels.get(config_name, config_name)
            ax.set_title(
                f'{target.upper()} stim — {config_title}',
                fontsize=10, fontweight='bold',
            )
            #ax.grid(True, alpha=0.2)
            ax.set_axisbelow(True)

            if show_legend and row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='best', framealpha=0.8)

    fig.suptitle(
        'Fraction Excited vs. Opsin Expression Level',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            save_dir / 'expression_level_overlay.pdf',
            dpi=300, bbox_inches='tight',
        )
        fig.savefig(
            save_dir / 'expression_level_overlay.png',
            dpi=300, bbox_inches='tight',
        )
        logger.info(f"Saved expression level overlay plots to {save_dir}")

    plt.show()
    return fig


def plot_expression_level_results(
    results: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot how paradoxical excitation varies with opsin expression level
    
    Creates plots showing:
    - Expression level dependence for full network
    - Comparison with ablation conditions (if available)
    - Separate panels for PV and SST stimulation
    - All populations including non-target interneurons
    
    Args:
        results: Dictionary from test_opsin_expression_levels()
        save_path: Optional directory to save figure
    """
    
    # Extract configuration names and expression levels
    config_names = list(results.keys())
    targets = list(results[config_names[0]].keys())
    
    # Get expression levels from first config/target
    expression_levels = sorted(results[config_names[0]][targets[0]].keys())
    
    # Create figure - 2 rows x 4 columns
    n_configs = len(config_names)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # Define colors and markers
    colors = {
        'full_network': '#2ecc71',
        'blocked_exc_to_int': '#e67e22',
        'blocked_int_int': '#e74c3c',
        'blocked_recurrent': '#3498db'
    }
    markers = {
        'full_network': 'o',
        'blocked_exc_to_int': 's',
        'blocked_int_int': '^',
        'blocked_recurrent': 'D'
    }
    labels = {
        'full_network': 'Full Network',
        'blocked_exc_to_int': 'Block Exc->Int',
        'blocked_int_int': 'Block Int-Int',
        'blocked_recurrent': 'Block Recurrent'
    }
    
    # Define populations to plot for each target
    pop_map = {
        'pv': ['gc', 'mc', 'pv_nonexpr', 'sst'],
        'sst': ['gc', 'mc', 'pv', 'sst_nonexpr']
    }
    
    for target_idx, target in enumerate(targets):
        populations = pop_map[target]
        
        for pop_idx, pop in enumerate(populations):
            ax = axes[target_idx, pop_idx]
            
            for config_name in config_names:
                excited_fractions = []
                excited_errors = []
                
                for expr_level in expression_levels:
                    data = results[config_name][target][expr_level]
                    
                    # Handle non-expressing target cells
                    if pop.endswith('_nonexpr'):
                        base_pop = pop.replace('_nonexpr', '')
                        excited_fractions.append(data.get(f'{base_pop}_nonexpr_excited', 0.0) * 100)
                        
                        if f'{base_pop}_nonexpr_excited_std' in data:
                            excited_errors.append(data[f'{base_pop}_nonexpr_excited_std'] * 100)
                        else:
                            excited_errors.append(0)
                    else:
                        # Normal populations
                        excited_fractions.append(data[f'{pop}_excited'] * 100)
                        
                        if f'{pop}_excited_std' in data:
                            excited_errors.append(data[f'{pop}_excited_std'] * 100)
                        else:
                            excited_errors.append(0)
                
                # Plot with error bars
                ax.errorbar(expression_levels, excited_fractions,
                           yerr=excited_errors if any(excited_errors) else None,
                           marker=markers.get(config_name, 'o'),
                           color=colors.get(config_name, 'gray'),
                           label=labels.get(config_name, config_name),
                           linewidth=2, markersize=8, capsize=5,
                           alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Mean Opsin Expression Level', fontsize=10)
            ax.set_ylabel('% Cells Paradoxically Excited', fontsize=10)
            
            if pop.endswith('_nonexpr'):
                base_pop = pop.replace('_nonexpr', '')
                ax.set_title(f'{target.upper()} Stim -> {target.upper()} (non-expr)',
                            fontsize=11, fontweight='bold', color='purple')
            elif pop in ['pv', 'sst']:
                ax.set_title(f'{target.upper()} Stim -> {pop.upper()} (non-target IN)',
                            fontsize=11, fontweight='bold', color='darkred')
            else:
                ax.set_title(f'{target.upper()} Stim -> {pop.upper()}',
                            fontsize=11, fontweight='bold')
            
            #ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(fontsize=10, loc='best')
            
            # Set x-axis limits
            ax.set_xlim(min(expression_levels) - 0.05, max(expression_levels) + 0.05)
    
    # Overall title
    fig.suptitle('Opsin Expression Level Dependence',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / 'expression_level_results.pdf',
                   dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'expression_level_results.png',
                   dpi=300, bbox_inches='tight')
        logger.info(f"\nSaved expression level plots to {save_dir}")
    
    plt.show()

def export_expression_results_csv(
    results: Dict,
    csv_path: str,
    target_populations: Optional[List[str]] = None,
) -> None:
    """
    Export opsin expression level experiment results to CSV.

    Produces one row per (config, target, expression_level, connectivity_idx)
    in wide format, with all population metrics as columns.  When per-connectivity
    data is available (paired mode with ``_by_conn`` keys), each connectivity
    instance is emitted as a separate row.  For unpaired / aggregate-only data,
    a single row is written with ``connectivity_idx`` set to NaN.

    The CSV is designed for downstream analysis in R (e.g. paired mixed-effects
    models across expression levels on identical circuit realizations).

    Args:
        results: Dictionary returned by ``test_opsin_expression_levels()``.
            Structure: ``results[config_name][target][expr_level] -> analysis_dict``
        csv_path: Output file path for the CSV.
        target_populations: Subset of targets to export (default: all in results).

    CSV columns (wide format):
        config, target, expression_level, connectivity_idx,
        {pop}_excited, {pop}_mean_change   (for each non-target population),
        {target}_nonexpr_excited, {target}_nonexpr_mean_change,
        {target}_nonexpr_count,
        n_connectivity
    """
    import csv

    config_names = list(results.keys())
    if not config_names:
        logger.warning("export_expression_results_csv: empty results dict")
        return

    # Determine target populations
    if target_populations is None:
        target_populations = sorted(results[config_names[0]].keys())

    # Canonical population order for column output
    all_pops = ['gc', 'mc', 'pv', 'sst']

    # --- collect rows lazily via a generator -----------------------------------
    def _generate_rows():
        for config_name in config_names:
            for target in target_populations:
                if target not in results[config_name]:
                    continue

                expression_levels = sorted(results[config_name][target].keys())

                for expr_level in expression_levels:
                    data = results[config_name][target][expr_level]

                    # Determine non-target populations for this target
                    non_target_pops = [p for p in all_pops if p != target]

                    # Check for paired (per-connectivity) data
                    first_by_conn_key = next(
                        (k for k in data if k.endswith('_by_conn')), None
                    )
                    n_conn = data.get('n_connectivity', None)

                    if first_by_conn_key is not None and n_conn is not None:
                        # --- Paired mode: one row per connectivity instance ---
                        for conn_idx in range(n_conn):
                            row = {
                                'config': config_name,
                                'target': target,
                                'expression_level': expr_level,
                                'connectivity_idx': conn_idx,
                                'n_connectivity': n_conn,
                            }

                            # Non-target population metrics
                            for pop in non_target_pops:
                                exc_key = f'{pop}_excited_by_conn'
                                chg_key = f'{pop}_mean_change_by_conn'

                                row[f'{pop}_excited'] = (
                                    data[exc_key][conn_idx]
                                    if exc_key in data
                                    and conn_idx < len(data[exc_key])
                                    else None
                                )
                                row[f'{pop}_mean_change'] = (
                                    data[chg_key][conn_idx]
                                    if chg_key in data
                                    and conn_idx < len(data[chg_key])
                                    else None
                                )

                            # Non-expressing target cells
                            ne_exc_key = f'{target}_nonexpr_excited_by_conn'
                            ne_chg_key = f'{target}_nonexpr_mean_change_by_conn'
                            ne_cnt_key = f'{target}_nonexpr_count_by_conn'

                            row[f'{target}_nonexpr_excited'] = (
                                data[ne_exc_key][conn_idx]
                                if ne_exc_key in data
                                and conn_idx < len(data[ne_exc_key])
                                else None
                            )
                            row[f'{target}_nonexpr_mean_change'] = (
                                data[ne_chg_key][conn_idx]
                                if ne_chg_key in data
                                and conn_idx < len(data[ne_chg_key])
                                else None
                            )
                            row[f'{target}_nonexpr_count'] = (
                                data[ne_cnt_key][conn_idx]
                                if ne_cnt_key in data
                                and conn_idx < len(data[ne_cnt_key])
                                else data.get(f'{target}_nonexpr_count', None)
                            )

                            yield row
                    else:
                        # --- Unpaired / aggregate mode: single row -----------
                        row = {
                            'config': config_name,
                            'target': target,
                            'expression_level': expr_level,
                            'connectivity_idx': float('nan'),
                            'n_connectivity': n_conn if n_conn else 0,
                        }

                        for pop in non_target_pops:
                            row[f'{pop}_excited'] = data.get(
                                f'{pop}_excited', None
                            )
                            row[f'{pop}_mean_change'] = data.get(
                                f'{pop}_mean_change', None
                            )

                        row[f'{target}_nonexpr_excited'] = data.get(
                            f'{target}_nonexpr_excited', None
                        )
                        row[f'{target}_nonexpr_mean_change'] = data.get(
                            f'{target}_nonexpr_mean_change', None
                        )
                        row[f'{target}_nonexpr_count'] = data.get(
                            f'{target}_nonexpr_count', None
                        )

                        yield row

    # --- build fieldnames in deterministic order --------------------------------
    # Fixed identifier columns
    fieldnames = [
        'config',
        'target',
        'expression_level',
        'connectivity_idx',
    ]

    # Population metric columns in canonical order
    for pop in all_pops:
        fieldnames.append(f'{pop}_excited')
        fieldnames.append(f'{pop}_mean_change')

    # Non-expressing target columns (one set per possible target)
    for target in target_populations:
        fieldnames.append(f'{target}_nonexpr_excited')
        fieldnames.append(f'{target}_nonexpr_mean_change')
        fieldnames.append(f'{target}_nonexpr_count')

    fieldnames.append('n_connectivity')

    # Deduplicate while preserving order
    seen = set()
    unique_fieldnames = []
    for f in fieldnames:
        if f not in seen:
            seen.add(f)
            unique_fieldnames.append(f)
    fieldnames = unique_fieldnames

    # --- write CSV ---------------------------------------------------------------
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction='ignore'
        )
        writer.writeheader()
        for row in _generate_rows():
            writer.writerow(row)
            n_rows += 1

    logger.info(f"Exported expression level results to: {csv_path}")
    logger.info(f"  Rows: {n_rows}")
    logger.info(f"  Configs: {config_names}")
    logger.info(f"  Targets: {target_populations}")

    
# ============================================================================
# Subcommand Handlers
# ============================================================================

def cmd_plot_comparative(args):
    """
    Handle comparative experiment plotting
    
    Plots PV vs SST stimulation results including:
    - Firing rate modulation
    - Network effects (excited/inhibited fractions)
    - Rate changes
    - Response correlations
    """
    logger.info("="*80)
    logger.info("Plotting comparative experiment results")
    logger.info("="*80)
    
    # Load and validate results
    try:
        data = load_results_with_validation(args.input, 'comparative')
        results, conn_analysis, cond_analysis, metadata = data
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Determine intensities to plot
    available_intensities = set()
    for target in results.keys():
        available_intensities.update(results[target].keys())
    
    try:
        intensities = validate_intensities(args.intensities, list(available_intensities))
    except Exception as e:
        logger.error(f"Invalid intensities: {e}")
        return 1
    
    # Generate plots for each intensity
    logger.info(f"\nGenerating plots for intensities: {intensities}")
    
    for intensity in intensities:
        logger.info(f"\n  Plotting intensity {intensity}...")
        
        try:
            plot_comparative_experiment_results(
                results, 
                conn_analysis,
                stimulation_level=intensity,
                metadata=metadata,
                save_path=str(output_dir)
            )
            logger.info(f"    Saved to {output_dir}")
        except Exception as e:
            logger.error(f"    Failed to plot intensity {intensity}: {e}")
            continue
    
    return 0


def cmd_plot_ablations(args):
    """
    Handle ablation test plotting
    
    Plots results from mechanistic ablation tests including:
    - Interneuron interaction blocking
    - Excitation to interneurons blocking
    - Recurrent excitation blocking
    - Intrinsic excitation blocking
    """
    logger.info("="*80)
    logger.info("Plotting ablation test results")
    logger.info("="*80)
    
    # Load and validate results
    try:
        ablation_results = load_results_with_validation(args.input, 'ablation')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Determine available intensities
    # Ablation results have structure: ablation_results[test_type][network_type][target][intensity]
    available_intensities = set()
    for test_type in ablation_results.values():
        if isinstance(test_type, dict):
            for network_type in test_type.values():
                if isinstance(network_type, dict):
                    for target_data in network_type.values():
                        if isinstance(target_data, dict):
                            available_intensities.update(target_data.keys())
    
    # Use specified intensity or default to first available
    if args.intensity is not None:
        intensity = args.intensity
        if intensity not in available_intensities:
            logger.warning(f"Requested intensity {intensity} not found in results")
            logger.warning(f"Available intensities: {sorted(available_intensities)}")
            logger.info(f"Using {sorted(available_intensities)[0]} instead")
            intensity = sorted(available_intensities)[0]
    else:
        intensity = sorted(available_intensities)[0] if available_intensities else 1.0
        logger.info(f"Using intensity {intensity}")
    
    # Generate plot
    logger.info(f"\nGenerating ablation plot for intensity {intensity}...")
    
    try:
        plot_ablation_results_violin(
            ablation_results,
            intensity=intensity,
            save_path=str(output_dir)
        )
        logger.info(f"Saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to plot ablation results: {e}")
        return 1
    logger.info(f"\nGenerating ablation plot for intensity {intensity}...")

    if args.export_csv:
        try:
            export_ablation_results_csv(
                ablation_results,
                output_dir=str(output_dir),
                intensities=[intensity]
            )
        except Exception as e:
            logger.error(f"Failed to export ablation results: {e}")
            return 1
    
    return 0


def cmd_plot_weights(args):
    """
    Handle synaptic weight distribution plotting
    
    Plots weight distributions for each post-synaptic population including:
    - Histogram distributions
    - Violin plots by response type
    - Heatmaps by cell
    - Correlation matrices
    """
    logger.info("="*80)
    logger.info("Plotting synaptic weight distributions")
    logger.info("="*80)
    
    # Load and validate results
    try:
        data = load_results_with_validation(args.input, 'comparative')
        results, conn_analysis, cond_analysis, metadata = data
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Plot weight distributions
    logger.info("\nGenerating weight distribution plots...")
    logger.info("(This requires circuit reconstruction and may take a moment)")
    
    try:
        plot_synaptic_weights_from_results(
            results=results,
            metadata=metadata,
            output_path=output_dir,
            optimization_json_file=args.optimization_file,
            device=None  # Auto-detect
        )
        logger.info(f"Saved to {output_dir / 'synaptic_weights'}")
    except Exception as e:
        logger.error(f"Failed to plot weight distributions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


def cmd_plot_expression(args):
    """
    Handle expression level plotting
    
    Plots how paradoxical excitation varies with opsin expression level
    """
    logger.info("="*80)
    logger.info("Plotting expression level dependence")
    logger.info("="*80)
    
    # Load and validate results
    try:
        expression_results = load_results_with_validation(args.input, 'expression')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Generate plot
    logger.info("\nGenerating expression level plots...")
    
    try:
        plot_expression_level_overlay(
            expression_results,
            save_path=str(output_dir)
        )
        logger.info(f"Saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to plot expression results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    if args.export_csv:
        try:
            export_expression_level_csv(
                expression_results,
                output_dir=str(output_dir),
                intensities=[intensity]
            )
        except Exception as e:
            logger.error(f"Failed to export expression level results: {e}")
            return 1
    
    return 0


def cmd_plot_failure(args):
    """
    Handle failure rate plotting
    
    Plots how paradoxical excitation varies with opsin failure rate,
    showing effective expression (accounting for failures) on x-axis
    """
    logger.info("="*80)
    logger.info("Plotting failure rate dependence")
    logger.info("="*80)
    
    # Load and validate results
    try:
        failure_results = load_results_with_validation(args.input, 'expression')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Generate plot
    logger.info("\nGenerating failure rate plots...")
    
    try:
        plot_failure_rate_overlay(
            failure_results,
            save_path=str(output_dir)
        )
        logger.info(f"Saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to plot failure rate results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

def cmd_plot_combined(args):
    """
    Handle combined ablation + expression plotting
    
    Creates figure combining ablation and expression results
    """
    logger.info("="*80)
    logger.info("Generating combined ablation + expression figure")
    logger.info("="*80)
    
    # Load comparative results
    logger.info("\nLoading comparative results...")
    try:
        comp_data = load_results_with_validation(args.comparative, 'comparative')
        results, conn_analysis, cond_analysis, metadata = comp_data
    except Exception as e:
        logger.error(f"Failed to load comparative results: {e}")
        return 1
    
    # Load ablation results
    logger.info("Loading ablation results...")
    try:
        ablation_results = load_results_with_validation(args.ablation, 'ablation')
    except Exception as e:
        logger.error(f"Failed to load ablation results: {e}")
        return 1
    
    # Load expression results
    logger.info("Loading expression results...")
    try:
        expression_results = load_results_with_validation(args.expression, 'expression')
    except Exception as e:
        logger.error(f"Failed to load expression results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Generate combined plot
    logger.info(f"\nGenerating combined figure for intensity {args.intensity}...")
    
    try:
        plot_combined_ablation_and_expression(
            ablation_results,
            expression_results,
            intensity=args.intensity,
            save_path=str(output_dir)
        )
        logger.info(f"Saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate combined figure: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    logger.info("\n" + "="*80)
    logger.info("Combined figure generation complete")
    logger.info("="*80)
    
    return 0


def cmd_plot_currents(args):
    """
    Handle synaptic current plotting
    
    Plots recorded synaptic currents (requires experiments run with --record-currents)
    """
    logger.info("="*80)
    logger.info("Plotting synaptic currents")
    logger.info("="*80)
    
    # Load and validate results
    try:
        data = load_results_with_validation(args.input, 'comparative')
        results, conn_analysis, cond_analysis, metadata = data
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Check if results contain current recordings
    has_currents = False
    for target in results.keys():
        for intensity, exp_data in results[target].items():
            if 'current_analysis' in exp_data or 'recorded_currents' in exp_data:
                has_currents = True
                break
        if has_currents:
            break
    
    if not has_currents:
        logger.error("No current recording data found in results")
        logger.error("Note: Experiments must be run with --record-currents flag")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Generate current plots
    logger.info("\nGenerating current plots...")
    logger.info("(This requires circuit reconstruction and may take a moment)")
    
    try:
        plot_currents_from_results(
            results=results,
            metadata=metadata,
            output_path=output_dir,
            optimization_json_file=args.optimization_file,
            device=None  # Auto-detect
        )
        logger.info(f"Saved to {output_dir / 'synaptic_currents'}")
    except Exception as e:
        logger.error(f"Failed to plot currents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


def cmd_analyze_weights_by_response(args):
    """
    Handle weight-response relationship analysis
    
    Performs statistical analysis comparing synaptic weights between
    excited and suppressed cells
    """
    logger.info("="*80)
    logger.info("Analyzing synaptic weights by response type")
    logger.info("="*80)
    
    # Load and validate results
    try:
        data = load_results_with_validation(args.input, 'comparative')
        results, conn_analysis, cond_analysis, metadata = data
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Analyze weights by response
    logger.info(f"\nAnalyzing weights by response (n_bootstrap={args.n_bootstrap})...")
    logger.info("(This requires circuit reconstruction and may take several minutes)")
    
    try:
        analyze_synaptic_weights_by_response_from_results(
            results=results,
            metadata=metadata,
            output_path=output_dir,
            optimization_json_file=args.optimization_file
        )
        logger.info(f"Saved to {output_dir / 'weights_by_response'}")
    except Exception as e:
        logger.error(f"Failed to analyze weights: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


def cmd_plot_variance(args):
    """
    Handle variance decomposition plotting
    
    Plots variance decomposition from nested experiments
    """
    logger.info("="*80)
    logger.info("Plotting variance decomposition")
    logger.info("="*80)
    
    # Load and validate results
    try:
        nested_data = load_results_with_validation(args.input, 'nested')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Check for required components
    if 'variance_analysis' not in nested_data:
        logger.error("Variance analysis not found in nested results")
        logger.error("Note: Run nested experiments with variance decomposition enabled")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Generate variance decomposition plot
    logger.info("\nGenerating variance decomposition plot...")
    
    try:
        plot_variance_decomposition(
            nested_data['variance_analysis'],
            nested_data.get('regime_classification'),
            save_path=str(output_dir / 'variance_decomposition.pdf')
        )
        logger.info(f"Saved to {output_dir / 'variance_decomposition.pdf'}")
    except Exception as e:
        logger.error(f"Failed to plot variance decomposition: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Generate detailed plots if requested
    if args.detailed:
        logger.info("\nGenerating detailed connectivity variance plots...")
        
        # Get highest intensity
        metadata = nested_data.get('metadata', {})
        intensities = metadata.get('intensities', [1.0])
        intensity = max(intensities)
        
        try:
            plot_connectivity_instance_variance(
                nested_data['aggregated_results'],
                intensity=intensity,
                save_path=str(output_dir / 'connectivity_instance_variance.pdf')
            )
            logger.info(f"Saved to {output_dir / 'connectivity_instance_variance.pdf'}")
            
            plot_connectivity_instance_variance_detailed(
                nested_data,
                intensity=intensity,
                warmup=metadata.get('warmup', 500.0),
                stim_start=metadata.get('stim_start', 1500.0),
                stim_duration=metadata.get('stim_duration', 1000.0),
                save_path=str(output_dir / 'connectivity_instance_variance_detailed.pdf')
            )
            logger.info(f"Saved to {output_dir / 'connectivity_instance_variance_detailed.pdf'}")
        except Exception as e:
            logger.error(f"Failed to plot detailed variance: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Print summary
    print_nested_experiment_summary(nested_data)
    
    return 0


def cmd_bootstrap_analysis(args):
    """
    Handle bootstrap effect size analysis
    
    Performs bootstrap analysis on nested experiment results
    """
    logger.info("="*80)
    logger.info("Bootstrap effect size analysis")
    logger.info("="*80)
    
    # Load and validate results
    try:
        nested_data = load_results_with_validation(args.input, 'nested')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Run bootstrap analysis
    logger.info(f"\nRunning bootstrap analysis (n={args.n_samples})...")
    logger.info("(This may take several minutes)")
    
    try:
        bootstrap_results = run_nested_effect_size_analysis(
            nested_data=nested_data,
            target_populations=['pv', 'sst'],
            post_populations=['gc', 'mc', 'pv', 'sst'],
            intensities=args.intensities,
            source_populations=['pv', 'sst', 'mc', 'mec'],
            n_bootstrap=args.n_samples,
            threshold_std=args.threshold_std,
            expression_threshold=args.expression_threshold,
            random_seed=args.seed,
            output_dir=str(output_dir),
            device=None  # Auto-detect
        )
        logger.info(f"Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to run bootstrap analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


def cmd_nested_weights_analysis(args):
    """
    Handle nested weights analysis command (supports HDF5 and pickle formats)
    
    Includes optional distributional analysis using:
    - Geometric mean ratios
    - Mann-Whitney U / CLES
    - Quantile profiles
    """
    logger.info("="*80)
    logger.info("Nested Weights Analysis")
    logger.info("="*80)
    
    # Load and validate results
    try:
        nested_data = load_results_with_validation(args.input, 'nested')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Extract metadata
    metadata = nested_data.get('metadata', {})
    file_format = nested_data.get('file_format', 'pickle')
    
    # Extract experiment parameters
    stim_start = metadata['stim_start']
    stim_duration = metadata['stim_duration']
    warmup = metadata['warmup']
    
    # Determine intensities and populations to analyze
    target_populations = args.target_populations or ['pv', 'sst']
    post_populations = args.post_populations or ['gc', 'mc', 'pv', 'sst']
    source_populations = args.source_populations or ['gc', 'mc', 'pv', 'sst', 'mec']
    
    if args.intensities is None:
        intensities = metadata['intensities']
    else:
        intensities = args.intensities
    
    logger.info(f"\nFile format: {file_format.upper()}")
    logger.info(f"Analysis parameters:")
    logger.info(f"  Target populations: {target_populations}")
    logger.info(f"  Post populations: {post_populations}")
    logger.info(f"  Source populations: {source_populations}")
    logger.info(f"  Intensities: {intensities}")
    logger.info(f"  Bootstrap samples: {args.n_bootstrap}")
    logger.info(f"  Classification threshold: {args.threshold_std} std")
    if args.run_distributional:
        logger.info(f"  Distributional analysis: ENABLED")
        logger.info(f"    Quantiles: {args.distribution_quantiles}")
    
    
    device = get_default_device()
    
    # Storage for all analyses
    all_analyses = {}
    all_distributional_analyses = {} if args.run_distributional else None
    
    # Handle HDF5 vs pickle differently
    if file_format == 'hdf5':
        
        hdf5_file = nested_data['hdf5_file']
        n_connectivity = metadata['n_connectivity_instances']
        n_mec_patterns = metadata['n_mec_patterns_per_connectivity']
        
        logger.info(f"\nLoading trials from HDF5: {hdf5_file}")
        
        with h5py.File(hdf5_file, 'r') as f:
            for target in target_populations:
                if target not in f:
                    logger.warning(f"Skipping {target.upper()} - no data in HDF5 file")
                    continue
                
                all_analyses[target] = {}
                if args.run_distributional:
                    all_distributional_analyses[target] = {}
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Analyzing {target.upper()} stimulation")
                logger.info('='*60)
                
                for intensity in intensities:
                    logger.info(f"\n  Intensity: {intensity}")
                    
                    # Load trials for this condition from HDF5
                    trials = load_nested_trials_from_hdf5(
                        f, target, intensity, n_connectivity, n_mec_patterns,
                        require_full_activity=True
                    )
                    
                    if len(trials) == 0:
                        logger.warning(f"  Skipping intensity {intensity} - no trials")
                        continue
                    
                    logger.info(f"    Loaded {len(trials)} trials from HDF5")
                    logger.info(f"    Connectivity instances: {len(set(t.connectivity_idx for t in trials))}")
                    
                    # Recreate circuit (use seed from first connectivity instance)
                    seed_structure = metadata.get('seed_structure', {})
                    connectivity_seeds = seed_structure.get('connectivity_seeds', [42])
                    first_conn_seed = connectivity_seeds[0]
                    
                    logger.info(f"    Recreating circuit with seed: {first_conn_seed}")
                    
                    circuit_params = CircuitParams()
                    synaptic_params = PerConnectionSynapticParams()
                    opsin_params = OpsinParams()
                    
                    optimization_file = metadata.get('optimization_file')
                    if optimization_file:
                        logger.info(f"    Applying optimization from: {optimization_file}")
                    
                    # Create circuit
                    set_random_seed(first_conn_seed, device)
                    experiment = OptogeneticExperiment(
                        circuit_params,
                        synaptic_params,
                        opsin_params,
                        optimization_json_file=optimization_file,
                        device=device,
                        base_seed=first_conn_seed
                    )
                    
                    # Analyze each post-synaptic population
                    for post_pop in post_populations:
                        logger.info(f"\n    Analyzing {target.upper()} -> {post_pop.upper()}")
                        
                        try:
                            # Mean-based analysis
                            analysis_results = analyze_weights_by_average_response_nested(
                                nested_results=trials,
                                circuit=experiment.circuit,
                                target_population=target,
                                post_population=post_pop,
                                source_populations=source_populations,
                                stim_start=stim_start,
                                stim_duration=stim_duration,
                                warmup=warmup,
                                threshold_std=args.threshold_std,
                                expression_threshold=args.expression_threshold,
                                n_bootstrap=args.n_bootstrap,
                                run_pca=args.run_pca,
                                n_pca_permutations=args.n_pca_permutations,
                                random_seed=args.seed
                            )
                            
                            # Store mean-based results
                            if target not in all_analyses:
                                all_analyses[target] = {}
                            if intensity not in all_analyses[target]:
                                all_analyses[target][intensity] = {}
                            all_analyses[target][intensity][post_pop] = analysis_results
                            
                            # Distributional analysis (if enabled)
                            if args.run_distributional:
                                logger.info(f"      Running distributional analysis...")
                                
                                dist_results = analyze_weights_distributional_nested(
                                    nested_results=trials,
                                    circuit=experiment.circuit,
                                    target_population=target,
                                    post_population=post_pop,
                                    source_populations=source_populations,
                                    stim_start=stim_start,
                                    stim_duration=stim_duration,
                                    warmup=warmup,
                                    threshold_std=args.threshold_std,
                                    expression_threshold=args.expression_threshold,
                                    n_bootstrap=args.n_bootstrap,
                                    quantiles=args.distribution_quantiles,
                                    random_seed=args.seed,
                                    export_csv_path=args.weight_distribution_csv
                                )
                                
                                # Store distributional results
                                if target not in all_distributional_analyses:
                                    all_distributional_analyses[target] = {}
                                if intensity not in all_distributional_analyses[target]:
                                    all_distributional_analyses[target][intensity] = {}
                                all_distributional_analyses[target][intensity][post_pop] = dist_results
                            
                            # Generate visualizations
                            if args.plot:
                                vis_dir = output_dir / f"{target}_intensity_{intensity}"
                                vis_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Mean-based plot
                                logger.info(f"      Generating mean-based plot...")
                                fig = plot_weights_by_average_response_nested(
                                    analysis_results,
                                    save_path=str(vis_dir / f'nested_weights_{post_pop}.pdf')
                                )
                                plt.close(fig)
                                
                                # Distributional plot (if enabled)
                                if args.run_distributional:
                                    logger.info(f"      Generating distributional plot...")
                                    fig_dist = plot_distributional_analysis_nested_opsin_aware(
                                        dist_results,
                                        save_path=str(vis_dir / f'nested_weights_{post_pop}_distributional.pdf')
                                    )
                                    plt.close(fig_dist)


                                # Generate detailed connectivity comparison if requested
                                if args.detailed:
                                    logger.info(f"      Generating detailed connectivity comparison...")
                                    n_conn = analysis_results['metadata']['n_connectivity_instances']
                                    conn_indices = list(range(min(3, n_conn)))
                                    fig = plot_connectivity_weight_comparison(
                                        analysis_results,
                                        connectivity_indices=conn_indices,
                                        save_path=str(vis_dir / f'nested_weights_{post_pop}_detailed.pdf')
                                    )
                                    plt.close(fig)
                        
                        except Exception as e:
                            logger.error(f"      Failed to analyze {post_pop.upper()}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            continue
                    
                    # Generate summary forest plot across all post-populations
                    if args.plot and intensity in all_analyses.get(target, {}):
                        analysis_results_by_post = all_analyses[target][intensity]
                        if len(analysis_results_by_post) > 0:
                            vis_dir = output_dir / f"{target}_intensity_{intensity}"
                            vis_dir.mkdir(parents=True, exist_ok=True)
                            logger.info(f"\n    Generating summary forest plot across all targets...")
                            try:
                                summary_fig = plot_summary_forest_plot_all_targets(
                                    analysis_results_by_target=analysis_results_by_post,
                                    stimulated_population=target,
                                    save_path=str(vis_dir / f'{target}_all_targets_summary_forest.pdf')
                                )
                                if summary_fig is not None:
                                    plt.close(summary_fig)
                                    logger.info(f"      Saved summary forest plot")
                            except Exception as e:
                                logger.error(f"      Failed to generate summary forest plot: {e}")
                                import traceback
                                logger.error(traceback.format_exc())

                            if args.run_distributional:
                                # Summary of geometric ratio and CLES across all post-populations
                                fig_summary = plot_distributional_violin_grid_across_populations_with_boxplot(
                                    all_distributional_analyses[target][intensity],
                                    save_path=str(vis_dir / f'{target}_distributional_summary_metrics.pdf'),
                                    figsize=(16, 12),
                                    show_stats=False
                                )
                                plt.close(fig_summary)

                                # Summary of quantile profiles across all post-populations
                                fig_quantiles = plot_quantile_summary_across_populations(
                                    all_distributional_analyses[target][intensity],
                                    save_path=str(vis_dir / f'{target}_distributional_summary_quantiles.pdf'),
                                    figsize=(18, 12)
                                )
                                plt.close(fig_quantiles)

                            if args.run_pca:
                                try:
                                    pca_results_by_target = {
                                        target_pop: results['pca_results']
                                        for target_pop, results in analysis_results_by_post.items()
                                        if results.get('pca_results') is not None
                                    }

                                    pca_fig = plot_pca_summary_all_targets(
                                        pca_results_by_target=pca_results_by_target,
                                        stimulated_population=target,
                                        save_path=str(vis_dir / f'{target}_all_targets_summary_pca.pdf')
                                    )
                                    if pca_fig is not None:
                                        plt.close(pca_fig)
                                        logger.info(f"      Saved PCA plot")
                                except Exception as e:
                                    logger.error(f"      Failed to generate summary PCA plot: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                    
                    # Clean up circuit
                    del experiment
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
    
    else:  # Pickle format
        nested_results = nested_data['nested_results']
        seed_structure = nested_data['seed_structure']
        
        # Original pickle-based analysis (same changes as HDF5 version)
        for target in target_populations:
            if target not in nested_results:
                logger.warning(f"Skipping {target.upper()} - no results found")
                continue
            
            all_analyses[target] = {}
            if args.run_distributional:
                all_distributional_analyses[target] = {}
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing {target.upper()} stimulation")
            logger.info('='*60)
            
            for intensity in intensities:
                if intensity not in nested_results[target]:
                    logger.warning(f"  Skipping intensity {intensity} - no results")
                    continue
                
                logger.info(f"\n  Intensity: {intensity}")
                
                trials = nested_results[target][intensity]
                
                logger.info(f"    Total trials: {len(trials)}")
                logger.info(f"    Connectivity instances: {len(set(t.connectivity_idx for t in trials))}")
                
                # Recreate circuit
                first_conn_seed = seed_structure['connectivity_seeds'][0]
                logger.info(f"    Recreating circuit with seed: {first_conn_seed}")
                
                circuit_params = CircuitParams()
                synaptic_params = PerConnectionSynapticParams()
                opsin_params = OpsinParams()
                
                optimization_file = metadata.get('optimization_file')
                if optimization_file:
                    logger.info(f"    Applying optimization from: {optimization_file}")
                
                set_random_seed(first_conn_seed, device)
                experiment = OptogeneticExperiment(
                    circuit_params,
                    synaptic_params,
                    opsin_params,
                    optimization_json_file=optimization_file,
                    device=device,
                    base_seed=first_conn_seed
                )
                
                # Analyze each post-synaptic population
                for post_pop in post_populations:
                    logger.info(f"\n    Analyzing {target.upper()} -> {post_pop.upper()}")
                    
                    try:
                        # Mean-based analysis
                        analysis_results = analyze_weights_by_average_response_nested(
                            nested_results=trials,
                            circuit=experiment.circuit,
                            target_population=target,
                            post_population=post_pop,
                            source_populations=source_populations,
                            stim_start=stim_start,
                            stim_duration=stim_duration,
                            warmup=warmup,
                            threshold_std=args.threshold_std,
                            expression_threshold=args.expression_threshold,
                            n_bootstrap=args.n_bootstrap,
                            run_pca=args.run_pca,
                            n_pca_permutations=args.n_pca_permutations,
                            random_seed=args.seed
                        )
                        
                        if target not in all_analyses:
                            all_analyses[target] = {}
                        if intensity not in all_analyses[target]:
                            all_analyses[target][intensity] = {}
                        all_analyses[target][intensity][post_pop] = analysis_results
                        
                        # Distributional analysis (if enabled)
                        if args.run_distributional:
                            logger.info(f"      Running distributional analysis...")
                            
                            dist_results = analyze_weights_distributional_nested(
                                nested_results=trials,
                                circuit=experiment.circuit,
                                target_population=target,
                                post_population=post_pop,
                                source_populations=source_populations,
                                stim_start=stim_start,
                                stim_duration=stim_duration,
                                warmup=warmup,
                                threshold_std=args.threshold_std,
                                expression_threshold=args.expression_threshold,
                                n_bootstrap=args.n_bootstrap,
                                quantiles=args.distribution_quantiles,
                                random_seed=args.seed,
                                export_csv_path=args.weight_distribution_csv
                            )
                            
                            if target not in all_distributional_analyses:
                                all_distributional_analyses[target] = {}
                            if intensity not in all_distributional_analyses[target]:
                                all_distributional_analyses[target][intensity] = {}
                            all_distributional_analyses[target][intensity][post_pop] = dist_results
                        
                        if args.plot:
                            vis_dir = output_dir / f"{target}_intensity_{intensity}"
                            vis_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Mean-based plot
                            logger.info(f"      Generating mean-based plot...")
                            fig = plot_weights_by_average_response_nested(
                                analysis_results,
                                save_path=str(vis_dir / f'nested_weights_{post_pop}.pdf')
                            )
                            plt.close(fig)
                            
                            # Distributional plot (if enabled)
                            if args.run_distributional:
                                logger.info(f"      Generating distributional plot...")
                                fig_dist = plot_distributional_analysis_nested_opsin_aware(
                                    dist_results,
                                    save_path=str(vis_dir / f'nested_weights_{post_pop}_distributional.pdf')
                                )
                                plt.close(fig_dist)
                            
                            if args.detailed:
                                logger.info(f"      Generating detailed connectivity comparison...")
                                n_conn = analysis_results['metadata']['n_connectivity_instances']
                                conn_indices = list(range(min(3, n_conn)))
                                fig = plot_connectivity_weight_comparison(
                                    analysis_results,
                                    connectivity_indices=conn_indices,
                                    save_path=str(vis_dir / f'nested_weights_{post_pop}_detailed.pdf')
                                )
                                plt.close(fig)
                    
                    except Exception as e:
                        logger.error(f"      Failed to analyze {post_pop.upper()}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                
                # Generate summary forest plot
                if args.plot and intensity in all_analyses.get(target, {}):
                    analysis_results_by_post = all_analyses[target][intensity]
                    if len(analysis_results_by_post) > 0:
                        vis_dir = output_dir / f"{target}_intensity_{intensity}"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"\n    Generating summary forest plot across all targets...")
                        try:
                            summary_fig = plot_summary_forest_plot_all_targets(
                                analysis_results_by_target=analysis_results_by_post,
                                stimulated_population=target,
                                save_path=str(vis_dir / f'{target}_all_targets_summary_forest.pdf')
                            )
                            if summary_fig is not None:
                                plt.close(summary_fig)
                                logger.info(f"      Saved summary forest plot")
                        except Exception as e:
                            logger.error(f"      Failed to generate summary forest plot: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        
                        if args.run_distributional:
                            # Summary of geometric ratio and CLES across all post-populations
                            fig_summary = plot_distributional_violin_grid_across_populations_with_boxplot(
                                all_distributional_analyses[target][intensity],
                                save_path=str(vis_dir / f'{target}_distributional_summary_metrics.pdf'),
                                figsize=(16, 12),
                                show_stats=False
                            )
                            plt.close(fig_summary)

                            # Summary of quantile profiles across all post-populations
                            fig_quantiles = plot_quantile_summary_across_populations(
                                all_distributional_analyses[target][intensity],
                                save_path=str(vis_dir / f'{target}_distributional_summary_quantiles.pdf'),
                                figsize=(18, 12)
                            )
                            plt.close(fig_quantiles)
                                
                        if args.run_pca:
                            try:
                                pca_results_by_target = {
                                    target_pop: results['pca_results']
                                    for target_pop, results in analysis_results_by_post.items()
                                    if results.get('pca_results') is not None
                                }
                                
                                pca_fig = plot_pca_summary_all_targets(
                                    pca_results_by_target=pca_results_by_target,
                                    stimulated_population=target,
                                    save_path=str(vis_dir / f'{target}_all_targets_summary_pca.pdf')
                                )
                                if pca_fig is not None:
                                    plt.close(pca_fig)
                                    logger.info(f"      Saved PCA plot")
                            except Exception as e:
                                logger.error(f"      Failed to generate summary PCA plot: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                
                # Clean up circuit
                del experiment
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    # Save complete analysis results
    if args.save_results:
        analysis_file = output_dir / 'nested_weights_analysis_results.pkl'
        save_dict = {
            'all_analyses': all_analyses,
            'metadata': metadata,
            'parameters': {
                'threshold_std': args.threshold_std,
                'expression_threshold': args.expression_threshold,
                'n_bootstrap': args.n_bootstrap,
                'random_seed': args.seed
            }
        }
        
        if args.run_distributional:
            save_dict['all_distributional_analyses'] = all_distributional_analyses
            save_dict['parameters']['quantiles'] = args.distribution_quantiles
        
        with open(analysis_file, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"\nAnalysis results saved to: {analysis_file}")
        if args.run_distributional:
            logger.info(f"  - Includes distributional analysis results")
    
    logger.info(f"\nNested weights analysis output saved to: {output_dir}")
    
    return 0


def cmd_decision_analysis(args):
    """
    Handle effect size decision framework
    
    Determines if additional data collection is needed based on
    effect sizes, power, and biological significance
    """
    logger.info("="*80)
    logger.info("Effect size decision framework analysis")
    logger.info("="*80)
    
    # Load and validate results
    try:
        nested_data = load_results_with_validation(args.input, 'nested')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Run decision analysis
    logger.info(f"\nRunning decision framework analysis...")
    logger.info(f"  Target power: {args.target_power}")
    logger.info(f"  Max feasible N: {args.max_n}")
    logger.info(f"  Min meaningful effect: {args.min_effect}")
    logger.info(f"  Min meaningful diff: {args.min_diff_nS} nS")
    logger.info("(This may take several minutes)")
    
    try:
        decision_results = run_nested_effect_size_decision_analysis(
            nested_data=nested_data,
            target_populations=['pv', 'sst'],
            post_populations=['gc', 'mc'],
            intensities=args.intensities,
            source_populations=['pv', 'sst', 'mc', 'mec'],
            n_bootstrap=args.n_samples,
            threshold_std=args.threshold_std,
            expression_threshold=args.expression_threshold,
            current_n=None,  # Auto-detect
            target_power=args.target_power,
            max_feasible_n=args.max_n,
            min_meaningful_effect=args.min_effect,
            min_meaningful_diff_nS=args.min_diff_nS,
            random_seed=args.seed,
            output_dir=str(output_dir),
            device=None  # Auto-detect
        )
        logger.info(f"Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to run decision analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    logger.info("Decision analysis saved to {output_dir / 'effect_size_decision_summary.txt'}")
    
    return 0


def cmd_plot_connectivity_activity(args):
    """
    Handle connectivity-specific aggregated activity plotting
    
    Plots the average activity across all trials (different MEC patterns)
    for a specific connectivity instance
    """
    logger.info("="*80)
    logger.info("Plotting Connectivity-Specific Aggregated Activity")
    logger.info("="*80)
    
    # Load and validate results
    try:
        nested_data = load_results_with_validation(args.input, 'nested')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Extract metadata
    metadata = nested_data.get('metadata', {})
    file_format = nested_data.get('file_format', 'pickle')

    # Get device for tensor operations
    device = get_default_device()
    
    # Validate connectivity index
    n_connectivity = metadata.get('n_connectivity_instances', 0)
    if args.connectivity_idx >= n_connectivity:
        logger.error(f"Connectivity index {args.connectivity_idx} out of range "
                    f"(max: {n_connectivity - 1})")
        return 1
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Extract stimulation parameters
    stim_start = metadata['stim_start']
    stim_duration = metadata['stim_duration']
    warmup = metadata.get('warmup', 500.0)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  File format: {file_format.upper()}")
    logger.info(f"  Target population: {args.target_population.upper()}")
    logger.info(f"  Intensity: {args.intensity}")
    logger.info(f"  Connectivity instance: {args.connectivity_idx}")
    logger.info(f"  Total connectivity instances: {n_connectivity}")
    
    # Filter trials for this connectivity
    logger.info(f"\nFiltering trials...")
    
    try:
        trials = filter_trials_for_connectivity(
            nested_data,
            args.target_population,
            args.intensity,
            args.connectivity_idx
        )
    except Exception as e:
        logger.error(f"Failed to filter trials: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    if len(trials) == 0:
        logger.error(f"No trials found for connectivity {args.connectivity_idx}")
        return 1
    
    logger.info(f"  Found {len(trials)} trials for this connectivity")
    logger.info(f"  MEC pattern indices: {sorted(set(t.mec_pattern_idx for t in trials))}")
    
    # Extract trial results and add opsin_expression from nested trial wrapper
    # (In nested experiments, opsin_expression is stored at NestedTrialResult level)
    trial_results = []
    for trial in trials:
        result = trial.results.copy()
        # Add opsin_expression from trial wrapper if not already in results
        if 'opsin_expression' not in result:
            result['opsin_expression'] = trial.opsin_expression
        trial_results.append(result)
    
    trial_results = convert_trial_results_to_torch(trial_results, device=device)
    
    # Reconstruct circuit with the specific connectivity seed
    logger.info(f"\nReconstructing circuit...")

    seed_structure = nested_data.get('seed_structure', {})
    connectivity_seeds = seed_structure.get('connectivity_seeds', [42])
    
    if args.connectivity_idx >= len(connectivity_seeds):
        logger.error(f"Connectivity seed not found for index {args.connectivity_idx}")
        return 1
    
    connectivity_seed = connectivity_seeds[args.connectivity_idx]
    logger.info(f"  Using connectivity seed: {connectivity_seed}")
    
    
    circuit_params = CircuitParams()
    synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()
    
    # Apply optimization if available
    optimization_file = metadata.get('optimization_file')
    if optimization_file:
        logger.info(f"  Applying optimization from: {optimization_file}")
    
    # Create circuit with this specific connectivity
    set_random_seed(connectivity_seed, device)
    
    experiment = OptogeneticExperiment(
        circuit_params,
        synaptic_params,
        opsin_params,
        optimization_json_file=optimization_file,
        device=device,
        base_seed=connectivity_seed
    )
    
    logger.info(f"  Circuit reconstructed successfully")
    
    # Add layout, connectivity, and stimulated indices to trial results
    # (These are needed for visualization but aren't saved in nested experiment results)
    activation_threshold = 1e-2  # Same threshold used in simulate_stimulation
    
    for trial_result in trial_results:
        trial_result['layout'] = experiment.circuit.layout
        trial_result['connectivity'] = experiment.circuit.connectivity
        
        # Reconstruct stimulated/non-stimulated indices from opsin expression
        opsin_expr = trial_result['opsin_expression']
        if isinstance(opsin_expr, torch.Tensor):
            opsin_expr_np = opsin_expr.cpu().numpy()
        else:
            opsin_expr_np = np.array(opsin_expr)
        
        stimulated_mask = opsin_expr_np >= activation_threshold
        trial_result['stimulated_indices'] = np.where(stimulated_mask)[0]
        trial_result['non_stimulated_indices'] = np.where(~stimulated_mask)[0]        

    # Aggregate across MEC patterns using the factored-out function
    logger.info(f"\nAggregating activity across MEC patterns...")
    aggregated_results = aggregate_trial_results(
        trial_results,
        n_trials=len(trial_results)
    )
    logger.info(f"  Aggregated {len(trial_results)} trials")
    
    # Extract opsin expression from aggregated results
    # (Already averaged across trials, no need to regenerate)
    opsin_expression_mean = aggregated_results['opsin_expression_mean']
    if isinstance(opsin_expression_mean, torch.Tensor):
        opsin_expression_array = opsin_expression_mean.cpu().numpy()
    else:
        opsin_expression_array = np.array(opsin_expression_mean)

    # Generate plot
    logger.info(f"\nGenerating aggregated activity plot...")
    
    vis = DGCircuitVisualization(experiment.circuit)
    
    # Create filename suffix
    suffix = f"_connectivity_{args.connectivity_idx}"
    if args.baseline_normalize:
        suffix += "_normalized"
    
    save_path = output_dir / f"DG_{args.target_population}_stimulation_connectivity_{args.connectivity_idx}_intensity_{args.intensity}_raster_aggregated.pdf"
    
    try:
        fig, _ = vis.plot_aggregated_activity(
            aggregated_results=aggregated_results,
            target_population=args.target_population,
            opsin_expression_levels=opsin_expression_array,
            light_intensity=args.intensity,
            stim_start=stim_start,
            warmup=warmup,
            baseline_normalize=args.baseline_normalize,
            sort_by_activity=args.sort_by_activity,
            save_path=str(save_path)
        )
        plt.close(fig)
        
        logger.info(f"  Saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Print summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary Statistics (Connectivity {args.connectivity_idx})")
    logger.info('='*60)
    
    time = aggregated_results['time']
    activity_mean = aggregated_results['activity_trace_mean']
    activity_std = aggregated_results['activity_trace_std']
    
    baseline_mask = (time >= warmup) & (time < stim_start)
    stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
    
    for pop in ['gc', 'mc', 'pv', 'sst']:
        baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
        stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
        
        mean_baseline = torch.mean(baseline_rate).item()
        mean_stim = torch.mean(stim_rate).item()
        mean_change = mean_stim - mean_baseline
        
        logger.info(f"\n{pop.upper()}:")
        logger.info(f"  Baseline: {mean_baseline:.2f} Hz")
        logger.info(f"  Stimulation: {mean_stim:.2f} Hz")
        logger.info(f"  Change: {mean_change:+.2f} Hz")
        
        # Variability across MEC patterns
        if len(trial_results) > 1:
            std_baseline = torch.mean(activity_std[pop][:, baseline_mask]).item()
            std_stim = torch.mean(activity_std[pop][:, stim_mask]).item()
            logger.info(f"  Variability (std across MEC patterns):")
            logger.info(f"    Baseline: {std_baseline:.2f} Hz")
            logger.info(f"    Stimulation: {std_stim:.2f} Hz")
    
    logger.info(f"\n{'='*60}")
    logger.info("Connectivity activity plotting complete")
    logger.info('='*60)
    
    return 0

def cmd_export_nested_csv(args):
    """
    Export per-(connectivity, MEC pattern) aggregate statistics to CSV.

    Each row contains population-level proportions (excited / suppressed /
    unchanged), mean firing-rate change, and log2 modulation ratio.
    For the target population, separate columns are emitted for
    opsin-expressing vs non-expressing cells.
    """
    logger.info("=" * 80)
    logger.info("Exporting nested experiment statistics to CSV")
    logger.info("=" * 80)

    # Load and validate nested results
    try:
        nested_data = load_results_with_validation(args.input, 'nested')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1

    metadata = nested_data.get('metadata', {})
    file_format = nested_data.get('file_format', 'pickle')

    # Resolve stimulation parameters from metadata
    stim_start = metadata.get('stim_start', args.stim_start)
    stim_duration = metadata.get('stim_duration', args.stim_duration)
    warmup = metadata.get('warmup', args.warmup)

    # Resolve output path
    if args.csv_path is not None:
        csv_path = args.csv_path
    else:
        output_dir = setup_output_directory(args.output)
        csv_path = str(output_dir / 'nested_experiment_stats.csv')

    # Resolve target populations and intensities
    target_populations = args.target_populations
    intensities = args.intensities

    logger.info(f"\nFile format: {file_format.upper()}")
    logger.info(f"Stimulation parameters:")
    logger.info(f"  stim_start   = {stim_start} ms")
    logger.info(f"  stim_duration = {stim_duration} ms")
    logger.info(f"  warmup       = {warmup} ms")
    logger.info(f"  threshold_std = {args.threshold_std}")
    logger.info(f"  expression_threshold = {args.expression_threshold}")
    logger.info(f"Output: {csv_path}")

    try:
        if file_format == 'hdf5':
            n_rows = export_nested_experiment_csv_from_hdf5(
                hdf5_filepath=nested_data['hdf5_file'],
                csv_path=csv_path,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                target_populations=target_populations,
                intensities=intensities,
                threshold_std=args.threshold_std,
                expression_threshold=args.expression_threshold,
            )
        else:
            nested_results = nested_data['nested_results']
            if nested_results is None:
                logger.error(
                    "In-memory nested_results is None. "
                    "Use HDF5 format or re-run with save_nested_trials=True."
                )
                return 1

            n_rows = export_nested_experiment_csv(
                nested_results=nested_results,
                csv_path=csv_path,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                target_populations=target_populations,
                intensities=intensities,
                threshold_std=args.threshold_std,
                expression_threshold=args.expression_threshold,
            )

        logger.info(f"\nExported {n_rows} rows to {csv_path}")

    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0



def cmd_export_per_cell_csv(args):
    """
    Handle per-cell weights and firing rates CSV export
    
    Exports detailed per-cell data including:
    - Response classification (excited/suppressed/unchanged)
    - Baseline and stimulation firing rates
    - Incoming synaptic weights from all source populations
    - Opsin expression levels
    """
    logger.info("="*80)
    logger.info("Exporting per-cell weights and firing rates to CSV")
    logger.info("="*80)
    
    # Load and validate results
    try:
        nested_data = load_results_with_validation(args.input, 'nested')
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Extract metadata
    metadata = nested_data.get('metadata', {})
    file_format = nested_data.get('file_format', 'pickle')
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.output)
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}")
        return 1
    
    # Determine CSV path
    csv_path = output_dir / args.csv_filename
    
    # Extract experiment parameters
    stim_start = metadata.get('stim_start', 1500.0)
    stim_duration = metadata.get('stim_duration', 1000.0)
    warmup = metadata.get('warmup', 500.0)
    optimization_file = metadata.get('optimization_file')
    
    # Determine populations to export
    source_populations = args.source_populations or ('gc', 'mc', 'pv', 'sst')
    post_populations = args.post_populations or ('gc', 'mc', 'pv', 'sst')
    target_populations = args.target_populations or ['pv', 'sst']
    intensities = args.intensities or metadata.get('intensities')
    
    logger.info(f"\nExport configuration:")
    logger.info(f"  File format: {file_format.upper()}")
    logger.info(f"  Output CSV: {csv_path}")
    logger.info(f"  Target populations: {target_populations}")
    logger.info(f"  Post populations: {post_populations}")
    logger.info(f"  Source populations: {source_populations}")
    logger.info(f"  Intensities: {intensities}")
    logger.info(f"  Classification threshold: {args.threshold_std} std")
    logger.info(f"  Expression threshold: {args.expression_threshold}")
    
    # Handle HDF5 format (memory-efficient streaming)
    if file_format == 'hdf5':
        hdf5_file = nested_data['hdf5_file']
        
        logger.info(f"\nProcessing HDF5 file: {hdf5_file}")
        logger.info("Using memory-efficient streaming approach...")
        
        try:
            n_rows = export_per_cell_weights_from_hdf5(
                hdf5_filepath=hdf5_file,
                csv_path=str(csv_path),
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                optimization_json_file=optimization_file,
                threshold_std=args.threshold_std,
                expression_threshold=args.expression_threshold,
                source_populations=tuple(source_populations),
                post_populations=tuple(post_populations),
                target_populations=target_populations,
                intensities=intensities,
                device=None,  # Auto-detect
            )
            
            logger.info(f"\nExport complete!")
            logger.info(f"  Total rows: {n_rows}")
            logger.info(f"  Output file: {csv_path}")
            logger.info(f"  File size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
    
    else:  # Pickle format
        logger.error("CSV export for pickle format not yet implemented")
        logger.error("Please convert to HDF5 format first, or use HDF5-based nested experiments")
        return 1
    
    logger.info("\n" + "="*80)
    logger.info("Per-cell CSV export complete")
    logger.info("="*80)
    
    return 0

# ============================================================================
# Argument Parser Setup
# ============================================================================

def create_parser():
    """
    Create argument parser with subcommands
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Analysis of DG optogenetic experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot comparative experiment results
  %(prog)s plot-comparative results.pkl
  %(prog)s plot-comparative results.pkl --intensities 0.5 1.0 1.5
  
  # Plot ablation test results
  %(prog)s plot-ablations ablation_results.pkl
  %(prog)s plot-ablations ablation_results.pkl --intensity 1.0
  
  # Plot synaptic weight distributions
  %(prog)s plot-weights results.pkl
  %(prog)s plot-weights results.pkl --optimization-file optimization.json
  
  # Get help for specific subcommand
  %(prog)s plot-comparative --help

        """
    )
    
    # Global options
    parser.add_argument('--output', '-o', 
                       default='./analysis_output',
                       help='Output directory for plots and results (default: ./analysis_output)')
    
    parser.add_argument('--optimization-file',
                       type=str,
                       default=None,
                       help='Path to optimization parameters JSON file (for circuit reconstruction)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    # Create subcommand parsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Analysis command to execute',
        metavar='COMMAND'
    )
    
    # ========== plot-comparative ==========
    parser_comp = subparsers.add_parser(
        'plot-comparative',
        help='Plot comparative experiment results (PV vs SST stimulation)',
        description='Generate publication-quality plots of comparative PV vs SST stimulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all available intensities
  %(prog)s results.pkl
  
  # Plot specific intensities
  %(prog)s results.pkl --intensities 1.0 1.5
  
  # Specify output directory
  %(prog)s results.pkl -o comparative_plots/
        """
    )
    parser_comp.add_argument('input',
                            help='Comparative results file (*.pkl)')
    parser_comp.add_argument('--intensities',
                            type=float,
                            nargs='+',
                            default=None,
                            metavar='INTENSITY',
                            help='Light intensities to plot (default: all available)')
    parser_comp.set_defaults(func=cmd_plot_comparative)
    
    # ========== plot-ablations ==========
    parser_abl = subparsers.add_parser(
        'plot-ablations',
        help='Plot ablation test results',
        description='Generate plots comparing full network vs. ablated conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot with default intensity
  %(prog)s ablation_results.pkl
  
  # Plot specific intensity
  %(prog)s ablation_results.pkl --intensity 1.5
        """
    )
    parser_abl.add_argument('input',
                            help='Ablation results file (*.pkl)')
    parser_abl.add_argument('--intensity',
                            type=float,
                            default=None,
                            metavar='VALUE',
                            help='Light intensity to plot (default: highest available)')
    parser_abl.add_argument('--export-csv',
                            action='store_true',
                            help='Export ablation data to CSV')
    
    parser_abl.set_defaults(func=cmd_plot_ablations)
    
    # ========== plot-weights ==========
    parser_weights = subparsers.add_parser(
        'plot-weights',
        help='Plot synaptic weight distributions',
        description='Generate weight distribution plots for each post-synaptic population',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot weights from comparative results
  %(prog)s results.pkl
  
  # Use optimized parameters for circuit reconstruction
  %(prog)s results.pkl --optimization-file optimization.json
        """
    )
    parser_weights.add_argument('input',
                               help='Comparative results file (*.pkl)')
    parser_weights.set_defaults(func=cmd_plot_weights)
    
    # ========== plot-expression ==========
    parser_expr = subparsers.add_parser(
        'plot-expression',
        help='Plot expression level dependence',
        description='Plot how paradoxical excitation varies with opsin expression level',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot expression level results
  %(prog)s expression_results.pkl
        """
    )
    parser_expr.add_argument('input',
                            help='Expression results file (*.pkl)')
    parser_expr.add_argument('--export-csv',
                             action='store_true',
                            help='Export expression level data to CSV')
    parser_expr.set_defaults(func=cmd_plot_expression)

    # ========== plot-failure ==========
    parser_fail = subparsers.add_parser(
        'plot-failure',
        help='Plot failure rate dependence',
        description='Plot how paradoxical excitation varies with opsin failure rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot failure rate results
  %(prog)s failure_rate_results.pkl
  
  # Specify output directory
  %(prog)s failure_rate_results.pkl -o failure_plots/
  
Note:
  X-axis shows "functional opsin expression" = (1 - failure_rate) * expression_mean
  This represents the fraction of cells with working opsins.
    """
)
    parser_fail.add_argument('input',
                             help='Failure rate results file (*.pkl)')
    parser_fail.set_defaults(func=cmd_plot_failure)
    
    # ========== plot-combined ==========
    parser_comb = subparsers.add_parser(
        'plot-combined',
        help='Generate combined ablation + expression figure',
        description='Create figure combining ablation and expression results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined figure
  %(prog)s results.pkl ablation_results.pkl expression_results.pkl
  
  # Specify intensity
  %(prog)s results.pkl ablation_results.pkl expression_results.pkl --intensity 1.5
        """
    )
    parser_comb.add_argument('comparative',
                            help='Comparative results file (*.pkl)')
    parser_comb.add_argument('ablation',
                            help='Ablation results file (*.pkl)')
    parser_comb.add_argument('expression',
                            help='Expression results file (*.pkl)')
    parser_comb.add_argument('--intensity',
                            type=float,
                            default=1.0,
                            metavar='VALUE',
                            help='Light intensity to plot (default: 1.0)')
    parser_comb.set_defaults(func=cmd_plot_combined)
    
    # ========== plot-currents ==========
    parser_curr = subparsers.add_parser(
        'plot-currents',
        help='Plot synaptic currents',
        description='Plot recorded synaptic currents from experiments with current recording enabled',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot currents (requires results with --record-currents)
  %(prog)s results.pkl
  
  # Use optimized parameters
  %(prog)s results.pkl --optimization-file optimization.json
        """
    )
    parser_curr.add_argument('input',
                            help='Comparative results file with current recordings (*.pkl)')
    parser_curr.set_defaults(func=cmd_plot_currents)
    
    # ========== analyze-weights-by-response ==========
    parser_wbr = subparsers.add_parser(
        'analyze-weights-by-response',
        help='Analyze weights by response type',
        description='Statistical analysis of synaptic weights comparing excited vs suppressed cells',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic weight-response analysis
  %(prog)s results.pkl
  
  # With bootstrap confidence intervals
  %(prog)s results.pkl --n-bootstrap 10000
  
  # Custom threshold
  %(prog)s results.pkl --threshold-std 1.5
        """
    )
    parser_wbr.add_argument('input',
                           help='Comparative results file (*.pkl)')
    parser_wbr.add_argument('--n-bootstrap',
                           type=int,
                           default=10000,
                           metavar='N',
                           help='Number of bootstrap samples (default: 10000)')
    parser_wbr.add_argument('--threshold-std',
                           type=float,
                           default=1.0,
                           metavar='VALUE',
                           help='Classification threshold in standard deviations (default: 1.0)')
    parser_wbr.set_defaults(func=cmd_analyze_weights_by_response)
    
    # ========== plot-variance ==========
    parser_var = subparsers.add_parser(
        'plot-variance',
        help='Plot variance decomposition',
        description='Plot variance decomposition from nested experiments (connectivity × MEC patterns)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot variance decomposition
  %(prog)s nested_results.pkl
  
  # Also plot connectivity variance details
  %(prog)s nested_results.pkl --detailed
        """
    )
    parser_var.add_argument('input',
                           help='Nested experiment results file (*.pkl)')
    parser_var.add_argument('--detailed',
                           action='store_true',
                           help='Generate detailed connectivity variance plots')
    parser_var.set_defaults(func=cmd_plot_variance)
    
    # ========== bootstrap-analysis ==========
    parser_boot = subparsers.add_parser(
        'bootstrap-analysis',
        help='Bootstrap effect size analysis',
        description='Perform bootstrap effect size analysis on nested experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic bootstrap analysis
  %(prog)s nested_results.pkl
  
  # Specify parameters
  %(prog)s nested_results.pkl --n-samples 10000 --seed 42
  
  # Analyze specific intensities
  %(prog)s nested_results.pkl --intensities 1.0 1.5
        """
    )
    parser_boot.add_argument('input',
                            help='Nested experiment results file (*.pkl)')
    parser_boot.add_argument('--n-samples',
                            type=int,
                            default=10000,
                            metavar='N',
                            help='Number of bootstrap samples (default: 10000)')
    parser_boot.add_argument('--threshold-std',
                            type=float,
                            default=1.0,
                            metavar='VALUE',
                            help='Classification threshold in std deviations (default: 1.0)')
    parser_boot.add_argument('--expression-threshold',
                            type=float,
                            default=0.2,
                            metavar='VALUE',
                            help='Opsin expression threshold (default: 0.2)')
    parser_boot.add_argument('--seed',
                            type=int,
                            default=None,
                            metavar='VALUE',
                            help='Random seed for reproducibility')
    parser_boot.add_argument('--intensities',
                            type=float,
                            nargs='+',
                            default=None,
                            metavar='INTENSITY',
                            help='Intensities to analyze (default: all)')
    parser_boot.set_defaults(func=cmd_bootstrap_analysis)

# ========== nested-weights-analysis ==========
    parser_weights = subparsers.add_parser(
        'nested-weights-analysis',
        help='Analyze synaptic weights by response across connectivity instances',
        description='Perform weight analysis on nested experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic nested weights analysis
  %(prog)s nested_results.pkl
  
  # With detailed connectivity comparison plots
  %(prog)s nested_results.pkl --detailed
  
  # With distributional analysis
  %(prog)s nested_results.pkl --run-distributional
        
  # Analyze specific populations
  %(prog)s nested_results.pkl --target-populations pv \\
                              --post-populations gc mc \\
                              --source-populations pv sst mc mec
  
  # Custom parameters
  %(prog)s nested_results.pkl --n-bootstrap 20000 \\
                              --threshold-std 1.5 \\
                              --expression-threshold 0.3
        """
    )
    parser_weights.add_argument('input',
                                help='Nested experiment results file (*.pkl)')
    parser_weights.add_argument('--target-populations',
                                type=str,
                                nargs='+',
                                default=None,
                                metavar='POP',
                                help='Target populations to analyze (default: pv sst)')
    parser_weights.add_argument('--post-populations',
                                type=str,
                                nargs='+',
                                default=None,
                                metavar='POP',
                                help='Post-synaptic populations to analyze (default: gc mc pv sst)')
    parser_weights.add_argument('--source-populations',
                                type=str,
                                nargs='+',
                                default=None,
                                metavar='POP',
                                help='Source populations for weight analysis (default: gc pv sst mc mec)')
    parser_weights.add_argument('--intensities',
                                type=float,
                                nargs='+',
                                default=None,
                                metavar='INTENSITY',
                                help='Intensities to analyze (default: all available)')
    parser_weights.add_argument('--n-bootstrap',
                                type=int,
                                default=10000,
                                metavar='N',
                                help='Number of bootstrap samples (default: 10000)')
    parser_weights.add_argument('--threshold-std',
                                type=float,
                                default=1.0,
                                metavar='VALUE',
                                help='Classification threshold in std deviations (default: 1.0)')
    parser_weights.add_argument('--expression-threshold',
                                type=float,
                                default=0.2,
                                metavar='VALUE',
                                help='Opsin expression threshold (default: 0.2)')

    parser_weights.add_argument('--run-distributional',
                                action='store_true',
                                default=False,
                                help='Include distributional analysis (geometric mean, CLES, quantiles)')
    parser_weights.add_argument('--distribution-quantiles',
                                type=float,
                                nargs='+',
                                default=[0.25, 0.50, 0.75, 0.90],
                                metavar='Q',
                                help='Quantiles for distributional analysis (default: 0.25 0.50 0.75 0.90)')
    parser_weights.add_argument('--weight-distribution-csv',
                                type=str,
                                default=None,
                                help='Export weight distribution data to CSV')

    parser_weights.add_argument('--run-pca',
                                action='store_true',
                                default=False,
                                help='Include PCA analysis of multivariate input weight patterns'
                                )
    parser_weights.add_argument('--n-pca-permutations',
                                type=int,
                                default=1000,
                                help='Number of permutations for PCA null distribution (default: 1000)'
                                )
    parser_weights.add_argument('--seed',
                                type=int,
                                default=None,
                                metavar='VALUE',
                                help='Random seed for reproducibility')
    parser_weights.add_argument('--plot',
                                action='store_true',
                                default=True,
                                help='Generate plots (default: True)')
    parser_weights.add_argument('--no-plot',
                                action='store_false',
                                dest='plot',
                                help='Skip plot generation')
    parser_weights.add_argument('--detailed',
                                action='store_true',
                                help='Generate detailed connectivity comparison plots')
    parser_weights.add_argument('--save-results',
                                action='store_true',
                                default=True,
                                help='Save analysis results to file (default: True)')
    parser_weights.add_argument('--no-save-results',
                                action='store_false',
                                dest='save_results',
                                help='Do not save analysis results')
    parser_weights.set_defaults(func=cmd_nested_weights_analysis)
    
    # ========== decision-analysis ==========
    parser_dec = subparsers.add_parser(
        'decision-analysis',
        help='Effect size decision framework',
        description='Run effect size decision framework to determine if more data is needed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic decision analysis
  %(prog)s nested_results.pkl
  
  # Specify power and feasibility constraints
  %(prog)s nested_results.pkl --target-power 0.8 --max-n 30
  
  # Set biological significance thresholds
  %(prog)s nested_results.pkl --min-effect 0.5 --min-diff-nS 0.1
        """
    )
    parser_dec.add_argument('input',
                           help='Nested experiment results file (*.pkl)')
    parser_dec.add_argument('--target-power',
                           type=float,
                           default=0.80,
                           metavar='VALUE',
                           help='Target statistical power (default: 0.80)')
    parser_dec.add_argument('--max-n',
                           type=int,
                           default=30,
                           metavar='VALUE',
                           help='Maximum feasible sample size (default: 30)')
    parser_dec.add_argument('--min-effect',
                           type=float,
                           default=0.5,
                           metavar='VALUE',
                           help='Minimum meaningful effect size (Cohen\'s d, default: 0.5)')
    parser_dec.add_argument('--min-diff-nS',
                           type=float,
                           default=0.05,
                           metavar='VALUE',
                           help='Minimum meaningful weight difference (nS, default: 0.05)')
    parser_dec.add_argument('--n-samples',
                           type=int,
                           default=10000,
                           metavar='N',
                           help='Number of bootstrap samples (default: 10000)')
    parser_dec.add_argument('--threshold-std',
                           type=float,
                           default=1.0,
                           metavar='VALUE',
                           help='Classification threshold (default: 1.0)')
    parser_dec.add_argument('--expression-threshold',
                           type=float,
                           default=0.2,
                           metavar='VALUE',
                           help='Opsin expression threshold (default: 0.2)')
    parser_dec.add_argument('--seed',
                           type=int,
                           default=None,
                           metavar='VALUE',
                           help='Random seed for reproducibility')
    parser_dec.add_argument('--intensities',
                           type=float,
                           nargs='+',
                           default=None,
                           metavar='INTENSITY',
                           help='Intensities to analyze (default: all)')
    parser_dec.set_defaults(func=cmd_decision_analysis)

    # ========== plot-connectivity-activity ==========
    parser_conn = subparsers.add_parser(
        'plot-connectivity-activity',
        help='Plot aggregated activity for specific connectivity instance',
        description='Plot trial-averaged activity for a single connectivity instance across MEC patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot connectivity 0 for PV stimulation
  %(prog)s nested_results.h5 --connectivity-idx 0 \\
                             --target-population pv --intensity 1.0
  
  # Plot with baseline normalization
  %(prog)s nested_results.pkl --connectivity-idx 2 \\
                              --target-population sst --intensity 1.5 \\
                              --baseline-normalize
  
  # Plot multiple connectivities
  for i in 0 1 2; do
    %(prog)s nested_results.h5 --connectivity-idx $i \\
                               --target-population pv --intensity 1.0
  done
        """
    )
    parser_conn.add_argument('input',
                            help='Nested experiment results file (*.pkl or *.h5)')
    parser_conn.add_argument('--connectivity-idx',
                            type=int,
                            required=True,
                            metavar='INDEX',
                            help='Connectivity instance index to plot')
    parser_conn.add_argument('--target-population',
                            type=str,
                            required=True,
                            choices=['pv', 'sst'],
                            metavar='POP',
                            help='Target population that was stimulated')
    parser_conn.add_argument('--intensity',
                            type=float,
                            required=True,
                            metavar='VALUE',
                            help='Light intensity to plot')
    parser_conn.add_argument('--baseline-normalize',
                            action='store_true',
                            help='Normalize activity relative to baseline')
    parser_conn.add_argument('--sort-by-activity',
                            action='store_true',
                            default=True,
                            help='Sort cells by activity (default: True)')
    parser_conn.add_argument('--no-sort',
                            action='store_false',
                            dest='sort_by_activity',
                            help='Do not sort cells by activity')
    parser_conn.set_defaults(func=cmd_plot_connectivity_activity)

    # ========== export-nested-csv ==========
    parser_csv = subparsers.add_parser(
        'export-nested-csv',
        help='Export per-trial aggregate statistics from nested experiments to CSV',
        description=(
            'Export one row per (connectivity, MEC pattern) with population-level '
            'proportions (excited / suppressed / unchanged), mean rate change, '
            'and log2 modulation ratio.  For the target population, separate '
            'columns are provided for expressing vs non-expressing cells.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from HDF5
  %(prog)s nested_results.h5

  # Export from pickle with explicit output path
  %(prog)s nested_results.pkl --csv-path results/trial_stats.csv

  # Export specific targets and intensities
  %(prog)s nested_results.h5 --target-populations pv sst \\
                             --intensities 1.0 1.5

  # Custom classification threshold
  %(prog)s nested_results.h5 --threshold-std 1.5
        """
    )
    parser_csv.add_argument(
        'input',
        help='Nested experiment results file (*.pkl or *.h5)',
    )
    parser_csv.add_argument(
        '--csv-path',
        type=str,
        default=None,
        metavar='PATH',
        help='Output CSV file path (default: <output>/nested_experiment_stats.csv)',
    )
    parser_csv.add_argument(
        '--target-populations',
        type=str,
        nargs='+',
        default=None,
        metavar='POP',
        help='Target populations to export (default: all in file)',
    )
    parser_csv.add_argument(
        '--intensities',
        type=float,
        nargs='+',
        default=None,
        metavar='INTENSITY',
        help='Intensities to export (default: all in file)',
    )
    parser_csv.add_argument(
        '--threshold-std',
        type=float,
        default=1.0,
        metavar='VALUE',
        help='Classification threshold in baseline std deviations (default: 1.0)',
    )
    parser_csv.add_argument(
        '--expression-threshold',
        type=float,
        default=0.2,
        metavar='VALUE',
        help='Opsin expression threshold for non-expressing classification (default: 0.2)',
    )
    parser_csv.add_argument(
        '--stim-start',
        type=float,
        default=1500.0,
        metavar='MS',
        help='Stimulation start time in ms (default: from metadata, fallback 1500)',
    )
    parser_csv.add_argument(
        '--stim-duration',
        type=float,
        default=1000.0,
        metavar='MS',
        help='Stimulation duration in ms (default: from metadata, fallback 1000)',
    )
    parser_csv.add_argument(
        '--warmup',
        type=float,
        default=500.0,
        metavar='MS',
        help='Baseline window start in ms (default: from metadata, fallback 500)',
    )
    parser_csv.set_defaults(func=cmd_export_nested_csv)

    # ========== export-per-cell-csv ==========
    parser_export = subparsers.add_parser(
        'export-per-cell-csv',
        help='Export per-cell weights and firing rates to CSV',
        description='Export detailed per-cell data for statistical analysis in R or Python',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export with defaults
  %(prog)s nested_results.h5
  
  # Custom output location and filename
  %(prog)s nested_results.h5 -o analysis/ \\
                             --csv-filename per_cell_data.csv
  
  # Export specific populations and intensities
  %(prog)s nested_results.h5 \\
      --target-populations pv \\
      --post-populations gc mc \\
      --source-populations gc mc pv sst mec \\
      --intensities 1.0 1.5
  
  # Custom classification thresholds
  %(prog)s nested_results.h5 \\
      --threshold-std 1.5 \\
      --expression-threshold 0.3

Output CSV structure:
  One row per (connectivity_idx, mec_pattern_idx, intensity, 
               post_population, post_cell_idx)
  
  Columns include:
  - Identifiers: connectivity_idx, mec_pattern_idx, intensity, 
                  post_population, post_cell_idx
  - Response: classification, baseline_rate, stim_rate, 
              rate_change, modulation_ratio
  - Opsin: post_opsin_expression, post_has_opsin (if target pop)
  - Weights: For each source population:
      - {source}_weight_sum, _mean, _median, _n_synapses
      - If source == target: _opsin_plus and _opsin_minus variants

Memory usage:
  HDF5 format uses streaming approach (peak ~500 MB)
  Processes one connectivity instance at a time
        """
    )
    parser_export.add_argument('input',
                               help='Nested experiment results file (*.h5)')
    parser_export.add_argument('--csv-filename',
                               type=str,
                               default='per_cell_weights_and_rates.csv',
                               metavar='FILENAME',
                               help='Output CSV filename (default: per_cell_weights_and_rates.csv)')
    parser_export.add_argument('--target-populations',
                               type=str,
                               nargs='+',
                               default=None,
                               metavar='POP',
                               help='Target populations to export (default: pv sst)')
    parser_export.add_argument('--post-populations',
                               type=str,
                               nargs='+',
                               default=None,
                               metavar='POP',
                               help='Post-synaptic populations (default: gc mc pv sst)')
    parser_export.add_argument('--source-populations',
                               type=str,
                               nargs='+',
                               default=None,
                               metavar='POP',
                               help='Source populations for weights (default: gc mc pv sst)')
    parser_export.add_argument('--intensities',
                               type=float,
                               nargs='+',
                               default=None,
                               metavar='INTENSITY',
                               help='Intensities to export (default: all)')
    parser_export.add_argument('--threshold-std',
                               type=float,
                               default=1.0,
                               metavar='VALUE',
                               help='Classification threshold in std (default: 1.0)')
    parser_export.add_argument('--expression-threshold',
                               type=float,
                               default=0.2,
                               metavar='VALUE',
                               help='Opsin expression threshold (default: 0.2)')
    parser_export.set_defaults(func=cmd_export_per_cell_csv)
    
    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for DG analysis script"""
    
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Check if subcommand was provided
    if not hasattr(args, 'func'):
        parser.print_help()
        print("\nError: No command specified", file=sys.stderr)
        print("Use --help to see available commands", file=sys.stderr)
        return 1
    
    # Execute subcommand
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
