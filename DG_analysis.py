#!/usr/bin/env python3
"""DG Analysis Script - Offline analysis of dentate gyrus experiments

This script provides analysis and visualization capabilities for
results from DG optogenetic experiments.

Available Commands:
    plot-comparative         Plot PV vs SST comparative results
    plot-ablations           Plot ablation test results
    plot-weights             Plot synaptic weight distributions
    plot-expression          Plot expression level dependence
    plot-combined            Generate combined ablation + expression figure
    plot-currents            Plot synaptic currents
    plot-connectivity-activity  Plot aggregated activity for specific connectivity instance
    analyze-weights-by-response  Statistical weight analysis by response
    plot-variance            Plot variance decomposition
    bootstrap-analysis       Bootstrap effect size analysis
    nested-weights-analysis  Analyze weights by response in nested experiments
    decision-analysis        Effect size decision framework

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

# Import functions from DG_protocol module
from DG_protocol import (
    OptogeneticExperiment, set_random_seed, get_default_device,
    
    # Loading
    load_experiment_results,

    # Aggregation utilities
    aggregate_trial_results,
    aggregate_adaptive_stats,
    
    # Plotting - Comparative
    plot_comparative_experiment_results,
    
    # Plotting - Ablations
    plot_ablation_test_results,
    
    # Plotting - Expression
    plot_expression_level_results,
    plot_combined_ablation_and_expression,
    
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
                require_full_activity=True
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
        plot_ablation_test_results(
            ablation_results,
            intensity=intensity,
            save_path=str(output_dir)
        )
        logger.info(f"Saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to plot ablation results: {e}")
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
        plot_expression_level_results(
            expression_results,
            save_path=str(output_dir)
        )
        logger.info(f"Saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to plot expression results: {e}")
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

For more information, see the documentation in the project memory bank.
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
    parser_expr.set_defaults(func=cmd_plot_expression)
    
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
