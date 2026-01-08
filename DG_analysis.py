#!/usr/bin/env python3
"""DG Analysis Script - Offline analysis of dentate gyrus experiments

This script provides analysis and visualization capabilities for
results from DG optogenetic experiments.

Available Commands:
    plot-comparative          Plot PV vs SST comparative results
    plot-ablations           Plot ablation test results
    plot-weights             Plot synaptic weight distributions
    plot-expression          Plot expression level dependence
    plot-combined            Generate combined ablation + expression figure
    plot-currents            Plot synaptic currents
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
from typing import Dict, Tuple, Optional, Any
import logging

# Import functions from DG_protocol module
from DG_protocol import (
    # Loading
    load_experiment_results,
    reconstruct_circuit_from_metadata,
    
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
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def load_results_with_validation(filepath: str, result_type: str) -> Dict:
    """
    Load and validate results file
    
    Args:
        filepath: Path to pickle file
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
    
    # Check file extension
    if filepath.suffix != '.pkl':
        logger.warning(f"Unexpected file extension: {filepath.suffix} (expected .pkl)")
    
    # Load file
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load results file: {e}")
    
    # Validate based on expected type
    if result_type == 'comparative':
        # Comparative results should have specific structure
        if not isinstance(data, tuple) or len(data) != 4:
            raise ValueError(
                "Invalid comparative results format. Expected tuple of "
                "(results, connectivity_analysis, conductance_analysis, metadata)"
            )
        results, conn_analysis, cond_analysis, metadata = data
        
        # Check for required keys
        if 'pv' not in results and 'sst' not in results:
            raise ValueError("Comparative results missing 'pv' and 'sst' populations")
        
        logger.info(f"Loaded comparative results from {filepath}")
        logger.info(f"  Populations: {list(results.keys())}")
        if metadata and 'n_trials' in metadata:
            logger.info(f"  Trials: {metadata['n_trials']}")
        
        return data
        
    elif result_type == 'ablation':
        # Ablation results should be a dict with test names
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
        # Expression results should be a dict
        if not isinstance(data, dict):
            raise ValueError("Invalid expression results format. Expected dictionary")
        
        # Check for expected structure
        if 'full_network' not in data:
            raise ValueError("Expression results missing 'full_network' key")
        
        logger.info(f"Loaded expression results from {filepath}")
        logger.info(f"  Conditions: {list(data.keys())}")
        
        return data
        
    elif result_type == 'nested':
        # Nested results should have specific keys
        if not isinstance(data, dict):
            raise ValueError("Invalid nested results format. Expected dictionary")
        
        required_keys = ['nested_results', 'metadata']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValueError(
                f"Nested results missing required keys: {missing_keys}"
            )
        
        logger.info(f"Loaded nested results from {filepath}")
        if 'metadata' in data:
            metadata = data['metadata']
            if 'n_connectivity' in metadata:
                logger.info(f"  Connectivity instances: {metadata['n_connectivity']}")
            if 'n_mec_patterns' in metadata:
                logger.info(f"  MEC patterns: {metadata['n_mec_patterns']}")
        
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
    Handle nested weights analysis command
    
    Analyzes synaptic weight distributions by post-synaptic response
    across connectivity instances in nested experiments.
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
    nested_results = nested_data['nested_results']
    seed_structure = nested_data['seed_structure']
    
    # Extract experiment parameters
    stim_start = metadata['stim_start']
    stim_duration = metadata['stim_duration']
    warmup = metadata['warmup']
    
    # Determine intensities and populations to analyze
    target_populations = args.target_populations or ['pv', 'sst']
    post_populations = args.post_populations or ['gc', 'mc', 'pv', 'sst']
    source_populations = args.source_populations or ['pv', 'sst', 'mc', 'mec']
    
    if args.intensities is None:
        intensities = metadata['intensities']
    else:
        intensities = args.intensities
    
    logger.info(f"\nAnalysis parameters:")
    logger.info(f"  Target populations: {target_populations}")
    logger.info(f"  Post populations: {post_populations}")
    logger.info(f"  Source populations: {source_populations}")
    logger.info(f"  Intensities: {intensities}")
    logger.info(f"  Bootstrap samples: {args.n_bootstrap}")
    logger.info(f"  Classification threshold: {args.threshold_std} std")
    
    # Import necessary classes
    from DG_circuit_dendritic_somatic_transfer import (
        CircuitParams, PerConnectionSynapticParams, OpsinParams
    )
    from DG_protocol import OptogeneticExperiment, set_random_seed, get_default_device
    
    device = get_default_device()
    
    # Storage for all analyses
    all_analyses = {}
    
    # Analyze each target population
    for target in target_populations:
        if target not in nested_results:
            logger.warning(f"Skipping {target.upper()} - no results found")
            continue
        
        all_analyses[target] = {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {target.upper()} stimulation")
        logger.info('='*60)
        
        for intensity in intensities:
            if intensity not in nested_results[target]:
                logger.warning(f"  Skipping intensity {intensity} - no results")
                continue
            
            logger.info(f"\n  Intensity: {intensity}")
            
            # Get trials for this condition
            trials = nested_results[target][intensity]
            
            logger.info(f"    Total trials: {len(trials)}")
            logger.info(f"    Connectivity instances: {len(set(t.connectivity_idx for t in trials))}")
            
            # Recreate circuit (use seed from first connectivity instance)
            first_conn_seed = seed_structure['connectivity_seeds'][0]
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
                
                # Run nested weights analysis
                try:
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
                        random_seed=args.seed
                    )
                    
                    # Store results
                    if target not in all_analyses:
                        all_analyses[target] = {}
                    if intensity not in all_analyses[target]:
                        all_analyses[target][intensity] = {}
                    all_analyses[target][intensity][post_pop] = analysis_results
                    
                    # Generate visualization
                    if args.plot:
                        vis_dir = output_dir / f"{target}_intensity_{intensity}"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        
                        logger.info(f"      Generating plot...")
                        fig = plot_weights_by_average_response_nested(
                            analysis_results,
                            save_path=str(vis_dir / f'nested_weights_{post_pop}.pdf')
                        )
                        plt.close(fig)
                        
                        # Generate detailed connectivity comparison if requested
                        if args.detailed:
                            logger.info(f"      Generating detailed connectivity comparison...")
                            n_conn = analysis_results['metadata']['n_connectivity_instances']
                            # Plot first few connectivities for detail
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
            
            # Clean up circuit
            del experiment
            if device.type == 'cuda':
                import torch
                torch.cuda.empty_cache()
    
    # Save complete analysis results
    if args.save_results:
        analysis_file = output_dir / 'nested_weights_analysis_results.pkl'
        import pickle
        with open(analysis_file, 'wb') as f:
            pickle.dump({
                'all_analyses': all_analyses,
                'metadata': metadata,
                'parameters': {
                    'threshold_std': args.threshold_std,
                    'expression_threshold': args.expression_threshold,
                    'n_bootstrap': args.n_bootstrap,
                    'random_seed': args.seed
                }
            }, f)
        logger.info(f"\nAnalysis results saved to: {analysis_file}")
    
    logger.info(f"Nested weights analysis output saved to: {output_dir}")
    
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
                                help='Source populations for weight analysis (default: pv sst mc mec)')
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
