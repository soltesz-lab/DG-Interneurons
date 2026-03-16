"""
Nested Experimental Framework for Dentate Gyrus Circuit:
Core nested simulation loop and aggregation functions

This framework enables hierarchical variance decomposition to
investigate whether paradoxical excitation is driven by specific
synaptic weight patterns (connectivity-driven) or population-level
dynamics (input-driven).

Design:
    Outer loop: n_connectivity_instances (different circuit realizations)
    Inner loop: n_mec_patterns (different MEC input patterns per connectivity)
    
This allows decomposition of variance:
    Total variance = Var(connectivity) + Var(input|connectivity) + Var(residual)

"""

from typing import Dict, List, Tuple, Optional, Iterator, NamedTuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import csv
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import time

import h5py
from hdf5_storage import (
    create_hdf5_experiment_file,
    save_trial_to_hdf5,
    load_trial_from_hdf5,
    load_metadata_from_hdf5,
    load_nested_trials_from_hdf5,
    extract_trial_statistics_from_hdf5
)

logger = logging.getLogger('nested_experiment')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


@dataclass
class NestedExperimentConfig:
    """Configuration for nested experimental framework"""
    
    # Nested structure
    n_connectivity_instances: int = 3
    n_mec_patterns_per_connectivity: int = 5
    
    # Random seed management
    base_seed: int = 42
    connectivity_seed_offset: int = 10000
    mec_pattern_seed_offset: int = 1000
    
    # Whether to save full nested structure
    save_nested_trials: bool = True
    save_full_activity: bool = False
    
    # Statistical thresholds
    variance_significance_alpha: float = 0.001
    effect_size_threshold: float = 0.2  # Cohen's d


class NestedTrialResult(NamedTuple):
    """Results from a single trial in nested framework"""
    connectivity_idx: int
    mec_pattern_idx: int
    seed: int
    results: Dict  # Standard trial results
    target_population: str  # Which population was stimulated ('pv', 'sst', etc.)
    opsin_expression: np.ndarray  # Expression levels for target population [n_cells]


def generate_nested_seeds(config: NestedExperimentConfig) -> Dict[str, List[List[int]]]:
    """
    Generate seeds for nested experimental structure
    
    Returns:
        Dict with 'connectivity_seeds' and 'mec_pattern_seeds'
        where mec_pattern_seeds[i][j] is the seed for connectivity i, pattern j
    """
    connectivity_seeds = [
        config.base_seed + config.connectivity_seed_offset * (i + 1)
        for i in range(config.n_connectivity_instances)
    ]
    
    mec_pattern_seeds = [
        [
            conn_seed + config.mec_pattern_seed_offset * (j + 1)
            for j in range(config.n_mec_patterns_per_connectivity)
        ]
        for conn_seed in connectivity_seeds
    ]
    
    return {
        'connectivity_seeds': connectivity_seeds,
        'mec_pattern_seeds': mec_pattern_seeds
    }


# ============================================================================
# Analysis Functions (Moved from hdf5_storage.py)
# ============================================================================

def _aggregate_precomputed_stats(trials: List[Dict], target: str) -> Dict:
    """
    Aggregate pre-computed statistics from multiple trials
    
    Used when save_full_activity=False to avoid loading full traces.
    
    Args:
        trials: List of trial statistics dicts (from extract_trial_statistics_from_hdf5)
        target: Target population being stimulated
        
    Returns:
        Aggregated statistics across trials
    """
    aggregated = {}
    
    # Collect stats for each population
    for pop in ['gc', 'mc', 'pv', 'sst']:
        metrics = {
            'excited': [],
            'mean_change': [],
            'baseline_rate': [],
            'stim_rate': []
        }
        
        # Add non-expressing metrics for target population
        if pop == target:
            metrics['excited_nonexpr'] = []
            metrics['mean_change_nonexpr'] = []
            metrics['n_nonexpr'] = []
        
        # Collect from all trials
        for trial in trials:
            if pop in trial['stats']:
                stats = trial['stats'][pop]
                metrics['excited'].append(stats['excited'])
                metrics['mean_change'].append(stats['mean_change'])
                metrics['baseline_rate'].append(stats['baseline_rate'])
                metrics['stim_rate'].append(stats['stim_rate'])
                
                if pop == target and 'excited_nonexpr' in stats:
                    metrics['excited_nonexpr'].append(stats['excited_nonexpr'])
                    metrics['mean_change_nonexpr'].append(stats['mean_change_nonexpr'])
                    metrics['n_nonexpr'].append(stats['n_nonexpr'])
        
        # Compute aggregated statistics
        aggregated[pop] = {
            'excited_mean': np.mean(metrics['excited']),
            'excited_std': np.std(metrics['excited']),
            'excited_sem': np.std(metrics['excited']) / np.sqrt(len(metrics['excited'])),
            'mean_change_mean': np.mean(metrics['mean_change']),
            'mean_change_std': np.std(metrics['mean_change']),
            'baseline_rate_mean': np.mean(metrics['baseline_rate']),
            'stim_rate_mean': np.mean(metrics['stim_rate']),
            'n_trials': len(trials)
        }
        
        # Add non-expressing statistics for target population
        if pop == target and metrics['excited_nonexpr']:
            aggregated[pop]['excited_nonexpr_mean'] = np.mean(metrics['excited_nonexpr'])
            aggregated[pop]['excited_nonexpr_std'] = np.std(metrics['excited_nonexpr'])
            aggregated[pop]['mean_change_nonexpr_mean'] = np.mean(metrics['mean_change_nonexpr'])
            aggregated[pop]['mean_change_nonexpr_std'] = np.std(metrics['mean_change_nonexpr'])
            aggregated[pop]['n_nonexpr_mean'] = np.mean(metrics['n_nonexpr'])
    
    return aggregated


def _aggregate_from_hdf5_storage(f: h5py.File,
                                  target: str,
                                  intensity: float,
                                  stim_start: float,
                                  stim_duration: float,
                                  warmup: float,
                                  expression_threshold: float = 0.2) -> Dict:
    """
    Compute aggregated statistics directly from HDF5 file
    
    This enables analysis without loading all data into memory.
    Works with both full activity traces and pre-computed summary stats.
    
    Args:
        f: Open h5py.File object
        target: Target population
        intensity: Light intensity
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period start (ms)
        expression_threshold: Opsin expression threshold
        
    Returns:
        Dict mapping connectivity_idx to aggregated statistics
    """
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in f[target]:
        return {}
    
    # Group by connectivity
    conn_groups = {}
    
    for conn_key in f[target][intensity_key].keys():
        conn_idx = int(conn_key.split('_')[1])
        conn_groups[conn_idx] = []
        
        for pattern_key in f[target][intensity_key][conn_key].keys():
            pattern_idx = int(pattern_key.split('_')[1])
            trial_grp = f[target][intensity_key][conn_key][pattern_key]
            
            # Check if we have pre-computed stats or need to load full activity
            has_precomputed = 'gc_excited' in trial_grp.attrs
            
            if has_precomputed:
                # Use pre-computed statistics (memory efficient)
                trial_data = {
                    'precomputed_stats': True,
                    'connectivity_idx': conn_idx,
                    'mec_pattern_idx': pattern_idx,
                    'target_population': trial_grp.attrs['target_population'],
                    'opsin_expression': trial_grp['opsin_expression'][:],
                    'stats': {}
                }
                
                # Extract stats for all populations
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if f'{pop}_excited' in trial_grp.attrs:
                        trial_data['stats'][pop] = {
                            'excited': trial_grp.attrs[f'{pop}_excited'],
                            'mean_change': trial_grp.attrs[f'{pop}_mean_change'],
                            'baseline_rate': trial_grp.attrs[f'{pop}_baseline_rate'],
                            'stim_rate': trial_grp.attrs[f'{pop}_stim_rate']
                        }
                        
                        # Add non-expressing cell stats if this is target population
                        if pop == target and f'{pop}_excited_nonexpr' in trial_grp.attrs:
                            trial_data['stats'][pop]['excited_nonexpr'] = trial_grp.attrs[f'{pop}_excited_nonexpr']
                            trial_data['stats'][pop]['mean_change_nonexpr'] = trial_grp.attrs[f'{pop}_mean_change_nonexpr']
                            trial_data['stats'][pop]['baseline_rate_nonexpr'] = trial_grp.attrs[f'{pop}_baseline_rate_nonexpr']
                            trial_data['stats'][pop]['stim_rate_nonexpr'] = trial_grp.attrs[f'{pop}_stim_rate_nonexpr']
                            trial_data['stats'][pop]['n_nonexpr'] = trial_grp.attrs[f'{pop}_n_nonexpr']
                
                conn_groups[conn_idx].append(trial_data)
            else:
                # Load full activity and compute
                trial_data = load_trial_from_hdf5(f, target, intensity, 
                                                 conn_idx, pattern_idx)
                
                # Create NestedTrialResult-like object
                trial_result = NestedTrialResult(
                    connectivity_idx=conn_idx,
                    mec_pattern_idx=pattern_idx,
                    seed=0,
                    results=trial_data,
                    target_population=trial_data['target_population'],
                    opsin_expression=trial_data['opsin_expression']
                )
                
                conn_groups[conn_idx].append(trial_result)
    
    # Aggregate within each connectivity instance
    by_conn = {}
    for conn_idx, trials in conn_groups.items():
        # Check if we have pre-computed stats
        if trials and isinstance(trials[0], dict) and trials[0].get('precomputed_stats'):
            # Aggregate from pre-computed stats
            by_conn[conn_idx] = _aggregate_precomputed_stats(trials, target)
        else:
            # Use original aggregation (requires full activity)
            by_conn[conn_idx] = _aggregate_trial_group(
                trials, stim_start, stim_duration, warmup, expression_threshold
            )
    
    return by_conn


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_nested_comparative_experiment(
    optimization_json_file: Optional[str] = None,
    intensities: List[float] = [1.0],
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    device: Optional[torch.device] = None,
    nested_config: Optional[NestedExperimentConfig] = None,
    save_results_file: Optional[str] = None,
    use_hdf5: bool = True,
    **optogenetic_kwargs
) -> Dict:
    """
    Run nested comparative experiment with hierarchical structure
    
    Structure:
        For each connectivity instance:
            For each MEC pattern:
                Run optogenetic stimulation
                
    This enables variance decomposition:
        - Connectivity-driven variance (between connectivity instances)
        - Input-driven variance (between MEC patterns within connectivity)
        - Residual variance (within-trial noise)
        
    Args:
        optimization_json_file: Path to optimization results
        intensities: Light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Pre-stimulation baseline (ms)
        device: Device to run on
        nested_config: Configuration for nested structure
        save_results_file: Path to save results
        **optogenetic_kwargs: Additional arguments for OptogeneticExperiment
        
    Returns:
        Dict containing nested results and aggregations
    """
    from DG_protocol import (
        OptogeneticExperiment, CircuitParams, PerConnectionSynapticParams,
        OpsinParams, set_random_seed, get_default_device
    )
    
    if device is None:
        device = get_default_device()
    
    if nested_config is None:
        nested_config = NestedExperimentConfig()
    
    logger.info("\n" + "="*80)
    logger.info("Nested comparative experiment")
    logger.info("="*80)
    logger.info(f"Structure:")
    logger.info(f"  Connectivity instances: {nested_config.n_connectivity_instances}")
    logger.info(f"  MEC patterns per connectivity: {nested_config.n_mec_patterns_per_connectivity}")
    logger.info(f"  Total trials per condition: {nested_config.n_connectivity_instances * nested_config.n_mec_patterns_per_connectivity}")
    logger.info("="*80 + "\n")
    
    # Generate seeds
    seed_structure = generate_nested_seeds(nested_config)
    
    # Initialize circuit parameters (shared across all instances)
    circuit_params = CircuitParams()
    synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()

    metadata = {
        'optimization_file': optimization_json_file,
        'intensities': intensities,
        'mec_current': mec_current,
        'opsin_current': opsin_current,
        'stim_start': stim_start,
        'stim_duration': stim_duration,
        'warmup': warmup,
        'seed_structure': seed_structure,
        'n_connectivity_instances': nested_config.n_connectivity_instances,
        'n_mec_patterns_per_connectivity': nested_config.n_mec_patterns_per_connectivity,
        'base_seed': nested_config.base_seed
    }
    
    # Initialize storage
    if use_hdf5:
        # Create HDF5 file
        if save_results_file is None:
            save_results_file = 'nested_experiment_results.h5'
        
        # Ensure .h5 extension
        if not save_results_file.endswith('.h5'):
            save_results_file = save_results_file.replace('.pkl', '.h5')
        
        hdf5_file = create_hdf5_experiment_file(save_results_file, metadata)
        nested_results = None  # Don't store in memory
    else:
        # store everything in memory
        hdf5_file = None
        nested_results = {
            'pv': {intensity: [] for intensity in intensities},
            'sst': {intensity: [] for intensity in intensities}
        }
    
    # Main nested loop
    try:

        for target in ['pv', 'sst']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {target.upper()} stimulation")
            logger.info('='*60)

            for intensity in intensities:
                logger.info(f"\nIntensity: {intensity}")
                logger.info(f"{'-'*60}")

                for conn_idx in range(nested_config.n_connectivity_instances):
                    connectivity_seed = seed_structure['connectivity_seeds'][conn_idx]

                    logger.info(f"\n  Connectivity instance {conn_idx + 1}/{nested_config.n_connectivity_instances}")
                    logger.info(f"    Seed: {connectivity_seed}")

                    # Create experiment with this connectivity
                    experiment = OptogeneticExperiment(
                        circuit_params,
                        synaptic_params,
                        opsin_params,
                        optimization_json_file=optimization_json_file,
                        device=device,
                        base_seed=connectivity_seed,
                        **optogenetic_kwargs
                    )

                    # Inner loop: MEC patterns
                    for mec_idx in range(nested_config.n_mec_patterns_per_connectivity):
                        mec_seed = seed_structure['mec_pattern_seeds'][conn_idx][mec_idx]

                        logger.info(f"    MEC pattern {mec_idx + 1}/{nested_config.n_mec_patterns_per_connectivity} (seed: {mec_seed})")

                        # Set seed for MEC pattern generation
                        set_random_seed(mec_seed, device)

                        # Run single trial with this connectivity and MEC pattern
                        result = experiment.simulate_stimulation(
                            target_population=target,
                            light_intensity=intensity,
                            stim_start=stim_start,
                            stim_duration=stim_duration,
                            post_duration=500.0,
                            mec_current=mec_current,
                            opsin_current=opsin_current,
                            plot_activity=False,
                            n_trials=1,  # Single trial per combination
                            regenerate_connectivity_per_trial=False  # Connectivity fixed
                        )

                        # Extract opsin expression for target population
                        if target in experiment.opsin_expression:
                            opsin_expr = experiment.opsin_expression[target].expression_levels
                            if hasattr(opsin_expr, 'cpu'):
                                opsin_expr = opsin_expr.cpu().numpy()
                            else:
                                opsin_expr = np.array(opsin_expr)
                        else:
                            # Fallback: create zero array if expression not available
                            n_cells = getattr(experiment.circuit_params, f'n_{target}')
                            opsin_expr = np.zeros(n_cells)

                        # Save to HDF5 immediately or store in memory
                        if use_hdf5:
                            save_trial_to_hdf5(
                                hdf5_file,
                                target,
                                intensity,
                                conn_idx,
                                mec_idx,
                                result,
                                save_full_activity=nested_config.save_full_activity,
                                stim_start=stim_start,
                                stim_duration=stim_duration,
                                warmup=warmup
                            )
                            # Flush to disk
                            hdf5_file.flush()
                        else:
                            # Store in memory with indexing and opsin expression
                            if target in experiment.opsin_expression:
                                opsin_expr = experiment.opsin_expression[target].expression_levels
                                if hasattr(opsin_expr, 'cpu'):
                                    opsin_expr = opsin_expr.cpu().numpy()
                                else:
                                    opsin_expr = np.array(opsin_expr)
                            else:
                                n_cells = getattr(experiment.circuit_params, f'n_{target}')
                                opsin_expr = np.zeros(n_cells)

                            trial_result = NestedTrialResult(
                                connectivity_idx=conn_idx,
                                mec_pattern_idx=mec_idx,
                                seed=mec_seed,
                                results=result,
                                target_population=target,
                                opsin_expression=opsin_expr
                            )
                            nested_results[target][intensity].append(trial_result)

                    # Clean up circuit to save memory
                    del experiment
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

        # Aggregate results
        logger.info("\n" + "="*80)
        logger.info("Aggregating nested results")
        logger.info("="*80)
        if use_hdf5:
            # Compute from HDF5 file
            aggregated_results = aggregate_nested_results_from_hdf5(
                hdf5_file, nested_config, stim_start, stim_duration, warmup
            )

            variance_analysis = compute_variance_decomposition_from_hdf5(
                hdf5_file, nested_config, stim_start, stim_duration, warmup
            )
        else:
            aggregated_results = aggregate_nested_results(
                nested_results,
                nested_config,
                stim_start,
                stim_duration,
                warmup
            )

            # Compute variance decomposition
            variance_analysis = compute_variance_decomposition(
                nested_results,
                nested_config,
                stim_start,
                stim_duration,
                warmup
            )

        # Classify regime (connectivity-driven vs. input-driven)
        regime_classification = classify_mechanism_regime(
            variance_analysis,
            nested_config
        )

        # Package complete results
        complete_results = {
            'nested_results': nested_results if nested_config.save_nested_trials else None,
            'aggregated_results': aggregated_results,
            'variance_analysis': variance_analysis,
            'regime_classification': regime_classification,
            'config': nested_config,
            'seed_structure': seed_structure,
            'metadata': metadata,
            'hdf5_file': save_results_file if use_hdf5 else None
        }

        # Save or update HDF5
        if use_hdf5:
            # Store aggregated results and variance analysis in HDF5
            _save_analysis_to_hdf5(hdf5_file, aggregated_results, 
                                   variance_analysis, regime_classification)
        elif save_results_file:
            save_nested_experiment_results(complete_results, save_results_file)

        # Print summary
        print_nested_experiment_summary(complete_results)

        return complete_results

    finally:
        # Always close HDF5 file
        if hdf5_file is not None:
            hdf5_file.close()
            logger.info(f"\nHDF5 file closed: {save_results_file}")


# ============================================================================
# Aggregation Functions
# ============================================================================

def aggregate_nested_results(
    nested_results: Dict,
    config: NestedExperimentConfig,
    stim_start: float,
    stim_duration: float,
    warmup: float
) -> Dict:
    """
    Aggregate nested results at multiple levels
    
    Aggregation levels:
        1. Within connectivity (across MEC patterns)
        2. Across all trials (grand mean)
        3. By connectivity instance
        
    Returns:
        Dict with aggregated statistics at each level
    """
    aggregated = {
        'by_connectivity': {},
        'grand_mean': {},
        'across_connectivity': {}
    }
    
    for target in ['pv', 'sst']:
        aggregated['by_connectivity'][target] = {}
        aggregated['grand_mean'][target] = {}
        aggregated['across_connectivity'][target] = {}
        
        for intensity, trials in nested_results[target].items():
            # Group by connectivity instance
            conn_groups = {}
            for trial in trials:
                conn_idx = trial.connectivity_idx
                if conn_idx not in conn_groups:
                    conn_groups[conn_idx] = []
                conn_groups[conn_idx].append(trial)
            
            # 1. Aggregate within each connectivity instance
            by_conn = {}
            for conn_idx, conn_trials in conn_groups.items():
                by_conn[conn_idx] = _aggregate_trial_group(
                    conn_trials,
                    stim_start,
                    stim_duration,
                    warmup
                )
            
            aggregated['by_connectivity'][target][intensity] = by_conn
            
            # 2. Grand mean across all trials
            aggregated['grand_mean'][target][intensity] = _aggregate_trial_group(
                trials,
                stim_start,
                stim_duration,
                warmup
            )
            
            # 3. Aggregate across connectivity instances
            # (mean and std of the within-connectivity means)
            aggregated['across_connectivity'][target][intensity] = \
                _aggregate_connectivity_means(by_conn)
    
    return aggregated


def _aggregate_trial_group(
    trials: List[NestedTrialResult],
    stim_start: float,
    stim_duration: float,
    warmup: float,
    expression_threshold: float = 0.2
) -> Dict:
    """
    Aggregate a group of trials (e.g., all trials with same connectivity)
    
    Computes mean and std of response metrics across trials.
    For the target population, only analyzes non-expressing cells to measure
    true paradoxical excitation (not direct optogenetic effects).
    
    Args:
        trials: List of trial results to aggregate
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period start (ms)
        expression_threshold: Opsin expression threshold for classification
                             (cells with expression < threshold are "non-expressing")
    
    Returns:
        Aggregated statistics across trials
    """
    if len(trials) == 0:
        return {}
    
    # Extract activity traces
    time = trials[0].results['time']
    baseline_mask = (time >= warmup) & (time < stim_start)
    stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
    
    # Get target population (should be same for all trials)
    target_population = trials[0].target_population
    
    # Collect metrics across trials
    metrics = {
        'gc': {'excited': [], 'mean_change': [], 'baseline_rate': [], 'stim_rate': []},
        'mc': {'excited': [], 'mean_change': [], 'baseline_rate': [], 'stim_rate': []},
        'pv': {'excited': [], 'mean_change': [], 'baseline_rate': [], 'stim_rate': []},
        'sst': {'excited': [], 'mean_change': [], 'baseline_rate': [], 'stim_rate': []}
    }
    
    # Add metrics for non-expressing cells in target population
    metrics[target_population]['excited_nonexpr'] = []
    metrics[target_population]['mean_change_nonexpr'] = []
    metrics[target_population]['n_nonexpr'] = []
    
    for trial in trials:
        # Handle both aggregated (activity_trace_mean) and raw (activity_trace) results
        if 'activity_trace_mean' in trial.results:
            activity = trial.results['activity_trace_mean']
        elif 'activity_trace' in trial.results:
            activity = trial.results['activity_trace']
        else:
            raise KeyError(f"Trial results missing both 'activity_trace' and 'activity_trace_mean'. "
                          f"Available keys: {list(trial.results.keys())}")
        
        # Get opsin expression levels for this trial
        opsin_expression = trial.opsin_expression
        
        for pop in metrics.keys():
            pop_activity = activity[pop]
            
            baseline_rate = torch.mean(pop_activity[:, baseline_mask], dim=1)
            stim_rate = torch.mean(pop_activity[:, stim_mask], dim=1)
            rate_change = stim_rate - baseline_rate
            baseline_std = torch.std(baseline_rate)
            
            # For target population, analyze non-expressing cells separately
            if pop == target_population:
                # Identify non-expressing cells
                non_expressing_mask = opsin_expression < expression_threshold
                n_nonexpr = np.sum(non_expressing_mask)
                
                if n_nonexpr > 0:
                    # Convert to torch tensor for indexing
                    if not isinstance(non_expressing_mask, torch.Tensor):
                        non_expressing_mask = torch.from_numpy(non_expressing_mask)
                    
                    # Get responses for non-expressing cells only
                    baseline_rate_nonexpr = baseline_rate[non_expressing_mask]
                    stim_rate_nonexpr = stim_rate[non_expressing_mask]
                    rate_change_nonexpr = rate_change[non_expressing_mask]
                    baseline_std_nonexpr = torch.std(baseline_rate_nonexpr)
                    
                    excited_fraction_nonexpr = torch.mean(
                        (rate_change_nonexpr > baseline_std_nonexpr).float()
                    ).item()
                    
                    metrics[pop]['excited_nonexpr'].append(excited_fraction_nonexpr)
                    metrics[pop]['mean_change_nonexpr'].append(torch.mean(rate_change_nonexpr).item())
                    metrics[pop]['n_nonexpr'].append(n_nonexpr)
                else:
                    # No non-expressing cells
                    metrics[pop]['excited_nonexpr'].append(0.0)
                    metrics[pop]['mean_change_nonexpr'].append(0.0)
                    metrics[pop]['n_nonexpr'].append(0)
            
            # Standard metrics (all cells) for all populations
            excited_fraction = torch.mean((rate_change > baseline_std).float()).item()
            
            metrics[pop]['excited'].append(excited_fraction)
            metrics[pop]['mean_change'].append(torch.mean(rate_change).item())
            metrics[pop]['baseline_rate'].append(torch.mean(baseline_rate).item())
            metrics[pop]['stim_rate'].append(torch.mean(stim_rate).item())
    
    # Compute statistics
    aggregated = {}
    for pop, pop_metrics in metrics.items():
        aggregated[pop] = {
            'excited_mean': np.mean(pop_metrics['excited']),
            'excited_std': np.std(pop_metrics['excited']),
            'excited_sem': np.std(pop_metrics['excited']) / np.sqrt(len(pop_metrics['excited'])),
            'mean_change_mean': np.mean(pop_metrics['mean_change']),
            'mean_change_std': np.std(pop_metrics['mean_change']),
            'baseline_rate_mean': np.mean(pop_metrics['baseline_rate']),
            'stim_rate_mean': np.mean(pop_metrics['stim_rate']),
            'n_trials': len(trials)
        }
        
        # Add non-expressing cell statistics for target population
        if pop == target_population and 'excited_nonexpr' in pop_metrics:
            aggregated[pop]['excited_nonexpr_mean'] = np.mean(pop_metrics['excited_nonexpr'])
            aggregated[pop]['excited_nonexpr_std'] = np.std(pop_metrics['excited_nonexpr'])
            aggregated[pop]['mean_change_nonexpr_mean'] = np.mean(pop_metrics['mean_change_nonexpr'])
            aggregated[pop]['mean_change_nonexpr_std'] = np.std(pop_metrics['mean_change_nonexpr'])
            aggregated[pop]['n_nonexpr_mean'] = np.mean(pop_metrics['n_nonexpr'])
    
    return aggregated


def _aggregate_connectivity_means(by_conn: Dict) -> Dict:
    """
    Aggregate statistics across connectivity instances
    
    Takes mean of within-connectivity means and computes between-connectivity variance
    """
    if len(by_conn) == 0:
        return {}
    
    populations = list(next(iter(by_conn.values())).keys())
    
    aggregated = {}
    for pop in populations:
        # Collect means from each connectivity instance
        excited_means = [conn_stats[pop]['excited_mean'] for conn_stats in by_conn.values()]
        change_means = [conn_stats[pop]['mean_change_mean'] for conn_stats in by_conn.values()]
        
        aggregated[pop] = {
            'excited_grand_mean': np.mean(excited_means),
            'excited_between_conn_std': np.std(excited_means),
            'excited_between_conn_sem': np.std(excited_means) / np.sqrt(len(excited_means)),
            'mean_change_grand_mean': np.mean(change_means),
            'mean_change_between_conn_std': np.std(change_means),
            'n_connectivity_instances': len(by_conn)
        }
    
    return aggregated


def aggregate_nested_results_from_hdf5(
    f: h5py.File,
    config: NestedExperimentConfig,
    stim_start: float,
    stim_duration: float,
    warmup: float
) -> Dict:
    """
    Aggregate nested results directly from HDF5 file
    
    Memory-efficient version that doesn't load all data at once.
    """
    aggregated = {
        'by_connectivity': {},
        'grand_mean': {},
        'across_connectivity': {}
    }
    
    for target in ['pv', 'sst']:
        aggregated['by_connectivity'][target] = {}
        aggregated['grand_mean'][target] = {}
        aggregated['across_connectivity'][target] = {}
        
        # Get all intensities from HDF5
        intensities = [float(k.split('_')[1]) for k in f[target].keys() 
                      if k.startswith('intensity_')]
        
        for intensity in intensities:
            # Aggregate within each connectivity using the moved function
            by_conn = _aggregate_from_hdf5_storage(
                f, target, intensity, stim_start, stim_duration, warmup
            )
            
            aggregated['by_connectivity'][target][intensity] = by_conn
            
            # Grand mean: compute from by-connectivity aggregates
            aggregated['grand_mean'][target][intensity] = \
                _compute_grand_mean_from_connectivity_aggregates(by_conn)
            
            # Across connectivity
            aggregated['across_connectivity'][target][intensity] = \
                _aggregate_connectivity_means(by_conn)
    
    return aggregated


def _compute_grand_mean_from_connectivity_aggregates(by_conn: Dict) -> Dict:
    """
    Compute grand mean statistics from connectivity-level aggregates
    with proper hierarchical variance decomposition.
    
    For nested data (connectivity instances x MEC patterns), properly computes:
    - Grand mean (weighted by sample sizes)
    - Total variance (between-connectivity + pooled within-connectivity)
    - Proper standard error accounting for hierarchical structure
    
    Args:
        by_conn: Dict mapping connectivity_idx to aggregated statistics
                 (output from _aggregate_from_hdf5_storage)
    
    Returns:
        Grand mean statistics across all connectivity instances
    """
    if not by_conn:
        return {}
    
    # Get populations from first connectivity instance
    populations = list(next(iter(by_conn.values())).keys())
    
    grand_mean = {}
    
    for pop in populations:
        # Collect statistics across all connectivity instances
        connectivity_means = []
        connectivity_vars = []
        connectivity_ns = []
        
        baseline_rates = []
        stim_rates = []
        
        # For target population (if non-expressing stats available)
        excited_nonexpr_means = []
        excited_nonexpr_vars = []
        mean_change_nonexpr_means = []
        mean_change_nonexpr_vars = []
        n_nonexpr_list = []
        
        for conn_data in by_conn.values():
            pop_data = conn_data[pop]
            
            # Store connectivity-level statistics
            connectivity_means.append(pop_data['excited_mean'])
            connectivity_vars.append(pop_data['excited_std']**2)
            connectivity_ns.append(pop_data['n_trials'])
            
            baseline_rates.append(pop_data['baseline_rate_mean'])
            stim_rates.append(pop_data['stim_rate_mean'])
            
            # Check for non-expressing cell data
            if 'excited_nonexpr_mean' in pop_data:
                excited_nonexpr_means.append(pop_data['excited_nonexpr_mean'])
                excited_nonexpr_vars.append(pop_data.get('excited_nonexpr_std', 0)**2)
                mean_change_nonexpr_means.append(pop_data['mean_change_nonexpr_mean'])
                mean_change_nonexpr_vars.append(pop_data.get('mean_change_nonexpr_std', 0)**2)
                n_nonexpr_list.append(pop_data.get('n_nonexpr_mean', 0))
        
        # Convert to numpy arrays
        connectivity_means = np.array(connectivity_means)
        connectivity_vars = np.array(connectivity_vars)
        connectivity_ns = np.array(connectivity_ns)
        
        # Compute grand mean (weighted by sample size)
        total_n = np.sum(connectivity_ns)
        weights = connectivity_ns / total_n if total_n > 0 else np.ones(len(connectivity_ns)) / len(connectivity_ns)
        grand_mean_value = np.sum(weights * connectivity_means)
        
        # Compute between-connectivity variance (weighted)
        k = len(connectivity_means)
        between_var = np.sum(connectivity_ns * (connectivity_means - grand_mean_value)**2) / (k - 1) if k > 1 else 0.0
        
        # Compute pooled within-connectivity variance
        within_var = np.sum((connectivity_ns - 1) * connectivity_vars) / (total_n - k) if total_n > k else 0.0
        
        # Total variance
        total_var = between_var + within_var
        total_std = np.sqrt(total_var)
        
        # Standard error of the grand mean
        se_grand_mean = np.sqrt(between_var / k + within_var / total_n) if k > 0 and total_n > 0 else 0.0
        
        # Do the same for mean_change
        change_means = []
        change_vars = []
        for conn_data in by_conn.values():
            pop_data = conn_data[pop]
            change_means.append(pop_data['mean_change_mean'])
            change_vars.append(pop_data['mean_change_std']**2)
        
        change_means = np.array(change_means)
        change_vars = np.array(change_vars)
        
        grand_change_mean = np.sum(weights * change_means)
        change_between_var = np.sum(connectivity_ns * (change_means - grand_change_mean)**2) / (k - 1) if k > 1 else 0.0
        change_within_var = np.sum((connectivity_ns - 1) * change_vars) / (total_n - k) if total_n > k else 0.0
        change_total_var = change_between_var + change_within_var
        
        grand_mean[pop] = {
            'excited_mean': grand_mean_value,
            'excited_total_var': total_var,
            'excited_between_var': between_var,
            'excited_within_var': within_var,
            'excited_std': total_std,
            'excited_between_std': np.sqrt(between_var),
            'excited_within_std': np.sqrt(within_var),
            'excited_sem': se_grand_mean,
            'mean_change_mean': grand_change_mean,
            'mean_change_total_var': change_total_var,
            'mean_change_std': np.sqrt(change_total_var),
            'baseline_rate_mean': np.sum(weights * np.array(baseline_rates)),
            'stim_rate_mean': np.sum(weights * np.array(stim_rates)),
            'n_trials': int(total_n),
            'n_connectivity_instances': k
        }
        
        # Add non-expressing cell statistics if available
        if excited_nonexpr_means:
            excited_nonexpr_means = np.array(excited_nonexpr_means)
            excited_nonexpr_vars = np.array(excited_nonexpr_vars)
            
            grand_nonexpr_mean = np.sum(weights * excited_nonexpr_means)
            nonexpr_between_var = np.sum(connectivity_ns * (excited_nonexpr_means - grand_nonexpr_mean)**2) / (k - 1) if k > 1 else 0.0
            nonexpr_within_var = np.sum((connectivity_ns - 1) * excited_nonexpr_vars) / (total_n - k) if total_n > k else 0.0
            nonexpr_total_var = nonexpr_between_var + nonexpr_within_var
            
            grand_mean[pop].update({
                'excited_nonexpr_mean': grand_nonexpr_mean,
                'excited_nonexpr_std': np.sqrt(nonexpr_total_var),
                'excited_nonexpr_sem': np.sqrt(nonexpr_between_var / k + nonexpr_within_var / total_n) if k > 0 and total_n > 0 else 0.0,
                'mean_change_nonexpr_mean': np.sum(weights * np.array(mean_change_nonexpr_means)),
                'mean_change_nonexpr_std': np.sqrt(np.sum(connectivity_ns * (np.array(mean_change_nonexpr_means) - np.sum(weights * np.array(mean_change_nonexpr_means)))**2) / (k - 1) if k > 1 else 0.0 + 
                                                   np.sum((connectivity_ns - 1) * np.array(mean_change_nonexpr_vars)) / (total_n - k) if total_n > k else 0.0),
                'n_nonexpr_mean': np.mean(n_nonexpr_list)
            })
    
    return grand_mean


# ============================================================================
# Variance Decomposition
# ============================================================================

def compute_variance_decomposition(
    nested_results: Dict,
    config: NestedExperimentConfig,
    stim_start: float,
    stim_duration: float,
    warmup: float
) -> Dict:
    """
    Decompose variance into connectivity-driven and input-driven components
    
    Uses hierarchical ANOVA framework:
        Total variance = Var(between connectivity) + Var(within connectivity)
        
    Where:
        Var(between connectivity) = variance of connectivity means
        Var(within connectivity) = mean of within-connectivity variances
        
    Returns:
        Dict with variance components and statistical tests
    """
    variance_analysis = {}
    
    for target in ['pv', 'sst']:
        variance_analysis[target] = {}
        
        for intensity, trials in nested_results[target].items():
            # Group by connectivity
            conn_groups = {}
            for trial in trials:
                conn_idx = trial.connectivity_idx
                if conn_idx not in conn_groups:
                    conn_groups[conn_idx] = []
                conn_groups[conn_idx].append(trial)
            
            # Analyze for each population
            pop_variance = {}
            for pop in ['gc', 'mc', 'pv', 'sst']:
                pop_variance[pop] = _decompose_population_variance(
                    conn_groups,
                    pop,
                    stim_start,
                    stim_duration,
                    warmup
                )
            
            variance_analysis[target][intensity] = pop_variance
    
    return variance_analysis


def _decompose_population_variance(
    conn_groups: Dict[int, List[NestedTrialResult]],
    population: str,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    expression_threshold: float = 0.2
) -> Dict:
    """
    Decompose variance for a single population
    
    For the target population (the one being stimulated), analyzes both
    all cells and non-expressing cells separately.
    
    Returns variance components and effect sizes
    """
    # Collect data organized by connectivity
    time = conn_groups[0][0].results['time']
    baseline_mask = (time >= warmup) & (time < stim_start)
    stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
    
    # Check if this is the target population
    target_population = conn_groups[0][0].target_population
    is_target = (population == target_population)
    
    # Store excited fractions by connectivity
    excited_by_conn = []
    change_by_conn = []
    
    # For target population, also track non-expressing cells
    if is_target:
        excited_nonexpr_by_conn = []
        change_nonexpr_by_conn = []
    
    for conn_idx in sorted(conn_groups.keys()):
        conn_trials = conn_groups[conn_idx]
        
        excited_vals = []
        change_vals = []
        
        if is_target:
            excited_nonexpr_vals = []
            change_nonexpr_vals = []
        
        for trial in conn_trials:
            # Handle both aggregated and raw results
            if 'activity_trace_mean' in trial.results:
                activity = trial.results['activity_trace_mean'][population]
            elif 'activity_trace' in trial.results:
                activity = trial.results['activity_trace'][population]
            else:
                raise KeyError(f"Trial results missing both 'activity_trace' and 'activity_trace_mean'")
            
            baseline_rate = torch.mean(activity[:, baseline_mask], dim=1)
            stim_rate = torch.mean(activity[:, stim_mask], dim=1)
            rate_change = stim_rate - baseline_rate
            baseline_std = torch.std(baseline_rate)
            
            # All cells
            excited_fraction = torch.mean((rate_change > baseline_std).float()).item()
            mean_change = torch.mean(rate_change).item()
            
            excited_vals.append(excited_fraction)
            change_vals.append(mean_change)
            
            # Non-expressing cells (for target population only)
            if is_target:
                opsin_expression = trial.opsin_expression
                non_expressing_mask = opsin_expression < expression_threshold
                n_nonexpr = np.sum(non_expressing_mask)
                
                if n_nonexpr > 0:
                    if not isinstance(non_expressing_mask, torch.Tensor):
                        non_expressing_mask = torch.from_numpy(non_expressing_mask)
                    
                    baseline_rate_nonexpr = baseline_rate[non_expressing_mask]
                    stim_rate_nonexpr = stim_rate[non_expressing_mask]
                    rate_change_nonexpr = rate_change[non_expressing_mask]
                    baseline_std_nonexpr = torch.std(baseline_rate_nonexpr)
                    
                    excited_nonexpr = torch.mean(
                        (rate_change_nonexpr > baseline_std_nonexpr).float()
                    ).item()
                    change_nonexpr = torch.mean(rate_change_nonexpr).item()
                    
                    excited_nonexpr_vals.append(excited_nonexpr)
                    change_nonexpr_vals.append(change_nonexpr)
                else:
                    excited_nonexpr_vals.append(0.0)
                    change_nonexpr_vals.append(0.0)
        
        excited_by_conn.append(excited_vals)
        change_by_conn.append(change_vals)
        
        if is_target:
            excited_nonexpr_by_conn.append(excited_nonexpr_vals)
            change_nonexpr_by_conn.append(change_nonexpr_vals)
    
    # Compute variance components for excited fraction (all cells)
    excited_variance = _hierarchical_variance_decomposition(excited_by_conn)
    
    # Compute variance components for mean change (all cells)
    change_variance = _hierarchical_variance_decomposition(change_by_conn)
    
    # Compute intraclass correlation coefficient (ICC)
    excited_icc = (excited_variance['between_group_var'] / 
                   (excited_variance['between_group_var'] + excited_variance['within_group_var'])
                   if (excited_variance['between_group_var'] + excited_variance['within_group_var']) > 0
                   else 0.0)
    
    change_icc = (change_variance['between_group_var'] /
                  (change_variance['between_group_var'] + change_variance['within_group_var'])
                  if (change_variance['between_group_var'] + change_variance['within_group_var']) > 0
                  else 0.0)
    
    result = {
        'excited_fraction': {
            **excited_variance,
            'icc': excited_icc
        },
        'mean_change': {
            **change_variance,
            'icc': change_icc
        }
    }
    
    # Add non-expressing cell analysis for target population
    if is_target:
        excited_nonexpr_variance = _hierarchical_variance_decomposition(excited_nonexpr_by_conn)
        change_nonexpr_variance = _hierarchical_variance_decomposition(change_nonexpr_by_conn)
        
        excited_nonexpr_icc = (
            excited_nonexpr_variance['between_group_var'] / 
            (excited_nonexpr_variance['between_group_var'] + excited_nonexpr_variance['within_group_var'])
            if (excited_nonexpr_variance['between_group_var'] + excited_nonexpr_variance['within_group_var']) > 0
            else 0.0
        )
        
        result['excited_fraction_nonexpr'] = {
            **excited_nonexpr_variance,
            'icc': excited_nonexpr_icc
        }
        result['mean_change_nonexpr'] = {
            **change_nonexpr_variance,
            'icc': (change_nonexpr_variance['between_group_var'] /
                   (change_nonexpr_variance['between_group_var'] + change_nonexpr_variance['within_group_var'])
                   if (change_nonexpr_variance['between_group_var'] + change_nonexpr_variance['within_group_var']) > 0
                   else 0.0)
        }
    
    return result


def _decompose_population_variance_from_stats(
    conn_groups_stats: Dict,
    population: str,
    target_population: str
) -> Dict:
    """
    Decompose variance using pre-computed per-trial statistics
    
    This version works with summary statistics stored in HDF5,
    avoiding the need to reload full activity traces.
    
    Args:
        conn_groups_stats: Dict mapping conn_idx to list of trial statistics
                          (output from extract_trial_statistics_from_hdf5)
        population: Population to analyze ('gc', 'mc', 'pv', 'sst')
        target_population: Which population was stimulated
        
    Returns:
        Variance components and ICC values
    """
    is_target = (population == target_population)
    
    # Collect data organized by connectivity
    excited_by_conn = []
    change_by_conn = []
    
    # For target population, also track non-expressing cells
    if is_target:
        excited_nonexpr_by_conn = []
        change_nonexpr_by_conn = []
    
    for conn_idx in sorted(conn_groups_stats.keys()):
        conn_trials = conn_groups_stats[conn_idx]
        
        excited_vals = []
        change_vals = []
        
        if is_target:
            excited_nonexpr_vals = []
            change_nonexpr_vals = []
        
        for trial_stats in conn_trials:
            # Check if this population has data
            if population not in trial_stats['populations']:
                continue
            
            pop_stats = trial_stats['populations'][population]
            
            # All cells statistics
            excited_vals.append(pop_stats['excited'])
            change_vals.append(pop_stats['mean_change'])
            
            # Non-expressing cells (for target population only)
            if is_target and 'excited_nonexpr' in pop_stats:
                excited_nonexpr_vals.append(pop_stats['excited_nonexpr'])
                change_nonexpr_vals.append(pop_stats['mean_change_nonexpr'])
        
        if excited_vals:  # Only add if we have data
            excited_by_conn.append(excited_vals)
            change_by_conn.append(change_vals)
            
            if is_target and excited_nonexpr_vals:
                excited_nonexpr_by_conn.append(excited_nonexpr_vals)
                change_nonexpr_by_conn.append(change_nonexpr_vals)
    
    # Check if we have any data
    if not excited_by_conn:
        return {
            'excited_fraction': {
                'total_var': 0.0,
                'between_group_var': 0.0,
                'within_group_var': 0.0,
                'icc': 0.0,
                'f_statistic': np.nan,
                'p_value': np.nan,
                'n_groups': 0,
                'n_total': 0,
                'test_method': 'no_data'
            },
            'mean_change': {
                'total_var': 0.0,
                'between_group_var': 0.0,
                'within_group_var': 0.0,
                'icc': 0.0,
                'f_statistic': np.nan,
                'p_value': np.nan,
                'n_groups': 0,
                'n_total': 0,
                'test_method': 'no_data'
            }
        }
    
    # Compute variance components
    excited_variance = _hierarchical_variance_decomposition(excited_by_conn)
    change_variance = _hierarchical_variance_decomposition(change_by_conn)
    
    # Compute ICC
    excited_icc = (excited_variance['between_group_var'] / 
                   (excited_variance['between_group_var'] + excited_variance['within_group_var'])
                   if (excited_variance['between_group_var'] + excited_variance['within_group_var']) > 0
                   else 0.0)
    
    change_icc = (change_variance['between_group_var'] /
                  (change_variance['between_group_var'] + change_variance['within_group_var'])
                  if (change_variance['between_group_var'] + change_variance['within_group_var']) > 0
                  else 0.0)
    
    result = {
        'excited_fraction': {
            **excited_variance,
            'icc': excited_icc
        },
        'mean_change': {
            **change_variance,
            'icc': change_icc
        }
    }
    
    # Add non-expressing cell analysis for target population
    if is_target and excited_nonexpr_by_conn:
        excited_nonexpr_variance = _hierarchical_variance_decomposition(excited_nonexpr_by_conn)
        change_nonexpr_variance = _hierarchical_variance_decomposition(change_nonexpr_by_conn)
        
        excited_nonexpr_icc = (
            excited_nonexpr_variance['between_group_var'] / 
            (excited_nonexpr_variance['between_group_var'] + excited_nonexpr_variance['within_group_var'])
            if (excited_nonexpr_variance['between_group_var'] + excited_nonexpr_variance['within_group_var']) > 0
            else 0.0
        )
        
        result['excited_fraction_nonexpr'] = {
            **excited_nonexpr_variance,
            'icc': excited_nonexpr_icc
        }
        result['mean_change_nonexpr'] = {
            **change_nonexpr_variance,
            'icc': (change_nonexpr_variance['between_group_var'] /
                   (change_nonexpr_variance['between_group_var'] + change_nonexpr_variance['within_group_var'])
                   if (change_nonexpr_variance['between_group_var'] + change_nonexpr_variance['within_group_var']) > 0
                   else 0.0)
        }
    
    return result


def compute_variance_from_hdf5(f: h5py.File,
                               target: str,
                               intensity: float,
                               stim_start: float,
                               stim_duration: float,
                               warmup: float) -> Dict:
    """
    Compute variance decomposition directly from HDF5
    
    Memory-efficient version that uses pre-computed per-trial statistics
    instead of loading full activity traces.
    """
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in f[target]:
        return {}
    
    # Extract metadata to determine dimensions
    metadata = dict(f['metadata'].attrs)
    n_connectivity = metadata['n_connectivity_instances']
    n_mec_patterns = metadata['n_mec_patterns_per_connectivity']
    
    # Extract per-trial statistics (I/O operation)
    conn_groups_stats = extract_trial_statistics_from_hdf5(
        f, target, intensity, n_connectivity, n_mec_patterns
    )
    
    # Compute variance for each population (analysis operation)
    pop_variance = {}
    for pop in ['gc', 'mc', 'pv', 'sst']:
        pop_variance[pop] = _decompose_population_variance_from_stats(
            conn_groups_stats, pop, target
        )
    
    return pop_variance


def compute_variance_decomposition_from_hdf5(
    f: h5py.File,
    config: NestedExperimentConfig,
    stim_start: float,
    stim_duration: float,
    warmup: float
) -> Dict:
    """
    Compute variance decomposition from HDF5 file
    
    Memory-efficient version.
    """
    variance_analysis = {}
    
    for target in ['pv', 'sst']:
        variance_analysis[target] = {}
        
        # Get intensities
        intensities = [float(k.split('_')[1]) for k in f[target].keys()
                      if k.startswith('intensity_')]
        
        for intensity in intensities:
            pop_variance = compute_variance_from_hdf5(
                f, target, intensity, stim_start, stim_duration, warmup
            )
            
            variance_analysis[target][intensity] = pop_variance
    
    return variance_analysis


def _hierarchical_variance_decomposition(grouped_data: List[List[float]],
                                         permutation_random_seed=47) -> Dict:
    """
    Compute hierarchical variance decomposition with robust statistical testing.
    
    Uses permutation test as primary method (better for small samples and discrete data),
    with fallback to F-test and explicit handling of edge cases.
    
    Args:
        grouped_data: List of lists, where each inner list is data from one group
        
    Returns:
        Dict with variance components and test results
    """
    from scipy import stats
    
    # Flatten for grand mean
    all_data = [val for group in grouped_data for val in group]
    grand_mean = np.mean(all_data)
    
    # Group means and sizes
    group_means = [np.mean(group) for group in grouped_data]
    group_sizes = [len(group) for group in grouped_data]
    
    # Total variance
    total_var = np.var(all_data, ddof=1) if len(all_data) > 1 else 0.0
    
    # Between-group variance
    n_total = sum(group_sizes)
    if len(group_means) > 1:
        between_group_var = sum(
            n * (mean - grand_mean)**2 
            for n, mean in zip(group_sizes, group_means)
        ) / (len(group_means) - 1)
    else:
        between_group_var = 0.0
    
    # Within-group variance
    within_group_var = np.mean([
        np.var(group, ddof=1) if len(group) > 1 else 0.0
        for group in grouped_data
    ])
    
    # Statistical test
    if len(grouped_data) <= 1:
        f_statistic = np.nan
        p_value = np.nan
        test_method = 'insufficient_groups'
        
    elif within_group_var < 1e-10:
        # Zero or near-zero within-group variance
        group_means_array = np.array(group_means)
        unique_means = np.unique(group_means_array)
        
        if len(unique_means) > 1:
            f_statistic = np.inf
            p_value = 0.0
            test_method = 'deterministic_separation'
        else:
            f_statistic = 0.0
            p_value = 1.0
            test_method = 'no_variation'
    
    else:
        # Normal case: use permutation test
        try:
            def test_statistic(*groups):
                means = [np.mean(g) for g in groups]
                return np.var(means, ddof=1)
            
            from scipy.stats import permutation_test
            
            result = permutation_test(
                grouped_data,
                test_statistic,
                permutation_type='independent',
                n_resamples=10000,
                random_state=permutation_random_seed
            )
            
            p_value = result.pvalue
            f_statistic = test_statistic(*grouped_data)
            test_method = 'permutation'
            
        except Exception as e:
            # Fallback to parametric F-test
            try:
                f_statistic, p_value = stats.f_oneway(*grouped_data)
                test_method = 'anova_fallback'
            except Exception as e2:
                f_statistic = np.nan
                p_value = np.nan
                test_method = f'failed: {str(e2)[:50]}'
    
    return {
        'total_var': total_var,
        'between_group_var': between_group_var,
        'within_group_var': within_group_var,
        'f_statistic': f_statistic,
        'p_value': p_value,
        'n_groups': len(grouped_data),
        'n_total': n_total,
        'test_method': test_method
    }


# ============================================================================
# Regime Classification
# ============================================================================

def classify_mechanism_regime(
    variance_analysis: Dict,
    config: NestedExperimentConfig
) -> Dict:
    """
    Classify mechanism regime with robust handling of edge cases
    """
    # Classification thresholds
    ICC_CONNECTIVITY_THRESHOLD = 0.5
    ICC_INPUT_THRESHOLD = 0.2
    P_VALUE_THRESHOLD = config.variance_significance_alpha
    
    classification = {}
    
    for target in variance_analysis.keys():
        classification[target] = {}
        
        for intensity, pop_analysis in variance_analysis[target].items():
            classification[target][intensity] = {}
            
            for pop, variance_data in pop_analysis.items():
                excited_data = variance_data['excited_fraction']
                excited_icc = excited_data['icc']
                excited_p = excited_data['p_value']
                test_method = excited_data.get('test_method', 'unknown')
                
                # Classify based on test results
                if np.isnan(excited_p) or np.isnan(excited_icc):
                    regime = 'insufficient_data'
                    
                elif test_method == 'deterministic_separation':
                    regime = 'connectivity_driven'
                    
                elif test_method == 'no_variation':
                    regime = 'no_significant_variance'
                    
                elif excited_p > P_VALUE_THRESHOLD:
                    regime = 'no_significant_variance'
                    
                elif excited_icc > ICC_CONNECTIVITY_THRESHOLD:
                    regime = 'connectivity_driven'
                    
                elif excited_icc < ICC_INPUT_THRESHOLD:
                    regime = 'input_driven'
                    
                else:
                    regime = 'mixed'
                
                classification[target][intensity][pop] = {
                    'regime': regime,
                    'excited_icc': excited_icc,
                    'excited_p_value': excited_p,
                    'test_method': test_method, 
                    'interpretation': _get_regime_interpretation(regime, excited_icc, test_method)
                }
    
    return classification


def _get_regime_interpretation(regime: str, icc: float, test_method: str = '') -> str:
    """Generate interpretation text with test method context"""
    
    method_note = ""
    if test_method == 'deterministic_separation':
        method_note = " (deterministic: zero within-group variance)"
    elif test_method == 'permutation':
        method_note = " (permutation test)"
    elif test_method == 'anova_fallback':
        method_note = " (ANOVA F-test)"
    
    interpretations = {
        'connectivity_driven': (
            f"High ICC ({icc:.3f}) indicates paradoxical excitation is primarily "
            "determined by specific synaptic weight patterns. Different circuit "
            f"instantiations show distinct response profiles{method_note}."
        ),
        'input_driven': (
            f"Low ICC ({icc:.3f}) indicates paradoxical excitation is primarily "
            "determined by population-level dynamics. The same connectivity can "
            f"produce different responses depending on input patterns{method_note}."
        ),
        'mixed': (
            f"Moderate ICC ({icc:.3f}) indicates both connectivity patterns and "
            f"population dynamics contribute to paradoxical excitation{method_note}."
        ),
        'no_significant_variance': (
            "No significant variance between connectivity instances. "
            f"Effect may be too small or sample size insufficient{method_note}."
        ),
        'insufficient_data': (
            "Insufficient data for regime classification."
        )
    }
    return interpretations.get(regime, "Unknown regime")


# ============================================================================
# HDF5 Save/Load Utilities
# ============================================================================

def _save_analysis_to_hdf5(f: h5py.File,
                           aggregated_results: Dict,
                           variance_analysis: Dict,
                           regime_classification: Dict) -> None:
    """Save aggregated analysis results to HDF5 file"""
    # Create analysis group
    if 'analysis' in f:
        del f['analysis']
    
    analysis_grp = f.create_group('analysis')
    
    # Save regime classification
    regime_grp = analysis_grp.create_group('regime_classification')
    for target in regime_classification.keys():
        target_grp = regime_grp.create_group(target)
        for intensity, pop_data in regime_classification[target].items():
            intensity_grp = target_grp.create_group(f'intensity_{intensity}')
            for pop, regime_data in pop_data.items():
                pop_grp = intensity_grp.create_group(pop)
                for key, value in regime_data.items():
                    if isinstance(value, str):
                        pop_grp.attrs[key] = value
                    else:
                        pop_grp.attrs[key] = value if value is not None else np.nan
    
    # Save variance analysis
    var_grp = analysis_grp.create_group('variance_analysis')
    for target in variance_analysis.keys():
        target_grp = var_grp.create_group(target)
        for intensity, pop_data in variance_analysis[target].items():
            intensity_grp = target_grp.create_group(f'intensity_{intensity}')
            for pop, var_data in pop_data.items():
                pop_grp = intensity_grp.create_group(pop)
                
                # Flatten nested dict structure
                def save_nested_dict(grp, data, prefix=''):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            save_nested_dict(grp, value, f'{prefix}{key}_')
                        else:
                            grp.attrs[f'{prefix}{key}'] = value if value is not None else np.nan
                
                save_nested_dict(pop_grp, var_data)
    
    f.flush()


def load_nested_experiment_from_hdf5(filepath: str) -> Dict:
    """Load complete nested experiment results from HDF5 file"""
    from dataclasses import asdict
    
    metadata = load_metadata_from_hdf5(filepath)
    
    # Open file for reading
    with h5py.File(filepath, 'r') as f:
        # Load analysis results
        analysis_grp = f.get('analysis')
        
        if analysis_grp is not None:
            # Load regime classification
            regime_classification = {}
            regime_grp = analysis_grp['regime_classification']
            for target in regime_grp.keys():
                regime_classification[target] = {}
                for intensity_key in regime_grp[target].keys():
                    intensity = float(intensity_key.split('_')[1])
                    regime_classification[target][intensity] = {}
                    
                    for pop in regime_grp[target][intensity_key].keys():
                        regime_data = dict(regime_grp[target][intensity_key][pop].attrs)
                        regime_classification[target][intensity][pop] = regime_data
            
            # Load variance analysis
            variance_analysis = {}
            var_grp = analysis_grp['variance_analysis']
            for target in var_grp.keys():
                variance_analysis[target] = {}
                for intensity_key in var_grp[target].keys():
                    intensity = float(intensity_key.split('_')[1])
                    variance_analysis[target][intensity] = {}
                    
                    for pop in var_grp[target][intensity_key].keys():
                        var_data_flat = dict(var_grp[target][intensity_key][pop].attrs)
                        
                        # Reconstruct nested structure
                        var_data = {}
                        for key, value in var_data_flat.items():
                            parts = key.split('_')
                            if len(parts) >= 2:
                                metric = parts[0]
                                subkey = '_'.join(parts[1:])
                                
                                if metric not in var_data:
                                    var_data[metric] = {}
                                var_data[metric][subkey] = value
                            else:
                                var_data[key] = value
                        
                        variance_analysis[target][intensity][pop] = var_data
        else:
            regime_classification = None
            variance_analysis = None
        
        # Reconstruct config
        config = NestedExperimentConfig(
            n_connectivity_instances=metadata['n_connectivity_instances'],
            n_mec_patterns_per_connectivity=metadata['n_mec_patterns_per_connectivity'],
            base_seed=metadata.get('seed_structure', {}).get('connectivity_seeds', [42])[0]
        )
    
    return {
        'nested_results': None,
        'aggregated_results': None,
        'variance_analysis': variance_analysis,
        'regime_classification': regime_classification,
        'config': config,
        'seed_structure': metadata.get('seed_structure'),
        'metadata': metadata,
        'hdf5_file': filepath
    }


def save_nested_experiment_results(results: Dict, filepath: str):
    """Save nested experiment results with compression"""
    import pickle
    from datetime import datetime
    
    results['timestamp'] = datetime.now().isoformat()
    results['version'] = '1.0_nested'
    
    # Convert tensors to numpy
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        elif isinstance(obj, NestedTrialResult):
            return NestedTrialResult(
                connectivity_idx=obj.connectivity_idx,
                mec_pattern_idx=obj.mec_pattern_idx,
                seed=obj.seed,
                results=convert_tensors(obj.results),
                target_population=obj.target_population,
                opsin_expression=convert_tensors(obj.opsin_expression)
            )
        else:
            return obj
    
    results_converted = convert_tensors(results)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results_converted, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"\nNested experiment results saved to: {filepath}")
    logger.info(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def print_nested_experiment_summary(results: Dict):
    """Print summary of nested experiment results"""
    config = results['config']
    variance_analysis = results['variance_analysis']
    regime_classification = results['regime_classification']
    
    logger.info("\n" + "="*80)
    logger.info("Nested experiment summary")
    logger.info("="*80)
    
    logger.info(f"\nExperimental Design:")
    logger.info(f"  Connectivity instances: {config.n_connectivity_instances}")
    logger.info(f"  MEC patterns per connectivity: {config.n_mec_patterns_per_connectivity}")
    logger.info(f"  Total trials per condition: {config.n_connectivity_instances * config.n_mec_patterns_per_connectivity}")
    
    for target in ['pv', 'sst']:
        logger.info(f"\n{target.upper()} Stimulation:")
        logger.info("-"*60)
        
        for intensity in sorted(variance_analysis[target].keys()):
            logger.info(f"\n  Intensity {intensity}:")
            
            for pop in ['gc', 'mc', 'pv', 'sst']:
                var_data = variance_analysis[target][intensity][pop]
                regime_data = regime_classification[target][intensity][pop]
                
                excited_icc = var_data['excited_fraction']['icc']
                excited_p = var_data['excited_fraction']['p_value']
                
                logger.info(f"\n    {pop.upper()}:")
                logger.info(f"      ICC (excited fraction): {excited_icc:.3f}")
                logger.info(f"      ANOVA p-value: {excited_p:.4f}")
                logger.info(f"      Regime: {regime_data['regime']}")
                logger.info(f"      {regime_data['interpretation']}")
    
    logger.info("\n" + "="*80)


def save_nested_experiment_summary(results: Dict, output_dir: str):
    """
    Save nested experiment results summary to markdown file
    
    Extracts key findings about regime classification and variance decomposition
    and writes them to 'nested_experiment_summary.md' in the specified directory.
    
    Args:
        results: Complete nested experiment results dictionary
        output_dir: Directory to save the summary file
    """
    config = results['config']
    variance_analysis = results['variance_analysis']
    regime_classification = results['regime_classification']
    
    # Build summary content
    content = f"""# Nested Experiment Results

## Experimental Design
- Connectivity instances: {config.n_connectivity_instances}
- MEC patterns per connectivity: {config.n_mec_patterns_per_connectivity}
- Total trials per condition: {config.n_connectivity_instances * config.n_mec_patterns_per_connectivity}
- Base seed: {config.base_seed}

## Key Findings

"""
    
    for target in ['pv', 'sst']:
        content += f"\n### {target.upper()} Stimulation\n\n"
        
        for intensity in sorted(variance_analysis[target].keys()):
            content += f"#### Intensity {intensity}\n\n"
            content += "| Population | ICC | Regime | Between Var | Within Var | p-value |\n"
            content += "|-----------|-----|--------|-------------|------------|----------|\n"
            
            for pop in ['gc', 'mc', 'pv', 'sst']:
                var_data = variance_analysis[target][intensity][pop]['excited_fraction']
                regime_data = regime_classification[target][intensity][pop]
                
                content += (f"| {pop.upper()} | {var_data['icc']:.3f} | "
                          f"{regime_data['regime']} | {var_data['between_group_var']:.6f} | "
                          f"{var_data['within_group_var']:.6f} | "
                          f"{var_data['p_value']:.4f} |\n")
            
            content += "\n"
    
    # Add interpretations
    content += "\n## Mechanistic Interpretations\n\n"
    
    for target in ['pv', 'sst']:
        for intensity in sorted(variance_analysis[target].keys()):
            for pop in ['gc', 'mc']:
                regime_data = regime_classification[target][intensity][pop]
                
                if regime_data['regime'] in ['connectivity_driven', 'input_driven']:
                    content += f"\n**{target.upper()} → {pop.upper()} (intensity {intensity}):**\n"
                    content += f"{regime_data['interpretation']}\n"
    
    # Write to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / "nested_experiment_summary.md"
    with open(summary_file, 'w') as f:
        f.write(content)
    
    logger.info(f"\nSaved nested experiment summary to: {summary_file}")


def plot_variance_decomposition(variance_analysis: Dict,
                                regime_classification: Dict,
                                save_path: Optional[str] = None):
    """
    Visualize variance decomposition and regime classification
    
    Creates a multi-panel figure showing:
    - ICC values for each population
    - Variance components (between vs. within connectivity)
    - Regime classification
    - Statistical significance
    """
    fig, axes = plt.subplots(2, 5, figsize=(18, 10))  # Extra column for non-expressing
    
    # Define colors for regimes
    regime_colors = {
        'connectivity_driven': '#e74c3c',
        'input_driven': '#3498db',
        'mixed': '#f39c12',
        'no_significant_variance': '#95a5a6',
        'insufficient_data': '#ecf0f1'
    }
    
    targets = list(variance_analysis.keys())
    
    for target_idx, target in enumerate(targets):
        intensities = sorted(variance_analysis[target].keys())
        intensity = intensities[-1]
        
        var_data = variance_analysis[target][intensity]
        regime_data = regime_classification[target][intensity]
        
        # Standard populations
        populations = ['gc', 'mc', 'pv', 'sst']
        
        # Panel A: ICC values (columns 0-3)
        for pop_idx, pop in enumerate(populations):
            ax_icc = axes[target_idx, pop_idx]
            
            icc_value = var_data[pop]['excited_fraction']['icc']
            regime = regime_data[pop]['regime']
            color = regime_colors[regime]
            
            bars = ax_icc.bar([0], [icc_value], color=color,
                             alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add significance marker
            p_val = var_data[pop]['excited_fraction']['p_value']
            if not np.isnan(p_val):
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = 'n.s.'
                
                ax_icc.text(0, icc_value + 0.05, marker,
                           ha='center', fontsize=12, fontweight='bold')
            
            # Add threshold lines
            if pop_idx == 0:  # Only on first panel
                ax_icc.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                              label='Connectivity threshold')
                ax_icc.axhline(y=0.2, color='blue', linestyle='--', alpha=0.5,
                              label='Input threshold')
                ax_icc.legend(fontsize=8, loc='upper right')
            
            ax_icc.set_xlim(-0.5, 0.5)
            ax_icc.set_xticks([])
            
            # Color-code title if this is the target population
            if pop == target:
                title_color = 'darkred'
                title_text = f'{pop.upper()}\n(all cells)'
            else:
                title_color = 'black'
                title_text = pop.upper()
            
            if target_idx == 0:
                ax_icc.set_title(title_text, fontsize=11, 
                               fontweight='bold', color=title_color)
            
            if pop_idx == 0:
                ax_icc.set_ylabel(f'{target.upper()} Stim\nICC', fontsize=10)
            
            ax_icc.set_ylim(0, 1.0)
            ax_icc.grid(True, alpha=0.3, axis='y')
        
        # Panel B: Non-expressing cells (column 4)
        ax_nonexpr = axes[target_idx, 4]
        
        # Check if non-expressing data exists
        if 'excited_fraction_nonexpr' in var_data[target]:
            nonexpr_data = var_data[target]['excited_fraction_nonexpr']
            icc_nonexpr = nonexpr_data['icc']
            
            # Use purple color for non-expressing
            bar = ax_nonexpr.bar([0], [icc_nonexpr], color='purple',
                                alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add significance marker
            p_val = nonexpr_data['p_value']
            if not np.isnan(p_val):
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = 'n.s.'
                
                ax_nonexpr.text(0, icc_nonexpr + 0.05, marker,
                              ha='center', fontsize=12, fontweight='bold')
            
            ax_nonexpr.set_xlim(-0.5, 0.5)
            ax_nonexpr.set_xticks([])
            
            if target_idx == 0:
                ax_nonexpr.set_title(f'{target.upper()}\n(non-expr)', 
                                    fontsize=11, fontweight='bold', color='purple')
            
            ax_nonexpr.set_ylim(0, 1.0)
            ax_nonexpr.grid(True, alpha=0.3, axis='y')
        else:
            # No non-expressing data
            ax_nonexpr.text(0.5, 0.5, 'No data',
                          ha='center', va='center', transform=ax_nonexpr.transAxes,
                          fontsize=10, style='italic')
            ax_nonexpr.axis('off')
    
    # Update legend to include non-expressing explanation
    legend_elements = [
        mpatches.Patch(color=regime_colors['connectivity_driven'], 
                      label='Connectivity-driven (ICC > 0.5)'),
        mpatches.Patch(color=regime_colors['input_driven'],
                      label='Input-driven (ICC < 0.2)'),
        mpatches.Patch(color=regime_colors['mixed'],
                      label='Mixed (0.2 <= ICC <= 0.5)'),
        mpatches.Patch(color='purple',
                      label='Non-expressing cells (target pop)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=2, fontsize=10, frameon=True)
    
    plt.suptitle('Variance Decomposition: Connectivity vs. Input Patterns\n'
                 '(Non-expressing cells = true paradoxical excitation)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved variance decomposition plot to: {save_path}")
    
    plt.show()
    return fig    


def plot_connectivity_instance_variance(
    aggregated_results: Dict,
    target_populations: list = ['pv', 'sst'],
    post_populations: list = ['gc', 'mc', 'pv', 'sst'],
    intensity: float = 1.5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot mean +/- std of excited/suppressed fractions for each connectivity instance
    
    This visualization shows:
    - X-axis: Connectivity instance number
    - Y-axis: Fraction of excited (or suppressed) cells
    - Error bars: Standard deviation across MEC input patterns
    - Separate panels for each target x post-synaptic population combination
    
    The plot reveals:
    - Between-connectivity variance: Differences in means across instances
    - Within-connectivity variance: Error bar sizes (variation across input patterns)
    
    Args:
        aggregated_results: Output from aggregate_nested_results()
        target_populations: Populations that were stimulated (e.g., ['pv', 'sst'])
        post_populations: Post-synaptic populations to plot (e.g., ['gc', 'mc'])
        intensity: Light intensity to plot
        save_path: Optional path to save figure
        
    Returns:
        Figure object
        
    Example:
        >>> aggregated = results['aggregated_results']
        >>> fig = plot_connectivity_instance_variance(
        ...     aggregated,
        ...     save_path='protocol/connectivity_variance.pdf'
        ... )
    """
    
    n_targets = len(target_populations)
    n_posts = len(post_populations)
    
    fig, axes = plt.subplots(n_targets, n_posts, figsize=(6 * n_posts, 5 * n_targets),
                             squeeze=False)
    
    excited_color = '#e74c3c'  # Red
    suppressed_color = '#3498db'  # Blue
    
    for target_idx, target in enumerate(target_populations):
        for post_idx, post_pop in enumerate(post_populations):
            ax = axes[target_idx, post_idx]
            
            # Get data for this target/intensity
            by_conn = aggregated_results['by_connectivity'][target][intensity]
            
            if not by_conn:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                continue
            
            conn_indices = sorted(by_conn.keys())
            n_conn = len(conn_indices)
            
            excited_means = []
            excited_stds = []
            suppressed_means = []
            suppressed_stds = []
            
            # Check if this is the target population
            is_target_population = (post_pop == target)
            
            for conn_idx in conn_indices:
                conn_data = by_conn[conn_idx][post_pop]
                
                # Use different keys based on whether this is target population
                if is_target_population:
                    # For target population, use non-expressing cell statistics
                    excited_key = 'excited_nonexpr_mean'
                    excited_std_key = 'excited_nonexpr_std'
                    inhibited_key = 'inhibited_nonexpr_mean'
                    inhibited_std_key = 'inhibited_nonexpr_std'
                else:
                    # For other populations, use standard statistics
                    excited_key = 'excited_mean'
                    excited_std_key = 'excited_std'
                    inhibited_key = 'inhibited_mean'
                    inhibited_std_key = 'inhibited_std'
                
                # Get excited fraction statistics
                excited_means.append(conn_data.get(excited_key, 0.0))
                excited_stds.append(conn_data.get(excited_std_key, 0.0))
                
                # Get suppressed/inhibited fraction statistics
                if inhibited_key in conn_data:
                    suppressed_means.append(conn_data[inhibited_key])
                    suppressed_stds.append(conn_data.get(inhibited_std_key, 0.0))
                else:
                    suppressed_means.append(0.0)
                    suppressed_stds.append(0.0)
            
            x_pos = np.arange(n_conn)
            width = 0.35
            
            # Plot excited cells
            ax.bar(x_pos - width/2, excited_means, width,
                  yerr=excited_stds,
                  label='Excited', color=excited_color, alpha=0.7,
                  edgecolor='black', linewidth=1.5,
                  capsize=5, error_kw={'linewidth': 2})
            
            # Plot suppressed cells if data available
            if any(s > 0 for s in suppressed_means):
                ax.bar(x_pos + width/2, suppressed_means, width,
                      yerr=suppressed_stds,
                      label='Suppressed', color=suppressed_color, alpha=0.7,
                      edgecolor='black', linewidth=1.5,
                      capsize=5, error_kw={'linewidth': 2})
            
            # Add labels with mean ± std
            for i, (x, exc_mean, exc_std) in enumerate(zip(x_pos, excited_means, excited_stds)):
                ax.text(x - width/2, exc_mean + exc_std + 0.02,
                       f'{exc_mean:.2f}\n±{exc_std:.2f}',
                       ha='center', va='bottom', fontsize=7, rotation=0)
            
            # Formatting
            ax.set_xlabel('Connectivity Instance', fontsize=11)
            ax.set_ylabel('Fraction of Cells', fontsize=11)
            
            # Color-code title to indicate target population
            if is_target_population:
                # Get n_nonexpr if available
                n_nonexpr = conn_data.get('n_nonexpr_mean', 'N/A')
                title_text = f'{target.upper()} Stim -> {post_pop.upper()}\n(non-expr, n≈{n_nonexpr:.0f})'
                title_color = 'purple'
            else:
                title_text = f'{target.upper()} Stim -> {post_pop.upper()}'
                title_color = 'black'
            
            ax.set_title(title_text + f'\n(Intensity {intensity})',
                        fontsize=12, fontweight='bold', color=title_color)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'C{i}' for i in conn_indices], fontsize=10)
            ax.set_ylim(0, max(max(excited_means), max(suppressed_means)) * 1.3 
                       if excited_means else 1.0)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            # Add horizontal line at 0
            ax.axhline(y=0, color='black', linewidth=1)
    
    plt.suptitle('Response Variability: Between vs. Within Connectivity\n'
                 '(Error bars = std across input patterns)\n'
                 '(Purple = non-expressing cells in target population)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved connectivity variance plot to: {save_path}")
    
    return fig    

def plot_connectivity_instance_variance_detailed(
    nested_results: Dict,
    target_populations: list = ['pv', 'sst'],
    post_populations: list = ['gc', 'mc', 'pv', 'sst'],
    intensity: float = 1.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    expression_threshold: float = 0.2,  # opsin expression threshold
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Alternative version that computes statistics directly from nested trial results
    
    This provides more control and can compute additional metrics beyond what's
    in the aggregated results.
    
    Args:
        nested_results: Full nested results dict with trial-level data
        target_populations: Populations that were stimulated
        post_populations: Post-synaptic populations to plot
        intensity: Light intensity to plot
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period start (ms)
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    
    n_targets = len(target_populations)
    n_posts = len(post_populations)
    
    fig, axes = plt.subplots(n_targets, n_posts, figsize=(6 * n_posts, 5 * n_targets),
                             squeeze=False)
    
    excited_color = '#e74c3c'
    suppressed_color = '#3498db'
    
    for target_idx, target in enumerate(target_populations):
        for post_idx, post_pop in enumerate(post_populations):
            ax = axes[target_idx, post_idx]
            
            # Get all trials for this target/intensity
            trials = nested_results['nested_results'][target][intensity]
            
            # Group by connectivity instance
            conn_groups = {}
            for trial in trials:
                conn_idx = trial.connectivity_idx
                if conn_idx not in conn_groups:
                    conn_groups[conn_idx] = []
                conn_groups[conn_idx].append(trial)
            
            conn_indices = sorted(conn_groups.keys())
            
            excited_means = []
            excited_stds = []
            suppressed_means = []
            suppressed_stds = []
            n_cells_analyzed = []  # Track how many cells analyzed
            
            # Compute statistics for each connectivity instance
            for conn_idx in conn_indices:
                conn_trials = conn_groups[conn_idx]
                
                excited_fracs = []
                suppressed_fracs = []
                n_cells_list = []
                
                for trial in conn_trials:
                    # Check if this is the target population
                    is_target_population = (post_pop == trial.target_population)
                    
                    # Get activity trace
                    if 'activity_trace_mean' in trial.results:
                        activity = trial.results['activity_trace_mean'][post_pop]
                    else:
                        activity = trial.results['activity_trace'][post_pop]
                    
                    time = trial.results['time']
                    
                    # Convert to numpy if needed
                    if hasattr(activity, 'cpu'):
                        activity = activity.cpu().numpy()
                    if hasattr(time, 'cpu'):
                        time = time.cpu().numpy()
                    
                    # Create time masks
                    baseline_mask = (time >= warmup) & (time < stim_start)
                    stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
                    
                    # Compute response for ALL cells first
                    baseline_rate_all = np.mean(activity[:, baseline_mask], axis=1)
                    stim_rate_all = np.mean(activity[:, stim_mask], axis=1)
                    rate_change_all = stim_rate_all - baseline_rate_all
                    baseline_std_all = np.std(baseline_rate_all)
                    
                    # Filter for non-expressing cells if this is target population
                    if is_target_population:
                        # Get opsin expression
                        opsin_expression = trial.opsin_expression
                        if hasattr(opsin_expression, 'cpu'):
                            opsin_expression = opsin_expression.cpu().numpy()
                        
                        # Create non-expressing mask
                        non_expressing_mask = opsin_expression < expression_threshold
                        n_nonexpr = np.sum(non_expressing_mask)
                        
                        if n_nonexpr > 0:
                            # Filter to non-expressing cells only
                            baseline_rate = baseline_rate_all[non_expressing_mask]
                            stim_rate = stim_rate_all[non_expressing_mask]
                            rate_change = rate_change_all[non_expressing_mask]
                            baseline_std = np.std(baseline_rate)
                            
                            n_cells_list.append(n_nonexpr)
                        else:
                            # No non-expressing cells - use empty arrays
                            baseline_rate = np.array([])
                            rate_change = np.array([])
                            baseline_std = 0.0
                            n_cells_list.append(0)
                    else:
                        # Not target population - use all cells
                        baseline_rate = baseline_rate_all
                        rate_change = rate_change_all
                        baseline_std = baseline_std_all
                        n_cells_list.append(len(baseline_rate))
                    
                    # Classify cells (only if we have data)
                    if len(rate_change) > 0:
                        excited = np.mean((rate_change > baseline_std).astype(float))
                        suppressed = np.mean((rate_change < -baseline_std).astype(float))
                    else:
                        excited = 0.0
                        suppressed = 0.0
                    
                    excited_fracs.append(excited)
                    suppressed_fracs.append(suppressed)
                
                # Compute mean and std across input patterns for this connectivity
                excited_means.append(np.mean(excited_fracs))
                excited_stds.append(np.std(excited_fracs))
                suppressed_means.append(np.mean(suppressed_fracs))
                suppressed_stds.append(np.std(suppressed_fracs))
                n_cells_analyzed.append(np.mean(n_cells_list))
            
            x_pos = np.arange(len(conn_indices))
            width = 0.35
            
            # Plot bars with error bars
            ax.bar(x_pos - width/2, excited_means, width,
                  yerr=excited_stds,
                  label='Excited', color=excited_color, alpha=0.7,
                  edgecolor='black', linewidth=1.5,
                  capsize=5, error_kw={'linewidth': 2})
            
            ax.bar(x_pos + width/2, suppressed_means, width,
                  yerr=suppressed_stds,
                  label='Suppressed', color=suppressed_color, alpha=0.7,
                  edgecolor='black', linewidth=1.5,
                  capsize=5, error_kw={'linewidth': 2})
            
            # Formatting
            ax.set_xlabel('Connectivity Instance', fontsize=11)
            ax.set_ylabel('Fraction of Cells', fontsize=11)
            
            # FIXED: Color-code title to indicate target population
            is_target_population = (post_pop == target)
            if is_target_population:
                avg_n_cells = np.mean(n_cells_analyzed)
                title_text = f'{target.upper()} Stim → {post_pop.upper()}\n(non-expr, n≈{avg_n_cells:.0f})'
                title_color = 'purple'
            else:
                title_text = f'{target.upper()} Stim → {post_pop.upper()}'
                title_color = 'black'
            
            ax.set_title(title_text + f'\n(Intensity {intensity})',
                        fontsize=12, fontweight='bold', color=title_color)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'C{i}' for i in conn_indices], fontsize=10)
            ax.set_ylim(0, max(max(excited_means), max(suppressed_means)) * 1.4 
                       if excited_means else 1.0)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
    
    plt.suptitle('Response Variability by Connectivity Instance\n'
                 '(Error bars = std across input patterns)\n'
                 f'(Purple = non-expressing cells, threshold={expression_threshold})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved detailed connectivity variance plot to: {save_path}")
    
    return fig


def load_nested_experiment_seeds_and_opsin(
    filepath: str,
    target_populations: List[str] = ['pv', 'sst'],
    max_n_experiments: Optional[bool] = None
) -> Dict:
    """
    Load connectivity seeds and opsin expressions from a nested experiment HDF5 file.

    For each target population and connectivity instance, extracts the opsin
    expression array from the first available intensity/pattern combination.
    Expression is constant across MEC patterns and intensities for a given
    (target, connectivity) pair due to deterministic seeding.

    Args:
        filepath: Path to nested experiment HDF5 file
        target_populations: Populations to load opsin expressions for
        max_n_experiments: optional argument to specify maximum number of experiments to load
    
    Returns:
        Dict with keys:
            'connectivity_seeds': List[int] — one seed per connectivity instance
            'opsin_expressions': Dict[str, Dict[int, np.ndarray]]
                mapping target_pop -> {conn_idx: expression_array}
            'metadata': Dict — full experiment metadata
            'intensities': List[float] — intensities used in nested experiment
    """
    metadata = load_metadata_from_hdf5(filepath)
    seed_structure = metadata['seed_structure']
    connectivity_seeds = seed_structure['connectivity_seeds']
    if max_n_experiments is not None:
        connectivity_seeds = connectivity_seeds[:max_n_experiments]
    
    opsin_expressions = {}
    intensities = metadata.get('intensities', [])

    with h5py.File(filepath, 'r') as f:
        for target in target_populations:
            if target not in f:
                logger.warning(
                    f"Target population '{target}' not found in {filepath}"
                )
                continue

            opsin_expressions[target] = {}

            # Use first available intensity (expression is intensity-independent)
            intensity_keys = sorted(
                k for k in f[target].keys() if k.startswith('intensity_')
            )
            if not intensity_keys:
                logger.warning(
                    f"No intensity groups found for '{target}' in {filepath}"
                )
                continue

            intensity_key = intensity_keys[0]
            
            for conn_idx in range(len(connectivity_seeds)):
                conn_key = f'connectivity_{conn_idx}'
                if conn_key not in f[target][intensity_key]:
                    logger.warning(
                        f"Missing conn_{conn_idx} for {target}/{intensity_key}"
                    )
                    continue

                # Load from first MEC pattern (constant across patterns)
                pattern_key = 'pattern_0'
                trial_grp = f[target][intensity_key][conn_key].get(pattern_key)
                if trial_grp is None or 'opsin_expression' not in trial_grp:
                    logger.warning(
                        f"Missing opsin_expression for "
                        f"{target}/{intensity_key}/{conn_key}/{pattern_key}"
                    )
                    continue

                opsin_expressions[target][conn_idx] = trial_grp['opsin_expression'][:]

    n_loaded = {
        t: len(exprs) for t, exprs in opsin_expressions.items()
    }
    logger.info(
        f"Loaded from {filepath}:\n"
        f"  Connectivity seeds: {len(connectivity_seeds)}\n"
        f"  Opsin expressions: {n_loaded}\n"
        f"  Intensities: {intensities}"
    )

    return {
        'connectivity_seeds': connectivity_seeds,
        'opsin_expressions': opsin_expressions,
        'metadata': metadata,
        'intensities': intensities,
    }


ALL_POPULATIONS = ('gc', 'mc', 'pv', 'sst')

def _build_csv_fieldnames(
    populations: Tuple[str, ...] = ALL_POPULATIONS,
) -> List[str]:
    """
    Build the ordered list of CSV column names.

    Identifier columns come first, then per-population metric columns.
    For the target population the metrics are split into ``_expr`` and
    ``_nonexpr`` variants; the plain (unsuffixed) columns are omitted
    for the target because the expressing/non-expressing split is more
    informative.  Non-target populations use unsuffixed columns only.

    Because target population varies per row, we emit *all* possible
    suffixed columns for every population and let the writer fill in
    only the relevant ones (the rest stay empty).

    Returns:
        Ordered list of field names.
    """
    id_cols = [
        'connectivity_idx',
        'mec_pattern_idx',
        'target_population',
        'intensity',
    ]

    metric_suffixes = [
        'n_cells',
        'fraction_excited',
        'fraction_suppressed',
        'fraction_unchanged',
        'mean_rate_change',
        'mean_modulation_ratio',
        'mean_baseline_rate',
        'mean_stim_rate',
    ]

    pop_cols = []
    for pop in populations:
        # Unsuffixed columns (used for non-target pops)
        for suffix in metric_suffixes:
            pop_cols.append(f'{pop}_{suffix}')
        # Expressing-cell columns (target pop only)
        for suffix in metric_suffixes:
            pop_cols.append(f'{pop}_expr_{suffix}')
        # Non-expressing-cell columns (target pop only)
        for suffix in metric_suffixes:
            pop_cols.append(f'{pop}_nonexpr_{suffix}')

    return id_cols + pop_cols


# ============================================================================
# Per-trial statistics
# ============================================================================

def compute_trial_population_stats(
    activity: torch.Tensor,
    time: np.ndarray,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    threshold_std: float = 1.0,
    cell_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute aggregate statistics for a subset of cells in one population
    for a single trial.

    Classification uses the same +/- ``threshold_std`` * baseline_std
    criterion as ``classify_cells_by_connectivity``.

    Args:
        activity: Activity tensor ``[n_cells, n_timesteps]``.
        time: Time array (numpy, ms).
        stim_start: Stimulation onset (ms).
        stim_duration: Duration of stimulation (ms).
        warmup: Start of the baseline window (ms).
        threshold_std: Number of baseline standard deviations for
            the excited / suppressed classification threshold.
        cell_mask: Optional boolean mask selecting a subset of cells
            along the first axis of *activity*.  ``None`` means use
            all cells.

    Returns:
        Dictionary with keys ``n_cells``, ``fraction_excited``,
        ``fraction_suppressed``, ``fraction_unchanged``,
        ``mean_rate_change``, ``mean_modulation_ratio``,
        ``mean_baseline_rate``, ``mean_stim_rate``.
    """
    baseline_mask = (time >= warmup) & (time < stim_start)
    stim_mask = (
        (time >= stim_start) & (time <= (stim_start + stim_duration))
    )

    # Select cells --------------------------------------------------------
    if cell_mask is not None:
        if not isinstance(cell_mask, torch.Tensor):
            cell_mask_t = torch.from_numpy(cell_mask).to(activity.device)
        else:
            cell_mask_t = cell_mask
        act = activity[cell_mask_t]
    else:
        act = activity

    n_cells = act.shape[0]
    if n_cells == 0:
        return {
            'n_cells': 0,
            'fraction_excited': float('nan'),
            'fraction_suppressed': float('nan'),
            'fraction_unchanged': float('nan'),
            'mean_rate_change': float('nan'),
            'mean_modulation_ratio': float('nan'),
            'mean_baseline_rate': float('nan'),
            'mean_stim_rate': float('nan'),
        }

    # Per-cell rates ------------------------------------------------------
    baseline_rate = torch.mean(act[:, baseline_mask], dim=1)  # [n_cells]
    stim_rate = torch.mean(act[:, stim_mask], dim=1)          # [n_cells]
    rate_change = stim_rate - baseline_rate
    baseline_std = torch.std(baseline_rate)

    # Classification ------------------------------------------------------
    threshold = threshold_std * baseline_std
    excited = rate_change > threshold
    suppressed = rate_change < -threshold
    unchanged = ~(excited | suppressed)

    frac_excited = torch.mean(excited.float()).item()
    frac_suppressed = torch.mean(suppressed.float()).item()
    frac_unchanged = torch.mean(unchanged.float()).item()

    # Aggregate rates -----------------------------------------------------
    mean_baseline = torch.mean(baseline_rate).item()
    mean_stim = torch.mean(stim_rate).item()
    mean_change = torch.mean(rate_change).item()

    # Modulation ratio: log2(stim / baseline), safe for near-zero baselines
    eps = 1e-6
    if mean_baseline > eps:
        modulation_ratio = float(np.log2(mean_stim / mean_baseline))
    else:
        modulation_ratio = float('nan')

    return {
        'n_cells': n_cells,
        'fraction_excited': frac_excited,
        'fraction_suppressed': frac_suppressed,
        'fraction_unchanged': frac_unchanged,
        'mean_rate_change': mean_change,
        'mean_modulation_ratio': modulation_ratio,
        'mean_baseline_rate': mean_baseline,
        'mean_stim_rate': mean_stim,
    }


# ============================================================================
# Row generator
# ============================================================================

def _generate_csv_rows(
    trials: List[NestedTrialResult],
    target_population: str,
    intensity: float,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    populations: Tuple[str, ...] = ALL_POPULATIONS,
) -> Iterator[Dict]:
    """
    Lazily yield one CSV row dict per trial.

    For the target population the row contains ``_expr`` and ``_nonexpr``
    columns; for all other populations the unsuffixed columns are filled.

    Args:
        trials: Flat list of ``NestedTrialResult`` for a single
            (target, intensity) condition.
        target_population: Which population was optogenetically stimulated.
        intensity: Light intensity for this batch of trials.
        stim_start: Stimulation onset (ms).
        stim_duration: Stimulation duration (ms).
        warmup: Baseline window start (ms).
        threshold_std: Classification threshold in baseline std units.
        expression_threshold: Opsin expression level below which a cell
            is considered non-expressing.
        populations: Tuple of population names to include.

    Yields:
        One dict per trial, keyed by CSV column names.
    """
    for trial in trials:
        # --- resolve activity tensor and time array ----------------------
        if 'activity_trace_mean' in trial.results:
            activity_dict = trial.results['activity_trace_mean']
        elif 'activity_trace' in trial.results:
            activity_dict = trial.results['activity_trace']
        else:
            raise KeyError(
                "Trial results missing 'activity_trace' and "
                "'activity_trace_mean'"
            )

        time = trial.results['time']
        if isinstance(time, torch.Tensor):
            time = time.cpu().numpy()
        else:
            time = np.asarray(time)

        # --- opsin expression for target population ----------------------
        opsin_expr = trial.opsin_expression
        if isinstance(opsin_expr, torch.Tensor):
            opsin_expr = opsin_expr.cpu().numpy()
        else:
            opsin_expr = np.asarray(opsin_expr)

        # --- build row ---------------------------------------------------
        row: Dict = {
            'connectivity_idx': trial.connectivity_idx,
            'mec_pattern_idx': trial.mec_pattern_idx,
            'target_population': target_population,
            'intensity': intensity,
        }

        for pop in populations:
            pop_activity = activity_dict[pop]
            if isinstance(pop_activity, np.ndarray):
                pop_activity = torch.from_numpy(pop_activity)

            if pop == target_population:
                # --- expressing cells ------------------------------------
                expr_mask = opsin_expr >= expression_threshold
                expr_stats = compute_trial_population_stats(
                    pop_activity, time,
                    stim_start, stim_duration, warmup,
                    threshold_std, cell_mask=expr_mask,
                )
                for key, val in expr_stats.items():
                    row[f'{pop}_expr_{key}'] = val

                # --- non-expressing cells --------------------------------
                nonexpr_mask = opsin_expr < expression_threshold
                nonexpr_stats = compute_trial_population_stats(
                    pop_activity, time,
                    stim_start, stim_duration, warmup,
                    threshold_std, cell_mask=nonexpr_mask,
                )
                for key, val in nonexpr_stats.items():
                    row[f'{pop}_nonexpr_{key}'] = val

            else:
                # --- all cells (non-target population) -------------------
                stats = compute_trial_population_stats(
                    pop_activity, time,
                    stim_start, stim_duration, warmup,
                    threshold_std,
                )
                for key, val in stats.items():
                    row[f'{pop}_{key}'] = val

        yield row


def export_nested_experiment_csv(
    nested_results: Dict,
    csv_path: str,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    target_populations: Optional[List[str]] = None,
    intensities: Optional[List[float]] = None,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    populations: Tuple[str, ...] = ALL_POPULATIONS,
) -> int:
    """
    Export per-(connectivity, MEC pattern) aggregate statistics to CSV.

    Each row represents a single trial identified by
    ``(connectivity_idx, mec_pattern_idx, target_population, intensity)``.

    For the target population, separate ``_expr`` and ``_nonexpr`` columns
    are emitted so that direct optogenetic effects can be distinguished from
    true network-mediated (paradoxical) responses.  Non-target populations
    use unsuffixed columns.

    Works with in-memory ``nested_results`` dictionaries of the form
    ``{target: {intensity: [NestedTrialResult, ...], ...}, ...}``.
    For HDF5-backed experiments, load trials first via
    ``load_nested_trials_from_hdf5`` and construct the same dict.

    Args:
        nested_results: ``{target_pop: {intensity: [NestedTrialResult, ...]}}``
        csv_path: Output file path.
        stim_start: Stimulation onset (ms).
        stim_duration: Stimulation duration (ms).
        warmup: Baseline window start (ms).
        target_populations: Subset of targets to export (default: all keys
            in *nested_results*).
        intensities: Subset of intensities to export (default: all).
        threshold_std: Classification threshold in baseline std units.
        expression_threshold: Opsin expression threshold.
        populations: Tuple of population names to include.

    Returns:
        Number of rows written.
    """
    if target_populations is None:
        target_populations = sorted(nested_results.keys())

    fieldnames = _build_csv_fieldnames(populations)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            extrasaction='ignore',   # silently drop unexpected keys
            restval='',              # empty string for missing columns
        )
        writer.writeheader()

        for target in target_populations:
            if target not in nested_results:
                logger.warning(
                    f"Target '{target}' not found in nested_results, skipping"
                )
                continue

            target_data = nested_results[target]
            iter_intensities = (
                sorted(target_data.keys())
                if intensities is None
                else [i for i in intensities if i in target_data]
            )

            for intensity in iter_intensities:
                trials = target_data[intensity]
                logger.info(
                    f"  Exporting {target.upper()} intensity={intensity}: "
                    f"{len(trials)} trials"
                )

                for row in _generate_csv_rows(
                    trials=trials,
                    target_population=target,
                    intensity=intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    warmup=warmup,
                    threshold_std=threshold_std,
                    expression_threshold=expression_threshold,
                    populations=populations,
                ):
                    writer.writerow(row)
                    n_rows += 1

    logger.info(f"Exported {n_rows} rows to {csv_path}")
    return n_rows


def export_nested_experiment_csv_from_hdf5(
    hdf5_filepath: str,
    csv_path: str,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    target_populations: Optional[List[str]] = None,
    intensities: Optional[List[float]] = None,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    populations: Tuple[str, ...] = ALL_POPULATIONS,
) -> int:
    """
    Export nested experiment CSV directly from an HDF5 file.

    Loads trials one (target, intensity) block at a time to limit memory
    usage, then delegates to the row generator.

    Args:
        hdf5_filepath: Path to the nested experiment HDF5 file.
        csv_path: Output CSV path.
        stim_start: Stimulation onset (ms).
        stim_duration: Stimulation duration (ms).
        warmup: Baseline window start (ms).
        target_populations: Targets to export (default: ``['pv', 'sst']``).
        intensities: Intensities to export (default: all in file).
        threshold_std: Classification threshold in baseline std units.
        expression_threshold: Opsin expression threshold.
        populations: Tuple of population names to include.

    Returns:
        Number of rows written.
    """

    metadata = load_metadata_from_hdf5(hdf5_filepath)
    n_connectivity = metadata['n_connectivity_instances']
    n_mec_patterns = metadata['n_mec_patterns_per_connectivity']

    if target_populations is None:
        target_populations = ['pv', 'sst']

    fieldnames = _build_csv_fieldnames(populations)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with (
        open(csv_path, 'w', newline='') as fh,
        h5py.File(hdf5_filepath, 'r') as f,
    ):
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            extrasaction='ignore',
            restval='',
        )
        writer.writeheader()

        for target in target_populations:
            if target not in f:
                logger.warning(
                    f"Target '{target}' not in HDF5, skipping"
                )
                continue

            # Discover intensities in this target group
            available = [
                float(k.split('_')[1])
                for k in f[target].keys()
                if k.startswith('intensity_')
            ]
            iter_intensities = (
                sorted(available)
                if intensities is None
                else [i for i in intensities if i in available]
            )

            for intensity in iter_intensities:
                trials = load_nested_trials_from_hdf5(
                    f, target, intensity,
                    n_connectivity, n_mec_patterns,
                    require_full_activity=True,
                )
                logger.info(
                    f"  Exporting {target.upper()} intensity={intensity}: "
                    f"{len(trials)} trials from HDF5"
                )

                for row in _generate_csv_rows(
                    trials=trials,
                    target_population=target,
                    intensity=intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    warmup=warmup,
                    threshold_std=threshold_std,
                    expression_threshold=expression_threshold,
                    populations=populations,
                ):
                    writer.writerow(row)
                    n_rows += 1

    logger.info(f"Exported {n_rows} rows to {csv_path}")
    return n_rows


def export_per_cell_weights_to_csv(
    classifications: Dict,
    circuit,
    source_populations: List[str],
    post_population: str,
    target_population: str,
    trials_by_connectivity: Dict,
    expression_threshold: float,
    csv_path: str
) -> None:
    """
    Export per-cell weight statistics to CSV.
    
    Each row represents one post-synaptic cell with:
    - Connectivity instance ID
    - Post-synaptic cell information (population, unit ID, opsin expression, classification)
    - For each presynaptic source: sum, mean, median of incoming weights
    - If presynaptic source is stimulated (opsin-expressing), separate stats for opsin+/- sources
    
    Args:
        classifications: Output from classify_cells_by_connectivity()
        circuit: DentateCircuit instance
        source_populations: List of source populations to analyze
        post_population: Post-synaptic population name
        target_population: Stimulated population name
        trials_by_connectivity: Organized trials by connectivity
        expression_threshold: Opsin expression threshold
        csv_path: Path to save CSV file
    """
    import csv
    
    # Collect all cell data
    cell_records = []
    
    for conn_idx in sorted(classifications.keys()):
        conn_data = classifications[conn_idx]

        # Get all cell indices for this connectivity
        all_cell_indices = set()
        all_cell_indices.update(conn_data['excited_cells'])
        all_cell_indices.update(conn_data['suppressed_cells'])
        all_cell_indices.update(conn_data['unchanged_cells'])
        
        # Get opsin expression from trials for this connectivity
        trials = trials_by_connectivity[conn_idx]
        if len(trials) == 0:
            continue
        
        # Get opsin expression from first trial (same for all trials in connectivity)
        opsin_expression = trials[0].opsin_expression
        if hasattr(opsin_expression, 'cpu'):
            opsin_expression = opsin_expression.cpu().numpy()
        
        for cell_idx in sorted(all_cell_indices):
            # Determine classification
            if cell_idx in conn_data['excited_cells']:
                classification = 'excited'
            elif cell_idx in conn_data['suppressed_cells']:
                classification = 'suppressed'
            else:
                classification = 'unchanged'
            
            # Determine opsin expression for post-synaptic cell
            post_opsin_expr = None
            post_has_opsin = None
            if post_population == target_population:
                post_opsin_expr = float(opsin_expression[cell_idx])
                post_has_opsin = post_opsin_expr >= expression_threshold
            
            # Build record
            record = {
                'connectivity_idx': conn_idx,
                'post_population': post_population,
                'post_cell_idx': cell_idx,
                'classification': classification
            }
            
            # Add opsin expression if applicable
            if post_population == target_population:
                record['post_opsin_expression'] = post_opsin_expr
                record['post_has_opsin'] = post_has_opsin
            
            # For each source population, get weight statistics
            for source_pop in source_populations:
                # Get weights from source to this target cell
                conn_name = f'{source_pop}_{post_population}'
                
                if conn_name not in circuit.connectivity.conductance_matrices:
                    # No connection from this source to post population
                    record[f'{source_pop}_weight_sum'] = 0.0
                    record[f'{source_pop}_weight_mean'] = 0.0
                    record[f'{source_pop}_weight_median'] = 0.0
                    record[f'{source_pop}_n_synapses'] = 0
                    
                    if source_pop == target_population:
                        record[f'{source_pop}_opsin_plus_weight_sum'] = 0.0
                        record[f'{source_pop}_opsin_plus_weight_mean'] = 0.0
                        record[f'{source_pop}_opsin_plus_weight_median'] = 0.0
                        record[f'{source_pop}_opsin_plus_n_synapses'] = 0
                        record[f'{source_pop}_opsin_minus_weight_sum'] = 0.0
                        record[f'{source_pop}_opsin_minus_weight_mean'] = 0.0
                        record[f'{source_pop}_opsin_minus_weight_median'] = 0.0
                        record[f'{source_pop}_opsin_minus_n_synapses'] = 0
                    continue
                
                cond_matrix = circuit.connectivity.conductance_matrices[conn_name]
                weight_matrix = cond_matrix.conductances  # [n_pre, n_post]
                
                # Convert to numpy for easier indexing
                if hasattr(weight_matrix, 'cpu'):
                    weight_matrix_np = weight_matrix.cpu().numpy()
                else:
                    weight_matrix_np = np.array(weight_matrix)
                
                weights_to_cell = weight_matrix_np[:, cell_idx]
                
                # Filter non-zero weights
                weights_to_cell = weights_to_cell[weights_to_cell > 0]
                
                # Basic statistics
                if len(weights_to_cell) > 0:
                    record[f'{source_pop}_weight_sum'] = float(np.sum(weights_to_cell))
                    record[f'{source_pop}_weight_mean'] = float(np.mean(weights_to_cell))
                    record[f'{source_pop}_weight_median'] = float(np.median(weights_to_cell))
                    record[f'{source_pop}_n_synapses'] = int(len(weights_to_cell))
                else:
                    record[f'{source_pop}_weight_sum'] = 0.0
                    record[f'{source_pop}_weight_mean'] = 0.0
                    record[f'{source_pop}_weight_median'] = 0.0
                    record[f'{source_pop}_n_synapses'] = 0
                
                # If source is stimulated population, separate by opsin expression
                if source_pop == target_population:
                    # Create opsin masks
                    opsin_plus_mask = opsin_expression >= expression_threshold
                    opsin_minus_mask = opsin_expression < expression_threshold
                    
                    # Get weights from opsin+ sources
                    weights_from_opsin_plus = weight_matrix_np[opsin_plus_mask, cell_idx]
                    weights_from_opsin_plus = weights_from_opsin_plus[weights_from_opsin_plus > 0]
                    
                    if len(weights_from_opsin_plus) > 0:
                        record[f'{source_pop}_opsin_plus_weight_sum'] = float(np.sum(weights_from_opsin_plus))
                        record[f'{source_pop}_opsin_plus_weight_mean'] = float(np.mean(weights_from_opsin_plus))
                        record[f'{source_pop}_opsin_plus_weight_median'] = float(np.median(weights_from_opsin_plus))
                        record[f'{source_pop}_opsin_plus_n_synapses'] = int(len(weights_from_opsin_plus))
                    else:
                        record[f'{source_pop}_opsin_plus_weight_sum'] = 0.0
                        record[f'{source_pop}_opsin_plus_weight_mean'] = 0.0
                        record[f'{source_pop}_opsin_plus_weight_median'] = 0.0
                        record[f'{source_pop}_opsin_plus_n_synapses'] = 0
                    
                    # Get weights from opsin- sources
                    weights_from_opsin_minus = weight_matrix_np[opsin_minus_mask, cell_idx]
                    weights_from_opsin_minus = weights_from_opsin_minus[weights_from_opsin_minus > 0]
                    
                    if len(weights_from_opsin_minus) > 0:
                        record[f'{source_pop}_opsin_minus_weight_sum'] = float(np.sum(weights_from_opsin_minus))
                        record[f'{source_pop}_opsin_minus_weight_mean'] = float(np.mean(weights_from_opsin_minus))
                        record[f'{source_pop}_opsin_minus_weight_median'] = float(np.median(weights_from_opsin_minus))
                        record[f'{source_pop}_opsin_minus_n_synapses'] = int(len(weights_from_opsin_minus))
                    else:
                        record[f'{source_pop}_opsin_minus_weight_sum'] = 0.0
                        record[f'{source_pop}_opsin_minus_weight_mean'] = 0.0
                        record[f'{source_pop}_opsin_minus_weight_median'] = 0.0
                        record[f'{source_pop}_opsin_minus_n_synapses'] = 0
            
            cell_records.append(record)
    
    # Write to CSV
    if len(cell_records) > 0:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all field names (some records may have opsin fields, others may not)
        all_fieldnames = set()
        for record in cell_records:
            all_fieldnames.update(record.keys())
        
        # Order fieldnames logically
        fieldnames = ['connectivity_idx', 'post_population', 'post_cell_idx', 'classification']
        
        # Add opsin fields if present
        if 'post_opsin_expression' in all_fieldnames:
            fieldnames.extend(['post_opsin_expression', 'post_has_opsin'])
        
        # Add source population fields in order
        for source_pop in source_populations:
            source_fields = [f for f in all_fieldnames if f.startswith(f'{source_pop}_')]
            # Sort to get: weight_sum, weight_mean, weight_median, n_synapses, then opsin variants
            source_fields_sorted = sorted(source_fields, key=lambda x: (
                'opsin' in x,  # Regular fields first, then opsin fields
                'minus' in x,  # opsin_plus before opsin_minus
                x.split('_')[-1]  # Then by statistic type
            ))
            fieldnames.extend(source_fields_sorted)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(cell_records)
        
        print(f"Exported per-cell weight data to: {csv_path}")
        print(f"  Total cells: {len(cell_records)}")
        print(f"  Connectivity instances: {len(set(r['connectivity_idx'] for r in cell_records))}")



def _build_per_cell_csv_fieldnames(
    source_populations: Tuple[str, ...],
    target_population: str,
) -> List[str]:
    """
    Build ordered list of CSV column names for per-cell export.
    
    Structure:
    - Identifiers: connectivity_idx, mec_pattern_idx, intensity, 
                   post_population, post_cell_idx
    - Response: classification, baseline_rate, stim_rate, rate_change, 
                modulation_ratio
    - Opsin (if target): post_opsin_expression, post_has_opsin
    - For each source population:
        - Overall: sum, mean, median, n_synapses
        - If source == target: separate opsin_plus and opsin_minus stats
    
    Args:
        source_populations: Pre-synaptic populations to analyze
        target_population: Which population is optogenetically stimulated
        
    Returns:
        Ordered list of field names
    """
    # Identifier columns
    id_cols = [
        'connectivity_idx',
        'mec_pattern_idx', 
        'intensity',
        'post_population',
        'post_cell_idx',
    ]
    
    # Response columns
    response_cols = [
        'classification',  # excited, suppressed, unchanged
        'baseline_rate',
        'stim_rate',
        'rate_change',
        'modulation_ratio',
    ]
    
    # Opsin columns (for target population cells)
    opsin_cols = [
        'post_opsin_expression',
        'post_has_opsin',
    ]
    
    # Weight statistics columns (per source population)
    weight_stats = [
        'weight_sum',
        'weight_mean', 
        'weight_median',
        'n_synapses',
    ]
    
    # Build source population columns
    source_cols = []
    for source_pop in source_populations:
        # Overall weight statistics
        for stat in weight_stats:
            source_cols.append(f'{source_pop}_{stat}')
        
        # If this source is the target, add opsin-specific columns
        if source_pop == target_population:
            for stat in weight_stats:
                source_cols.append(f'{source_pop}_opsin_plus_{stat}')
            for stat in weight_stats:
                source_cols.append(f'{source_pop}_opsin_minus_{stat}')
    
    # Combine all fieldnames
    return id_cols + response_cols + opsin_cols + source_cols


def _compute_per_cell_response_stats(
    activity: torch.Tensor,
    time: np.ndarray,
    cell_idx: int,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    threshold_std: float = 1.0,
) -> Dict:
    """
    Compute response statistics for a single cell.
    
    Args:
        activity: Activity tensor [n_cells, n_timesteps]
        time: Time array (ms)
        cell_idx: Index of cell to analyze
        stim_start: Stimulation onset (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline window start (ms)
        threshold_std: Classification threshold in baseline std units
        
    Returns:
        Dict with baseline_rate, stim_rate, rate_change, 
        modulation_ratio, classification
    """
    baseline_mask = (time >= warmup) & (time < stim_start)
    stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
    
    # Extract single cell activity
    cell_activity = activity[cell_idx, :]
    
    # Compute rates
    baseline_rate = torch.mean(cell_activity[baseline_mask]).item()
    stim_rate = torch.mean(cell_activity[stim_mask]).item()
    rate_change = stim_rate - baseline_rate
    
    # Modulation ratio: log2(stim / baseline)
    eps = 1e-6
    if baseline_rate > eps:
        modulation_ratio = float(np.log2(stim_rate / baseline_rate))
    else:
        modulation_ratio = float('nan')
    
    # Classification (using population-level baseline std for consistency)
    population_baseline = torch.mean(activity[:, baseline_mask], dim=1)
    baseline_std = torch.std(population_baseline).item()
    threshold = threshold_std * baseline_std
    
    if rate_change > threshold:
        classification = 'excited'
    elif rate_change < -threshold:
        classification = 'suppressed'
    else:
        classification = 'unchanged'
    
    return {
        'baseline_rate': baseline_rate,
        'stim_rate': stim_rate,
        'rate_change': rate_change,
        'modulation_ratio': modulation_ratio,
        'classification': classification,
    }


def _extract_incoming_weights(
    circuit,
    source_pop: str,
    post_pop: str,
    post_cell_idx: int,
    source_opsin_expression: Optional[np.ndarray] = None,
    expression_threshold: float = 0.2,
) -> Dict:
    """
    Extract incoming synaptic weights for a single post-synaptic cell.
    
    Args:
        circuit: DentateCircuit instance
        source_pop: Pre-synaptic population name
        post_pop: Post-synaptic population name
        post_cell_idx: Index of post-synaptic cell
        source_opsin_expression: Opsin expression levels for source population
                                 (if source is target population)
        expression_threshold: Threshold for opsin-expressing classification
        
    Returns:
        Dict with weight statistics (sum, mean, median, n_synapses)
        and optionally opsin_plus/opsin_minus variants
    """
    conn_name = f'{source_pop}_{post_pop}'
    
    # Check if connection exists
    if conn_name not in circuit.connectivity.conductance_matrices:
        stats = {
            'weight_sum': 0.0,
            'weight_mean': 0.0,
            'weight_median': 0.0,
            'n_synapses': 0,
        }
        
        # Add opsin variants if source has opsin expression
        if source_opsin_expression is not None:
            for suffix in ['opsin_plus', 'opsin_minus']:
                stats[f'{suffix}_weight_sum'] = 0.0
                stats[f'{suffix}_weight_mean'] = 0.0
                stats[f'{suffix}_weight_median'] = 0.0
                stats[f'{suffix}_n_synapses'] = 0
        
        return stats
    
    # Get conductance matrix [n_pre, n_post]
    cond_matrix = circuit.connectivity.conductance_matrices[conn_name]
    weight_matrix = cond_matrix.conductances
    
    # Convert to numpy
    if hasattr(weight_matrix, 'cpu'):
        weight_matrix_np = weight_matrix.cpu().numpy()
    else:
        weight_matrix_np = np.array(weight_matrix)
    
    # Extract weights to this post-synaptic cell
    weights_to_cell = weight_matrix_np[:, post_cell_idx]
    
    # Filter non-zero weights (actual synapses)
    nonzero_mask = weights_to_cell > 0
    weights_to_cell = weights_to_cell[nonzero_mask]
    
    # Overall statistics
    stats = {}
    if len(weights_to_cell) > 0:
        stats['weight_sum'] = float(np.sum(weights_to_cell))
        stats['weight_mean'] = float(np.mean(weights_to_cell))
        stats['weight_median'] = float(np.median(weights_to_cell))
        stats['n_synapses'] = int(len(weights_to_cell))
    else:
        stats['weight_sum'] = 0.0
        stats['weight_mean'] = 0.0
        stats['weight_median'] = 0.0
        stats['n_synapses'] = 0
    
    # Opsin-specific statistics (if source is target population)
    if source_opsin_expression is not None:
        # Create masks for opsin-expressing vs non-expressing sources
        opsin_plus_mask = (source_opsin_expression >= expression_threshold) & nonzero_mask
        opsin_minus_mask = (source_opsin_expression < expression_threshold) & nonzero_mask
        
        # Opsin-expressing sources
        weights_opsin_plus = weight_matrix_np[opsin_plus_mask, post_cell_idx]
        if len(weights_opsin_plus) > 0:
            stats['opsin_plus_weight_sum'] = float(np.sum(weights_opsin_plus))
            stats['opsin_plus_weight_mean'] = float(np.mean(weights_opsin_plus))
            stats['opsin_plus_weight_median'] = float(np.median(weights_opsin_plus))
            stats['opsin_plus_n_synapses'] = int(len(weights_opsin_plus))
        else:
            stats['opsin_plus_weight_sum'] = 0.0
            stats['opsin_plus_weight_mean'] = 0.0
            stats['opsin_plus_weight_median'] = 0.0
            stats['opsin_plus_n_synapses'] = 0
        
        # Non-expressing sources
        weights_opsin_minus = weight_matrix_np[opsin_minus_mask, post_cell_idx]
        if len(weights_opsin_minus) > 0:
            stats['opsin_minus_weight_sum'] = float(np.sum(weights_opsin_minus))
            stats['opsin_minus_weight_mean'] = float(np.mean(weights_opsin_minus))
            stats['opsin_minus_weight_median'] = float(np.median(weights_opsin_minus))
            stats['opsin_minus_n_synapses'] = int(len(weights_opsin_minus))
        else:
            stats['opsin_minus_weight_sum'] = 0.0
            stats['opsin_minus_weight_mean'] = 0.0
            stats['opsin_minus_weight_median'] = 0.0
            stats['opsin_minus_n_synapses'] = 0
    
    return stats


def _generate_per_cell_rows_for_trial(
    trial_data: Dict,
    circuit,
    connectivity_idx: int,
    mec_pattern_idx: int,
    target_population: str,
    intensity: float,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    threshold_std: float,
    expression_threshold: float,
    source_populations: Tuple[str, ...],
    post_populations: Tuple[str, ...],
    opsin_expression: np.ndarray,
) -> Iterator[Dict]:
    """
    Generate CSV rows for all cells in a single trial.
    
    Yields one row per (post_population, cell_idx) combination.
    
    Args:
        trial_data: Trial results from load_trial_from_hdf5
        circuit: Reconstructed DentateCircuit for this connectivity
        connectivity_idx: Connectivity instance index
        mec_pattern_idx: MEC pattern index
        target_population: Stimulated population
        intensity: Light intensity
        stim_start: Stimulation onset (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline window start (ms)
        threshold_std: Classification threshold
        expression_threshold: Opsin threshold
        source_populations: Pre-synaptic populations to analyze
        post_populations: Post-synaptic populations to analyze
        opsin_expression: Opsin expression for target population
        
    Yields:
        One dict per cell (CSV row)
    """
    # Get activity traces
    if 'activity_trace_mean' in trial_data:
        activity_dict = trial_data['activity_trace_mean']
    elif 'activity_trace' in trial_data:
        activity_dict = trial_data['activity_trace']
    else:
        raise KeyError("Trial data missing activity traces")
    
    time = trial_data['time']
    if isinstance(time, torch.Tensor):
        time = time.cpu().numpy()
    else:
        time = np.asarray(time)
    
    # Process each post-synaptic population
    for post_pop in post_populations:
        pop_activity = activity_dict[post_pop]
        if isinstance(pop_activity, np.ndarray):
            pop_activity = torch.from_numpy(pop_activity)
        
        n_cells = pop_activity.shape[0]
        
        # Determine if this is the target population
        is_target_pop = (post_pop == target_population)
        
        # Process each cell
        for cell_idx in range(n_cells):
            # Base row data
            row = {
                'connectivity_idx': connectivity_idx,
                'mec_pattern_idx': mec_pattern_idx,
                'intensity': intensity,
                'post_population': post_pop,
                'post_cell_idx': cell_idx,
            }
            
            # Compute response statistics
            response_stats = _compute_per_cell_response_stats(
                pop_activity, time, cell_idx,
                stim_start, stim_duration, warmup, threshold_std
            )
            row.update(response_stats)
            
            # Add opsin expression if target population
            if is_target_pop:
                post_opsin_expr = float(opsin_expression[cell_idx])
                row['post_opsin_expression'] = post_opsin_expr
                row['post_has_opsin'] = post_opsin_expr >= expression_threshold
            else:
                row['post_opsin_expression'] = float('nan')
                row['post_has_opsin'] = False
            
            # Extract incoming weights from each source population
            for source_pop in source_populations:
                # Determine if source has opsin expression data
                source_opsin_expr = (
                    opsin_expression if source_pop == target_population
                    else None
                )
                
                # Get weight statistics
                weight_stats = _extract_incoming_weights(
                    circuit, source_pop, post_pop, cell_idx,
                    source_opsin_expr, expression_threshold
                )
                
                # Add to row with source population prefix
                for key, value in weight_stats.items():
                    row[f'{source_pop}_{key}'] = value
            
            yield row


def export_per_cell_weights_from_hdf5(
    hdf5_filepath: str,
    csv_path: str,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    optimization_json_file: Optional[str] = None,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    source_populations: Tuple[str, ...] = ('gc', 'mc', 'pv', 'sst'),
    post_populations: Tuple[str, ...] = ('gc', 'mc', 'pv', 'sst'),
    target_populations: Optional[List[str]] = None,
    intensities: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> int:
    """
    Export per-cell weights and firing rates to CSV from HDF5 nested experiment.
    
    Memory-efficient streaming approach that processes one connectivity 
    instance at a time, reconstructing circuits on-demand and loading 
    trials incrementally from HDF5.
    
    Output CSV structure:
    - One row per (connectivity_idx, mec_pattern_idx, intensity, 
                   post_population, post_cell_idx)
    - Columns: identifiers, response stats, opsin info (if target), 
               incoming weights from each source population
    
    Args:
        hdf5_filepath: Path to nested experiment HDF5 file
        csv_path: Output CSV file path
        stim_start: Stimulation onset (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline window start (ms)
        optimization_json_file: Path to optimization results 
                                (overrides HDF5 metadata)
        threshold_std: Classification threshold in baseline std units
        expression_threshold: Opsin expression threshold
        source_populations: Pre-synaptic populations to analyze
        post_populations: Post-synaptic populations to include
        target_populations: Targets to export (default: all in HDF5)
        intensities: Intensities to export (default: all in HDF5)
        device: PyTorch device for circuit reconstruction
        
    Returns:
        Number of rows written
        
    Example:
        >>> n_rows = export_per_cell_weights_from_hdf5(
        ...     'results/nested_experiment.h5',
        ...     'analysis/per_cell_weights.csv',
        ...     stim_start=1500.0,
        ...     stim_duration=1000.0,
        ...     warmup=500.0,
        ...     optimization_json_file='protocol/optimization_results.json'
        ... )
        >>> print(f"Exported {n_rows} rows")
    """
    from DG_protocol import (
        OptogeneticExperiment, CircuitParams, PerConnectionSynapticParams,
        OpsinParams, set_random_seed, get_default_device
    )
    
    # Setup
    if device is None:
        device = get_default_device()
    
    # Load metadata
    metadata = load_metadata_from_hdf5(hdf5_filepath)
    seed_structure = metadata['seed_structure']
    connectivity_seeds = seed_structure['connectivity_seeds']
    n_connectivity = metadata['n_connectivity_instances']
    n_mec_patterns = metadata['n_mec_patterns_per_connectivity']
    
    # Get optimization file from metadata if not provided
    if optimization_json_file is None:
        optimization_json_file = metadata.get('optimization_file')
    
    # Determine which targets and intensities to process
    if target_populations is None:
        target_populations = ['pv', 'sst']
    
    logger.info(
        f"\nExporting per-cell weights and firing rates from {hdf5_filepath}\n"
        f"  Connectivity instances: {n_connectivity}\n"
        f"  MEC patterns per connectivity: {n_mec_patterns}\n"
        f"  Target populations: {target_populations}\n"
        f"  Source populations: {source_populations}\n"
        f"  Post populations: {post_populations}"
    )
    
    # Open HDF5 and CSV files
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_rows = 0
    batch_size = 5000  # Write in batches for better performance
    row_batch = []
    
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        # We'll write to CSV for the first target to determine fieldnames
        # Then we know the structure for subsequent targets
        
        # Determine fieldnames from first target
        first_target = target_populations[0]
        fieldnames = _build_per_cell_csv_fieldnames(
            source_populations, first_target
        )
        
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                extrasaction='ignore',
                restval='',
            )
            writer.writeheader()
            
            # Process each connectivity instance
            for conn_idx in range(n_connectivity):
                connectivity_seed = connectivity_seeds[conn_idx]
                
                logger.info(
                    f"\nProcessing connectivity instance {conn_idx + 1}/{n_connectivity} "
                    f"(seed: {connectivity_seed})"
                )
                
                # Reconstruct circuit with this connectivity seed
                circuit_params = CircuitParams()
                synaptic_params = PerConnectionSynapticParams()
                opsin_params = OpsinParams()
                
                experiment = OptogeneticExperiment(
                    circuit_params,
                    synaptic_params,
                    opsin_params,
                    optimization_json_file=optimization_json_file,
                    device=device,
                    base_seed=connectivity_seed,
                )

                # Process each target population
                for target in target_populations:
                    if target not in hdf5_file:
                        logger.warning(f"  Target '{target}' not in HDF5, skipping")
                        continue
                    
                    # Discover available intensities
                    available_intensities = [
                        float(k.split('_')[1])
                        for k in hdf5_file[target].keys()
                        if k.startswith('intensity_')
                    ]
                    
                    iter_intensities = (
                        sorted(available_intensities)
                        if intensities is None
                        else [i for i in intensities if i in available_intensities]
                    )
                    
                    # Get opsin expression for target population (same across all trials)
                    # Load from first available trial
                    if iter_intensities:
                        intensity_key = f'intensity_{iter_intensities[0]}'
                        conn_key = f'connectivity_{conn_idx}'
                        pattern_key = 'pattern_0'
                        
                        if (conn_key in hdf5_file[target][intensity_key] and
                            pattern_key in hdf5_file[target][intensity_key][conn_key]):
                            trial_grp = hdf5_file[target][intensity_key][conn_key][pattern_key]
                            if 'opsin_expression' in trial_grp:
                                opsin_expression = trial_grp['opsin_expression'][:]
                            else:
                                # Fallback: create zero array
                                n_target_cells = getattr(circuit_params, f'n_{target}')
                                opsin_expression = np.zeros(n_target_cells)
                        else:
                            n_target_cells = getattr(circuit_params, f'n_{target}')
                            opsin_expression = np.zeros(n_target_cells)

                    else:
                        n_target_cells = getattr(circuit_params, f'n_{target}')
                        opsin_expression = np.zeros(n_target_cells)
                    
                    # Process each intensity
                    for intensity in iter_intensities:
                        logger.info(f"  {target.upper()} intensity={intensity}")
                        
                        # Process each MEC pattern
                        for mec_idx in range(n_mec_patterns):
                            # Load trial from HDF5
                            trial_data = load_trial_from_hdf5(
                                hdf5_file, target, intensity, conn_idx, mec_idx
                            )
                            
                            # Generate rows for all cells in this trial
                            for row in _generate_per_cell_rows_for_trial(
                                trial_data=trial_data,
                                circuit=experiment.circuit,
                                connectivity_idx=conn_idx,
                                mec_pattern_idx=mec_idx,
                                target_population=target,
                                intensity=intensity,
                                stim_start=stim_start,
                                stim_duration=stim_duration,
                                warmup=warmup,
                                threshold_std=threshold_std,
                                expression_threshold=expression_threshold,
                                source_populations=source_populations,
                                post_populations=post_populations,
                                opsin_expression=opsin_expression,
                            ):
                                row_batch.append(row)
                                n_rows += 1

                                # Write batch when it reaches batch_size
                                if len(row_batch) >= batch_size:
                                    writer.writerows(row_batch)
                                    row_batch = []
                                    csv_file.flush()
                            
                            # Periodic progress update
                            if (mec_idx + 1) % 5 == 0:
                                logger.info(
                                    f"    Completed {mec_idx + 1}/{n_mec_patterns} "
                                    f"MEC patterns ({n_rows} total rows)"
                                )

                # Write any remaining rows in batch
                if row_batch:
                    writer.writerows(row_batch)
                    row_batch = []
                    csv_file.flush()
                # Clean up circuit to free memory
                del experiment
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    logger.info(f"\nExport complete: {n_rows} rows written to {csv_path}")
    return n_rows        
