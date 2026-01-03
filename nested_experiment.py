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

from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

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
    
    # Storage for nested results
    nested_results = {
        'pv': {intensity: [] for intensity in intensities},
        'sst': {intensity: [] for intensity in intensities}
    }
    
    # Main nested loop
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
                    
                    # Store with proper indexing and opsin expression
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
        'metadata': {
            'optimization_file': optimization_json_file,
            'intensities': intensities,
            'mec_current': mec_current,
            'opsin_current': opsin_current,
            'stim_start': stim_start,
            'stim_duration': stim_duration,
            'warmup': warmup
        }
    }
    
    # Save if requested
    if save_results_file:
        save_nested_experiment_results(complete_results, save_results_file)
    
    # Print summary
    print_nested_experiment_summary(complete_results)
    
    return complete_results


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
        # When simulate_stimulation is called with n_trials=1, it aggregates and uses 'activity_trace_mean'
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
                raise KeyError(f"Trial results missing both 'activity_trace' and 'activity_trace_mean'. "
                              f"Available keys: {list(trial.results.keys())}")
            
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

def _hierarchical_variance_decomposition(grouped_data: List[List[float]],
                                         permutation_random_seed=47) -> Dict:
    """
    Compute hierarchical variance decomposition with robust statistical testing.
    
    Uses permutation test as primary method (better for small samples and discrete data),
    with fallback to F-test and explicit handling of edge cases.
    
    Args:
        grouped_data: List of lists, where each inner list is data from one group
        
    Returns:
        Dict with variance components:
            - total_var: Total variance
            - between_group_var: Variance between group means
            - within_group_var: Mean variance within groups
            - f_statistic: F-statistic for ANOVA
            - p_value: p-value for F-test
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
        # Only one group - no test possible
        f_statistic = np.nan
        p_value = np.nan
        test_method = 'insufficient_groups'
        
    elif within_group_var < 1e-10:
        # Zero or near-zero within-group variance - handle explicitly
        group_means_array = np.array(group_means)
        unique_means = np.unique(group_means_array)
        
        if len(unique_means) > 1:
            # Means differ but no within-group variance
            # This indicates deterministic separation by group
            f_statistic = np.inf
            p_value = 0.0
            test_method = 'deterministic_separation'
        else:
            # All values identical everywhere
            f_statistic = 0.0
            p_value = 1.0
            test_method = 'no_variation'
    
    else:
        # Normal case: use permutation test (primary method)
        try:
            def test_statistic(*groups):
                """F-statistic equivalent: variance of group means"""
                means = [np.mean(g) for g in groups]
                return np.var(means, ddof=1)
            
            # Run permutation test
            from scipy.stats import permutation_test
            
            result = permutation_test(
                grouped_data,
                test_statistic,
                permutation_type='independent',
                n_resamples=10000,
                random_state=permutation_random_seed  # For reproducibility
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
                # Ultimate fallback - should rarely happen
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
        'test_method': test_method  # track which test was used
    }    


def classify_mechanism_regime(
    variance_analysis: Dict,
    config: NestedExperimentConfig
) -> Dict:
    """
    Classify whether paradoxical excitation is driven by:
        - Connectivity patterns (high ICC, significant between-connectivity variance)
        - Population dynamics (low ICC, high within-connectivity variance)
        
    Uses ICC thresholds and statistical significance of ANOVA
    
    Returns:
        Classification for each target/intensity/population
    """
    # Classification thresholds
    ICC_CONNECTIVITY_THRESHOLD = 0.5  # ICC > 0.5 suggests connectivity-driven
    ICC_INPUT_THRESHOLD = 0.2  # ICC < 0.2 suggests input-driven
    P_VALUE_THRESHOLD = config.variance_significance_alpha
    
    classification = {}
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
                    # Perfect separation by connectivity - clearly connectivity-driven
                    regime = 'connectivity_driven'
                    
                elif test_method == 'no_variation':
                    # No variation at all
                    regime = 'no_significant_variance'
                    
                elif excited_p > P_VALUE_THRESHOLD:
                    # Not statistically significant
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
# Saving and Loading
# ============================================================================

def save_nested_experiment_results(results: Dict, filepath: str):
    """Save nested experiment results with compression"""
    import pickle
    from datetime import datetime
    
    # Add timestamp
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
    - Separate panels for each target × post-synaptic population combination
    
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
