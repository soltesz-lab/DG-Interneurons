"""
Nested Weights Analysis Framework

Extends the single-connectivity weight analysis to properly handle hierarchical
structure of nested experiments (connectivity instances x MEC patterns).

Key principles:
- Classify cells within each connectivity instance (averaged across MEC patterns)
- Extract weights for excited/suppressed cells per connectivity
- Bootstrap at connectivity level (independent units)
- Visualize between-connectivity and within-connectivity variance
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from nested_experiment import NestedTrialResult
from nested_effect_size import (organize_nested_trials,
                                average_activity_across_trials,
                                classify_cells_by_connectivity,
                                extract_weights_by_connectivity)
from pca_input_weight_patterns import analyze_input_weight_patterns_pca, plot_pca_summary_all_targets



# ============================================================================
# Weight Distribution Analysis (Nested)
# ============================================================================


def extract_weight_distributions_by_connectivity(
    circuit,
    classifications: Dict,
    post_population: str,
    source_populations: List[str],
    target_population: str,  # which population was stimulated
    trials_by_connectivity: Dict,  # used to get opsin expression
    expression_threshold: float = 0.2  # threshold for opsin+/-
) -> Dict:
    """
    Extract full weight distributions (not just totals) for each connectivity instance. Separates opsin+ and opsin- cells when source == target population.
    
    Args:
        circuit: DentateCircuit instance
        classifications: Output from classify_cells_by_connectivity()
        post_population: Post-synaptic population
        source_populations: List of source populations to analyze
        target_population: Which population was optogenetically stimulated
        trials_by_connectivity: Dict mapping connectivity_idx to trial list
        expression_threshold: Threshold for opsin+ classification (default: 0.2)
        
    Returns:
        dict: {
            connectivity_idx: {
                source_population: {
                    # Standard keys (all cells)
                    'weights_excited': array of individual synapse weights,
                    'weights_suppressed': array of individual synapse weights,
                    'total_input_excited': array of total input per cell,
                    'total_input_suppressed': array of total input per cell,
                    
                    # If source == target, also include:
                    'weights_excited_opsin_plus': array for opsin+ -> excited,
                    'weights_excited_opsin_minus': array for opsin- -> excited,
                    'weights_suppressed_opsin_plus': array for opsin+ -> suppressed,
                    'weights_suppressed_opsin_minus': array for opsin- -> suppressed,
                    'total_input_excited_opsin_plus': array,
                    'total_input_excited_opsin_minus': array,
                    'total_input_suppressed_opsin_plus': array,
                    'total_input_suppressed_opsin_minus': array,
                    'n_cells_opsin_plus': int,
                    'n_cells_opsin_minus': int
                }
            }
        }
    """
    weights_by_connectivity = {}
    
    for source_pop in source_populations:
        conn_name = f'{source_pop}_{post_population}'
        
        if conn_name not in circuit.connectivity.conductance_matrices:
            continue
        
        conductance_matrix = circuit.connectivity.conductance_matrices[conn_name]
        conductances = conductance_matrix.conductances  # [n_pre, n_post]
        
        # Convert to numpy for easier indexing
        if hasattr(conductances, 'cpu'):
            conductances_np = conductances.cpu().numpy()
        else:
            conductances_np = np.array(conductances)
        
        # Check if we need to separate by opsin expression
        separate_by_opsin = (source_pop == target_population)
        
        for conn_idx, classification in classifications.items():
            if conn_idx not in weights_by_connectivity:
                weights_by_connectivity[conn_idx] = {}
            
            if source_pop not in weights_by_connectivity[conn_idx]:
                weights_by_connectivity[conn_idx][source_pop] = {}
            
            # Get opsin expression if needed
            if separate_by_opsin:
                # Get opsin expression from first trial of this connectivity
                trials = trials_by_connectivity[conn_idx]
                opsin_expression = trials[0].opsin_expression
                if hasattr(opsin_expression, 'cpu'):
                    opsin_expression = opsin_expression.cpu().numpy()
                
                # Create masks for opsin+ and opsin- cells
                opsin_plus_mask = opsin_expression >= expression_threshold
                opsin_minus_mask = opsin_expression < expression_threshold
                
                n_opsin_plus = np.sum(opsin_plus_mask)
                n_opsin_minus = np.sum(opsin_minus_mask)
                
                weights_by_connectivity[conn_idx][source_pop]['n_cells_opsin_plus'] = n_opsin_plus
                weights_by_connectivity[conn_idx][source_pop]['n_cells_opsin_minus'] = n_opsin_minus
            
            # Process each response type
            for response_type in ['excited', 'suppressed']:
                cell_key = f'{response_type}_cells'
                if cell_key not in classification:
                    # Initialize empty arrays for all variants
                    weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}'] = np.array([])
                    weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}'] = np.array([])
                    
                    if separate_by_opsin:
                        weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}_opsin_plus'] = np.array([])
                        weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}_opsin_minus'] = np.array([])
                        weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}_opsin_plus'] = np.array([])
                        weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}_opsin_minus'] = np.array([])
                    continue
                
                cell_indices = classification[cell_key]
                
                if len(cell_indices) > 0:
                    # Extract all weights to these post-synaptic cells
                    weights_to_cells = conductances_np[:, cell_indices]  # [n_pre, n_post_response]
                    
                    # Standard analysis (all pre-synaptic cells)
                    individual_weights = weights_to_cells[weights_to_cells > 0]
                    total_input = np.sum(weights_to_cells, axis=0)
                    
                    weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}'] = individual_weights
                    weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}'] = total_input
                    
                    # Opsin-specific analysis (if applicable)
                    if separate_by_opsin:
                        # Opsin+ cells -> post-synaptic cells
                        weights_opsin_plus = weights_to_cells[opsin_plus_mask, :]  # [n_opsin+, n_post_response]
                        individual_weights_opsin_plus = weights_opsin_plus[weights_opsin_plus > 0]
                        total_input_opsin_plus = np.sum(weights_opsin_plus, axis=0)
                        
                        # Opsin- cells -> post-synaptic cells
                        weights_opsin_minus = weights_to_cells[opsin_minus_mask, :]  # [n_opsin-, n_post_response]
                        individual_weights_opsin_minus = weights_opsin_minus[weights_opsin_minus > 0]
                        total_input_opsin_minus = np.sum(weights_opsin_minus, axis=0)
                        
                        # Store
                        weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}_opsin_plus'] = individual_weights_opsin_plus
                        weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}_opsin_minus'] = individual_weights_opsin_minus
                        weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}_opsin_plus'] = total_input_opsin_plus
                        weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}_opsin_minus'] = total_input_opsin_minus
                else:
                    # No cells of this response type
                    weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}'] = np.array([])
                    weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}'] = np.array([])
                    
                    if separate_by_opsin:
                        weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}_opsin_plus'] = np.array([])
                        weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}_opsin_minus'] = np.array([])
                        weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}_opsin_plus'] = np.array([])
                        weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}_opsin_minus'] = np.array([])
    
    return weights_by_connectivity



def compute_weight_statistics_by_connectivity(
    weights_by_connectivity: Dict,
    source_populations: List[str],
    target_population: str  # to identify which source needs opsin separation
) -> Dict:
    """
    Compute statistics for each connectivity instance. Computes separate statistics for opsin+ and opsin- cells.
    
    Args:
        weights_by_connectivity: Output from extract_weight_distributions_by_connectivity()
        source_populations: List of source populations
        target_population: Which population was stimulated (for opsin separation)
        
    Returns:
        dict: {
            connectivity_idx: {
                source_population: {
                    # Standard statistics (all cells)
                    'mean_excited': float,
                    'mean_suppressed': float,
                    'mean_diff_exc_sup': float,
                    ...
                    
                    # If source == target, also include:
                    'mean_excited_opsin_plus': float,
                    'mean_excited_opsin_minus': float,
                    'mean_suppressed_opsin_plus': float,
                    'mean_suppressed_opsin_minus': float,
                    'mean_diff_exc_sup_opsin_plus': float,
                    'mean_diff_exc_sup_opsin_minus': float,
                    'mean_diff_opsin_plus_minus_to_excited': float,  # opsin+ - opsin- -> excited
                    'mean_diff_opsin_plus_minus_to_suppressed': float,  # opsin+ - opsin- -> suppressed
                    ...
                }
            }
        }
    """
    stats_by_connectivity = {}
    
    for conn_idx, conn_data in weights_by_connectivity.items():
        stats_by_connectivity[conn_idx] = {}
        
        for source_pop in source_populations:
            if source_pop not in conn_data:
                continue
            
            source_data = conn_data[source_pop]
            
            # Check if this source has opsin separation
            has_opsin_separation = ('weights_excited_opsin_plus' in source_data)
            
            # Standard statistics (all cells)
            weights_excited = source_data['weights_excited']
            weights_suppressed = source_data['weights_suppressed']
            total_input_excited = source_data['total_input_excited']
            total_input_suppressed = source_data['total_input_suppressed']
            
            stats = {
                # Individual synapse statistics
                'mean_excited': np.mean(weights_excited) if len(weights_excited) > 0 else np.nan,
                'mean_suppressed': np.mean(weights_suppressed) if len(weights_suppressed) > 0 else np.nan,
                'std_excited': np.std(weights_excited) if len(weights_excited) > 0 else np.nan,
                'std_suppressed': np.std(weights_suppressed) if len(weights_suppressed) > 0 else np.nan,
                
                # Total input statistics
                'mean_total_input_excited': np.mean(total_input_excited) if len(total_input_excited) > 0 else np.nan,
                'mean_total_input_suppressed': np.mean(total_input_suppressed) if len(total_input_suppressed) > 0 else np.nan,
                'std_total_input_excited': np.std(total_input_excited) if len(total_input_excited) > 0 else np.nan,
                'std_total_input_suppressed': np.std(total_input_suppressed) if len(total_input_suppressed) > 0 else np.nan,
                
                # Difference
                'mean_diff_exc_sup': (np.mean(weights_excited) - np.mean(weights_suppressed) 
                                     if len(weights_excited) > 0 and len(weights_suppressed) > 0 
                                     else np.nan),
                
                # Sample sizes
                'n_synapses_excited': len(weights_excited),
                'n_synapses_suppressed': len(weights_suppressed),
                'n_cells_excited': len(total_input_excited),
                'n_cells_suppressed': len(total_input_suppressed)
            }
            
            # Add opsin-specific statistics if available
            if has_opsin_separation:
                # Extract opsin+ and opsin- data
                weights_exc_opsin_plus = source_data['weights_excited_opsin_plus']
                weights_exc_opsin_minus = source_data['weights_excited_opsin_minus']
                weights_sup_opsin_plus = source_data['weights_suppressed_opsin_plus']
                weights_sup_opsin_minus = source_data['weights_suppressed_opsin_minus']
                
                total_exc_opsin_plus = source_data['total_input_excited_opsin_plus']
                total_exc_opsin_minus = source_data['total_input_excited_opsin_minus']
                total_sup_opsin_plus = source_data['total_input_suppressed_opsin_plus']
                total_sup_opsin_minus = source_data['total_input_suppressed_opsin_minus']
                
                # Opsin+ statistics
                stats['mean_excited_opsin_plus'] = np.mean(weights_exc_opsin_plus) if len(weights_exc_opsin_plus) > 0 else np.nan
                stats['mean_suppressed_opsin_plus'] = np.mean(weights_sup_opsin_plus) if len(weights_sup_opsin_plus) > 0 else np.nan
                stats['std_excited_opsin_plus'] = np.std(weights_exc_opsin_plus) if len(weights_exc_opsin_plus) > 0 else np.nan
                stats['std_suppressed_opsin_plus'] = np.std(weights_sup_opsin_plus) if len(weights_sup_opsin_plus) > 0 else np.nan
                
                # Opsin- statistics
                stats['mean_excited_opsin_minus'] = np.mean(weights_exc_opsin_minus) if len(weights_exc_opsin_minus) > 0 else np.nan
                stats['mean_suppressed_opsin_minus'] = np.mean(weights_sup_opsin_minus) if len(weights_sup_opsin_minus) > 0 else np.nan
                stats['std_excited_opsin_minus'] = np.std(weights_exc_opsin_minus) if len(weights_exc_opsin_minus) > 0 else np.nan
                stats['std_suppressed_opsin_minus'] = np.std(weights_sup_opsin_minus) if len(weights_sup_opsin_minus) > 0 else np.nan
                
                # Opsin+ differences (excited - suppressed within opsin+)
                stats['mean_diff_exc_sup_opsin_plus'] = (
                    stats['mean_excited_opsin_plus'] - stats['mean_suppressed_opsin_plus']
                    if not np.isnan(stats['mean_excited_opsin_plus']) and not np.isnan(stats['mean_suppressed_opsin_plus'])
                    else np.nan
                )
                
                # Opsin- differences (excited - suppressed within opsin-)
                stats['mean_diff_exc_sup_opsin_minus'] = (
                    stats['mean_excited_opsin_minus'] - stats['mean_suppressed_opsin_minus']
                    if not np.isnan(stats['mean_excited_opsin_minus']) and not np.isnan(stats['mean_suppressed_opsin_minus'])
                    else np.nan
                )
                
                # Cross-comparison: opsin+ vs opsin- to excited cells
                stats['mean_diff_opsin_plus_minus_to_excited'] = (
                    stats['mean_excited_opsin_plus'] - stats['mean_excited_opsin_minus']
                    if not np.isnan(stats['mean_excited_opsin_plus']) and not np.isnan(stats['mean_excited_opsin_minus'])
                    else np.nan
                )
                
                # Cross-comparison: opsin+ vs opsin- to suppressed cells
                stats['mean_diff_opsin_plus_minus_to_suppressed'] = (
                    stats['mean_suppressed_opsin_plus'] - stats['mean_suppressed_opsin_minus']
                    if not np.isnan(stats['mean_suppressed_opsin_plus']) and not np.isnan(stats['mean_suppressed_opsin_minus'])
                    else np.nan
                )
                
                # Total input comparisons
                stats['mean_total_input_excited_opsin_plus'] = np.mean(total_exc_opsin_plus) if len(total_exc_opsin_plus) > 0 else np.nan
                stats['mean_total_input_excited_opsin_minus'] = np.mean(total_exc_opsin_minus) if len(total_exc_opsin_minus) > 0 else np.nan
                stats['mean_total_input_suppressed_opsin_plus'] = np.mean(total_sup_opsin_plus) if len(total_sup_opsin_plus) > 0 else np.nan
                stats['mean_total_input_suppressed_opsin_minus'] = np.mean(total_sup_opsin_minus) if len(total_sup_opsin_minus) > 0 else np.nan
                
                # Sample sizes
                stats['n_synapses_excited_opsin_plus'] = len(weights_exc_opsin_plus)
                stats['n_synapses_excited_opsin_minus'] = len(weights_exc_opsin_minus)
                stats['n_synapses_suppressed_opsin_plus'] = len(weights_sup_opsin_plus)
                stats['n_synapses_suppressed_opsin_minus'] = len(weights_sup_opsin_minus)
                
                stats['n_cells_opsin_plus'] = source_data['n_cells_opsin_plus']
                stats['n_cells_opsin_minus'] = source_data['n_cells_opsin_minus']
            
            stats_by_connectivity[conn_idx][source_pop] = stats
    
    return stats_by_connectivity




def bootstrap_weight_statistics_nested(
    stats_by_connectivity: Dict,
    source_population: str,
    statistic: str = 'mean_diff_exc_sup',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Bootstrap confidence intervals for weight statistics at connectivity level
    
    Args:
        stats_by_connectivity: Output from compute_weight_statistics_by_connectivity()
        source_population: Source population to analyze
        statistic: Which statistic to bootstrap ('mean_diff_exc_sup', etc.)
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level
        random_seed: Random seed for reproducibility
        
    Returns:
        dict with effect size, CI, p-value, bootstrap distribution
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Extract statistic values across connectivity instances
    values = []
    conn_indices = []
    
    for conn_idx in sorted(stats_by_connectivity.keys()):
        if source_population in stats_by_connectivity[conn_idx]:
            val = stats_by_connectivity[conn_idx][source_population][statistic]
            if not np.isnan(val):
                values.append(val)
                conn_indices.append(conn_idx)
    
    if len(values) == 0:
        return {
            'mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'p_value': np.nan,
            'n_connectivity': 0
        }
    
    values = np.array(values)
    n_conn = len(values)
    
    # Original statistics
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if n_conn > 1 else 0.0
    
    if n_conn == 1:
        return {
            'mean': mean_val,
            'ci_lower': mean_val,
            'ci_upper': mean_val,
            'std': 0.0,
            'p_value': np.nan,
            'n_connectivity': n_conn,
            'connectivity_indices': conn_indices
        }
    
    # Bootstrap
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        resampled = np.random.choice(values, size=n_conn, replace=True)
        bootstrap_means.append(np.mean(resampled))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    # Two-sided p-value: proportion of bootstrap samples with opposite sign
    p_value = np.sum(np.sign(bootstrap_means) != np.sign(mean_val)) / len(bootstrap_means)
    
    return {
        'mean': mean_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': std_val,
        'p_value': p_value,
        'n_connectivity': n_conn,
        'bootstrap_distribution': bootstrap_means,
        'connectivity_indices': conn_indices,
        'original_values': values
    }


def compute_effect_sizes_nested(
    stats_by_connectivity: Dict,
    source_populations: List[str]
) -> Dict:
    """
    Compute Cohen's d effect sizes at connectivity level
    
    For each source population, compute effect size of the difference
    between excited and suppressed cells across connectivity instances.
    
    Args:
        stats_by_connectivity: Output from compute_weight_statistics_by_connectivity()
        source_populations: List of source populations
        
    Returns:
        dict: {
            source_population: {
                'cohens_d': float,
                'mean_diff': float,
                'std_diff': float,
                'n_connectivity': int
            }
        }
    """
    effect_sizes = {}
    
    for source_pop in source_populations:
        differences = []
        
        for conn_idx in sorted(stats_by_connectivity.keys()):
            if source_pop in stats_by_connectivity[conn_idx]:
                diff = stats_by_connectivity[conn_idx][source_pop]['mean_diff_exc_sup']
                if not np.isnan(diff):
                    differences.append(diff)
        
        if len(differences) == 0:
            effect_sizes[source_pop] = {
                'cohens_d': np.nan,
                'mean_diff': np.nan,
                'std_diff': np.nan,
                'n_connectivity': 0
            }
            continue
        
        differences = np.array(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1) if len(differences) > 1 else 0.0
        
        cohens_d = mean_diff / std_diff if std_diff > 0 else (np.inf if mean_diff != 0 else 0.0)
        
        effect_sizes[source_pop] = {
            'cohens_d': cohens_d,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'n_connectivity': len(differences)
        }
    
    return effect_sizes


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_weights_by_average_response_nested(
    nested_results: List,
    circuit,
    target_population: str,
    post_population: str,
    source_populations: List[str],
    stim_start: float,
    stim_duration: float,
    warmup: float,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    n_bootstrap: int = 10000,
    run_pca: bool = False,  
    n_pca_permutations: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Perform comprehensive weight analysis on nested experiment results
    
    This function:
    1. Groups trials by connectivity instance
    2. Averages activity across MEC patterns within each connectivity
    3. Classifies cells as excited/suppressed/unchanged per connectivity
    4. Extracts weight distributions for each response type
    5. Computes statistics at connectivity level
    6. Performs bootstrap analysis treating connectivity as independent units
    
    Args:
        nested_results: List of NestedTrialResult objects
        circuit: DentateCircuit instance
        target_population: Stimulated population
        post_population: Post-synaptic population to analyze
        source_populations: List of source populations to analyze
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period start (ms)
        threshold_std: Classification threshold (std deviations)
        expression_threshold: Opsin expression threshold
        n_bootstrap: Number of bootstrap samples
        run_pca: Whether to include PCA analysis
        n_pca_permutations: Number of permutations for PCA null distribution
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: {
            'classifications': {...},
            'weights_by_connectivity': {...},
            'stats_by_connectivity': {...},
            'bootstrap_results': {
                source_population: {...}
            },
            'effect_sizes': {...},
            'summary_statistics': {...}
        }
    """
    print(f"\n{'='*80}")
    print(f"Nested Weight Analysis: {target_population.upper()} -> {post_population.upper()}")
    print('='*80)
    
    # Organize trials by connectivity
    trials_by_connectivity = organize_nested_trials(nested_results)
    n_conn = len(trials_by_connectivity)
    print(f"Analyzing {n_conn} connectivity instances")
    
    # Classify cells within each connectivity
    classifications = classify_cells_by_connectivity(
        trials_by_connectivity,
        target_population,
        post_population,
        stim_start,
        stim_duration,
        warmup,
        threshold_std,
        expression_threshold
    )
    
    print(f"\nCell classifications per connectivity:")
    for conn_idx in sorted(classifications.keys()):
        class_data = classifications[conn_idx]
        print(f"  Connectivity {conn_idx}: "
              f"Excited={class_data['n_excited']}, "
              f"Suppressed={class_data['n_suppressed']}")
    
    # Extract weight distributions
    weights_by_connectivity = extract_weight_distributions_by_connectivity(
        circuit,
        classifications,
        post_population,
        source_populations,
        target_population=target_population,
        trials_by_connectivity=trials_by_connectivity,
        expression_threshold=expression_threshold
    )
    
    # Compute statistics per connectivity
    stats_by_connectivity = compute_weight_statistics_by_connectivity(
        weights_by_connectivity,
        source_populations,
        target_population=target_population
    )
    
    # Bootstrap analysis for each source population
    bootstrap_results = {}
    
    print(f"\nBootstrap Analysis (n={n_bootstrap}):")
    print(f"{'Source':<8} {'Mean Diff':<20} {'p-value':<12} {'Sig':<5}")
    print(f"{'-'*8} {'-'*20} {'-'*12} {'-'*5}")
    
    for source_pop in source_populations:
        bootstrap_results[source_pop] = bootstrap_weight_statistics_nested(
            stats_by_connectivity,
            source_pop,
            statistic='mean_diff_exc_sup',
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            random_seed=random_seed
        )
        
        result = bootstrap_results[source_pop]
        
        if not np.isnan(result['mean']):
            mean_str = f"{result['mean']:>6.3f} [{result['ci_lower']:>6.3f}, {result['ci_upper']:>6.3f}]"
            sig = '***' if result['p_value'] < 0.001 else ('**' if result['p_value'] < 0.01 else ('*' if result['p_value'] < 0.05 else 'n.s.'))
            print(f"{source_pop.upper():<8} {mean_str:<20} {result['p_value']:<12.4f} {sig:<5}")

        # Bootstrap for opsin-specific comparisons if applicable
        if source_pop == target_population:
            print(f"\n  Opsin-specific analysis for {source_pop.upper()}:")
            
            # Opsin+ excited vs suppressed
            bootstrap_results[f'{source_pop}_opsin_plus'] = bootstrap_weight_statistics_nested(
                stats_by_connectivity,
                source_pop,
                statistic='mean_diff_exc_sup_opsin_plus',
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed
            )
            
            # Opsin- excited vs suppressed
            bootstrap_results[f'{source_pop}_opsin_minus'] = bootstrap_weight_statistics_nested(
                stats_by_connectivity,
                source_pop,
                statistic='mean_diff_exc_sup_opsin_minus',
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed
            )
            
            # Opsin+ vs opsin- to excited cells
            bootstrap_results[f'{source_pop}_opsin_diff_to_excited'] = bootstrap_weight_statistics_nested(
                stats_by_connectivity,
                source_pop,
                statistic='mean_diff_opsin_plus_minus_to_excited',
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed
            )
            
            # Opsin+ vs opsin- to suppressed cells
            bootstrap_results[f'{source_pop}_opsin_diff_to_suppressed'] = bootstrap_weight_statistics_nested(
                stats_by_connectivity,
                source_pop,
                statistic='mean_diff_opsin_plus_minus_to_suppressed',
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed
            )
            
            # Print results
            for key, label in [
                (f'{source_pop}_opsin_plus', 'Opsin+ (exc-sup)'),
                (f'{source_pop}_opsin_minus', 'Opsin- (exc-sup)'),
                (f'{source_pop}_opsin_diff_to_excited', 'Opsin+/- (->excited)'),
                (f'{source_pop}_opsin_diff_to_suppressed', 'Opsin+/- (->suppressed)')
            ]:
                res = bootstrap_results[key]
                if not np.isnan(res['mean']):
                    mean_str = f"{res['mean']:>6.3f} [{res['ci_lower']:>6.3f}, {res['ci_upper']:>6.3f}]"
                    sig = '***' if res['p_value'] < 0.001 else ('**' if res['p_value'] < 0.01 else ('*' if res['p_value'] < 0.05 else 'n.s.'))
                    print(f"    {label:<25} {mean_str:<20} {res['p_value']:<12.4f} {sig:<5}")
            
    # Compute effect sizes
    effect_sizes = compute_effect_sizes_nested(
        stats_by_connectivity,
        source_populations
    )
    
    print(f"\nEffect Sizes (Cohen's d):")
    for source_pop in source_populations:
        es = effect_sizes[source_pop]
        if not np.isnan(es['cohens_d']):
            print(f"  {source_pop.upper()}: d={es['cohens_d']:.3f} "
                  f"(mean_diff={es['mean_diff']:.3f}, std={es['std_diff']:.3f})")

    # PCA Analysis (optional)
    pca_results = None
    if run_pca:
        print(f"\n{'='*80}")
        print("MULTIVARIATE PCA ANALYSIS")
        print('='*80)
        
        try:
            pca_results = analyze_input_weight_patterns_pca(
                nested_results=nested_results,
                circuit=circuit,
                target_population=target_population,
                post_population=post_population,
                source_populations=source_populations,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                n_bootstrap=n_bootstrap,
                n_permutations=n_pca_permutations,
                n_components=None,  # Auto-select
                threshold_std=threshold_std,
                expression_threshold=expression_threshold,
                save_dir=None,  # Will be set externally
                random_seed=random_seed
            )
            
            print("\nPCA Summary:")
            print(f"  Hotelling's T^2: {pca_results['bootstrap_result'].observed_t2:.2f}")
            print(f"  Bootstrap p-value: {pca_results['bootstrap_result'].p_value_bootstrap:.4f}")
            print(f"  Permutation p-value: {pca_results['permutation_result']['p_value_permutation']:.4f}")
            
        except Exception as e:
            print(f"\nWarning: PCA analysis failed with error: {e}")
            print("Continuing without PCA results...")
            pca_results = None
    
            
    # Compute summary statistics across all connectivities
    summary_statistics = _compute_summary_statistics(
        stats_by_connectivity,
        source_populations
    )
    
    print(f"\n{'='*80}\n")
    
    return {
        'classifications': classifications,
        'weights_by_connectivity': weights_by_connectivity,
        'stats_by_connectivity': stats_by_connectivity,
        'bootstrap_results': bootstrap_results,
        'effect_sizes': effect_sizes,
        'pca_results': pca_results,
        'summary_statistics': summary_statistics,
        'metadata': {
            'target_population': target_population,
            'post_population': post_population,
            'n_connectivity_instances': n_conn,
            'threshold_std': threshold_std,
            'expression_threshold': expression_threshold,
            'pca_analysis_performed': run_pca
        }
    }


def _compute_summary_statistics(
    stats_by_connectivity: Dict,
    source_populations: List[str]
) -> Dict:
    """
    Compute grand mean and between-connectivity variance
    
    Args:
        stats_by_connectivity: Output from compute_weight_statistics_by_connectivity()
        source_populations: List of source populations
        
    Returns:
        dict with summary statistics
    """
    summary = {}
    
    for source_pop in source_populations:
        # Collect statistics across connectivities
        mean_excited_vals = []
        mean_suppressed_vals = []
        mean_diff_vals = []
        
        for conn_idx in sorted(stats_by_connectivity.keys()):
            if source_pop in stats_by_connectivity[conn_idx]:
                data = stats_by_connectivity[conn_idx][source_pop]
                
                if not np.isnan(data['mean_excited']):
                    mean_excited_vals.append(data['mean_excited'])
                if not np.isnan(data['mean_suppressed']):
                    mean_suppressed_vals.append(data['mean_suppressed'])
                if not np.isnan(data['mean_diff_exc_sup']):
                    mean_diff_vals.append(data['mean_diff_exc_sup'])
        
        summary[source_pop] = {
            # Grand means (mean of connectivity means)
            'grand_mean_excited': np.mean(mean_excited_vals) if len(mean_excited_vals) > 0 else np.nan,
            'grand_mean_suppressed': np.mean(mean_suppressed_vals) if len(mean_suppressed_vals) > 0 else np.nan,
            'grand_mean_diff': np.mean(mean_diff_vals) if len(mean_diff_vals) > 0 else np.nan,
            
            # Between-connectivity variance
            'between_conn_std_excited': np.std(mean_excited_vals) if len(mean_excited_vals) > 1 else 0.0,
            'between_conn_std_suppressed': np.std(mean_suppressed_vals) if len(mean_suppressed_vals) > 1 else 0.0,
            'between_conn_std_diff': np.std(mean_diff_vals) if len(mean_diff_vals) > 1 else 0.0,
            
            # Sample sizes
            'n_connectivity_with_data': len(mean_diff_vals)
        }
    
    return summary


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_weights_by_average_response_nested(
    analysis_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 14),
) -> plt.Figure:
    """
    Create visualizations of nested weight analysis.
    Includes opsin+/- panels when source == target population.
    
    Creates multi-panel figure showing:
    - Panel A: Weight distributions by connectivity instance (violin plots)
    - Panel B: Mean weights with between-connectivity variance (bar + error bars)
    - Panel C: Effect sizes with bootstrap CIs (forest plot)
    - Panel D: Bootstrap distributions for each source
    - Panel E: Opsin+/- comparison (when source == target)
    
    Args:
        analysis_results: Output from analyze_weights_by_average_response_nested()
        include_pca_panel: Whether to add PCA scatter subplot
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    metadata = analysis_results['metadata']
    target = metadata['target_population']
    post_pop = metadata['post_population']
    
    weights_by_conn = analysis_results['weights_by_connectivity']
    stats_by_conn = analysis_results['stats_by_connectivity']
    bootstrap_results = analysis_results['bootstrap_results']
    effect_sizes = analysis_results['effect_sizes']
    summary_stats = analysis_results['summary_statistics']
    
    # Get source populations
    source_populations = sorted([k for k in bootstrap_results.keys() 
                                if not k.endswith(('_opsin_plus', '_opsin_minus', 
                                                   '_opsin_diff_to_excited', 
                                                   '_opsin_diff_to_suppressed'))])
    n_sources = len(source_populations)
    n_conn = metadata['n_connectivity_instances']
    
    # Check if we have opsin-specific data
    has_opsin_data = any(f'{src}_opsin_plus' in bootstrap_results 
                         for src in source_populations)
    
    # Adjust figure layout based on whether we have opsin data
    if has_opsin_data:
        # 5 rows: violin, bars, forest (all sources), bootstrap, opsin details
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 6))
        gs = fig.add_gridspec(5, n_sources, hspace=0.4, wspace=0.3,
                              top=0.93, bottom=0.05, left=0.07, right=0.97,
                              height_ratios=[1, 1, 1, 1, 1.2])
    else:
        # Original 4 rows
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, n_sources, hspace=0.4, wspace=0.3,
                              top=0.93, bottom=0.05, left=0.07, right=0.97)
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db',
        'unchanged': '#95a5a6',
        'opsin_plus': '#e74c3c',    # Red for opsin+
        'opsin_minus': '#3498db'     # Blue for opsin-
    }
    
    # ========================================================================
    # Panel A: Weight distributions by connectivity (violin plots)
    # ========================================================================
    for source_idx, source_pop in enumerate(source_populations):
        ax = fig.add_subplot(gs[0, source_idx])
        
        # Collect data for violin plot
        data_by_conn = {conn_idx: {'excited': [], 'suppressed': []} 
                        for conn_idx in sorted(weights_by_conn.keys())}
        
        for conn_idx in sorted(weights_by_conn.keys()):
            if source_pop in weights_by_conn[conn_idx]:
                data = weights_by_conn[conn_idx][source_pop]
                data_by_conn[conn_idx]['excited'] = data['weights_excited']
                data_by_conn[conn_idx]['suppressed'] = data['weights_suppressed']
        
        # Plot violins for each connectivity
        positions = []
        data_to_plot = []
        plot_colors = []
        
        for i, conn_idx in enumerate(sorted(data_by_conn.keys())):
            if len(data_by_conn[conn_idx]['excited']) > 0:
                positions.append(i * 3)
                data_to_plot.append(data_by_conn[conn_idx]['excited'])
                plot_colors.append(colors['excited'])
            
            if len(data_by_conn[conn_idx]['suppressed']) > 0:
                positions.append(i * 3 + 1)
                data_to_plot.append(data_by_conn[conn_idx]['suppressed'])
                plot_colors.append(colors['suppressed'])
        
        if len(data_to_plot) > 0:
            parts = ax.violinplot(data_to_plot, positions=positions,
                                 showmeans=True, showmedians=False, widths=0.7)
            
            for pc, color in zip(parts['bodies'], plot_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
        
        ax.set_ylabel('Synaptic Weight (nS)', fontsize=10)
        ax.set_title(f'{source_pop.upper()} $\\rightarrow$ {post_pop.upper()}',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([])
        
        if positions:
            conn_positions = [i * 3 + 0.5 for i in range(n_conn)]
            ax.set_xticks(conn_positions)
            ax.set_xticklabels([f'C{i}' for i in range(n_conn)], fontsize=8)
    
    # ========================================================================
    # Panel B: Mean weights with between-connectivity error bars
    # ========================================================================
    for source_idx, source_pop in enumerate(source_populations):
        ax = fig.add_subplot(gs[1, source_idx])
        
        conn_indices = sorted(stats_by_conn.keys())
        excited_means = []
        suppressed_means = []
        
        for conn_idx in conn_indices:
            if source_pop in stats_by_conn[conn_idx]:
                data = stats_by_conn[conn_idx][source_pop]
                excited_means.append(data['mean_excited'] if not np.isnan(data['mean_excited']) else np.nan)
                suppressed_means.append(data['mean_suppressed'] if not np.isnan(data['mean_suppressed']) else np.nan)
            else:
                excited_means.append(np.nan)
                suppressed_means.append(np.nan)
        
        excited_means = np.array(excited_means)
        suppressed_means = np.array(suppressed_means)
        
        if np.all(np.isnan(excited_means)) and np.all(np.isnan(suppressed_means)):
            ax.text(0.5, 0.5, f'No {source_pop.upper()} connections',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic')
            ax.set_ylabel('Mean Weight (nS)', fontsize=10)
            ax.set_xlabel('Connectivity Instance', fontsize=10)
            continue
        
        x_pos = np.arange(len(conn_indices))
        width = 0.35
        
        ax.bar(x_pos - width/2, excited_means, width,
               label='Excited', color=colors['excited'], alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax.bar(x_pos + width/2, suppressed_means, width,
               label='Suppressed', color=colors['suppressed'], alpha=0.7,
               edgecolor='black', linewidth=1.5)
        
        if source_pop in summary_stats:
            summary = summary_stats[source_pop]
            if not np.isnan(summary['grand_mean_excited']):
                ax.axhline(summary['grand_mean_excited'], color=colors['excited'],
                          linestyle='--', linewidth=2, alpha=0.5)
            if not np.isnan(summary['grand_mean_suppressed']):
                ax.axhline(summary['grand_mean_suppressed'], color=colors['suppressed'],
                          linestyle='--', linewidth=2, alpha=0.5)
                
        ax.set_ylabel('Mean Weight (nS)', fontsize=10)
        ax.set_xlabel('Connectivity Instance', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'C{i}' for i in conn_indices], fontsize=9)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        boot_result = bootstrap_results[source_pop]
        if not np.isnan(boot_result['mean']):
            ax.text(0.02, 0.98,
                   f"$\\Delta$ = {boot_result['mean']:.3f} nS\n"
                   f"95% CI: [{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]\n"
                   f"p = {boot_result['p_value']:.4f}",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========================================================================
    # Panel C: Effect sizes (forest plot across all sources)
    # ========================================================================
    ax_forest = fig.add_subplot(gs[2, :])
    
    y_positions = np.arange(n_sources)
    x_min, x_max = np.inf, -np.inf
    
    for i, source_pop in enumerate(source_populations):
        boot_result = bootstrap_results[source_pop]
        
        if not np.isnan(boot_result['mean']):
            mean_val = boot_result['mean']
            ci_lower = boot_result['ci_lower']
            ci_upper = boot_result['ci_upper']
            p_val = boot_result['p_value']
            
            x_min = min(x_min, ci_lower)
            x_max = max(x_max, ci_upper)
            
            color = 'red' if p_val < 0.001 else ('orange' if p_val < 0.05 else 'gray')
            
            ax_forest.plot(mean_val, y_positions[i], 'o',
                          color=color, markersize=12, zorder=3)
            ax_forest.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]],
                          '-', color=color, linewidth=3, zorder=2)
            
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            elif p_val < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            
            ax_forest.text(0.95, y_positions[i],
                           f"{mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {stars}",
                           va='center', ha='right', fontsize=9,
                           transform=ax_forest.get_yaxis_transform())
    
    if np.isfinite(x_min) and np.isfinite(x_max):
        x_range = x_max - x_min
        ax_forest.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax_forest.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax_forest.set_yticks(y_positions)
    ax_forest.set_yticklabels([f'{s.upper()} $\\rightarrow$ {post_pop.upper()}' 
                               for s in source_populations], fontsize=10)
    ax_forest.set_xlabel('Weight Difference: Excited - Suppressed (nS)\n(Bootstrap 95% CI)',
                         fontsize=11, fontweight='bold')
    ax_forest.set_title('Between-Connectivity Weight Differences (All Cells)',
                        fontsize=12, fontweight='bold')
    ax_forest.grid(True, alpha=0.3, axis='x')
    
    # ========================================================================
    # Panel D: Bootstrap distributions
    # ========================================================================
    for source_idx, source_pop in enumerate(source_populations):
        ax = fig.add_subplot(gs[3, source_idx])
        
        boot_result = bootstrap_results[source_pop]
        
        if 'bootstrap_distribution' in boot_result and boot_result['bootstrap_distribution'] is not None:
            bootstrap_dist = boot_result['bootstrap_distribution']
            
            ax.hist(bootstrap_dist, bins=50, color='steelblue',
                    alpha=0.7, edgecolor='black')
            
            ax.axvline(boot_result['mean'], color='red',
                       linestyle='--', linewidth=2, label='Observed', zorder=3)
            ax.axvline(boot_result['ci_lower'], color='orange',
                       linestyle=':', linewidth=2, zorder=3)
            ax.axvline(boot_result['ci_upper'], color='orange',
                       linestyle=':', linewidth=2, label='95% CI', zorder=3)
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=2)
            
            ax.set_xlabel('Weight Difference (nS)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            es = effect_sizes[source_pop]
            ax.text(0.02, 0.98,
                   f"Cohen's d = {es['cohens_d']:.3f}\n"
                   f"N conn = {boot_result['n_connectivity']}",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ========================================================================
    # Panel E: Opsin-specific analysis (when source == target)
    # ========================================================================
    if has_opsin_data:
        # Create a subplot that spans all columns for opsin analysis
        ax_opsin = fig.add_subplot(gs[4, :])
        
        # Collect opsin-specific bootstrap results
        opsin_data = []
        for source_pop in source_populations:
            if source_pop == target:  # Only for target population
                # Check for opsin+ data
                opsin_keys = [
                    (f'{source_pop}_opsin_plus', 'Opsin$^+$ (exc$-$sup)'),
                    (f'{source_pop}_opsin_minus', 'Opsin$^-$ (exc$-$sup)'),
                    (f'{source_pop}_opsin_diff_to_excited', 'Opsin$^+$ $-$ Opsin$^-$ ($\\rightarrow$ excited)'),
                    (f'{source_pop}_opsin_diff_to_suppressed', 'Opsin$^+$ $-$ Opsin$^-$ ($\\rightarrow$ suppressed)')
                ]
                
                for key, label in opsin_keys:
                    if key in bootstrap_results:
                        res = bootstrap_results[key]
                        if not np.isnan(res['mean']):
                            opsin_data.append({
                                'label': label,
                                'mean': res['mean'],
                                'ci_lower': res['ci_lower'],
                                'ci_upper': res['ci_upper'],
                                'p_value': res['p_value']
                            })
        
        if len(opsin_data) > 0:
            y_pos_opsin = np.arange(len(opsin_data))
            
            for i, data in enumerate(opsin_data):
                p_val = data['p_value']
                color = 'red' if p_val < 0.001 else ('darkorange' if p_val < 0.01 else ('orange' if p_val < 0.05 else 'gray'))
                
                ax_opsin.plot(data['mean'], y_pos_opsin[i], 'o',
                            color=color, markersize=12, zorder=3)
                ax_opsin.plot([data['ci_lower'], data['ci_upper']], 
                             [y_pos_opsin[i], y_pos_opsin[i]],
                             '-', color=color, linewidth=3, zorder=2)
                
                stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
                
                ax_opsin.text(0.95, y_pos_opsin[i],
                             f"{data['mean']:.3f} [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}] {stars}",
                             va='center', ha='right', fontsize=9,
                             transform=ax_opsin.get_yaxis_transform())
            
            ax_opsin.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
            ax_opsin.set_yticks(y_pos_opsin)
            ax_opsin.set_yticklabels([d['label'] for d in opsin_data], fontsize=10)
            ax_opsin.set_xlabel('Weight Difference (nS) | Bootstrap 95% CI', fontsize=11, fontweight='bold')
            ax_opsin.set_title(f'Opsin Expression Analysis: {target.upper()}$^{{+/-}}$ $\\rightarrow$ {post_pop.upper()}',
                             fontsize=12, fontweight='bold')
            ax_opsin.grid(True, alpha=0.3, axis='x')
            
            # Add legend for significance
            legend_elements = [
                mpatches.Patch(color='red', label='p < 0.001 (***)'),
                mpatches.Patch(color='darkorange', label='p < 0.01 (**)'),
                mpatches.Patch(color='orange', label='p < 0.05 (*)'),
                mpatches.Patch(color='gray', label='n.s.')
            ]
            ax_opsin.legend(handles=legend_elements, loc='lower right', fontsize=8)
        else:
            ax_opsin.text(0.5, 0.5, 'No opsin-specific data available',
                         ha='center', va='center', transform=ax_opsin.transAxes,
                         fontsize=12, style='italic')
            ax_opsin.axis('off')
    
    # Overall title
    if has_opsin_data:
        fig.suptitle(f'Nested Weight Analysis: {target.upper()} Stimulation $\\rightarrow$ {post_pop.upper()} Response\n'
                    f'(N={n_conn} connectivity instances | Including Opsin$^{{+/-}}$ Separation)',
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Nested Weight Analysis: {target.upper()} Stimulation $\\rightarrow$ {post_pop.upper()} Response\n'
                    f'(N={n_conn} connectivity instances, averaged across MEC patterns)',
                    fontsize=14, fontweight='bold')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved nested weight analysis plot to: {save_path}")
    
    return fig


def plot_connectivity_weight_comparison(
    analysis_results: Dict,
    connectivity_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """Plot detailed comparison across specific connectivity instances.
    
    Shows weight distributions and statistics for selected
    connectivities to visualize between-connectivity variance. Shows
    opsin+/- separation when source == target population.
    
    Args:
        analysis_results: Output from analyze_weights_by_average_response_nested()
        connectivity_indices: Which connectivities to plot (default: all)
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object

    """
    metadata = analysis_results['metadata']
    target = metadata['target_population']
    post_pop = metadata['post_population']
    weights_by_conn = analysis_results['weights_by_connectivity']
    stats_by_conn = analysis_results['stats_by_connectivity']
    
    if connectivity_indices is None:
        connectivity_indices = sorted(weights_by_conn.keys())
    
    n_conn = len(connectivity_indices)
    source_populations = sorted(list(weights_by_conn[connectivity_indices[0]].keys()))
    n_sources = len(source_populations)
    
    # Check if we have opsin data for any source
    has_opsin_data = {}
    for source_pop in source_populations:
        has_opsin_data[source_pop] = (
            source_pop == target and 
            'weights_excited_opsin_plus' in weights_by_conn[connectivity_indices[0]][source_pop]
        )
    
    # Calculate number of columns needed
    # For sources with opsin data: show 3 columns (all, opsin+, opsin-)
    # For sources without: show 1 column
    n_cols = sum(3 if has_opsin_data[s] else 1 for s in source_populations)
    
    fig, axes = plt.subplots(n_conn, n_cols, figsize=(figsize[0], figsize[1]),
                            squeeze=False)
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db',
        'opsin_plus': '#e74c3c',
        'opsin_minus': '#3498db'
    }
    
    col_idx = 0
    for source_pop in source_populations:
        has_opsin = has_opsin_data[source_pop]
        
        # Determine how many columns this source needs
        source_cols = 3 if has_opsin else 1
        
        for conn_idx_pos, conn_idx in enumerate(connectivity_indices):
            # Get data
            weights_data = weights_by_conn[conn_idx][source_pop]
            stats_data = stats_by_conn[conn_idx][source_pop]
            
            # Column 1: All cells (standard analysis)
            ax_all = axes[conn_idx_pos, col_idx]
            
            weights_excited = weights_data['weights_excited']
            weights_suppressed = weights_data['weights_suppressed']
            
            if len(weights_excited) > 0 and len(weights_suppressed) > 0:
                bins = np.linspace(0, max(np.max(weights_excited), np.max(weights_suppressed)), 30)
                
                ax_all.hist(weights_excited, bins=bins, alpha=0.6,
                           color=colors['excited'], label='Excited',
                           edgecolor='black')
                ax_all.hist(weights_suppressed, bins=bins, alpha=0.6,
                           color=colors['suppressed'], label='Suppressed',
                           edgecolor='black')
                
                ax_all.axvline(stats_data['mean_excited'], color=colors['excited'],
                              linestyle='--', linewidth=2)
                ax_all.axvline(stats_data['mean_suppressed'], color=colors['suppressed'],
                              linestyle='--', linewidth=2)
                
                ax_all.text(0.98, 0.98,
                           f"$\\Delta$ = {stats_data['mean_diff_exc_sup']:.3f} nS\n"
                           f"n_exc = {stats_data['n_synapses_excited']}\n"
                           f"n_sup = {stats_data['n_synapses_suppressed']}",
                           transform=ax_all.transAxes, fontsize=8,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax_all.set_xlabel('Weight (nS)', fontsize=9)
            ax_all.set_ylabel('Count', fontsize=9)
            
            if conn_idx_pos == 0:
                title = f'{source_pop.upper()}\n(All Cells)' if has_opsin else source_pop.upper()
                ax_all.set_title(title, fontsize=10, fontweight='bold')
            
            if col_idx == 0:
                ax_all.text(-0.3, 0.5, f'Connectivity {conn_idx}',
                           transform=ax_all.transAxes, fontsize=11, fontweight='bold',
                           rotation=90, verticalalignment='center')
            
            if conn_idx_pos == 0 and col_idx == 0:
                ax_all.legend(fontsize=8, loc='upper right')
            
            ax_all.grid(True, alpha=0.3)
            
            # If opsin data available, add two more columns
            if has_opsin:
                # Column 2: Opsin+ cells
                ax_opsin_plus = axes[conn_idx_pos, col_idx + 1]
                
                weights_exc_opsin_plus = weights_data['weights_excited_opsin_plus']
                weights_sup_opsin_plus = weights_data['weights_suppressed_opsin_plus']
                
                if len(weights_exc_opsin_plus) > 0 and len(weights_sup_opsin_plus) > 0:
                    bins = np.linspace(0, max(np.max(weights_exc_opsin_plus), 
                                             np.max(weights_sup_opsin_plus)), 30)
                    
                    ax_opsin_plus.hist(weights_exc_opsin_plus, bins=bins, alpha=0.6,
                                      color=colors['excited'], label='Excited',
                                      edgecolor='black')
                    ax_opsin_plus.hist(weights_sup_opsin_plus, bins=bins, alpha=0.6,
                                      color=colors['suppressed'], label='Suppressed',
                                      edgecolor='black')
                    
                    ax_opsin_plus.axvline(stats_data['mean_excited_opsin_plus'], 
                                         color=colors['excited'], linestyle='--', linewidth=2)
                    ax_opsin_plus.axvline(stats_data['mean_suppressed_opsin_plus'],
                                         color=colors['suppressed'], linestyle='--', linewidth=2)
                    
                    ax_opsin_plus.text(0.98, 0.98,
                                      f"$\\Delta$ = {stats_data['mean_diff_exc_sup_opsin_plus']:.3f} nS\n"
                                      f"n_exc = {stats_data['n_synapses_excited_opsin_plus']}\n"
                                      f"n_sup = {stats_data['n_synapses_suppressed_opsin_plus']}",
                                      transform=ax_opsin_plus.transAxes, fontsize=8,
                                      verticalalignment='top', horizontalalignment='right',
                                      bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                
                ax_opsin_plus.set_xlabel('Weight (nS)', fontsize=9)
                ax_opsin_plus.set_ylabel('Count', fontsize=9)
                
                if conn_idx_pos == 0:
                    ax_opsin_plus.set_title(f'{source_pop.upper()}\n(Opsin$^+$)',
                                           fontsize=10, fontweight='bold', color=colors['opsin_plus'])
                
                ax_opsin_plus.grid(True, alpha=0.3)
                
                # Column 3: Opsin- cells
                ax_opsin_minus = axes[conn_idx_pos, col_idx + 2]
                
                weights_exc_opsin_minus = weights_data['weights_excited_opsin_minus']
                weights_sup_opsin_minus = weights_data['weights_suppressed_opsin_minus']
                
                if len(weights_exc_opsin_minus) > 0 and len(weights_sup_opsin_minus) > 0:
                    bins = np.linspace(0, max(np.max(weights_exc_opsin_minus),
                                             np.max(weights_sup_opsin_minus)), 30)
                    
                    ax_opsin_minus.hist(weights_exc_opsin_minus, bins=bins, alpha=0.6,
                                       color=colors['excited'], label='Excited',
                                       edgecolor='black')
                    ax_opsin_minus.hist(weights_sup_opsin_minus, bins=bins, alpha=0.6,
                                       color=colors['suppressed'], label='Suppressed',
                                       edgecolor='black')
                    
                    ax_opsin_minus.axvline(stats_data['mean_excited_opsin_minus'],
                                          color=colors['excited'], linestyle='--', linewidth=2)
                    ax_opsin_minus.axvline(stats_data['mean_suppressed_opsin_minus'],
                                          color=colors['suppressed'], linestyle='--', linewidth=2)
                    
                    ax_opsin_minus.text(0.98, 0.98,
                                       f"$\\Delta$ = {stats_data['mean_diff_exc_sup_opsin_minus']:.3f} nS\n"
                                       f"n_exc = {stats_data['n_synapses_excited_opsin_minus']}\n"
                                       f"n_sup = {stats_data['n_synapses_suppressed_opsin_minus']}",
                                       transform=ax_opsin_minus.transAxes, fontsize=8,
                                       verticalalignment='top', horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                ax_opsin_minus.set_xlabel('Weight (nS)', fontsize=9)
                ax_opsin_minus.set_ylabel('Count', fontsize=9)
                
                if conn_idx_pos == 0:
                    ax_opsin_minus.set_title(f'{source_pop.upper()}\n(Opsin$^-$)',
                                            fontsize=10, fontweight='bold', color=colors['opsin_minus'])
                
                ax_opsin_minus.grid(True, alpha=0.3)
        
        # Move to next set of columns
        col_idx += source_cols
    
    # Overall title
    opsin_note = ' | Opsin$^{+/-}$ Separation Shown' if any(has_opsin_data.values()) else ''
    fig.suptitle(f'Weight Distributions by Connectivity Instance\n'
                f'{target.upper()} $\\rightarrow$ {post_pop.upper()}{opsin_note}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved connectivity comparison plot to: {save_path}")
    
    return fig

def plot_summary_forest_plot_all_targets(
    analysis_results_by_target: Dict[str, Dict],
    stimulated_population: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Create unified forest plot showing weight differences across all postsynaptic targets
    
    This function consolidates results from multiple nested weight analyses
    (one per postsynaptic population) into a single comprehensive forest plot.
    
    For each target population, shows:
    1. All source -> target comparisons (standard, all cells)
    2. If target has opsin data (source == stimulated pop), shows:
       - Opsin+ (excited - suppressed) -> this target
       - Opsin- (excited - suppressed) -> this target  
       - Opsin+/- difference -> excited cells of this target
       - Opsin+/- difference -> suppressed cells of this target

    Appropriate legend:

    Forest plot showing mean synaptic weight differences from each
    source population to excited versus suppressed post-synaptic cells
    (excited - suppressed). Positive values (red) indicate that cells
    receiving stronger input from this source tend to become excited,
    suggesting the pathway contributes to excitatory
    responses. Negative values (blue) indicate association with
    suppression. Values near zero (gray) indicate the source provides
    input independent of response type. For the optogenetically
    stimulated population (squares), opsin-specific comparisons
    separate pre-synaptic cells by expression level: "Opsin+
    (exc-sup)" and "Opsin- (exc-sup)" show whether opsin-expressing or
    non-expressing cells preferentially connect to excited versus
    suppressed targets; "Opsin+/- diff → excited" and "Opsin+/- diff →
    suppressed" show whether excited or suppressed targets
    preferentially receive input from opsin+ versus opsin-
    cells. Error bars represent 95% bootstrap confidence intervals (N
    = X connectivity instances). Significance levels: ***p < 0.001,
    **p < 0.01, *p < 0.05.
    
    Args:
        analysis_results_by_target: Dict mapping postsynaptic population names to
            their respective analysis results from analyze_weights_by_average_response_nested()
        stimulated_population: Name of stimulated population
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object

    """
    # Organize data by target population
    data_by_target = {}
    
    for target_pop, analysis_results in analysis_results_by_target.items():
        bootstrap_results = analysis_results['bootstrap_results']
        
        data_by_target[target_pop] = {
            'standard': [],  # Standard comparisons (all cells)
            'opsin': []      # Opsin-specific comparisons
        }
        
        # Standard comparisons (all cells) - all sources
        for source_pop in sorted([k for k in bootstrap_results.keys() 
                                 if not k.endswith(('_opsin_plus', '_opsin_minus',
                                                   '_opsin_diff_to_excited', 
                                                   '_opsin_diff_to_suppressed'))]):
            boot_result = bootstrap_results[source_pop]
            if not np.isnan(boot_result['mean']):
                data_by_target[target_pop]['standard'].append({
                    'source': source_pop,
                    'target': target_pop,
                    'label': f"{source_pop.upper()} $\\rightarrow$ {target_pop.upper()}",
                    'mean': boot_result['mean'],
                    'ci_lower': boot_result['ci_lower'],
                    'ci_upper': boot_result['ci_upper'],
                    'p_value': boot_result['p_value'],
                    'n_connectivity': boot_result['n_connectivity']
                })
        
        # Opsin-specific comparisons (only if stimulated pop is a source to this target)
        # Check if we have opsin data for this target
        stim_key = f'{stimulated_population}_opsin_plus'
        if stim_key in bootstrap_results:
            # We have opsin data - this target receives input from stimulated population
            opsin_comparisons = [
                (f'{stimulated_population}_opsin_plus', 
                 f"  {stimulated_population.upper()} Opsin$^+$ $\\rightarrow$ {target_pop.upper()} (exc$-$sup)"),
                (f'{stimulated_population}_opsin_minus',
                 f"  {stimulated_population.upper()} Opsin$^-$ $\\rightarrow$ {target_pop.upper()} (exc$-$sup)"),
                (f'{stimulated_population}_opsin_diff_to_excited',
                 f"  {stimulated_population.upper()} Opsin$^{{+/-}}$ diff $\\rightarrow$ {target_pop.upper()} excited"),
                (f'{stimulated_population}_opsin_diff_to_suppressed',
                 f"  {stimulated_population.upper()} Opsin$^{{+/-}}$ diff $\\rightarrow$ {target_pop.upper()} suppressed")
            ]
            
            for key, label in opsin_comparisons:
                if key in bootstrap_results:
                    boot_result = bootstrap_results[key]
                    if not np.isnan(boot_result['mean']):
                        data_by_target[target_pop]['opsin'].append({
                            'source': stimulated_population,
                            'target': target_pop,
                            'label': label,
                            'mean': boot_result['mean'],
                            'ci_lower': boot_result['ci_lower'],
                            'ci_upper': boot_result['ci_upper'],
                            'p_value': boot_result['p_value'],
                            'n_connectivity': boot_result['n_connectivity']
                        })
    
    # Flatten into single list with target grouping
    data_rows = []
    target_order = {'gc': 0, 'mc': 1, 'pv': 2, 'sst': 3}
    
    for target_pop in sorted(data_by_target.keys(), key=lambda x: target_order.get(x, 99)):
        # Add standard comparisons for this target
        # Sort by source population
        source_order = {'gc': 0, 'mc': 1, 'pv': 2, 'sst': 3, 'mec': 4}
        standard_rows = sorted(data_by_target[target_pop]['standard'],
                              key=lambda x: source_order.get(x['source'], 99))
        data_rows.extend(standard_rows)
        
        # Add opsin comparisons for this target (if any)
        data_rows.extend(data_by_target[target_pop]['opsin'])
    
    if len(data_rows) == 0:
        print("Warning: No valid data to plot in summary forest plot")
        return None
    
    n_rows = len(data_rows)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n_rows)
    
    # Define colors and markers
    def get_color(p_val):
        if p_val < 0.001:
            return 'red'
        elif p_val < 0.01:
            return 'darkorange'
        elif p_val < 0.05:
            return 'orange'
        else:
            return 'gray'
    
    def get_stars(p_val):
        if p_val < 0.001:
            return '***'
        elif p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        else:
            return 'n.s.'
    
    def is_opsin_row(label):
        return label.strip().startswith(stimulated_population.upper() + ' Opsin')
    
    # Track target boundaries for separator lines
    target_boundaries = []
    current_target = data_rows[0]['target']
    x_min, x_max = np.inf, -np.inf
    
    # Plot each row
    for i, row_data in enumerate(data_rows):
        mean_val = row_data['mean']
        ci_lower = row_data['ci_lower']
        ci_upper = row_data['ci_upper']
        p_val = row_data['p_value']
        label = row_data['label']
        
        # Track target boundaries
        if row_data['target'] != current_target:
            target_boundaries.append(i - 0.5)
            current_target = row_data['target']
        
        # Update x-axis bounds
        x_min = min(x_min, ci_lower)
        x_max = max(x_max, ci_upper)
        
        color = get_color(p_val)
        stars = get_stars(p_val)
        
        # Different marker for opsin rows
        if is_opsin_row(label):
            marker = 's'  # Square
            markersize = 8
            linewidth = 2.0
        else:
            marker = 'o'  # Circle
            markersize = 10
            linewidth = 2.5
        
        # Plot point estimate
        ax.plot(mean_val, y_positions[i], marker,
                color=color, markersize=markersize, zorder=3)
        
        # Plot CI
        ax.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]],
                '-', color=color, linewidth=linewidth, zorder=2)
        
        # Add text annotation
        ax.text(0.95, y_positions[i],
                f"{mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {stars}",
                va='center', ha='right', fontsize=9,
                transform=ax.get_yaxis_transform())
    
    # Set x-axis limits with padding
    if np.isfinite(x_min) and np.isfinite(x_max):
        x_range = x_max - x_min
        ax.set_xlim(x_min - 0.15 * x_range, x_max + 0.15 * x_range)
    
    # Reference line at zero
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
    
    # Y-axis labels
    y_labels = [row['label'] for row in data_rows]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Add horizontal lines to separate targets
    for boundary in target_boundaries:
        ax.axhline(boundary, color='black', linestyle='-', alpha=0.4, linewidth=2)
    
    # Labels and title
    ax.set_xlabel('Weight Difference (nS) | Bootstrap 95% CI',
                  fontsize=12, fontweight='bold')
    
    # Count opsin rows
    n_opsin_rows = sum(1 for row in data_rows if is_opsin_row(row['label']))
    
    if n_opsin_rows > 0:
        title = (f'Synaptic Weight Differences: {stimulated_population.upper()} Stimulation\n'
                 f'Across All Targets | Opsin$^{{+/-}}$ Separated by Target Population')
    else:
        title = (f'Synaptic Weight Differences: {stimulated_population.upper()} Stimulation\n'
                 f'Across All Targets')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='red', label='p < 0.001 (***)'),
        mpatches.Patch(color='darkorange', label='p < 0.01 (**)'),
        mpatches.Patch(color='orange', label='p < 0.05 (*)'),
        mpatches.Patch(color='gray', label='n.s.')
    ]
    
    if n_opsin_rows > 0:
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markersize=8, label='All cells', linestyle='None'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                       markersize=8, label='Opsin$^{+/-}$ specific', linestyle='None')
        ])
    
    ax.legend(handles=legend_elements, loc='lower left', 
              fontsize=9, title='Significance & Type', title_fontsize=10,
              ncol=2 if n_opsin_rows > 0 else 1)
    
    # Add sample size note
    n_conn_values = set(row['n_connectivity'] for row in data_rows)
    if len(n_conn_values) == 1:
        n_conn_text = f"N = {n_conn_values.pop()} connectivity instances"
    else:
        n_conn_text = f"N = {min(n_conn_values)}-{max(n_conn_values)} connectivity instances"
    
    ax.text(0.02, 0.98, n_conn_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary forest plot to: {save_path}")
    
    return fig


def create_summary_forest_plots_batch(
    nested_results: List,
    circuit,
    stimulated_population: str,
    target_populations: List[str],
    source_populations: List[str],
    stim_start: float,
    stim_duration: float,
    warmup: float,
    output_dir: str,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    n_bootstrap: int = 10000,
    run_pca: bool = False,
    n_pca_permutations: int = 1000,
    random_seed: Optional[int] = None
):
    """
    Batch function to run nested weight analysis for all targets and create summary plot
    
    Args:
        nested_results: List of NestedTrialResult objects
        circuit: DentateCircuit instance
        stimulated_population: Stimulated population name
        target_populations: List of postsynaptic populations to analyze
        source_populations: List of source populations
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period start (ms)
        output_dir: Directory to save outputs
        threshold_std: Classification threshold
        expression_threshold: Opsin expression threshold
        n_bootstrap: Number of bootstrap samples
        run_pca: 
        n_pca_permutations: 
        random_seed: Random seed
        
    Returns:
        tuple: (analysis_results_by_target, summary_fig)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analysis_results_by_target = {}
    
    # Run analysis for each target population
    print(f"\n{'='*80}")
    print(f"Running nested weight analysis for all postsynaptic targets")
    print(f"Stimulated population: {stimulated_population.upper()}")
    print('='*80)
    
    for target_pop in target_populations:
        print(f"\n--- Analyzing {target_pop.upper()} target population ---")
        
        analysis_results = analyze_weights_by_average_response_nested(
            nested_results=nested_results,
            circuit=circuit,
            target_population=stimulated_population,
            post_population=target_pop,
            source_populations=source_populations,
            stim_start=stim_start,
            stim_duration=stim_duration,
            warmup=warmup,
            threshold_std=threshold_std,
            expression_threshold=expression_threshold,
            n_bootstrap=n_bootstrap,
            run_pca=run_pca,
            n_pca_permutations=n_pca_permutations,
            random_seed=random_seed
        )
        
        analysis_results_by_target[target_pop] = analysis_results
        
        # Save individual target plot
        individual_plot_path = output_path / f'{stimulated_population}_to_{target_pop}_nested_weights.pdf'
        plot_weights_by_average_response_nested(
            analysis_results,
            save_path=individual_plot_path
        )

        #  Save PCA plots if analysis was performed
        if run_pca and analysis_results['pca_results'] is not None:
            pca_output_dir = output_path / f'{stimulated_population}_to_{target_pop}_pca'
            pca_output_dir.mkdir(exist_ok=True, parents=True)
            
            from pca_input_patterns import (
                plot_pca_scatter_by_connectivity,
                plot_pca_loadings_by_connectivity,
                plot_pca_separation_forest,
                plot_variance_explained
            )
            
            pca_res = analysis_results['pca_results']
            
            # Generate all PCA plots
            plot_pca_scatter_by_connectivity(
                pca_res['pca_results'],
                pca_res['weight_matrices'],
                pca_res['separation_stats'],
                stimulated_population,
                target_pop,
                save_path=str(pca_output_dir / f'pca_scatter.pdf')
            )
            
            plot_pca_loadings_by_connectivity(
                pca_res['pca_results'],
                pca_res['weight_matrices'],
                save_path=str(pca_output_dir / f'pca_loadings.pdf')
            )
            
            plot_pca_separation_forest(
                pca_res['bootstrap_result'],
                stimulated_population,
                target_pop,
                save_path=str(pca_output_dir / f'pca_forest.pdf')
            )
            
            plot_variance_explained(
                pca_res['pca_results'],
                save_path=str(pca_output_dir / f'pca_variance.pdf')
            )
            
            print(f"   Saved PCA plots to: {pca_output_dir}")
        
    # Create summary forest plot
    print(f"\n{'='*80}")
    print(f"Creating summary forest plot across all targets")
    print('='*80)
    
    summary_plot_path = output_path / f'{stimulated_population}_all_targets_summary_forest.pdf'
    summary_fig = plot_summary_forest_plot_all_targets(
        analysis_results_by_target=analysis_results_by_target,
        stimulated_population=stimulated_population,
        save_path=summary_plot_path
    )

    if run_pca:
        print(f"\n{'='*80}")
        print(f"Creating PCA summary plot across all targets")
        print('='*80)
        
        # Extract only PCA results
        pca_results_by_target = {
            target_pop: results['pca_results']
            for target_pop, results in analysis_results_by_target.items()
            if results.get('pca_results') is not None
        }
        
        if pca_results_by_target:
            pca_summary_path = output_path / f'{stimulated_population}_all_targets_pca_summary.pdf'
            pca_summary_fig = plot_pca_summary_all_targets(
                pca_results_by_target=pca_results_by_target,
                stimulated_population=stimulated_population,
                save_path=pca_summary_path
            )
            print(f"   Saved PCA summary plot to: {pca_summary_path}")
        else:
            print("   Warning: No PCA results available for summary plot")
            pca_summary_fig = None
    else:
        pca_summary_fig = None
        
    print(f"\nBatch analysis complete. Outputs saved to: {output_path}")
    
    return analysis_results_by_target, summary_fig
