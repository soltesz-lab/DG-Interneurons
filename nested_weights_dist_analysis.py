"""
Nested Weights Distributional Analysis

Extends nested weights analysis to compare log-normal weight distributions using:
1. Geometric mean ratios (parametric, assumes log-normal)
2. Mann-Whitney U / CLES (non-parametric, distribution-free)
3. Quantile differences (robust, reveals mechanism)

All analyses maintain nested bootstrap framework (resampling at connectivity level).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from nested_experiment import export_per_cell_weights_to_csv
from nested_weights_analysis import (
    organize_nested_trials,
    classify_cells_by_connectivity,
    extract_weight_distributions_by_connectivity,
)

plt.rcParams['axes.grid'] = False

# ============================================================================
# Distributional Analysis: Geometric Mean (Log-Normal)
# ============================================================================


def compute_geometric_mean_statistics_by_connectivity(
    weights_by_connectivity: Dict,
    source_populations: List[str],
    target_population: str,
    epsilon: float = 0.0
) -> Dict:
    """
    Compute log-transformed weight statistics for each connectivity instance.
    
    Since synaptic weights follow log-normal distributions, geometric mean
    (antilog of mean log-weight) provides a more appropriate measure of
    "typical" synapse strength than arithmetic mean.
    
    Args:
        weights_by_connectivity: Output from extract_weight_distributions_by_connectivity()
        source_populations: List of source populations to analyze
        target_population: Which population was stimulated (for opsin separation)
        epsilon: Offset for log transform if needed (default: 0.0, weights already > 0)
        
    Returns:
        dict: {
            connectivity_idx: {
                source_population: {
                    # Standard statistics (all cells)
                    'mean_log_excited': float,
                    'mean_log_suppressed': float,
                    'std_log_excited': float,
                    'std_log_suppressed': float,
                    'diff_log_means': float,  # mean_log_exc - mean_log_sup
                    'geometric_mean_ratio': float,  # exp(diff_log_means)
                    
                    # If source == target, also include opsin+/- variants
                    'mean_log_excited_opsin_plus': float,
                    ...
                }
            }
        }
    """
    geometric_stats = {}
    
    for conn_idx, conn_data in weights_by_connectivity.items():
        geometric_stats[conn_idx] = {}
        
        for source_pop in source_populations:
            if source_pop not in conn_data:
                continue
                
            source_data = conn_data[source_pop]
            has_opsin_separation = ('weights_excited_opsin_plus' in source_data)
            
            stats_dict = {}
            
            # Standard statistics (all cells)
            weights_excited = source_data['weights_excited']
            weights_suppressed = source_data['weights_suppressed']
            
            if len(weights_excited) > 0:
                log_weights_excited = np.log(weights_excited + epsilon)
                stats_dict['mean_log_excited'] = np.mean(log_weights_excited)
                stats_dict['std_log_excited'] = np.std(log_weights_excited, ddof=1) if len(log_weights_excited) > 1 else 0.0
                stats_dict['geometric_mean_excited'] = np.exp(stats_dict['mean_log_excited'])
            else:
                stats_dict['mean_log_excited'] = np.nan
                stats_dict['std_log_excited'] = np.nan
                stats_dict['geometric_mean_excited'] = np.nan
            
            if len(weights_suppressed) > 0:
                log_weights_suppressed = np.log(weights_suppressed + epsilon)
                stats_dict['mean_log_suppressed'] = np.mean(log_weights_suppressed)
                stats_dict['std_log_suppressed'] = np.std(log_weights_suppressed, ddof=1) if len(log_weights_suppressed) > 1 else 0.0
                stats_dict['geometric_mean_suppressed'] = np.exp(stats_dict['mean_log_suppressed'])
            else:
                stats_dict['mean_log_suppressed'] = np.nan
                stats_dict['std_log_suppressed'] = np.nan
                stats_dict['geometric_mean_suppressed'] = np.nan
            
            # Difference and ratio
            if not np.isnan(stats_dict['mean_log_excited']) and not np.isnan(stats_dict['mean_log_suppressed']):
                stats_dict['diff_log_means'] = stats_dict['mean_log_excited'] - stats_dict['mean_log_suppressed']
                stats_dict['geometric_mean_ratio'] = np.exp(stats_dict['diff_log_means'])
            else:
                stats_dict['diff_log_means'] = np.nan
                stats_dict['geometric_mean_ratio'] = np.nan
            
            # Sample sizes
            stats_dict['n_synapses_excited'] = len(weights_excited)
            stats_dict['n_synapses_suppressed'] = len(weights_suppressed)
            
            # Opsin-specific analysis (if applicable)
            if has_opsin_separation:
                # Opsin+ cells
                weights_exc_opsin_plus = source_data['weights_excited_opsin_plus']
                weights_sup_opsin_plus = source_data['weights_suppressed_opsin_plus']
                
                if len(weights_exc_opsin_plus) > 0:
                    log_exc_opsin_plus = np.log(weights_exc_opsin_plus + epsilon)
                    stats_dict['mean_log_excited_opsin_plus'] = np.mean(log_exc_opsin_plus)
                    stats_dict['std_log_excited_opsin_plus'] = np.std(log_exc_opsin_plus, ddof=1) if len(log_exc_opsin_plus) > 1 else 0.0
                    stats_dict['geometric_mean_excited_opsin_plus'] = np.exp(stats_dict['mean_log_excited_opsin_plus'])
                else:
                    stats_dict['mean_log_excited_opsin_plus'] = np.nan
                    stats_dict['std_log_excited_opsin_plus'] = np.nan
                    stats_dict['geometric_mean_excited_opsin_plus'] = np.nan
                
                if len(weights_sup_opsin_plus) > 0:
                    log_sup_opsin_plus = np.log(weights_sup_opsin_plus + epsilon)
                    stats_dict['mean_log_suppressed_opsin_plus'] = np.mean(log_sup_opsin_plus)
                    stats_dict['std_log_suppressed_opsin_plus'] = np.std(log_sup_opsin_plus, ddof=1) if len(log_sup_opsin_plus) > 1 else 0.0
                    stats_dict['geometric_mean_suppressed_opsin_plus'] = np.exp(stats_dict['mean_log_suppressed_opsin_plus'])
                else:
                    stats_dict['mean_log_suppressed_opsin_plus'] = np.nan
                    stats_dict['std_log_suppressed_opsin_plus'] = np.nan
                    stats_dict['geometric_mean_suppressed_opsin_plus'] = np.nan
                
                # Opsin+ difference and ratio
                if not np.isnan(stats_dict['mean_log_excited_opsin_plus']) and not np.isnan(stats_dict['mean_log_suppressed_opsin_plus']):
                    stats_dict['diff_log_means_opsin_plus'] = stats_dict['mean_log_excited_opsin_plus'] - stats_dict['mean_log_suppressed_opsin_plus']
                    stats_dict['geometric_mean_ratio_opsin_plus'] = np.exp(stats_dict['diff_log_means_opsin_plus'])
                else:
                    stats_dict['diff_log_means_opsin_plus'] = np.nan
                    stats_dict['geometric_mean_ratio_opsin_plus'] = np.nan
                
                # Opsin- cells
                weights_exc_opsin_minus = source_data['weights_excited_opsin_minus']
                weights_sup_opsin_minus = source_data['weights_suppressed_opsin_minus']
                
                if len(weights_exc_opsin_minus) > 0:
                    log_exc_opsin_minus = np.log(weights_exc_opsin_minus + epsilon)
                    stats_dict['mean_log_excited_opsin_minus'] = np.mean(log_exc_opsin_minus)
                    stats_dict['std_log_excited_opsin_minus'] = np.std(log_exc_opsin_minus, ddof=1) if len(log_exc_opsin_minus) > 1 else 0.0
                    stats_dict['geometric_mean_excited_opsin_minus'] = np.exp(stats_dict['mean_log_excited_opsin_minus'])
                else:
                    stats_dict['mean_log_excited_opsin_minus'] = np.nan
                    stats_dict['std_log_excited_opsin_minus'] = np.nan
                    stats_dict['geometric_mean_excited_opsin_minus'] = np.nan
                
                if len(weights_sup_opsin_minus) > 0:
                    log_sup_opsin_minus = np.log(weights_sup_opsin_minus + epsilon)
                    stats_dict['mean_log_suppressed_opsin_minus'] = np.mean(log_sup_opsin_minus)
                    stats_dict['std_log_suppressed_opsin_minus'] = np.std(log_sup_opsin_minus, ddof=1) if len(log_sup_opsin_minus) > 1 else 0.0
                    stats_dict['geometric_mean_suppressed_opsin_minus'] = np.exp(stats_dict['mean_log_suppressed_opsin_minus'])
                else:
                    stats_dict['mean_log_suppressed_opsin_minus'] = np.nan
                    stats_dict['std_log_suppressed_opsin_minus'] = np.nan
                    stats_dict['geometric_mean_suppressed_opsin_minus'] = np.nan
                
                # Opsin- difference and ratio
                if not np.isnan(stats_dict['mean_log_excited_opsin_minus']) and not np.isnan(stats_dict['mean_log_suppressed_opsin_minus']):
                    stats_dict['diff_log_means_opsin_minus'] = stats_dict['mean_log_excited_opsin_minus'] - stats_dict['mean_log_suppressed_opsin_minus']
                    stats_dict['geometric_mean_ratio_opsin_minus'] = np.exp(stats_dict['diff_log_means_opsin_minus'])
                else:
                    stats_dict['diff_log_means_opsin_minus'] = np.nan
                    stats_dict['geometric_mean_ratio_opsin_minus'] = np.nan
                
                # Cross-comparison: opsin+ vs opsin- to excited cells
                if not np.isnan(stats_dict['mean_log_excited_opsin_plus']) and not np.isnan(stats_dict['mean_log_excited_opsin_minus']):
                    stats_dict['diff_log_means_opsin_plus_minus_to_excited'] = stats_dict['mean_log_excited_opsin_plus'] - stats_dict['mean_log_excited_opsin_minus']
                    stats_dict['geometric_mean_ratio_opsin_plus_minus_to_excited'] = np.exp(stats_dict['diff_log_means_opsin_plus_minus_to_excited'])
                else:
                    stats_dict['diff_log_means_opsin_plus_minus_to_excited'] = np.nan
                    stats_dict['geometric_mean_ratio_opsin_plus_minus_to_excited'] = np.nan
                
                # Cross-comparison: opsin+ vs opsin- to suppressed cells
                if not np.isnan(stats_dict['mean_log_suppressed_opsin_plus']) and not np.isnan(stats_dict['mean_log_suppressed_opsin_minus']):
                    stats_dict['diff_log_means_opsin_plus_minus_to_suppressed'] = stats_dict['mean_log_suppressed_opsin_plus'] - stats_dict['mean_log_suppressed_opsin_minus']
                    stats_dict['geometric_mean_ratio_opsin_plus_minus_to_suppressed'] = np.exp(stats_dict['diff_log_means_opsin_plus_minus_to_suppressed'])
                else:
                    stats_dict['diff_log_means_opsin_plus_minus_to_suppressed'] = np.nan
                    stats_dict['geometric_mean_ratio_opsin_plus_minus_to_suppressed'] = np.nan
                
                # Sample sizes
                stats_dict['n_synapses_excited_opsin_plus'] = len(weights_exc_opsin_plus)
                stats_dict['n_synapses_excited_opsin_minus'] = len(weights_exc_opsin_minus)
                stats_dict['n_synapses_suppressed_opsin_plus'] = len(weights_sup_opsin_plus)
                stats_dict['n_synapses_suppressed_opsin_minus'] = len(weights_sup_opsin_minus)
            
            geometric_stats[conn_idx][source_pop] = stats_dict
    
    return geometric_stats


def bootstrap_geometric_mean_nested(
    geometric_stats_by_connectivity: Dict,
    source_population: str,
    statistic: str = 'geometric_mean_ratio',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Bootstrap geometric mean ratios at connectivity level.
    
    Args:
        geometric_stats_by_connectivity: Output from compute_geometric_mean_statistics_by_connectivity()
        source_population: Source population to analyze
        statistic: Which statistic to bootstrap (default: 'geometric_mean_ratio')
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: {
            'mean': float,  # grand mean of ratios across connectivities
            'ci_lower': float,
            'ci_upper': float,
            'std': float,
            'p_value': float,
            'n_connectivity': int,
            'bootstrap_distribution': np.ndarray,
            'connectivity_indices': List[int],
            'original_values': np.ndarray,
            'interpretation': str  # e.g., "1.45-fold stronger (45% increase)"
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Extract statistic values across connectivity instances
    values = []
    conn_indices = []
    
    for conn_idx in sorted(geometric_stats_by_connectivity.keys()):
        if source_population in geometric_stats_by_connectivity[conn_idx]:
            val = geometric_stats_by_connectivity[conn_idx][source_population][statistic]
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
            'n_connectivity': 0,
            'interpretation': 'No data'
        }
    
    values = np.array(values)
    n_conn = len(values)
    
    # Original statistics
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if n_conn > 1 else 0.0
    
    if n_conn == 1:
        interpretation = _interpret_geometric_mean_ratio(mean_val)
        return {
            'mean': mean_val,
            'ci_lower': mean_val,
            'ci_upper': mean_val,
            'std': 0.0,
            'p_value': np.nan,
            'n_connectivity': n_conn,
            'connectivity_indices': conn_indices,
            'interpretation': interpretation
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
    
    # Two-sided p-value: test if ratio differs from 1.0
    # For ratios, null hypothesis is ratio = 1.0 (no difference)
    # For CLES, null hypothesis is CLES = 0.5

    if statistic == 'cles':
        null_value = 0.5
    elif 'ratio' in statistic:
        null_value = 1.0
    else:
        null_value = 0.0

    # Count proportion of bootstrap samples as or more extreme than observed
    p_value = np.sum(np.abs(bootstrap_means - null_value) >= np.abs(mean_val - null_value)) / len(bootstrap_means)
    
    interpretation = _interpret_geometric_mean_ratio(mean_val)
    
    return {
        'mean': mean_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': std_val,
        'p_value': p_value,
        'n_connectivity': n_conn,
        'bootstrap_distribution': bootstrap_means,
        'connectivity_indices': conn_indices,
        'original_values': values,
        'interpretation': interpretation
    }


def _interpret_geometric_mean_ratio(ratio: float) -> str:
    """Generate interpretation string for geometric mean ratio."""
    if np.isnan(ratio):
        return 'No data'
    
    if ratio > 1.0:
        percent_increase = (ratio - 1.0) * 100
        return f"{ratio:.2f}-fold stronger ({percent_increase:.0f}% increase)"
    elif ratio < 1.0:
        percent_decrease = (1.0 - ratio) * 100
        return f"{ratio:.2f}-fold weaker ({percent_decrease:.0f}% decrease)"
    else:
        return "No difference (ratio = 1.0)"


# ============================================================================
# Distributional Analysis: Mann-Whitney U / CLES
# ============================================================================


def compute_mann_whitney_statistics_by_connectivity(
    weights_by_connectivity: Dict,
    source_populations: List[str],
    target_population: str,
    alternative: str = 'two-sided'
) -> Dict:
    """
    Compute Mann-Whitney U statistic and CLES for each connectivity.
    
    Mann-Whitney U is a non-parametric test that makes no assumptions about
    the distribution shape. CLES (Common Language Effect Size) provides an
    intuitive interpretation: P(weight to excited > weight to suppressed).
    
    Includes comprehensive opsin+/- separation and cross-comparisons.
    
    Args:
        weights_by_connectivity: Output from extract_weight_distributions_by_connectivity()
        source_populations: List of source populations to analyze
        target_population: Which population was stimulated (for opsin separation)
        alternative: Direction of test ('two-sided', 'greater', 'less')
        
    Returns:
        dict: {
            connectivity_idx: {
                source_population: {
                    'u_statistic': float,
                    'p_value': float,
                    'cles': float,  # Common Language Effect Size
                    'z_score': float,
                    'r_effect': float,  # Z / sqrt(n_total)
                    'n_excited': int,
                    'n_suppressed': int,
                    
                    # Opsin-specific (if applicable)
                    'u_statistic_opsin_plus': float,
                    'cles_opsin_plus': float,
                    ...
                }
            }
        }
    """
    mann_whitney_stats = {}
    
    for conn_idx, conn_data in weights_by_connectivity.items():
        mann_whitney_stats[conn_idx] = {}
        
        for source_pop in source_populations:
            if source_pop not in conn_data:
                continue
                
            source_data = conn_data[source_pop]
            has_opsin_separation = ('weights_excited_opsin_plus' in source_data)
            
            stats_dict = {}
            
            # Standard statistics (all cells)
            weights_excited = source_data['weights_excited']
            weights_suppressed = source_data['weights_suppressed']
            
            if len(weights_excited) > 0 and len(weights_suppressed) > 0:
                u_stat, p_val = stats.mannwhitneyu(
                    weights_excited,
                    weights_suppressed,
                    alternative=alternative
                )
                
                n_exc = len(weights_excited)
                n_sup = len(weights_suppressed)
                
                # CLES: Common Language Effect Size
                cles = u_stat / (n_exc * n_sup)
                
                # Z-score and r effect size
                n_total = n_exc + n_sup
                mean_u = n_exc * n_sup / 2
                std_u = np.sqrt(n_exc * n_sup * (n_total + 1) / 12)
                z_score = (u_stat - mean_u) / std_u if std_u > 0 else 0.0
                r_effect = z_score / np.sqrt(n_total) if n_total > 0 else 0.0
                
                stats_dict['u_statistic'] = u_stat
                stats_dict['p_value'] = p_val
                stats_dict['cles'] = cles
                stats_dict['z_score'] = z_score
                stats_dict['r_effect'] = r_effect
                stats_dict['n_excited'] = n_exc
                stats_dict['n_suppressed'] = n_sup
            else:
                stats_dict['u_statistic'] = np.nan
                stats_dict['p_value'] = np.nan
                stats_dict['cles'] = np.nan
                stats_dict['z_score'] = np.nan
                stats_dict['r_effect'] = np.nan
                stats_dict['n_excited'] = len(weights_excited)
                stats_dict['n_suppressed'] = len(weights_suppressed)
            
            # Opsin-specific analysis (if applicable)
            if has_opsin_separation:
                # Opsin+ cells (exc vs sup)
                weights_exc_opsin_plus = source_data['weights_excited_opsin_plus']
                weights_sup_opsin_plus = source_data['weights_suppressed_opsin_plus']
                
                if len(weights_exc_opsin_plus) > 0 and len(weights_sup_opsin_plus) > 0:
                    u_stat_op, p_val_op = stats.mannwhitneyu(
                        weights_exc_opsin_plus,
                        weights_sup_opsin_plus,
                        alternative=alternative
                    )
                    
                    n_exc_op = len(weights_exc_opsin_plus)
                    n_sup_op = len(weights_sup_opsin_plus)
                    cles_op = u_stat_op / (n_exc_op * n_sup_op)
                    
                    n_total_op = n_exc_op + n_sup_op
                    mean_u_op = n_exc_op * n_sup_op / 2
                    std_u_op = np.sqrt(n_exc_op * n_sup_op * (n_total_op + 1) / 12)
                    z_score_op = (u_stat_op - mean_u_op) / std_u_op if std_u_op > 0 else 0.0
                    r_effect_op = z_score_op / np.sqrt(n_total_op) if n_total_op > 0 else 0.0
                    
                    stats_dict['u_statistic_opsin_plus'] = u_stat_op
                    stats_dict['p_value_opsin_plus'] = p_val_op
                    stats_dict['cles_opsin_plus'] = cles_op
                    stats_dict['z_score_opsin_plus'] = z_score_op
                    stats_dict['r_effect_opsin_plus'] = r_effect_op
                else:
                    stats_dict['u_statistic_opsin_plus'] = np.nan
                    stats_dict['p_value_opsin_plus'] = np.nan
                    stats_dict['cles_opsin_plus'] = np.nan
                    stats_dict['z_score_opsin_plus'] = np.nan
                    stats_dict['r_effect_opsin_plus'] = np.nan
                
                # Opsin- cells (exc vs sup)
                weights_exc_opsin_minus = source_data['weights_excited_opsin_minus']
                weights_sup_opsin_minus = source_data['weights_suppressed_opsin_minus']
                
                if len(weights_exc_opsin_minus) > 0 and len(weights_sup_opsin_minus) > 0:
                    u_stat_om, p_val_om = stats.mannwhitneyu(
                        weights_exc_opsin_minus,
                        weights_sup_opsin_minus,
                        alternative=alternative
                    )
                    
                    n_exc_om = len(weights_exc_opsin_minus)
                    n_sup_om = len(weights_sup_opsin_minus)
                    cles_om = u_stat_om / (n_exc_om * n_sup_om)
                    
                    n_total_om = n_exc_om + n_sup_om
                    mean_u_om = n_exc_om * n_sup_om / 2
                    std_u_om = np.sqrt(n_exc_om * n_sup_om * (n_total_om + 1) / 12)
                    z_score_om = (u_stat_om - mean_u_om) / std_u_om if std_u_om > 0 else 0.0
                    r_effect_om = z_score_om / np.sqrt(n_total_om) if n_total_om > 0 else 0.0
                    
                    stats_dict['u_statistic_opsin_minus'] = u_stat_om
                    stats_dict['p_value_opsin_minus'] = p_val_om
                    stats_dict['cles_opsin_minus'] = cles_om
                    stats_dict['z_score_opsin_minus'] = z_score_om
                    stats_dict['r_effect_opsin_minus'] = r_effect_om
                else:
                    stats_dict['u_statistic_opsin_minus'] = np.nan
                    stats_dict['p_value_opsin_minus'] = np.nan
                    stats_dict['cles_opsin_minus'] = np.nan
                    stats_dict['z_score_opsin_minus'] = np.nan
                    stats_dict['r_effect_opsin_minus'] = np.nan
                
                # Cross-comparison: opsin+ vs opsin- to excited cells
                if len(weights_exc_opsin_plus) > 0 and len(weights_exc_opsin_minus) > 0:
                    u_stat_cross_exc, p_val_cross_exc = stats.mannwhitneyu(
                        weights_exc_opsin_plus,
                        weights_exc_opsin_minus,
                        alternative=alternative
                    )
                    
                    cles_cross_exc = u_stat_cross_exc / (len(weights_exc_opsin_plus) * len(weights_exc_opsin_minus))
                    stats_dict['u_statistic_opsin_plus_minus_to_excited'] = u_stat_cross_exc
                    stats_dict['p_value_opsin_plus_minus_to_excited'] = p_val_cross_exc
                    stats_dict['cles_opsin_plus_minus_to_excited'] = cles_cross_exc
                else:
                    stats_dict['u_statistic_opsin_plus_minus_to_excited'] = np.nan
                    stats_dict['p_value_opsin_plus_minus_to_excited'] = np.nan
                    stats_dict['cles_opsin_plus_minus_to_excited'] = np.nan
                
                # Cross-comparison: opsin+ vs opsin- to suppressed cells
                if len(weights_sup_opsin_plus) > 0 and len(weights_sup_opsin_minus) > 0:
                    u_stat_cross_sup, p_val_cross_sup = stats.mannwhitneyu(
                        weights_sup_opsin_plus,
                        weights_sup_opsin_minus,
                        alternative=alternative
                    )
                    
                    cles_cross_sup = u_stat_cross_sup / (len(weights_sup_opsin_plus) * len(weights_sup_opsin_minus))
                    stats_dict['u_statistic_opsin_plus_minus_to_suppressed'] = u_stat_cross_sup
                    stats_dict['p_value_opsin_plus_minus_to_suppressed'] = p_val_cross_sup
                    stats_dict['cles_opsin_plus_minus_to_suppressed'] = cles_cross_sup
                else:
                    stats_dict['u_statistic_opsin_plus_minus_to_suppressed'] = np.nan
                    stats_dict['p_value_opsin_plus_minus_to_suppressed'] = np.nan
                    stats_dict['cles_opsin_plus_minus_to_suppressed'] = np.nan
            
            mann_whitney_stats[conn_idx][source_pop] = stats_dict
    
    return mann_whitney_stats


def bootstrap_mann_whitney_nested(
    mann_whitney_stats_by_connectivity: Dict,
    source_population: str,
    statistic: str = 'cles',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Bootstrap Mann-Whitney statistics (typically CLES) at connectivity level.
    
    Args:
        mann_whitney_stats_by_connectivity: Output from compute_mann_whitney_statistics_by_connectivity()
        source_population: Source population to analyze
        statistic: Which statistic to bootstrap (default: 'cles')
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level
        random_seed: Random seed for reproducibility
        
    Returns:
        dict with mean, CI, p-value, bootstrap distribution, interpretation
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Extract statistic values across connectivity instances
    values = []
    conn_indices = []
    
    for conn_idx in sorted(mann_whitney_stats_by_connectivity.keys()):
        if source_population in mann_whitney_stats_by_connectivity[conn_idx]:
            val = mann_whitney_stats_by_connectivity[conn_idx][source_population][statistic]
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
            'n_connectivity': 0,
            'interpretation': 'No data'
        }
    
    values = np.array(values)
    n_conn = len(values)
    
    # Original statistics
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if n_conn > 1 else 0.0
    
    if n_conn == 1:
        interpretation = _interpret_cles(mean_val) if statistic == 'cles' else f'{statistic} = {mean_val:.3f}'
        return {
            'mean': mean_val,
            'ci_lower': mean_val,
            'ci_upper': mean_val,
            'std': 0.0,
            'p_value': np.nan,
            'n_connectivity': n_conn,
            'connectivity_indices': conn_indices,
            'interpretation': interpretation
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
    
    # Two-sided p-value: test if CLES differs from 0.5 (no difference)
    # For CLES, null hypothesis is CLES = 0.5
    null_value = 0.5 if statistic == 'cles' else 0.0
    p_value = np.sum(np.abs(bootstrap_means - null_value) >= np.abs(mean_val - null_value)) / len(bootstrap_means)

    interpretation = _interpret_cles(mean_val) if statistic == 'cles' else f'{statistic} = {mean_val:.3f}'
    
    return {
        'mean': mean_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': std_val,
        'p_value': p_value,
        'n_connectivity': n_conn,
        'bootstrap_distribution': bootstrap_means,
        'connectivity_indices': conn_indices,
        'original_values': values,
        'interpretation': interpretation
    }


def _interpret_cles(cles: float) -> str:
    """Generate interpretation string for CLES."""
    if np.isnan(cles):
        return 'No data'
    
    prob_percent = cles * 100
    
    if cles > 0.5:
        return f"{prob_percent:.0f}% probability excited > suppressed"
    elif cles < 0.5:
        return f"{(100 - prob_percent):.0f}% probability suppressed > excited"
    else:
        return "No difference (CLES = 0.5)"


# ============================================================================
# Distributional Analysis: Quantile Profiles
# ============================================================================


def compute_quantile_statistics_by_connectivity(
    weights_by_connectivity: Dict,
    source_populations: List[str],
    target_population: str,
    quantiles: List[float] = [0.25, 0.50, 0.75, 0.90]
) -> Dict:
    """
    Compute weight differences at multiple quantiles.
    
    Quantile analysis reveals WHERE in the distribution differences occur:
    - Uniform across quantiles → entire distribution shifts
    - Large at 75th/90th, small at 25th/50th → tail-driven (strong synapses)
    - Large at 25th/50th, convergence at 75th/90th → bulk-driven (typical synapses)
    
    Includes comprehensive opsin+/- separation and cross-comparisons.
    
    Args:
        weights_by_connectivity: Output from extract_weight_distributions_by_connectivity()
        source_populations: List of source populations to analyze
        target_population: Which population was stimulated (for opsin separation)
        quantiles: List of quantiles to compute (default: [0.25, 0.50, 0.75, 0.90])
        
    Returns:
        dict: {
            connectivity_idx: {
                source_population: {
                    'quantiles': List[float],
                    'quantile_diffs': List[float],
                    'quantile_excited': List[float],
                    'quantile_suppressed': List[float],
                    
                    # Opsin-specific (if applicable)
                    'quantile_diffs_opsin_plus': List[float],
                    ...
                }
            }
        }
    """
    quantile_stats = {}
    
    for conn_idx, conn_data in weights_by_connectivity.items():
        quantile_stats[conn_idx] = {}
        
        for source_pop in source_populations:
            if source_pop not in conn_data:
                continue
                
            source_data = conn_data[source_pop]
            has_opsin_separation = ('weights_excited_opsin_plus' in source_data)
            
            stats_dict = {'quantiles': quantiles}
            
            # Standard statistics (all cells)
            weights_excited = source_data['weights_excited']
            weights_suppressed = source_data['weights_suppressed']
            
            if len(weights_excited) > 0 and len(weights_suppressed) > 0:
                q_excited = []
                q_suppressed = []
                q_diffs = []
                
                for q in quantiles:
                    q_exc = np.percentile(weights_excited, q * 100)
                    q_sup = np.percentile(weights_suppressed, q * 100)
                    q_diff = q_exc - q_sup
                    
                    q_excited.append(q_exc)
                    q_suppressed.append(q_sup)
                    q_diffs.append(q_diff)
                
                stats_dict['quantile_excited'] = q_excited
                stats_dict['quantile_suppressed'] = q_suppressed
                stats_dict['quantile_diffs'] = q_diffs
            else:
                stats_dict['quantile_excited'] = [np.nan] * len(quantiles)
                stats_dict['quantile_suppressed'] = [np.nan] * len(quantiles)
                stats_dict['quantile_diffs'] = [np.nan] * len(quantiles)
            
            # Opsin-specific analysis (if applicable)
            if has_opsin_separation:
                # Opsin+ cells (exc vs sup)
                weights_exc_opsin_plus = source_data['weights_excited_opsin_plus']
                weights_sup_opsin_plus = source_data['weights_suppressed_opsin_plus']
                
                if len(weights_exc_opsin_plus) > 0 and len(weights_sup_opsin_plus) > 0:
                    q_diffs_op = []
                    q_exc_op = []
                    q_sup_op = []
                    
                    for q in quantiles:
                        q_exc = np.percentile(weights_exc_opsin_plus, q * 100)
                        q_sup = np.percentile(weights_sup_opsin_plus, q * 100)
                        q_diffs_op.append(q_exc - q_sup)
                        q_exc_op.append(q_exc)
                        q_sup_op.append(q_sup)
                    
                    stats_dict['quantile_diffs_opsin_plus'] = q_diffs_op
                    stats_dict['quantile_excited_opsin_plus'] = q_exc_op
                    stats_dict['quantile_suppressed_opsin_plus'] = q_sup_op
                else:
                    stats_dict['quantile_diffs_opsin_plus'] = [np.nan] * len(quantiles)
                    stats_dict['quantile_excited_opsin_plus'] = [np.nan] * len(quantiles)
                    stats_dict['quantile_suppressed_opsin_plus'] = [np.nan] * len(quantiles)
                
                # Opsin- cells (exc vs sup)
                weights_exc_opsin_minus = source_data['weights_excited_opsin_minus']
                weights_sup_opsin_minus = source_data['weights_suppressed_opsin_minus']
                
                if len(weights_exc_opsin_minus) > 0 and len(weights_sup_opsin_minus) > 0:
                    q_diffs_om = []
                    q_exc_om = []
                    q_sup_om = []
                    
                    for q in quantiles:
                        q_exc = np.percentile(weights_exc_opsin_minus, q * 100)
                        q_sup = np.percentile(weights_sup_opsin_minus, q * 100)
                        q_diffs_om.append(q_exc - q_sup)
                        q_exc_om.append(q_exc)
                        q_sup_om.append(q_sup)
                    
                    stats_dict['quantile_diffs_opsin_minus'] = q_diffs_om
                    stats_dict['quantile_excited_opsin_minus'] = q_exc_om
                    stats_dict['quantile_suppressed_opsin_minus'] = q_sup_om
                else:
                    stats_dict['quantile_diffs_opsin_minus'] = [np.nan] * len(quantiles)
                    stats_dict['quantile_excited_opsin_minus'] = [np.nan] * len(quantiles)
                    stats_dict['quantile_suppressed_opsin_minus'] = [np.nan] * len(quantiles)
                
                # Cross-comparison: opsin+ vs opsin- to excited cells
                if len(weights_exc_opsin_plus) > 0 and len(weights_exc_opsin_minus) > 0:
                    q_diffs_cross_exc = []
                    for q in quantiles:
                        q_exc_op = np.percentile(weights_exc_opsin_plus, q * 100)
                        q_exc_om = np.percentile(weights_exc_opsin_minus, q * 100)
                        q_diffs_cross_exc.append(q_exc_op - q_exc_om)
                    stats_dict['quantile_diffs_opsin_plus_minus_to_excited'] = q_diffs_cross_exc
                else:
                    stats_dict['quantile_diffs_opsin_plus_minus_to_excited'] = [np.nan] * len(quantiles)
                
                # Cross-comparison: opsin+ vs opsin- to suppressed cells
                if len(weights_sup_opsin_plus) > 0 and len(weights_sup_opsin_minus) > 0:
                    q_diffs_cross_sup = []
                    for q in quantiles:
                        q_sup_op = np.percentile(weights_sup_opsin_plus, q * 100)
                        q_sup_om = np.percentile(weights_sup_opsin_minus, q * 100)
                        q_diffs_cross_sup.append(q_sup_op - q_sup_om)
                    stats_dict['quantile_diffs_opsin_plus_minus_to_suppressed'] = q_diffs_cross_sup
                else:
                    stats_dict['quantile_diffs_opsin_plus_minus_to_suppressed'] = [np.nan] * len(quantiles)
            
            quantile_stats[conn_idx][source_pop] = stats_dict
    
    return quantile_stats


def bootstrap_quantile_differences_nested(
    quantile_stats_by_connectivity: Dict,
    source_population: str,
    quantiles: List[float],
    statistic_key: str = 'quantile_diffs',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Bootstrap each quantile difference with CI.
    
    Args:
        quantile_stats_by_connectivity: Output from compute_quantile_statistics_by_connectivity()
        source_population: Source population to analyze
        quantiles: List of quantiles to analyze
        statistic_key: Which statistic to bootstrap (default: 'quantile_diffs')
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: {
            'quantiles': List[float],
            'results_by_quantile': {
                q: {
                    'mean': float,
                    'ci_lower': float,
                    'ci_upper': float,
                    'p_value': float,
                    'bootstrap_distribution': np.ndarray
                }
            },
            'n_connectivity': int
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get connectivity indices
    conn_indices = sorted(quantile_stats_by_connectivity.keys())
    n_conn = len(conn_indices)
    
    if n_conn == 0:
        return {
            'quantiles': quantiles,
            'results_by_quantile': {},
            'n_connectivity': 0
        }
    
    results_by_quantile = {}
    
    for q_idx, q in enumerate(quantiles):
        # Extract differences at this quantile across connectivities
        diffs_at_q = []
        
        for conn_idx in conn_indices:
            if source_population in quantile_stats_by_connectivity[conn_idx]:
                q_diffs = quantile_stats_by_connectivity[conn_idx][source_population][statistic_key]
                if q_idx < len(q_diffs) and not np.isnan(q_diffs[q_idx]):
                    diffs_at_q.append(q_diffs[q_idx])
        
        if len(diffs_at_q) == 0:
            results_by_quantile[q] = {
                'mean': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan,
                'bootstrap_distribution': np.array([])
            }
            continue
        
        diffs_at_q = np.array(diffs_at_q)
        mean_diff = np.mean(diffs_at_q)
        
        if len(diffs_at_q) == 1:
            results_by_quantile[q] = {
                'mean': mean_diff,
                'ci_lower': mean_diff,
                'ci_upper': mean_diff,
                'p_value': np.nan,
                'bootstrap_distribution': np.array([mean_diff])
            }
            continue
        
        # Bootstrap
        bootstrap_means_q = []
        
        for _ in range(n_bootstrap):
            resampled = np.random.choice(diffs_at_q, size=len(diffs_at_q), replace=True)
            bootstrap_means_q.append(np.mean(resampled))
        
        bootstrap_means_q = np.array(bootstrap_means_q)
        
        # CI and p-value
        alpha = 1 - confidence_level
        ci_lower_q = np.percentile(bootstrap_means_q, (alpha / 2) * 100)
        ci_upper_q = np.percentile(bootstrap_means_q, (1 - alpha / 2) * 100)
        p_value_q = np.sum(np.abs(bootstrap_means_q) >= np.abs(mean_diff)) / len(bootstrap_means_q)
        
        results_by_quantile[q] = {
            'mean': mean_diff,
            'ci_lower': ci_lower_q,
            'ci_upper': ci_upper_q,
            'p_value': p_value_q,
            'bootstrap_distribution': bootstrap_means_q
        }
    
    return {
        'quantiles': quantiles,
        'results_by_quantile': results_by_quantile,
        'n_connectivity': n_conn
    }


# ============================================================================
# Master Distributional Analysis
# ============================================================================


def analyze_weights_distributional_nested(
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
    quantiles: List[float] = [0.25, 0.50, 0.75, 0.90],
    random_seed: Optional[int] = None,
    export_csv_path: Optional[str] = None
) -> Dict:
    """
    Perform comprehensive distributional analysis on nested experiment results.
    
    This function orchestrates three complementary distributional analyses:
    1. Geometric mean ratios (parametric, assumes log-normal)
    2. Mann-Whitney U / CLES (non-parametric, no assumptions)
    3. Quantile differences (robust, reveals mechanism)
    
    All analyses maintain nested bootstrap framework (resampling at connectivity level).
    Includes comprehensive opsin+/- separation for stimulated population.
    
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
        quantiles: List of quantiles to analyze
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: {
            'classifications': Dict,
            'weights_by_connectivity': Dict,
            'geometric_stats_by_connectivity': Dict,
            'mann_whitney_stats_by_connectivity': Dict,
            'quantile_stats_by_connectivity': Dict,
            'geometric_bootstrap_results': Dict,
            'mann_whitney_bootstrap_results': Dict,
            'quantile_bootstrap_results': Dict,
            'metadata': Dict
        }
    """
    print(f"\n{'='*80}")
    print(f"Distributional Analysis: {target_population.upper()} -> {post_population.upper()}")
    print('='*80)
    
    # Organize trials by connectivity
    trials_by_connectivity = organize_nested_trials(nested_results)
    n_conn = len(trials_by_connectivity)
    print(f"Analyzing {n_conn} connectivity instances")
    
    # Classify cells within each connectivity (reuse from existing code)
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
    
    # Extract weight distributions (reuse from existing code)
    weights_by_connectivity = extract_weight_distributions_by_connectivity(
        circuit,
        classifications,
        post_population,
        source_populations,
        target_population=target_population,
        trials_by_connectivity=trials_by_connectivity,
        expression_threshold=expression_threshold
    )
    
    # Compute distributional statistics
    print("\nComputing distributional statistics...")
    
    geometric_stats = compute_geometric_mean_statistics_by_connectivity(
        weights_by_connectivity,
        source_populations,
        target_population
    )
    
    mann_whitney_stats = compute_mann_whitney_statistics_by_connectivity(
        weights_by_connectivity,
        source_populations,
        target_population
    )
    
    quantile_stats = compute_quantile_statistics_by_connectivity(
        weights_by_connectivity,
        source_populations,
        target_population,
        quantiles=quantiles
    )
    
    # Bootstrap analyses
    print(f"\nBootstrap Analysis (n={n_bootstrap}):")
    
    geometric_bootstrap = {}
    mann_whitney_bootstrap = {}
    quantile_bootstrap = {}
    
    for source_pop in source_populations:
        # Standard comparisons (all cells)
        geometric_bootstrap[source_pop] = bootstrap_geometric_mean_nested(
            geometric_stats,
            source_pop,
            statistic='geometric_mean_ratio',
            n_bootstrap=n_bootstrap,
            random_seed=random_seed
        )
        
        mann_whitney_bootstrap[source_pop] = bootstrap_mann_whitney_nested(
            mann_whitney_stats,
            source_pop,
            statistic='cles',
            n_bootstrap=n_bootstrap,
            random_seed=random_seed
        )
        
        quantile_bootstrap[source_pop] = bootstrap_quantile_differences_nested(
            quantile_stats,
            source_pop,
            quantiles=quantiles,
            statistic_key='quantile_diffs',
            n_bootstrap=n_bootstrap,
            random_seed=random_seed
        )
        
        # Opsin-specific if applicable (matching nested_weights_analysis pattern)
        if source_pop == target_population:
            print(f"\n  Opsin-specific analysis for {source_pop.upper()}:")
            
            # 1. Opsin+ (excited - suppressed)
            geometric_bootstrap[f'{source_pop}_opsin_plus'] = bootstrap_geometric_mean_nested(
                geometric_stats,
                source_pop,
                statistic='geometric_mean_ratio_opsin_plus',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            mann_whitney_bootstrap[f'{source_pop}_opsin_plus'] = bootstrap_mann_whitney_nested(
                mann_whitney_stats,
                source_pop,
                statistic='cles_opsin_plus',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            quantile_bootstrap[f'{source_pop}_opsin_plus'] = bootstrap_quantile_differences_nested(
                quantile_stats,
                source_pop,
                quantiles=quantiles,
                statistic_key='quantile_diffs_opsin_plus',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            # 2. Opsin- (excited - suppressed)
            geometric_bootstrap[f'{source_pop}_opsin_minus'] = bootstrap_geometric_mean_nested(
                geometric_stats,
                source_pop,
                statistic='geometric_mean_ratio_opsin_minus',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            mann_whitney_bootstrap[f'{source_pop}_opsin_minus'] = bootstrap_mann_whitney_nested(
                mann_whitney_stats,
                source_pop,
                statistic='cles_opsin_minus',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            quantile_bootstrap[f'{source_pop}_opsin_minus'] = bootstrap_quantile_differences_nested(
                quantile_stats,
                source_pop,
                quantiles=quantiles,
                statistic_key='quantile_diffs_opsin_minus',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            # 3. Opsin+ vs Opsin- to excited cells
            geometric_bootstrap[f'{source_pop}_opsin_diff_to_excited'] = bootstrap_geometric_mean_nested(
                geometric_stats,
                source_pop,
                statistic='geometric_mean_ratio_opsin_plus_minus_to_excited',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            mann_whitney_bootstrap[f'{source_pop}_opsin_diff_to_excited'] = bootstrap_mann_whitney_nested(
                mann_whitney_stats,
                source_pop,
                statistic='cles_opsin_plus_minus_to_excited',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            quantile_bootstrap[f'{source_pop}_opsin_diff_to_excited'] = bootstrap_quantile_differences_nested(
                quantile_stats,
                source_pop,
                quantiles=quantiles,
                statistic_key='quantile_diffs_opsin_plus_minus_to_excited',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            # 4. Opsin+ vs Opsin- to suppressed cells
            geometric_bootstrap[f'{source_pop}_opsin_diff_to_suppressed'] = bootstrap_geometric_mean_nested(
                geometric_stats,
                source_pop,
                statistic='geometric_mean_ratio_opsin_plus_minus_to_suppressed',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            mann_whitney_bootstrap[f'{source_pop}_opsin_diff_to_suppressed'] = bootstrap_mann_whitney_nested(
                mann_whitney_stats,
                source_pop,
                statistic='cles_opsin_plus_minus_to_suppressed',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
            
            quantile_bootstrap[f'{source_pop}_opsin_diff_to_suppressed'] = bootstrap_quantile_differences_nested(
                quantile_stats,
                source_pop,
                quantiles=quantiles,
                statistic_key='quantile_diffs_opsin_plus_minus_to_suppressed',
                n_bootstrap=n_bootstrap,
                random_seed=random_seed
            )
    
    # Print summary
    _print_distributional_summary(
        source_populations,
        target_population,
        geometric_bootstrap,
        mann_whitney_bootstrap,
        quantile_bootstrap
    )

    # Export per-cell data to CSV if requested
    if export_csv_path is not None:
        export_per_cell_weights_to_csv(
            classifications=classifications,
            circuit=circuit,
            source_populations=source_populations,
            post_population=post_population,
            target_population=target_population,
            trials_by_connectivity=trials_by_connectivity,
            expression_threshold=expression_threshold,
            csv_path=f"{target_population}_{post_population}_{export_csv_path}"
        )
        
    print(f"\n{'='*80}\n")
    
    return {
        'classifications': classifications,
        'weights_by_connectivity': weights_by_connectivity,
        'geometric_stats_by_connectivity': geometric_stats,
        'mann_whitney_stats_by_connectivity': mann_whitney_stats,
        'quantile_stats_by_connectivity': quantile_stats,
        'geometric_bootstrap_results': geometric_bootstrap,
        'mann_whitney_bootstrap_results': mann_whitney_bootstrap,
        'quantile_bootstrap_results': quantile_bootstrap,
        'metadata': {
            'target_population': target_population,
            'post_population': post_population,
            'n_connectivity_instances': n_conn,
            'quantiles_analyzed': quantiles,
            'threshold_std': threshold_std,
            'expression_threshold': expression_threshold
        }
    }


def _print_distributional_summary(
    source_populations: List[str],
    target_population: str,
    geometric_bootstrap: Dict,
    mann_whitney_bootstrap: Dict,
    quantile_bootstrap: Dict
):
    """Print formatted summary table to console."""
    
    print("\n" + "="*80)
    print("DISTRIBUTIONAL ANALYSIS SUMMARY")
    print("="*80)
    
    for source_pop in source_populations:
        print(f"\nSource Population: {source_pop.upper()}")
        print("-" * 80)
        
        # Geometric mean
        geo_res = geometric_bootstrap.get(source_pop, {})
        if 'mean' in geo_res and not np.isnan(geo_res['mean']):
            print(f"Geometric Mean Analysis:")
            print(f"  Ratio (excited/suppressed): {geo_res['mean']:.3f} "
                  f"[95% CI: {geo_res['ci_lower']:.3f}, {geo_res['ci_upper']:.3f}]")
            print(f"  Interpretation: {geo_res['interpretation']}")
            sig = '***' if geo_res['p_value'] < 0.001 else ('**' if geo_res['p_value'] < 0.01 else ('*' if geo_res['p_value'] < 0.05 else 'n.s.'))
            print(f"  p-value: {geo_res['p_value']:.4f} {sig}")
        
        # Mann-Whitney
        mw_res = mann_whitney_bootstrap.get(source_pop, {})
        if 'mean' in mw_res and not np.isnan(mw_res['mean']):
            print(f"\nMann-Whitney U Analysis:")
            print(f"  CLES: {mw_res['mean']:.3f} "
                  f"[95% CI: {mw_res['ci_lower']:.3f}, {mw_res['ci_upper']:.3f}]")
            print(f"  Interpretation: {mw_res['interpretation']}")
            sig = '***' if mw_res['p_value'] < 0.001 else ('**' if mw_res['p_value'] < 0.01 else ('*' if mw_res['p_value'] < 0.05 else 'n.s.'))
            print(f"  p-value: {mw_res['p_value']:.4f} {sig}")
        
        # Quantile
        q_res = quantile_bootstrap.get(source_pop, {})
        if 'results_by_quantile' in q_res and len(q_res['results_by_quantile']) > 0:
            print(f"\nQuantile Analysis:")
            for q, q_data in q_res['results_by_quantile'].items():
                if not np.isnan(q_data['mean']):
                    sig = '***' if q_data['p_value'] < 0.001 else ('**' if q_data['p_value'] < 0.01 else ('*' if q_data['p_value'] < 0.05 else 'n.s.'))
                    print(f"  {int(q*100)}th percentile diff: {q_data['mean']:.3f} nS "
                          f"[{q_data['ci_lower']:.3f}, {q_data['ci_upper']:.3f}] {sig}")
        
        # Opsin-specific if applicable
        if source_pop == target_population:
            print(f"\n  --- Opsin-Specific Results ---")
            
            # Opsin+ (exc - sup)
            for opsin_label, opsin_key in [('Opsin+', f'{source_pop}_opsin_plus'),
                                            ('Opsin-', f'{source_pop}_opsin_minus')]:
                if opsin_key in geometric_bootstrap:
                    geo_op = geometric_bootstrap[opsin_key]
                    mw_op = mann_whitney_bootstrap.get(opsin_key, {})
                    
                    if 'mean' in geo_op and not np.isnan(geo_op['mean']):
                        sig_geo = '***' if geo_op['p_value'] < 0.001 else ('**' if geo_op['p_value'] < 0.01 else ('*' if geo_op['p_value'] < 0.05 else 'n.s.'))
                        print(f"\n  {opsin_label} (excited - suppressed):")
                        print(f"    Geometric ratio: {geo_op['mean']:.3f} "
                              f"[{geo_op['ci_lower']:.3f}, {geo_op['ci_upper']:.3f}] {sig_geo}")
                    
                    if 'mean' in mw_op and not np.isnan(mw_op['mean']):
                        sig_mw = '***' if mw_op['p_value'] < 0.001 else ('**' if mw_op['p_value'] < 0.01 else ('*' if mw_op['p_value'] < 0.05 else 'n.s.'))
                        print(f"    CLES: {mw_op['mean']:.3f} "
                              f"[{mw_op['ci_lower']:.3f}, {mw_op['ci_upper']:.3f}] {sig_mw}")
            
            # Cross-comparisons
            for comp_label, comp_key in [('Opsin+/- → excited', f'{source_pop}_opsin_diff_to_excited'),
                                         ('Opsin+/- → suppressed', f'{source_pop}_opsin_diff_to_suppressed')]:
                if comp_key in geometric_bootstrap:
                    geo_comp = geometric_bootstrap[comp_key]
                    mw_comp = mann_whitney_bootstrap.get(comp_key, {})
                    
                    if 'mean' in geo_comp and not np.isnan(geo_comp['mean']):
                        sig_geo = '***' if geo_comp['p_value'] < 0.001 else ('**' if geo_comp['p_value'] < 0.01 else ('*' if geo_comp['p_value'] < 0.05 else 'n.s.'))
                        print(f"\n  {comp_label}:")
                        print(f"    Geometric ratio: {geo_comp['mean']:.3f} "
                              f"[{geo_comp['ci_lower']:.3f}, {geo_comp['ci_upper']:.3f}] {sig_geo}")
                    
                    if 'mean' in mw_comp and not np.isnan(mw_comp['mean']):
                        sig_mw = '***' if mw_comp['p_value'] < 0.001 else ('**' if mw_comp['p_value'] < 0.01 else ('*' if mw_comp['p_value'] < 0.05 else 'n.s.'))
                        print(f"    CLES: {mw_comp['mean']:.3f} "
                              f"[{mw_comp['ci_lower']:.3f}, {mw_comp['ci_upper']:.3f}] {sig_mw}")


# ============================================================================
# Distributional Visualization
# ============================================================================


def plot_distributional_analysis_nested(
    distributional_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 16)
) -> plt.Figure:
    """
    Create comprehensive distributional comparison plots.
    
    Layout:
    - Row 1: Log-scale violin plots (one per source population)
    - Row 2: Quantile difference plots (one per source population)
    - Row 3: Summary statistics table
    
    Args:
        distributional_results: Output from analyze_weights_distributional_nested()
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    metadata = distributional_results['metadata']
    target = metadata['target_population']
    post_pop = metadata['post_population']
    
    weights_by_conn = distributional_results['weights_by_connectivity']
    geometric_stats = distributional_results['geometric_stats_by_connectivity']
    mann_whitney_stats = distributional_results['mann_whitney_stats_by_connectivity']
    quantile_stats = distributional_results['quantile_stats_by_connectivity']
    
    geometric_bootstrap = distributional_results['geometric_bootstrap_results']
    mann_whitney_bootstrap = distributional_results['mann_whitney_bootstrap_results']
    quantile_bootstrap = distributional_results['quantile_bootstrap_results']
    
    # Get source populations (excluding opsin variants)
    source_populations = sorted([k for k in geometric_bootstrap.keys()
                                 if not k.endswith(('_opsin_plus', '_opsin_minus',
                                                    '_opsin_diff_to_excited',
                                                    '_opsin_diff_to_suppressed'))])
    n_sources = len(source_populations)
    
    # Create figure with 3 rows
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_sources, hspace=0.4, wspace=0.3,
                          top=0.93, bottom=0.05, left=0.07, right=0.97,
                          height_ratios=[1.2, 1, 0.8])
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db'
    }
    
    # ========================================================================
    # Row 1: Log-scale violin plots
    # ========================================================================
    for source_idx, source_pop in enumerate(source_populations):
        ax = fig.add_subplot(gs[0, source_idx])
        
        # Collect all weights across connectivities
        all_excited = []
        all_suppressed = []
        
        for conn_idx in sorted(weights_by_conn.keys()):
            if source_pop in weights_by_conn[conn_idx]:
                weights_exc = weights_by_conn[conn_idx][source_pop]['weights_excited']
                weights_sup = weights_by_conn[conn_idx][source_pop]['weights_suppressed']
                
                if len(weights_exc) > 0:
                    all_excited.extend(weights_exc)
                if len(weights_sup) > 0:
                    all_suppressed.extend(weights_sup)
        
        if len(all_excited) > 0 and len(all_suppressed) > 0:
            # Violin plot on log scale
            data_to_plot = [all_excited, all_suppressed]
            positions = [1, 2]
            
            parts = ax.violinplot(data_to_plot, positions=positions,
                                 showmeans=False, showmedians=False, widths=0.6)
            
            for pc, color in zip(parts['bodies'], [colors['excited'], colors['suppressed']]):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Add geometric means
            geo_res = geometric_bootstrap[source_pop]
            if 'mean' in geo_res and not np.isnan(geo_res['mean']):
                geo_mean_exc = np.mean([geometric_stats[c][source_pop]['geometric_mean_excited']
                                       for c in sorted(geometric_stats.keys())
                                       if source_pop in geometric_stats[c] and
                                       not np.isnan(geometric_stats[c][source_pop]['geometric_mean_excited'])])
                geo_mean_sup = np.mean([geometric_stats[c][source_pop]['geometric_mean_suppressed']
                                       for c in sorted(geometric_stats.keys())
                                       if source_pop in geometric_stats[c] and
                                       not np.isnan(geometric_stats[c][source_pop]['geometric_mean_suppressed'])])
                
                ax.scatter([1], [geo_mean_exc], color=colors['excited'],
                          marker='D', s=100, edgecolor='black', linewidth=2, zorder=5,
                          label='Geometric mean')
                ax.scatter([2], [geo_mean_sup], color=colors['suppressed'],
                          marker='D', s=100, edgecolor='black', linewidth=2, zorder=5)
            
            # Add medians
            median_exc = np.median(all_excited)
            median_sup = np.median(all_suppressed)
            ax.scatter([1], [median_exc], color=colors['excited'],
                      marker='_', s=200, linewidth=3, zorder=5, label='Median')
            ax.scatter([2], [median_sup], color=colors['suppressed'],
                      marker='_', s=200, linewidth=3, zorder=5)
            
            ax.set_yscale('log')
            ax.set_ylabel('Synaptic Weight (nS)\n[log scale]', fontsize=10)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Excited', 'Suppressed'], fontsize=10)
            ax.set_title(f'{source_pop.upper()} $\\rightarrow$ {post_pop.upper()}',
                        fontsize=11, fontweight='bold')
            
            # Add statistics annotation
            geo_text = (f"Geom. ratio: {geo_res['mean']:.2f}\n"
                       f"CLES: {mann_whitney_bootstrap[source_pop]['mean']:.2f}\n"
                       f"N conn: {geo_res['n_connectivity']}")
            ax.text(0.98, 0.98, geo_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            if source_idx == 0:
                ax.legend(fontsize=8, loc='upper left')
            
            ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Row 2: Quantile difference plots
    # ========================================================================
    for source_idx, source_pop in enumerate(source_populations):
        ax = fig.add_subplot(gs[1, source_idx])
        
        q_res = quantile_bootstrap[source_pop]
        
        if 'results_by_quantile' in q_res and len(q_res['results_by_quantile']) > 0:
            quantiles = q_res['quantiles']
            means = []
            ci_lowers = []
            ci_uppers = []
            p_values = []
            
            for q in quantiles:
                if q in q_res['results_by_quantile']:
                    q_data = q_res['results_by_quantile'][q]
                    means.append(q_data['mean'])
                    ci_lowers.append(q_data['ci_lower'])
                    ci_uppers.append(q_data['ci_upper'])
                    p_values.append(q_data['p_value'])
                else:
                    means.append(np.nan)
                    ci_lowers.append(np.nan)
                    ci_uppers.append(np.nan)
                    p_values.append(np.nan)
            
            # Convert to percentiles for x-axis
            percentiles = [q * 100 for q in quantiles]
            
            # Plot with color by significance
            for i, (pct, mean, ci_low, ci_up, p) in enumerate(zip(percentiles, means, ci_lowers, ci_uppers, p_values)):
                if not np.isnan(mean):
                    color = 'red' if p < 0.001 else ('orange' if p < 0.05 else 'gray')
                    
                    ax.plot(pct, mean, 'o', color=color, markersize=10, zorder=3)
                    ax.plot([pct, pct], [ci_low, ci_up], '-', color=color,
                           linewidth=2.5, zorder=2)
            
            # Connect points with line
            valid_idx = ~np.isnan(means)
            if np.any(valid_idx):
                ax.plot(np.array(percentiles)[valid_idx], np.array(means)[valid_idx],
                       '-', color='black', alpha=0.3, linewidth=1, zorder=1)
            
            ax.axhline(0, color='black', linestyle='--', alpha=0.5, zorder=0)
            ax.set_xlabel('Quantile (Percentile)', fontsize=10)
            ax.set_ylabel('Weight Difference (nS)\n[Excited - Suppressed]', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='red', label='p < 0.001'),
                mpatches.Patch(color='orange', label='p < 0.05'),
                mpatches.Patch(color='gray', label='n.s.')
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
    
    # ========================================================================
    # Row 3: Summary statistics table (text)
    # ========================================================================
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Create summary text
    summary_lines = []
    summary_lines.append("DISTRIBUTIONAL ANALYSIS SUMMARY")
    summary_lines.append("=" * 100)
    
    for source_pop in source_populations:
        summary_lines.append(f"\n{source_pop.upper()} -> {post_pop.upper()}:")
        
        geo_res = geometric_bootstrap[source_pop]
        mw_res = mann_whitney_bootstrap[source_pop]
        
        if 'mean' in geo_res and not np.isnan(geo_res['mean']):
            sig_geo = '***' if geo_res['p_value'] < 0.001 else ('**' if geo_res['p_value'] < 0.01 else ('*' if geo_res['p_value'] < 0.05 else 'n.s.'))
            summary_lines.append(f"  Geometric mean ratio: {geo_res['mean']:.3f} [{geo_res['ci_lower']:.3f}, {geo_res['ci_upper']:.3f}] {sig_geo}")
            summary_lines.append(f"    {geo_res['interpretation']}")
        
        if 'mean' in mw_res and not np.isnan(mw_res['mean']):
            sig_mw = '***' if mw_res['p_value'] < 0.001 else ('**' if mw_res['p_value'] < 0.01 else ('*' if mw_res['p_value'] < 0.05 else 'n.s.'))
            summary_lines.append(f"  CLES: {mw_res['mean']:.3f} [{mw_res['ci_lower']:.3f}, {mw_res['ci_upper']:.3f}] {sig_mw}")
            summary_lines.append(f"    {mw_res['interpretation']}")
    
    summary_text = "\n".join(summary_lines)
    ax_table.text(0.05, 0.95, summary_text,
                 transform=ax_table.transAxes,
                 fontsize=9, verticalalignment='top',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Overall title
    fig.suptitle(f'Distributional Analysis: {target.upper()} Stimulation $\\rightarrow$ {post_pop.upper()} Response\n'
                f'(N={metadata["n_connectivity_instances"]} connectivity instances)',
                fontsize=14, fontweight='bold')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distributional analysis plot to: {save_path}")
    
    return fig

"""
Enhanced Distributional Analysis Plotting with Opsin Separation

Modifications to nested_weights_dist_analysis.py plotting functions:
1. Separate panels for opsin+ and opsin- when applicable
2. P-values displayed on top row panels
3. Summary plots across all post-synaptic populations

Add these functions to nested_weights_dist_analysis.py or import them separately.
"""


def plot_distributional_analysis_nested_opsin_aware(
    distributional_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 16)
) -> plt.Figure:
    """
    Enhanced distributional comparison plots with automatic opsin separation.
    
    Layout (dynamically adjusted):
    - Row 1: Log-scale violin plots (separate panels for opsin+ and opsin- if applicable)
    - Row 2: Quantile difference plots (separate panels for opsin+ and opsin- if applicable)
    - Row 3: Summary statistics table
    
    Args:
        distributional_results: Output from analyze_weights_distributional_nested()
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    metadata = distributional_results['metadata']
    target = metadata['target_population']
    post_pop = metadata['post_population']
    
    weights_by_conn = distributional_results['weights_by_connectivity']
    geometric_stats = distributional_results['geometric_stats_by_connectivity']
    mann_whitney_stats = distributional_results['mann_whitney_stats_by_connectivity']
    quantile_stats = distributional_results['quantile_stats_by_connectivity']
    
    geometric_bootstrap = distributional_results['geometric_bootstrap_results']
    mann_whitney_bootstrap = distributional_results['mann_whitney_bootstrap_results']
    quantile_bootstrap = distributional_results['quantile_bootstrap_results']
    
    # Get source populations (excluding opsin variants)
    source_populations = sorted([k for k in geometric_bootstrap.keys()
                                 if not k.endswith(('_opsin_plus', '_opsin_minus',
                                                    '_opsin_diff_to_excited',
                                                    '_opsin_diff_to_suppressed'))])
    
    # Build panel structure: each source may have 1 or 3 panels (all, opsin+, opsin-)
    panel_structure = []
    for source_pop in source_populations:
        has_opsin = f'{source_pop}_opsin_plus' in geometric_bootstrap
        if has_opsin:
            panel_structure.append((source_pop, 'all'))
            panel_structure.append((source_pop, 'opsin_plus'))
            panel_structure.append((source_pop, 'opsin_minus'))
        else:
            panel_structure.append((source_pop, 'all'))
    
    n_panels = len(panel_structure)
    
    # Create figure with 3 rows
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_panels, hspace=0.4, wspace=0.3,
                          top=0.93, bottom=0.05, left=0.05, right=0.98,
                          height_ratios=[1.2, 1, 0.8])
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db'
    }
    
    # ========================================================================
    # Row 1: Log-scale violin plots with p-values
    # ========================================================================
    for panel_idx, (source_pop, panel_type) in enumerate(panel_structure):
        ax = fig.add_subplot(gs[0, panel_idx])
        
        # Determine which weights to plot
        if panel_type == 'all':
            suffix = ''
            title_suffix = ''
            geo_key = source_pop
            mw_key = source_pop
        elif panel_type == 'opsin_plus':
            suffix = '_opsin_plus'
            title_suffix = ' (Opsin+)'
            geo_key = f'{source_pop}_opsin_plus'
            mw_key = f'{source_pop}_opsin_plus'
        else:  # opsin_minus
            suffix = '_opsin_minus'
            title_suffix = ' (Opsin-)'
            geo_key = f'{source_pop}_opsin_minus'
            mw_key = f'{source_pop}_opsin_minus'
        
        # Collect weights across connectivities
        all_excited = []
        all_suppressed = []
        
        for conn_idx in sorted(weights_by_conn.keys()):
            if source_pop in weights_by_conn[conn_idx]:
                if panel_type == 'all':
                    weights_exc = weights_by_conn[conn_idx][source_pop]['weights_excited']
                    weights_sup = weights_by_conn[conn_idx][source_pop]['weights_suppressed']
                elif panel_type == 'opsin_plus':
                    weights_exc = weights_by_conn[conn_idx][source_pop].get('weights_excited_opsin_plus', [])
                    weights_sup = weights_by_conn[conn_idx][source_pop].get('weights_suppressed_opsin_plus', [])
                else:  # opsin_minus
                    weights_exc = weights_by_conn[conn_idx][source_pop].get('weights_excited_opsin_minus', [])
                    weights_sup = weights_by_conn[conn_idx][source_pop].get('weights_suppressed_opsin_minus', [])
                
                if len(weights_exc) > 0:
                    all_excited.extend(weights_exc)
                if len(weights_sup) > 0:
                    all_suppressed.extend(weights_sup)
        
        if len(all_excited) > 0 and len(all_suppressed) > 0:
            # Violin plot on log scale
            data_to_plot = [all_excited, all_suppressed]
            positions = [1, 2]
            
            parts = ax.violinplot(data_to_plot, positions=positions,
                                 showmeans=False, showmedians=False, widths=0.6)
            
            for pc, color in zip(parts['bodies'], [colors['excited'], colors['suppressed']]):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Add geometric means
            geo_res = geometric_bootstrap.get(geo_key, {})
            if 'mean' in geo_res and not np.isnan(geo_res['mean']):
                # Compute geometric means from stats
                geo_means_exc = []
                geo_means_sup = []
                
                for c in sorted(geometric_stats.keys()):
                    if source_pop in geometric_stats[c]:
                        if panel_type == 'all':
                            gm_exc = geometric_stats[c][source_pop].get('geometric_mean_excited', np.nan)
                            gm_sup = geometric_stats[c][source_pop].get('geometric_mean_suppressed', np.nan)
                        elif panel_type == 'opsin_plus':
                            gm_exc = geometric_stats[c][source_pop].get('geometric_mean_excited_opsin_plus', np.nan)
                            gm_sup = geometric_stats[c][source_pop].get('geometric_mean_suppressed_opsin_plus', np.nan)
                        else:  # opsin_minus
                            gm_exc = geometric_stats[c][source_pop].get('geometric_mean_excited_opsin_minus', np.nan)
                            gm_sup = geometric_stats[c][source_pop].get('geometric_mean_suppressed_opsin_minus', np.nan)
                        
                        if not np.isnan(gm_exc):
                            geo_means_exc.append(gm_exc)
                        if not np.isnan(gm_sup):
                            geo_means_sup.append(gm_sup)
                
                if len(geo_means_exc) > 0:
                    geo_mean_exc = np.mean(geo_means_exc)
                    ax.scatter([1], [geo_mean_exc], color=colors['excited'],
                              marker='D', s=100, edgecolor='black', linewidth=2, zorder=5,
                              label='Geometric mean')
                
                if len(geo_means_sup) > 0:
                    geo_mean_sup = np.mean(geo_means_sup)
                    ax.scatter([2], [geo_mean_sup], color=colors['suppressed'],
                              marker='D', s=100, edgecolor='black', linewidth=2, zorder=5)
            
            # Add medians
            median_exc = np.median(all_excited)
            median_sup = np.median(all_suppressed)
            ax.scatter([1], [median_exc], color=colors['excited'],
                      marker='_', s=200, linewidth=3, zorder=5, label='Median')
            ax.scatter([2], [median_sup], color=colors['suppressed'],
                      marker='_', s=200, linewidth=3, zorder=5)
            
            ax.set_yscale('log')
            ax.set_ylabel('Synaptic Weight (nS)\n[log scale]', fontsize=9)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Excited', 'Suppressed'], fontsize=9)
            ax.set_title(f'{source_pop.upper()} $\\rightarrow$ {post_pop.upper()}{title_suffix}',
                        fontsize=10, fontweight='bold')
            
            # Add statistics annotation WITH P-VALUES
            mw_res = mann_whitney_bootstrap.get(mw_key, {})
            if 'mean' in geo_res and 'mean' in mw_res:
                # Format p-values
                p_geo = geo_res.get('p_value', np.nan)
                p_mw = mw_res.get('p_value', np.nan)
                
                sig_geo = _format_significance(p_geo)
                sig_mw = _format_significance(p_mw)
                
                stats_text = (f"Geom. ratio: {geo_res['mean']:.2f}\n"
                             f"  p = {p_geo:.4f} {sig_geo}\n"
                             f"CLES: {mw_res['mean']:.2f}\n"
                             f"  p = {p_mw:.4f} {sig_mw}\n"
                             f"N conn: {geo_res['n_connectivity']}")
                
                ax.text(0.98, 0.98, stats_text,
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            if panel_idx == 0:
                ax.legend(fontsize=7, loc='upper left')
            
            ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Row 2: Quantile difference plots
    # ========================================================================
    for panel_idx, (source_pop, panel_type) in enumerate(panel_structure):
        ax = fig.add_subplot(gs[1, panel_idx])
        
        # Determine which quantile results to plot
        if panel_type == 'all':
            q_key = source_pop
        elif panel_type == 'opsin_plus':
            q_key = f'{source_pop}_opsin_plus'
        else:  # opsin_minus
            q_key = f'{source_pop}_opsin_minus'
        
        q_res = quantile_bootstrap.get(q_key, {})
        
        if 'results_by_quantile' in q_res and len(q_res['results_by_quantile']) > 0:
            quantiles = q_res['quantiles']
            means = []
            ci_lowers = []
            ci_uppers = []
            p_values = []
            
            for q in quantiles:
                if q in q_res['results_by_quantile']:
                    q_data = q_res['results_by_quantile'][q]
                    means.append(q_data['mean'])
                    ci_lowers.append(q_data['ci_lower'])
                    ci_uppers.append(q_data['ci_upper'])
                    p_values.append(q_data['p_value'])
                else:
                    means.append(np.nan)
                    ci_lowers.append(np.nan)
                    ci_uppers.append(np.nan)
                    p_values.append(np.nan)
            
            # Convert to percentiles for x-axis
            percentiles = [q * 100 for q in quantiles]
            
            # Plot with color by significance
            for i, (pct, mean, ci_low, ci_up, p) in enumerate(zip(percentiles, means, ci_lowers, ci_uppers, p_values)):
                if not np.isnan(mean):
                    color = 'red' if p < 0.001 else ('orange' if p < 0.05 else 'gray')
                    
                    ax.plot(pct, mean, 'o', color=color, markersize=10, zorder=3)
                    ax.plot([pct, pct], [ci_low, ci_up], '-', color=color,
                           linewidth=2.5, zorder=2)
            
            # Connect points with line
            valid_idx = ~np.isnan(means)
            if np.any(valid_idx):
                ax.plot(np.array(percentiles)[valid_idx], np.array(means)[valid_idx],
                       '-', color='black', alpha=0.3, linewidth=1, zorder=1)
            
            ax.axhline(0, color='black', linestyle='--', alpha=0.5, zorder=0)
            ax.set_xlabel('Quantile (Percentile)', fontsize=9)
            ax.set_ylabel('Weight Difference (nS)\n[Excited - Suppressed]', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add legend only on first panel
            if panel_idx == 0:
                legend_elements = [
                    mpatches.Patch(color='red', label='p < 0.001'),
                    mpatches.Patch(color='orange', label='p < 0.05'),
                    mpatches.Patch(color='gray', label='n.s.')
                ]
                ax.legend(handles=legend_elements, fontsize=7, loc='upper left')
    
    # ========================================================================
    # Row 3: Summary statistics table (text)
    # ========================================================================
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Create summary text
    summary_lines = []
    summary_lines.append("DISTRIBUTIONAL ANALYSIS SUMMARY")
    summary_lines.append("=" * 120)
    
    for source_pop in source_populations:
        has_opsin = f'{source_pop}_opsin_plus' in geometric_bootstrap
        
        if has_opsin:
            summary_lines.append(f"\n{source_pop.upper()} -> {post_pop.upper()} [WITH OPSIN SEPARATION]:")
        else:
            summary_lines.append(f"\n{source_pop.upper()} -> {post_pop.upper()}:")
        
        # Standard results
        geo_res = geometric_bootstrap[source_pop]
        mw_res = mann_whitney_bootstrap[source_pop]
        
        if 'mean' in geo_res and not np.isnan(geo_res['mean']):
            sig_geo = _format_significance(geo_res['p_value'])
            summary_lines.append(f"  ALL CELLS: Geom. ratio = {geo_res['mean']:.3f} [{geo_res['ci_lower']:.3f}, {geo_res['ci_upper']:.3f}] {sig_geo}")
        
        if 'mean' in mw_res and not np.isnan(mw_res['mean']):
            sig_mw = _format_significance(mw_res['p_value'])
            summary_lines.append(f"             CLES = {mw_res['mean']:.3f} [{mw_res['ci_lower']:.3f}, {mw_res['ci_upper']:.3f}] {sig_mw}")
        
        # Opsin-specific results
        if has_opsin:
            for opsin_label, opsin_key in [('OPSIN+', f'{source_pop}_opsin_plus'),
                                           ('OPSIN-', f'{source_pop}_opsin_minus')]:
                geo_op = geometric_bootstrap.get(opsin_key, {})
                mw_op = mann_whitney_bootstrap.get(opsin_key, {})
                
                if 'mean' in geo_op and not np.isnan(geo_op['mean']):
                    sig_geo = _format_significance(geo_op['p_value'])
                    summary_lines.append(f"  {opsin_label}:      Geom. ratio = {geo_op['mean']:.3f} [{geo_op['ci_lower']:.3f}, {geo_op['ci_upper']:.3f}] {sig_geo}")
                
                if 'mean' in mw_op and not np.isnan(mw_op['mean']):
                    sig_mw = _format_significance(mw_op['p_value'])
                    summary_lines.append(f"             CLES = {mw_op['mean']:.3f} [{mw_op['ci_lower']:.3f}, {mw_op['ci_upper']:.3f}] {sig_mw}")
    
    summary_text = "\n".join(summary_lines)
    ax_table.text(0.02, 0.98, summary_text,
                 transform=ax_table.transAxes,
                 fontsize=8, verticalalignment='top',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Overall title
    fig.suptitle(f'Distributional Analysis: {target.upper()} Stimulation $\\rightarrow$ {post_pop.upper()} Response\n'
                f'(N={metadata["n_connectivity_instances"]} connectivity instances)',
                fontsize=14, fontweight='bold')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distributional analysis plot to: {save_path}")
    
    return fig


def plot_distributional_violin_grid_across_populations(
    all_distributional_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    show_stats: bool = True,
    show_pvalues: bool = True
) -> plt.Figure:
    """
    Create violin plot grid showing weight distributions across all post-synaptic populations.
    
    Layout:
    - Each row = one post-synaptic population
    - Each column = one source population (opsin variants shown as separate columns)
    - Each cell = violin plot of excited vs suppressed weight distributions
    
    Args:
        all_distributional_results: Dict mapping post_pop -> distributional_results
        save_path: Optional path to save figure
        figsize: Figure size (width, height). If None, auto-calculated from grid size
        show_stats: Whether to show geometric mean and median markers
        show_pvalues: Whether to show p-values in text boxes
        
    Returns:
        matplotlib Figure object
    """
    post_populations = sorted(all_distributional_results.keys())
    n_post = len(post_populations)
    
    # Get source organization from first result
    first_result = all_distributional_results[post_populations[0]]
    geometric_bootstrap = first_result['geometric_bootstrap_results']
    source_entries = _organize_sources_with_opsin_separation(geometric_bootstrap)
    n_sources = len(source_entries)
    
    target = first_result['metadata']['target_population']
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        # Each violin panel needs ~2-3 inches width, ~2 inches height
        fig_width = max(20, n_sources * 2.5 + 2)
        fig_height = max(12, n_post * 2.5 + 2)
        figsize = (fig_width, fig_height)
    
    # Create figure
    fig, axes = plt.subplots(n_post, n_sources, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.05, left=0.05, right=0.98)
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db'
    }
    
    # Iterate over post-populations (rows)
    for row_idx, post_pop in enumerate(post_populations):
        results = all_distributional_results[post_pop]
        weights_by_conn = results['weights_by_connectivity']
        geometric_stats = results['geometric_stats_by_connectivity']
        geometric_bootstrap = results['geometric_bootstrap_results']
        mann_whitney_bootstrap = results['mann_whitney_bootstrap_results']
        
        # Iterate over source populations (columns)
        for col_idx, entry in enumerate(source_entries):
            ax = axes[row_idx, col_idx]
            
            source_key = entry['key']
            source_label = entry['label']
            base_source = entry['base_source']
            
            # Determine panel type for weight extraction
            if source_key == base_source:
                panel_type = 'all'
            elif 'opsin_plus' in source_key:
                panel_type = 'opsin_plus'
            else:  # opsin_minus
                panel_type = 'opsin_minus'
            
            # Collect weights across connectivities
            all_excited = []
            all_suppressed = []
            
            for conn_idx in sorted(weights_by_conn.keys()):
                if base_source in weights_by_conn[conn_idx]:
                    if panel_type == 'all':
                        weights_exc = weights_by_conn[conn_idx][base_source]['weights_excited']
                        weights_sup = weights_by_conn[conn_idx][base_source]['weights_suppressed']
                    elif panel_type == 'opsin_plus':
                        weights_exc = weights_by_conn[conn_idx][base_source].get('weights_excited_opsin_plus', [])
                        weights_sup = weights_by_conn[conn_idx][base_source].get('weights_suppressed_opsin_plus', [])
                    else:  # opsin_minus
                        weights_exc = weights_by_conn[conn_idx][base_source].get('weights_excited_opsin_minus', [])
                        weights_sup = weights_by_conn[conn_idx][base_source].get('weights_suppressed_opsin_minus', [])
                    
                    if len(weights_exc) > 0:
                        all_excited.extend(weights_exc)
                    if len(weights_sup) > 0:
                        all_suppressed.extend(weights_sup)
            
            # Plot violin if we have data
            if len(all_excited) > 0 and len(all_suppressed) > 0:
                data_to_plot = [all_excited, all_suppressed]
                positions = [1, 2]
                
                # Create violin plot
                parts = ax.violinplot(data_to_plot, positions=positions,
                                     showmeans=False, showmedians=False, widths=0.6)

                for pc, color in zip(parts['bodies'], [colors['excited'], colors['suppressed']]):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.2)


                # Plot individual connectivity-level geometric means
                geo_means_exc_per_conn = []
                geo_means_sup_per_conn = []

                for c in sorted(geometric_stats.keys()):
                    if base_source in geometric_stats[c]:
                        if panel_type == 'all':
                            gm_exc = geometric_stats[c][base_source].get('geometric_mean_excited', np.nan)
                            gm_sup = geometric_stats[c][base_source].get('geometric_mean_suppressed', np.nan)
                        elif panel_type == 'opsin_plus':
                            gm_exc = geometric_stats[c][base_source].get('geometric_mean_excited_opsin_plus', np.nan)
                            gm_sup = geometric_stats[c][base_source].get('geometric_mean_suppressed_opsin_plus', np.nan)
                        else:  # opsin_minus
                            gm_exc = geometric_stats[c][base_source].get('geometric_mean_excited_opsin_minus', np.nan)
                            gm_sup = geometric_stats[c][base_source].get('geometric_mean_suppressed_opsin_minus', np.nan)

                        if not np.isnan(gm_exc):
                            geo_means_exc_per_conn.append(gm_exc)
                        if not np.isnan(gm_sup):
                            geo_means_sup_per_conn.append(gm_sup)

                # Plot individual points with jitter for visibility
                if len(geo_means_exc_per_conn) > 0:
                    jitter_exc = np.random.normal(0, 0.04, len(geo_means_exc_per_conn))
                    ax.scatter(1 + jitter_exc, geo_means_exc_per_conn, 
                               color=colors['excited'], s=20, alpha=0.4, 
                               edgecolor='black', linewidth=0.5, zorder=4)

                if len(geo_means_sup_per_conn) > 0:
                    jitter_sup = np.random.normal(0, 0.04, len(geo_means_sup_per_conn))
                    ax.scatter(2 + jitter_sup, geo_means_sup_per_conn,
                               color=colors['suppressed'], s=20, alpha=0.4,
                               edgecolor='black', linewidth=0.5, zorder=4)
                
                if show_stats:
                    # Add geometric means
                    geo_res = geometric_bootstrap.get(source_key, {})
                    if 'mean' in geo_res and not np.isnan(geo_res['mean']):
                        geo_means_exc = []
                        geo_means_sup = []
                        
                        for c in sorted(geometric_stats.keys()):
                            if base_source in geometric_stats[c]:
                                if panel_type == 'all':
                                    gm_exc = geometric_stats[c][base_source].get('geometric_mean_excited', np.nan)
                                    gm_sup = geometric_stats[c][base_source].get('geometric_mean_suppressed', np.nan)
                                elif panel_type == 'opsin_plus':
                                    gm_exc = geometric_stats[c][base_source].get('geometric_mean_excited_opsin_plus', np.nan)
                                    gm_sup = geometric_stats[c][base_source].get('geometric_mean_suppressed_opsin_plus', np.nan)
                                else:  # opsin_minus
                                    gm_exc = geometric_stats[c][base_source].get('geometric_mean_excited_opsin_minus', np.nan)
                                    gm_sup = geometric_stats[c][base_source].get('geometric_mean_suppressed_opsin_minus', np.nan)
                                
                                if not np.isnan(gm_exc):
                                    geo_means_exc.append(gm_exc)
                                if not np.isnan(gm_sup):
                                    geo_means_sup.append(gm_sup)
                        
                        if len(geo_means_exc) > 0:
                            geo_mean_exc = np.mean(geo_means_exc)
                            ax.scatter([1], [geo_mean_exc], color=colors['excited'],
                                       marker='D', s=60, edgecolor='black', linewidth=1.5, zorder=5)
                        
                        if len(geo_means_sup) > 0:
                            geo_mean_sup = np.mean(geo_means_sup)
                            ax.scatter([2], [geo_mean_sup], color=colors['suppressed'],
                                       marker='D', s=60, edgecolor='black', linewidth=1.5, zorder=5)
                    
                    # Add medians
                    median_exc = np.median(all_excited)
                    median_sup = np.median(all_suppressed)
                    ax.scatter([1], [median_exc], color=colors['excited'],
                              marker='_', s=150, linewidth=2.5, zorder=5)
                    ax.scatter([2], [median_sup], color=colors['suppressed'],
                              marker='_', s=150, linewidth=2.5, zorder=5)
                
                # Add p-values annotation if requested
                if show_pvalues:
                    geo_res = geometric_bootstrap.get(source_key, {})
                    mw_res = mann_whitney_bootstrap.get(source_key, {})
                    
                    if 'mean' in geo_res and 'mean' in mw_res:
                        p_geo = geo_res.get('p_value', np.nan)
                        p_mw = mw_res.get('p_value', np.nan)
                        
                        sig_geo = _format_significance(p_geo)
                        sig_mw = _format_significance(p_mw)
                        
                        stats_text = (f"W: {geo_res['mean']:.2f} {sig_geo}\n"
                                      f"M: {mw_res['mean']:.2f} {sig_mw}")
                        
                        ax.text(0.65, 0.98, stats_text,
                               transform=ax.transAxes, fontsize=7,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                
                # Formatting
                ax.set_yscale('log')
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Exc', 'Sup'], fontsize=8)
                ax.grid(True, alpha=0.2, axis='y')
                
                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel(f'{post_pop.upper()}\nWeight (nS)', fontsize=9, fontweight='bold')
                else:
                    ax.set_ylabel('')
                
                # Column titles on first row
                if row_idx == 0:
                    ax.set_title(source_label, fontsize=9, fontweight='bold')
                
            else:
                # No data - show empty panel
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=8, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                
                if col_idx == 0:
                    ax.set_ylabel(f'{post_pop.upper()}', fontsize=9, fontweight='bold')
                
                if row_idx == 0:
                    ax.set_title(source_label, fontsize=9, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'Weight Distribution Grid: {target.upper()} Stimulation Across All Post-Synaptic Populations\n'
                f'(Excited vs Suppressed, Log Scale)',
                fontsize=13, fontweight='bold')
    
    # Add legend
    if show_stats:
        legend_elements = [
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                      markeredgecolor='black', markersize=8, label='Log-transformed mean of weights'),
            plt.Line2D([0], [0], marker='_', color='gray', markersize=10,
                       linewidth=2.5, label='Median'),
            mpatches.Patch(color='none', label='W = Mean ratio of weights (Exc/Sup)'),
            mpatches.Patch(color='none', label='M = Mann-Whitney CLES (Common Language Effect Size)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, 0.01), fontsize=9, frameon=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distributional violin grid plot to: {save_path}")
    
    return fig

def plot_distributional_violin_grid_across_populations_with_boxplot(
    all_distributional_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    show_stats: bool = True,
    show_pvalues: bool = True
) -> plt.Figure:
    """
    Create violin plot grid with box plots showing connectivity-level variance.
    
    Visualization strategy:
    - Violin plots (semi-transparent): within-connectivity variance (all synaptic weights)
    - Box plots (overlaid): between-connectivity variance (geometric means per connectivity)
    - Individual dots (jittered): individual connectivity geometric means
    
    Args:
        all_distributional_results: Dict mapping post_pop -> distributional_results
        save_path: Optional path to save figure
        figsize: Figure size (width, height). If None, auto-calculated from grid size
        show_stats: Whether to show statistics in text boxes
        show_pvalues: Whether to show p-values in text boxes
        
    Returns:
        matplotlib Figure object
    """
    post_populations = sorted(all_distributional_results.keys())
    n_post = len(post_populations)
    
    # Get source organization from first result
    first_result = all_distributional_results[post_populations[0]]
    geometric_bootstrap = first_result['geometric_bootstrap_results']
    source_entries = _organize_sources_with_opsin_separation(geometric_bootstrap)
    n_sources = len(source_entries)
    
    target = first_result['metadata']['target_population']
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        fig_width = max(20, n_sources * 2.5 + 2)
        fig_height = max(12, n_post * 2.5 + 2)
        figsize = (fig_width, fig_height)
    
    # Create figure
    fig, axes = plt.subplots(n_post, n_sources, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.08, left=0.05, right=0.98)
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db'
    }
    
    # Iterate over post-populations (rows)
    for row_idx, post_pop in enumerate(post_populations):
        results = all_distributional_results[post_pop]
        weights_by_conn = results['weights_by_connectivity']
        geometric_stats = results['geometric_stats_by_connectivity']
        geometric_bootstrap = results['geometric_bootstrap_results']
        mann_whitney_bootstrap = results['mann_whitney_bootstrap_results']
        
        # Iterate over source populations (columns)
        for col_idx, entry in enumerate(source_entries):
            ax = axes[row_idx, col_idx]
            
            source_key = entry['key']
            source_label = entry['label']
            base_source = entry['base_source']
            
            # Determine panel type for weight extraction
            if source_key == base_source:
                panel_type = 'all'
            elif 'opsin_plus' in source_key:
                panel_type = 'opsin_plus'
            else:  # opsin_minus
                panel_type = 'opsin_minus'
            
            # Collect weights across connectivities
            all_excited = []
            all_suppressed = []
            
            # Collect connectivity-level geometric means
            geo_means_exc_per_conn = []
            geo_means_sup_per_conn = []
            
            for conn_idx in sorted(weights_by_conn.keys()):
                if base_source in weights_by_conn[conn_idx]:
                    # Get weights for violin plot
                    if panel_type == 'all':
                        weights_exc = weights_by_conn[conn_idx][base_source]['weights_excited']
                        weights_sup = weights_by_conn[conn_idx][base_source]['weights_suppressed']
                    elif panel_type == 'opsin_plus':
                        weights_exc = weights_by_conn[conn_idx][base_source].get('weights_excited_opsin_plus', [])
                        weights_sup = weights_by_conn[conn_idx][base_source].get('weights_suppressed_opsin_plus', [])
                    else:  # opsin_minus
                        weights_exc = weights_by_conn[conn_idx][base_source].get('weights_excited_opsin_minus', [])
                        weights_sup = weights_by_conn[conn_idx][base_source].get('weights_suppressed_opsin_minus', [])
                    
                    if len(weights_exc) > 0:
                        all_excited.extend(weights_exc)
                    if len(weights_sup) > 0:
                        all_suppressed.extend(weights_sup)
                    
                    # Get geometric means for box plot
                    if base_source in geometric_stats[conn_idx]:
                        if panel_type == 'all':
                            gm_exc = geometric_stats[conn_idx][base_source].get('geometric_mean_excited', np.nan)
                            gm_sup = geometric_stats[conn_idx][base_source].get('geometric_mean_suppressed', np.nan)
                        elif panel_type == 'opsin_plus':
                            gm_exc = geometric_stats[conn_idx][base_source].get('geometric_mean_excited_opsin_plus', np.nan)
                            gm_sup = geometric_stats[conn_idx][base_source].get('geometric_mean_suppressed_opsin_plus', np.nan)
                        else:  # opsin_minus
                            gm_exc = geometric_stats[conn_idx][base_source].get('geometric_mean_excited_opsin_minus', np.nan)
                            gm_sup = geometric_stats[conn_idx][base_source].get('geometric_mean_suppressed_opsin_minus', np.nan)
                        
                        if not np.isnan(gm_exc):
                            geo_means_exc_per_conn.append(gm_exc)
                        if not np.isnan(gm_sup):
                            geo_means_sup_per_conn.append(gm_sup)
            
            # Plot if we have data
            if len(all_excited) > 0 and len(all_suppressed) > 0:
                # 1. Violin plot (within-connectivity variance)
                data_to_plot = [all_excited, all_suppressed]
                positions = [1, 2]
                
                parts = ax.violinplot(data_to_plot, positions=positions,
                                     showmeans=False, showmedians=False, widths=0.6)
                
                for pc, color in zip(parts['bodies'], [colors['excited'], colors['suppressed']]):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.25)  # More transparent to see box plot
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.0)
                    pc.set_label('All synapses' if color == colors['excited'] else None)
                
                # 2. Box plot (between-connectivity variance)
                if len(geo_means_exc_per_conn) > 0 and len(geo_means_sup_per_conn) > 0:
                    box_data = [geo_means_exc_per_conn, geo_means_sup_per_conn]
                    box_positions = [1, 2]
                    
                    bp = ax.boxplot(box_data, positions=box_positions, widths=0.3,
                                   patch_artist=True, showfliers=False,
                                   boxprops=dict(linewidth=2, zorder=5),
                                   whiskerprops=dict(linewidth=2, zorder=5),
                                   capprops=dict(linewidth=2, zorder=5),
                                   medianprops=dict(linewidth=2.5, color='black', zorder=6))
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], [colors['excited'], colors['suppressed']]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.5)
                        patch.set_edgecolor('black')
                    
                    # 3. Individual connectivity points with jitter
                    jitter_exc = np.random.normal(0, 0.04, len(geo_means_exc_per_conn))
                    jitter_sup = np.random.normal(0, 0.04, len(geo_means_sup_per_conn))
                    
                    ax.scatter(1 + jitter_exc, geo_means_exc_per_conn, 
                               color=colors['excited'], s=20, alpha=0.6, 
                               edgecolor='black', linewidth=0.5, zorder=7)
                    ax.scatter(2 + jitter_sup, geo_means_sup_per_conn,
                               color=colors['suppressed'], s=20, alpha=0.6,
                               edgecolor='black', linewidth=0.5, zorder=7)
                
                if show_stats:
                    # Grand geometric mean marker
                    geo_res = geometric_bootstrap.get(source_key, {})
                    if 'mean' in geo_res and not np.isnan(geo_res['mean']):
                        if len(geo_means_exc_per_conn) > 0:
                            grand_mean_exc = np.mean(geo_means_exc_per_conn)
                            ax.scatter([1], [grand_mean_exc], color=colors['excited'],
                                       marker='D', s=80, edgecolor='black', linewidth=2, zorder=8)
                        
                        if len(geo_means_sup_per_conn) > 0:
                            grand_mean_sup = np.mean(geo_means_sup_per_conn)
                            ax.scatter([2], [grand_mean_sup], color=colors['suppressed'],
                                       marker='D', s=80, edgecolor='black', linewidth=2, zorder=8)
                
                # Add p-values annotation if requested
                if show_pvalues:
                    geo_res = geometric_bootstrap.get(source_key, {})
                    mw_res = mann_whitney_bootstrap.get(source_key, {})
                    
                    if 'mean' in geo_res and 'mean' in mw_res:
                        p_geo = geo_res.get('p_value', np.nan)
                        p_mw = mw_res.get('p_value', np.nan)
                        
                        sig_geo = _format_significance(p_geo)
                        sig_mw = _format_significance(p_mw)
                        
                        stats_text = (f"W: {geo_res['mean']:.2f} {sig_geo}\n"
                                      f"M: {mw_res['mean']:.2f} {sig_mw}")
                        
                        ax.text(0.65, 0.98, stats_text,
                               transform=ax.transAxes, fontsize=7,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                
                # Formatting
                ax.set_yscale('log')
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Exc', 'Sup'], fontsize=8)
                ax.grid(True, alpha=0.2, axis='y')
                
                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel(f'{post_pop.upper()}\nWeight (nS)', fontsize=9, fontweight='bold')
                else:
                    ax.set_ylabel('')
                
                # Column titles on first row
                if row_idx == 0:
                    ax.set_title(source_label, fontsize=9, fontweight='bold')
                
            else:
                # No data - show empty panel
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=8, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                
                if col_idx == 0:
                    ax.set_ylabel(f'{post_pop.upper()}', fontsize=9, fontweight='bold')
                
                if row_idx == 0:
                    ax.set_title(source_label, fontsize=9, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'Weight Distribution Grid: {target.upper()} Stimulation Across All Post-Synaptic Populations\n'
                f'(Excited vs Suppressed, Log Scale)',
                fontsize=13, fontweight='bold')
    
    # Enhanced legend explaining the two variance levels
    legend_elements = [
        mpatches.Patch(facecolor='gray', alpha=0.25, edgecolor='black', linewidth=1,
                      label='Violin: All synapses (within-connectivity variance)'),
        mpatches.Patch(facecolor='gray', alpha=0.8, edgecolor='black', linewidth=2,
                      label='Box: Connectivity means (between-connectivity variance)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markeredgecolor='black', markersize=5, alpha=0.6,
                  label='Individual connectivity geometric means'),
        #plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
        #          markeredgecolor='black', markersize=8,
        #          label='Grand mean (across connectivities)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.01), fontsize=9, frameon=True)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distributional violin grid plot with box overlays to: {save_path}")
    
    return fig


def plot_quantile_summary_across_populations(
    all_distributional_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Create summary plots showing quantile profiles across all post-synaptic populations.
    
    Each row = one post-synaptic population
    Each source population (with opsin variants if applicable) shown in separate color/style
    
    Args:
        all_distributional_results: Dict mapping post_pop -> distributional_results
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    post_populations = sorted(all_distributional_results.keys())
    n_post = len(post_populations)
    
    fig, axes = plt.subplots(n_post, 1, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.4)
    
    # Color palette for different sources
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    linestyles = {
        'all': '-',
        'opsin_plus': '--',
        'opsin_minus': ':',
        'opsin_diff_to_excited': '-.',
        'opsin_diff_to_suppressed': '-.'
    }
    
    for row_idx, post_pop in enumerate(post_populations):
        ax = axes[row_idx, 0]
        results = all_distributional_results[post_pop]
        quantile_bootstrap = results['quantile_bootstrap_results']
        target = results['metadata']['target_population']
        
        # Get all source populations
        all_keys = sorted(quantile_bootstrap.keys())
        
        # Organize by source
        sources_organized = {}
        for key in all_keys:
            if '_opsin_' in key:
                base_source = key.split('_opsin_')[0]
                if 'diff_to_excited' in key:
                    variant = 'opsin_diff_to_excited'
                elif 'diff_to_suppressed' in key:
                    variant = 'opsin_diff_to_suppressed'
                elif 'plus' in key:
                    variant = 'opsin_plus'
                else:
                    variant = 'opsin_minus'
                
                if base_source not in sources_organized:
                    sources_organized[base_source] = {}
                sources_organized[base_source][variant] = key
            else:
                if key not in sources_organized:
                    sources_organized[key] = {}
                sources_organized[key]['all'] = key
        
        # Plot each source
        color_idx = 0
        for source_base in sorted(sources_organized.keys()):
            variants = sources_organized[source_base]
            color = colors[color_idx % len(colors)]
            
            for variant_type, full_key in variants.items():
                q_res = quantile_bootstrap[full_key]
                
                if 'results_by_quantile' in q_res and len(q_res['results_by_quantile']) > 0:
                    quantiles = q_res['quantiles']
                    means = []
                    ci_lowers = []
                    ci_uppers = []
                    
                    for q in quantiles:
                        if q in q_res['results_by_quantile']:
                            q_data = q_res['results_by_quantile'][q]
                            means.append(q_data['mean'])
                            ci_lowers.append(q_data['ci_lower'])
                            ci_uppers.append(q_data['ci_upper'])
                        else:
                            means.append(np.nan)
                            ci_lowers.append(np.nan)
                            ci_uppers.append(np.nan)
                    
                    percentiles = [q * 100 for q in quantiles]
                    
                    # Plot mean line
                    valid_idx = ~np.isnan(means)
                    if np.any(valid_idx):
                        label = f"{source_base.upper()}"
                        if variant_type != 'all':
                            label += f" ({variant_type.replace('_', ' ')})"
                        
                        ax.plot(np.array(percentiles)[valid_idx],
                               np.array(means)[valid_idx],
                               linestyle=linestyles[variant_type],
                               color=color, linewidth=2, marker='o',
                               markersize=6, label=label, alpha=0.8)
                        
                        # Add CI as shaded region
                        ax.fill_between(np.array(percentiles)[valid_idx],
                                       np.array(ci_lowers)[valid_idx],
                                       np.array(ci_uppers)[valid_idx],
                                       color=color, alpha=0.15)
            
            color_idx += 1
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, zorder=0)
        ax.set_xlabel('Quantile (Percentile)', fontsize=10)
        ax.set_ylabel('Weight Difference (nS)\n[Excited - Suppressed]', fontsize=10)
        ax.set_title(f'{target.upper()} $\\rightarrow$ {post_pop.upper()}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best', ncol=2)
    
    fig.suptitle(f'Quantile Analysis Summary Across Post-Synaptic Populations',
                fontsize=14, fontweight='bold')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved quantile summary plot to: {save_path}")
    
    return fig


# ============================================================================
# Helper Functions
# ============================================================================


def _format_significance(p_value: float) -> str:
    """Format p-value with significance stars."""
    if np.isnan(p_value):
        return ''
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s.'


def _plot_metric_comparison(
    ax,
    sources_organized: Dict,
    bootstrap_results: Dict,
    metric_name: str,
    null_value: float,
    post_pop: str,
    target: str
):
    """
    Plot comparison of a single metric across source populations.
    
    Args:
        ax: Matplotlib axis
        sources_organized: Organized source populations with variants
        bootstrap_results: Bootstrap results dict
        metric_name: Name of metric being plotted
        null_value: Null hypothesis value (1.0 for ratio, 0.5 for CLES)
        post_pop: Post-synaptic population name
        target: Target (stimulated) population name
    """
    # Collect all data points
    positions = []
    means = []
    ci_lowers = []
    ci_uppers = []
    labels = []
    colors_list = []
    p_values = []
    
    pos = 0
    base_colors = plt.cm.Set2(np.linspace(0, 1, len(sources_organized)))
    
    for color_idx, (source_base, variants_dict) in enumerate(sorted(sources_organized.items())):
        base_color = base_colors[color_idx]
        
        # Plot 'all' first if exists
        if variants_dict.get('all') is not None:
            key = variants_dict['all']
            res = bootstrap_results.get(key, {})
            
            if 'mean' in res and not np.isnan(res['mean']):
                positions.append(pos)
                means.append(res['mean'])
                ci_lowers.append(res['ci_lower'])
                ci_uppers.append(res['ci_upper'])
                labels.append(source_base.upper())
                colors_list.append(base_color)
                p_values.append(res.get('p_value', np.nan))
                pos += 1
        
        # Plot variants
        variant_order = ['opsin_plus', 'opsin_minus', 'opsin_diff_to_excited', 'opsin_diff_to_suppressed']
        for variant in variant_order:
            if variant in variants_dict:
                key = variants_dict[variant]
                res = bootstrap_results.get(key, {})
                
                if 'mean' in res and not np.isnan(res['mean']):
                    positions.append(pos)
                    means.append(res['mean'])
                    ci_lowers.append(res['ci_lower'])
                    ci_uppers.append(res['ci_upper'])
                    
                    # Create label
                    if variant == 'opsin_plus':
                        label = f"{source_base.upper()}\nOpsin+"
                    elif variant == 'opsin_minus':
                        label = f"{source_base.upper()}\nOpsin-"
                    elif variant == 'opsin_diff_to_excited':
                        label = f"{source_base.upper()}\nOp+/- → Exc"
                    else:
                        label = f"{source_base.upper()}\nOp+/- → Sup"
                    
                    labels.append(label)
                    # Use lighter shade for variants
                    variant_color = tuple(list(base_color[:3]) + [0.6])
                    colors_list.append(variant_color)
                    p_values.append(res.get('p_value', np.nan))
                    pos += 1
        
        # Add spacing between different source populations
        pos += 0.5
    
    # Plot
    for i, (p, m, cl, cu, c, pval) in enumerate(zip(positions, means, ci_lowers, ci_uppers, colors_list, p_values)):
        # Color by significance
        edge_color = 'red' if pval < 0.001 else ('orange' if pval < 0.05 else 'black')
        edge_width = 2.5 if pval < 0.05 else 1.5
        
        ax.errorbar(p, m, yerr=[[m - cl], [cu - m]],
                   fmt='o', color=c, markersize=10,
                   capsize=5, capthick=2, elinewidth=2,
                   markeredgecolor=edge_color, markeredgewidth=edge_width,
                   zorder=3)
    
    # Add null line
    ax.axhline(null_value, color='black', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_title(f'{target.upper()} $\\rightarrow$ {post_pop.upper()}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for significance
    legend_elements = [
        mpatches.Patch(edgecolor='red', facecolor='white', linewidth=2.5, label='p < 0.001'),
        mpatches.Patch(edgecolor='orange', facecolor='white', linewidth=2.5, label='p < 0.05'),
        mpatches.Patch(edgecolor='black', facecolor='white', linewidth=1.5, label='n.s.')
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')




def _organize_sources_with_opsin_separation(bootstrap_results: Dict) -> List[Dict]:
    """
    Organize source populations with opsin-expressing and non-expressing as separate entities.
    
    Returns:
        List of dicts with keys: 'key' (result key), 'label' (display label), 
                                 'base_source' (for grouping), 'is_opsin' (bool)
    """
    # Get all keys
    all_keys = sorted(bootstrap_results.keys())
    
    # Identify base sources and their opsin variants
    base_sources = set()
    opsin_keys = set()
    
    for key in all_keys:
        if '_opsin_' in key:
            opsin_keys.add(key)
            # Extract base source
            base = key.split('_opsin_')[0]
            base_sources.add(base)
        else:
            base_sources.add(key)
    
    # Build source entries list
    source_entries = []
    
    for base_source in sorted(base_sources):
        # Check if this source has opsin variants
        has_opsin_plus = f'{base_source}_opsin_plus' in bootstrap_results
        has_opsin_minus = f'{base_source}_opsin_minus' in bootstrap_results
        
        if has_opsin_plus or has_opsin_minus:
            # This source has opsin separation
            # Add all cells entry
            if base_source in bootstrap_results:
                source_entries.append({
                    'key': base_source,
                    'label': f'{base_source.upper()} (All)',
                    'base_source': base_source,
                    'is_opsin': False
                })
            
            # Add opsin+ entry
            if has_opsin_plus:
                source_entries.append({
                    'key': f'{base_source}_opsin_plus',
                    'label': f'{base_source.upper()} (Opsin+)',
                    'base_source': base_source,
                    'is_opsin': True
                })
            
            # Add opsin- entry
            if has_opsin_minus:
                source_entries.append({
                    'key': f'{base_source}_opsin_minus',
                    'label': f'{base_source.upper()} (Opsin-)',
                    'base_source': base_source,
                    'is_opsin': True
                })
        else:
            # No opsin separation, just add the base
            if base_source in bootstrap_results:
                source_entries.append({
                    'key': base_source,
                    'label': base_source.upper(),
                    'base_source': base_source,
                    'is_opsin': False
                })
    
    return source_entries


def _plot_metric_comparison_opsin_separated(
    ax,
    source_entries: List[Dict],
    bootstrap_results: Dict,
    metric_name: str,
    null_value: float,
    post_pop: str,
    target: str
):
    """
    Plot comparison of a single metric across source populations with opsin separation.
    Opsin+ and opsin- are treated as separate, equal entities.
    
    Args:
        ax: Matplotlib axis
        source_entries: List of source entry dicts from _organize_sources_with_opsin_separation()
        bootstrap_results: Bootstrap results dict
        metric_name: Name of metric being plotted
        null_value: Null hypothesis value (1.0 for ratio, 0.5 for CLES)
        post_pop: Post-synaptic population name
        target: Target (stimulated) population name
    """
    # Collect data points
    positions = []
    means = []
    ci_lowers = []
    ci_uppers = []
    labels = []
    colors_list = []
    p_values = []
    
    # Color scheme: different shades for same base source
    base_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    base_color_map = {}
    color_idx = 0
    
    pos = 0
    last_base_source = None
    
    for entry in source_entries:
        key = entry['key']
        label = entry['label']
        base_source = entry['base_source']
        is_opsin = entry['is_opsin']
        
        res = bootstrap_results.get(key, {})
        
        if 'mean' in res and not np.isnan(res['mean']):
            # Assign color
            if base_source not in base_color_map:
                base_color_map[base_source] = base_colors[color_idx % len(base_colors)]
                color_idx += 1
            
            base_color = base_color_map[base_source]
            
            # Adjust shade for opsin variants
            if 'Opsin+' in label:
                # Darker shade for opsin+
                color = tuple(np.array(base_color[:3]) * 0.7)
            elif 'Opsin-' in label:
                # Lighter shade for opsin-
                color = tuple(np.array(base_color[:3]) * 1.0 + 0.15)
                color = tuple(np.clip(color, 0, 1))
            elif 'All' in label:
                # Medium shade for all
                color = base_color
            else:
                color = base_color
            
            positions.append(pos)
            means.append(res['mean'])
            ci_lowers.append(res['ci_lower'])
            ci_uppers.append(res['ci_upper'])
            labels.append(label)
            colors_list.append(color)
            p_values.append(res.get('p_value', np.nan))
            
            # Add small spacing between different base sources
            if last_base_source is not None and base_source != last_base_source:
                pos += 1.5
            else:
                pos += 1.0
            
            last_base_source = base_source
    
    # Plot
    for i, (p, m, cl, cu, c, pval) in enumerate(zip(positions, means, ci_lowers, ci_uppers, colors_list, p_values)):
        # Color by significance
        if pval < 0.001:
            edge_color = 'darkred'
            edge_width = 3.0
            marker_size = 12
        elif pval < 0.05:
            edge_color = 'darkorange'
            edge_width = 2.5
            marker_size = 11
        else:
            edge_color = 'black'
            edge_width = 1.5
            marker_size = 10
        
        ax.errorbar(p, m, yerr=[[m - cl], [cu - m]],
                   fmt='o', color=c, markersize=marker_size,
                   capsize=5, capthick=2, elinewidth=2,
                   markeredgecolor=edge_color, markeredgewidth=edge_width,
                   zorder=3)
    
    # Add null line
    ax.axhline(null_value, color='black', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'{target.upper()} $\\rightarrow$ {post_pop.upper()}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add y-axis limits with padding
    if len(means) > 0:
        y_min = min([cl for cl in ci_lowers if not np.isnan(cl)])
        y_max = max([cu for cu in ci_uppers if not np.isnan(cu)])
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Add legend for significance
    legend_elements = [
        mpatches.Patch(edgecolor='darkred', facecolor='white', linewidth=3.0, label='p < 0.001'),
        mpatches.Patch(edgecolor='darkorange', facecolor='white', linewidth=2.5, label='p < 0.05'),
        mpatches.Patch(edgecolor='black', facecolor='white', linewidth=1.5, label='n.s.')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
