"""
Nested Weights Analysis Framework

Extends the single-connectivity weight analysis to properly handle hierarchical
structure of nested experiments (connectivity instances × MEC patterns).

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

# Import from existing modules
from nested_experiment import NestedTrialResult
from nested_effect_size import (organize_nested_trials,
                                average_activity_across_trials,
                                classify_cells_by_connectivity,
                                extract_weights_by_connectivity)


# ============================================================================
# Weight Distribution Analysis (Nested)
# ============================================================================

def extract_weight_distributions_by_connectivity(
    circuit,
    classifications: Dict,
    post_population: str,
    source_populations: List[str]
) -> Dict:
    """
    Extract full weight distributions (not just totals) for each connectivity instance
    
    Unlike extract_weights_by_connectivity which sums weights, this extracts
    individual synaptic weight values for distribution analysis.
    
    Args:
        circuit: DentateCircuit instance
        classifications: Output from classify_cells_by_connectivity()
        post_population: Post-synaptic population
        source_populations: List of source populations to analyze
        
    Returns:
        dict: {
            connectivity_idx: {
                source_population: {
                    'weights_excited': array of individual synapse weights,
                    'weights_suppressed': array of individual synapse weights,
                    'weights_unchanged': array of individual synapse weights,
                    'total_input_excited': array of total input per cell,
                    'total_input_suppressed': array of total input per cell,
                    'total_input_unchanged': array of total input per cell
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
        
        for conn_idx, classification in classifications.items():
            if conn_idx not in weights_by_connectivity:
                weights_by_connectivity[conn_idx] = {}
            
            if source_pop not in weights_by_connectivity[conn_idx]:
                weights_by_connectivity[conn_idx][source_pop] = {}
            
            for response_type in ['excited', 'suppressed', 'unchanged']:
                cell_indices = classification[f'{response_type}_cells']
                
                if len(cell_indices) > 0:
                    # Individual synapse weights (all incoming connections)
                    weights_to_cells = conductances_np[:, cell_indices]
                    individual_weights = weights_to_cells[weights_to_cells > 0]
                    
                    # Total input per cell
                    total_input = np.sum(conductances_np[:, cell_indices], axis=0)
                    
                    weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}'] = individual_weights
                    weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}'] = total_input
                else:
                    weights_by_connectivity[conn_idx][source_pop][f'weights_{response_type}'] = np.array([])
                    weights_by_connectivity[conn_idx][source_pop][f'total_input_{response_type}'] = np.array([])
    
    return weights_by_connectivity


def compute_weight_statistics_by_connectivity(
    weights_by_connectivity: Dict,
    source_populations: List[str]
) -> Dict:
    """
    Compute statistics for each connectivity instance
    
    Args:
        weights_by_connectivity: Output from extract_weight_distributions_by_connectivity()
        source_populations: List of source populations
        
    Returns:
        dict: {
            connectivity_idx: {
                source_population: {
                    'mean_excited': float,
                    'mean_suppressed': float,
                    'mean_unchanged': float,
                    'std_excited': float,
                    'std_suppressed': float,
                    'std_unchanged': float,
                    'mean_diff_exc_sup': float,  # excited - suppressed
                    'n_synapses_excited': int,
                    'n_synapses_suppressed': int,
                    'n_cells_excited': int,
                    'n_cells_suppressed': int
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
            
            weights_excited = source_data['weights_excited']
            weights_suppressed = source_data['weights_suppressed']
            weights_unchanged = source_data.get('weights_unchanged', np.array([]))
            
            total_input_excited = source_data['total_input_excited']
            total_input_suppressed = source_data['total_input_suppressed']
            
            stats_by_connectivity[conn_idx][source_pop] = {
                # Individual synapse statistics
                'mean_excited': np.mean(weights_excited) if len(weights_excited) > 0 else np.nan,
                'mean_suppressed': np.mean(weights_suppressed) if len(weights_suppressed) > 0 else np.nan,
                'mean_unchanged': np.mean(weights_unchanged) if len(weights_unchanged) > 0 else np.nan,
                'std_excited': np.std(weights_excited) if len(weights_excited) > 0 else np.nan,
                'std_suppressed': np.std(weights_suppressed) if len(weights_suppressed) > 0 else np.nan,
                'std_unchanged': np.std(weights_unchanged) if len(weights_unchanged) > 0 else np.nan,
                
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
    random_seed: Optional[int] = None
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
        source_populations
    )
    
    # Compute statistics per connectivity
    stats_by_connectivity = compute_weight_statistics_by_connectivity(
        weights_by_connectivity,
        source_populations
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
        'summary_statistics': summary_statistics,
        'metadata': {
            'target_population': target_population,
            'post_population': post_population,
            'n_connectivity_instances': n_conn,
            'threshold_std': threshold_std,
            'expression_threshold': expression_threshold
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
    figsize: Tuple[int, int] = (20, 14)
) -> plt.Figure:
    """
    Create comprehensive visualization of nested weight analysis
    
    Creates multi-panel figure showing:
    - Panel A: Weight distributions by connectivity instance (violin plots)
    - Panel B: Mean weights with between-connectivity variance (bar + error bars)
    - Panel C: Effect sizes with bootstrap CIs (forest plot)
    - Panel D: Bootstrap distributions for each source
    
    Args:
        analysis_results: Output from analyze_weights_by_average_response_nested()
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
    source_populations = sorted(bootstrap_results.keys())
    n_sources = len(source_populations)
    n_conn = metadata['n_connectivity_instances']
    
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, n_sources, hspace=0.4, wspace=0.3,
                         top=0.93, bottom=0.05, left=0.07, right=0.97)
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db',
        'unchanged': '#95a5a6'
    }
    
    # Panel A: Weight distributions by connectivity (violin plots)
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
        labels = []
        
        for i, conn_idx in enumerate(sorted(data_by_conn.keys())):
            # Excited
            if len(data_by_conn[conn_idx]['excited']) > 0:
                positions.append(i * 3)
                data_to_plot.append(data_by_conn[conn_idx]['excited'])
                plot_colors.append(colors['excited'])
                labels.append(f'C{conn_idx}\nExc')
            
            # Suppressed
            if len(data_by_conn[conn_idx]['suppressed']) > 0:
                positions.append(i * 3 + 1)
                data_to_plot.append(data_by_conn[conn_idx]['suppressed'])
                plot_colors.append(colors['suppressed'])
                labels.append(f'C{conn_idx}\nSup')
        
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
        
        # Add connectivity labels
        if positions:
            conn_positions = [i * 3 + 0.5 for i in range(n_conn)]
            ax.set_xticks(conn_positions)
            ax.set_xticklabels([f'C{i}' for i in range(n_conn)], fontsize=8)
    
    # Panel B: Mean weights with between-connectivity error bars
    for source_idx, source_pop in enumerate(source_populations):
        ax = fig.add_subplot(gs[1, source_idx])
        
        # Collect means and stds across connectivities
        conn_indices = sorted(stats_by_conn.keys())
        excited_means = []
        suppressed_means = []
        
        for conn_idx in conn_indices:
            if source_pop in stats_by_conn[conn_idx]:
                data = stats_by_conn[conn_idx][source_pop]
                excited_means.append(data['mean_excited'] if not np.isnan(data['mean_excited']) else 0)
                suppressed_means.append(data['mean_suppressed'] if not np.isnan(data['mean_suppressed']) else 0)
        
        x_pos = np.arange(len(conn_indices))
        width = 0.35
        
        # Plot bars
        ax.bar(x_pos - width/2, excited_means, width,
              label='Excited', color=colors['excited'], alpha=0.7,
              edgecolor='black', linewidth=1.5)
        ax.bar(x_pos + width/2, suppressed_means, width,
              label='Suppressed', color=colors['suppressed'], alpha=0.7,
              edgecolor='black', linewidth=1.5)
        
        # Add grand means as horizontal lines
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
        
        # Add difference annotation
        boot_result = bootstrap_results[source_pop]
        if not np.isnan(boot_result['mean']):
            ax.text(0.02, 0.98,
                   f"$\\Delta$ = {boot_result['mean']:.3f} nS\n"
                   f"95% CI: [{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]\n"
                   f"p = {boot_result['p_value']:.4f}",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Panel C: Effect sizes (forest plot across all sources)
    ax_forest = fig.add_subplot(gs[2, :])
    
    y_positions = np.arange(n_sources)
    
    for i, source_pop in enumerate(source_populations):
        boot_result = bootstrap_results[source_pop]
        
        if not np.isnan(boot_result['mean']):
            mean_val = boot_result['mean']
            ci_lower = boot_result['ci_lower']
            ci_upper = boot_result['ci_upper']
            p_val = boot_result['p_value']
            
            # Color by significance
            color = 'red' if p_val < 0.001 else ('orange' if p_val < 0.05 else 'gray')
            
            # Plot point estimate
            ax_forest.plot(mean_val, y_positions[i], 'o',
                          color=color, markersize=12, zorder=3)
            
            # Plot CI
            ax_forest.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]],
                          '-', color=color, linewidth=3, zorder=2)
            
            # Significance stars
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            elif p_val < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            
            # Annotation
            ax_forest.text(ci_upper + 0.02, y_positions[i],
                          f"{mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] {stars}",
                          va='center', fontsize=9)
    
    ax_forest.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax_forest.set_yticks(y_positions)
    ax_forest.set_yticklabels([f'{s.upper()} $\\rightarrow$ {post_pop.upper()}' 
                               for s in source_populations], fontsize=10)
    ax_forest.set_xlabel('Weight Difference: Excited - Suppressed (nS)\n(Bootstrap 95% CI)',
                        fontsize=11, fontweight='bold')
    ax_forest.set_title('Between-Connectivity Weight Differences',
                       fontsize=12, fontweight='bold')
    ax_forest.grid(True, alpha=0.3, axis='x')
    
    # Panel D: Bootstrap distributions
    for source_idx, source_pop in enumerate(source_populations):
        ax = fig.add_subplot(gs[3, source_idx])
        
        boot_result = bootstrap_results[source_pop]
        
        if 'bootstrap_distribution' in boot_result and boot_result['bootstrap_distribution'] is not None:
            bootstrap_dist = boot_result['bootstrap_distribution']
            
            ax.hist(bootstrap_dist, bins=50, color='steelblue',
                   alpha=0.7, edgecolor='black')
            
            # Mark observed mean
            ax.axvline(boot_result['mean'], color='red',
                      linestyle='--', linewidth=2, label='Observed', zorder=3)
            
            # Mark CI bounds
            ax.axvline(boot_result['ci_lower'], color='orange',
                      linestyle=':', linewidth=2, zorder=3)
            ax.axvline(boot_result['ci_upper'], color='orange',
                      linestyle=':', linewidth=2, label='95% CI', zorder=3)
            
            # Mark zero
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=2)
            
            ax.set_xlabel('Weight Difference (nS)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add effect size
            es = effect_sizes[source_pop]
            ax.text(0.02, 0.98,
                   f"Cohen's d = {es['cohens_d']:.3f}\n"
                   f"N conn = {boot_result['n_connectivity']}",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Overall title
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
    """
    Plot detailed comparison across specific connectivity instances
    
    Shows weight distributions and statistics for selected connectivities
    to visualize between-connectivity variance.
    
    Args:
        analysis_results: Output from analyze_weights_by_average_response_nested()
        connectivity_indices: Which connectivities to plot (default: all)
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    metadata = analysis_results['metadata']
    weights_by_conn = analysis_results['weights_by_connectivity']
    stats_by_conn = analysis_results['stats_by_connectivity']
    
    if connectivity_indices is None:
        connectivity_indices = sorted(weights_by_conn.keys())
    
    n_conn = len(connectivity_indices)
    source_populations = sorted(list(weights_by_conn[connectivity_indices[0]].keys()))
    n_sources = len(source_populations)
    
    fig, axes = plt.subplots(n_conn, n_sources, figsize=figsize,
                            squeeze=False)
    
    colors = {
        'excited': '#e74c3c',
        'suppressed': '#3498db'
    }
    
    for conn_idx_pos, conn_idx in enumerate(connectivity_indices):
        for source_idx, source_pop in enumerate(source_populations):
            ax = axes[conn_idx_pos, source_idx]
            
            # Get data
            weights_data = weights_by_conn[conn_idx][source_pop]
            stats_data = stats_by_conn[conn_idx][source_pop]
            
            weights_excited = weights_data['weights_excited']
            weights_suppressed = weights_data['weights_suppressed']
            
            # Plot histograms
            if len(weights_excited) > 0 and len(weights_suppressed) > 0:
                bins = np.linspace(0, max(np.max(weights_excited), np.max(weights_suppressed)), 30)
                
                ax.hist(weights_excited, bins=bins, alpha=0.6,
                       color=colors['excited'], label='Excited',
                       edgecolor='black')
                ax.hist(weights_suppressed, bins=bins, alpha=0.6,
                       color=colors['suppressed'], label='Suppressed',
                       edgecolor='black')
                
                # Add means
                ax.axvline(stats_data['mean_excited'], color=colors['excited'],
                          linestyle='--', linewidth=2)
                ax.axvline(stats_data['mean_suppressed'], color=colors['suppressed'],
                          linestyle='--', linewidth=2)
                
                # Annotation
                ax.text(0.98, 0.98,
                       f"$\\Delta$ = {stats_data['mean_diff_exc_sup']:.3f} nS\n"
                       f"n_exc = {stats_data['n_synapses_excited']}\n"
                       f"n_sup = {stats_data['n_synapses_suppressed']}",
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Weight (nS)', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            
            if conn_idx_pos == 0:
                ax.set_title(f'{source_pop.upper()}', fontsize=10, fontweight='bold')
            
            if source_idx == 0:
                ax.text(-0.3, 0.5, f'Connectivity {conn_idx}',
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       rotation=90, verticalalignment='center')
            
            if conn_idx_pos == 0 and source_idx == 0:
                ax.legend(fontsize=8, loc='upper right')
            
            ax.grid(True, alpha=0.3)
    
    target = metadata['target_population']
    post_pop = metadata['post_population']
    
    fig.suptitle(f'Weight Distributions by Connectivity Instance\n'
                f'{target.upper()} $\\rightarrow$ {post_pop.upper()}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved connectivity comparison plot to: {save_path}")
    
    return fig
