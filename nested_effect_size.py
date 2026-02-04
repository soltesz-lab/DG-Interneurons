"""
Bootstrap Effect Size Analysis for Nested Experiment Data

Computes bootstrap confidence intervals on effect sizes for synaptic weight
differences between excited and suppressed cells, properly accounting for the
hierarchical structure of nested experiments (connectivity * MEC patterns).

Key statistical property:
- ICC ~= 1.0: Cell responses deterministic within connectivity
- Bootstrap at connectivity level (independent units)
- Proper handling of hierarchical variance structure
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# Data Organization
# ============================================================================

def organize_nested_trials(nested_results: List) -> Dict[int, List]:
    """
    Group trials by connectivity instance
    
    Args:
        nested_results: List of NestedTrialResult objects
        
    Returns:
        dict: {connectivity_idx: [list of trial results], ...}
    """
    trials_by_connectivity = defaultdict(list)
    
    for trial in nested_results:
        conn_idx = trial.connectivity_idx
        trials_by_connectivity[conn_idx].append(trial)
    
    return dict(trials_by_connectivity)


def average_activity_across_trials(
    trials: List,
    population: str
) -> torch.Tensor:
    """
    Average activity across all trials (MEC patterns) for a connectivity
    
    Args:
        trials: List of trial results from same connectivity
        population: Population name to extract
        
    Returns:
        Averaged activity tensor [n_cells, n_timesteps]
    """
    activities = []
    
    for trial in trials:
        # Handle both aggregated and raw results
        if 'activity_trace_mean' in trial.results:
            activity = trial.results['activity_trace_mean'][population]
        elif 'activity_trace' in trial.results:
            activity = trial.results['activity_trace'][population]
        else:
            raise KeyError(f"Trial results missing both 'activity_trace' and 'activity_trace_mean'")

        # Convert to tensor if needed (handles loaded NumPy arrays)
        if isinstance(activity, np.ndarray):
            activity = torch.from_numpy(activity)
        
        activities.append(activity)
    
    # Average across trials
    return torch.mean(torch.stack(activities), dim=0)


# ============================================================================
# Cell Classification
# ============================================================================

def classify_cells_by_connectivity(
    trials_by_connectivity: Dict[int, List],
    target_population: str,
    post_population: str,
    stim_start: float,
    stim_duration: float,
    warmup: float,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2
) -> Dict:
    """
    For each connectivity, classify cells as excited/suppressed
    
    Args:
        trials_by_connectivity: dict from organize_nested_trials()
        target_population: Stimulated population
        post_population: Post-synaptic population to analyze
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period start (ms)
        threshold_std: Classification threshold (std deviations)
        expression_threshold: Opsin expression threshold for non-expressing cells
        
    Returns:
        dict: {
            connectivity_idx: {
                'excited_cells': array of cell indices,
                'suppressed_cells': array of cell indices,
                'unchanged_cells': array of cell indices,
                'n_excited': int,
                'n_suppressed': int,
                'n_unchanged': int,
                'rate_change': tensor of rate changes for all cells
            },
            ...
        }
    """
    classifications = {}
    
    for conn_idx, trials in trials_by_connectivity.items():
        # Average activity across all MEC patterns for this connectivity
        averaged_activity = average_activity_across_trials(trials, post_population)
        
        # Compute baseline and stimulation periods
        time = trials[0].results['time']
        if hasattr(time, 'cpu'):
            time = time.cpu().numpy()
        
        baseline_mask = (time >= warmup) & (time < stim_start)
        stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
        
        # Compute rate changes
        baseline_rate = torch.mean(averaged_activity[:, baseline_mask], dim=1)
        stim_rate = torch.mean(averaged_activity[:, stim_mask], dim=1)
        rate_change = stim_rate - baseline_rate
        baseline_std = torch.std(baseline_rate)
        
        # For target population, filter non-expressing cells
        if post_population == target_population:
            opsin_expression = trials[0].opsin_expression
            if hasattr(opsin_expression, 'cpu'):
                opsin_expression = opsin_expression.cpu().numpy()
            
            non_expressing_mask = opsin_expression < expression_threshold
            
            # Apply mask to get indices
            all_indices = np.arange(len(rate_change))
            cell_indices = all_indices[non_expressing_mask]
            
            # Filter rate changes
            if isinstance(rate_change, torch.Tensor):
                non_expressing_mask_tensor = torch.from_numpy(non_expressing_mask).to(rate_change.device)
                rate_change_filtered = rate_change[non_expressing_mask_tensor]
            else:
                rate_change_filtered = rate_change[non_expressing_mask]
        else:
            cell_indices = np.arange(len(rate_change))
            rate_change_filtered = rate_change
        
        # Classify cells
        if isinstance(rate_change_filtered, torch.Tensor):
            excited_mask = rate_change_filtered > (threshold_std * baseline_std)
            suppressed_mask = rate_change_filtered < (-threshold_std * baseline_std)
            unchanged_mask = ~(excited_mask | suppressed_mask)  # NEW

            excited_cells = cell_indices[excited_mask.cpu().numpy()]
            suppressed_cells = cell_indices[suppressed_mask.cpu().numpy()]
            unchanged_cells = cell_indices[unchanged_mask.cpu().numpy()]  # NEW
        else:
            excited_mask = rate_change_filtered > (threshold_std * baseline_std)
            suppressed_mask = rate_change_filtered < (-threshold_std * baseline_std)
            unchanged_mask = ~(excited_mask | suppressed_mask)  # NEW
                        
            excited_cells = cell_indices[excited_mask]
            suppressed_cells = cell_indices[suppressed_mask]
            unchanged_cells = cell_indices[unchanged_mask]  # NEW
            
        classifications[conn_idx] = {
            'excited_cells': excited_cells,
            'suppressed_cells': suppressed_cells,
            'unchanged_cells': unchanged_cells,
            'n_excited': len(excited_cells),
            'n_suppressed': len(suppressed_cells),
            'n_unchanged': len(unchanged_cells),
            'rate_change': rate_change_filtered
        }
    
    return classifications


# ============================================================================
# Weight Extraction
# ============================================================================

def extract_weights_by_connectivity(
    circuit,
    classifications: Dict,
    post_population: str,
    source_population: str
) -> Dict:
    """
    Extract synaptic weights for excited vs suppressed cells
    
    Note: Assumes circuit connectivity is the same across trials within
    a connectivity instance (which it is by design)
    
    Args:
        circuit: DentateCircuit instance (from any trial in connectivity)
        classifications: Output from classify_cells_by_connectivity()
        post_population: Post-synaptic population
        source_population: Pre-synaptic source population
        
    Returns:
        dict: {
            connectivity_idx: {
                'weights_excited': array of weights to excited cells,
                'weights_suppressed': array of weights to suppressed cells,
                'weights_unchanged': array of weights to unchanged cells,
                'n_weights_excited': number of non-zero weights,
                'n_weights_suppressed': number of non-zero weights
                'n_weights_unchanged': number of non-zero weights

            },
            ...
        }
    """
    conn_name = f'{source_population}_{post_population}'
    
    if conn_name not in circuit.connectivity.conductance_matrices:
        return {}
    
    conductance_matrix = circuit.connectivity.conductance_matrices[conn_name]
    weights = conductance_matrix.conductances  # [n_pre, n_post]
    
    weights_by_connectivity = {}
    
    for conn_idx, classification in classifications.items():
        excited_cells = classification['excited_cells']
        suppressed_cells = classification['suppressed_cells']
        unchanged_cells = classification['unchanged_cells']
        
        # Extract weights for each group
        # weights[:, excited_cells] gives all inputs to excited cells
        if len(excited_cells) > 0:
            weights_to_excited = weights[:, excited_cells]
            # Flatten and keep only non-zero weights
            weights_excited = weights_to_excited[weights_to_excited > 0]
            if hasattr(weights_excited, 'cpu'):
                weights_excited = weights_excited.cpu().numpy()
            else:
                weights_excited = np.array(weights_excited)
        else:
            weights_excited = np.array([])
        
        if len(suppressed_cells) > 0:
            weights_to_suppressed = weights[:, suppressed_cells]
            weights_suppressed = weights_to_suppressed[weights_to_suppressed > 0]
            if hasattr(weights_suppressed, 'cpu'):
                weights_suppressed = weights_suppressed.cpu().numpy()
            else:
                weights_suppressed = np.array(weights_suppressed)
        else:
            weights_suppressed = np.array([])
        
        weights_by_connectivity[conn_idx] = {
            'weights_excited': weights_excited,
            'weights_suppressed': weights_suppressed,
            'weights_unchanged': weights_unchanged,
            'n_weights_excited': len(weights_excited),
            'n_weights_suppressed': len(weights_suppressed),
            'n_weights_unchanged': len(weights_unchanged)
        }
    
    return weights_by_connectivity


# ============================================================================
# Weight Difference Computation
# ============================================================================

def compute_connectivity_weight_differences(
    weights_by_connectivity: Dict,
    metric: str = 'mean'
) -> Dict[str, np.ndarray]:
    """
    Compute weight differences for all three pairwise comparisons
    
    Args:
        weights_by_connectivity: Output from extract_weights_by_connectivity()
        metric: 'mean' or 'median'
        
    Returns:
        dict: {
            'excited_vs_suppressed': array of Δw per connectivity,
            'excited_vs_unchanged': array of Δw per connectivity,
            'suppressed_vs_unchanged': array of Δw per connectivity
        }
    """
    diff_exc_sup = []
    diff_exc_unch = []
    diff_sup_unch = []
    
    for conn_idx in sorted(weights_by_connectivity.keys()):
        weights_excited = weights_by_connectivity[conn_idx]['weights_excited']
        weights_suppressed = weights_by_connectivity[conn_idx]['weights_suppressed']
        weights_unchanged = weights_by_connectivity[conn_idx]['weights_unchanged']
        
        # Compute statistics based on metric
        if metric == 'mean':
            stat_func = np.mean
        elif metric == 'median':
            stat_func = np.median
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Excited vs Suppressed (original comparison)
        if len(weights_excited) > 0 and len(weights_suppressed) > 0:
            delta_exc_sup = stat_func(weights_excited) - stat_func(weights_suppressed)
            diff_exc_sup.append(delta_exc_sup)
        
        # Excited vs Unchanged
        if len(weights_excited) > 0 and len(weights_unchanged) > 0:
            delta_exc_unch = stat_func(weights_excited) - stat_func(weights_unchanged)
            diff_exc_unch.append(delta_exc_unch)
        
        # Suppressed vs Unchanged
        if len(weights_suppressed) > 0 and len(weights_unchanged) > 0:
            delta_sup_unch = stat_func(weights_suppressed) - stat_func(weights_unchanged)
            diff_sup_unch.append(delta_sup_unch)
    
    return {
        'excited_vs_suppressed': np.array(diff_exc_sup),
        'excited_vs_unchanged': np.array(diff_exc_unch),
        'suppressed_vs_unchanged': np.array(diff_sup_unch)
    }


# ============================================================================
# Bootstrap Analysis
# ============================================================================

def bootstrap_effect_size_nested(
    weight_differences: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Bootstrap confidence intervals for effect size
    
    Args:
        weight_differences: Array of Δw for each connectivity
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level (default 0.95)
        random_seed: Random seed for reproducibility
        
    Returns:
        dict with effect size, CI, p-value, bootstrap distribution
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_conn = len(weight_differences)
    
    if n_conn == 0:
        return {
            'effect_size': np.nan,
            'effect_size_ci_lower': np.nan,
            'effect_size_ci_upper': np.nan,
            'mean_diff': np.nan,
            'mean_diff_ci_lower': np.nan,
            'mean_diff_ci_upper': np.nan,
            'std_diff': np.nan,
            'p_value': np.nan,
            'n_connectivity': 0,
            'n_bootstrap': n_bootstrap
        }
    
    # Original effect size (Cohen's d for one-sample)
    mean_diff = np.mean(weight_differences)
    std_diff = np.std(weight_differences, ddof=1) if n_conn > 1 else 0.0
    
    if std_diff == 0:
        return {
            'effect_size': np.inf if mean_diff != 0 else 0.0,
            'effect_size_ci_lower': np.inf if mean_diff != 0 else 0.0,
            'effect_size_ci_upper': np.inf if mean_diff != 0 else 0.0,
            'mean_diff': mean_diff,
            'mean_diff_ci_lower': mean_diff,
            'mean_diff_ci_upper': mean_diff,
            'std_diff': 0.0,
            'p_value': 0.0 if mean_diff != 0 else 1.0,
            'n_connectivity': n_conn,
            'n_bootstrap': n_bootstrap
        }
    
    effect_size = mean_diff / std_diff
    
    # Bootstrap
    bootstrap_effects = []
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample connectivities with replacement
        resampled = np.random.choice(
            weight_differences,
            size=n_conn,
            replace=True
        )
        
        # Compute effect size for resample
        boot_mean = np.mean(resampled)
        boot_std = np.std(resampled, ddof=1) if n_conn > 1 else 0.0
        
        if boot_std > 0:
            boot_effect = boot_mean / boot_std
            bootstrap_effects.append(boot_effect)
            bootstrap_means.append(boot_mean)
    
    # Confidence intervals
    alpha = 1 - confidence_level
    ci_lower_percentile = (alpha / 2) * 100
    ci_upper_percentile = (1 - alpha / 2) * 100
    
    effect_ci_lower = np.percentile(bootstrap_effects, ci_lower_percentile)
    effect_ci_upper = np.percentile(bootstrap_effects, ci_upper_percentile)
    
    mean_ci_lower = np.percentile(bootstrap_means, ci_lower_percentile)
    mean_ci_upper = np.percentile(bootstrap_means, ci_upper_percentile)
    
    # Two-sided p-value: proportion of bootstrap samples with opposite sign
    p_value = np.sum(np.sign(bootstrap_means) != np.sign(mean_diff)) / len(bootstrap_means)
    
    return {
        'effect_size': effect_size,
        'effect_size_ci_lower': effect_ci_lower,
        'effect_size_ci_upper': effect_ci_upper,
        'mean_diff': mean_diff,
        'mean_diff_ci_lower': mean_ci_lower,
        'mean_diff_ci_upper': mean_ci_upper,
        'std_diff': std_diff,
        'p_value': p_value,
        'n_connectivity': n_conn,
        'n_bootstrap': n_bootstrap,
        'bootstrap_distribution': np.array(bootstrap_effects),
        'mean_distribution': np.array(bootstrap_means)
    }


# ============================================================================
# Multi-Source Analysis
# ============================================================================

def analyze_effect_size_all_sources_nested(
    nested_results: List,
    circuit,
    target_population: str,
    post_population: str,
    source_populations: List[str],
    stim_start: float,
    stim_duration: float,
    warmup: float,
    n_bootstrap: int = 10000,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Perform bootstrap effect size analysis for all source populations
    
    Args:
        nested_results: List of NestedTrialResult objects
        circuit: DentateCircuit instance (connectivity from first trial)
        target_population: Stimulated population
        post_population: Post-synaptic population to analyze
        source_populations: List of source populations to analyze
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period start (ms)
        n_bootstrap: Number of bootstrap samples
        threshold_std: Classification threshold (std deviations)
        expression_threshold: Opsin expression threshold
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: {
            source_population: {
                'bootstrap_results': {...},
                'weights_by_connectivity': {...},
                'weight_differences': array,
                'classifications': {...}
            },
            ...
        }
    """
    # Organize trials by connectivity
    trials_by_connectivity = organize_nested_trials(nested_results)
    
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
    
    # Analyze each source population
    all_results = {}
    
    for source_pop in source_populations:
        # Extract weights
        weights_by_conn = extract_weights_by_connectivity(
            circuit,
            classifications,
            post_population,
            source_pop
        )
        
        if not weights_by_conn:
            continue
        
        # Compute connectivity-level differences
        weight_diffs_all = compute_connectivity_weight_differences(
            weights_by_conn,
            metric='mean'
        )
        weight_diffs = weights_diffs_all['excited_vs_suppressed']
        
        if len(weight_diffs) == 0:
            continue
        
        # Bootstrap analysis
        bootstrap_results = bootstrap_effect_size_nested(
            weight_diffs,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed
        )
        
        all_results[source_pop] = {
            'bootstrap_results': bootstrap_results,
            'weights_by_connectivity': weights_by_conn,
            'weight_differences': weight_diffs,
            'classifications': classifications
        }
    
    return all_results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_effect_sizes_forest(
    analysis_results: Dict,
    target_population: str,
    post_population: str,
    alpha: float = 0.001,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create forest plot of effect sizes with bootstrap CIs
    
    Args:
        analysis_results: Output from analyze_effect_size_all_sources_nested()
        target_population: Stimulated population
        post_population: Post-synaptic population
        alpha: Significance level
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    source_pops = sorted(analysis_results.keys())
    n_sources = len(source_pops)
    
    fig, ax = plt.subplots(figsize=(10, max(6, n_sources * 0.8)))
    
    y_positions = np.arange(n_sources)
    
    for i, source_pop in enumerate(source_pops):
        results = analysis_results[source_pop]['bootstrap_results']
        
        effect_size = results['effect_size']
        ci_lower = results['effect_size_ci_lower']
        ci_upper = results['effect_size_ci_upper']
        p_value = results['p_value']
        n_conn = results['n_connectivity']
        
        # Color by significance
        color = 'red' if p_value < alpha else 'gray'
        
        # Plot point estimate
        ax.plot(effect_size, y_positions[i], 'o', 
                color=color, markersize=10, zorder=3)
        
        # Plot CI
        ax.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]],
                '-', color=color, linewidth=2, zorder=2)
        
        # Add significance stars
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        
        # Add text annotation
        ax.text(ci_upper + 0.15, y_positions[i],
                f'd={effect_size:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]\n'
                f'p={p_value:.3f} {stars} (N={n_conn})',
                va='center', fontsize=9)
    
    # Reference line at zero
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
    
    # Add effect size interpretation regions
    ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Small effect (|d|<0.2)')
    ax.axvspan(0.5, ax.get_xlim()[1], alpha=0.1, color='orange', label='Medium effect (|d|>0.5)')
    ax.axvspan(ax.get_xlim()[0], -0.5, alpha=0.1, color='orange')
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s.upper() for s in source_pops], fontsize=11)
    ax.set_xlabel("Effect Size (Cohen's d)\nExcited - Suppressed", fontsize=12, fontweight='bold')
    ax.set_ylabel("Source Population", fontsize=12, fontweight='bold')
    ax.set_title(f"{target_population.upper()} -> {post_population.upper()}\n"
                 f"Weight Difference: Excited vs Suppressed Cells\n"
                 f"(Bootstrap 95% CI)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved forest plot to: {save_path}")
    
    return fig


def plot_bootstrap_distributions(
    analysis_results: Dict,
    source_population: str,
    target_population: str,
    post_population: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bootstrap distribution for a single source population
    
    Args:
        analysis_results: Output from analyze_effect_size_all_sources_nested()
        source_population: Source population to plot
        target_population: Stimulated population (for title)
        post_population: Post-synaptic population (for title)
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    results = analysis_results[source_population]['bootstrap_results']
    weight_diffs = analysis_results[source_population]['weight_differences']
    
    # Check if bootstrap distribution exists (may be missing if insufficient data)
    if 'bootstrap_distribution' not in results or results['bootstrap_distribution'] is None:
        print(f"Warning: No bootstrap distribution available for {source_population}")
        # Create simple figure showing the issue
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.text(0.5, 0.5, f'Insufficient data for bootstrap analysis\n'
                          f'{source_population.upper()} -> {post_population.upper()}\n'
                          f'N connectivity instances: {results.get("n_connectivity", 0)}',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Connectivity-level weight differences
    ax1 = axes[0]
    
    x_pos = np.arange(len(weight_diffs))
    colors = ['red' if w > 0 else 'blue' for w in weight_diffs]
    ax1.bar(x_pos, weight_diffs, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Connectivity Instance', fontsize=12)
    ax1.set_ylabel('Weight Difference (nS)', fontsize=12)
    ax1.set_title(f'Per-Connectivity Weight Differences\n'
                  f'{source_population.upper()} -> {post_population.upper()}\n'
                  f'Mean = {results["mean_diff"]:.3f} +/- {results["std_diff"]:.3f} nS',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'C{i}' for i in range(len(weight_diffs))])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Bootstrap distribution
    ax2 = axes[1]
    
    bootstrap_dist = results['bootstrap_distribution']
    
    ax2.hist(bootstrap_dist, bins=50, color='steelblue', 
             alpha=0.7, edgecolor='black')
    
    # Mark original effect size
    ax2.axvline(results['effect_size'], color='red', 
                linestyle='--', linewidth=2, label='Observed', zorder=3)
    
    # Mark CI bounds
    ax2.axvline(results['effect_size_ci_lower'], color='orange',
                linestyle=':', linewidth=2, zorder=3)
    ax2.axvline(results['effect_size_ci_upper'], color='orange',
                linestyle=':', linewidth=2, label='95% CI', zorder=3)
    
    # Mark zero
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=2)
    
    ax2.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Bootstrap Distribution (N={results["n_bootstrap"]:,})\n'
                  f'd = {results["effect_size"]:.2f} '
                  f'[{results["effect_size_ci_lower"]:.2f}, '
                  f'{results["effect_size_ci_upper"]:.2f}], '
                  f'p={results["p_value"]:.4f}',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{target_population.upper()} Stimulation Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bootstrap distribution plot to: {save_path}")
    
    return fig

def plot_weight_distributions_by_response(
    analysis_results: Dict,
    source_population: str,
    connectivity_idx: int,
    target_population: str,
    post_population: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot weight distributions for excited vs suppressed cells (single connectivity)
    
    Args:
        analysis_results: Output from analyze_effect_size_all_sources_nested()
        source_population: Source population to plot
        connectivity_idx: Which connectivity instance to plot
        target_population: Stimulated population (for title)
        post_population: Post-synaptic population (for title)
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
def plot_weight_distributions_by_response(
    analysis_results: Dict,
    source_population: str,
    connectivity_idx: int,
    target_population: str,
    post_population: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot weight distributions for excited vs suppressed cells (single connectivity)
    
    Args:
        analysis_results: Output from analyze_all_sources_nested()
        source_population: Source population to plot
        connectivity_idx: Which connectivity instance to plot
        target_population: Stimulated population (for title)
        post_population: Post-synaptic population (for title)
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    weights_by_conn = analysis_results[source_population]['weights_by_connectivity']
    
    if connectivity_idx not in weights_by_conn:
        raise ValueError(f"Connectivity {connectivity_idx} not found in results")
    
    weights_data = weights_by_conn[connectivity_idx]
    weights_excited = weights_data['weights_excited']
    weights_suppressed = weights_data['weights_suppressed']
    
    # Check for empty arrays
    if len(weights_excited) == 0 and len(weights_suppressed) == 0:
        raise ValueError(f"No weights found for connectivity {connectivity_idx}")
    
    if len(weights_excited) == 0 or len(weights_suppressed) == 0:
        # Handle case where one group is empty
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.text(0.5, 0.5, 
                f'Insufficient data: {"Excited" if len(weights_excited)==0 else "Suppressed"} '
                f'cells have no synaptic weights\n'
                f'Connectivity {connectivity_idx}\n'
                f'{source_population.upper()} → {post_population.upper()}',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Histograms
    ax1 = axes[0]
    
    max_weight = max(np.max(weights_excited) if len(weights_excited) > 0 else 0, 
                     np.max(weights_suppressed) if len(weights_suppressed) > 0 else 0)
    
    if max_weight > 0:
        bins = np.linspace(0, max_weight, 30)
    else:
        bins = 30
    
    ax1.hist(weights_excited, bins=bins, alpha=0.6, color='red', 
             label=f'Excited (n={len(weights_excited)})', edgecolor='black')
    ax1.hist(weights_suppressed, bins=bins, alpha=0.6, color='blue',
             label=f'Suppressed (n={len(weights_suppressed)})', edgecolor='black')
    
    ax1.axvline(np.mean(weights_excited), color='red', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(weights_suppressed), color='blue', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Synaptic Weight (nS)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Weight Distributions\nConnectivity {connectivity_idx}', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Violin plots
    ax2 = axes[1]
    
    parts = ax2.violinplot([weights_excited, weights_suppressed],
                           positions=[0, 1],
                           showmeans=True,
                           showextrema=True)
    
    for pc, color in zip(parts['bodies'], ['red', 'blue']):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Excited', 'Suppressed'], fontsize=11)
    ax2.set_ylabel('Synaptic Weight (nS)', fontsize=12)
    ax2.set_title(f'Weight Comparison\n'
                  f'Δ = {np.mean(weights_excited) - np.mean(weights_suppressed):.3f} nS',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{target_population.upper()} → {post_population.upper()}\n'
                 f'{source_population.upper()} inputs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved weight distribution plot to: {save_path}")
    
    return fig    

# ============================================================================
# Summary Statistics
# ============================================================================

def print_nested_effect_size_analysis_summary(
    analysis_results: Dict,
    target_population: str,
    post_population: str,
    alpha: float = 0.05
):
    """
    Print summary of bootstrap effect size analysis
    
    Args:
        analysis_results: Output from analyze_effect_size_all_sources_nested()
        target_population: Stimulated population
        post_population: Post-synaptic population
        alpha: Significance level
    """
    print("\n" + "="*80)
    print(f"Bootstrap Effect Size Analysis: {target_population.upper()} -> {post_population.upper()}")
    print("="*80)
    
    for source_pop in sorted(analysis_results.keys()):
        results = analysis_results[source_pop]['bootstrap_results']
        
        print(f"\n{source_pop.upper()} inputs:")
        print(f"  Effect size (Cohen's d): {results['effect_size']:.3f} "
              f"[{results['effect_size_ci_lower']:.3f}, {results['effect_size_ci_upper']:.3f}]")
        print(f"  Mean weight difference: {results['mean_diff']:.4f} nS "
              f"[{results['mean_diff_ci_lower']:.4f}, {results['mean_diff_ci_upper']:.4f}]")
        print(f"  p-value: {results['p_value']:.4f}", end="")
        
        if results['p_value'] < 0.001:
            print(" ***")
        elif results['p_value'] < 0.01:
            print(" **")
        elif results['p_value'] < alpha:
            print(" *")
        else:
            print(" (n.s.)")
        
        print(f"  N connectivity instances: {results['n_connectivity']}")
        print(f"  Bootstrap samples: {results['n_bootstrap']:,}")
        
        # Interpretation
        if abs(results['effect_size']) > 0.8:
            effect_label = "LARGE"
        elif abs(results['effect_size']) > 0.5:
            effect_label = "MEDIUM"
        elif abs(results['effect_size']) > 0.2:
            effect_label = "SMALL"
        else:
            effect_label = "NEGLIGIBLE"
        
        print(f"  Interpretation: {effect_label} effect size")
        
        if results['p_value'] < alpha:
            direction = "receive MORE" if results['mean_diff'] > 0 else "receive LESS"
            print(f"  -> Excited cells {direction} input from {source_pop.upper()} (p < {alpha})")
    
    print("\n" + "="*80)
