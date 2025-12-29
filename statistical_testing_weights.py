#!/usr/bin/env python3
"""
Statistical testing framework for weights and cell response.
"""

import sys
import torch
import numpy as np
import scipy.stats as stats
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import pickle
import json
import pprint
from pathlib import Path
import tqdm

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# Synaptic weights comparison framework
# ============================================================================

def classify_cells_by_average_response(
    activity_trace_mean: Dict[str, torch.Tensor],
    population: str,
    baseline_mask: torch.Tensor,
    stim_mask: torch.Tensor,
    threshold_std: float = 1.0
) -> Dict:
    """
    Classify cells based on trial-averaged activity
    
    Args:
        activity_trace_mean: Dict of trial-averaged activity traces
        population: Population to classify ('gc', 'mc', 'pv', 'sst')
        baseline_mask: Boolean mask for baseline period
        stim_mask: Boolean mask for stimulation period
        threshold_std: Standard deviation threshold for classification
        
    Returns:
        Dict with classification results:
        {
            'excited': array of cell indices,
            'suppressed': array of cell indices,
            'unchanged': array of cell indices,
            'baseline_rates': array of baseline rates,
            'stim_rates': array of stim rates,
            'rate_changes': array of rate changes,
            'n_excited': int,
            'n_suppressed': int,
            'n_unchanged': int
        }
    """
    activity = activity_trace_mean[population]
    
    # Compute average rates
    baseline_rate = torch.mean(activity[:, baseline_mask], dim=1)
    stim_rate = torch.mean(activity[:, stim_mask], dim=1)
    rate_change = stim_rate - baseline_rate
    
    # Classification threshold
    baseline_std = torch.std(baseline_rate)
    threshold = threshold_std * baseline_std
    
    # Classify cells
    excited_mask = rate_change > threshold
    suppressed_mask = rate_change < -threshold
    unchanged_mask = ~(excited_mask | suppressed_mask)
    
    # Convert to numpy
    if hasattr(excited_mask, 'cpu'):
        excited = np.where(excited_mask.cpu().numpy())[0]
        suppressed = np.where(suppressed_mask.cpu().numpy())[0]
        unchanged = np.where(unchanged_mask.cpu().numpy())[0]
        baseline_rates = baseline_rate.cpu().numpy()
        stim_rates = stim_rate.cpu().numpy()
        rate_changes = rate_change.cpu().numpy()
    else:
        excited = np.where(excited_mask)[0]
        suppressed = np.where(suppressed_mask)[0]
        unchanged = np.where(unchanged_mask)[0]
        baseline_rates = baseline_rate
        stim_rates = stim_rate
        rate_changes = rate_change
    
    return {
        'excited': excited,
        'suppressed': suppressed,
        'unchanged': unchanged,
        'baseline_rates': baseline_rates,
        'stim_rates': stim_rates,
        'rate_changes': rate_changes,
        'n_excited': len(excited),
        'n_suppressed': len(suppressed),
        'n_unchanged': len(unchanged)
    }


def extract_weights_by_response(
    circuit,
    post_population: str,
    classification: Dict
) -> Dict:
    """
    Extract synaptic weights for excited vs suppressed cells
    
    Args:
        circuit: DentateCircuit instance
        post_population: Target population ('gc', 'mc', 'pv', 'sst')
        classification: Output from classify_cells_by_average_response
        
    Returns:
        Dict mapping response types to source-specific weights:
        {
            'excited': {
                'mec': array of total input weights,
                'gc': array of total input weights,
                ...
            },
            'suppressed': {...},
            'unchanged': {...}
        }
    """
    conductance_matrices = circuit.connectivity.conductance_matrices
    
    weights_by_response = {
        'excited': {},
        'suppressed': {},
        'unchanged': {}
    }
    
    # Possible source populations
    all_sources = ['mec', 'gc', 'mc', 'pv', 'sst']
    
    for source in all_sources:
        conn_name = f'{source}_{post_population}'
        
        if conn_name not in conductance_matrices:
            continue
        
        cond_matrix = conductance_matrices[conn_name]
        conductances = cond_matrix.conductances
        
        # Sum over presynaptic cells to get total input weight per postsynaptic cell
        if hasattr(conductances, 'cpu'):
            total_input = torch.sum(conductances, dim=0).cpu().numpy()
        else:
            total_input = np.sum(conductances, axis=0)
        
        # Extract for each response type
        for response_type in ['excited', 'suppressed', 'unchanged']:
            cell_indices = classification[response_type]
            
            if len(cell_indices) > 0:
                weights_by_response[response_type][source] = total_input[cell_indices]
            else:
                weights_by_response[response_type][source] = np.array([])
    
    return weights_by_response


def bootstrap_confidence_interval(
    data1: np.ndarray,
    data2: np.ndarray,
    statistic_func,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for difference between groups
    
    Args:
        data1: First group data
        data2: Second group data
        statistic_func: Function to compute statistic (e.g., mean difference)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (observed_statistic, ci_lower, ci_upper)
    """
    # Observed statistic
    observed = statistic_func(data1, data2)
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        
        bootstrap_stats[i] = statistic_func(sample1, sample2)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return observed, ci_lower, ci_upper


def compute_statistical_tests_with_bootstrap(
    weights_by_response: Dict,
    n_permutations: int = 10000,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    alpha: float = 0.05
) -> Dict:
    """
    Compute comprehensive statistical tests with bootstrap confidence intervals
    
    Args:
        weights_by_response: Output from extract_weights_by_response
        n_permutations: Number of permutations for permutation test
        n_bootstrap: Number of bootstrap samples for CI
        confidence_level: Confidence level for bootstrap CI
        alpha: Significance level
        
    Returns:
        Dict with statistical results for each source:
        {
            'mec': {
                'mann_whitney_u': float,
                'mann_whitney_p': float,
                'cohens_d': float,
                'cohens_d_ci_lower': float,
                'cohens_d_ci_upper': float,
                'mean_diff': float,
                'mean_diff_ci_lower': float,
                'mean_diff_ci_upper': float,
                'permutation_p': float,
                'mean_excited': float,
                'mean_suppressed': float,
                'std_excited': float,
                'std_suppressed': float,
                'significant': bool,
                'bonferroni_significant': bool
            },
            ...
        }
    """
    from scipy import stats
    
    statistics = {}
    
    # Get all sources present in the data
    sources = set()
    for response_type in weights_by_response.values():
        sources.update(response_type.keys())
    
    for source in sources:
        excited_weights = weights_by_response['excited'].get(source, np.array([]))
        suppressed_weights = weights_by_response['suppressed'].get(source, np.array([]))
        
        # Skip if either group is too small
        if len(excited_weights) < 3 or len(suppressed_weights) < 3:
            statistics[source] = {
                'mann_whitney_u': np.nan,
                'mann_whitney_p': np.nan,
                'cohens_d': np.nan,
                'cohens_d_ci_lower': np.nan,
                'cohens_d_ci_upper': np.nan,
                'mean_diff': np.nan,
                'mean_diff_ci_lower': np.nan,
                'mean_diff_ci_upper': np.nan,
                'permutation_p': np.nan,
                'mean_excited': np.mean(excited_weights) if len(excited_weights) > 0 else np.nan,
                'mean_suppressed': np.mean(suppressed_weights) if len(suppressed_weights) > 0 else np.nan,
                'std_excited': np.std(excited_weights) if len(excited_weights) > 0 else np.nan,
                'std_suppressed': np.std(suppressed_weights) if len(suppressed_weights) > 0 else np.nan,
                'significant': False,
                'bonferroni_significant': False,
                'n_excited': len(excited_weights),
                'n_suppressed': len(suppressed_weights)
            }
            continue
        
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(excited_weights, suppressed_weights, 
                                             alternative='two-sided')
        
        # Basic statistics
        mean_excited = np.mean(excited_weights)
        mean_suppressed = np.mean(suppressed_weights)
        std_excited = np.std(excited_weights)
        std_suppressed = np.std(suppressed_weights)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.var(excited_weights) + np.var(suppressed_weights)) / 2)
        
        if pooled_std > 0:
            cohens_d = (mean_excited - mean_suppressed) / pooled_std
        else:
            cohens_d = 0.0
        
        # Bootstrap CI for Cohen's d
        def cohens_d_func(data1, data2):
            pooled = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            if pooled > 0:
                return (np.mean(data1) - np.mean(data2)) / pooled
            else:
                return 0.0
        
        _, cohens_d_ci_lower, cohens_d_ci_upper = bootstrap_confidence_interval(
            excited_weights, suppressed_weights,
            cohens_d_func,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
        
        # Bootstrap CI for mean difference
        def mean_diff_func(data1, data2):
            return np.mean(data1) - np.mean(data2)
        
        mean_diff, mean_diff_ci_lower, mean_diff_ci_upper = bootstrap_confidence_interval(
            excited_weights, suppressed_weights,
            mean_diff_func,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
        
        # Permutation test
        combined = np.concatenate([excited_weights, suppressed_weights])
        n_excited = len(excited_weights)
        
        perm_diffs = np.zeros(n_permutations)
        for i in range(n_permutations):
            np.random.shuffle(combined)
            perm_excited = combined[:n_excited]
            perm_suppressed = combined[n_excited:]
            perm_diffs[i] = np.mean(perm_excited) - np.mean(perm_suppressed)
        
        permutation_p = np.mean(np.abs(perm_diffs) >= np.abs(mean_diff))
        
        statistics[source] = {
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_value,
            'cohens_d': cohens_d,
            'cohens_d_ci_lower': cohens_d_ci_lower,
            'cohens_d_ci_upper': cohens_d_ci_upper,
            'mean_diff': mean_diff,
            'mean_diff_ci_lower': mean_diff_ci_lower,
            'mean_diff_ci_upper': mean_diff_ci_upper,
            'permutation_p': permutation_p,
            'mean_excited': mean_excited,
            'mean_suppressed': mean_suppressed,
            'std_excited': std_excited,
            'std_suppressed': std_suppressed,
            'significant': p_value < alpha,
            'n_excited': len(excited_weights),
            'n_suppressed': len(suppressed_weights)
        }
    
    # Apply Bonferroni correction
    valid_p_values = [s['mann_whitney_p'] for s in statistics.values() 
                      if not np.isnan(s['mann_whitney_p'])]
    if len(valid_p_values) > 0:
        bonferroni_alpha = alpha / len(valid_p_values)
        for source in statistics:
            if not np.isnan(statistics[source]['mann_whitney_p']):
                statistics[source]['bonferroni_significant'] = \
                    statistics[source]['mann_whitney_p'] < bonferroni_alpha
            else:
                statistics[source]['bonferroni_significant'] = False
    
    return statistics


def analyze_weights_by_average_response(
    comparative_results: Dict,
    circuit,
    target_population: str,
    intensity: float,
    baseline_start: float = 500.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    post_populations: Optional[List[str]] = None,
    threshold_std: float = 1.0,
    n_permutations: int = 10000,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Main analysis function using trial-averaged responses
    
    Args:
        comparative_results: Output from run_comparative_experiment
        circuit: DentateCircuit instance
        target_population: Stimulated population ('pv' or 'sst')
        intensity: Light intensity to analyze
        baseline_start: Start of baseline period (ms)
        stim_start: Start of stimulation (ms)
        stim_duration: Duration of stimulation (ms)
        post_populations: Populations to analyze (default: ['gc', 'mc'])
        threshold_std: Std threshold for classification
        n_permutations: Number of permutations for statistical tests
        n_bootstrap: Number of bootstrap samples for CIs
        confidence_level: Confidence level for bootstrap CIs
        
    Returns:
        Dict with complete analysis results:
        {
            'gc': {
                'classification': {...},
                'weights': {...},
                'statistics': {...}
            },
            'mc': {...},
            ...
        }
    """
    if post_populations is None:
        post_populations = ['gc', 'mc']
    
    # Extract trial-averaged data
    experiment_data = comparative_results[target_population][intensity]
    activity_trace_mean = experiment_data['activity_trace_mean']
    time = experiment_data['time']
    n_trials = experiment_data.get('n_trials', 1)
    
    # Create time masks
    baseline_mask = (time >= baseline_start) & (time < stim_start)
    stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
    
    # Analyze each post-synaptic population
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Weight Analysis by Average Response ({n_trials} trials averaged)")
    print(f"Target: {target_population.upper()} stimulation at intensity {intensity}")
    print('='*70)
    
    for post_pop in post_populations:
        print(f"\n{post_pop.upper()} Analysis:")
        print('-'*70)
        
        # Classify cells based on trial-averaged activity
        classification = classify_cells_by_average_response(
            activity_trace_mean=activity_trace_mean,
            population=post_pop,
            baseline_mask=baseline_mask,
            stim_mask=stim_mask,
            threshold_std=threshold_std
        )
        
        print(f"  Excited cells: {classification['n_excited']}")
        print(f"  Suppressed cells: {classification['n_suppressed']}")
        print(f"  Unchanged cells: {classification['n_unchanged']}")
        
        # Extract weights
        weights = extract_weights_by_response(
            circuit=circuit,
            post_population=post_pop,
            classification=classification
        )
        
        # Compute statistics with bootstrap CIs
        statistics = compute_statistical_tests_with_bootstrap(
            weights_by_response=weights,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
        
        # Print results
        print(f"\n  Statistical Comparisons (Excited vs Suppressed):")
        print(f"  {'Source':<8} {'Cohen d':<20} {'Mean Diff (nS)':<25} {'p-value':<12} {'Sig':<5}")
        print(f"  {'-'*8} {'-'*20} {'-'*25} {'-'*12} {'-'*5}")
        
        for source in sorted(statistics.keys()):
            stats = statistics[source]
            
            if not np.isnan(stats['mann_whitney_p']):
                # Format Cohen's d with CI
                d_str = f"{stats['cohens_d']:>6.3f} [{stats['cohens_d_ci_lower']:>6.3f}, {stats['cohens_d_ci_upper']:>6.3f}]"
                
                # Format mean difference with CI
                diff_str = f"{stats['mean_diff']:>6.3f} [{stats['mean_diff_ci_lower']:>6.3f}, {stats['mean_diff_ci_upper']:>6.3f}]"
                
                # Significance markers
                if stats['bonferroni_significant']:
                    sig = '***'
                elif stats['significant']:
                    sig = '*'
                else:
                    sig = 'n.s.'
                
                print(f"  {source.upper():<8} {d_str:<20} {diff_str:<25} {stats['mann_whitney_p']:<12.4f} {sig:<5}")
        
        results[post_pop] = {
            'classification': classification,
            'weights': weights,
            'statistics': statistics
        }
    
    return results


def plot_weights_by_average_response(
    analysis_results: Dict,
    target_population: str,
    post_populations: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create visualization of weight analysis with bootstrap CIs
    
    Args:
        analysis_results: Output from analyze_weights_by_average_response
        target_population: Stimulated population for plot title
        post_populations: Populations to plot (default: ['gc', 'mc'])
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if post_populations is None:
        post_populations = ['gc', 'mc']
    
    n_pops = len(post_populations)
    fig = plt.figure(figsize=(18, 6 * n_pops))
    
    # Color scheme
    colors = {
        'excited': '#e74c3c',  # Red
        'suppressed': '#3498db',  # Blue
        'unchanged': '#95a5a6'  # Gray
    }
    
    for pop_idx, post_pop in enumerate(post_populations):
        pop_results = analysis_results[post_pop]
        weights = pop_results['weights']
        statistics = pop_results['statistics']
        classification = pop_results['classification']
        
        # Get all sources
        sources = sorted(set().union(*[w.keys() for w in weights.values()]))
        n_sources = len(sources)
        
        # Create subplot grid
        gs = fig.add_gridspec(3, n_sources,
                             left=0.05, right=0.95,
                             top=0.95 - pop_idx * (1.0 / n_pops),
                             bottom=0.95 - (pop_idx + 1) * (1.0 / n_pops) + 0.05,
                             hspace=0.4, wspace=0.3)
        
        # Panel A: Violin plots
        for source_idx, source in enumerate(sources):
            ax = fig.add_subplot(gs[0, source_idx])
            
            data_to_plot = []
            labels = []
            plot_colors = []
            
            for response_type in ['excited', 'unchanged', 'suppressed']:
                if source in weights[response_type] and len(weights[response_type][source]) > 0:
                    data_to_plot.append(weights[response_type][source])
                    labels.append(response_type.replace('_', ' ').title())
                    plot_colors.append(colors[response_type])
            
            if len(data_to_plot) > 0:
                parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                                     showmeans=True, showmedians=True)
                
                for pc, color in zip(parts['bodies'], plot_colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Total Input Weight (nS)', fontsize=10)
                
                # Statistical annotation with bootstrap CI
                if source in statistics and not np.isnan(statistics[source]['mann_whitney_p']):
                    stats = statistics[source]
                    d_val = stats['cohens_d']
                    d_ci = f"[{stats['cohens_d_ci_lower']:.2f}, {stats['cohens_d_ci_upper']:.2f}]"
                    bonf_sig = stats.get('bonferroni_significant', False)
                    
                    sig_text = '***' if bonf_sig else ('*' if stats['significant'] else 'n.s.')
                    ax.text(0.5, 0.98, f"d={d_val:.2f}\n{d_ci}\n{sig_text}",
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'{source.upper()} → {post_pop.upper()}',
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        # Panel B: Mean comparison with error bars
        for source_idx, source in enumerate(sources):
            ax = fig.add_subplot(gs[1, source_idx])
            
            if source in statistics and not np.isnan(statistics[source]['mean_diff']):
                stats = statistics[source]
                
                # Plot means with error bars (bootstrap CIs)
                x_pos = [0, 1]
                means = [stats['mean_excited'], stats['mean_suppressed']]
                
                # Approximate error bars from bootstrap CIs
                # (For visualization; actual CIs are for the difference)
                ax.bar(x_pos, means, color=[colors['excited'], colors['suppressed']],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
                
                # Add value labels
                for x, mean in zip(x_pos, means):
                    ax.text(x, mean, f'{mean:.2f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                # Show difference with CI
                diff_text = (f"Δ = {stats['mean_diff']:.2f} nS\n"
                           f"95% CI: [{stats['mean_diff_ci_lower']:.2f}, {stats['mean_diff_ci_upper']:.2f}]")
                ax.text(0.5, 0.98, diff_text,
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(['Excited', 'Suppressed'], fontsize=10)
                ax.set_ylabel('Mean Weight (nS)', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
        
        # Panel C: Effect size summary
        ax = fig.add_subplot(gs[2, :])
        
        effect_sizes = []
        effect_ci_lower = []
        effect_ci_upper = []
        source_labels = []
        bonf_sig = []
        
        for source in sources:
            if source in statistics and not np.isnan(statistics[source]['cohens_d']):
                effect_sizes.append(statistics[source]['cohens_d'])
                effect_ci_lower.append(statistics[source]['cohens_d_ci_lower'])
                effect_ci_upper.append(statistics[source]['cohens_d_ci_upper'])
                source_labels.append(f"{source.upper()}→{post_pop.upper()}")
                bonf_sig.append(statistics[source].get('bonferroni_significant', False))
        
        if len(effect_sizes) > 0:
            x_pos = np.arange(len(source_labels))
            bar_colors = ['#e74c3c' if sig else '#95a5a6' for sig in bonf_sig]
            
            # Compute error bars from bootstrap CIs
            yerr_lower = np.array(effect_sizes) - np.array(effect_ci_lower)
            yerr_upper = np.array(effect_ci_upper) - np.array(effect_sizes)
            yerr = np.array([yerr_lower, yerr_upper])
            
            bars = ax.bar(x_pos, effect_sizes, yerr=yerr, capsize=5,
                         color=bar_colors, alpha=0.7,
                         edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar, d_val in zip(bars, effect_sizes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{d_val:.2f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=10, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(source_labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel("Cohen's d (Effect Size)", fontsize=11)
            ax.set_title(f"Effect Size: Excited vs Suppressed {post_pop.upper()} Cells\n"
                        f"(Error bars: 95% Bootstrap CI)",
                        fontsize=12, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', alpha=0.7, label='Significant (Bonferroni)'),
                Patch(facecolor='#95a5a6', alpha=0.7, label='Not Significant')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add population label
        fig.text(0.02, 0.95 - pop_idx * (1.0 / n_pops) - 0.02,
                f'{post_pop.upper()} Response to {target_population.upper()} Stimulation',
                fontsize=13, fontweight='bold', va='top')
        
        # Add sample sizes
        n_excited = classification['n_excited']
        n_suppressed = classification['n_suppressed']
        n_unchanged = classification['n_unchanged']
        
        fig.text(0.02, 0.95 - pop_idx * (1.0 / n_pops) - 0.05,
                f'n_excited={n_excited}, n_suppressed={n_suppressed}, n_unchanged={n_unchanged}',
                fontsize=10, va='top', style='italic')
    
    fig.suptitle(f'Synaptic Weight Analysis by Average Response\n'
                f'{target_population.upper()} Stimulation (Trial-Averaged Classification)',
                fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved weight analysis plot to: {save_path}")
    
    return fig        

