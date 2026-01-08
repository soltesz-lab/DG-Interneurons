"""
Effect Size Decision Framework for Nested Experiment Data

Integrates bootstrap effect size analysis with power analysis, precision assessment,
and biological significance to make data collection recommendations.

This framework determines whether additional connectivity instances are needed
for conclusive mechanistic insights about paradoxical excitation.
"""

from typing import Dict, Tuple, Optional, NamedTuple, List
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ============================================================================
# Data Structures
# ============================================================================

class PowerAnalysisResult(NamedTuple):
    """Results from statistical power analysis"""
    observed_effect_size: float
    current_n: int
    required_n_80: int  # For 80% power
    required_n_90: int  # For 90% power
    current_power: float
    additional_n_needed_80: int
    additional_n_needed_90: int


class PrecisionAssessment(NamedTuple):
    """Assessment of CI precision"""
    ci_width: float
    ci_width_relative: float  # Relative to effect size
    is_precise: bool
    needs_narrower_ci: bool
    recommended_n: int


class BiologicalSignificanceAssessment(NamedTuple):
    """Assessment of biological significance"""
    effect_size: float
    mean_diff_nS: float
    is_biologically_meaningful: bool
    strength_category: str  # 'negligible', 'small', 'medium', 'large', 'very_large'
    interpretation: str


class DataCollectionRecommendation(NamedTuple):
    """Recommendation for data collection"""
    action: str  # 'STOP', 'CONTINUE', 'CONSIDER'
    priority: str  # 'HIGH', 'MEDIUM', 'LOW', 'NONE'
    recommended_total_n: int
    additional_n_needed: int
    justification: str
    details: Dict


# ============================================================================
# Power Analysis Functions
# ============================================================================

def compute_power_one_sample(
    effect_size: float,
    n: int,
    alpha: float = 0.001
) -> float:
    """
    Compute statistical power for one-sample t-test
    
    Tests H0: effect_size = 0 vs H1: effect_size ≠ 0
    
    Args:
        effect_size: Cohen's d
        n: Sample size
        alpha: Significance level (two-sided)
        
    Returns:
        Statistical power (0-1)
    """
    if n <= 2 or effect_size == 0:
        return alpha  # No power for very small n or zero effect
    
    # Non-centrality parameter for t-distribution
    ncp = effect_size * np.sqrt(n)
    
    # Critical value for two-sided test
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Power = P(reject H0 | H1 is true)
    # = P(|t| > t_crit | ncp ≠ 0)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    
    return power


def estimate_required_n(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.001
) -> int:
    """
    Estimate required sample size for desired power
    
    Uses iterative search for one-sample t-test
    
    Args:
        effect_size: Cohen's d
        power: Desired power (default 0.80)
        alpha: Significance level (default 0.001)
        
    Returns:
        Required sample size
    """
    if abs(effect_size) < 0.01:
        return 9999  # Essentially zero effect
    
    # Use analytical approximation as starting point
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    n_approx = ((z_alpha + z_beta) / effect_size) ** 2
    
    # Refine with exact t-distribution
    n = max(3, int(np.ceil(n_approx)))
    
    # Iterative search
    while compute_power_one_sample(effect_size, n, alpha) < power:
        n += 1
        if n > 1000:  # Safety limit
            break
    
    return n


def analyze_statistical_power(
    bootstrap_results: Dict,
    current_n: int = 5,
    alpha: float = 0.001
) -> Dict[str, PowerAnalysisResult]:
    """
    Perform power analysis for all source populations
    
    Args:
        bootstrap_results: Results from analyze_effect_size_all_sources_nested()
        current_n: Current number of connectivity instances
        alpha: Significance level
        
    Returns:
        Dict mapping source_pop to PowerAnalysisResult
    """
    power_results = {}
    
    for source_pop, results in bootstrap_results.items():
        boot_results = results['bootstrap_results']
        effect_size = boot_results['effect_size']
        
        # Current power
        current_power = compute_power_one_sample(effect_size, current_n, alpha)
        
        # Required N for different power levels
        required_n_80 = estimate_required_n(effect_size, power=0.80, alpha=alpha)
        required_n_90 = estimate_required_n(effect_size, power=0.90, alpha=alpha)
        
        # Additional N needed
        add_n_80 = max(0, required_n_80 - current_n)
        add_n_90 = max(0, required_n_90 - current_n)
        
        power_results[source_pop] = PowerAnalysisResult(
            observed_effect_size=effect_size,
            current_n=current_n,
            required_n_80=required_n_80,
            required_n_90=required_n_90,
            current_power=current_power,
            additional_n_needed_80=add_n_80,
            additional_n_needed_90=add_n_90
        )
    
    return power_results


# ============================================================================
# Precision Analysis Functions
# ============================================================================

def assess_ci_precision(
    bootstrap_results: Dict,
    max_absolute_width: float = 1.0,
    max_relative_width: float = 0.5,  # CI width <= 50% of effect size
    current_n: int = 5
) -> Dict[str, PrecisionAssessment]:
    """
    Assess whether confidence intervals are precise enough
    
    Two criteria:
    1. Absolute width: CI should be < max_absolute_width (e.g., 1.0 effect size units)
    2. Relative width: CI width / effect_size should be < max_relative_width
    
    Args:
        bootstrap_results: Results from analyze_effect_size_all_sources_nested()
        max_absolute_width: Maximum acceptable absolute CI width
        max_relative_width: Maximum acceptable relative CI width
        current_n: Current sample size
        
    Returns:
        Dict mapping source_pop to PrecisionAssessment
    """
    assessments = {}
    
    for source_pop, results in bootstrap_results.items():
        boot_results = results['bootstrap_results']
        
        ci_lower = boot_results['effect_size_ci_lower']
        ci_upper = boot_results['effect_size_ci_upper']
        effect_size = boot_results['effect_size']
        
        # CI width
        ci_width = ci_upper - ci_lower
        
        # Relative width (handle near-zero effects)
        if abs(effect_size) > 0.01:
            ci_width_relative = ci_width / abs(effect_size)
        else:
            ci_width_relative = np.inf
        
        # Precision checks
        is_precise_absolute = ci_width <= max_absolute_width
        is_precise_relative = ci_width_relative <= max_relative_width
        is_precise = is_precise_absolute and is_precise_relative
        
        # Estimate N needed for desired precision
        # CI width scales as ~ 1/sqrt(n)
        if not is_precise:
            target_width = min(max_absolute_width, max_relative_width * abs(effect_size))
            if target_width > 0:
                n_for_precision = int(np.ceil(current_n * (ci_width / target_width) ** 2))
            else:
                n_for_precision = 9999
        else:
            n_for_precision = current_n
        
        assessments[source_pop] = PrecisionAssessment(
            ci_width=ci_width,
            ci_width_relative=ci_width_relative,
            is_precise=is_precise,
            needs_narrower_ci=not is_precise,
            recommended_n=n_for_precision
        )
    
    return assessments


# ============================================================================
# Biological Significance Functions
# ============================================================================

def assess_biological_significance(
    bootstrap_results: Dict,
    min_meaningful_effect_size: float = 0.5,
    min_meaningful_diff_nS: float = 0.1
) -> Dict[str, BiologicalSignificanceAssessment]:
    """
    Determine if effects are biologically meaningful
    
    Uses two criteria:
    1. Effect size threshold (Cohen's d)
    2. Absolute weight difference threshold (nS)
    
    Args:
        bootstrap_results: Results from analyze_effect_size_all_sources_nested()
        min_meaningful_effect_size: Minimum Cohen's d to be meaningful
        min_meaningful_diff_nS: Minimum weight difference (nS) to be meaningful
        
    Returns:
        Dict mapping source_pop to BiologicalSignificanceAssessment
    """
    assessments = {}
    
    for source_pop, results in bootstrap_results.items():
        boot_results = results['bootstrap_results']
        
        effect_size = boot_results['effect_size']
        mean_diff = boot_results['mean_diff']
        
        # Effect size categories (Cohen's conventions)
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            strength = 'negligible'
        elif abs_d < 0.5:
            strength = 'small'
        elif abs_d < 0.8:
            strength = 'medium'
        elif abs_d < 1.2:
            strength = 'large'
        else:
            strength = 'very_large'
        
        # Biological meaningfulness
        is_meaningful = (abs_d >= min_meaningful_effect_size and 
                         abs(mean_diff) >= min_meaningful_diff_nS)
        
        # Interpretation
        if is_meaningful:
            if strength == 'very_large':
                interpretation = (f"Very large effect (d={effect_size:.2f}). "
                                f"Excited cells receive {mean_diff:+.2f} nS "
                                f"different input - highly biologically relevant.")
            elif strength == 'large':
                interpretation = (f"Large effect (d={effect_size:.2f}). "
                                f"Substantial difference in synaptic input "
                                f"({mean_diff:+.2f} nS) - biologically meaningful.")
            else:
                interpretation = (f"Moderate effect (d={effect_size:.2f}). "
                                f"Detectable difference in weights "
                                f"({mean_diff:+.2f} nS) - may be functionally relevant.")
        else:
            interpretation = (f"Effect size (d={effect_size:.2f}) or weight difference "
                            f"({mean_diff:+.2f} nS) below biological significance threshold. "
                            f"Unlikely to have major functional impact.")
        
        assessments[source_pop] = BiologicalSignificanceAssessment(
            effect_size=effect_size,
            mean_diff_nS=mean_diff,
            is_biologically_meaningful=is_meaningful,
            strength_category=strength,
            interpretation=interpretation
        )
    
    return assessments


# ============================================================================
# Integrated Decision Framework
# ============================================================================

def make_data_collection_decision(
    bootstrap_results: Dict,
    power_results: Dict[str, PowerAnalysisResult],
    precision_results: Dict[str, PrecisionAssessment],
    bio_sig_results: Dict[str, BiologicalSignificanceAssessment],
    current_n: int = 5,
    target_power: float = 0.80,
    max_feasible_n: int = 20
) -> Dict[str, DataCollectionRecommendation]:
    """
    Integrate all criteria to make data collection recommendation
    
    Decision logic:
    1. If already significant + biologically meaningful + precise -> STOP
    2. If biologically meaningful but underpowered/imprecise + feasible N -> CONTINUE
    3. If promising but uncertain -> CONSIDER
    4. If not biologically meaningful or infeasible N -> STOP
    
    Args:
        bootstrap_results: Results from analyze_effect_size_all_sources_nested()
        power_results: Results from analyze_statistical_power()
        precision_results: Results from assess_ci_precision()
        bio_sig_results: Results from assess_biological_significance()
        current_n: Current sample size
        target_power: Desired statistical power
        max_feasible_n: Maximum feasible sample size
        
    Returns:
        Dict mapping source_pop to DataCollectionRecommendation
    """
    recommendations = {}
    
    for source_pop in bootstrap_results.keys():
        # Extract results
        boot_res = bootstrap_results[source_pop]['bootstrap_results']
        power_res = power_results[source_pop]
        precision_res = precision_results[source_pop]
        bio_res = bio_sig_results[source_pop]
        
        p_value = boot_res['p_value']
        effect_size = boot_res['effect_size']
        ci_lower = boot_res['effect_size_ci_lower']
        ci_upper = boot_res['effect_size_ci_upper']
        
        # Decision variables
        is_significant = p_value < 0.05
        is_bio_meaningful = bio_res.is_biologically_meaningful
        is_precise = precision_res.is_precise
        current_power = power_res.current_power
        
        # Required N (use maximum of power and precision requirements)
        n_for_power = power_res.required_n_80 if target_power >= 0.80 else power_res.required_n_90
        n_for_precision = precision_res.recommended_n
        required_n = max(n_for_power, n_for_precision)
        is_feasible = required_n <= max_feasible_n
        
        # CI crosses zero?
        ci_includes_zero = ci_lower * ci_upper <= 0
        
        # Make decision
        if is_significant and is_bio_meaningful and is_precise:
            # Clear positive result
            action = 'STOP'
            priority = 'NONE'
            recommended_n = current_n
            justification = (f"Already significant (p={p_value:.3f}) with "
                           f"biologically meaningful effect (d={effect_size:.2f}) "
                           f"and precise CI. No additional data needed.")
        
        elif not is_bio_meaningful:
            # Effect too small to matter
            action = 'STOP'
            priority = 'NONE'
            recommended_n = current_n
            justification = (f"Effect size (d={effect_size:.2f}) below biological "
                           f"significance threshold. Not worth pursuing.")
        
        elif is_bio_meaningful and not is_feasible:
            # Would need too much data
            action = 'STOP'
            priority = 'LOW'
            recommended_n = current_n
            justification = (f"Biologically meaningful effect but requires "
                           f"N={required_n} (infeasible). Current evidence "
                           f"(d={effect_size:.2f}, p={p_value:.3f}) is suggestive "
                           f"but underpowered. Report as exploratory finding.")
        
        elif is_bio_meaningful and is_feasible and current_power < 0.60:
            # Clear need for more data
            action = 'CONTINUE'
            priority = 'HIGH'
            recommended_n = required_n
            justification = (f"Large, biologically meaningful effect "
                           f"(d={effect_size:.2f}) but currently underpowered "
                           f"(power={current_power:.2f}). Adding "
                           f"{required_n - current_n} connectivity instances "
                           f"will achieve {target_power:.0%} power.")
        
        elif is_bio_meaningful and is_feasible and 0.60 <= current_power < 0.80:
            # Borderline - consider adding data
            action = 'CONSIDER'
            priority = 'MEDIUM'
            recommended_n = required_n
            justification = (f"Meaningful effect (d={effect_size:.2f}) with "
                           f"moderate power ({current_power:.2f}). Consider adding "
                           f"{required_n - current_n} instances for more definitive "
                           f"results, though current evidence is suggestive "
                           f"(p={p_value:.3f}).")
        
        elif is_significant and not is_precise:
            # Significant but imprecise
            action = 'CONSIDER'
            priority = 'MEDIUM'
            recommended_n = precision_res.recommended_n
            justification = (f"Significant effect (p={p_value:.3f}) but CI is wide "
                           f"({ci_lower:.2f} to {ci_upper:.2f}). Consider adding "
                           f"{recommended_n - current_n} instances for better precision.")
        
        else:
            # Default: borderline cases
            action = 'CONSIDER'
            priority = 'LOW'
            recommended_n = min(required_n, max_feasible_n)
            justification = (f"Uncertain evidence (d={effect_size:.2f}, p={p_value:.3f}). "
                           f"May benefit from additional data but not high priority.")
        
        additional_n = max(0, recommended_n - current_n)
        
        recommendations[source_pop] = DataCollectionRecommendation(
            action=action,
            priority=priority,
            recommended_total_n=recommended_n,
            additional_n_needed=additional_n,
            justification=justification,
            details={
                'p_value': p_value,
                'effect_size': effect_size,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'current_power': current_power,
                'is_significant': is_significant,
                'is_bio_meaningful': is_bio_meaningful,
                'is_precise': is_precise,
                'bio_strength': bio_res.strength_category
            }
        )
    
    return recommendations


# ============================================================================
# Reporting Functions
# ============================================================================

def print_effect_size_decision_report(
    bootstrap_results: Dict,
    power_results: Dict[str, PowerAnalysisResult],
    precision_results: Dict[str, PrecisionAssessment],
    bio_sig_results: Dict[str, BiologicalSignificanceAssessment],
    recommendations: Dict[str, DataCollectionRecommendation],
    target_population: str,
    post_population: str
):
    """Print report integrating all decision criteria"""
    print("\n" + "="*80)
    print(f"EFFECT SIZE DECISION REPORT: {target_population.upper()} -> {post_population.upper()}")
    print("="*80)
    
    # Summary table
    print("\n" + "-"*80)
    print(f"{'Source':<8} {'d':<8} {'p':<8} {'Power':<8} {'CI Width':<10} {'Bio Sig':<10} {'Action':<12}")
    print("-"*80)
    
    for source_pop in sorted(bootstrap_results.keys()):
        boot = bootstrap_results[source_pop]['bootstrap_results']
        power = power_results[source_pop]
        precision = precision_results[source_pop]
        bio = bio_sig_results[source_pop]
        rec = recommendations[source_pop]
        
        # Format values
        d_str = f"{boot['effect_size']:.2f}"
        p_str = f"{boot['p_value']:.3f}"
        pow_str = f"{power.current_power:.2f}"
        ci_str = f"{precision.ci_width:.2f}"
        bio_str = bio.strength_category[:6]
        action_str = f"{rec.action} ({rec.priority})"
        
        print(f"{source_pop.upper():<8} {d_str:<8} {p_str:<8} {pow_str:<8} "
              f"{ci_str:<10} {bio_str:<10} {action_str:<12}")
    
    # Detailed recommendations
    print("\n" + "="*80)
    print("DETAILED RECOMMENDATIONS")
    print("="*80)
    
    for source_pop in sorted(bootstrap_results.keys()):
        rec = recommendations[source_pop]
        bio = bio_sig_results[source_pop]
        
        print(f"\n{source_pop.upper()} -> {post_population.upper()}:")
        print(f"  Action: {rec.action} (Priority: {rec.priority})")
        print(f"  Current N: {power_results[source_pop].current_n}")
        print(f"  Recommended N: {rec.recommended_total_n}")
        print(f"  Additional needed: {rec.additional_n_needed}")
        print(f"\n  Biological significance: {bio.interpretation}")
        print(f"\n  Justification: {rec.justification}")
    
    # Overall recommendation
    print("\n" + "="*80)
    print("OVERALL RECOMMENDATION")
    print("="*80)
    
    high_priority = [s for s, r in recommendations.items() if r.priority == 'HIGH']
    medium_priority = [s for s, r in recommendations.items() if r.priority == 'MEDIUM']
    
    if high_priority:
        max_add_n = max(recommendations[s].additional_n_needed for s in high_priority)
        print(f"\nHIGH PRIORITY: Add {max_add_n} connectivity instances")
        print(f"  Sources requiring more data: {', '.join(s.upper() for s in high_priority)}")
        print(f"  These show large, biologically meaningful effects but are underpowered")
    elif medium_priority:
        max_add_n = max(recommendations[s].additional_n_needed for s in medium_priority)
        print(f"\nMEDIUM PRIORITY: Consider adding {max_add_n} connectivity instances")
        print(f"  Sources that would benefit: {', '.join(s.upper() for s in medium_priority)}")
        print(f"  Current evidence is suggestive but could be strengthened")
    else:
        print("\nNO ADDITIONAL DATA NEEDED")
        print(f"  All effects either: (1) already well-characterized, or")
        print(f"                      (2) not biologically meaningful")
    
    print("\n" + "="*80)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_effect_size_decision_summary(
    bootstrap_results: Dict,
    recommendations: Dict[str, DataCollectionRecommendation],
    bio_sig_results: Dict[str, BiologicalSignificanceAssessment],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create visual summary of decision framework results
    
    Three-panel figure:
    1. Effect sizes with CIs, colored by recommendation
    2. Current vs required sample size
    3. Statistical vs biological significance
    """
    
    source_pops = sorted(bootstrap_results.keys())
    n_sources = len(source_pops)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color scheme
    action_colors = {
        'STOP': '#95a5a6',      # Gray
        'CONSIDER': '#f39c12',  # Orange
        'CONTINUE': '#e74c3c'   # Red
    }
    
    # Panel 1: Effect sizes with CIs
    ax1 = axes[0]
    y_pos = np.arange(n_sources)
    
    for i, source_pop in enumerate(source_pops):
        boot = bootstrap_results[source_pop]['bootstrap_results']
        rec = recommendations[source_pop]
        
        d = boot['effect_size']
        ci_lower = boot['effect_size_ci_lower']
        ci_upper = boot['effect_size_ci_upper']
        
        color = action_colors[rec.action]
        
        # Plot point and CI
        ax1.plot(d, y_pos[i], 'o', color=color, markersize=12)
        ax1.plot([ci_lower, ci_upper], [y_pos[i], y_pos[i]], 
                '-', color=color, linewidth=3)
        
        # Significance marker
        if boot['p_value'] < 0.05:
            ax1.plot(d, y_pos[i], '*', color='gold', markersize=20)
    
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(0.5, color='green', linestyle=':', alpha=0.3, label='Medium effect')
    ax1.axvline(0.8, color='blue', linestyle=':', alpha=0.3, label='Large effect')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s.upper() for s in source_pops])
    ax1.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax1.set_title("Effect Sizes with 95% CI\n(* = p<0.05)", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Panel 2: Current vs required N
    ax2 = axes[1]
    
    current_n_list = []
    required_n_list = []
    colors_n = []
    
    for source_pop in source_pops:
        rec = recommendations[source_pop]
        current_n_list.append(5)  # Assuming pilot is N=5
        required_n_list.append(rec.recommended_total_n if rec.recommended_total_n < 100 else 100)
        colors_n.append(action_colors[rec.action])
    
    x = np.arange(n_sources)
    width = 0.35
    
    ax2.bar(x - width/2, current_n_list, width, label='Current N', 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, required_n_list, width, label='Required N',
            color=colors_n, alpha=0.7, edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in source_pops], rotation=45, ha='right')
    ax2.set_ylabel('Sample Size (N)', fontsize=12)
    ax2.set_title('Current vs Required Sample Size', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Statistical vs biological significance
    ax3 = axes[2]
    
    # Create 2x2 grid
    bio_meaningful = []
    stat_significant = []
    colors_grid = []
    
    for source_pop in source_pops:
        boot = bootstrap_results[source_pop]['bootstrap_results']
        bio = bio_sig_results[source_pop]
        rec = recommendations[source_pop]
        
        bio_meaningful.append(1 if bio.is_biologically_meaningful else 0)
        stat_significant.append(1 if boot['p_value'] < 0.05 else 0)
        colors_grid.append(action_colors[rec.action])
    
    # Scatter plot
    for i, source_pop in enumerate(source_pops):
        ax3.scatter(stat_significant[i], bio_meaningful[i],
                   s=300, c=colors_grid[i], alpha=0.7,
                   edgecolor='black', linewidth=2)
        ax3.text(stat_significant[i], bio_meaningful[i], source_pop.upper(),
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Quadrant labels
    ax3.text(0.25, 0.25, 'Null finding\n(Stop)', 
            ha='center', va='center', fontsize=10, alpha=0.5, style='italic')
    ax3.text(0.75, 0.25, 'Type I error?\n(Check effect)', 
            ha='center', va='center', fontsize=10, alpha=0.5, style='italic')
    ax3.text(0.25, 0.75, 'Underpowered\n(Add data)', 
            ha='center', va='center', fontsize=10, alpha=0.5, style='italic')
    ax3.text(0.75, 0.75, 'Clear finding\n(Report)', 
            ha='center', va='center', fontsize=10, alpha=0.5, style='italic')
    
    ax3.set_xlim(-0.2, 1.2)
    ax3.set_ylim(-0.2, 1.2)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Not Sig\n(p≥0.05)', 'Significant\n(p<0.05)'])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Not Bio.\nMeaningful', 'Bio.\nMeaningful'])
    ax3.set_title('Statistical vs Biological Significance', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_axisbelow(True)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=action_colors['STOP'], label='STOP (sufficient)'),
        mpatches.Patch(color=action_colors['CONSIDER'], label='CONSIDER (optional)'),
        mpatches.Patch(color=action_colors['CONTINUE'], label='CONTINUE (needed)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=3, fontsize=11, frameon=True)
    
    plt.suptitle('Effect Size Decision Framework Summary', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved decision summary plot to: {save_path}")
    
    return fig


# ============================================================================
# Complete Workflow
# ============================================================================

def run_effect_size_decision_analysis(
    bootstrap_results: Dict,
    target_population: str,
    post_population: str,
    current_n: int = 5,
    target_power: float = 0.80,
    max_feasible_n: int = 20,
    min_meaningful_effect: float = 0.5,
    min_meaningful_diff_nS: float = 0.1,
    icc_value: Optional[float] = None,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Effect size decision analysis workflow
    
    Takes bootstrap effect size results and produces
    data collection recommendations.
    
    Args:
        bootstrap_results: Results from analyze_effect_size_all_sources_nested()
        target_population: Stimulated population
        post_population: Post-synaptic population
        current_n: Current number of connectivity instances
        target_power: Desired statistical power
        max_feasible_n: Maximum feasible sample size
        min_meaningful_effect: Minimum biologically meaningful effect size
        min_meaningful_diff_nS: Minimum biologically meaningful weight difference
        icc_value: Optional ICC value for contextualization
        save_dir: Directory to save results and plots
        
    Returns:
        Complete results dict with all analyses
    """
    
    print("\n" + "="*80)
    print("EFFECT SIZE DECISION FRAMEWORK")
    print("="*80)
    
    # Power analysis
    print("\nAnalyzing statistical power...")
    power_results = analyze_statistical_power(
        bootstrap_results,
        current_n=current_n
    )
    
    # Precision analysis
    print("\nAssessing CI precision...")
    precision_results = assess_ci_precision(
        bootstrap_results,
        current_n=current_n
    )
    
    # Biological significance
    print("\nEvaluating biological significance...")
    bio_sig_results = assess_biological_significance(
        bootstrap_results,
        min_meaningful_effect_size=min_meaningful_effect,
        min_meaningful_diff_nS=min_meaningful_diff_nS
    )
    
    # Integrated decision
    print("\nMaking data collection recommendations...")
    recommendations = make_data_collection_decision(
        bootstrap_results,
        power_results,
        precision_results,
        bio_sig_results,
        current_n=current_n,
        target_power=target_power,
        max_feasible_n=max_feasible_n
    )
    
    # Step 5: Generate report
    print("\nGenerating report...")
    print_effect_size_decision_report(
        bootstrap_results,
        power_results,
        precision_results,
        bio_sig_results,
        recommendations,
        target_population,
        post_population
    )
    
    
    # Create visualizations
    if save_dir:
        print("\nCreating visualizations...")
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Decision summary plot
        fig_decision = plot_effect_size_decision_summary(
            bootstrap_results,
            recommendations,
            bio_sig_results,
            save_path=str(save_path / f'decision_summary_{target_population}_{post_population}.pdf')
        )
        plt.close(fig_decision)
        
        print(f"  Saved plots to: {save_path}")
    
    # Package results
    complete_results = {
        'bootstrap_results': bootstrap_results,
        'power_analysis': power_results,
        'precision_analysis': precision_results,
        'biological_significance': bio_sig_results,
        'recommendations': recommendations,
        'metadata': {
            'target_population': target_population,
            'post_population': post_population,
            'current_n': current_n,
            'target_power': target_power,
            'max_feasible_n': max_feasible_n,
            'min_meaningful_effect': min_meaningful_effect,
            'min_meaningful_diff_nS': min_meaningful_diff_nS
        }
    }
    
    print("\n" + "="*80)
    print("DECISION ANALYSIS COMPLETE")
    print("="*80)
    
    return complete_results
