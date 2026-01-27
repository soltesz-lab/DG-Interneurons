"""
PCA Analysis of Input Weight Patterns

This module provides Principal Components Analysis (PCA) of synaptic input
weight patterns to test whether excited and suppressed cells differ in their
multivariate input combinations rather than individual source means.

Complements univariate nested weights analysis by testing for combinatorial
patterns that might exist even when individual source means are identical.

Key Functions:
    - extract_input_weight_matrix_by_connectivity: Build input weight matrices
    - perform_pca_by_connectivity: Perform PCA within each connectivity instance
    - compute_separation_statistics: Quantify excited vs suppressed separation
    - bootstrap_pca_separation: Bootstrap confidence intervals
    - permutation_test_pca_separation: Generate null distribution
    - analyze_input_patterns_pca: Complete workflow function

Design Philosophy:
    - Connectivity instances are the independent statistical units (ICC ≈ 1.0)
    - Bootstrap resamples at connectivity level
    - Separate analysis within each connectivity preserves structure
    - Multi-panel visualizations show consistency across instances
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger('pca_input_patterns')
logger.setLevel(logging.INFO)


# ============================================================================
# Data Structures
# ============================================================================

class PCAResult(NamedTuple):
    """Results from PCA analysis for a single connectivity instance"""
    pc_scores: np.ndarray           # [n_cells × n_components]
    explained_variance: np.ndarray  # Variance explained by each PC
    explained_variance_ratio: np.ndarray
    components: np.ndarray          # PC loadings [n_components × n_sources]
    mean: np.ndarray                # Mean input weight pattern
    
    excited_scores: np.ndarray      # PC scores for excited cells
    suppressed_scores: np.ndarray   # PC scores for suppressed cells


class SeparationStatistics(NamedTuple):
    """Statistics quantifying separation between excited and suppressed groups"""
    hotellings_t2: float
    p_value_hotelling: float
    cohens_d_pc1: float
    cohens_d_pc2: float
    mahalanobis_distance: float
    overlap_coefficient: float  # Proportion of overlapping distributions


class BootstrapPCAResult(NamedTuple):
    """Bootstrap results for PCA separation"""
    observed_t2: float
    bootstrap_t2: np.ndarray
    p_value_bootstrap: float
    ci_lower_t2: float
    ci_upper_t2: float
    
    observed_d_pc1: float
    bootstrap_d_pc1: np.ndarray
    ci_lower_d_pc1: float
    ci_upper_d_pc1: float
    
    observed_d_pc2: float
    bootstrap_d_pc2: np.ndarray
    ci_lower_d_pc2: float
    ci_upper_d_pc2: float


# ============================================================================
# Core Analysis Functions
# ============================================================================

def extract_input_weight_matrix_by_connectivity(
    circuit,
    classifications: Dict,
    post_population: str,
    source_populations: List[str],
    normalize: bool = True
) -> Dict:
    """
    Extract matrix of total input weights from each source to each cell
    
    For each connectivity instance, builds a matrix where:
    - Rows = cells (excited cells first, then suppressed)
    - Columns = source populations
    - Values = total synaptic conductance from that source to that cell
    
    Args:
        circuit: DentateCircuit instance
        classifications: Output from classify_cells_by_connectivity()
        post_population: Post-synaptic population ('gc', 'mc', 'pv', 'sst')
        source_populations: List of source populations to include
        normalize: Whether to z-score weights by source (recommended)
        
    Returns:
        dict: {
            connectivity_idx: {
                'weight_matrix': [n_cells × n_sources],
                'source_names': List[str],
                'cell_indices': np.ndarray,
                'excited_mask': np.ndarray,
                'suppressed_mask': np.ndarray,
                'n_excited': int,
                'n_suppressed': int
            }
        }
    """
    weight_matrices_by_conn = {}
    
    for conn_idx, classification in classifications.items():
        excited_cells = classification['excited_cells']
        suppressed_cells = classification['suppressed_cells']
        n_cells = len(excited_cells) + len(suppressed_cells)
        
        if n_cells == 0:
            logger.warning(f"Connectivity {conn_idx}: No cells to analyze")
            continue
        
        # Build weight matrix
        all_cells = np.concatenate([excited_cells, suppressed_cells])
        weight_matrix = np.zeros((n_cells, len(source_populations)))
        
        for source_idx, source_pop in enumerate(source_populations):
            conn_name = f'{source_pop}_{post_population}'
            
            if conn_name not in circuit.connectivity.conductance_matrices:
                # No connection from this source
                weight_matrix[:, source_idx] = 0.0
                continue
            
            conductance_matrix = circuit.connectivity.conductance_matrices[conn_name]
            weights = conductance_matrix.conductances  # [n_pre, n_post]
            
            # Convert to numpy if needed
            if hasattr(weights, 'cpu'):
                weights = weights.cpu().numpy()
            else:
                weights = np.array(weights)
            
            # Sum inputs from all presynaptic cells to each postsynaptic cell
            total_weights = weights.sum(axis=0)  # [n_post]
            
            # Extract for relevant cells
            weight_matrix[:, source_idx] = total_weights[all_cells]
        
        # Optional normalization (z-score by source)
        if normalize:
            means = weight_matrix.mean(axis=0)
            stds = weight_matrix.std(axis=0)
            # Avoid division by zero
            stds[stds < 1e-10] = 1.0
            weight_matrix = (weight_matrix - means) / stds
        
        # Create masks
        excited_mask = np.zeros(n_cells, dtype=bool)
        excited_mask[:len(excited_cells)] = True
        suppressed_mask = ~excited_mask
        
        weight_matrices_by_conn[conn_idx] = {
            'weight_matrix': weight_matrix,
            'source_names': source_populations,
            'cell_indices': all_cells,
            'excited_mask': excited_mask,
            'suppressed_mask': suppressed_mask,
            'n_excited': len(excited_cells),
            'n_suppressed': len(suppressed_cells)
        }
    
    return weight_matrices_by_conn


def perform_pca_by_connectivity(
    weight_matrices_by_conn: Dict,
    n_components: Optional[int] = None,
    min_explained_variance: float = 0.95
) -> Dict[int, PCAResult]:
    """
    Perform PCA on input weight patterns within each connectivity
    
    Args:
        weight_matrices_by_conn: Output from extract_input_weight_matrix_by_connectivity()
        n_components: Number of PCs to retain (None = auto-select)
        min_explained_variance: Keep PCs until this much variance explained
        
    Returns:
        Dict mapping connectivity_idx to PCAResult
    """
    pca_results = {}
    
    for conn_idx, data in weight_matrices_by_conn.items():
        weight_matrix = data['weight_matrix']
        excited_mask = data['excited_mask']
        suppressed_mask = data['suppressed_mask']
        
        # Determine number of components
        if n_components is None:
            # Auto-select based on explained variance
            pca_full = PCA()
            pca_full.fit(weight_matrix)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = np.argmax(cumvar >= min_explained_variance) + 1
            n_comp = max(2, min(n_comp, weight_matrix.shape[1]))  # At least 2, at most n_sources
        else:
            n_comp = min(n_components, weight_matrix.shape[1])
        
        # Perform PCA
        pca = PCA(n_components=n_comp)
        pc_scores = pca.fit_transform(weight_matrix)
        
        # Separate by response type
        excited_scores = pc_scores[excited_mask]
        suppressed_scores = pc_scores[suppressed_mask]
        
        pca_results[conn_idx] = PCAResult(
            pc_scores=pc_scores,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            components=pca.components_,
            mean=pca.mean_,
            excited_scores=excited_scores,
            suppressed_scores=suppressed_scores
        )
    
    return pca_results


def compute_separation_statistics(
    pca_result: PCAResult,
    n_components: int = 2
) -> SeparationStatistics:
    """
    Compute statistics quantifying separation in PC space
    
    Uses Hotelling's T^2 test for multivariate separation and computes
    effect sizes (Cohen's d) for individual principal components.
    
    Args:
        pca_result: PCAResult from perform_pca_by_connectivity()
        n_components: Number of PCs to use for multivariate tests
        
    Returns:
        SeparationStatistics
    """
    excited = pca_result.excited_scores[:, :n_components]
    suppressed = pca_result.suppressed_scores[:, :n_components]
    
    n_excited = excited.shape[0]
    n_suppressed = suppressed.shape[0]
    
    if n_excited == 0 or n_suppressed == 0:
        return SeparationStatistics(
            hotellings_t2=0.0,
            p_value_hotelling=1.0,
            cohens_d_pc1=0.0,
            cohens_d_pc2=0.0,
            mahalanobis_distance=0.0,
            overlap_coefficient=1.0
        )
    
    # Means
    mean_excited = excited.mean(axis=0)
    mean_suppressed = suppressed.mean(axis=0)
    
    # Pooled covariance
    cov_excited = np.cov(excited.T)
    cov_suppressed = np.cov(suppressed.T)
    
    # Handle single-component case
    if n_components == 1:
        cov_excited = np.array([[cov_excited]])
        cov_suppressed = np.array([[cov_suppressed]])
    
    cov_pooled = ((n_excited - 1) * cov_excited + (n_suppressed - 1) * cov_suppressed) / (n_excited + n_suppressed - 2)
    
    # Hotelling's T²
    mean_diff = mean_excited - mean_suppressed
    
    try:
        cov_pooled_inv = np.linalg.pinv(cov_pooled)
        t2 = (n_excited * n_suppressed) / (n_excited + n_suppressed) * mean_diff @ cov_pooled_inv @ mean_diff
        
        # F-statistic and p-value
        p = n_components  # Dimensionality
        n = n_excited + n_suppressed
        f_stat = ((n - p - 1) / (p * (n - 2))) * t2
        p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    except np.linalg.LinAlgError:
        t2 = 0.0
        p_value = 1.0
    
    # Cohen's d for individual PCs
    pooled_std_pc1 = np.sqrt(
        ((n_excited - 1) * excited[:, 0].var() + (n_suppressed - 1) * suppressed[:, 0].var()) / 
        (n_excited + n_suppressed - 2)
    )
    d_pc1 = (mean_excited[0] - mean_suppressed[0]) / pooled_std_pc1 if pooled_std_pc1 > 0 else 0.0
    
    if n_components >= 2:
        pooled_std_pc2 = np.sqrt(
            ((n_excited - 1) * excited[:, 1].var() + (n_suppressed - 1) * suppressed[:, 1].var()) / 
            (n_excited + n_suppressed - 2)
        )
        d_pc2 = (mean_excited[1] - mean_suppressed[1]) / pooled_std_pc2 if pooled_std_pc2 > 0 else 0.0
    else:
        d_pc2 = 0.0
    
    # Mahalanobis distance between group centroids
    try:
        mahal_dist = np.sqrt(mean_diff @ cov_pooled_inv @ mean_diff)
    except:
        mahal_dist = 0.0
    
    # Overlap coefficient
    overlap = compute_overlap_coefficient(excited, suppressed, cov_pooled)
    
    return SeparationStatistics(
        hotellings_t2=t2,
        p_value_hotelling=p_value,
        cohens_d_pc1=d_pc1,
        cohens_d_pc2=d_pc2,
        mahalanobis_distance=mahal_dist,
        overlap_coefficient=overlap
    )


def compute_overlap_coefficient(excited, suppressed, cov_pooled):
    """
    Compute overlap between two multivariate distributions
    
    Uses Mahalanobis distance to define typical range of suppressed
    distribution and computes proportion of excited cells within that range.
    """
    if excited.shape[0] == 0 or suppressed.shape[0] == 0:
        return 1.0
    
    mean_suppressed = suppressed.mean(axis=0)
    
    try:
        cov_inv = np.linalg.pinv(cov_pooled)
        
        # For each excited cell, compute Mahalanobis distance to suppressed centroid
        distances_excited = []
        for cell in excited:
            diff = cell - mean_suppressed
            dist = np.sqrt(diff @ cov_inv @ diff)
            distances_excited.append(dist)
        
        # For each suppressed cell, compute distance to own centroid
        distances_suppressed = []
        for cell in suppressed:
            diff = cell - mean_suppressed
            dist = np.sqrt(diff @ cov_inv @ diff)
            distances_suppressed.append(dist)
        
        # Overlap = proportion of excited cells within typical suppressed range
        typical_range = np.percentile(distances_suppressed, 95)
        overlap = np.mean(np.array(distances_excited) < typical_range)
        
        return overlap
    except:
        return 1.0


def bootstrap_pca_separation(
    pca_results: Dict[int, PCAResult],
    separation_stats: Dict[int, SeparationStatistics],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> BootstrapPCAResult:
    """
    Bootstrap confidence intervals for PCA separation statistics
    
    Resamples connectivity instances with replacement (treating connectivity
    as the independent statistical unit).
    
    Args:
        pca_results: Dict from perform_pca_by_connectivity()
        separation_stats: Dict from compute_separation_statistics()
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level
        random_seed: Random seed for reproducibility
        
    Returns:
        BootstrapPCAResult
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_conn = len(pca_results)
    conn_indices = list(pca_results.keys())
    
    # Observed values (mean across connectivity instances)
    observed_t2 = np.mean([separation_stats[i].hotellings_t2 for i in conn_indices])
    observed_d_pc1 = np.mean([separation_stats[i].cohens_d_pc1 for i in conn_indices])
    observed_d_pc2 = np.mean([separation_stats[i].cohens_d_pc2 for i in conn_indices])
    
    # Bootstrap
    bootstrap_t2 = []
    bootstrap_d_pc1 = []
    bootstrap_d_pc2 = []
    
    for _ in range(n_bootstrap):
        # Resample connectivity instances
        resampled_indices = np.random.choice(conn_indices, size=n_conn, replace=True)
        
        # Compute statistics on resampled data
        boot_t2 = np.mean([separation_stats[i].hotellings_t2 for i in resampled_indices])
        boot_d_pc1 = np.mean([separation_stats[i].cohens_d_pc1 for i in resampled_indices])
        boot_d_pc2 = np.mean([separation_stats[i].cohens_d_pc2 for i in resampled_indices])
        
        bootstrap_t2.append(boot_t2)
        bootstrap_d_pc1.append(boot_d_pc1)
        bootstrap_d_pc2.append(boot_d_pc2)
    
    bootstrap_t2 = np.array(bootstrap_t2)
    bootstrap_d_pc1 = np.array(bootstrap_d_pc1)
    bootstrap_d_pc2 = np.array(bootstrap_d_pc2)
    
    # Confidence intervals
    alpha = 1 - confidence_level
    ci_lower_percentile = (alpha / 2) * 100
    ci_upper_percentile = (1 - alpha / 2) * 100
    
    ci_t2 = np.percentile(bootstrap_t2, [ci_lower_percentile, ci_upper_percentile])
    ci_d_pc1 = np.percentile(bootstrap_d_pc1, [ci_lower_percentile, ci_upper_percentile])
    ci_d_pc2 = np.percentile(bootstrap_d_pc2, [ci_lower_percentile, ci_upper_percentile])
    
    # Two-sided p-value
    p_value = 2 * min(np.sum(bootstrap_t2 <= 0) / n_bootstrap,
                     np.sum(bootstrap_t2 >= 0) / n_bootstrap)
    
    return BootstrapPCAResult(
        observed_t2=observed_t2,
        bootstrap_t2=bootstrap_t2,
        p_value_bootstrap=p_value,
        ci_lower_t2=ci_t2[0],
        ci_upper_t2=ci_t2[1],
        observed_d_pc1=observed_d_pc1,
        bootstrap_d_pc1=bootstrap_d_pc1,
        ci_lower_d_pc1=ci_d_pc1[0],
        ci_upper_d_pc1=ci_d_pc1[1],
        observed_d_pc2=observed_d_pc2,
        bootstrap_d_pc2=bootstrap_d_pc2,
        ci_lower_d_pc2=ci_d_pc2[0],
        ci_upper_d_pc2=ci_d_pc2[1]
    )


def permutation_test_pca_separation(
    weight_matrices_by_conn: Dict,
    n_permutations: int = 1000,
    n_components: int = 2,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Generate null distribution by permuting cell labels within each connectivity
    
    Tests whether observed separation is greater than expected by chance.
    
    Args:
        weight_matrices_by_conn: Output from extract_input_weight_matrix_by_connectivity()
        n_permutations: Number of random permutations
        n_components: Number of PCs to use
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with null distribution statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Observed statistics
    observed_pca = perform_pca_by_connectivity(weight_matrices_by_conn, n_components=n_components)
    observed_stats = {
        conn_idx: compute_separation_statistics(pca_result, n_components)
        for conn_idx, pca_result in observed_pca.items()
    }
    
    observed_t2_mean = np.mean([s.hotellings_t2 for s in observed_stats.values()])
    
    # Null distribution
    null_t2_values = []
    
    for perm in range(n_permutations):
        # For each connectivity, randomly permute excited/suppressed labels
        permuted_weights = {}
        
        for conn_idx, data in weight_matrices_by_conn.items():
            permuted_data = data.copy()
            
            # Randomly permute the masks
            n_cells = len(permuted_data['excited_mask'])
            permuted_labels = np.random.permutation(n_cells)
            n_excited = permuted_data['excited_mask'].sum()
            
            new_excited_mask = np.zeros(n_cells, dtype=bool)
            new_excited_mask[permuted_labels[:n_excited]] = True
            
            permuted_data['excited_mask'] = new_excited_mask
            permuted_data['suppressed_mask'] = ~new_excited_mask
            
            permuted_weights[conn_idx] = permuted_data
        
        # Compute PCA on permuted data
        permuted_pca = perform_pca_by_connectivity(permuted_weights, n_components=n_components)
        permuted_stats = {
            conn_idx: compute_separation_statistics(pca_result, n_components)
            for conn_idx, pca_result in permuted_pca.items()
        }
        
        perm_t2_mean = np.mean([s.hotellings_t2 for s in permuted_stats.values()])
        null_t2_values.append(perm_t2_mean)
    
    null_t2_values = np.array(null_t2_values)
    
    # P-value: proportion of null values >= observed
    p_value_permutation = np.sum(null_t2_values >= observed_t2_mean) / n_permutations
    
    return {
        'observed_t2': observed_t2_mean,
        'null_t2_distribution': null_t2_values,
        'p_value_permutation': p_value_permutation,
        'null_t2_mean': null_t2_values.mean(),
        'null_t2_std': null_t2_values.std()
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_pca_scatter_by_connectivity(
    pca_results: Dict[int, PCAResult],
    weight_matrices_by_conn: Dict,
    separation_stats: Dict[int, SeparationStatistics],
    target_population: str,
    post_population: str,
    save_path: Optional[str] = None
):
    """
    Create scatter plots showing excited vs suppressed cells in PC space
    
    Multi-panel figure with one panel per connectivity instance.
    """
    n_conn = len(pca_results)
    n_cols = min(3, n_conn)
    n_rows = int(np.ceil(n_conn / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                            squeeze=False)
    axes = axes.flatten()
    
    for idx, (conn_idx, pca_result) in enumerate(sorted(pca_results.items())):
        ax = axes[idx]
        
        # Extract PC scores
        excited_scores = pca_result.excited_scores
        suppressed_scores = pca_result.suppressed_scores
        
        # Scatter plot
        ax.scatter(suppressed_scores[:, 0], suppressed_scores[:, 1],
                  c='#3498DB', alpha=0.6, s=50, label='Suppressed', 
                  edgecolors='black', linewidths=0.5)
        ax.scatter(excited_scores[:, 0], excited_scores[:, 1],
                  c='#E74C3C', alpha=0.6, s=50, label='Excited', 
                  edgecolors='black', linewidths=0.5)
        
        # Add confidence ellipses
        for scores, color in [(excited_scores, '#E74C3C'), 
                              (suppressed_scores, '#3498DB')]:
            if len(scores) > 2:
                mean = scores[:, :2].mean(axis=0)
                cov = np.cov(scores[:, :2].T)
                
                # 95% confidence ellipse
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width, height = 2 * 2.447 * np.sqrt(eigenvalues)  # 2.447 for 95% CI
                
                ellipse = Ellipse(mean, width, height, angle=angle,
                                facecolor='none', edgecolor=color, linewidth=2, linestyle='--')
                ax.add_patch(ellipse)
        
        # Statistics
        stats = separation_stats[conn_idx]
        
        # Format plot
        var_explained = pca_result.explained_variance_ratio
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=11)
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=11)
        ax.set_title(f'Connectivity {conn_idx}\n' +
                    f'$T^2$={stats.hotellings_t2:.2f}, p={stats.p_value_hotelling:.3f}\n' +
                    f'd(PC1)={stats.cohens_d_pc1:.2f}, d(PC2)={stats.cohens_d_pc2:.2f}',
                    fontsize=10, fontweight='bold')
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Remove unused subplots
    for idx in range(n_conn, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'PCA of Input Weights: {target_population.upper()} $\\rightarrow$ {post_population.upper()}\n' +
                 f'Excited vs Suppressed Cells in PC Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA scatter plot to: {save_path}")
    
    return fig


def plot_pca_loadings_by_connectivity(
    pca_results: Dict[int, PCAResult],
    weight_matrices_by_conn: Dict,
    n_components: int = 3,
    save_path: Optional[str] = None
):
    """
    Create bar plots showing PC loadings (which sources contribute to each PC)
    
    Multi-panel figure showing loadings for each connectivity instance.
    """
    n_conn = len(pca_results)
    n_cols = min(3, n_conn)
    n_rows = int(np.ceil(n_conn / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                            squeeze=False)
    axes = axes.flatten()
    
    for idx, (conn_idx, pca_result) in enumerate(sorted(pca_results.items())):
        ax = axes[idx]
        source_names = weight_matrices_by_conn[conn_idx]['source_names']
        
        # Extract loadings for first n_components PCs
        loadings = pca_result.components[:n_components, :]
        
        # Create grouped bar plot
        x = np.arange(len(source_names))
        width = 0.25
        
        for pc in range(min(n_components, loadings.shape[0])):
            offset = width * (pc - n_components/2 + 0.5)
            ax.bar(x + offset, loadings[pc, :], width,
                  label=f'PC{pc+1}', alpha=0.8)
        
        ax.set_xlabel('Source Population', fontsize=11)
        ax.set_ylabel('Loading', fontsize=11)
        ax.set_title(f'Connectivity {conn_idx}\nPC Loadings', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in source_names], rotation=45, ha='right')
        if idx == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)
    
    # Remove unused subplots
    for idx in range(n_conn, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Principal Component Loadings Across Connectivity Instances',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA loadings plot to: {save_path}")
    
    return fig


def plot_pca_separation_forest(
    bootstrap_result: BootstrapPCAResult,
    target_population: str,
    post_population: str,
    save_path: Optional[str] = None
):
    """
    Create forest plot showing effect sizes with bootstrap CIs
    
    Shows Cohen's d for PC1 and PC2 with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = [
        ('PC1', bootstrap_result.observed_d_pc1, 
         bootstrap_result.ci_lower_d_pc1, bootstrap_result.ci_upper_d_pc1),
        ('PC2', bootstrap_result.observed_d_pc2,
         bootstrap_result.ci_lower_d_pc2, bootstrap_result.ci_upper_d_pc2)
    ]
    
    y_pos = np.arange(len(metrics))
    
    for i, (label, observed, ci_lower, ci_upper) in enumerate(metrics):
        # Determine color based on significance
        ci_crosses_zero = ci_lower * ci_upper <= 0
        color = '#95a5a6' if ci_crosses_zero else '#e74c3c'
        
        # Plot point estimate
        ax.plot(observed, y_pos[i], 'o', color=color, markersize=12)
        
        # Plot CI
        ax.plot([ci_lower, ci_upper], [y_pos[i], y_pos[i]],
               '-', color=color, linewidth=3)
        
        # Add text annotation
        sig_marker = '' if ci_crosses_zero else '*'
        ax.text(ci_upper + 0.1, y_pos[i],
               f"d={observed:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]{sig_marker}",
               va='center', fontsize=11)
    
    # Reference line at zero
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlabel("Cohen's d (Excited vs Suppressed)", fontsize=12)
    ax.set_title(f'PCA Separation Effect Sizes: {target_population.upper()} $\\rightarrow$ {post_population.upper()}\n' +
                f'$T^2$={bootstrap_result.observed_t2:.2f}, p={bootstrap_result.p_value_bootstrap:.4f}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add reference lines for effect size interpretation
    ax.axvline(0.5, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax.axvline(-0.5, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax.text(0.5, -0.5, 'Medium', fontsize=9, alpha=0.5, ha='center')
    
    ax.axvline(0.8, color='blue', linestyle=':', alpha=0.3, linewidth=1)
    ax.axvline(-0.8, color='blue', linestyle=':', alpha=0.3, linewidth=1)
    ax.text(0.8, -0.5, 'Large', fontsize=9, alpha=0.5, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA forest plot to: {save_path}")
    
    return fig


def plot_variance_explained(
    pca_results: Dict[int, PCAResult],
    save_path: Optional[str] = None
):
    """
    Create scree plot showing variance explained by each PC
    
    Shows consistency across connectivity instances.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Individual variance ratios
    for conn_idx, pca_result in sorted(pca_results.items()):
        var_ratio = pca_result.explained_variance_ratio
        n_comp = len(var_ratio)
        ax1.plot(np.arange(1, n_comp + 1), var_ratio,
                'o-', alpha=0.6, label=f'Conn {conn_idx}')
    
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each PC\nAcross Connectivity Instances',
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Cumulative variance
    for conn_idx, pca_result in sorted(pca_results.items()):
        var_ratio = pca_result.explained_variance_ratio
        cumvar = np.cumsum(var_ratio)
        n_comp = len(cumvar)
        ax2.plot(np.arange(1, n_comp + 1), cumvar,
                'o-', alpha=0.6, label=f'Conn {conn_idx}')
    
    ax2.axhline(0.95, color='red', linestyle='--', alpha=0.5, linewidth=1, label='95% threshold')
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained',
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved variance explained plot to: {save_path}")
    
    return fig

def plot_pca_summary_all_targets(
    pca_results_by_target: Dict[str, Dict],
    stimulated_population: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create unified summary plot showing PCA separation across all target populations
    
    Consolidates PCA results from multiple target populations into a comprehensive
    visualization with multiple panels:
    - Panel A: Effect sizes (Cohen's d) for PC1 and PC2 (forest plot)
    - Panel B: Hotelling's T² statistics
    - Panel C: Variance explained by PC1 and PC2
    - Panel D: p-value comparison (bootstrap vs permutation)
    
    Args:
        pca_results_by_target: Dict mapping target population names to their
            PCA analysis results from analyze_input_patterns_pca()
        stimulated_population: Name of stimulated population
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> pca_by_target = {
        ...     'gc': analyze_input_patterns_pca(..., post_population='gc'),
        ...     'mc': analyze_input_patterns_pca(..., post_population='mc'),
        ...     'pv': analyze_input_patterns_pca(..., post_population='pv'),
        ...     'sst': analyze_input_patterns_pca(..., post_population='sst')
        ... }
        >>> fig = plot_pca_summary_all_targets(pca_by_target, 'sst')
    """
    # Extract data from all targets
    data_rows = []
    target_order = {'gc': 0, 'mc': 1, 'pv': 2, 'sst': 3}
    
    for target_pop in sorted(pca_results_by_target.keys(), key=lambda x: target_order.get(x, 99)):
        pca_res = pca_results_by_target[target_pop]
        
        bootstrap_result = pca_res['bootstrap_result']
        permutation_result = pca_res['permutation_result']
        
        # Get mean variance explained across connectivity instances
        pca_results = pca_res['pca_results']
        mean_var_pc1 = np.mean([r.explained_variance_ratio[0] for r in pca_results.values()])
        mean_var_pc2 = np.mean([r.explained_variance_ratio[1] for r in pca_results.values()]) if len(pca_results.values()) > 0 else 0.0
        
        data_rows.append({
            'target': target_pop,
            'label': f'{stimulated_population.upper()} $\\rightarrow$ {target_pop.upper()}',
            # Effect sizes
            'd_pc1': bootstrap_result.observed_d_pc1,
            'd_pc1_ci_lower': bootstrap_result.ci_lower_d_pc1,
            'd_pc1_ci_upper': bootstrap_result.ci_upper_d_pc1,
            'd_pc2': bootstrap_result.observed_d_pc2,
            'd_pc2_ci_lower': bootstrap_result.ci_lower_d_pc2,
            'd_pc2_ci_upper': bootstrap_result.ci_upper_d_pc2,
            # T² statistic
            't2': bootstrap_result.observed_t2,
            't2_ci_lower': bootstrap_result.ci_lower_t2,
            't2_ci_upper': bootstrap_result.ci_upper_t2,
            # P-values
            'p_bootstrap': bootstrap_result.p_value_bootstrap,
            'p_permutation': permutation_result['p_value_permutation'],
            'p_hotelling': np.mean([s.p_value_hotelling for s in pca_res['separation_stats'].values()]),
            # Variance explained
            'var_pc1': mean_var_pc1,
            'var_pc2': mean_var_pc2,
            # Sample size
            'n_connectivity': pca_res['metadata']['n_connectivity_instances']
        })
    
    if len(data_rows) == 0:
        print("Warning: No valid PCA results to plot")
        return None
    
    # Create figure with 4 panels
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Color scheme
    def get_color(p_val):
        if p_val < 0.001:
            return '#e74c3c'  # Red
        elif p_val < 0.01:
            return '#e67e22'  # Dark orange
        elif p_val < 0.05:
            return '#f39c12'  # Orange
        else:
            return '#95a5a6'  # Gray
    
    def get_stars(p_val):
        if p_val < 0.001:
            return '***'
        elif p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        else:
            return 'n.s.'
    
    # ========================================================================
    # Panel A: Effect Sizes (Cohen's d) Forest Plot
    # ========================================================================
    ax_forest = fig.add_subplot(gs[0, :])  # Top row, both columns
    
    # Create y-positions (2 rows per target: PC1 and PC2)
    n_targets = len(data_rows)
    y_positions = []
    y_labels = []
    
    for i, row_data in enumerate(data_rows):
        base_y = i * 3  # Space between targets
        y_positions.extend([base_y, base_y + 1])
        y_labels.extend([
            f"{row_data['label']}\nPC1",
            f"PC2"
        ])
    
    # Plot PC1 and PC2 for each target
    for i, row_data in enumerate(data_rows):
        base_y = i * 3
        
        # PC1
        p_val = row_data['p_bootstrap']
        color = get_color(p_val)
        
        ax_forest.plot(row_data['d_pc1'], base_y, 'o',
                      color=color, markersize=10, zorder=3)
        ax_forest.plot([row_data['d_pc1_ci_lower'], row_data['d_pc1_ci_upper']], 
                      [base_y, base_y],
                      '-', color=color, linewidth=2.5, zorder=2)
        
        # PC2
        ax_forest.plot(row_data['d_pc2'], base_y + 1, 's',
                      color=color, markersize=8, zorder=3, alpha=0.7)
        ax_forest.plot([row_data['d_pc2_ci_lower'], row_data['d_pc2_ci_upper']], 
                      [base_y + 1, base_y + 1],
                      '-', color=color, linewidth=2, zorder=2, alpha=0.7)
        
        # Add annotation
        stars = get_stars(p_val)
        ax_forest.text(0.95, base_y + 0.5,
                      f"d$_{{PC1}}$={row_data['d_pc1']:.2f}, d$_{{PC2}}$={row_data['d_pc2']:.2f} {stars}",
                      va='center', ha='right', fontsize=9,
                      transform=ax_forest.get_yaxis_transform())
    
    # Reference lines
    ax_forest.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax_forest.axvline(0.5, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax_forest.axvline(-0.5, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax_forest.text(0.5, 0.02, 'Medium', fontsize=8, alpha=0.5, ha='center',
                  transform=ax_forest.get_xaxis_transform())
    
    ax_forest.set_yticks(y_positions)
    ax_forest.set_yticklabels(y_labels, fontsize=9)
    ax_forest.set_xlabel("Cohen's d (Excited vs Suppressed in PC Space)", fontsize=11, fontweight='bold')
    ax_forest.set_title(f'PCA Separation Effect Sizes: {stimulated_population.upper()} Stimulation',
                       fontsize=12, fontweight='bold')
    ax_forest.grid(True, alpha=0.3, axis='x')
    
    # Legend for PC1 vs PC2
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=8, label='PC1', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   markersize=7, label='PC2', linestyle='None')
    ]
    ax_forest.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # ========================================================================
    # Panel B: Hotelling's T^2 Statistics
    # ========================================================================
    ax_t2 = fig.add_subplot(gs[1, 0])  # Bottom left
    
    x_pos = np.arange(n_targets)
    t2_values = [row['t2'] for row in data_rows]
    t2_ci_lower = [row['t2_ci_lower'] for row in data_rows]
    t2_ci_upper = [row['t2_ci_upper'] for row in data_rows]
    p_values = [row['p_bootstrap'] for row in data_rows]
    
    colors = [get_color(p) for p in p_values]
    
    ax_t2.bar(x_pos, t2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax_t2.errorbar(x_pos, t2_values, 
                   yerr=[np.array(t2_values) - np.array(t2_ci_lower),
                         np.array(t2_ci_upper) - np.array(t2_values)],
                   fmt='none', ecolor='black', capsize=5, linewidth=2)
    
    # Add significance markers
    for i, (t2, p_val) in enumerate(zip(t2_values, p_values)):
        stars = get_stars(p_val)
        ax_t2.text(i, t2 + (t2_ci_upper[i] - t2) * 1.1, stars,
                  ha='center', fontsize=11, fontweight='bold')
    
    ax_t2.set_xticks(x_pos)
    ax_t2.set_xticklabels([row['target'].upper() for row in data_rows], fontsize=10)
    ax_t2.set_ylabel("Hotelling's $T^2$", fontsize=11, fontweight='bold')
    ax_t2.set_xlabel('Target Population', fontsize=11)
    ax_t2.set_title('Multivariate Separation Statistics', fontsize=11, fontweight='bold')
    ax_t2.grid(True, alpha=0.3, axis='y')
    ax_t2.set_axisbelow(True)
    
    # ========================================================================
    # Panel C: Variance Explained
    # ========================================================================
    ax_var = fig.add_subplot(gs[1, 1])  # Bottom right
    
    var_pc1 = [row['var_pc1'] * 100 for row in data_rows]  # Convert to percentage
    var_pc2 = [row['var_pc2'] * 100 for row in data_rows]
    
    width = 0.35
    ax_var.bar(x_pos - width/2, var_pc1, width, label='PC1',
              color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax_var.bar(x_pos + width/2, var_pc2, width, label='PC2',
              color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(var_pc1, var_pc2)):
        ax_var.text(i - width/2, v1 + 1, f'{v1:.1f}%',
                   ha='center', va='bottom', fontsize=8)
        ax_var.text(i + width/2, v2 + 1, f'{v2:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    ax_var.set_xticks(x_pos)
    ax_var.set_xticklabels([row['target'].upper() for row in data_rows], fontsize=10)
    ax_var.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
    ax_var.set_xlabel('Target Population', fontsize=11)
    ax_var.set_title('Variance Captured by Principal Components', fontsize=11, fontweight='bold')
    ax_var.legend(fontsize=9, loc='upper right')
    ax_var.grid(True, alpha=0.3, axis='y')
    ax_var.set_axisbelow(True)
    ax_var.set_ylim([0, max(max(var_pc1), max(var_pc2)) * 1.2])
    
    # Overall title and layout
    sample_sizes = set(row['n_connectivity'] for row in data_rows)
    if len(sample_sizes) == 1:
        n_conn_text = f"N = {sample_sizes.pop()} connectivity instances"
    else:
        n_conn_text = f"N = {min(sample_sizes)}-{max(sample_sizes)} connectivity instances"
    
    fig.suptitle(f'PCA Multivariate Pattern Analysis: {stimulated_population.upper()} Stimulation Across All Targets\n' +
                f'{n_conn_text}',
                fontsize=14, fontweight='bold')
    
    # Add legend for significance
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='p < 0.001 (***)'),
        mpatches.Patch(color='#e67e22', label='p < 0.01 (**)'),
        mpatches.Patch(color='#f39c12', label='p < 0.05 (*)'),
        mpatches.Patch(color='#95a5a6', label='n.s.')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=9, frameon=True, title='Significance Level')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA summary plot to: {save_path}")
    
    return fig


# ============================================================================
# High-Level Workflow Function
# ============================================================================

def analyze_input_weight_patterns_pca(
    nested_results,
    circuit,
    target_population: str,
    post_population: str,
    source_populations: List[str],
    stim_start: float,
    stim_duration: float,
    warmup: float,
    n_bootstrap: int = 10000,
    n_permutations: int = 1000,
    n_components: int = None,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    save_dir: Optional[str] = None,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Complete workflow for PCA analysis of input weight patterns
    
    Args:
        nested_results: Nested experiment results (list of NestedTrialResult)
        circuit: DentateCircuit instance
        target_population: Stimulated population
        post_population: Post-synaptic population to analyze
        source_populations: List of source populations
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline period (ms)
        n_bootstrap: Number of bootstrap samples
        n_permutations: Number of permutations for null distribution
        n_components: Number of PCs to retain (None = auto)
        threshold_std: Classification threshold
        expression_threshold: Opsin expression threshold
        save_dir: Directory to save results
        random_seed: Random seed for reproducibility
        
    Returns:
        Complete results dictionary
    """
    print("\n" + "="*80)
    print("PCA ANALYSIS OF INPUT WEIGHT PATTERNS")
    print("="*80)
    print(f"\nTarget: {target_population.upper()} $\\rightarrow$ {post_population.upper()}")
    print(f"Sources: {', '.join(s.upper() for s in source_populations)}")
    
    # Import here to avoid circular dependency
    from nested_effect_size import (organize_nested_trials,
                                   classify_cells_by_connectivity)
    
    # Organize trials by connectivity
    print("\nOrganizing nested trials...")
    trials_by_connectivity = organize_nested_trials(nested_results)
    print(f"   Found {len(trials_by_connectivity)} connectivity instances")
    
    # Classify cells within each connectivity
    print("\nClassifying cells as excited/suppressed...")
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
    
    for conn_idx, clf in classifications.items():
        print(f"   Conn {conn_idx}: {clf['n_excited']} excited, {clf['n_suppressed']} suppressed")
    
    # Extract input weight matrices
    print("\nExtracting input weight matrices...")
    weight_matrices = extract_input_weight_matrix_by_connectivity(
        circuit,
        classifications,
        post_population,
        source_populations,
        normalize=True
    )
    
    # Perform PCA
    print("\nPerforming PCA within each connectivity...")
    pca_results = perform_pca_by_connectivity(
        weight_matrices,
        n_components=n_components
    )
    
    # Print variance explained
    for conn_idx, pca_result in pca_results.items():
        var_ratio = pca_result.explained_variance_ratio
        print(f"   Conn {conn_idx}: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%")
    
    # Compute separation statistics
    print("\nComputing separation statistics...")
    separation_stats = {
        conn_idx: compute_separation_statistics(pca_result)
        for conn_idx, pca_result in pca_results.items()
    }
    
    for conn_idx, stats in separation_stats.items():
        print(f"   Conn {conn_idx}: $T^2$={stats.hotellings_t2:.2f}, p={stats.p_value_hotelling:.4f}")
    
    # Bootstrap analysis
    print(f"\nBootstrap analysis ({n_bootstrap} samples)...")
    bootstrap_result = bootstrap_pca_separation(
        pca_results,
        separation_stats,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed
    )
    
    print(f"   Observed $T^2$: {bootstrap_result.observed_t2:.2f} " +
          f"[{bootstrap_result.ci_lower_t2:.2f}, {bootstrap_result.ci_upper_t2:.2f}]")
    print(f"   PC1 Cohen's d: {bootstrap_result.observed_d_pc1:.2f} " +
          f"[{bootstrap_result.ci_lower_d_pc1:.2f}, {bootstrap_result.ci_upper_d_pc1:.2f}]")
    print(f"   PC2 Cohen's d: {bootstrap_result.observed_d_pc2:.2f} " +
          f"[{bootstrap_result.ci_lower_d_pc2:.2f}, {bootstrap_result.ci_upper_d_pc2:.2f}]")
    
    # Permutation test
    print(f"\nPermutation test ({n_permutations} permutations)...")
    permutation_result = permutation_test_pca_separation(
        weight_matrices,
        n_permutations=n_permutations,
        random_seed=random_seed
    )
    
    print(f"   Observed vs Null $T^2$: {permutation_result['observed_t2']:.2f} vs " +
          f"{permutation_result['null_t2_mean']:.2f} $\\pm$ {permutation_result['null_t2_std']:.2f}")
    print(f"   Permutation p-value: {permutation_result['p_value_permutation']:.4f}")
    
    # Generate visualizations
    if save_dir:
        print("\nGenerating visualizations...")
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Scatter plots
        fig1 = plot_pca_scatter_by_connectivity(
            pca_results,
            weight_matrices,
            separation_stats,
            target_population,
            post_population,
            save_path=str(save_path / f'pca_scatter_{target_population}_{post_population}.pdf')
        )
        plt.close(fig1)
        
        # Loadings
        fig2 = plot_pca_loadings_by_connectivity(
            pca_results,
            weight_matrices,
            save_path=str(save_path / f'pca_loadings_{target_population}_{post_population}.pdf')
        )
        plt.close(fig2)
        
        # Forest plot
        fig3 = plot_pca_separation_forest(
            bootstrap_result,
            target_population,
            post_population,
            save_path=str(save_path / f'pca_forest_{target_population}_{post_population}.pdf')
        )
        plt.close(fig3)
        
        # Variance explained
        fig4 = plot_variance_explained(
            pca_results,
            save_path=str(save_path / f'pca_variance_{target_population}_{post_population}.pdf')
        )
        plt.close(fig4)
        
        print(f"   Saved plots to: {save_path}")
    
    complete_results = {
        'weight_matrices': weight_matrices,
        'pca_results': pca_results,
        'separation_stats': separation_stats,
        'bootstrap_result': bootstrap_result,
        'permutation_result': permutation_result,
        'classifications': classifications,
        'metadata': {
            'target_population': target_population,
            'post_population': post_population,
            'source_populations': source_populations,
            'n_connectivity_instances': len(pca_results),
            'n_components': n_components,
            'n_bootstrap': n_bootstrap,
            'n_permutations': n_permutations
        }
    }
    
    print("\n" + "="*80)
    print("PCA ANALYSIS COMPLETE")
    print("="*80)
    
    return complete_results
