import sys
import math
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, NamedTuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pathlib import Path
import tqdm
import logging
import h5py


logger = logging.getLogger('DG_protocol')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

from dendritic_somatic_transfer import (
    get_default_device
)
from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit,
    CircuitParams,
    PerConnectionSynapticParams,
    OpsinParams
)
from optogenetic_experiment import (
    OptogeneticExperiment
    )
from ablation_tests import (
    run_all_ablation_tests,
    _analyze_ablation_trial,
    _aggregate_paired_ablation_results
)
from nested_experiment import (
    NestedExperimentConfig,
    run_nested_comparative_experiment,
    aggregate_nested_results,
    compute_variance_decomposition,
    classify_mechanism_regime,
    save_nested_experiment_results,
    print_nested_experiment_summary,
    save_nested_experiment_summary,
    plot_variance_decomposition,
    plot_connectivity_instance_variance,
    plot_connectivity_instance_variance_detailed,
)
from nested_effect_size import (
    analyze_effect_size_all_sources_nested,
    plot_effect_sizes_forest,
    plot_bootstrap_distributions,
    plot_weight_distributions_by_response,
    print_nested_effect_size_analysis_summary
)
from effect_size_decision_framework import (
    PowerAnalysisResult,
    PrecisionAssessment,
    BiologicalSignificanceAssessment,
    DataCollectionRecommendation,
    compute_power_one_sample,
    estimate_required_n,
    analyze_statistical_power,
    assess_ci_precision,
    assess_biological_significance,
    make_data_collection_decision,
    print_effect_size_decision_report,
    plot_effect_size_decision_summary,
    run_effect_size_decision_analysis
)

from DG_visualization import (
    DGCircuitVisualization
)
from statistical_testing_weights import (analyze_weights_by_average_response,
                                         plot_weights_by_average_response)
from gradient_adaptive_stepper import (
    GradientAdaptiveStepConfig,
    GradientAdaptiveStepper,
)

# ============================================================================
# Random Seed Management (consistent with optimization module)
# ============================================================================

def set_random_seed(seed: int, device: Optional[torch.device] = None):
    """
    Set random seeds for reproducible connectivity generation
    
    Args:
        seed: Random seed value
        device: Device to set CUDA seed for (if applicable)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device is not None and device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    

def analyze_connectivity_patterns(experiment: OptogeneticExperiment) -> Dict:
    """
    Analyze the anatomical connectivity patterns including MEC
    
    Note: Uses connectivity from current circuit instance
    """
    layout = experiment.circuit.layout
    conductance_matrices = experiment.circuit.connectivity.conductance_matrices
    
    analysis = {}
    
    # Helper function to get connectivity matrix by name
    def get_connectivity(conn_name: str) -> torch.Tensor:
        """Get connectivity matrix for a specific connection"""
        if conn_name in conductance_matrices:
            return conductance_matrices[conn_name].connectivity
        else:
            return torch.zeros((1, 1), device=experiment.device)
    
    # Analyze GC connections (should be local)
    gc_mc_distances = []
    gc_pv_distances = []
    gc_sst_distances = []
    
    gc_mc_conn = get_connectivity('gc_mc')
    gc_pv_conn = get_connectivity('gc_pv')
    gc_sst_conn = get_connectivity('gc_sst')
    
    for i in range(experiment.circuit_params.n_gc):
        # Connected MCs
        if i < gc_mc_conn.size(0):
            connected_mc = torch.where(gc_mc_conn[i] > 0)[0]
            if len(connected_mc) > 0:
                gc_pos = layout.positions['gc'][i:i+1]
                mc_pos = layout.positions['mc'][connected_mc]
                distances = torch.norm(gc_pos - mc_pos, dim=1)
                gc_mc_distances.extend(distances.cpu().tolist())
        
        # Connected PVs
        if i < gc_pv_conn.size(0):
            connected_pv = torch.where(gc_pv_conn[i] > 0)[0]
            if len(connected_pv) > 0:
                gc_pos = layout.positions['gc'][i:i+1]
                pv_pos = layout.positions['pv'][connected_pv]
                distances = torch.norm(gc_pos - pv_pos, dim=1)
                gc_pv_distances.extend(distances.cpu().tolist())
            
        # Connected SSTs
        if i < gc_sst_conn.size(0):
            connected_sst = torch.where(gc_sst_conn[i] > 0)[0]
            if len(connected_sst) > 0:
                gc_pos = layout.positions['gc'][i:i+1]
                sst_pos = layout.positions['sst'][connected_sst]
                distances = torch.norm(gc_pos - sst_pos, dim=1)
                gc_sst_distances.extend(distances.cpu().tolist())
    
    # Analyze MC connections (should include distant)
    mc_gc_distances = []
    mc_sst_distances = []
    
    mc_gc_conn = get_connectivity('mc_gc')
    mc_sst_conn = get_connectivity('mc_sst')
    
    for i in range(experiment.circuit_params.n_mc):
        # Connected GCs
        if i < mc_gc_conn.size(0):
            connected_gc = torch.where(mc_gc_conn[i] > 0)[0]
            if len(connected_gc) > 0:
                mc_pos = layout.positions['mc'][i:i+1]
                gc_pos = layout.positions['gc'][connected_gc]
                distances = torch.norm(mc_pos - gc_pos, dim=1)
                mc_gc_distances.extend(distances.cpu().tolist())
        
        # Connected SSTs
        if i < mc_sst_conn.size(0):
            connected_sst = torch.where(mc_sst_conn[i] > 0)[0]
            if len(connected_sst) > 0:
                mc_pos = layout.positions['mc'][i:i+1]
                sst_pos = layout.positions['sst'][connected_sst]
                distances = torch.norm(mc_pos - sst_pos, dim=1)
                mc_sst_distances.extend(distances.cpu().tolist())
    
    # Analyze MEC connections (asymmetry analysis)
    mec_pv_conn = get_connectivity('mec_pv')
    mec_gc_conn = get_connectivity('mec_gc')
    mec_mc_conn = get_connectivity('mec_mc')
    mec_sst_conn = get_connectivity('mec_sst')
    
    mec_pv_connections = torch.sum(mec_pv_conn).item()
    mec_gc_connections = torch.sum(mec_gc_conn).item()
    mec_mc_connections = torch.sum(mec_mc_conn).item()
    mec_sst_connections = torch.sum(mec_sst_conn).item()
    
    analysis = {
        'gc_mc_distances': gc_mc_distances,
        'gc_pv_distances': gc_pv_distances,
        'gc_sst_distances': gc_sst_distances,
        'mc_gc_distances': mc_gc_distances,
        'mc_sst_distances': mc_sst_distances,
        'local_radius': experiment.circuit_params.local_radius,
        'distant_min': experiment.circuit_params.distant_min,
        'mec_connectivity': {
            'mec_to_pv': mec_pv_connections,
            'mec_to_gc': mec_gc_connections,
            'mec_to_mc': mec_mc_connections,
            'mec_to_sst': mec_sst_connections,
            'pv_fraction': mec_pv_connections / (experiment.circuit_params.n_mec * experiment.circuit_params.n_pv) if experiment.circuit_params.n_mec > 0 and experiment.circuit_params.n_pv > 0 else 0.0,
            'gc_fraction': mec_gc_connections / (experiment.circuit_params.n_mec * experiment.circuit_params.n_gc) if experiment.circuit_params.n_mec > 0 and experiment.circuit_params.n_gc > 0 else 0.0
        }
    }
    
    return analysis


def analyze_conductance_patterns(experiment: OptogeneticExperiment) -> Dict:
    """Analyze the per-connection conductance patterns"""
    conductance_matrices = experiment.circuit.connectivity.conductance_matrices
    
    analysis = {}
    
    for conn_name, cond_matrix in conductance_matrices.items():
        # Get conductances for existing connections only
        existing_connections = cond_matrix.connectivity > 0
        active_conductances = cond_matrix.conductances[existing_connections]
        
        if len(active_conductances) > 0:
            analysis[conn_name] = {
                'mean_conductance': float(torch.mean(active_conductances)),
                'std_conductance': float(torch.std(active_conductances)),
                'min_conductance': float(torch.min(active_conductances)),
                'max_conductance': float(torch.max(active_conductances)),
                'n_connections': len(active_conductances),
                'synapse_type': cond_matrix.synapse_type,
                'cv_conductance': float(torch.std(active_conductances) / torch.mean(active_conductances)) if torch.mean(active_conductances) > 0 else 0.0
            }
        else:
            analysis[conn_name] = {
                'mean_conductance': 0.0, 
                'std_conductance': 0.0, 
                'min_conductance': 0.0, 
                'max_conductance': 0.0,
                'n_connections': 0,
                'synapse_type': cond_matrix.synapse_type,
                'cv_conductance': 0.0
            }
    
    return analysis

def run_comparative_experiment(optimization_json_file: Optional[str] = None,
                               intensities: List[float] = [0.5, 1.0, 2.0],
                               mec_current: float = 100.0,
                               opsin_current: float = 100.0,
                               stim_start: float = 1500.0,
                               stim_duration: float = 1000.0,
                               warmup: float = 500.0,
                               plot_activity: bool = True,
                               plot_baseline_normalize: bool = False,
                               device: Optional[torch.device] = None,
                               n_trials: int = 1,
                               regenerate_connectivity_per_trial: bool = False,
                               base_seed: int = 42,
                               load_results_file: Optional[str] = None,
                               save_results_file: Optional[str] = None,
                               auto_save: bool = True,
                               save_full_activity: bool = False,
                               record_currents: bool = False,
                               adaptive_step: bool = True,
                               adaptive_config: Optional[GradientAdaptiveStepConfig] = None,
                               use_time_varying_mec: bool = False,
                               mec_pattern_type: str = 'oscillatory',
                               mec_theta_freq: float = 5.0,
                               mec_theta_amplitude: float = 0.3,
                               mec_gamma_freq: float = 20.0,
                               mec_gamma_amplitude: float = 0.15,
                               mec_gamma_coupling_strength: float = 0.8,
                               mec_gamma_preferred_phase: float = 0.0,
                               mec_drift_timescale: float = 200.0,
                               mec_drift_amplitude: float = 0.1,
                               mec_rotation_groups: int = 3):
    """
    Compare PV vs SST stimulation with anatomical connectivity
    
    Args:
        optimization_json_file: Path to optimization results (optional)
        intensities: List of light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: When to start stimulation (ms)
        stim_duration: Duration of stimulation (ms)
        warmup: Pre-stimulation period (ms)
        plot_activity: Whether to plot activity traces
        device: Device to run on (None for auto-detect)
        n_trials: Number of trials to average (default: 1)
        base_seed: Base random seed for reproducibility (default: 42)
        load_results_file: If provided, load results from this file instead of running
        save_results_file: If provided, save results to this file
        auto_save: If True, automatically save results with default filename (default: True)
        save_full_activity: If True, save complete activity traces (larger files, default: False)
        record_currents: If True, record and save (if save_full_activity specified) currents

    Returns:
        results: Dict with averaged results
        connectivity_analysis: Connectivity pattern analysis
        conductance_analysis: Conductance pattern analysis
    """

    # Create adaptive config if using adaptive stepping
    if adaptive_step and adaptive_config is None:
        adaptive_config = GradientAdaptiveStepConfig(
            dt_min=0.05,
            dt_max=0.3,
            gradient_low=0.5,
            gradient_high=10.0,
        )
    
    # Check if we should load from file
    if load_results_file is not None:
        logger.info(f"\nLoading experiment results from file: {load_results_file}")
        results, conn_analysis, conductance_analysis, metadata = load_experiment_results(load_results_file)
        
        # Print loaded configuration
        logger.info("\nLoaded experiment configuration:")
        logger.info(f"  Light intensities: {metadata.get('intensities', 'Unknown')}")
        logger.info(f"  MEC current: {metadata.get('mec_current', 'Unknown')} pA")
        logger.info(f"  Opsin current: {metadata.get('opsin_current', 'Unknown')} pA")
        logger.info(f"  Stimulation: {metadata.get('stim_start', 'Unknown')} ms start, "
              f"{metadata.get('stim_duration', 'Unknown')} ms duration")
        
        return results, conn_analysis, conductance_analysis
    
    # Otherwise, run the experiment
    if device is None:
        device = get_default_device()
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()

    # Configure current recording
    current_config = CurrentRecordingConfig(
        enabled=True,
        record_by_source=True,
        record_by_type=True,
        populations=['gc', 'mc', 'pv', 'sst'],
        downsample_factor=10
    )

    experiment = OptogeneticExperiment(
        circuit_params, synaptic_params, opsin_params, 
        optimization_json_file=optimization_json_file,
        device=device,
        base_seed=base_seed,
        adaptive_config=adaptive_config,
        use_time_varying_mec=use_time_varying_mec,
        mec_pattern_type=mec_pattern_type,
        mec_theta_freq=mec_theta_freq,
        mec_theta_amplitude=mec_theta_amplitude,
        mec_gamma_freq=mec_gamma_freq,
        mec_gamma_amplitude=mec_gamma_amplitude,
        mec_gamma_coupling_strength=mec_gamma_coupling_strength,
        mec_gamma_preferred_phase=mec_gamma_preferred_phase,
        mec_drift_timescale=mec_drift_timescale,
        mec_drift_amplitude=mec_drift_amplitude,
        mec_rotation_groups=mec_rotation_groups,
        record_currents=record_currents,
        current_recording_config=current_config
    )
    
    logger.info(f"\nRunning comparative experiment with {n_trials} trial(s) per condition")
    logger.info(f"Base seed: {base_seed}\n")
    
    # Analyze connectivity patterns (from last trial's circuit)
    conn_analysis = analyze_connectivity_patterns(experiment)
    
    # Analyze conductance patterns
    conductance_analysis = analyze_conductance_patterns(experiment)
    
    logger.info("Connectivity Analysis:")
    logger.info(f"GC->MC distances (mean): {np.mean(conn_analysis['gc_mc_distances']):.3f} mm")
    logger.info(f"GC->PV distances (mean): {np.mean(conn_analysis['gc_pv_distances']):.3f} mm")
    logger.info(f"GC->SST distances (mean): {np.mean(conn_analysis['gc_sst_distances']):.3f} mm")
    logger.info(f"MC->GC distances (mean): {np.mean(conn_analysis['mc_gc_distances']):.3f} mm")
    logger.info(f"MC->SST distances (mean): {np.mean(conn_analysis['mc_sst_distances']):.3f} mm")
    logger.info(f"Local radius threshold: {conn_analysis['local_radius']} mm")
    logger.info(f"Distant minimum threshold: {conn_analysis['distant_min']} mm")

    logger.info("\nConductance Analysis:")
    logger.info("-" * 50)
    for conn_name, stats in conductance_analysis.items():
        if stats['n_connections'] > 0:
            logger.info(f"{conn_name} ({stats['synapse_type']}):")
            logger.info(f"  Connections: {stats['n_connections']}")
            logger.info(f"  Conductance: {stats['mean_conductance']:.3f} +/- {stats['std_conductance']:.3f} nS")
            logger.info(f"  CV: {stats['cv_conductance']:.2f}")
            logger.info(f"  Range: [{stats['min_conductance']:.3f}, {stats['max_conductance']:.3f}] nS")
    
    # Test different stimulation intensities
    results = {}

    if stim_start < warmup:
        stim_start = warmup
    
    for target in ['pv', 'sst']:
        results[target] = {}
        logger.info(f"\nTesting {target.upper()} stimulation...")
        
        for intensity in intensities:
            logger.info(f"\n  Intensity: {intensity}")
            
            # Run multi-trial stimulation
            result = experiment.simulate_stimulation(
                target, intensity,
                stim_start=stim_start,
                stim_duration=stim_duration,
                plot_activity=plot_activity,
                plot_aggregated=True,
                plot_baseline_normalize=plot_baseline_normalize,
                mec_current=mec_current,
                opsin_current=opsin_current,
                n_trials=n_trials,
                regenerate_connectivity_per_trial=regenerate_connectivity_per_trial,
                adaptive_step=adaptive_step
            )
            if adaptive_step and 'adaptive_stats' in result:
                adaptive_stats = result['adaptive_stats']
    
                # For multi-trial: show mean +/- std
                if n_trials > 1:
                    logger.info(f"  Steps: {adaptive_stats['n_steps_mean']:.0f} +/- {adaptive_stats['n_steps_std']:.0f} "
                          f"(range: {adaptive_stats['n_steps_min']}-{adaptive_stats['n_steps_max']})")
                    logger.info(f"  Avg dt: {adaptive_stats['avg_dt_mean']:.3f} +/- {adaptive_stats['avg_dt_std']:.3f} ms")
                    logger.info(f"  dt range: [{adaptive_stats['min_dt_mean']:.3f}, {adaptive_stats['max_dt_mean']:.3f}] ms")
                # For single trial: show direct values
                else:
                    logger.info(f"  Steps: {adaptive_stats['n_steps_mean']:.0f} (avg dt: {adaptive_stats['avg_dt_mean']:.3f} ms)")
                    logger.info(f"  dt range: [{adaptive_stats['min_dt_min']:.3f}, {adaptive_stats['max_dt_max']:.3f}] ms")


                
            # Analyze network effects (using mean across trials)
            time = result['time']
            activity_mean = result['activity_trace_mean']
            activity_std = result['activity_trace_std']
            opsin_expression_mean = result['opsin_expression_mean']

            # Extract currents
            recorded_currents = result.get('recorded_currents', None)
            
            baseline_mask = (time >= warmup) & (time < stim_start)
            stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
            
            analysis = {}
            analysis['opsin_expression_mean'] = opsin_expression_mean
            analysis['n_trials'] = n_trials

            if save_full_activity and 'trial_results' in result:
                analysis['trial_results'] = result['trial_results']

            if recorded_currents is not None:
                analysis['recorded_currents'] = recorded_currents

            # Store adaptive stats if present
            if 'adaptive_stats' in result:
                analysis['adaptive_stats'] = result['adaptive_stats']
            
            if save_full_activity:
                analysis['time'] = time
                analysis['activity_trace_mean'] = activity_mean
                analysis['activity_trace_std'] = activity_std
                analysis['opsin_expression_std'] = result['opsin_expression_std']
                
            for pop in ['gc', 'mc', 'pv', 'sst']:
                baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                
                if pop == target:
                    # For target population, analyze both all cells and non-expressing cells

                    # Store all-cell metrics for target (includes directly stimulated cells)
                    analysis[f'{pop}_stim_rates_mean'] = stim_rate.numpy()
                    analysis[f'{pop}_baseline_rates_mean'] = baseline_rate.numpy()

                    # Calculate trial-to-trial variability for all cells
                    stim_rates_all_trials = []
                    baseline_rates_all_trials = []
                    for trial_result in result['trial_results']:
                        trial_activity = trial_result['activity_trace'][pop]
                        trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                        trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                        baseline_rates_all_trials.append(trial_baseline.numpy())
                        stim_rates_all_trials.append(trial_stim.numpy())

                    analysis[f'{pop}_baseline_rates_std'] = np.std(baseline_rates_all_trials, axis=0)
                    analysis[f'{pop}_stim_rates_std'] = np.std(stim_rates_all_trials, axis=0)

                    # Analyze non-expressing cells separately
                    # Get non-stimulated indices from first trial (same across trials if not regenerating opsin)
                    non_stimulated_indices = result['trial_results'][0]['non_stimulated_indices']
                    n_non_expressing = len(non_stimulated_indices)

                    if n_non_expressing > 0:
                        # Analyze only non-expressing cells
                        rate_change = stim_rate - baseline_rate
                        baseline_std = torch.std(baseline_rate)

                        # Get responses for non-expressing cells
                        baseline_rate_nonexpr = baseline_rate[non_stimulated_indices]
                        stim_rate_nonexpr = stim_rate[non_stimulated_indices]
                        rate_change_nonexpr = rate_change[non_stimulated_indices]
                        baseline_std_nonexpr = torch.std(baseline_rate_nonexpr)

                        # Compute excited/inhibited fractions for non-expressing cells
                        excited_fraction_nonexpr = torch.mean(
                            (rate_change_nonexpr > baseline_std_nonexpr).float()
                        )
                        inhibited_fraction_nonexpr = torch.mean(
                            (rate_change_nonexpr < -baseline_std_nonexpr).float()
                        )

                        analysis[f'{pop}_nonexpr_excited'] = excited_fraction_nonexpr.item()
                        analysis[f'{pop}_nonexpr_inhibited'] = inhibited_fraction_nonexpr.item()
                        analysis[f'{pop}_nonexpr_mean_change'] = torch.mean(rate_change_nonexpr).item()
                        analysis[f'{pop}_nonexpr_mean_stim_rate'] = torch.mean(stim_rate_nonexpr).item()
                        analysis[f'{pop}_nonexpr_mean_baseline_rate'] = torch.mean(baseline_rate_nonexpr).item()
                        analysis[f'{pop}_nonexpr_count'] = n_non_expressing

                        # Calculate trial-to-trial variability for non-expressing cells
                        excited_fractions_nonexpr_all = []
                        mean_changes_nonexpr_all = []

                        for trial_result in result['trial_results']:
                            trial_activity = trial_result['activity_trace'][pop]
                            trial_non_stim_idx = trial_result['non_stimulated_indices']

                            if len(trial_non_stim_idx) > 0:
                                trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                                trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                                trial_change = trial_stim - trial_baseline
                                trial_baseline_std = torch.std(trial_baseline)

                                # Non-expressing cells only
                                trial_change_nonexpr = trial_change[trial_non_stim_idx]
                                trial_baseline_std_nonexpr = torch.std(trial_baseline[trial_non_stim_idx])

                                trial_excited_nonexpr = torch.mean(
                                    (trial_change_nonexpr > trial_baseline_std_nonexpr).float()
                                ).item()
                                excited_fractions_nonexpr_all.append(trial_excited_nonexpr)
                                mean_changes_nonexpr_all.append(torch.mean(trial_change_nonexpr).item())

                        analysis[f'{pop}_nonexpr_excited_std'] = np.std(excited_fractions_nonexpr_all)
                        analysis[f'{pop}_nonexpr_mean_change_std'] = np.std(mean_changes_nonexpr_all)
                    else:
                        # All cells express opsin - no non-expressing cells
                        analysis[f'{pop}_nonexpr_excited'] = 0.0
                        analysis[f'{pop}_nonexpr_inhibited'] = 0.0
                        analysis[f'{pop}_nonexpr_mean_change'] = 0.0
                        analysis[f'{pop}_nonexpr_count'] = 0
                        if n_trials > 1:
                            analysis[f'{pop}_nonexpr_excited_std'] = 0.0
                            analysis[f'{pop}_nonexpr_mean_change_std'] = 0.0

                    # Skip standard analysis for target population
                    continue                
                
                # Standard analysis for NON-TARGET populations
                rate_change = stim_rate - baseline_rate
                baseline_std = torch.std(baseline_rate)
                
                excited_fraction = torch.mean((rate_change > baseline_std).float())
                inhibited_fraction = torch.mean((rate_change < -baseline_std).float())

                analysis[f'{pop}_excited'] = excited_fraction.item()
                analysis[f'{pop}_inhibited'] = inhibited_fraction.item()
                analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()
                analysis[f'{pop}_mean_stim_rate'] = torch.mean(stim_rate).item()
                analysis[f'{pop}_mean_baseline_rate'] = torch.mean(baseline_rate).item()
                
                # Calculate statistics across trials
                excited_fractions_all = []
                mean_changes_all = []
                
                for trial_result in result['trial_results']:
                    trial_activity = trial_result['activity_trace'][pop]
                    trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                    trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                    trial_change = trial_stim - trial_baseline
                    trial_baseline_std = torch.std(trial_baseline)
                    
                    trial_excited = torch.mean((trial_change > trial_baseline_std).float()).item()
                    excited_fractions_all.append(trial_excited)
                    mean_changes_all.append(torch.mean(trial_change).item())
                
                analysis[f'{pop}_excited_std'] = np.std(excited_fractions_all)
                analysis[f'{pop}_mean_change_std'] = np.std(mean_changes_all)

            if record_currents:
                current_analysis = analyze_currents_by_period(
                    recorded_currents,
                    baseline_start=warmup,
                    baseline_end=stim_start,
                    stim_start=stim_start,
                    stim_end=stim_start+stim_duration
                )
                analysis['current_analysis'] = current_analysis
                
            results[target][intensity] = analysis
    
    # Prepare metadata for saving
    metadata = {
        'optimization_file': optimization_json_file,
        'intensities': intensities,
        'mec_current': mec_current,
        'opsin_current': opsin_current,
        'stim_start': stim_start,
        'stim_duration': stim_duration,
        'warmup': warmup,
        'n_trials': n_trials,
        'base_seed': base_seed,
        'device': str(device),
        'circuit_params': {
            'n_gc': circuit_params.n_gc,
            'n_mc': circuit_params.n_mc,
            'n_pv': circuit_params.n_pv,
            'n_sst': circuit_params.n_sst,
            'n_mec': circuit_params.n_mec,
        }
    }
    
    # Save results if requested
    if save_results_file is not None:
        save_experiment_results(results, conn_analysis, conductance_analysis, 
                                save_results_file, metadata)
    elif auto_save:
        # Auto-save with default filename
        default_filename = get_default_results_filename(optimization_json_file, 
                                                        n_trials, base_seed)
        save_path = Path("protocol") / default_filename
        save_experiment_results(results, conn_analysis, conductance_analysis,
                                str(save_path), metadata)
    
    return experiment, results, conn_analysis, conductance_analysis
    

def save_experiment_results(results: Dict, 
                            connectivity_analysis: Dict,
                            conductance_analysis: Dict,
                            filepath: str,
                            metadata: Optional[Dict] = None):
    """
    Save comparative experiment results to file
    
    Args:
        results: Results dictionary from run_comparative_experiment
        connectivity_analysis: Connectivity analysis dictionary
        conductance_analysis: Conductance analysis dictionary
        filepath: Path to save file (e.g., 'experiment_results.pkl')
        metadata: Optional metadata dict (parameters, timestamp, etc.)
    """
    
    # Prepare data for saving
    save_data = {
        'results': results,
        'connectivity_analysis': connectivity_analysis,
        'conductance_analysis': conductance_analysis,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    # Convert torch tensors to numpy for better compatibility
    def convert_tensors_to_numpy(obj):
        """Recursively convert torch tensors to numpy arrays"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        elif isinstance(obj, dict):
            return {k: convert_tensors_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_tensors_to_numpy(item) for item in obj)
        else:
            return obj
    
    save_data = convert_tensors_to_numpy(save_data)
    
    # Save to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"\nExperiment results saved to: {filepath}")
    logger.info(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    if metadata:
        logger.info(f"  Configuration: {metadata.get('n_trials', 'N/A')} trials, "
              f"seed {metadata.get('base_seed', 'N/A')}")


def load_experiment_results(filepath: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load comparative experiment results from file
    
    Args:
        filepath: Path to saved results file
        
    Returns:
        Tuple of (results, connectivity_analysis, conductance_analysis, metadata)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    # Extract components
    results = save_data['results']
    connectivity_analysis = save_data['connectivity_analysis']
    conductance_analysis = save_data['conductance_analysis']
    metadata = save_data.get('metadata', {})
    
    logger.info(f"\nLoaded experiment results from: {filepath}")
    logger.info(f"  Saved: {save_data.get('timestamp', 'Unknown')}")
    logger.info(f"  Version: {save_data.get('version', 'Unknown')}")
    if metadata:
        logger.info(f"  Configuration: {metadata.get('n_trials', 'N/A')} trials, "
              f"seed {metadata.get('base_seed', 'N/A')}")
        if 'optimization_file' in metadata and metadata['optimization_file']:
            logger.info(f"  Optimization: {metadata['optimization_file']}")
    
    # Convert numpy arrays back to torch tensors where needed
    def convert_numpy_to_tensors(obj):
        """Recursively convert numpy arrays to torch tensors for activity traces"""
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, dict):
            # Only convert specific keys that are expected to be tensors
            tensor_keys = ['recorded_currents',
                           'activity_trace_mean', 'activity_trace_std', 
                           'opsin_expression_mean', 'opsin_expression_std', 'time',
                           'trial_results']

            # Special handling for trial_results (list of trial dicts)
            if 'trial_results' in obj:
                converted_trials = []
                for trial in obj['trial_results']:
                    converted_trial = {}
                    for key, value in trial.items():
                        # Convert tensors in trial dict
                        if key in ['time', 'opsin_expression', 'target_positions']:
                            converted_trial[key] = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                        elif key == 'activity_trace':
                            # Convert activity traces for each population
                            converted_trial[key] = {
                                pop: torch.from_numpy(act) if isinstance(act, np.ndarray) else act
                                for pop, act in value.items()
                            }
                        else:
                            # Keep other fields as-is (layout, connectivity, indices)
                            converted_trial[key] = value
                    converted_trials.append(converted_trial)

                # Return dict with converted trial_results
                result = {k: convert_numpy_to_tensors(v) if k != 'trial_results' else converted_trials 
                         for k, v in obj.items()}
                return result

            # Original logic for other keys
            if any(k in obj for k in tensor_keys):
                return {k: convert_numpy_to_tensors(v) for k, v in obj.items()}
            else:
                return {k: convert_numpy_to_tensors(v) if isinstance(v, (dict, list, np.ndarray)) 
                       else v for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_tensors(item) for item in obj]
        else:
            return obj
    
    
    results = convert_numpy_to_tensors(results)
    
    return results, connectivity_analysis, conductance_analysis, metadata


def get_default_results_filename(optimization_file: Optional[str] = None,
                                 n_trials: int = 1,
                                 base_seed: int = 42) -> str:
    """
    Generate default filename for experiment results
    
    Args:
        optimization_file: Optimization file used (if any)
        n_trials: Number of trials run
        base_seed: Random seed used
        
    Returns:
        Default filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if optimization_file:
        opt_name = Path(optimization_file).stem
        filename = f"DG_experiment_{opt_name}_n{n_trials}_seed{base_seed}_{timestamp}.pkl"
    else:
        filename = f"DG_experiment_default_n{n_trials}_seed{base_seed}_{timestamp}.pkl"
    
    return filename


def calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for firing rate inequality analysis."""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return max(0.0, gini)


def lorenz_curve(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Lorenz curve for inequality visualization."""
    if len(values) == 0:
        return np.array([0, 1]), np.array([0, 1])
    sorted_values = np.sort(values)
    n = len(sorted_values)
    x = np.arange(1, n + 1) / n
    y = np.cumsum(sorted_values) / np.sum(sorted_values)
    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])
    return x, y


def get_opsin_expression_mask(target_pop: str,
                              opsin_expression: np.ndarray, 
                              expression_threshold: float = 0.2) -> Dict:
    """Extract mask for non-opsin and opsin expressing cells."""
    expressing_mask = opsin_expression >= expression_threshold
    return expressing_mask

def plot_adaptive_stepping_analysis(adaptive_stats: Dict,
                                    stim_start: float,
                                    stim_duration: float,
                                    save_path: Optional[str] = None) -> None:
    """
    Visualize adaptive stepping behavior
    
    Args:
        adaptive_stats: Aggregated adaptive statistics from multi-trial run
        stim_start: Stimulation start time (ms) for marking
        stim_duration: Stimulation duration (ms) for marking
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Use sample from first trial for detailed time series
    time_history = adaptive_stats['time_history_sample']
    dt_history = adaptive_stats['dt_history_sample']
    gradient_history = adaptive_stats['gradient_history_sample']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Time step over time
    ax1 = axes[0]
    ax1.plot(time_history[1:], dt_history, 'b-', linewidth=1.5, alpha=0.7)
    
    # Show mean and range across trials if multi-trial
    if 'avg_dt_std' in adaptive_stats and adaptive_stats['avg_dt_std'] > 0:
        n_trials = len(adaptive_stats['trial_avg_dt'])
        ax1.axhline(adaptive_stats['avg_dt_mean'], color='red', linestyle='--', 
                   label=f'Mean across {n_trials} trials', linewidth=2)
        ax1.fill_between([time_history[0], time_history[-1]], 
                        [adaptive_stats['min_dt_min']] * 2,
                        [adaptive_stats['max_dt_max']] * 2,
                        alpha=0.2, color='red', label='Range across trials')
    
    # Mark stimulation period
    ax1.axvspan(stim_start, stim_start + stim_duration, 
               alpha=0.1, color='orange', label='Stimulation')
    
    ax1.set_ylabel(r'Time Step $\Delta t$ (ms)', fontsize=11)
    ax1.set_title('Adaptive Time Step Evolution (Sample Trial)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Gradient magnitude over time
    ax2 = axes[1]
    ax2.plot(time_history[1:], gradient_history, 'g-', linewidth=1.5, alpha=0.7)
    ax2.axvspan(stim_start, stim_start + stim_duration, 
               alpha=0.1, color='orange', label='Stimulation')
    ax2.set_ylabel('Gradient Magnitude (Hz/ms)', fontsize=11)
    ax2.set_title('Activity Gradient Over Time', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Panel 3: Cumulative time vs steps
    ax3 = axes[2]
    cumulative_time = np.cumsum(dt_history)
    steps = np.arange(len(dt_history))
    ax3.plot(cumulative_time, steps, 'purple', linewidth=2)
    
    # Add reference line for fixed dt
    fixed_dt = 0.1  # Standard dt from circuit_params
    max_time = time_history[-1]
    fixed_steps = np.arange(0, max_time / fixed_dt)
    fixed_time = fixed_steps * fixed_dt
    ax3.plot(fixed_time, fixed_steps, 'k--', alpha=0.5, linewidth=1.5,
            label=f'Fixed dt={fixed_dt} ms')
    
    # Calculate efficiency
    n_steps_adaptive = len(dt_history)
    n_steps_fixed = int(max_time / fixed_dt)
    efficiency = (1 - n_steps_adaptive / n_steps_fixed) * 100
    
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_ylabel('Cumulative Steps', fontsize=11)
    ax3.set_title(f'Computational Efficiency: {efficiency:.1f}% Reduction '
                 f'({n_steps_adaptive} vs {n_steps_fixed} steps)', 
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Overall title with multi-trial info
    if 'n_steps_std' in adaptive_stats and adaptive_stats['n_steps_std'] > 0:
        n_trials = len(adaptive_stats['trial_avg_dt'])
        fig.suptitle(f'Adaptive Stepping Analysis (Aggregated over {n_trials} trials)\n'
                    f'Mean steps: {adaptive_stats["n_steps_mean"]:.0f} ± {adaptive_stats["n_steps_std"]:.0f}, '
                    f'Mean dt: {adaptive_stats["avg_dt_mean"]:.3f} ± {adaptive_stats["avg_dt_std"]:.3f} ms',
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Adaptive Stepping Analysis (Single Trial)', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved adaptive stepping analysis to: {save_path}")
    
    plt.show()


def analyze_mec_asymmetry_effects(experiment: OptogeneticExperiment) -> Dict:
    """Analyze how MEC -> PV (but not SST) asymmetry affects circuit dynamics"""
    
    # Reset circuit state
    experiment.circuit.reset_state()

    # Simulation parameters
    n_steps = 2000
    dt = 0.1
    time = torch.arange(n_steps, device=experiment.device) * dt
    mec_drive_time = 100.0  # Start MEC drive at 100ms
    
    activity_trace = {
        'gc': torch.zeros(experiment.circuit_params.n_gc, n_steps, device=experiment.device),
        'mc': torch.zeros(experiment.circuit_params.n_mc, n_steps, device=experiment.device), 
        'pv': torch.zeros(experiment.circuit_params.n_pv, n_steps, device=experiment.device),
        'sst': torch.zeros(experiment.circuit_params.n_sst, n_steps, device=experiment.device),
        'mec': torch.zeros(experiment.circuit_params.n_mec, n_steps, device=experiment.device)
    }
    
    # Store conductance information
    conductance_stats = analyze_conductance_patterns(experiment)
    
    for t in range(n_steps):
        current_time = t * dt
        
        # MEC drive (simulating dentate spike)
        external_drive = {}
        if mec_drive_time <= current_time <= mec_drive_time + 50.0:  # 50ms MEC drive
            mec_drive = torch.ones(experiment.circuit_params.n_mec, device=experiment.device) * 500.0
        else:
            mec_drive = torch.ones(experiment.circuit_params.n_mec, device=experiment.device) * 50.0
        
        external_drive['mec'] = mec_drive

        # No optogenetic stimulation
        direct_activation = {}
        
        # Update circuit
        current_activity = experiment.circuit(direct_activation, external_drive)
        
        # Store activity
        for pop in activity_trace:
            activity_trace[pop][:, t] = current_activity[pop]
    
    # Move to CPU for analysis
    time_cpu = time.cpu()
    activity_trace_cpu = {pop: activity.cpu() for pop, activity in activity_trace.items()}
    
    # Analyze temporal dynamics
    baseline_mask = time_cpu < mec_drive_time
    response_mask = (time_cpu >= mec_drive_time) & (time_cpu <= (mec_drive_time + 50.0))
    
    analysis = {'conductance_stats': conductance_stats}
    
    for pop in ['gc', 'mc', 'pv', 'sst']:
        baseline_rate = torch.mean(activity_trace_cpu[pop][:, baseline_mask], dim=1)
        response_rate = torch.mean(activity_trace_cpu[pop][:, response_mask], dim=1)
        
        # Calculate response latency (time to peak)
        mec_drive_start_idx = int(mec_drive_time / dt)
        mec_drive_end_idx = int((mec_drive_time + 50.0) / dt)
        pop_trace = torch.mean(activity_trace_cpu[pop], dim=0)
        baseline_mean = torch.mean(pop_trace[:mec_drive_start_idx])
        baseline_std = torch.std(pop_trace[:mec_drive_start_idx])

        if torch.max(pop_trace[mec_drive_start_idx:mec_drive_end_idx]) > baseline_mean + 2*baseline_std:
            peak_idx = torch.argmax(pop_trace[mec_drive_start_idx:mec_drive_end_idx]) + mec_drive_start_idx
            latency = (peak_idx * dt) - mec_drive_time
        else:
            latency = float('nan')
        
        analysis[f'{pop}_response'] = {
            'baseline_mean': torch.mean(baseline_rate).item(),
            'baseline_std': torch.std(baseline_rate).item(),
            'response_mean': torch.mean(response_rate).item(),
            'response_std': torch.std(response_rate).item(),
            'response_latency': latency,
            'activated_fraction': torch.mean(response_rate > baseline_rate + torch.std(baseline_rate)).item(),
            'response_change': torch.mean(response_rate - baseline_rate).item()
        }
    
    # MEC analysis
    mec_baseline = torch.mean(activity_trace_cpu['mec'][:, baseline_mask], dim=1)
    mec_response = torch.mean(activity_trace_cpu['mec'][:, response_mask], dim=1)
    
    analysis['mec_response'] = {
        'baseline_mean': torch.mean(mec_baseline).item(),
        'response_mean': torch.mean(mec_response).item(),
        'drive_effectiveness': torch.mean(mec_response - mec_baseline).item() / 450.0
    }
    # MEC analysis
    mec_baseline = torch.mean(activity_trace_cpu['mec'][:, baseline_mask], dim=1)
    mec_response = torch.mean(activity_trace_cpu['mec'][:, response_mask], dim=1)
    
    analysis['mec_response'] = {
        'baseline_mean': torch.mean(mec_baseline).item(),
        'response_mean': torch.mean(mec_response).item(),
        'drive_effectiveness': torch.mean(mec_response - mec_baseline).item() / 450.0
    }
    
    # Key asymmetry analysis
    pv_response_strength = analysis['pv_response']['response_change']
    sst_response_strength = analysis['sst_response']['response_change']
    
    analysis['asymmetry_effect'] = {
        'pv_direct_response': pv_response_strength,
        'sst_indirect_response': sst_response_strength,
        'asymmetry_ratio': pv_response_strength / (abs(sst_response_strength) + 1e-6),
        'pv_latency': analysis['pv_response']['response_latency'],
        'sst_latency': analysis['sst_response']['response_latency']
    }
    
    return analysis


        
def analyze_disinhibition_effects(experiment: OptogeneticExperiment,
                                  target_population: str, 
                                  light_intensity: float,
                                  stim_start = 500.0,
                                  stim_duration = 1000.0,
                                  mec_current: float = 40.0,
                                  opsin_current: float = 100.0) -> Dict:
    """Analyze disinhibition mechanisms with fixed connectivity access"""
    
    # Store original synaptic parameters
    original_synaptic_params = experiment.synaptic_params
    
    # Run simulation with full network
    result_full = experiment.simulate_stimulation(
        target_population, light_intensity,
        mec_current=mec_current,
        opsin_current=opsin_current
    )
    
    # Create modified synaptic parameters with reduced inhibition
    modified_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=original_synaptic_params.ampa_g_mean,
        ampa_g_std=original_synaptic_params.ampa_g_std,
        ampa_g_min=original_synaptic_params.ampa_g_min,
        ampa_g_max=original_synaptic_params.ampa_g_max,
        gaba_g_mean=original_synaptic_params.gaba_g_mean * 0.1,
        gaba_g_std=original_synaptic_params.gaba_g_std * 0.1,
        gaba_g_min=original_synaptic_params.gaba_g_min * 0.1,
        gaba_g_max=original_synaptic_params.gaba_g_max * 0.1,
        distribution=original_synaptic_params.distribution,
        connection_modulation=original_synaptic_params.connection_modulation,
    )

    # Create new experiment with reduced inhibition
    experiment_no_inh = OptogeneticExperiment(
        experiment.circuit_params, 
        modified_synaptic_params,
        experiment.opsin_params,
        device=experiment.device
    )
    
    result_no_inhibition = experiment_no_inh.simulate_stimulation(
        target_population, light_intensity,
        mec_current=mec_current,
        stim_start=stim_start,
        stim_duration=stim_duration,
        opsin_current=opsin_current
    )
    
    time = result_full['time']
    baseline_mask = time < stim_start
    stim_mask = (time >= stim_start) & (time < (stim_start + stim_duration)) 
    
    analysis = {}
    
    for pop in ['gc', 'mc', 'pv', 'sst']:
        if pop == target_population:
            continue

        # Full network response
        activity_full = result_full['activity_trace'][pop]
        baseline_full = torch.mean(activity_full[:, baseline_mask], dim=1)
        stim_full = torch.mean(activity_full[:, stim_mask], dim=1)
        change_full = stim_full - baseline_full
        
        # Reduced inhibition response
        activity_no_inh = result_no_inhibition['activity_trace'][pop]
        baseline_no_inh = torch.mean(activity_no_inh[:, baseline_mask], dim=1)
        stim_no_inh = torch.mean(activity_no_inh[:, stim_mask], dim=1)
        change_no_inh = stim_no_inh - baseline_no_inh
        
        # Analyze paradoxical excitation 
        baseline_std_full = torch.std(baseline_full)
        baseline_std_no_inh = torch.std(baseline_no_inh)
        
        excited_full = torch.sum(change_full > 2 * baseline_std_full)
        excited_no_inh = torch.sum(change_no_inh > 2 * baseline_std_no_inh)

        analysis[f'{pop}_paradoxical_excitation'] = {
            'with_inhibition': excited_full.item(),
            'without_inhibition': excited_no_inh.item(),
            'disinhibition_dependent': (excited_full - excited_no_inh).item(),
            'mean_change_full': torch.mean(change_full).item(),
            'mean_change_no_inh': torch.mean(change_no_inh).item(),
            'std_change_full': torch.std(change_full).item(),
            'std_change_no_inh': torch.std(change_no_inh).item()
        }
    
    return analysis


        
def test_disinhibition_hypothesis(optimization_json_file: Optional[str] = None,
                                  mec_current: float = 100.0,
                                  opsin_current: float = 100.0,
                                  device: Optional[torch.device] = None):
    """Test whether disinhibition mechanisms explain paradoxical excitation"""
    
    if device is None:
        device = get_default_device()
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    experiment = OptogeneticExperiment(
        circuit_params, synaptic_params, opsin_params,
        optimization_json_file=optimization_json_file,
        device=device
    )
    
    logger.info("Testing Disinhibition Hypothesis")
    logger.info("=" * 40)
    
    for target in ['pv', 'sst']:
        logger.info(f"\n{target.upper()} Stimulation:")
        logger.info("-" * 20)
        
        analysis = analyze_disinhibition_effects(
            experiment, target, 1.0,
            mec_current=mec_current,
            opsin_current=opsin_current
        )
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if f'{pop}_paradoxical_excitation' in analysis:
                result = analysis[f'{pop}_paradoxical_excitation']
                
                logger.info(f"{pop.upper()}:")
                logger.info(f"  With inhibition: {result['with_inhibition']} cells excited")
                logger.info(f"  Reduced inhibition: {result['without_inhibition']} cells excited") 
                logger.info(f"  Disinhibition-dependent: {result['disinhibition_dependent']} cells")
                logger.info(f"  Mean change: {result['mean_change_full']:.3f} -> {result['mean_change_no_inh']:.3f}")
                logger.info(f"  Change variability: {result['std_change_full']:.3f} -> {result['std_change_no_inh']:.3f}")


def _run_paired_expression_condition(
    circuit_params: 'CircuitParams',
    synaptic_params: 'PerConnectionSynapticParams',
    opsin_params: 'OpsinParams',
    target: str,
    intensity: float,
    connectivity_seeds: List[int],
    stim_start: float,
    stim_duration: float,
    warmup: float,
    mec_current: float,
    opsin_current: float,
    optimization_json_file: Optional[str],
    device: torch.device,
    condition_label: str = '',
    **optogenetic_experiment_kwargs,
) -> Dict:
    """
    Run one expression-level condition across all connectivity instances (paired design).

    Similar to _run_paired_ablation_condition in ablation_tests.py, but generates
    fresh opsin expressions at the level specified in opsin_params rather than
    injecting pre-loaded ones.  This allows the same circuit topology to be tested
    across different expression levels while keeping connectivity fixed.

    Args:
        circuit_params: Circuit parameters
        synaptic_params: Synaptic parameters (may include ablation modulations)
        opsin_params: Opsin parameters with the desired expression_mean
        target: Target population to stimulate
        intensity: Light intensity
        connectivity_seeds: Seeds for each connectivity instance
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline start (ms)
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        optimization_json_file: Path to optimization results
        device: Torch device
        condition_label: Label for logging
        **optogenetic_experiment_kwargs: Forwarded to OptogeneticExperiment

    Returns:
        Aggregated analysis dict with '_mean', '_std', '_sem', '_by_conn' keys
        produced by _aggregate_paired_ablation_results.
    """
    n_conn = len(connectivity_seeds)
    per_conn_analyses = []

    for conn_idx, conn_seed in enumerate(connectivity_seeds):
        logger.info(
            f"      {condition_label} conn {conn_idx + 1}/{n_conn} "
            f"(seed={conn_seed}), expr_mean={opsin_params.expression_mean:.2f}"
        )

        # Create experiment -- opsin expression is generated fresh from opsin_params
        experiment = OptogeneticExperiment(
            circuit_params,
            synaptic_params,
            opsin_params,
            optimization_json_file=optimization_json_file,
            device=device,
            base_seed=conn_seed,
            **optogenetic_experiment_kwargs,
        )

        # Single-trial simulation with fixed connectivity
        result = experiment.simulate_stimulation(
            target_population=target,
            light_intensity=intensity,
            stim_start=stim_start,
            stim_duration=stim_duration,
            plot_activity=False,
            mec_current=mec_current,
            opsin_current=opsin_current,
            n_trials=1,
            regenerate_connectivity_per_trial=False,
        )

        # Compute time masks
        time = result['time']
        baseline_mask = (time >= warmup) & (time < stim_start)
        stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

        # Reuse ablation analysis helper for consistent metrics
        analysis = _analyze_ablation_trial(result, target, baseline_mask, stim_mask)
        per_conn_analyses.append(analysis)

        # Free memory
        del experiment
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Aggregate across connectivity instances
    return _aggregate_paired_ablation_results(per_conn_analyses, target)


def test_opsin_expression_levels(
    optimization_json_file: Optional[str] = None,
    target_populations: List[str] = ['pv', 'sst'],
    expression_levels: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
    intensity: float = 1.0,
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    device: Optional[torch.device] = None,
    n_trials: int = 3,
    base_seed: int = 42,
    include_ablations: bool = False,
    save_results_file: Optional[str] = None,
    nested_experiment_file: Optional[str] = None,
    **optogenetic_experiment_kwargs
) -> Dict:
    """
    Test how paradoxical excitation varies with opsin expression level

    This test varies the mean opsin expression level and measures how
    paradoxical excitation changes. Optionally includes key ablation
    conditions to test mechanism robustness.

    When nested_experiment_file is specified, connectivity seeds are read
    from the HDF5 file so that each expression level is tested on the
    same circuit realizations used in the nested experiment (paired mode).
    In this mode n_trials and base_seed are ignored; one trial is run per
    connectivity instance.

    Args:
        optimization_json_file: Path to optimization results (optional)
        target_populations: List of populations to stimulate
        expression_levels: List of mean expression levels to test (0-1)
        intensity: Light intensity for stimulation
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: When to start stimulation (ms)
        stim_duration: Duration of stimulation (ms)
        warmup: Pre-stimulation period (ms)
        device: Device to run on (None for auto-detect)
        n_trials: Number of trials to average per condition (ignored in paired mode)
        base_seed: Base random seed (ignored in paired mode)
        include_ablations: Whether to test ablation conditions
        save_results_file: If provided, save results to this file
        nested_experiment_file: Path to nested experiment HDF5 file.
            When specified, connectivity seeds are loaded from this file
            for paired comparisons across expression levels.

    Returns:
        Dictionary with results for each expression level and condition
    """

    if device is None:
        from dendritic_somatic_transfer import get_default_device
        device = get_default_device()

    # --- Load connectivity seeds from nested experiment file
    connectivity_seeds = None
    paired_mode = False

    if nested_experiment_file is not None:
        from nested_experiment import load_nested_experiment_seeds_and_opsin

        logger.info(f"\nLoading connectivity from nested experiment: "
                    f"{nested_experiment_file}")
        nested_data = load_nested_experiment_seeds_and_opsin(
            nested_experiment_file, target_populations
        )
        connectivity_seeds = nested_data['connectivity_seeds']
        paired_mode = True
        # discard opsin_expressions -- we sweep expression levels
        logger.info(f"  Paired mode: {len(connectivity_seeds)} connectivity "
                    f"instances (opsin expressions will be generated fresh)")

    logger.info("\n" + "="*80)
    logger.info("TEST: Opsin Expression Level Dependence")
    logger.info("="*80)
    logger.info(f"\nTesting {len(expression_levels)} expression levels:")
    logger.info(f"  Expression levels: {expression_levels}")
    if paired_mode:                                                            logger.info(f"  Mode: PAIRED ({len(connectivity_seeds)} "      
                    f"connectivity instances)")                        
    else:                                                              
        logger.info(f"  Trials per condition: {n_trials}")
    if include_ablations:
        logger.info(f"  Including key ablation conditions")
    logger.info("="*80 + "\n")

    circuit_params = CircuitParams()
    base_synaptic_params = PerConnectionSynapticParams()

    # Define ablation conditions if requested
    ablation_configs = {'full_network': base_synaptic_params}

    if include_ablations:
        # Key ablation: Block excitation to interneurons (disinhibition test)
        ablation_configs['blocked_exc_to_int'] = PerConnectionSynapticParams(
            **{k: v for k, v in base_synaptic_params.__dict__.items()
               if k != 'connection_modulation'},
            connection_modulation={
                **base_synaptic_params.connection_modulation,
                'mec_pv': 0.01,
                'gc_pv': 0.01,
                'mc_pv': 0.01,
                'gc_sst': 0.01,
                'mc_sst': 0.01,
            }
        )

    results = {config_name: {target: {} for target in target_populations}
              for config_name in ablation_configs.keys()}

    for target in target_populations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {target.upper()} stimulation")
        logger.info('='*60)

        for config_name, synaptic_params in ablation_configs.items():
            logger.info(f"\n{config_name.replace('_', ' ').title()}")
            logger.info('-'*60)

            for expr_level in expression_levels:
                logger.info(f"  Expression level: {expr_level:.1f}")

                # Create opsin parameters with this expression level
                opsin_params = OpsinParams(
                    expression_mean=expr_level,
                    expression_std=0.05,
                    failure_rate=0.5,
                    light_decay=0.4,
                    hill_coeff=2.5,
                    half_sat=0.4
                )

                # ---- paired-mode branch ----
                if paired_mode:
                    analysis = _run_paired_expression_condition(
                        circuit_params=circuit_params,
                        synaptic_params=synaptic_params,
                        opsin_params=opsin_params,
                        target=target,
                        intensity=intensity,
                        connectivity_seeds=connectivity_seeds,
                        stim_start=stim_start,
                        stim_duration=stim_duration,
                        warmup=warmup,
                        mec_current=mec_current,
                        opsin_current=opsin_current,
                        optimization_json_file=optimization_json_file,
                        device=device,
                        condition_label=f'[{config_name}]',
                        **optogenetic_experiment_kwargs,
                    )
                    analysis['expression_level'] = expr_level
                    results[config_name][target][expr_level] = analysis

                # ---- unpaired branch ----
                else:
                    # Create experiment
                    experiment = OptogeneticExperiment(
                        circuit_params, synaptic_params, opsin_params,
                        optimization_json_file=optimization_json_file,
                        device=device,
                        base_seed=base_seed + int(expr_level * 1000),
                        **optogenetic_experiment_kwargs
                    )

                    # Run stimulation
                    result = experiment.simulate_stimulation(
                        target, intensity,
                        stim_start=stim_start,
                        stim_duration=stim_duration,
                        plot_activity=False,
                        mec_current=mec_current,
                        opsin_current=opsin_current,
                        n_trials=n_trials
                    )

                    # Analyze results
                    time = result['time']
                    activity_mean = result['activity_trace_mean']
                    baseline_mask = (time >= warmup) & (time < stim_start)
                    stim_mask = (time >= stim_start) & (
                        time <= (stim_start + stim_duration))

                    analysis = {'expression_level': expr_level}

                    for pop in ['gc', 'mc', 'pv', 'sst']:
                        if pop == target:
                            continue

                        baseline_rate = torch.mean(
                            activity_mean[pop][:, baseline_mask], dim=1)
                        stim_rate = torch.mean(
                            activity_mean[pop][:, stim_mask], dim=1)
                        rate_change = stim_rate - baseline_rate
                        baseline_std = torch.std(baseline_rate)

                        excited_fraction = torch.mean(
                            (rate_change > baseline_std).float())

                        analysis[f'{pop}_excited'] = excited_fraction.item()
                        analysis[f'{pop}_mean_change'] = torch.mean(
                            rate_change).item()

                        # Get trial-to-trial variability
                        if n_trials > 1:
                            excited_fractions = []
                            for trial_result in result['trial_results']:
                                trial_activity = (
                                    trial_result['activity_trace'][pop])
                                trial_baseline = torch.mean(
                                    trial_activity[:, baseline_mask], dim=1)
                                trial_stim = torch.mean(
                                    trial_activity[:, stim_mask], dim=1)
                                trial_change = trial_stim - trial_baseline
                                trial_baseline_std = torch.std(trial_baseline)
                                trial_excited = torch.mean(
                                    (trial_change > trial_baseline_std
                                     ).float()).item()
                                excited_fractions.append(trial_excited)

                            analysis[f'{pop}_excited_std'] = np.std(
                                excited_fractions)

                    results[config_name][target][expr_level] = analysis

    # Save results if requested
    if save_results_file:
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to: {save_results_file}")

    return results

    

def plot_combined_ablation_and_expression(
    ablation_results: Dict,
    expression_results: Dict,
    intensity: float = 1.0,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive figure combining ablation and expression level results
    
    Args:
        ablation_results: Results from run_all_ablation_tests()
        expression_results: Results from test_opsin_expression_levels()
        intensity: Light intensity for ablation results
        save_path: Optional directory to save figure
    """
    
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 8, hspace=0.35, wspace=0.35)
    
    # Top row: Ablation test bar plots
    ablation_axes = [fig.add_subplot(gs[0, i]) for i in range(8)]
    
    # Extract ablation data
    results_int_int = ablation_results['interneuron_interactions']
    results_exc_int = ablation_results['excitation_to_interneurons']
    results_recurrent = ablation_results['recurrent_excitation']
    
    colors = {
        'full': '#2ecc71',
        'int_int': '#e74c3c',
        'exc_int': '#e67e22',
        'recurrent': '#3498db'
    }
    
    conditions = ['Full', 'Int-Int', 'Exc->Int', 'MC<->GC']
    
    pop_map = {
        'pv': ['gc', 'mc', 'pv_nonexpr', 'sst'],
        'sst': ['gc', 'mc', 'pv', 'sst_nonexpr']
    }

    y_label_set = False
    ax_idx = 0
    for target in ['pv', 'sst']:
        populations = pop_map[target]
        for pop in populations:
            ax = ablation_axes[ax_idx]
            
            # Handle non-expressing target cells
            if pop.endswith('_nonexpr'):
                base_pop = pop.replace('_nonexpr', '')
                full_excited = results_exc_int['full_network'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                int_int_excited = results_int_int['blocked_int_int'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                exc_int_excited = results_exc_int['blocked_exc_to_int'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                recurrent_excited = results_recurrent['blocked_recurrent'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
            else:
                full_excited = results_exc_int['full_network'][target][intensity][f'{pop}_excited']
                int_int_excited = results_int_int['blocked_int_int'][target][intensity][f'{pop}_excited']
                exc_int_excited = results_exc_int['blocked_exc_to_int'][target][intensity][f'{pop}_excited']
                recurrent_excited = results_recurrent['blocked_recurrent'][target][intensity][f'{pop}_excited']
            
            data = [full_excited, int_int_excited, exc_int_excited, recurrent_excited]
            
            x_pos = np.arange(len(conditions))
            bars = ax.bar(x_pos, [d * 100 for d in data],
                         color=[colors['full'], colors['int_int'],
                               colors['exc_int'], colors['recurrent']],
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Value labels
            for bar, value in zip(bars, data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value*100:.0f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(conditions, fontsize=10, rotation=45, ha='right')
            if not y_label_set:
                ax.set_ylabel('% Excited', fontsize=10)
                y_label_set = True
            
            # Color interneuron titles differently
            # Color interneuron titles differently
            if pop.endswith('_nonexpr'):
                base_pop = pop.replace('_nonexpr', '')
                ax.set_title(f'{target.upper()}->{target.upper()}†', fontsize=10, 
                           fontweight='bold', color='purple')
            elif pop in ['pv', 'sst']:
                ax.set_title(f'{target.upper()}->{pop.upper()}*', fontsize=10, 
                           fontweight='bold', color='darkred')
            else:
                ax.set_title(f'{target.upper()}->{pop.upper()}', fontsize=10, 
                           fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max([d * 100 for d in data]) * 1.2)
            
            ax_idx += 1
    
    # Middle and bottom rows: Expression level line plots (12 panels)
    expression_axes = [fig.add_subplot(gs[1, col]) 
                       for col in range(8)]
    
    # Get expression levels
    config_names = list(expression_results.keys())
    targets = list(expression_results[config_names[0]].keys())
    expression_levels = sorted(expression_results[config_names[0]][targets[0]].keys())
    
    expression_colors = {
        'full_network': '#2ecc71',
        'blocked_exc_to_int': '#e67e22',
        'blocked_int_int': '#e74c3c',
        'blocked_recurrent': '#3498db'
    }
    markers = {
        'full_network': 'o',
        'blocked_exc_to_int': 's'
    }
    labels = {
        'full_network': 'Full Network',
        'blocked_exc_to_int': 'Block Exc->Int'
    }

    y_label_set = False
    ax_idx = 0
    for target in targets:
        populations = pop_map[target]
        for pop in populations:
            ax = expression_axes[ax_idx]
            
            for config_name in config_names:
                excited_fractions = []
                
                for expr_level in expression_levels:
                    data = expression_results[config_name][target][expr_level]
                    
                    # Handle non-expressing target cells
                    if pop.endswith('_nonexpr'):
                        base_pop = pop.replace('_nonexpr', '')
                        excited_fractions.append(data.get(f'{base_pop}_nonexpr_excited', 0.0) * 100)
                    else:
                        excited_fractions.append(data[f'{pop}_excited'] * 100)
                
                ax.plot(expression_levels, excited_fractions,
                        marker=markers.get(config_name, 'o'),
                        color=expression_colors.get(config_name, 'gray'),
                        label=labels.get(config_name, config_name),
                        linewidth=1.5, markersize=5, alpha=0.8)
            
            ax.set_xlabel('Opsin Expression', fontsize=10)
            if not y_label_set:
                ax.set_ylabel('% Excited', fontsize=10)
                y_label_set = True
            
            # Color interneuron titles differently
            if pop.endswith('_nonexpr'):
                base_pop = pop.replace('_nonexpr', '')
                ax.set_title(f'{target.upper()}->{target.upper()}†', fontsize=10, 
                           fontweight='bold', color='purple')
            elif pop in ['pv', 'sst']:
                ax.set_title(f'{target.upper()}->{pop.upper()}*', fontsize=10, 
                           fontweight='bold', color='darkred')
            else:
                ax.set_title(f'{target.upper()}->{pop.upper()}', fontsize=10, 
                           fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            
            ax_idx += 1
    
    # Add section labels
    fig.text(0.02, 0.95, 'A. Ablation Tests (* = non-target IN, † = non-expr target)', 
             fontsize=11, fontweight='bold')
    fig.text(0.02, 0.475, 'B. Expression Level Dependence', 
             fontsize=11, fontweight='bold')
    
    fig.suptitle('Dependence of Paradoxical Excitation on\n'
                 'Mechanisms and Expression Level\n',
                fontsize=13, fontweight='bold', y=0.98)
    
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / 'combined_ablation_expression.pdf',
                   dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'combined_ablation_expression.png',
                   dpi=300, bbox_inches='tight')
        logger.info(f"\nSaved combined plot to {save_dir}")
    
    plt.show()


def plot_recorded_currents(circuit, recorded_currents, current_analysis,
                           target_population='pv', baseline_start=500.0,
                           stim_start=1500.0, stim_duration=1000.0,
                           output_dir='./current_visualization'):
    """
    Workflow for visualizing synaptic currents
    
    Args:
        circuit: DentateCircuit instance
        recorded_currents: Output from SynapticCurrentRecorder.get_results()
        current_analysis: Output from analyze_currents_by_period()
        target_population: Stimulated population
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create visualization object
    vis = DGCircuitVisualization(circuit)
    
    stim_end = stim_start + stim_duration
    
    # Current traces for each population
    logger.info("\nGenerating current trace plots...")
    for pop in ['gc', 'mc', 'pv', 'sst']:
        if pop not in recorded_currents['by_type']:
            continue
        
        logger.info(f"   - {pop.upper()} current traces")
        fig = vis.plot_current_traces(
            recorded_currents,
            population=pop,
            stim_start=stim_start,
            stim_end=stim_end,
            baseline_start=baseline_start,
            save_path=str(output_path / f'current_traces_{pop}_{target_population}_stim.pdf')
        )
        plt.close(fig)
    
    # Current sources for each population
    logger.info("\nGenerating current source plots...")
    for pop in ['gc', 'mc', 'pv', 'sst']:
        if pop not in recorded_currents['by_source']:
            continue
        
        # Only plot if there are sources
        if len(recorded_currents['by_source'][pop]) > 0:
            logger.info(f"   - {pop.upper()} current sources")
            fig = vis.plot_current_sources(
                recorded_currents,
                population=pop,
                stim_start=stim_start,
                stim_end=stim_end,
                baseline_start=baseline_start,
                save_path=str(output_path / f'current_sources_{pop}_{target_population}_stim.pdf')
            )
            plt.close(fig)
    
    # Comparison bar plots
    logger.info("\nGenerating comparison bar plots...")
    fig = vis.plot_current_comparison_bar(
        current_analysis,
        target_population=target_population,
        populations=['gc', 'mc', 'pv', 'sst'],
        save_path=str(output_path / f'current_comparison_{target_population}_stim.pdf')
    )
    plt.close(fig)
    
    # Current heatmaps for key populations
    logger.info("\nGenerating current heatmaps...")
    for pop in ['gc', 'mc']:
        for current_type in ['net', 'total_exc', 'total_inh']:
            logger.info(f"   - {pop.upper()} {current_type} heatmap")
            fig = vis.plot_current_heatmap(
                recorded_currents,
                population=pop,
                current_type=current_type,
                stim_start=stim_start,
                stim_end=stim_end,
                baseline_start=baseline_start,
                sort_by_mean=True,
                save_path=str(output_path / f'current_heatmap_{pop}_{current_type}_{target_population}_stim.pdf')
            )
            if fig is not None:
                plt.close(fig)

def plot_synaptic_distributions(circuit,
                                target_population: str,
                                opsin_expression: Dict[str, np.ndarray],
                                output_dir: str = './synaptic_weights'):
    """
    Plot synaptic weight distributions for all post-synaptic populations
    
    Args:
        circuit: DentateCircuit instance
        target_population: Population that was optogenetically stimulated
        opsin_expression: Dict mapping population names to expression arrays
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nGenerating synaptic weight distributions for {target_population.upper()} stimulation:")
    logger.info(f"  Output directory: {output_path}")
    
    vis = DGCircuitVisualization(circuit)
    
    # Plot weights to each post-synaptic population
    for post_pop in ['gc', 'mc', 'pv', 'sst']:
        logger.info(f"  Plotting weights to {post_pop.upper()}...")
        
        # 1. Input weight distribution (histogram + violin plots)
        distribution_path = output_path / f'{target_population}_stim_to_{post_pop}_weight_distribution.pdf'
        try:
            fig = vis.plot_input_weight_distribution(
                post_population=post_pop,
                opsin_expression=opsin_expression,
                stimulated_population=target_population,
                plot_type='both',
                save_path=str(distribution_path)
            )
            if fig is not None:
                plt.close(fig)
                logger.info(f"  Saved distribution: {distribution_path.name}")
        except Exception as e:
            logger.info(f"    Error creating distribution plot: {e}")
        
        # 2. Weight heatmap by cell
        heatmap_path = output_path / f'{target_population}_stim_to_{post_pop}_weight_heatmap.pdf'
        try:
            fig = vis.plot_weight_heatmap_by_cell(
                post_population=post_pop,
                opsin_expression=opsin_expression,
                save_path=str(heatmap_path)
            )
            if fig is not None:
                plt.close(fig)
                logger.info(f"  Saved heatmap: {heatmap_path.name}")
        except Exception as e:
            logger.info(f"  Error creating heatmap: {e}")
        
        # 3. Weight correlation matrix
        correlation_path = output_path / f'{target_population}_stim_to_{post_pop}_weight_correlation.pdf'
        try:
            fig = vis.plot_weight_correlation_matrix(
                post_population=post_pop,
                opsin_expression=opsin_expression,
                save_path=str(correlation_path)
            )
            if fig is not None:
                plt.close(fig)
                logger.info(f"  Saved correlation: {correlation_path.name}")
        except Exception as e:
            logger.info(f"  Error creating correlation plot: {e}")
    
    logger.info(f"\nCompleted synaptic weight plotting for {target_population.upper()}")
                
            
        

    
                

def print_current_analysis(current_analysis):
    
    logger.info("\n" + "-"*80)
    logger.info("Current analysis summary")
    logger.info("-"*80)
    
    for pop in ['gc', 'mc', 'pv', 'sst']:
        if pop not in current_analysis['baseline']:
            continue
        
        logger.info(f"\n{pop.upper()}:")
        
        baseline = current_analysis['baseline'][pop]['by_type']
        stim = current_analysis['stimulation'][pop]['by_type']
        change = current_analysis['change'][pop]['by_type']
        
        logger.info(f"  Excitatory: {baseline['total_exc']['mean']:>7.2f} -> "
              f"{stim['total_exc']['mean']:>7.2f} pA "
              f"(Δ = {change['total_exc']['mean']:>+7.2f} pA)")
        
        logger.info(f"  Inhibitory: {baseline['total_inh']['mean']:>7.2f} -> "
              f"{stim['total_inh']['mean']:>7.2f} pA "
              f"(Δ = {change['total_inh']['mean']:>+7.2f} pA)")
        
        logger.info(f"  Net:        {baseline['net']['mean']:>7.2f} -> "
              f"{stim['net']['mean']:>7.2f} pA "
              f"(Δ = {change['net']['mean']:>+7.2f} pA)")
        
        # E/I ratio
        baseline_ei = abs(baseline['total_exc']['mean']) / (abs(baseline['total_inh']['mean']) + 1e-6)
        stim_ei = abs(stim['total_exc']['mean']) / (abs(stim['total_inh']['mean']) + 1e-6)
        logger.info(f"  E/I Ratio:  {baseline_ei:>7.3f} -> {stim_ei:>7.3f}")
    

def reconstruct_circuit_from_metadata(metadata: Dict,
                                      optimization_json_file: Optional[str] = None,
                                      device: Optional[torch.device] = None) -> DentateCircuit:
    """
    Reconstruct circuit from saved metadata
    
    NOTE: Does NOT recreate opsin expression due to lack of dedicated RNG seed.
    Opsin expression should always be extracted from saved results.
    
    Args:
        metadata: Metadata dict from saved experiment results
        optimization_json_file: Optional path to optimization file
        device: Device to create circuit on
        
    Returns:
        Reconstructed circuit (without opsin expression)
    """
    if device is None:
        device = get_default_device()
    
    logger.info("\nReconstructing circuit from saved metadata...")
    
    # Extract circuit parameters from metadata
    circuit_config = metadata.get('circuit_params', {})
    
    # Create circuit parameters with saved values
    circuit_params = CircuitParams(
        n_gc=circuit_config.get('n_gc', 1000),
        n_mc=circuit_config.get('n_mc', 30),
        n_pv=circuit_config.get('n_pv', 30),
        n_sst=circuit_config.get('n_sst', 20),
        n_mec=circuit_config.get('n_mec', 60)
    )
    
    # Create synaptic and opsin parameters
    synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()
    
    # Create circuit
    circuit = DentateCircuit(
        circuit_params,
        synaptic_params,
        opsin_params,
        device=device
    )
    
    # Apply optimization parameters if file provided
    if optimization_json_file:
        circuit.load_and_apply_optimization_results(optimization_json_file)
    
    # Use base_seed from metadata for circuit consistency
    base_seed = metadata.get('base_seed', 42)
    set_random_seed(base_seed, device)
    
    logger.info(f"  Circuit reconstructed with {circuit_params.n_gc} GC, "
          f"{circuit_params.n_mc} MC, {circuit_params.n_pv} PV, "
          f"{circuit_params.n_sst} SST cells")
    logger.info(f"  Using seed: {base_seed}")
    
    return circuit        


def plot_synaptic_weights_from_results(results: Dict,
                                       metadata: Dict,
                                       output_path: Path,
                                       optimization_json_file: Optional[str] = None,
                                       device: Optional[torch.device] = None):
    """
    Plot synaptic weight distributions from saved results
    
    Args:
        results: Loaded experiment results
        metadata: Metadata from saved results
        output_path: Base output directory (e.g., Path('protocol'))
        optimization_json_file: Optional path to optimization file
        device: Device to create circuit on
    """
    logger.info("\n" + "="*80)
    logger.info("Generating Synaptic Weight Distribution Plots")
    logger.info("="*80)
    
    # Reconstruct circuit
    circuit = reconstruct_circuit_from_metadata(
        metadata, 
        optimization_json_file,
        device
    )
    
    # Create base output directory for weights
    weights_base_dir = output_path / 'synaptic_weights'
    weights_base_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"\nBase output directory: {weights_base_dir}")
    
    # Plot for each target population
    for target_pop in ['pv', 'sst']:
        if target_pop not in results:
            logger.info(f"\nSkipping {target_pop.upper()} - no results found")
            continue
        
        # Use highest intensity for weight visualization
        intensities = sorted(results[target_pop].keys())
        if len(intensities) == 0:
            logger.info(f"\nSkipping {target_pop.upper()} - no intensities found")
            continue
        
        intensity = intensities[-1]  # Highest intensity
        experiment_data = results[target_pop][intensity]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {target_pop.upper()} stimulation at intensity {intensity}")
        logger.info('='*60)
        
        # Get opsin expression levels
        if 'opsin_expression_mean' in experiment_data:
            opsin_expression_array = experiment_data['opsin_expression_mean']
            if hasattr(opsin_expression_array, 'cpu'):
                opsin_expression_array = opsin_expression_array.cpu().numpy()
            else:
                opsin_expression_array = np.array(opsin_expression_array)
        else:
            raise RuntimeError("Unable to obtain opsin expression data")
        
        # Create opsin expression dict for visualization
        # This should include the target population's expression
        opsin_expr_dict = {
            target_pop: opsin_expression_array
        }
        
        # Create subdirectory for this target population
        target_output_dir = weights_base_dir / f'{target_pop}_stimulation'
        
        # Plot weight distributions
        plot_synaptic_distributions(
            circuit=circuit,
            target_population=target_pop,
            opsin_expression=opsin_expr_dict,
            output_dir=str(target_output_dir)
        )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"All synaptic weight plots saved to: {weights_base_dir}")
    logger.info('='*80)


def plot_currents_from_results(results: Dict,
                               metadata: Dict,
                               output_path: Path,
                               optimization_json_file: Optional[str] = None,
                               device: Optional[torch.device] = None):
    """
    Plot synaptic current analysis from saved results
    
    Args:
        results: Loaded experiment results
        metadata: Metadata from saved results
        output_path: Directory to save plots
        optimization_json_file: Optional path to optimization file
        device: Device to create circuit on
    """
    logger.info("\n" + "="*80)
    logger.info("Generating Synaptic Current Plots")
    logger.info("="*80)
    
    # Check if any results contain current data
    has_current_data = False
    for target_pop in ['pv', 'sst']:
        if target_pop in results:
            for intensity, data in results[target_pop].items():
                if 'current_analysis' in data and 'recorded_currents' in data:
                    has_current_data = True
                    break
            if has_current_data:
                break
    
    if not has_current_data:
        logger.info("No current recording data found in results")
        logger.info("Note: Current recording must be enabled during simulation with --record-currents")
        return
    
    # Reconstruct circuit from metadata
    circuit, _ = reconstruct_circuit_from_metadata(
        metadata,
        optimization_json_file,
        device
    )
    
    # Create output directory
    currents_output = output_path / 'synaptic_currents'
    currents_output.mkdir(exist_ok=True, parents=True)
    
    # Extract stimulation parameters from metadata
    stim_start = metadata.get('stim_start', 1500.0)
    stim_duration = metadata.get('stim_duration', 1000.0)
    baseline_start = metadata.get('warmup', 500.0)
    
    logger.info(f"\nStimulation parameters:")
    logger.info(f"  Baseline: {baseline_start} - {stim_start} ms")
    logger.info(f"  Stimulation: {stim_start} - {stim_start + stim_duration} ms")
    
    # Plot for each target population and intensity with current data
    for target_pop in ['pv', 'sst']:
        if target_pop not in results:
            continue
        
        logger.info(f"\nProcessing {target_pop.upper()} stimulation:")
        
        for intensity in sorted(results[target_pop].keys()):
            experiment_data = results[target_pop][intensity]
            
            if 'current_analysis' not in experiment_data:
                continue
            
            if 'recorded_currents' not in experiment_data:
                continue
            
            recorded_currents = experiment_data['recorded_currents']
            current_analysis = experiment_data['current_analysis']
            
            if recorded_currents is None or current_analysis is None:
                continue
            
            logger.info(f"  Plotting currents for intensity {intensity}")
            
            condition_dir = currents_output / f"{target_pop}_intensity_{intensity}"
            condition_dir.mkdir(exist_ok=True, parents=True)
            
            plot_recorded_currents(
                circuit=circuit,
                recorded_currents=recorded_currents,
                current_analysis=current_analysis,
                target_population=target_pop,
                baseline_start=baseline_start,
                stim_start=stim_start,
                stim_duration=stim_duration,
                output_dir=str(condition_dir)
            )
    
    logger.info(f"\nSynaptic current plots saved to: {currents_output}")
    

def analyze_and_plot_weights_by_response(
    circuit,
    target_population: str,
    experiment_results: Dict,
    stim_start: float,
    stim_duration: float,
    warmup: float = 500.0,
    post_populations: Optional[List[str]] = None,
    threshold_std: float = 1.0,
    save_path: Optional[str] = None
) -> Tuple[Dict, List[plt.Figure]]:
    """
    Combined analysis and plotting of weights by post-synaptic response
    
    For each post-synaptic population, classifies cells as excited vs suppressed
    during optogenetic stimulation, then plots violin plots of input weights
    from each source separately for excited and suppressed cells.
    
    This analysis reveals whether cells that show paradoxical excitation
    (or suppression) have different patterns of synaptic input compared to
    cells that respond in the opposite direction.
    
    Args:
        circuit: DentateCircuit instance
        target_population: Optogenetically stimulated population (e.g., 'pv')
        experiment_results: Results from simulate_stimulation (aggregated)
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Pre-stimulation period (ms)
        post_populations: List of populations to analyze (None = ['gc', 'mc'])
        threshold_std: Classification threshold in standard deviations
        save_path: Base directory for saving figures (creates subdirectory)
        
    Returns:
        Tuple of (analysis_results_dict, list_of_figures)
        
    Example:
        >>> analyses, figs = analyze_and_plot_weights_by_response(
        ...     circuit=circuit,
        ...     target_population='pv',
        ...     experiment_results=results,
        ...     stim_start=1500.0,
        ...     stim_duration=1000.0,
        ...     save_path='protocol/weight_analysis'
        ... )
    """
    if post_populations is None:
        post_populations = ['gc', 'mc', 'pv', 'sst']
    
    # Get activity traces and masks
    time = experiment_results['time']
    activity_trace_mean = experiment_results['activity_trace_mean']
    
    baseline_mask = (time >= warmup) & (time < stim_start)
    stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
    
    # Get opsin expression
    opsin_expression_mean = experiment_results.get('opsin_expression_mean', None)
    
    if opsin_expression_mean is not None:
        if hasattr(opsin_expression_mean, 'cpu'):
            opsin_expression_array = opsin_expression_mean.cpu().numpy()
        else:
            opsin_expression_array = np.array(opsin_expression_mean)
        
        opsin_expression = {target_population: opsin_expression_array}
    else:
        opsin_expression = None
    
    # Create visualization object
    vis = DGCircuitVisualization(circuit)
    
    # Create output directory if save_path provided
    if save_path:
        output_dir = Path(save_path) / f'{target_population}_weights_by_response'
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nSaving analysis of weights by response to: {output_dir}")
    
    # Analyze and plot for each post-synaptic population
    all_analyses = {}
    all_figures = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing synaptic weights by post-synaptic response")
    logger.info(f"Target: {target_population.upper()} stimulation")
    logger.info('='*60)
    
    for post_pop in post_populations:
        logger.info(f"\nAnalyzing weights to {post_pop.upper()}...")
        
        # Analyze
        analysis = vis.analyze_weights_by_response_type(
            target_population=target_population,
            post_population=post_pop,
            activity_trace=activity_trace_mean,
            baseline_mask=baseline_mask,
            stim_mask=stim_mask,
            opsin_expression=opsin_expression,
            threshold_std=threshold_std
        )
        
        all_analyses[post_pop] = analysis
        
        # Print summary
        logger.info(f"  Excited cells: {analysis['n_excited']}")
        logger.info(f"  Suppressed cells: {analysis['n_suppressed']}")
        logger.info(f"  Unchanged cells: {analysis['n_unchanged']}")
        
        # Plot
        if save_path:
            save_file = str(output_dir / f'weights_by_response_{post_pop}.pdf')
        else:
            save_file = None
        
        fig = vis.plot_weights_by_response_type(
            analysis,
            save_path=save_file
        )
        
        if fig is not None:
            all_figures.append(fig)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed weights by response analysis")
    logger.info('='*60)
    
    return all_analyses, all_figures


def analyze_synaptic_weights_by_response_from_results(results: Dict,
                                                      metadata: Dict,
                                                      output_path: Path,
                                                      optimization_json_file: Optional[str] = None,
                                                      device: Optional[torch.device] = None):
    """
    Analyze and plot synaptic weights by response from saved results
    
    Args:
        results: Loaded experiment results
        metadata: Metadata from saved results
        output_path: Base output directory (e.g., Path('protocol'))
        optimization_json_file: Optional path to optimization file
        device: Device to create circuit on
    """
    
    # Reconstruct circuit
    circuit = reconstruct_circuit_from_metadata(
        metadata, 
        optimization_json_file,
        device
    )
    
    # Extract stimulation parameters from metadata
    stim_start = metadata.get('stim_start', 1500.0)
    stim_duration = metadata.get('stim_duration', 1000.0)
    baseline_start = metadata.get('warmup', 500.0)
    
    # Create base output directory for weights
    weights_base_dir = output_path / 'synaptic_weights'
    weights_base_dir.mkdir(exist_ok=True, parents=True)

    analyses_dict = {}
    figures_dict = {}
    
    # Plot for each target population
    for target_pop in ['pv', 'sst']:
        if target_pop not in results:
            logger.info(f"\nSkipping {target_pop.upper()} - no results found")
            continue
        
        # Use highest intensity for weight visualization
        intensities = sorted(results[target_pop].keys())
        if len(intensities) == 0:
            logger.info(f"\nSkipping {target_pop.upper()} - no intensities found")
            continue
        
        intensity = intensities[-1]  # Highest intensity
        experiment_data = results[target_pop][intensity]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {target_pop.upper()} stimulation at intensity {intensity}")
        logger.info('='*60)
        
        # Create subdirectory for this target population
        target_output_dir = weights_base_dir / f'{target_pop}_stimulation'

        all_analyses, all_figures = analyze_and_plot_weights_by_response(
            circuit,
            target_population=target_pop,
            experiment_results=experiment_data,
            stim_start=stim_start,
            stim_duration=stim_duration,
            warmup=baseline_start,
            save_path=target_output_dir)

        analyses_dict[target_pop] = all_analyses
        figures_dict[target_pop] = all_figures
        
    return analyses_dict, figures_dict


def run_nested_effect_size_analysis(
    nested_data: Dict,
    target_populations: List[str] = ['pv', 'sst'],
    post_populations: List[str] = ['gc', 'mc', 'pv', 'sst'],
    intensities: Optional[List[float]] = None,
    source_populations: List[str] = ['pv', 'sst', 'mc', 'mec'],
    n_bootstrap: int = 10000,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    random_seed: Optional[int] = None,
    output_dir: str = './effect_size_analysis',
    device: Optional[torch.device] = None
) -> Dict:
    """
    Run bootstrap effect size analysis from saved nested experiment results
    
    Args:
        nested_results: nested experiment results
        target_populations: Populations that were stimulated
        post_populations: Post-synaptic populations to analyze
        intensities: Light intensities to analyze (None = all available)
        source_populations: Source populations to analyze for weights
        n_bootstrap: Number of bootstrap samples
        threshold_std: Classification threshold (std deviations)
        expression_threshold: Opsin expression threshold for non-expressing cells
        random_seed: Random seed for reproducibility
        output_dir: Directory to save results and plots
        device: Device for circuit creation
        
    Returns:
        Dict with analysis results for each target/intensity/post combination
    """
    if device is None:
        device = get_default_device()
    
    print("\n" + "="*80)
    print("Nested Bootstrap Effect Size Analysis")
    print("="*80)

    # Check if this is HDF5 or pickle format
    hdf5_file_path = nested_data.get('hdf5_file')
    use_hdf5 = hdf5_file_path is not None
    
    if use_hdf5:
        print(f"  Loading from HDF5: {hdf5_file_path}")
        metadata = nested_data['metadata']
        seed_structure = nested_data['seed_structure']
    else:
        # Original pickle format
        nested_results = nested_data['nested_results']
        metadata = nested_data['metadata']
        seed_structure = nested_data['seed_structure']
    
    # Extract experiment parameters
    stim_start = metadata['stim_start']
    stim_duration = metadata['stim_duration']
    warmup = metadata['warmup']
    
    # Determine intensities to analyze
    if intensities is None:
        intensities = metadata['intensities']
    
    print(f"\nExperiment parameters:")
    print(f"  Stimulation: {stim_start} - {stim_start + stim_duration} ms")
    print(f"  Baseline: {warmup} - {stim_start} ms")
    print(f"  Intensities: {intensities}")
    print(f"  Bootstrap samples: {n_bootstrap:,}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Storage for all results
    all_analyses = {}
    # Open HDF5 file if needed
    hdf5_file = h5py.File(hdf5_file_path, 'r') if use_hdf5 else None
    
    try:
        # Analyze each target population
        for target in target_populations:
            # Check if target exists
            if use_hdf5:
                if target not in hdf5_file:
                    print(f"\nSkipping {target.upper()} - no results found")
                    continue
            else:
                if target not in nested_results:
                    print(f"\nSkipping {target.upper()} - no results found")
                    continue
            
            all_analyses[target] = {}
            
            print(f"\n{'='*60}")
            print(f"Analyzing {target.upper()} stimulation")
            print('='*60)
            
            for intensity in intensities:
                all_analyses[target][intensity] = {}
                
                print(f"\n  Intensity: {intensity}")
                
                # Get trials for this condition
                if use_hdf5:
                    trials = _load_trials_from_hdf5(hdf5_file, target, intensity,
                                                   metadata, expression_threshold)
                else:
                    trials = nested_results[target][intensity]
                
                print(f"    Total trials: {len(trials)}")
                print(f"    Connectivity instances: {len(set(t.connectivity_idx for t in trials))}")
                
                # Recreate circuit (use seed from first connectivity instance)
                first_conn_seed = seed_structure['connectivity_seeds'][0]
                
                print(f"    Recreating circuit with seed: {first_conn_seed}")
                
                # Import necessary classes
                from DG_circuit_dendritic_somatic_transfer import (
                    CircuitParams, PerConnectionSynapticParams, OpsinParams
                )
                
                circuit_params = CircuitParams()
                synaptic_params = PerConnectionSynapticParams()
                opsin_params = OpsinParams()
                
                # Apply optimization if available
                optimization_file = metadata.get('optimization_file')
                if optimization_file:
                    print(f"    Applying optimization from: {optimization_file}")
                
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
                    print(f"\n    Analyzing {target.upper()} -> {post_pop.upper()}")
                    
                    # Run bootstrap analysis
                    analysis_results = analyze_effect_size_all_sources_nested(
                        nested_results=trials,
                        circuit=experiment.circuit,
                        target_population=target,
                        post_population=post_pop,
                        source_populations=source_populations,
                        stim_start=stim_start,
                        stim_duration=stim_duration,
                        warmup=warmup,
                        n_bootstrap=n_bootstrap,
                        threshold_std=threshold_std,
                        expression_threshold=expression_threshold,
                        random_seed=random_seed
                    )
                    
                    if not analysis_results:
                        print(f"      No valid results for {post_pop.upper()}")
                        continue
                    
                    # Store results
                    all_analyses[target][intensity][post_pop] = analysis_results
                    
                    # Print summary
                    print_nested_effect_size_analysis_summary(
                        analysis_results,
                        target,
                        post_pop
                    )
                    
                    # Create visualizations
                    vis_dir = output_path / f"{target}_intensity_{intensity}" / post_pop
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Forest plot
                    print(f"      Generating forest plot...")
                    fig_forest = plot_effect_sizes_forest(
                        analysis_results,
                        target,
                        post_pop,
                        save_path=str(vis_dir / 'effect_sizes_forest.pdf')
                    )
                    plt.close(fig_forest)
                    
                    # Bootstrap distributions for each source
                    for source_pop in analysis_results.keys():
                        print(f"      Generating bootstrap plot for {source_pop.upper()}...")
                        fig_bootstrap = plot_bootstrap_distributions(
                            analysis_results,
                            source_pop,
                            target,
                            post_pop,
                            save_path=str(vis_dir / f'bootstrap_dist_{source_pop}.pdf')
                        )
                        plt.close(fig_bootstrap)
                        
                        # Weight distributions for first connectivity
                        try:
                            fig_weights = plot_weight_distributions_by_response(
                                analysis_results,
                                source_pop,
                                connectivity_idx=0,
                                target_population=target,
                                post_population=post_pop,
                                save_path=str(vis_dir / f'weight_distributions_{source_pop}_conn0.pdf')
                            )
                            plt.close(fig_weights)
                        except Exception as e:
                            print(f"        Warning: Could not plot weight distributions for {source_pop}: {e}")
                
                # Clean up circuit
                del experiment
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    finally:
        # Close HDF5 file
        if hdf5_file is not None:
            hdf5_file.close()
    
    # Save complete analysis results
    analysis_file = output_path / 'bootstrap_analysis_results.pkl'
    with open(analysis_file, 'wb') as f:
        pickle.dump({
            'all_analyses': all_analyses,
            'metadata': metadata,
            'parameters': {
                'n_bootstrap': n_bootstrap,
                'threshold_std': threshold_std,
                'expression_threshold': expression_threshold,
                'random_seed': random_seed
            }
        }, f)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete. Results saved to: {output_path}")
    print(f"  Summary file: {analysis_file}")
    print('='*80)
    
    return all_analyses


def _load_trials_from_hdf5(hdf5_file: h5py.File,
                           target: str,
                           intensity: float,
                           metadata: Dict,
                           expression_threshold: float = 0.2) -> List:
    """
    Load all trials for a specific target/intensity from HDF5
    
    Returns list of NestedTrialResult objects.
    """
    from nested_experiment import NestedTrialResult
    from hdf5_storage import load_trial_from_hdf5
    
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in hdf5_file[target]:
        return []
    
    trials = []
    n_connectivity = metadata['n_connectivity_instances']
    n_mec_patterns = metadata['n_mec_patterns_per_connectivity']
    
    for conn_idx in range(n_connectivity):
        for pattern_idx in range(n_mec_patterns):
            try:
                trial_data = load_trial_from_hdf5(
                    hdf5_file, target, intensity, conn_idx, pattern_idx
                )
                
                # Create NestedTrialResult
                trial_result = NestedTrialResult(
                    connectivity_idx=conn_idx,
                    mec_pattern_idx=pattern_idx,
                    seed=0,  # Not critical for analysis
                    results=trial_data,
                    target_population=trial_data['target_population'],
                    opsin_expression=trial_data['opsin_expression']
                )
                
                trials.append(trial_result)
                
            except Exception as e:
                logger.warning(f"Could not load trial conn={conn_idx}, pattern={pattern_idx}: {e}")
                continue
    
    return trials



def run_nested_effect_size_decision_analysis(
    nested_data: Dict,
    target_populations: List[str] = ['pv', 'sst'],
    post_populations: List[str] = ['gc', 'mc'],
    intensities: Optional[List[float]] = None,
    source_populations: List[str] = ['pv', 'sst', 'mc', 'mec'],
    n_bootstrap: int = 10000,
    threshold_std: float = 1.0,
    expression_threshold: float = 0.2,
    current_n: Optional[int] = None,
    target_power: float = 0.80,
    max_feasible_n: int = 20,
    min_meaningful_effect: float = 0.5,
    min_meaningful_diff_nS: float = 0.1,
    random_seed: Optional[int] = None,
    output_dir: str = './effect_size_decision',
    device: Optional[torch.device] = None
) -> Dict:
    """
    Run effect size decision analysis from nested experiment results.
    
    This function integrates bootstrap effect size analysis with power analysis,
    precision assessment, and biological significance evaluation to make data
    collection recommendations.
    
    Args:
        nested_data: Nested experiment data dict with keys:
            - 'nested_results': Trial data
            - 'metadata': Experiment metadata
            - 'seed_structure': RNG seed information
            - 'variance_analysis': Variance decomposition results (optional)
        target_populations: Populations that were stimulated
        post_populations: Post-synaptic populations to analyze
        intensities: Light intensities to analyze (None = all available)
        source_populations: Source populations for weight analysis
        n_bootstrap: Number of bootstrap samples
        threshold_std: Classification threshold (std deviations)
        expression_threshold: Opsin expression threshold
        current_n: Current number of connectivity instances (auto-detected if None)
        target_power: Desired statistical power
        max_feasible_n: Maximum feasible sample size
        min_meaningful_effect: Minimum biologically meaningful effect size
        min_meaningful_diff_nS: Minimum biologically meaningful weight difference
        random_seed: Random seed for reproducibility
        output_dir: Directory to save results and plots
        device: Device for circuit creation
        
    Returns:
        Dict with complete analysis results for each target/intensity/post combination
    """
    if device is None:
        device = get_default_device()
    
    print("\n" + "="*80)
    print("Nested effect size decision framework")
    print("="*80)
    
    # Extract experiment data
    nested_results = nested_data['nested_results']
    metadata = nested_data['metadata']
    seed_structure = nested_data['seed_structure']
    
    # Extract experiment parameters
    stim_start = metadata['stim_start']
    stim_duration = metadata['stim_duration']
    warmup = metadata['warmup']
    
    # Determine intensities to analyze
    if intensities is None:
        intensities = metadata['intensities']
    
    # Auto-detect current N if not specified
    if current_n is None:
        # Get from seed structure
        current_n = len(seed_structure['connectivity_seeds'])
    
    print(f"\nExperiment parameters:")
    print(f"  Stimulation: {stim_start} - {stim_start + stim_duration} ms")
    print(f"  Baseline: {warmup} - {stim_start} ms")
    print(f"  Intensities: {intensities}")
    print(f"  Current N: {current_n} connectivity instances")
    print(f"  Bootstrap samples: {n_bootstrap:,}")
    print(f"  Target power: {target_power:.0%}")
    print(f"  Max feasible N: {max_feasible_n}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for variance decomposition results (ICC analysis)
    variance_analysis = nested_data.get('variance_analysis', None)
    regime_classification = nested_data.get('regime_classification', None)
    
    # Storage for all results
    all_analyses = {}
    
    # Analyze each target population
    for target in target_populations:
        if target not in nested_results:
            print(f"\nSkipping {target.upper()} - no results found")
            continue
        
        all_analyses[target] = {}
        
        print(f"\n{'='*60}")
        print(f"Analyzing {target.upper()} stimulation")
        print('='*60)
        
        for intensity in intensities:
            if intensity not in nested_results[target]:
                continue
            
            all_analyses[target][intensity] = {}
            
            print(f"\n  Intensity: {intensity}")
            
            # Get trials for this condition
            trials = nested_results[target][intensity]
            
            print(f"    Total trials: {len(trials)}")
            print(f"    Connectivity instances: {len(set(t.connectivity_idx for t in trials))}")
            
            # Recreate circuit (use seed from first connectivity instance)
            first_conn_seed = seed_structure['connectivity_seeds'][0]
            
            print(f"    Recreating circuit with seed: {first_conn_seed}")
            
            # Create circuit with optimization if available
            from DG_circuit_dendritic_somatic_transfer import (
                CircuitParams, PerConnectionSynapticParams, OpsinParams
            )
            
            circuit_params = CircuitParams()
            synaptic_params = PerConnectionSynapticParams()
            opsin_params = OpsinParams()
            
            optimization_file = metadata.get('optimization_file')
            if optimization_file:
                print(f"    Applying optimization from: {optimization_file}")
            
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
                print(f"\n    Analyzing {target.upper()} -> {post_pop.upper()}")
                
                # Create output directory for this combination
                combo_dir = output_path / f"{target}_intensity_{intensity}" / post_pop
                combo_dir.mkdir(parents=True, exist_ok=True)
                
                # Compute bootstrap effect sizes
                print(f"      Computing bootstrap effect sizes...")
                bootstrap_results = analyze_effect_size_all_sources_nested(
                    nested_results=trials,
                    circuit=experiment.circuit,
                    target_population=target,
                    post_population=post_pop,
                    source_populations=source_populations,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    warmup=warmup,
                    n_bootstrap=n_bootstrap,
                    threshold_std=threshold_std,
                    expression_threshold=expression_threshold,
                    random_seed=random_seed
                )
                
                if not bootstrap_results:
                    print(f"      No valid results for {post_pop.upper()}")
                    continue
                
                # Extract ICC if available
                icc_value = None
                if variance_analysis is not None:
                    if target in variance_analysis and intensity in variance_analysis[target]:
                        if post_pop in variance_analysis[target][intensity]:
                            var_results = variance_analysis[target][intensity][post_pop]
                            icc_value = var_results.get('icc', None)
                
                # Run complete decision analysis
                analysis_results = run_effect_size_decision_analysis(
                    bootstrap_results=bootstrap_results,
                    target_population=target,
                    post_population=post_pop,
                    current_n=current_n,
                    target_power=target_power,
                    max_feasible_n=max_feasible_n,
                    min_meaningful_effect=min_meaningful_effect,
                    min_meaningful_diff_nS=min_meaningful_diff_nS,
                    icc_value=icc_value,
                    save_dir=str(combo_dir)
                )
                
                # Store results
                all_analyses[target][intensity][post_pop] = analysis_results
                
                # Print summary
                print_nested_effect_size_analysis_summary(
                    analysis_results['bootstrap_results'],
                    target,
                    post_pop
                )
            
            # Clean up circuit
            del experiment
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Save complete analysis results
    analysis_file = output_path / 'effect_size_decision_analysis_results.pkl'
    with open(analysis_file, 'wb') as f:
        pickle.dump({
            'all_analyses': all_analyses,
            'metadata': metadata,
            'parameters': {
                'n_bootstrap': n_bootstrap,
                'threshold_std': threshold_std,
                'expression_threshold': expression_threshold,
                'current_n': current_n,
                'target_power': target_power,
                'max_feasible_n': max_feasible_n,
                'min_meaningful_effect': min_meaningful_effect,
                'min_meaningful_diff_nS': min_meaningful_diff_nS,
                'random_seed': random_seed
            }
        }, f)
    
    # Generate cross-population summary report
    print_cross_population_summary(all_analyses, output_path)
    
    print(f"\n{'='*80}")
    print(f"Decision analysis complete. Results saved to: {output_path}")
    print(f"  Summary file: {analysis_file}")
    print('='*80)
    
    return all_analyses


def print_cross_population_summary(
    all_analyses: Dict,
    output_path: Path
):
    """
    Print and save cross-population summary of recommendations
    
    Args:
        all_analyses: Complete analysis results
        output_path: Directory to save summary
    """
    print("\n" + "="*80)
    print("Cross-population effect size decision summary")
    print("="*80)
    
    summary_lines = []
    summary_lines.append("\n" + "="*80)
    summary_lines.append("Cross-population Effect Size Decision Summary")
    summary_lines.append("="*80)
    
    # Organize by intensity
    for target in sorted(all_analyses.keys()):
        for intensity in sorted(all_analyses[target].keys()):
            
            header = f"\n{target.upper()} Stimulation at Intensity {intensity}"
            print(header)
            summary_lines.append(header)
            
            divider = "-"*80
            print(divider)
            summary_lines.append(divider)
            
            table_header = f"{'Post Pop':<10} {'Source':<8} {'d':<8} {'Power':<8} {'Action':<15} {'Add N':<8} {'Priority':<10}"
            print(table_header)
            summary_lines.append(table_header)
            
            print(divider)
            summary_lines.append(divider)
            
            for post_pop in sorted(all_analyses[target][intensity].keys()):
                results = all_analyses[target][intensity][post_pop]
                recommendations = results['recommendations']
                power_analysis = results['power_analysis']
                bootstrap_results = results['bootstrap_results']
                
                for source_pop in sorted(recommendations.keys()):
                    rec = recommendations[source_pop]
                    power = power_analysis[source_pop]
                    boot = bootstrap_results[source_pop]['bootstrap_results']
                    
                    line = (f"{post_pop.upper():<10} {source_pop.upper():<8} "
                           f"{boot['effect_size']:>6.2f}  {power.current_power:>6.2f}  "
                           f"{rec.action:<15} {rec.additional_n_needed:>6}  "
                           f"{rec.priority:<10}")
                    print(line)
                    summary_lines.append(line)
    
    # Overall recommendations
    print("\n" + "="*80)
    print("Overall Recommendations")
    print("="*80)
    summary_lines.append("\n" + "="*80)
    summary_lines.append("Overall Recommendations")
    summary_lines.append("="*80)
    
    # Find maximum additional N needed across all conditions
    max_add_n = 0
    high_priority_items = []
    medium_priority_items = []
    
    for target in all_analyses:
        for intensity in all_analyses[target]:
            for post_pop in all_analyses[target][intensity]:
                recommendations = all_analyses[target][intensity][post_pop]['recommendations']
                for source_pop, rec in recommendations.items():
                    max_add_n = max(max_add_n, rec.additional_n_needed)
                    
                    if rec.priority == 'HIGH':
                        high_priority_items.append(
                            (target, intensity, post_pop, source_pop, rec.additional_n_needed)
                        )
                    elif rec.priority == 'MEDIUM':
                        medium_priority_items.append(
                            (target, intensity, post_pop, source_pop, rec.additional_n_needed)
                        )
    
    if high_priority_items:
        msg = f"\nHIGH PRIORITY: Add {max_add_n} connectivity instances"
        print(msg)
        summary_lines.append(msg)
        
        print("\nEffects requiring additional data:")
        summary_lines.append("\nEffects requiring additional data:")
        
        for target, intensity, post_pop, source_pop, add_n in high_priority_items:
            item = f"  • {target.upper()} → {post_pop.upper()} (from {source_pop.upper()}): +{add_n}"
            print(item)
            summary_lines.append(item)
    
    elif medium_priority_items:
        msg = f"\nMEDIUM PRIORITY: Consider adding {max_add_n} connectivity instances"
        print(msg)
        summary_lines.append(msg)
        
        print("\nEffects that would benefit from more data:")
        summary_lines.append("\nEffects that would benefit from more data:")
        
        for target, intensity, post_pop, source_pop, add_n in medium_priority_items:
            item = f"  • {target.upper()} → {post_pop.upper()} (from {source_pop.upper()}): +{add_n}"
            print(item)
            summary_lines.append(item)
    
    else:
        msg = "\nNO ADDITIONAL DATA NEEDED"
        print(msg)
        summary_lines.append(msg)
        
        msg2 = "All effects are either well-characterized or not biologically meaningful"
        print(msg2)
        summary_lines.append(msg2)
    
    print("\n" + "="*80)
    summary_lines.append("\n" + "="*80)
    
    # Save summary to file
    summary_file = output_path / 'effect_size_decision_summary.txt'
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nSummary saved to: {summary_file}")

    
if __name__ == "__main__":
    logger.info("Dentate Gyrus Optogenetic Protocol")
    logger.info("="*80)
    logger.info("\nFor analysis of saved results, use: python DG_analysis.py --help\n")
    logger.info("="*80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Run DG optogenetic experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comparative experiment
  %(prog)s --comparative --n-trials 5
  
  # Run nested experiment  
  %(prog)s --nested --n-connectivity 10 --n-mec-patterns 5
  
  # Run ablation tests
  %(prog)s --ablations --n-trials 3
  
  # Run all experiments
  %(prog)s --all --n-trials 5
  
  # Analyze results (separate script)
  python DG_analysis.py plot-comparative results.pkl
  python DG_analysis.py bootstrap-analysis nested_results.pkl

For detailed analysis options, see: python DG_analysis.py --help
        """
    )
    
    # Experiment selection
    parser.add_argument('--comparative', action='store_true',
                        help='Run comparative PV vs SST experiment')
    parser.add_argument('--nested', action='store_true',
                        help='Run nested (connectivity x MEC) experiment (implies --save-full-activity)')
    parser.add_argument('--ablations', action='store_true',
                        help='Run ablation tests')
    parser.add_argument('--nested-file', type=str, default=None,
                        metavar='PATH',
                        help='Path to nested experiment HDF5 file. When specified, '
                        'ablation tests use connectivity seeds and opsin '
                        'expressions from this file for paired comparisons.')
    
    parser.add_argument('--expression', action='store_true',
                        help='Run expression level tests')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='protocol',
                       metavar='DIR',
                       help='Output directory for results (default: protocol)')
    
    # Common simulation parameters
    parser.add_argument('--n-trials', type=int, default=3,
                       help='Number of trials to average (default: 3)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--regenerate-connectivity', action='store_true',
                       help='Regenerate circuit connectivity for each trial')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', None],
                       help='Device to run on (default: auto-detect)')
    
    # Circuit and stimulation parameters
    parser.add_argument('--optimization-file', type=str, default=None,
                       help='Path to optimization JSON file')
    parser.add_argument('--mec-current', type=float, default=40.0,
                       help='MEC drive current in pA (default: 40.0)')
    parser.add_argument('--opsin-current', type=float, default=200.0,
                       help='Optogenetic current in pA (default: 200.0)')
    parser.add_argument('--stim-start', type=float, default=1500.0,
                       help='Stimulus start time [ms] (default: 1500.0)')
    parser.add_argument('--stim-duration', type=float, default=1000.0,
                       help='Stimulus duration [ms] (default: 1000.0)')
    
    # Data saving options
    parser.add_argument('--no-auto-save', action='store_true',
                        help='Disable automatic saving of results')
    parser.add_argument('--save-full-activity', action='store_true',
                        help='Save complete activity traces (larger files)')
    parser.add_argument('--record-currents', action='store_true',
                        help='Record synaptic currents')
    parser.add_argument('--save-results-file', type=str, default=None,
                        help='Path to results file')
    
    # Adaptive stepping
    parser.add_argument('--adaptive-step', action='store_true',
                       help='Use gradient-driven adaptive time stepping')
    parser.add_argument('--adaptive-dt-min', type=float, default=0.05,
                       help='Minimum time step for adaptive stepping (ms)')
    parser.add_argument('--adaptive-dt-max', type=float, default=0.25,
                       help='Maximum time step for adaptive stepping (ms)')
    parser.add_argument('--adaptive-gradient-low', type=float, default=0.5,
                       help='Low gradient threshold (Hz/ms)')
    parser.add_argument('--adaptive-gradient-high', type=float, default=10.0,
                       help='High gradient threshold (Hz/ms)')
    
    # MEC input patterns
    parser.add_argument('--time-varying-mec', action='store_true',
                       help='Enable time-varying MEC input')
    parser.add_argument('--mec-pattern-type', type=str, default='oscillatory',
                       choices=['oscillatory', 'drift', 'noisy', 'constant'],
                       help='Type of temporal pattern for MEC input')
    parser.add_argument('--mec-theta-freq', type=float, default=5.0,
                       help='Theta oscillation frequency (Hz)')
    parser.add_argument('--mec-theta-amplitude', type=float, default=0.3,
                       help='Theta modulation depth (0-1)')
    parser.add_argument('--mec-gamma-freq', type=float, default=20.0,
                       help='Gamma oscillation frequency (Hz)')
    parser.add_argument('--mec-gamma-amplitude', type=float, default=0.15,
                       help='Gamma modulation depth (0-1)')
    parser.add_argument('--mec-gamma-coupling', type=float, default=0.8,
                       help='Gamma-theta coupling strength (0-1)')
    parser.add_argument('--mec-gamma-phase', type=float, default=0.0,
                       help='Preferred theta phase for gamma peak (radians)')
    parser.add_argument('--mec-rotation-groups', type=int, default=3,
                       help='Number of groups for spatial rotation')
    
    # Nested experiment options
    nested_group = parser.add_argument_group('Nested Experiment Options')
    nested_group.add_argument('--n-connectivity', type=int, default=5,
                             help='Number of connectivity instances (default: 5)')
    nested_group.add_argument('--n-mec-patterns', type=int, default=3,
                             help='Number of MEC patterns per connectivity (default: 3)')
    
    # Expression level test options
    parser.add_argument('--expression-levels', type=float, nargs='+',
                       default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       help='Expression levels to test (default: 0.1 - 1.0 at 0.1 increments)')
    
    args = parser.parse_args()
    
    # Validate that at least one experiment type is selected
    if not any([args.comparative, args.nested, args.ablations, args.expression, args.all]):
        parser.error("No experiment selected. Use --comparative, --nested, --ablations, "
                    "--expression, or --all")
    
    # Setup
    output_dir = args.output_dir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Auto-detect device
    if args.device is None:
        device = get_default_device()
    else:
        device = torch.device(args.device)
    
    logger.info(f"\nUsing device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # Create adaptive config if requested
    adaptive_config = None
    if args.adaptive_step:
        adaptive_config = GradientAdaptiveStepConfig(
            dt_min=args.adaptive_dt_min,
            dt_max=args.adaptive_dt_max,
            gradient_low=args.adaptive_gradient_low,
            gradient_high=args.adaptive_gradient_high,
        )
    
    logger.info(f"\nExperiment configuration:")
    logger.info(f"  Trials per condition: {args.n_trials}")
    logger.info(f"  Base seed: {args.base_seed}")
    logger.info(f"  Auto-save: {not args.no_auto_save}")
    logger.info(f"  Output directory: {output_path}")
    
    # =========================================================================
    # RUN EXPERIMENTS
    # =========================================================================
    
    # Nested experiment
    if args.nested or args.all:
        logger.info("\n" + "="*80)
        logger.info("Running Nested Experiment (Connectivity x MEC Patterns)")
        logger.info("="*80)

        nested_config = NestedExperimentConfig(
            n_connectivity_instances=args.n_connectivity,
            n_mec_patterns_per_connectivity=args.n_mec_patterns,
            base_seed=args.base_seed,
            save_nested_trials=True,
            save_full_activity=True
        )

        # Determine save file path
        if args.save_results_file:
            save_file = args.save_results_file
        else:
            save_file = str(output_path / 'nested_experiment_results.h5')
        
        nested_results = run_nested_comparative_experiment(
            optimization_json_file=args.optimization_file,
            intensities=[1.0, 1.5],
            mec_current=args.mec_current,
            opsin_current=args.opsin_current,
            stim_start=args.stim_start,
            stim_duration=args.stim_duration,
            warmup=500.0,
            device=device,
            nested_config=nested_config,
            save_results_file=save_file,
            use_time_varying_mec=args.time_varying_mec,
            mec_pattern_type=args.mec_pattern_type,
            mec_theta_freq=args.mec_theta_freq,
            mec_theta_amplitude=args.mec_theta_amplitude,
            mec_gamma_freq=args.mec_gamma_freq,
            mec_gamma_amplitude=args.mec_gamma_amplitude,
            mec_gamma_coupling_strength=args.mec_gamma_coupling,
            mec_gamma_preferred_phase=args.mec_gamma_phase,
            mec_rotation_groups=args.mec_rotation_groups,
            adaptive_config=adaptive_config
        )
        
        save_nested_experiment_summary(nested_results, output_dir=output_path)
    
    # Comparative experiment
    if args.comparative or args.all:
        logger.info("\n" + "="*80)
        logger.info("Running Comparative Experiment (PV vs SST)")
        logger.info("="*80)
        
        experiment, results, conn_analysis, cond_analysis = run_comparative_experiment(
            optimization_json_file=args.optimization_file,
            intensities=[0.5, 1.0, 1.5],
            mec_current=args.mec_current,
            opsin_current=args.opsin_current,
            device=device,
            n_trials=args.n_trials,
            regenerate_connectivity_per_trial=args.regenerate_connectivity,
            plot_baseline_normalize=True,
            base_seed=args.base_seed,
            stim_start=args.stim_start,
            stim_duration=args.stim_duration,
            auto_save=not args.no_auto_save,
            save_full_activity=args.save_full_activity,
            record_currents=args.record_currents,
            adaptive_step=args.adaptive_step,
            adaptive_config=adaptive_config,
            use_time_varying_mec=args.time_varying_mec,
            mec_pattern_type=args.mec_pattern_type,
            mec_theta_freq=args.mec_theta_freq,
            mec_theta_amplitude=args.mec_theta_amplitude,
            mec_gamma_freq=args.mec_gamma_freq,
            mec_gamma_amplitude=args.mec_gamma_amplitude,
            mec_gamma_coupling_strength=args.mec_gamma_coupling,
            mec_gamma_preferred_phase=args.mec_gamma_phase,
            mec_rotation_groups=args.mec_rotation_groups
        )
    
    # Ablation tests
    if args.ablations or args.all:
        logger.info("\n" + "="*80)
        logger.info("Running Ablation Tests")
        logger.info("="*80)
        
        ablation_output = output_path / "ablation_tests"
        ablation_results = run_all_ablation_tests(
            optimization_json_file=args.optimization_file,
            intensities=[0.5, 1.0, 1.5],
            mec_current=args.mec_current,
            opsin_current=args.opsin_current,
            stim_start=args.stim_start,
            stim_duration=args.stim_duration,
            device=device,
            n_trials=args.n_trials,
            base_seed=args.base_seed,
            output_dir=str(ablation_output),
            use_time_varying_mec=args.time_varying_mec,
            mec_pattern_type=args.mec_pattern_type,
            mec_theta_freq=args.mec_theta_freq,
            mec_theta_amplitude=args.mec_theta_amplitude,
            mec_gamma_freq=args.mec_gamma_freq,
            mec_gamma_amplitude=args.mec_gamma_amplitude,
            mec_gamma_coupling_strength=args.mec_gamma_coupling,
            mec_gamma_preferred_phase=args.mec_gamma_phase,
            mec_rotation_groups=args.mec_rotation_groups,
            nested_experiment_file=args.nested_file
        )
    
    # Expression level tests
    if args.expression or args.all:
        logger.info("\n" + "="*80)
        logger.info("Running Expression Level Tests")
        logger.info("="*80)
        
        expression_output = output_path / "expression_tests"
        expression_output.mkdir(exist_ok=True)
        
        expression_results = test_opsin_expression_levels(
            optimization_json_file=args.optimization_file,
            target_populations=['pv', 'sst'],
            expression_levels=args.expression_levels,
            intensity=1.5,
            mec_current=args.mec_current,
            opsin_current=args.opsin_current,
            stim_start=args.stim_start,
            stim_duration=args.stim_duration,
            device=device,
            n_trials=args.n_trials,
            base_seed=args.base_seed,
            include_ablations=True,
            save_results_file=str(expression_output / "expression_results.pkl"),
            nested_experiment_file=args.nested_file,
            use_time_varying_mec=args.time_varying_mec,
            mec_pattern_type=args.mec_pattern_type,
            mec_theta_freq=args.mec_theta_freq,
            mec_theta_amplitude=args.mec_theta_amplitude,
            mec_gamma_freq=args.mec_gamma_freq,
            mec_gamma_amplitude=args.mec_gamma_amplitude,
            mec_gamma_coupling_strength=args.mec_gamma_coupling,
            mec_gamma_preferred_phase=args.mec_gamma_phase,
            mec_rotation_groups=args.mec_rotation_groups
        )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("All experiments complete!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("\nTo analyze results, use DG_analysis.py:")
    logger.info("  python DG_analysis.py plot-comparative results.pkl")
    logger.info("  python DG_analysis.py plot-ablations ablation_tests/all_ablation_tests.pkl")
    logger.info("  python DG_analysis.py bootstrap-analysis nested_experiment_results.pkl")
    logger.info("\nFor all analysis options: python DG_analysis.py --help")
    logger.info("="*80)
