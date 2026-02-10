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
from nested_experiment import (
    load_nested_experiment_seeds_and_opsin
)
     

def test_interneuron_interactions(
    optimization_json_file: Optional[str] = None,
    target_populations: List[str] = ['pv', 'sst'],
    intensities: List[float] = [0.5, 1.0, 2.0],
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    plot_activity: bool = True,
    device: Optional[torch.device] = None,
    n_trials: int = 1,
    base_seed: int = 42,
    save_results_file: Optional[str] = None,
    connectivity_seeds: Optional[List[int]] = None,
    opsin_expressions: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    **optogenetic_experiment_kwargs
) -> Dict:
    """
    Test role of interneuron-interneuron interactions in paradoxical excitation.

    Blocks PV<->SST, PV<->PV, SST<->SST connections.

    When connectivity_seeds and opsin_expressions are provided (paired mode),
    runs each condition on the same connectivity instances for paired comparison.
    Otherwise falls back to original unpaired behavior.
    """
    if device is None:
        device = get_default_device()

    paired_mode = (connectivity_seeds is not None and opsin_expressions is not None)

    logger.info("\n" + "=" * 80)
    logger.info("TEST: Interneuron-Interneuron Interaction Hypothesis")
    logger.info("=" * 80)
    if paired_mode:
        logger.info(f"  Mode: PAIRED ({len(connectivity_seeds)} connectivity instances)")
    else:
        logger.info(f"  Mode: UNPAIRED (base_seed={base_seed}, n_trials={n_trials})")

    # Define blocked synaptic parameters
    circuit_params = CircuitParams()
    opsin_params_obj = OpsinParams()
    base_synaptic_params = PerConnectionSynapticParams()

    blocked_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params.ampa_g_mean,
        ampa_g_std=base_synaptic_params.ampa_g_std,
        ampa_g_min=base_synaptic_params.ampa_g_min,
        ampa_g_max=base_synaptic_params.ampa_g_max,
        gaba_g_mean=base_synaptic_params.gaba_g_mean,
        gaba_g_std=base_synaptic_params.gaba_g_std,
        gaba_g_min=0.0,
        gaba_g_max=base_synaptic_params.gaba_g_max,
        distribution=base_synaptic_params.distribution,
        connection_modulation={
            **base_synaptic_params.connection_modulation,
            'pv_pv': 0.0,
            'pv_sst': 0.0,
            'sst_pv': 0.0,
            'sst_sst': 0.0,
        }
    )

    results = {
        'full_network': {},
        'blocked_int_int': {},
    }

    if paired_mode:
        # ---- Paired mode ----
        for target in target_populations:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing {target.upper()} stimulation (paired)")
            logger.info('=' * 60)

            target_opsin = opsin_expressions.get(target, {})

            # Full network
            logger.info("\n  1. Full Network (Control)")
            results['full_network'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=base_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Full]',
                plot_activity=False,
                **optogenetic_experiment_kwargs,
            )

            # Blocked int-int (SAME connectivity seeds!)
            logger.info("\n  2. Blocked Int-Int Connections")
            results['blocked_int_int'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=blocked_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Blocked Int-Int]',
                plot_activity=False,
                **optogenetic_experiment_kwargs,
            )

    else:
        for target in target_populations:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {target.upper()} stimulation")
            logger.info('='*60)

            # Run full network (control)
            logger.info("\n1. Full Network (Control)")
            logger.info("-"*60)
            experiment_full = OptogeneticExperiment(
                circuit_params, base_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed,
                **optogenetic_experiment_kwargs
            )

            full_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_full.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_activity and intensity == intensities[-1]),
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                # Analyze results
                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}

                # First, analyze ALL non-target populations (including non-target interneurons)
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue  # Will handle target population separately below

                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                    # Get trial-to-trial variability
                    if n_trials > 1:
                        excited_fractions = []
                        for trial_result in result['trial_results']:
                            trial_activity = trial_result['activity_trace'][pop]
                            trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                            trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                            trial_change = trial_stim - trial_baseline
                            trial_baseline_std = torch.std(trial_baseline)
                            trial_excited = torch.mean((trial_change > trial_baseline_std).float()).item()
                            excited_fractions.append(trial_excited)

                        analysis[f'{pop}_excited_std'] = np.std(excited_fractions)

                # Second, analyze NON-EXPRESSING cells from the TARGET population
                # These are cells that failed to express opsin and were not directly stimulated
                target_baseline = torch.mean(activity_mean[target][:, baseline_mask], dim=1)
                target_stim = torch.mean(activity_mean[target][:, stim_mask], dim=1)
                target_change = target_stim - target_baseline
                target_baseline_std = torch.std(target_baseline)

                # Get non-stimulated indices from the first trial (same across trials if not regenerating opsin)
                non_stimulated_indices = result['trial_results'][0]['non_stimulated_indices']
                n_non_expressing = len(non_stimulated_indices)

                if n_non_expressing > 0:
                    # Analyze only non-expressing cells
                    non_expr_change = target_change[non_stimulated_indices]
                    non_expr_excited = torch.mean((non_expr_change > target_baseline_std).float())

                    analysis[f'{target}_nonexpr_excited'] = non_expr_excited.item()
                    analysis[f'{target}_nonexpr_mean_change'] = torch.mean(non_expr_change).item()
                    analysis[f'{target}_nonexpr_count'] = n_non_expressing

                    # Get trial-to-trial variability for non-expressing target cells
                    if n_trials > 1:
                        non_expr_excited_fractions = []
                        for trial_result in result['trial_results']:
                            trial_activity = trial_result['activity_trace'][target]
                            trial_non_stim_idx = trial_result['non_stimulated_indices']

                            trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                            trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                            trial_change = trial_stim - trial_baseline
                            trial_baseline_std = torch.std(trial_baseline)

                            if len(trial_non_stim_idx) > 0:
                                trial_non_expr_change = trial_change[trial_non_stim_idx]
                                trial_excited = torch.mean((trial_non_expr_change > trial_baseline_std).float()).item()
                                non_expr_excited_fractions.append(trial_excited)

                        analysis[f'{target}_nonexpr_excited_std'] = np.std(non_expr_excited_fractions)
                else:
                    # All cells express opsin - no non-expressing cells
                    analysis[f'{target}_nonexpr_excited'] = 0.0
                    analysis[f'{target}_nonexpr_mean_change'] = 0.0
                    analysis[f'{target}_nonexpr_count'] = 0
                    if n_trials > 1:
                        analysis[f'{target}_nonexpr_excited_std'] = 0.0

                # Store opsin expression statistics
                opsin_expr = result['opsin_expression_mean']
                expressing_mask = opsin_expr >= 0.2  # Same threshold as in code
                analysis[f'{target}_expression_fraction'] = torch.mean(expressing_mask.float()).item()
                analysis[f'{target}_mean_expression'] = torch.mean(opsin_expr).item()

                full_results[intensity] = analysis

            results['full_network'][target] = full_results

            # Run with blocked interneuron interactions
            logger.info("\n2. Blocked Interneuron-Interneuron Connections")
            logger.info("-"*60)
            experiment_blocked = OptogeneticExperiment(
                circuit_params, blocked_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed + 1000,  # Different connectivity
                **optogenetic_experiment_kwargs
            )

            blocked_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_blocked.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_activity and intensity == intensities[-1]),
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue
                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                blocked_results[intensity] = analysis

            results['blocked_int_int'][target] = blocked_results

            # Print comparison
            logger.info(f"\n{'='*60}")
            logger.info(f"SUMMARY: {target.upper()} stimulation at intensity {intensities[-1]}")
            logger.info('='*60)
            logger.info(f"{'Population':<12} {'Full Network':<20} {'Blocked Int-Int':<20} {'Change':<15}")
            logger.info('-'*60)

            for pop in ['gc', 'mc', 'pv', 'sst']:
                if pop == target:
                    continue
                full_exc = full_results[intensities[-1]][f'{pop}_excited']
                blocked_exc = blocked_results[intensities[-1]][f'{pop}_excited']
                change = blocked_exc - full_exc
                logger.info(f"{pop.upper():<12} {full_exc:>6.1%}              {blocked_exc:>6.1%}              {change:>+6.1%}")
            
    if save_results_file:
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to: {save_results_file}")

    return results


def test_excitation_to_interneurons(
    optimization_json_file: Optional[str] = None,
    target_populations: List[str] = ['pv', 'sst'],
    intensities: List[float] = [0.5, 1.0, 2.0],
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    plot_activity: bool = True,
    device: Optional[torch.device] = None,
    n_trials: int = 1,
    base_seed: int = 42,
    save_results_file: Optional[str] = None,
    connectivity_seeds: Optional[List[int]] = None,
    opsin_expressions: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    **optogenetic_experiment_kwargs
) -> Dict:
    """
    Test disinhibition hypothesis by blocking excitation to interneurons
    
    This is the cleanest test of the disinhibition hypothesis. By blocking all
    excitatory inputs to interneurons (MEC->PV, GC->PV, MC->PV, GC->SST, MC->SST)
    while keeping excitation to principal cells intact, we can test if paradoxical
    excitation requires recruitment of the inhibitory network.
    
    Prediction: If paradoxical excitation is mediated by disinhibition (i.e.,
               activating one interneuron population reduces inhibition from
               another), then blocking excitatory inputs to interneurons should
               eliminate the effect.
    
    Args:
        optimization_json_file: Path to optimization results (optional)
        target_populations: List of populations to stimulate
        intensities: List of light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: When to start stimulation (ms)
        stim_duration: Duration of stimulation (ms)
        warmup: Pre-stimulation period (ms)
        plot_activity: Whether to plot activity traces
        device: Device to run on (None for auto-detect)
        n_trials: Number of trials to average
        base_seed: Base random seed
        save_results_file: If provided, save results to this file
        
    Returns:
        Dictionary with results for each target population
    """
    
    if device is None:
        device = get_default_device()
    

    paired_mode = (connectivity_seeds is not None and opsin_expressions is not None)
    
    logger.info("\n" + "="*80)
    logger.info("TEST: Disinhibition Hypothesis")
    logger.info("="*80)
    logger.info("\nBlocking connections:")
    logger.info("  - MEC -> PV (blocks external excitation to PV)")
    logger.info("  - GC -> PV (blocks local excitation to PV)")
    logger.info("  - MC -> PV (blocks distant excitation to PV)")
    logger.info("  - GC -> SST (blocks local excitation to SST)")
    logger.info("  - MC -> SST (blocks distant excitation to SST)")
    logger.info("\nKept intact:")
    logger.info("  - All excitation to principal cells (GC, MC)")
    logger.info("  - All inhibition from interneurons")
    logger.info("  - Direct optogenetic activation of target interneurons")
    if paired_mode:
        logger.info(f"  Mode: PAIRED ({len(connectivity_seeds)} connectivity instances)")
    else:
        logger.info(f"  Mode: UNPAIRED (base_seed={base_seed}, n_trials={n_trials})")
    logger.info("="*80 + "\n")
    
    circuit_params = CircuitParams()
    opsin_params_obj = OpsinParams()
    
    base_synaptic_params = PerConnectionSynapticParams()
    
    blocked_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params.ampa_g_mean,
        ampa_g_std=base_synaptic_params.ampa_g_std,
        ampa_g_min=0.0,
        ampa_g_max=base_synaptic_params.ampa_g_max,
        gaba_g_mean=base_synaptic_params.gaba_g_mean,
        gaba_g_std=base_synaptic_params.gaba_g_std,
        gaba_g_min=base_synaptic_params.gaba_g_min,
        gaba_g_max=base_synaptic_params.gaba_g_max,
        distribution=base_synaptic_params.distribution,
        connection_modulation={
            **base_synaptic_params.connection_modulation,
            # Block all excitation TO interneurons
            'mec_pv': 0.0,
            'gc_pv': 0.0,
            'mc_pv': 0.0,
            'gc_sst': 0.0,
            'mc_sst': 0.0,
        }
    )
    
    results = {
        'full_network': {},
        'blocked_exc_to_int': {}
    }

    if paired_mode:
        # ---- Paired mode ----
        for target in target_populations:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing {target.upper()} stimulation (paired)")
            logger.info('=' * 60)

            target_opsin = opsin_expressions.get(target, {})

            # Full network
            logger.info("\n  1. Full Network (Control)")
            results['full_network'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=base_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Full]',
                **optogenetic_experiment_kwargs,
            )
            # Blocked exc-int (SAME connectivity seeds!)
            logger.info("\n2. Blocked Excitation to Interneurons")
            results['blocked_exc_to_int'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=blocked_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Blocked Exc-Int]',
                **optogenetic_experiment_kwargs,
            )

    else:
        for target in target_populations:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {target.upper()} stimulation")
            logger.info('='*60)

            # Run full network (control)
            logger.info("\n1. Full Network (Control)")
            logger.info("-"*60)
            experiment_full = OptogeneticExperiment(
                circuit_params, base_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed,
                **optogenetic_experiment_kwargs
            )

            full_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_full.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_activity and intensity == intensities[-1]),
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue  # Skip the directly stimulated population

                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                    # Get trial-to-trial variability
                    if n_trials > 1:
                        excited_fractions = []
                        for trial_result in result['trial_results']:
                            trial_activity = trial_result['activity_trace'][pop]
                            trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                            trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                            trial_change = trial_stim - trial_baseline
                            trial_baseline_std = torch.std(trial_baseline)
                            trial_excited = torch.mean((trial_change > trial_baseline_std).float()).item()
                            excited_fractions.append(trial_excited)

                        analysis[f'{pop}_excited_std'] = np.std(excited_fractions)

                full_results[intensity] = analysis

            results['full_network'][target] = full_results

            # Run with blocked excitation to interneurons
            logger.info("\n2. Blocked Excitation to Interneurons")
            logger.info("-"*60)
            experiment_blocked = OptogeneticExperiment(
                circuit_params, blocked_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed + 1000,
                **optogenetic_experiment_kwargs
            )

            blocked_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_blocked.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_activity and intensity == intensities[-1]),
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue
                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                blocked_results[intensity] = analysis

            results['blocked_exc_to_int'][target] = blocked_results

            # Print comparison
            logger.info(f"\n{'='*60}")
            logger.info(f"SUMMARY: {target.upper()} stimulation at intensity {intensities[-1]}")
            logger.info('='*60)
            logger.info(f"{'Population':<12} {'Full Network':<20} {'Blocked Exc->Int':<20} {'Change':<15}")
            logger.info('-'*60)

            for pop in ['gc', 'mc', 'pv', 'sst']:
                if pop == target:
                    continue
                full_exc = full_results[intensities[-1]][f'{pop}_excited']
                blocked_exc = blocked_results[intensities[-1]][f'{pop}_excited']
                change = blocked_exc - full_exc

                logger.info(f"{pop.upper():<12} {full_exc:>6.1%}              {blocked_exc:>6.1%}              {change:>+6.1%}")

                if abs(change) > 0.05:  # Significant reduction
                    logger.info(f"             -> {'STRONG' if abs(change) > 0.15 else 'MODERATE'} reduction suggests disinhibition mechanism")
    
    if save_results_file:
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to: {save_results_file}")
    
    return results


def test_recurrent_excitation(
    optimization_json_file: Optional[str] = None,
    target_populations: List[str] = ['pv', 'sst'],
    intensities: List[float] = [0.5, 1.0, 2.0],
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    plot_activity: bool = True,
    device: Optional[torch.device] = None,
    n_trials: int = 1,
    base_seed: int = 42,
    save_results_file: Optional[str] = None,
    connectivity_seeds: Optional[List[int]] = None,
    opsin_expressions: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    **optogenetic_experiment_kwargs
) -> Dict:
    """
    Test role of recurrent excitation among principal cells
    
    This test blocks recurrent excitatory connections among principal cells
    (GC->MC, MC->GC, MC->MC) while keeping all connections involving interneurons
    intact. This tests whether paradoxical excitation depends on amplification
    through recurrent excitatory loops.
    
    Args:
        optimization_json_file: Path to optimization results (optional)
        target_populations: List of populations to stimulate
        intensities: List of light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: When to start stimulation (ms)
        stim_duration: Duration of stimulation (ms)
        warmup: Pre-stimulation period (ms)
        plot_activity: Whether to plot activity traces
        device: Device to run on (None for auto-detect)
        n_trials: Number of trials to average
        base_seed: Base random seed
        save_results_file: If provided, save results to this file
        
    Returns:
        Dictionary with results for each target population
    """
    
    if device is None:
        device = get_default_device()
    
    paired_mode = (connectivity_seeds is not None and opsin_expressions is not None)
    
    logger.info("\n" + "="*80)
    logger.info("TEST: Recurrent Excitation Hypothesis")
    logger.info("="*80)
    logger.info("\nBlocking connections:")
    logger.info("  - GC -> MC (blocks feedforward excitation)")
    logger.info("  - MC -> GC (blocks feedback excitation)")
    logger.info("  - MC -> MC (blocks lateral excitation within MC)")
    logger.info("\nKept intact:")
    logger.info("  - All connections involving interneurons")
    logger.info("  - External drive (MEC -> GC, MEC -> MC)")
    if paired_mode:
        logger.info(f"  Mode: PAIRED ({len(connectivity_seeds)} connectivity instances)")
    else:
        logger.info(f"  Mode: UNPAIRED (base_seed={base_seed}, n_trials={n_trials})")
    logger.info("="*80 + "\n")
    
    circuit_params = CircuitParams()
    opsin_params_obj = OpsinParams()
    
    base_synaptic_params = PerConnectionSynapticParams()
    
    blocked_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params.ampa_g_mean,
        ampa_g_std=base_synaptic_params.ampa_g_std,
        ampa_g_min=0.0,
        ampa_g_max=base_synaptic_params.ampa_g_max,
        gaba_g_mean=base_synaptic_params.gaba_g_mean,
        gaba_g_std=base_synaptic_params.gaba_g_std,
        gaba_g_min=base_synaptic_params.gaba_g_min,
        gaba_g_max=base_synaptic_params.gaba_g_max,
        distribution=base_synaptic_params.distribution,
        connection_modulation={
            **base_synaptic_params.connection_modulation,
            # Block recurrent excitation among principal cells
            'gc_mc': 0.0,
            'mc_gc': 0.0,
            'mc_mc': 0.0,
        }
    )
    
    results = {
        'full_network': {},
        'blocked_recurrent': {}
    }

    if paired_mode:
        # ---- Paired mode ----
        for target in target_populations:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing {target.upper()} stimulation (paired)")
            logger.info('=' * 60)

            target_opsin = opsin_expressions.get(target, {})

            # Full network
            logger.info("\n  1. Full Network (Control)")
            results['full_network'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=base_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Full]',
                **optogenetic_experiment_kwargs,
            )
            # Blocked recurrent (SAME connectivity seeds!)
            logger.info("\n2. Blocked Recurrent Excitation")
            results['blocked_recurrent'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=blocked_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Blocked Recurrent]',
                **optogenetic_experiment_kwargs,
            )

    else:
        for target in target_populations:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {target.upper()} stimulation")
            logger.info('='*60)

            # Run full network (control)
            logger.info("\n1. Full Network (Control)")
            logger.info("-"*60)
            experiment_full = OptogeneticExperiment(
                circuit_params, base_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed,
                **optogenetic_experiment_kwargs
            )

            full_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_full.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_activity and intensity == intensities[-1]),
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue  # Skip the directly stimulated population

                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                    # Get trial-to-trial variability
                    if n_trials > 1:
                        excited_fractions = []
                        for trial_result in result['trial_results']:
                            trial_activity = trial_result['activity_trace'][pop]
                            trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                            trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                            trial_change = trial_stim - trial_baseline
                            trial_baseline_std = torch.std(trial_baseline)
                            trial_excited = torch.mean((trial_change > trial_baseline_std).float()).item()
                            excited_fractions.append(trial_excited)

                        analysis[f'{pop}_excited_std'] = np.std(excited_fractions)

                full_results[intensity] = analysis

            results['full_network'][target] = full_results

            # Run with blocked recurrent excitation
            logger.info("\n2. Blocked Recurrent Excitation")
            logger.info("-"*60)
            experiment_blocked = OptogeneticExperiment(
                circuit_params, blocked_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed + 1000,
                **optogenetic_experiment_kwargs
            )

            blocked_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_blocked.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_activity and intensity == intensities[-1]),
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue
                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                blocked_results[intensity] = analysis

            results['blocked_recurrent'][target] = blocked_results

            # Print comparison
            logger.info(f"\n{'='*60}")
            logger.info(f"SUMMARY: {target.upper()} stimulation at intensity {intensities[-1]}")
            logger.info('='*60)
            logger.info(f"{'Population':<12} {'Full Network':<20} {'Blocked Recurrent':<20} {'Change':<15}")
            logger.info('-'*60)

            for pop in ['gc', 'mc']:  # Focus on principal cells for this test
                full_exc = full_results[intensities[-1]][f'{pop}_excited']
                blocked_exc = blocked_results[intensities[-1]][f'{pop}_excited']
                change = blocked_exc - full_exc

                logger.info(f"{pop.upper():<12} {full_exc:>6.1%}              {blocked_exc:>6.1%}              {change:>+6.1%}")

                if abs(change) > 0.05:
                    logger.info(f"             -> Recurrent excitation {'amplifies' if change < 0 else 'suppresses'} paradoxical response")
    
    if save_results_file:
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to: {save_results_file}")
    
    return results


def test_intrinsic_excitation(
    optimization_json_file: Optional[str] = None,
    target_populations: List[str] = ['pv', 'sst'],
    intensities: List[float] = [0.5, 1.0, 2.0],
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    plot_blocked_activity: bool = False,
    plot_control_activity: bool = False,
    device: Optional[torch.device] = None,
    n_trials: int = 1,
    base_seed: int = 42,
    save_results_file: Optional[str] = None,
    connectivity_seeds: Optional[List[int]] = None,
    opsin_expressions: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    **optogenetic_experiment_kwargs
) -> Dict:
    """Test role of intrinsic excitation among principal cells and
    from principal cells to inhibitory interneurons.
    
    This test blocks recurrent excitatory connections among principal
    cells and from principal cells to inhibitory interneurons (GC->MC,
    MC->GC, MC->MC, GC->PV, MC->PV, GC->SST, MC->SST) while keeping
    all other connections involving interneurons intact. 
    
    Args:
        optimization_json_file: Path to optimization results (optional)
        target_populations: List of populations to stimulate
        intensities: List of light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: When to start stimulation (ms)
        stim_duration: Duration of stimulation (ms)
        warmup: Pre-stimulation period (ms)
        plot_activity: Whether to plot activity traces
        device: Device to run on (None for auto-detect)
        n_trials: Number of trials to average
        base_seed: Base random seed
        save_results_file: If provided, save results to this file
        
    Returns:
        Dictionary with results for each target population

    """
    
    if device is None:
        device = get_default_device()

    paired_mode = (connectivity_seeds is not None and opsin_expressions is not None)
    
    logger.info("\n" + "="*80)
    logger.info("TEST: Recurrent Excitation Hypothesis")
    logger.info("="*80)
    logger.info("\nBlocking connections:")
    logger.info("  - GC -> MC (blocks feedforward excitation)")
    logger.info("  - MC -> GC (blocks feedback excitation)")
    logger.info("  - MC -> MC (blocks lateral excitation within MC)")
    logger.info("  - GC -> PV")
    logger.info("  - GC -> SST")
    logger.info("  - MC -> PV")
    logger.info("  - MC -> SST")
    logger.info("\nKept intact:")
    logger.info("  - All inhibitory connections involving interneurons")
    logger.info("  - External drive (MEC -> GC, MEC -> MC)")
    if paired_mode:
        logger.info(f"  Mode: PAIRED ({len(connectivity_seeds)} connectivity instances)")
    else:
        logger.info(f"  Mode: UNPAIRED (base_seed={base_seed}, n_trials={n_trials})")
    logger.info("="*80 + "\n")
    
    circuit_params = CircuitParams()
    opsin_params_obj = OpsinParams()
    
    base_synaptic_params = PerConnectionSynapticParams()
    
    blocked_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params.ampa_g_mean,
        ampa_g_std=base_synaptic_params.ampa_g_std,
        ampa_g_min=0.0,
        ampa_g_max=base_synaptic_params.ampa_g_max,
        gaba_g_mean=base_synaptic_params.gaba_g_mean,
        gaba_g_std=base_synaptic_params.gaba_g_std,
        gaba_g_min=base_synaptic_params.gaba_g_min,
        gaba_g_max=base_synaptic_params.gaba_g_max,
        distribution=base_synaptic_params.distribution,
        connection_modulation={
            **base_synaptic_params.connection_modulation,
            'gc_mc': 0.0,
            'mc_gc': 0.0,
            'mc_mc': 0.0,
            'gc_pv': 0.0,
            'gc_sst': 0.0,
            'mc_pv': 0.0,
            'mc_sst': 0.0,
        }
    )
    
    results = {
        'full_network': {},
        'blocked_intrinsic_exc': {}
    }

    if paired_mode:
        # ---- Paired mode ----
        for target in target_populations:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing {target.upper()} stimulation (paired)")
            logger.info('=' * 60)

            target_opsin = opsin_expressions.get(target, {})

            # Full network
            logger.info("\n  1. Full Network (Control)")
            results['full_network'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=base_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Full]',
                **optogenetic_experiment_kwargs,
            )
            # Blocked intrinsic excitation (SAME connectivity seeds!)
            logger.info("\n2. Blocked Intrinsic Excitation")
            results['blocked_intrinsic_exc'][target] = _run_paired_ablation_condition(
                circuit_params=circuit_params,
                synaptic_params=blocked_synaptic_params,
                opsin_params=opsin_params_obj,
                target=target,
                intensities=intensities,
                connectivity_seeds=connectivity_seeds,
                opsin_expressions=target_opsin,
                stim_start=stim_start,
                stim_duration=stim_duration,
                warmup=warmup,
                mec_current=mec_current,
                opsin_current=opsin_current,
                optimization_json_file=optimization_json_file,
                device=device,
                condition_label='[Blocked Intrinsic Exc]',
                **optogenetic_experiment_kwargs,
            )

        
    else:
        for target in target_populations:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {target.upper()} stimulation")
            logger.info('='*60)

            # Run full network (control)
            logger.info("\n1. Full Network (Control)")
            logger.info("-"*60)
            experiment_full = OptogeneticExperiment(
                circuit_params, base_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed,
                **optogenetic_experiment_kwargs
            )

            full_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_full.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_control_activity and intensity == intensities[-1]),
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue  # Skip the directly stimulated population

                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                    # Get trial-to-trial variability
                    if n_trials > 1:
                        excited_fractions = []
                        for trial_result in result['trial_results']:
                            trial_activity = trial_result['activity_trace'][pop]
                            trial_baseline = torch.mean(trial_activity[:, baseline_mask], dim=1)
                            trial_stim = torch.mean(trial_activity[:, stim_mask], dim=1)
                            trial_change = trial_stim - trial_baseline
                            trial_baseline_std = torch.std(trial_baseline)
                            trial_excited = torch.mean((trial_change > trial_baseline_std).float()).item()
                            excited_fractions.append(trial_excited)

                        analysis[f'{pop}_excited_std'] = np.std(excited_fractions)

                full_results[intensity] = analysis

            results['full_network'][target] = full_results

            # Run with blocked recurrent excitation
            logger.info("\n2. Blocked Intrinsic Excitation")
            logger.info("-"*60)
            experiment_blocked = OptogeneticExperiment(
                circuit_params, blocked_synaptic_params, opsin_params_obj,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=base_seed + 1000,
                **optogenetic_experiment_kwargs
            )

            blocked_results = {}
            for intensity in intensities:
                logger.info(f"\n  Testing intensity: {intensity}")
                result = experiment_blocked.simulate_stimulation(
                    target, intensity,
                    stim_start=stim_start,
                    stim_duration=stim_duration,
                    plot_activity=(plot_blocked_activity and intensity == intensities[-1]),
                    plot_individual_trials=True,
                    mec_current=mec_current,
                    opsin_current=opsin_current,
                    n_trials=n_trials
                )

                time = result['time']
                activity_mean = result['activity_trace_mean']
                baseline_mask = (time >= warmup) & (time < stim_start)
                stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))

                analysis = {}
                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if pop == target:
                        continue
                    baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                    stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                    rate_change = stim_rate - baseline_rate
                    baseline_std = torch.std(baseline_rate)

                    excited_fraction = torch.mean((rate_change > baseline_std).float())

                    analysis[f'{pop}_excited'] = excited_fraction.item()
                    analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

                blocked_results[intensity] = analysis

            results['blocked_intrinsic_exc'][target] = blocked_results

            # Print comparison
            logger.info(f"\n{'='*60}")
            logger.info(f"SUMMARY: {target.upper()} stimulation at intensity {intensities[-1]}")
            logger.info('='*60)
            logger.info(f"{'Population':<12} {'Full Network':<20} {'Blocked Intrinsic Exc':<20} {'Change':<15}")
            logger.info('-'*60)

            for pop in ['gc', 'mc']:  # Focus on principal cells for this test
                full_exc = full_results[intensities[-1]][f'{pop}_excited']
                blocked_exc = blocked_results[intensities[-1]][f'{pop}_excited']
                change = blocked_exc - full_exc

                logger.info(f"{pop.upper():<12} {full_exc:>6.1%}              {blocked_exc:>6.1%}              {change:>+6.1%}")

                if abs(change) > 0.05:
                    logger.info(f"             -> Intrinsic excitation {'amplifies' if change < 0 else 'suppresses'} paradoxical response")
    
    if save_results_file:
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to: {save_results_file}")
    
    return results


def run_all_ablation_tests(
    optimization_json_file: Optional[str] = None,
    target_populations: List[str] = ['pv', 'sst'],
    intensities: List[float] = [1.0],
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    stim_start: float = 1500.0,
    stim_duration: float = 1000.0,
    warmup: float = 500.0,
    device: Optional[torch.device] = None,
    n_trials: int = 3,
    base_seed: int = 42,
    output_dir: str = "./ablation_tests",
    nested_experiment_file: Optional[str] = None,
    **optogenetic_experiment_kwargs
) -> Dict:
    """
    Run all ablation tests and create summary comparison.

    When nested_experiment_file is specified, connectivity seeds and opsin
    expressions are read from the HDF5 file. Each ablation test then runs
    on the exact same circuit realizations as the nested experiment,
    creating paired (full vs. blocked) comparisons for each connectivity
    instance.

    Args:
        optimization_json_file: Path to optimization results (optional)
        target_populations: Populations to stimulate
        intensities: Light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Pre-stimulation period (ms)
        device: Device to run on (None for auto-detect)
        n_trials: Trials per condition (ignored when nested file is used)
        base_seed: Base random seed (ignored when nested file is used)
        output_dir: Directory to save results
        nested_experiment_file: Path to nested experiment HDF5 file.
            When specified, connectivity seeds and opsin expressions are
            loaded from this file, overriding base_seed and n_trials.
        **optogenetic_experiment_kwargs: Forwarded to OptogeneticExperiment

    Returns:
        Dict with all test results. When nested file is used, results include
        '_by_conn' arrays enabling paired statistical tests.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if device is None:
        device = get_default_device()

    # ------------------------------------------------------------------
    # Load nested experiment data if specified
    # ------------------------------------------------------------------
    connectivity_seeds = None
    opsin_expressions = None
    paired_mode = False

    if nested_experiment_file is not None:
        logger.info(f"\nLoading connectivity from nested experiment: "
                    f"{nested_experiment_file}")

        nested_data = load_nested_experiment_seeds_and_opsin(
            nested_experiment_file, target_populations, max_n_experiments=None
        )
        connectivity_seeds = nested_data['connectivity_seeds']
        opsin_expressions = nested_data['opsin_expressions']
        paired_mode = True

        # Default to nested experiment's intensities if not overridden
        if intensities == [1.0]:  # default sentinel
            nested_intensities = nested_data.get('intensities')
            if nested_intensities:
                intensities = nested_intensities
                logger.info(f"  Using intensities from nested file: {intensities}")

        logger.info(f"  Paired mode: {len(connectivity_seeds)} connectivity instances")
    else:
        logger.info(f"\n  Unpaired mode: base_seed={base_seed}, n_trials={n_trials}")

    logger.info("\n" + "=" * 80)
    logger.info("ABLATION TESTS")
    logger.info("=" * 80)
    mode_str = (
        f"Paired mode ({len(connectivity_seeds)} connectivity instances from nested file)"
        if paired_mode
        else f"Unpaired mode ({n_trials} trials, base_seed={base_seed})"
    )
    logger.info(f"\n{mode_str}")
    logger.info(f"Results will be saved to: {output_dir}")
    logger.info("=" * 80)

    all_results = {}

    # ------------------------------------------------------------------
    # Common kwargs for test functions
    # ------------------------------------------------------------------
    common_kwargs = dict(
        optimization_json_file=optimization_json_file,
        target_populations=target_populations,
        intensities=intensities,
        mec_current=mec_current,
        opsin_current=opsin_current,
        stim_start=stim_start,
        stim_duration=stim_duration,
        warmup=warmup,
        plot_activity=False,
        device=device,
        connectivity_seeds=connectivity_seeds,
        opsin_expressions=opsin_expressions,
        **optogenetic_experiment_kwargs,
    )

    # Unpaired-only kwargs
    unpaired_kwargs = dict(
        n_trials=n_trials,
    )

    def merged_kwargs(extra_base_seed_offset: int, save_name: str):
        kw = {**common_kwargs}
        if not paired_mode:
            kw.update(unpaired_kwargs)
            kw['base_seed'] = base_seed + extra_base_seed_offset
        kw['save_results_file'] = str(output_path / save_name)
        return kw

    # ------------------------------------------------------------------
    # Excitation to interneurons
    # ------------------------------------------------------------------
    logger.info("\n\n" + "#" * 80)
    logger.info("# TEST 1: Excitation to Interneurons (Disinhibition Test)")
    logger.info("#" * 80)
    results_exc_int = test_excitation_to_interneurons(
        **merged_kwargs(2000, "test2_exc_to_int.pkl")
    )
    all_results['excitation_to_interneurons'] = results_exc_int

    # ------------------------------------------------------------------
    # Interneuron-interneuron interactions
    # ------------------------------------------------------------------
    logger.info("\n\n" + "#" * 80)
    logger.info("# TEST 2: Interneuron-Interneuron Interactions")
    logger.info("#" * 80)
    results_int_int = test_interneuron_interactions(
        **merged_kwargs(0, "test1_int_int.pkl")
    )
    all_results['interneuron_interactions'] = results_int_int

    # ------------------------------------------------------------------
    # Recurrent excitation
    # ------------------------------------------------------------------
    logger.info("\n\n" + "#" * 80)
    logger.info("# TEST 3: Recurrent Excitation")
    logger.info("#" * 80)
    results_recurrent = test_recurrent_excitation(
        **merged_kwargs(4000, "test3_recurrent.pkl")
    )
    all_results['recurrent_excitation'] = results_recurrent

    # ------------------------------------------------------------------
    # Test 4: All intrinsic excitation
    # ------------------------------------------------------------------
    logger.info("\n\n" + "#" * 80)
    logger.info("# TEST 4: All intrinsic excitation")
    logger.info("#" * 80)
    results_intrinsic_exc = test_intrinsic_excitation(
        **merged_kwargs(5000, "test4_intrinsic_excitation.pkl")
    )
    all_results['intrinsic_excitation'] = results_intrinsic_exc

    _print_ablation_summary(
        all_results, intensities, target_populations, paired_mode
    )

    # Save combined results
    combined_file = output_path / "all_ablation_tests.pkl"
    with open(combined_file, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"\n\nAll results saved to: {combined_file}")

    return all_results



def _print_ablation_summary(
    all_results: Dict,
    intensities: List[float],
    target_populations: List[str],
    paired_mode: bool,
):
    """Print ablation test summary with optional paired statistics."""
    from scipy import stats as scipy_stats

    logger.info("\n\n" + "=" * 80)
    logger.info("ABLATION TEST SUMMARY")
    logger.info("=" * 80)

    intensity = intensities[-1]

    results_int_int = all_results['interneuron_interactions']
    results_exc_int = all_results['excitation_to_interneurons']
    results_recurrent = all_results['recurrent_excitation']
    results_intrinsic_exc = all_results['intrinsic_excitation']

    for target in target_populations:
        logger.info(f"\n{target.upper()} Stimulation at intensity {intensity}")
        logger.info("-" * 80)

        header = f"{'Manipulation':<30} {'GC Excited':<15} {'MC Excited':<15}"
        if paired_mode:
            header += f" {'Paired p':<12}"
        header += f" {'Effect':<15}"
        logger.info(header)
        logger.info("-" * 80)

        # Full network baseline
        full_gc = results_exc_int['full_network'][target][intensity]['gc_excited']
        full_mc = results_exc_int['full_network'][target][intensity]['mc_excited']
        logger.info(
            f"{'Full Network (baseline)':<30} {full_gc:>6.1%}          "
            f"{full_mc:>6.1%}          -"
        )

        # Each ablation condition
        conditions = [
            ('Block Int-Int', results_int_int, 'blocked_int_int'),
            ('Block Exc->Int (KEY)', results_exc_int, 'blocked_exc_to_int'),
            ('Block Recurrent', results_recurrent, 'blocked_recurrent'),
            ('Block Intrinsic exc', results_intrinsic_exc, 'blocked_intrinsic_exc'),
        ]

        for label, result_dict, cond_key in conditions:
            cond_gc = result_dict[cond_key][target][intensity]['gc_excited']
            cond_mc = result_dict[cond_key][target][intensity]['mc_excited']
            change_gc = (cond_gc - full_gc) / (full_gc + 1e-6)
            change_mc = (cond_mc - full_mc) / (full_mc + 1e-6)
            avg_change = (change_gc + change_mc) / 2

            line = f"{label:<30} {cond_gc:>6.1%}          {cond_mc:>6.1%}          "

            # Paired test if available
            if paired_mode:
                full_gc_by_conn = results_exc_int['full_network'][target][intensity].get(
                    'gc_excited_by_conn'
                )
                cond_gc_by_conn = result_dict[cond_key][target][intensity].get(
                    'gc_excited_by_conn'
                )
                if full_gc_by_conn is not None and cond_gc_by_conn is not None:
                    try:
                        _, p_val = scipy_stats.wilcoxon(
                            full_gc_by_conn, cond_gc_by_conn
                        )
                        line += f"p={p_val:.3f}      "
                    except ValueError:
                        line += f"p=n/a        "
                else:
                    line += f"{'':12}"

            line += f"{avg_change:>+6.1%}"
            logger.info(line)





def _analyze_ablation_trial(
    result: Dict,
    target: str,
    baseline_mask,
    stim_mask,
    n_trials: int = 1,
) -> Dict:
    """
    Compute excited fractions and mean changes from a single ablation trial.

    Handles both target population (non-expressing cells) and non-target
    populations.

    Args:
        result: Output of simulate_stimulation (aggregated)
        target: Target population being stimulated
        baseline_mask: Boolean tensor for baseline period
        stim_mask: Boolean tensor for stimulation period
        n_trials: Number of trials (for trial-level variability)

    Returns:
        Analysis dict with per-population metrics
    """
    activity_mean = result['activity_trace_mean']
    analysis = {}

    for pop in ['gc', 'mc', 'pv', 'sst']:
        baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
        stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
        rate_change = stim_rate - baseline_rate
        baseline_std = torch.std(baseline_rate)

        if pop == target:
            # Non-expressing cells only
            non_stim_idx = result['trial_results'][0]['non_stimulated_indices']
            n_nonexpr = len(non_stim_idx)

            if n_nonexpr > 0:
                change_nonexpr = rate_change[non_stim_idx]
                baseline_std_nonexpr = torch.std(baseline_rate[non_stim_idx])
                excited_nonexpr = torch.mean(
                    (change_nonexpr > baseline_std_nonexpr).float()
                ).item()
                analysis[f'{pop}_nonexpr_excited'] = excited_nonexpr
                analysis[f'{pop}_nonexpr_mean_change'] = torch.mean(change_nonexpr).item()
                analysis[f'{pop}_nonexpr_count'] = n_nonexpr
            else:
                analysis[f'{pop}_nonexpr_excited'] = 0.0
                analysis[f'{pop}_nonexpr_mean_change'] = 0.0
                analysis[f'{pop}_nonexpr_count'] = 0
        else:
            excited_fraction = torch.mean(
                (rate_change > baseline_std).float()
            ).item()
            analysis[f'{pop}_excited'] = excited_fraction
            analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()

    return analysis


def _run_paired_ablation_condition(
    circuit_params: 'CircuitParams',
    synaptic_params: 'PerConnectionSynapticParams',
    opsin_params: 'OpsinParams',
    target: str,
    intensities: List[float],
    connectivity_seeds: List[int],
    opsin_expressions: Dict[int, np.ndarray],
    stim_start: float,
    stim_duration: float,
    warmup: float,
    mec_current: float,
    opsin_current: float,
    optimization_json_file: Optional[str],
    device: torch.device,
    condition_label: str = '',
    plot_activity: bool = False,
    **optogenetic_experiment_kwargs,
) -> Dict[float, Dict]:
    """
    Run one ablation condition across all connectivity instances (paired design).

    For each connectivity seed, creates an OptogeneticExperiment, injects the
    corresponding opsin expression, and runs a single-trial simulation.
    Results are aggregated across connectivity instances.

    Args:
        circuit_params: Circuit parameters
        synaptic_params: Synaptic parameters (may include ablation modulations)
        opsin_params: Opsin parameters
        target: Target population to stimulate
        intensities: Light intensities to test
        connectivity_seeds: Seeds for each connectivity instance
        opsin_expressions: Opsin expression arrays keyed by conn_idx
        stim_start: Stimulation start time (ms)
        stim_duration: Stimulation duration (ms)
        warmup: Baseline start (ms)
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        optimization_json_file: Path to optimization results
        device: Torch device
        condition_label: Label for logging
        plot_activity: Whether to plot individual trials
        **optogenetic_experiment_kwargs: Forwarded to OptogeneticExperiment

    Returns:
        Dict mapping intensity -> aggregated analysis dict.
        Each analysis dict contains per-population metrics with
        '_mean' and '_std' suffixes computed across connectivity instances,
        plus '_by_conn' lists for paired tests.
    """
    n_conn = len(connectivity_seeds)
    results_by_intensity = {}

    for intensity in intensities:
        # Collect per-connectivity results
        per_conn_analyses = []

        for conn_idx, conn_seed in enumerate(connectivity_seeds):
            logger.info(
                f"    {condition_label} conn {conn_idx + 1}/{n_conn} "
                f"(seed={conn_seed}), intensity={intensity}"
            )

            # Create experiment with this connectivity
            experiment = OptogeneticExperiment(
                circuit_params, synaptic_params, opsin_params,
                optimization_json_file=optimization_json_file,
                device=device,
                base_seed=conn_seed,
                **optogenetic_experiment_kwargs,
            )

            # Inject pre-loaded opsin expression
            if conn_idx in opsin_expressions:
                experiment.set_opsin_expression(target, opsin_expressions[conn_idx])

            # Run single-trial simulation
            result = experiment.simulate_stimulation(
                target_population=target,
                light_intensity=intensity,
                stim_start=stim_start,
                stim_duration=stim_duration,
                plot_activity=plot_activity,
                mec_current=mec_current,
                opsin_current=opsin_current,
                n_trials=1,
                regenerate_connectivity_per_trial=False,
            )

            # Compute time masks
            time = result['time']
            baseline_mask = (time >= warmup) & (time < stim_start)
            stim_mask = (
                (time >= stim_start)
                & (time <= (stim_start + stim_duration))
            )

            analysis = _analyze_ablation_trial(
                result, target, baseline_mask, stim_mask
            )
            per_conn_analyses.append(analysis)

            # Clean up
            del experiment
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Aggregate across connectivity instances
        aggregated = _aggregate_paired_ablation_results(per_conn_analyses, target)
        results_by_intensity[intensity] = aggregated

    return results_by_intensity


def _aggregate_paired_ablation_results(
    per_conn_analyses: List[Dict],
    target: str,
) -> Dict:
    """
    Aggregate per-connectivity ablation analyses into summary statistics.

    Computes mean, std, SEM across connectivity instances, and retains
    per-connectivity values for paired statistical tests.

    Args:
        per_conn_analyses: List of analysis dicts, one per connectivity instance
        target: Target population

    Returns:
        Aggregated dict with '_mean', '_std', '_sem', '_by_conn' keys
    """
    if not per_conn_analyses:
        return {}

    # Discover all metric keys from first analysis
    all_keys = list(per_conn_analyses[0].keys())
    aggregated = {}
    n = len(per_conn_analyses)

    for key in all_keys:
        values = [a[key] for a in per_conn_analyses if key in a]

        if not values:
            continue

        # Numeric keys get mean/std/sem/by_conn
        if isinstance(values[0], (int, float)):
            arr = np.array(values, dtype=float)
            aggregated[key] = float(np.mean(arr))
            aggregated[f'{key}_std'] = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            aggregated[f'{key}_sem'] = (
                float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            )
            aggregated[f'{key}_by_conn'] = arr.tolist()
        else:
            # Non-numeric (e.g., count): just store mean
            aggregated[key] = values[0]

    aggregated['n_connectivity'] = n
    return aggregated


def plot_ablation_test_results(all_results: Dict,
                               intensity: float = 1.0,
                               save_path: Optional[str] = None,
                               show_percentage_change: bool = False,
                               show_error_bars: bool = False) -> None:
    """
    Plot ablation test results including non-target interneurons
    
    Creates a multi-panel figure showing:
    - Bar plots comparing % excited cells across ablation conditions
    - Separate panels for PV and SST stimulation
    - All populations: GC, MC, and the non-target interneuron
    - Error bars for multi-trial experiments
    
    Args:
        all_results: Dictionary from run_all_ablation_tests()
        intensity: Light intensity to plot (default: 1.0)
        save_path: Optional directory to save figure
    """
    
    # Extract test results
    results_int_int = all_results['interneuron_interactions']
    results_exc_int = all_results['excitation_to_interneurons']
    results_recurrent = all_results['recurrent_excitation']
    results_intrinsic_exc = all_results['intrinsic_excitation']
    
    # Create figure with subplots - 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    
    # Define colors
    colors = {
        'full': '#2ecc71',  # Green
        'int_int': '#e74c3c',  # Red
        'exc_int': '#e67e22',  # Orange
        'recurrent': '#3498db',  # Blue
        'intrinsic': '#e1c16e'  # Brass
    }
    
    conditions = ['Full\nNetwork', 'Block\nInt-Int', 'Block\nExc-Int', 'Block\nRec.',
                  'Block\nIntrinsic']
    
    # Define populations to plot for each target
    pop_map = {
        'pv': ['gc', 'mc', 'pv_nonexpr', 'sst'],  # When stimulating PV, show GC, MC, PV-non-expressing, SST
        'sst': ['gc', 'mc', 'pv', 'sst_nonexpr']   # When stimulating SST, show GC, MC, PV, SST-non-expressing
    }
    
    for target_idx, target in enumerate(['pv', 'sst']):
        populations = pop_map[target]
        
        for pop_idx, pop in enumerate(populations):
            ax = axes[target_idx, pop_idx]
            
            # Handle non-expressing target cells specially
            if pop.endswith('_nonexpr'):
                base_pop = pop.replace('_nonexpr', '')
                full_excited = results_exc_int['full_network'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                int_int_excited = results_int_int['blocked_int_int'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                exc_int_excited = results_exc_int['blocked_exc_to_int'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                recurrent_excited = results_recurrent['blocked_recurrent'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                intrinsic_excited = results_intrinsic_exc['blocked_intrinsic_exc'][target][intensity].get(f'{base_pop}_nonexpr_excited', 0.0)
                
                # Get error bars
                errors = []
                if show_error_bars:
                    for result_dict, condition in [(results_exc_int, 'full_network'),
                                                   (results_int_int, 'blocked_int_int'),
                                                   (results_exc_int, 'blocked_exc_to_int'),
                                                   (results_recurrent, 'blocked_recurrent'),
                                                   (results_intrinsic_exc, 'blocked_intrinsic_exc')]:
                        std_key = f'{base_pop}_nonexpr_excited_std'
                        if condition in result_dict and std_key in result_dict[condition][target][intensity]:
                            errors.append(result_dict[condition][target][intensity][std_key])
                        else:
                            errors.append(0)
            else:
                # Normal populations
                full_excited = results_exc_int['full_network'][target][intensity][f'{pop}_excited']
                int_int_excited = results_int_int['blocked_int_int'][target][intensity][f'{pop}_excited']
                exc_int_excited = results_exc_int['blocked_exc_to_int'][target][intensity][f'{pop}_excited']
                recurrent_excited = results_recurrent['blocked_recurrent'][target][intensity][f'{pop}_excited']
                intrinsic_excited = results_intrinsic_exc['blocked_intrinsic_exc'][target][intensity][f'{pop}_excited']
                
                # Get error bars
                errors = []
                if show_error_bars:
                    for result_dict, condition in [(results_exc_int, 'full_network'),
                                                   (results_int_int, 'blocked_int_int'),
                                                   (results_exc_int, 'blocked_exc_to_int'),
                                                   (results_recurrent, 'blocked_recurrent'),
                                                   (results_intrinsic_exc, 'blocked_intrinsic_exc')]:
                        std_key = f'{pop}_excited_std'
                        if condition in result_dict and std_key in result_dict[condition][target][intensity]:
                            errors.append(result_dict[condition][target][intensity][std_key])
                        else:
                            errors.append(0)

            data = [full_excited, int_int_excited, exc_int_excited, recurrent_excited, intrinsic_excited]
                        
            # Create bar plot
            x_pos = np.arange(len(conditions))
            bars = ax.bar(x_pos, data, yerr=errors if any(errors) else None,
                          color=[colors['full'], colors['int_int'], 
                                 colors['exc_int'], colors['recurrent'],
                                 colors['intrinsic']],
                          alpha=0.7, edgecolor='black', width=0.5,
                          linewidth=1.5, capsize=5)
            
            # Add value labels on bars
            for bar, value in zip(bars, data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value*100:.1f}%',
                        ha='left' if show_error_bars else 'center',
                        va='bottom', fontsize=10, fontweight='bold')
            
            # Add percentage change labels for ablations
            if show_percentage_change:
                for i, (bar, value) in enumerate(zip(bars[1:], data[1:]), 1):
                    if data[0] > 0.0:
                        change = (value - data[0]) / (data[0] + 1e-6) * 100
                    else:
                        change = value * 100
                    height = bar.get_height()
                    color = 'red' if change < -10 else 'orange' if change < 0 else 'black'
                    ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                            f'{change:+.0f}%',
                            ha='center', va='center', fontsize=10, 
                            fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='white', alpha=0.7))
            
            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels(conditions, fontsize=10)
            ax.set_ylabel('Fraction Excited', fontsize=10)
            ax.set_ylim(0, max(data) * 1.2)
            
            # Format title based on population type
            if pop.endswith('_nonexpr'):
                base_pop = pop.replace('_nonexpr', '')
                # Get count of non-expressing cells for subtitle
                n_nonexpr = results_exc_int['full_network'][target][intensity].get(f'{base_pop}_nonexpr_count', 0)
                ax.set_title(f'{target.upper()} Stim -> {target.upper()} (non-expr)', 
                            fontsize=11, fontweight='bold', color='purple')
            elif pop in ['pv', 'sst']:
                ax.set_title(f'{target.upper()} Stim -> {pop.upper()} (non-target IN)', 
                            fontsize=11, fontweight='bold', color='darkred')
            else:
                ax.set_title(f'{target.upper()} Stim -> {pop.upper()}', 
                            fontsize=11, fontweight='bold')
            
            #ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
    
    # Overall title
    fig.suptitle(f'Ablation Test Results: Paradoxical Excitation\n'
                f'(Intensity = {intensity})',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / f'ablation_results_intensity_{intensity}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'ablation_results_intensity_{intensity}.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"\nSaved ablation plots to {save_dir}")
    
    plt.show()

