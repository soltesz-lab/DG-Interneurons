#!/usr/bin/env python3
"""
Optimization framework including optogenetic stimulation effects

Adds optogenetic stimulation objectives to the circuit optimization,
targeting the paradoxical excitation effects observed by Hainmueller et al.

Updated to use proper CPU/GPU device interface consistent with other modules.
"""
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import torch
import numpy as np
from functools import partial
import tqdm
import json
from datetime import datetime

# Import existing optimization components
from DG_circuit_optimization import (
    OptimizationTargets,
    OptimizationConfig,
    evaluate_rate_ordering_constraints,
    configure_torch_threads,
    get_default_device,
    create_default_targets
)

@dataclass
class OptogeneticTargets:
    """Target effects for optogenetic stimulation experiments"""
    
    # Target fractional increase in firing rates during stimulation
    # Format: {target_pop: {affected_pop: target_increase}}
    target_rate_increases: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'pv': {
            'gc': 0.1,   # 10% strong effect expected on GCs
            'mc': 0.2,   # 20% of MCs should increase firing
            'sst': 0.3,  # SST increase firing
        },
        'sst': {
            'gc': -0.15,  # GCs are inhibited (winner-take-all)
            'mc': 0.2,   # Effect on MCs
            'pv': -0.2,  # PV should be inhibited
        }
    })
    
    # Target for increased inequality (Gini coefficient change)
    target_gini_increase: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'pv': {
            'mc': 0.1,   # PV stimulation should increase MC inequality
            'gc': 0.1,   # Less effect on GCs
        },
        'sst': {
            'gc': 0.1,   # SST stimulation should increase GC inequality
            'mc': 0.1,   # Less effect on MCs
        }
    })
    
    # Light intensity for stimulation
    stimulation_intensity: float = 1.0
    

@dataclass
class CombinedOptimizationTargets(OptimizationTargets):
    """Combined targets including both baseline and optogenetic objectives"""
    
    # Optogenetic targets
    optogenetic_targets: OptogeneticTargets = field(default_factory=OptogeneticTargets)
    
    # Overall weight distribution
    baseline_weight: float = 1.0
    optogenetic_weight: float = 1.0


def calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for firing rate inequality"""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return max(0.0, gini)


def simulate_optogenetic_stimulation(circuit_factory_data, connection_modulation,
                                     target_pop: str, light_intensity: float,
                                     mec_current: float = 80.0,
                                     opsin_current: float = 200.0,
                                     device: Optional[torch.device] = None) -> Dict:
    """
    Run optogenetic stimulation experiment on circuit
    
    Args:
        circuit_factory_data: Tuple of (circuit_params, base_params_dict, opsin_params)
        connection_modulation: Dict of connection modulation parameters
        target_pop: Population to stimulate ('pv' or 'sst')
        light_intensity: Light intensity for stimulation
        mec_current: MEC drive current (pA)
        opsin_current: Opsin-induced current (pA)
        device: Device to run simulation on (None for auto-detect)
    
    Returns:
        Dict with firing rate statistics for all populations
    """
    from DG_circuit_dendritic_somatic_transfer import (
        PerConnectionSynapticParams
    )
    from DG_protocol import OpsinExpression, OptogeneticExperiment
    
    if device is None:
        device = get_default_device()
    
    # Unpack circuit factory data
    circuit_params, base_synaptic_params_dict, opsin_params = circuit_factory_data
    
    # Create synaptic parameters
    synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params_dict['ampa_g_mean'],
        ampa_g_std=base_synaptic_params_dict['ampa_g_std'],
        gaba_g_mean=base_synaptic_params_dict['gaba_g_mean'],
        gaba_g_std=base_synaptic_params_dict['gaba_g_std'],
        distribution=base_synaptic_params_dict['distribution'],
        connection_modulation=connection_modulation
    )
    
    # Create opto experiment on specified device
    experiment = OptogeneticExperiment(circuit_params,
                                       synaptic_params,
                                       opsin_params,
                                       device=device)

    stim_start = 550.
    duration = 1650.
    warmup = 150.
    
    exp_result = experiment.simulate_stimulation(target_pop,
                                                 light_intensity,
                                                 duration = duration,
                                                 stim_start = stim_start,
                                                 mec_current = mec_current,
                                                 opsin_current = opsin_current,
                                                 plot_activity = False)

    time = exp_result['time']
    activity = exp_result['activity_trace']
    baseline_mask = (time >= warmup) & (time < stim_start)  # Pre-stimulation
    stim_mask = time >= stim_start     # During stimulation

    # Calculate statistics
    results = {}
    
    for pop in activity:
        if len(activity[pop]) > 0:
            pop_time_series = activity[pop]  # (neurons, time)
            
            # Baseline and stimulation periods
            baseline_rates = torch.mean(pop_time_series[:, baseline_mask], dim=1)
            stim_rates = torch.mean(pop_time_series[:, stim_mask], dim=1)
            
            # Changes
            rate_changes = stim_rates - baseline_rates
            baseline_std = torch.std(baseline_rates)

            # Fraction activated
            activated_fraction = torch.mean((rate_changes > baseline_std).float()).item()

            baseline_mean = torch.mean(baseline_rates).item()
            stim_mean = torch.mean(stim_rates).item()
            
            # Gini coefficients
            baseline_gini = calculate_gini_coefficient(baseline_rates.cpu().numpy())
            stim_gini = calculate_gini_coefficient(stim_rates.cpu().numpy())
            gini_change = stim_gini - baseline_gini
            
            results[pop] = {
                'baseline_mean': baseline_mean,
                'stim_mean': stim_mean,
                'mean_change': torch.mean(rate_changes).item(),
                'activated_fraction': activated_fraction,
                'baseline_gini': baseline_gini,
                'stim_gini': stim_gini,
                'gini_change': gini_change,
            }
    
    return results


def evaluate_optogenetic_objectives(opto_results: Dict, 
                                    target_pop: str,
                                    opto_targets: OptogeneticTargets,
                                    verbose: bool = False) -> Tuple[float, Dict]:
    """
    Calculate loss for optogenetic stimulation objectives with detailed diagnostics
    
    Returns:
        total_loss: Combined optogenetic loss
        loss_components: Breakdown of loss components
    """
    loss_components = {}
    total_loss = 0.0
    
    # Target rate increases
    if target_pop in opto_targets.target_rate_increases:
        rate_increase_loss = 0.0
        
        for affected_pop, target_fraction in opto_targets.target_rate_increases[target_pop].items():
            if affected_pop in opto_results and affected_pop != target_pop:
                actual_fraction = opto_results[affected_pop]['activated_fraction']
                actual_mean_change = opto_results[affected_pop]['mean_change']
                baseline_mean = opto_results[affected_pop]['baseline_mean']
                stim_mean = opto_results[affected_pop]['stim_mean']

                if verbose:
                    print(f"  {affected_pop.upper()}:")
                    print(f"    Baseline rate: {baseline_mean:.3f} Hz")
                    print(f"    Stim rate: {stim_mean:.3f} Hz")
                    print(f"    Mean change: {actual_mean_change:+.3f} Hz")
                    print(f"    Activated fraction: {actual_fraction:.3f} (target: {target_fraction:.3f})")
                
                # Squared error
                error = (actual_fraction - target_fraction) ** 2

                # Strong penalty for zero activation when non-zero is expected
                if (np.abs(target_fraction) > 0) and (np.isclose(np.abs(actual_fraction), 0.0, 1e-2, 1e-2)):
                    penalty = 1e2
                    rate_increase_loss += penalty
                    if verbose:
                        print(f"    Zero activation penalty: {penalty:.1f}")
                else:
                    rate_increase_loss += error
                    if verbose:
                        print(f"    Squared error: {error:.6f}")
        
        if verbose:
            print(f"  Total rate increase loss: {rate_increase_loss:.6f}")
        
        loss_components['rate_increase'] = rate_increase_loss
        total_loss += rate_increase_loss
    
    # Target Gini increases (inequality)
    if target_pop in opto_targets.target_gini_increase:
        gini_loss = 0.0
        
        if verbose:
            print(f"\n  Gini coefficient changes:")
        
        for affected_pop, target_gini_change in opto_targets.target_gini_increase[target_pop].items():
            if affected_pop in opto_results and affected_pop != target_pop:
                baseline_gini = opto_results[affected_pop]['baseline_gini']
                stim_gini = opto_results[affected_pop]['stim_gini']
                actual_gini_change = opto_results[affected_pop]['gini_change']
                
                # We want to match the target Gini increase
                error = (actual_gini_change - target_gini_change) ** 2
                gini_loss += error
                
                if verbose:
                    print(f"  {affected_pop.upper()}:")
                    print(f"    Baseline Gini: {baseline_gini:.4f}")
                    print(f"    Stim Gini: {stim_gini:.4f}")
                    print(f"    Change: {actual_gini_change:+.4f} (target: {target_gini_change:+.4f})")
                    print(f"    Squared error: {error:.6f}")
        
        if verbose:
            print(f"  Total Gini loss: {gini_loss:.6f}")
        
        loss_components['gini_increase'] = gini_loss
        total_loss += gini_loss
    
    return total_loss, loss_components


<<<<<<< HEAD
def evaluate_de_candidate_worker(param_array, connection_names, circuit_factory_data,
                                 targets: CombinedOptimizationTargets, config,
                                 device: Optional[torch.device] = None,
                                 verbose: bool = False):
=======

def evaluate_candidate_worker(param_array, connection_names, circuit_factory_data,
                              targets: CombinedOptimizationTargets, config,
                              verbose: bool = False):
>>>>>>> 8c504ed274620d21a13a6a8594f9b4617da999d4
    """
    Objective worker function with detailed diagnostics
    
    Args:
        param_array: Array of connection modulation parameters
        connection_names: List of connection names
        circuit_factory_data: Tuple of circuit configuration data
        targets: Combined optimization targets
        config: Optimization configuration
        device: Device to run simulation on (None for auto-detect)
        verbose: Whether to print detailed diagnostics
    """
    try:
        from DG_circuit_dendritic_somatic_transfer import (
            DentateCircuit, PerConnectionSynapticParams
        )
        
        if device is None:
            device = get_default_device()
        
        # Convert parameter array to connection modulation dict
        connection_modulation = dict(zip(connection_names, param_array))
        
        # Unpack circuit factory data
        circuit_params, base_synaptic_params_dict, opsin_params = circuit_factory_data
        
        if verbose:
            print(f"\n{'='*80}")
            print("Detailed evaluation of candidate parameters")
            print(f"{'='*80}")
            print(f"Device: {device}")
        
        # === BASELINE OBJECTIVES ===
        synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=base_synaptic_params_dict['ampa_g_mean'],
            ampa_g_std=base_synaptic_params_dict['ampa_g_std'],
            gaba_g_mean=base_synaptic_params_dict['gaba_g_mean'],
            gaba_g_std=base_synaptic_params_dict['gaba_g_std'],
            distribution=base_synaptic_params_dict['distribution'],
            connection_modulation=connection_modulation
        )
        
        # Create circuit on specified device
        circuit = DentateCircuit(circuit_params, synaptic_params, opsin_params, device=device)
        
        baseline_loss = 0.0
        
        if verbose:
            print(f"\n{'='*80}")
            print("Baseline circuit evaluation")
            print(f"{'='*80}")
        
        for mec_drive in config.mec_drive_levels:
            if verbose:
                print(f"\n  MEC drive: {mec_drive} pA")
            
            for trial in range(config.n_trials):
                firing_rates = {}
                circuit.reset_state()
                
                mec_input = torch.ones(circuit.circuit_params.n_mec, device=device) * mec_drive
                activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
                
                for t in range(config.simulation_duration):
                    external_drive = {'mec': mec_input}
                    activities = circuit({}, external_drive)
                    
                    if t >= config.warmup_duration:
                        for pop in activities_over_time:
                            if pop in activities:
                                activities_over_time[pop].append(activities[pop].clone())
                
                trial_loss = 0.0
                for pop in activities_over_time:
                    if len(activities_over_time[pop]) > 0:
                        pop_time_series = torch.stack(activities_over_time[pop])
                        mean_rates = torch.mean(pop_time_series, dim=0)
                        
                        if pop in targets.target_rates:
                            target_rate = targets.target_rates[pop]
                            actual_rate = torch.mean(mean_rates).item()
                            firing_rates[pop] = actual_rate
                            tolerance = targets.rate_tolerance[pop]
                            
                            is_zero_rate = np.isclose(actual_rate, 0.0, 1e-2, 1e-2)
                            
                            if is_zero_rate:
                                rate_loss = 1e2
                            else:
                                error = abs(actual_rate - target_rate)
                                rate_loss = error if error > tolerance else 0.5 * (error / tolerance) ** 2
                            
                            trial_loss += rate_loss
                        
                        if pop in targets.sparsity_targets:
                            target_sparsity = targets.sparsity_targets[pop]
                            actual_sparsity = (torch.sum(mean_rates > targets.activity_threshold) / len(mean_rates)).item()
                            trial_loss += (actual_sparsity - target_sparsity) ** 2
                
                constraint_violation, _ = evaluate_rate_ordering_constraints(
                    firing_rates, targets.rate_ordering_constraints
                )
                trial_loss += targets.constraint_violation_weight * constraint_violation
                baseline_loss += trial_loss
        
        baseline_loss /= (len(config.mec_drive_levels) * config.n_trials)
        
        if verbose:
            print(f"\n  Average baseline loss: {baseline_loss:.6f}")

        # === OPTOGENETIC OBJECTIVES ===
        opto_loss = 0.0
        opto_targets = targets.optogenetic_targets
        
        if verbose:
            print(f"\n{'='*80}")
            print("Optogenetic stimulation evaluation")
            print(f"{'='*80}")
        
        for target_pop in ['pv', 'sst']:
            if verbose:
                print(f"\n--- {target_pop.upper()} Stimulation ---")
            
            target_pop_opto_loss = 0.0
    
            for trial in range(config.n_trials):
                opto_results = simulate_optogenetic_stimulation(
                    circuit_factory_data,
                    connection_modulation,
                    target_pop,
                    opto_targets.stimulation_intensity,
                    mec_current=config.mec_drive_levels[0],
                    device=device
                )

<<<<<<< HEAD
                trial_opto_loss, _ = evaluate_optogenetic_objectives(
                    opto_results, target_pop, opto_targets, verbose=verbose
                )
                target_pop_opto_loss += trial_opto_loss
=======
                if verbose and config.n_trials > 1:
                    print(f"\n  Trial {trial + 1}/{config.n_trials}:")

                opto_results = simulate_optogenetic_stimulation(
                    circuit_factory_data,
                    connection_modulation,
                    target_pop,
                    opto_targets.stimulation_intensity,
                    mec_current=config.mec_drive_levels[0]  # Use first MEC drive level
                )

                # Evaluate optogenetic objectives for this trial
                trial_opto_loss, loss_components = evaluate_optogenetic_objectives(
                    opto_results,
                    target_pop,
                    opto_targets,
                    verbose=verbose
                )

                target_pop_opto_loss += trial_opto_loss

>>>>>>> 8c504ed274620d21a13a6a8594f9b4617da999d4
                
            target_pop_opto_loss /= config.n_trials
            opto_loss += target_pop_opto_loss
            
        if verbose:
            print(f"\n  Total optogenetic loss: {opto_loss:.6f}")
        
        # === COMBINED LOSS ===
        total_loss = (targets.baseline_weight * baseline_loss + 
                     targets.optogenetic_weight * opto_loss)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"  Baseline: {baseline_loss:.6f} x {targets.baseline_weight} = {targets.baseline_weight * baseline_loss:.6f}")
            print(f"  Optogenetic: {opto_loss:.6f} x {targets.optogenetic_weight} = {targets.optogenetic_weight * opto_loss:.6f}")
            print(f"  TOTAL: {total_loss:.6f}")
            print(f"{'='*80}\n")
        
        return total_loss
        
    except Exception as e:
        print(f"Worker error: {e}")
        import traceback
        traceback.print_exc()
        return 1e6


def evaluate_particle_worker(args):
    """Particle evaluation including optogenetic objectives"""
    position, connection_names, circuit_factory_data, targets, config, device = args
    
    try:
<<<<<<< HEAD
        loss = evaluate_de_candidate_worker(
            position, connection_names, circuit_factory_data, targets, config, device=device
=======
        loss = evaluate_candidate_worker(
            position, connection_names, circuit_factory_data, targets, config
>>>>>>> 8c504ed274620d21a13a6a8594f9b4617da999d4
        )
        connection_modulation = dict(zip(connection_names, position))
        return position, loss, connection_modulation
    except Exception as e:
        print(f"Particle worker error: {e}")
        return position, 1e6, dict(zip(connection_names, position))


def print_new_best_diagnostics(position, loss, connection_names, 
                               circuit_factory_data, targets, config, device):
    """Print comprehensive diagnostics when a new best solution is found"""
    print(f"\n{'#'*80}")
    print("NEW BEST SOLUTION FOUND")
    print(f"{'#'*80}")
    print(f"Loss: {loss:.6f}\n")
    
<<<<<<< HEAD
    recomputed_loss = evaluate_de_candidate_worker(
        position, connection_names, circuit_factory_data, 
        targets, config, device=device, verbose=True
=======
    # Evaluate with verbose output
    recomputed_loss = evaluate_candidate_worker(
        position, 
        connection_names, 
        circuit_factory_data, 
        targets, 
        config,
        verbose=True
    )

    # Verify that optimizer loss matches recomputed loss
    print(f"\n{'='*80}")
    print("Loss verification")
    print(f"{'='*80}")
    print(f"  Loss from optimizer:  {loss:.6f}")
    print(f"  Recomputed loss:      {recomputed_loss:.6f}")
    
    # Additional summary statistics
    connection_modulation = dict(zip(connection_names, position))
    
    print(f"\n{'='*80}")
    print("Corner case detection")
    print(f"{'='*80}")
    
    # Quick check for potential issues
    from DG_circuit_dendritic_somatic_transfer import (
        DentateCircuit, PerConnectionSynapticParams
>>>>>>> 8c504ed274620d21a13a6a8594f9b4617da999d4
    )
    
    print(f"\nLoss verification: {loss:.6f} vs {recomputed_loss:.6f}")
    print(f"{'#'*80}\n")


def run_global_optimization(optimization_config,
                            device: Optional[torch.device] = None,
                            n_workers=1,
                            n_threads_per_worker=1,
                            method='particle_swarm',
                            n_particles=20,
                            diagnostic_frequency=5):
    """
    Run global optimization with optogenetic objectives
    
    Args:
        optimization_config: OptimizationConfig instance
        device: Device to run on (None for auto-detect, 'cpu', or 'cuda')
        n_workers: Number of parallel workers (for CPU multiprocessing)
        n_threads_per_worker: Threads per worker
        method: 'particle_swarm' or 'differential_evolution'
        diagnostic_frequency: How often to print detailed diagnostics
    """
    from DG_circuit_dendritic_somatic_transfer import (
        CircuitParams, PerConnectionSynapticParams, OpsinParams
    )
    from scipy.optimize import differential_evolution
    import multiprocessing as mp
    
    # Device setup
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    print(f"Optimization device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    # Setup
    circuit_params = CircuitParams()
    base_synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()
    
    # Create targets
    base_targets = create_default_targets()
    
    targets = CombinedOptimizationTargets(
        target_rates=base_targets.target_rates,
        sparsity_targets=base_targets.sparsity_targets,
        rate_ordering_constraints=base_targets.rate_ordering_constraints,
        optogenetic_targets=OptogeneticTargets(),
        baseline_weight=1.0,
        optogenetic_weight=1.0
    )
    
    # Circuit factory data
    circuit_factory_data = (
        circuit_params,
        {
            'ampa_g_mean': base_synaptic_params.ampa_g_mean,
            'ampa_g_std': base_synaptic_params.ampa_g_std,
            'gaba_g_mean': base_synaptic_params.gaba_g_mean,
            'gaba_g_std': base_synaptic_params.gaba_g_std,
            'distribution': base_synaptic_params.distribution,
        },
        opsin_params
    )
    
    connection_names = list(targets.connection_bounds.keys())
    bounds = [targets.connection_bounds.get(name, (0.1, 5.0)) for name in connection_names]
    
    if device.type == 'cpu':
        n_cores = mp.cpu_count()
        if n_workers is None:
            n_workers = max(1, n_cores // max(1, n_threads_per_worker))
        configure_torch_threads(n_threads_per_worker)
    
    print(f"Starting {method.upper()} optimization")
    print(f"  Device: {device}")
    if device.type == 'cpu':
        print(f"  Workers: {n_workers}, Threads/worker: {n_threads_per_worker}")
    print(f"  Baseline weight: {targets.baseline_weight}, Optogenetic weight: {targets.optogenetic_weight}")
    
    n_new_bests = [0]
    
<<<<<<< HEAD
    if method == 'particle_swarm':
=======
    if method == 'differential_evolution':
        # Use Differential Evolution
        objective = partial(
            evaluate_candidate_worker,
            connection_names=connection_names,
            circuit_factory_data=circuit_factory_data,
            targets=targets,
            config=optimization_config
        )
        
        best_loss = float('inf')
        best_params = None
        history = {'loss': [], 'parameters': []}
        
        def callback(xk, convergence):
            loss = evaluate_candidate_worker(
                xk, connection_names, circuit_factory_data, targets, optimization_config
            )
            connection_modulation = dict(zip(connection_names, xk))
            
            history['loss'].append(loss)
            history['parameters'].append(connection_modulation)
            
            nonlocal best_loss, best_params
            if loss < best_loss:
                best_loss = loss
                best_params = connection_modulation.copy()
                print(f"New best loss: {loss:.6f}")
            
            return False
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=optimization_config.max_iterations,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=True,
            polish=False,
            workers=n_workers if n_workers > 1 else 1,
            updating='deferred',
            callback=callback,
        )
        
        return {
            'optimized_connection_modulation': best_params,
            'best_loss': best_loss,
            'n_evaluations': result.nfev,
            'n_iterations': result.nit,
            'success': result.success,
            'history': history,
            'method': 'differential_evolution',
            'targets': targets
        }
    
    elif method == 'particle_swarm':
        # Use Particle Swarm Optimization
        n_particles = 142
        n_dimensions = len(connection_names)
>>>>>>> 8c504ed274620d21a13a6a8594f9b4617da999d4
        max_iterations = optimization_config.max_iterations
        w_max, w_min, w = 0.9, 0.2, 0.9
        
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, len(connection_names)))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, len(connection_names)))
        
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(n_particles, float('inf'))
        global_best_position, global_best_score = None, float('inf')
        
        history = {'loss': [], 'parameters': []}
        use_multiprocessing = device.type == 'cpu' and n_workers > 1
        
        if use_multiprocessing:
            ctx = mp.get_context('spawn')
        
        previous_best, no_improvement = float('inf'), 0
        
        for iteration in range(max_iterations):
            print(f"\nPSO Iteration {iteration+1}/{max_iterations}")
            c1, c2 = np.random.uniform(1.5, 2.5), np.random.uniform(1.5, 2.5)
            
            eval_args = [
                (positions[i], connection_names, circuit_factory_data, targets, optimization_config, device)
                for i in range(n_particles)
            ]
            
            if use_multiprocessing:
                with ctx.Pool(processes=n_workers) as pool:
                    results = list(tqdm.tqdm(pool.imap_unordered(evaluate_particle_worker, eval_args),
                                            total=n_particles, desc="Evaluating"))
            else:
                results = [evaluate_particle_worker(args) for args in tqdm.tqdm(eval_args, desc="Evaluating")]

            for i, (position, score, connection_modulation) in enumerate(results):
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = position.copy()
                
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = position.copy()
                    best_params = connection_modulation.copy()
                    print(f"\n  New best: {global_best_score:.6f}")
                    
                    if n_new_bests[0] % diagnostic_frequency == 0:
<<<<<<< HEAD
                        print_new_best_diagnostics(position, score, connection_names,
                                                  circuit_factory_data, targets, optimization_config, device)
                    n_new_bests[0] += 1

=======
                        print_new_best_diagnostics(
                            position, score, connection_names,
                            circuit_factory_data, targets, optimization_config
                        )

                    n_new_bests[0] += 1
                    
>>>>>>> 8c504ed274620d21a13a6a8594f9b4617da999d4
                history['loss'].append(score)
                history['parameters'].append(connection_modulation)

            # Adapt inertia
            improvement = previous_best - global_best_score
            if improvement > 0.1:
                w = min(w + 0.05, w_max)
                no_improvement = 0
            else:
                w = max(w - 0.2 * w, w_min)
                no_improvement += 1
                
            previous_best = global_best_score

            if no_improvement > 3:
                w = w_max
                no_improvement = 0
            
            # Update particles
            for i in range(n_particles):
                r1, r2 = np.random.random(len(connection_names)), np.random.random(len(connection_names))
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] = np.clip(positions[i] + velocities[i], lower_bounds, upper_bounds)
        
        print("\nFinal diagnostics:")
        print_new_best_diagnostics(global_best_position, global_best_score, connection_names,
                                  circuit_factory_data, targets, optimization_config, device)
        
        return {
            'optimized_connection_modulation': dict(zip(connection_names, global_best_position)),
            'best_loss': global_best_score,
            'n_evaluations': max_iterations * n_particles,
            'history': history,
            'method': 'particle_swarm',
            'targets': targets,
            'device': str(device)
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    from DG_circuit_optimization import create_default_global_opt_config
    import argparse
    
    parser = argparse.ArgumentParser(description='Optogenetic Circuit Optimizer')
    parser.add_argument('--device', type=str, default=None, help='Device: cpu, cuda, or None for auto')
    parser.add_argument('--method', type=str, default='particle_swarm')
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--n-threads', type=int, default=1)
    parser.add_argument('--n-particles', type=int, default=20)
    parser.add_argument('--max-iterations', type=int, default=6)
    
    args = parser.parse_args()
    
    config = create_default_global_opt_config()
    config.max_iterations = args.max_iterations
    
    results = run_global_optimization(
        config, device=args.device, n_workers=args.n_workers,
        n_threads_per_worker=args.n_threads, method=args.method,
        n_particles=args.n_particles
    )
    
    print(f"\nOptimization Complete!")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Device: {results['device']}")
