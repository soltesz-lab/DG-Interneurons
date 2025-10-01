#!/usr/bin/env python3
"""
Optimization framework including optogenetic stimulation effects

Adds optogenetic stimulation objectives to the circuit optimization,
targeting the paradoxical excitation effects observed by Hainmueller et al.
"""
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import torch
import numpy as np
from functools import partial
import tqdm

# Import existing optimization components
from DG_circuit_optimization import (
    OptimizationTargets,
    OptimizationConfig,
    evaluate_rate_ordering_constraints,
    configure_torch_threads
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
    
    # Minimum firing rate change to count as "activated"
    # (as fraction of baseline standard deviation)
    activation_threshold: float = 2.0
    
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
    
    # Weight for optogenetic objectives in total loss
    opto_loss_weight: float = 0.5

@dataclass
class CombinedOptimizationTargets(OptimizationTargets):
    """Combined targets including both baseline and optogenetic objectives"""
    
    # Optogenetic targets
    optogenetic_targets: OptogeneticTargets = field(default_factory=OptogeneticTargets)
    
    # Overall weight distribution
    baseline_weight: float = 0.5
    optogenetic_weight: float = 0.5


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
                                     mec_current: float = 80.0) -> Dict:
    """
    Run optogenetic stimulation experiment on circuit
    
    Returns firing rate statistics for all populations
    """
    from DG_circuit_dendritic_somatic_transfer import (
        DentateCircuit, PerConnectionSynapticParams
    )
    from DG_protocol import OpsinExpression
    
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
    
    # Create circuit
    circuit = DentateCircuit(circuit_params, synaptic_params, opsin_params)
    
    # Create opsin expression
    n_cells = getattr(circuit_params, f'n_{target_pop}')
    opsin = OpsinExpression(opsin_params, n_cells)
    target_positions = circuit.layout.positions[target_pop]
    
    # Calculate activation probabilities
    activation_prob = opsin.calculate_activation(target_positions, light_intensity)
    
    # Simulation parameters
    dt = circuit_params.dt
    duration = 1650.0  # ms
    stim_start = 650.0  # ms
    warmup = 150.0  # ms
    
    n_steps = int(duration / dt)
    stim_start_step = int(stim_start / dt)
    warmup_step = int(warmup / dt)
    
    # Storage
    device = torch.device('cpu')
    activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
    
    # Run simulation
    circuit.reset_state()

    for t in range(n_steps):
        current_time = t * dt
        
        # Optogenetic activation
        direct_activation = {}
        if t >= stim_start_step:
            direct_activation[target_pop] = activation_prob * 200.0
        
        # MEC drive
        external_drive = {'mec': torch.ones(circuit_params.n_mec, device=device) * mec_current}
        
        # Update circuit
        current_activity = circuit(direct_activation, external_drive)
        
        # Store activities after warmup
        if t >= warmup_step:
            for pop in activities_over_time:
                if pop in current_activity:
                    activities_over_time[pop].append(current_activity[pop].clone().cpu())

    # Calculate statistics
    baseline_end_step = stim_start_step - warmup_step
    results = {}
    
    for pop in activities_over_time:
        if len(activities_over_time[pop]) > 0:
            pop_time_series = torch.stack(activities_over_time[pop])  # (time, neurons)
            
            # Baseline and stimulation periods
            baseline_rates = torch.mean(pop_time_series[:baseline_end_step], dim=0)
            stim_rates = torch.mean(pop_time_series[baseline_end_step:], dim=0)
            
            # Changes
            rate_changes = stim_rates - baseline_rates
            baseline_std = torch.std(baseline_rates)
            
            # Fraction activated
            activated_fraction = (torch.sum(rate_changes > 2 * baseline_std) / len(rate_changes)).item()
            
            # Gini coefficients
            baseline_gini = calculate_gini_coefficient(baseline_rates.numpy())
            stim_gini = calculate_gini_coefficient(stim_rates.numpy())
            gini_change = stim_gini - baseline_gini
            
            results[pop] = {
                'baseline_mean': torch.mean(baseline_rates).item(),
                'stim_mean': torch.mean(stim_rates).item(),
                'mean_change': torch.mean(rate_changes).item(),
                'activated_fraction': activated_fraction,
                'baseline_gini': baseline_gini,
                'stim_gini': stim_gini,
                'gini_change': gini_change,
            }
    
    return results


def evaluate_optogenetic_objectives(opto_results: Dict, 
                                    target_pop: str,
                                    opto_targets: OptogeneticTargets) -> Tuple[float, Dict]:
    """
    Calculate loss for optogenetic stimulation objectives
    
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
                
                # Squared error
                error = (actual_fraction - target_fraction) ** 2
                rate_increase_loss += error
        
        loss_components['rate_increase'] = rate_increase_loss
        total_loss += rate_increase_loss
    
    # Target Gini increases (inequality)
    if target_pop in opto_targets.target_gini_increase:
        gini_loss = 0.0
        
        for affected_pop, target_gini_change in opto_targets.target_gini_increase[target_pop].items():
            if affected_pop in opto_results and affected_pop != target_pop:
                actual_gini_change = opto_results[affected_pop]['gini_change']
                
                # We want to match the target Gini increase
                error = (actual_gini_change - target_gini_change) ** 2
                gini_loss += error
        
        loss_components['gini_increase'] = gini_loss
        total_loss += gini_loss
    
    return total_loss, loss_components


def evaluate_de_candidate_worker(param_array, connection_names, circuit_factory_data,
                                          targets: CombinedOptimizationTargets, config):
    """
    Objective worker function including optogenetic objectives
    """
    try:
        from DG_circuit_dendritic_somatic_transfer import (
            DentateCircuit, PerConnectionSynapticParams
        )
        
        # Convert parameter array to connection modulation dict
        connection_modulation = dict(zip(connection_names, param_array))
        
        # Unpack circuit factory data
        circuit_params, base_synaptic_params_dict, opsin_params = circuit_factory_data
        
        # === BASELINE OBJECTIVES ===
        # Create synaptic parameters
        synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=base_synaptic_params_dict['ampa_g_mean'],
            ampa_g_std=base_synaptic_params_dict['ampa_g_std'],
            gaba_g_mean=base_synaptic_params_dict['gaba_g_mean'],
            gaba_g_std=base_synaptic_params_dict['gaba_g_std'],
            distribution=base_synaptic_params_dict['distribution'],
            connection_modulation=connection_modulation
        )
        
        # Create circuit
        circuit = DentateCircuit(circuit_params, synaptic_params, opsin_params)
        device = torch.device('cpu')
        
        # Evaluate baseline objectives
        baseline_loss = 0.0
        
        for mec_drive in config.mec_drive_levels:
            for trial in range(config.n_trials):
                firing_rates = {}
                circuit.reset_state()
                
                # Run simulation
                mec_input = torch.ones(circuit.circuit_params.n_mec, device=device) * mec_drive
                activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
                
                for t in range(config.simulation_duration):
                    external_drive = {'mec': mec_input}
                    activities = circuit({}, external_drive)
                    
                    if t >= config.warmup_duration:
                        for pop in activities_over_time:
                            if pop in activities:
                                activities_over_time[pop].append(activities[pop].clone().cpu())
                
                # Calculate baseline loss
                trial_loss = 0.0
                for pop in activities_over_time:
                    if len(activities_over_time[pop]) > 0:
                        pop_time_series = torch.stack(activities_over_time[pop])
                        mean_rates = torch.mean(pop_time_series, dim=0)
                        
                        # Firing rate loss
                        if pop in targets.target_rates:
                            target_rate = targets.target_rates[pop]
                            actual_rate = torch.mean(mean_rates).item()
                            firing_rates[pop] = actual_rate
                            tolerance = targets.rate_tolerance[pop]
                            
                            if np.isclose(actual_rate, 0.0, 1e-2, 1e-2):
                                rate_loss = 1e2
                            else:
                                error = abs(actual_rate - target_rate)
                                if error > tolerance:
                                    rate_loss = error - tolerance
                                else:
                                    rate_loss = 0.5 * (error / tolerance) ** 2
                            
                            trial_loss += rate_loss
                        
                        # Sparsity loss
                        if pop in targets.sparsity_targets:
                            target_sparsity = targets.sparsity_targets[pop]
                            actual_sparsity = (torch.sum(mean_rates > targets.activity_threshold) / len(mean_rates)).item()
                            sparsity_error = (actual_sparsity - target_sparsity) ** 2
                            trial_loss += sparsity_error
                
                # Add constraint violations
                constraint_violation, _ = evaluate_rate_ordering_constraints(
                    firing_rates, 
                    targets.rate_ordering_constraints
                )
                trial_loss += targets.constraint_violation_weight * constraint_violation
                
                baseline_loss += trial_loss
        
        # Average baseline loss
        baseline_loss /= (len(config.mec_drive_levels) * config.n_trials)

        # === OPTOGENETIC OBJECTIVES ===
        opto_loss = 0.0
        opto_targets = targets.optogenetic_targets
        
        # Run optogenetic stimulation for PV and SST
        for target_pop in ['pv', 'sst']:
            opto_results = simulate_optogenetic_stimulation(
                circuit_factory_data,
                connection_modulation,
                target_pop,
                opto_targets.stimulation_intensity,
                mec_current=config.mec_drive_levels[0]  # Use first MEC drive level
            )

            # Evaluate optogenetic objectives
            pop_opto_loss, _ = evaluate_optogenetic_objectives(
                opto_results,
                target_pop,
                opto_targets
            )
            
            opto_loss += pop_opto_loss
        
        
        # Average optogenetic loss across both stimulation types
        opto_loss /= 2.0
        
        # === COMBINED LOSS ===
        total_loss = (targets.baseline_weight * baseline_loss + 
                     targets.optogenetic_weight * opto_loss)
        
        return total_loss
        
    except Exception as e:
        print(f"DE worker error: {e}")
        import traceback
        traceback.print_exc()
        return 1e6


def evaluate_particle_worker(args):
    """
    Particle evaluation including optogenetic objectives
    """
    position, connection_names, circuit_factory_data, targets, config = args
    
    try:
        # Use the extended DE worker function
        loss = evaluate_de_candidate_worker(
            position, connection_names, circuit_factory_data, targets, config
        )
        
        connection_modulation = dict(zip(connection_names, position))
        return position, loss, connection_modulation
        
    except Exception as e:
        print(f"Particle worker error: {e}")
        return position, 1e6, dict(zip(connection_names, position))


def run_global_optimization(optimization_config, n_workers=1, n_threads_per_worker=1,
                            method='particle_swarm'):
    """
    Run global optimization with optogenetic objectives
    
    Args:
        optimization_config: OptimizationConfig instance
        n_workers: Number of parallel workers
        n_threads_per_worker: Threads per worker
        method: 'particle_swarm' or 'differential_evolution'
    """
    from DG_circuit_dendritic_somatic_transfer import (
        CircuitParams, PerConnectionSynapticParams, OpsinParams
    )
    from DG_circuit_optimization import GlobalCircuitOptimizer
    from scipy.optimize import differential_evolution
    import multiprocessing as mp
    
    # Setup
    circuit_params = CircuitParams()
    base_synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()
    
    # Create targets
    base_targets = OptimizationTargets(
        target_rates={
            'gc': 0.5,
            'mc': 1.1,
            'pv': 6.0,
            'sst': 4.0,
        },
        sparsity_targets={
            'gc': 0.08,
            'mc': 0.50,
            'pv': 0.85,
            'sst': 0.60,
        }
    )
    
    opto_targets = OptogeneticTargets()
    
    targets = CombinedOptimizationTargets(
        target_rates=base_targets.target_rates,
        sparsity_targets=base_targets.sparsity_targets,
        rate_ordering_constraints=base_targets.rate_ordering_constraints,
        optogenetic_targets=opto_targets,
        baseline_weight=0.5,
        optogenetic_weight=0.5
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
    
    # Connection names and bounds
    connection_names = list(targets.connection_bounds.keys())
    bounds = [targets.connection_bounds.get(name, (0.1, 5.0)) for name in connection_names]
    
    # Configure threading
    n_cores = mp.cpu_count()
    if n_workers is None:
        n_workers = max(1, n_cores // max(1, n_threads_per_worker))
    
    print(f"Starting combined {method.upper()} optimization")
    print(f"  Workers: {n_workers}")
    print(f"  Threads per worker: {n_threads_per_worker}")
    print(f"  Baseline weight: {targets.baseline_weight}")
    print(f"  Optogenetic weight: {targets.optogenetic_weight}")
    
    configure_torch_threads(n_threads_per_worker)
    
    if method == 'differential_evolution':
        # Use Differential Evolution
        objective = partial(
            evaluate_de_candidate_worker,
            connection_names=connection_names,
            circuit_factory_data=circuit_factory_data,
            targets=targets,
            config=optimization_config
        )
        
        best_loss = float('inf')
        best_params = None
        history = {'loss': [], 'parameters': []}
        
        def callback(xk, convergence):
            loss = evaluate_de_candidate_worker(
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
            'method': 'differential_evolution'
        }
    
    elif method == 'particle_swarm':
        # Use Particle Swarm Optimization
        n_particles = 20
        n_dimensions = len(connection_names)
        max_iterations = optimization_config.max_iterations
        
        # PSO parameters
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        # Initialize
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, n_dimensions))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_dimensions))
        
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(n_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        
        history = {'loss': [], 'parameters': []}
        
        ctx = mp.get_context('spawn')
        
        for iteration in range(max_iterations):
            print(f"PSO Iteration {iteration+1}/{max_iterations}")
            
            # Prepare evaluation arguments
            eval_args = [
                (positions[i], connection_names, circuit_factory_data, targets, optimization_config)
                for i in range(n_particles)
            ]
            
            # Evaluate in parallel
            with ctx.Pool(processes=n_workers) as pool:
                results = list(tqdm.tqdm(
                    pool.imap_unordered(evaluate_particle_worker, eval_args),
                    total=n_particles,
                    desc="Evaluating particles",
                    position=1,
                    leave=False
                ))

            # Process results
            for i, (position, score, connection_modulation) in enumerate(results):

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = position.copy()
                
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = position.copy()
                    best_params = connection_modulation.copy()

                    print(f"  New global best: {global_best_score:.6f}")
                    print(f"  Parameters: {best_params}")
                
                history['loss'].append(score)
                history['parameters'].append(connection_modulation)
            
            # Update velocities and positions
            for i in range(n_particles):
                r1, r2 = np.random.random(n_dimensions), np.random.random(n_dimensions)
                
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bounds, upper_bounds)
        
        return {
            'optimized_connection_modulation': dict(zip(connection_names, global_best_position)),
            'best_loss': global_best_score,
            'n_evaluations': max_iterations * n_particles,
            'history': history,
            'method': 'particle_swarm'
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    from DG_circuit_optimization import create_default_global_opt_config
    
    print("Optimization with Optogenetic Objectives")
    print("=" * 60)
    
    # Create configuration
    config = create_default_global_opt_config()
    
    # Run optimization
    results = run_global_optimization(
        config,
        n_workers=10,
        n_threads_per_worker=1,
        method='particle_swarm'
    )
    
    print("\nOptimization Complete!")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Method: {results['method']}")
    print(f"Evaluations: {results['n_evaluations']}")
    
    # Print best parameters
    print("\nOptimized connection modulation:")
    for conn, value in sorted(results['optimized_connection_modulation'].items()):
        print(f"  {conn}: {value:.3f}")
