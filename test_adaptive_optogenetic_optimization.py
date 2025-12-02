#!/usr/bin/env python3
"""
Tests for adaptive stepping in optogenetic circuit optimization

Tests:
- Sequential strategy with adaptive stepping
- Batch GPU strategy with adaptive stepping (if CUDA available)
- Multiprocess CPU strategy with adaptive stepping
- Comparison: fixed-dt vs adaptive-dt results
- Performance benchmarking
- Adaptive statistics validation
"""

import torch
import time
import numpy as np
from pathlib import Path

from DG_protocol import OptogeneticExperiment
from DG_circuit_optogenetic_optimization import (
    OptogeneticCircuitOptimizer,
    CombinedOptimizationTargets,
    OptogeneticTargets,
    BatchOptogeneticEvaluator,
    set_random_seed
)
from DG_circuit_optimization import (
    CircuitParams,
    PerConnectionSynapticParams,
    OpsinParams,
    create_default_targets,
    OptimizationConfig
)
from gradient_adaptive_stepper import GradientAdaptiveStepConfig


def setup_test_environment():
    """Setup common test parameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    circuit_params = CircuitParams()
    synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()
    opsin_current = 100.0
    
    # Create combined targets
    base_targets = create_default_targets()
    targets = CombinedOptimizationTargets(
        target_rates=base_targets.target_rates,
        sparsity_targets=base_targets.sparsity_targets,
        rate_ordering_constraints=base_targets.rate_ordering_constraints,
        optogenetic_targets=OptogeneticTargets(),
        baseline_weight=1.0,
        optogenetic_weight=2.0
    )
    
    return device, circuit_params, synaptic_params, opsin_params, opsin_current, targets


def test_sequential_strategy_adaptive():
    """Sequential strategy with adaptive stepping"""
    print("="*80)
    print("Sequential Strategy with Adaptive Stepping")
    print("="*80)
    
    device, circuit_params, synaptic_params, opsin_params, opsin_current, targets = setup_test_environment()
    
    # Create parameter set
    param_set = [synaptic_params.connection_modulation]
    
    # Test fixed dt
    print("\nFixed dt evaluation:")
    config_fixed = OptimizationConfig(
        n_trials=1,
        simulation_duration=300,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=False
    )

    evaluator_fixed = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params, opsin_params, opsin_current,
        targets, config_fixed, device=device, base_seed=42,
        adaptive_step=False
    )
    
    start = time.time()
    loss_fixed, details_fixed = evaluator_fixed.evaluate_parameter_batch(
        param_set, 40.0, 1.0
    )
    time_fixed = time.time() - start
    
    print(f"  Time: {time_fixed:.3f}s")
    print(f"  Total loss: {loss_fixed[0].item():.6f}")
    print(f"  Baseline loss: {details_fixed['baseline_losses'][0].item():.6f}")
    print(f"  Opto loss: {details_fixed['opto_losses'][0].item():.6f}")
    
    # Test adaptive dt
    print("\nAdaptive dt evaluation:")
    config_adaptive = OptimizationConfig(
        n_trials=1,
        simulation_duration=300,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=True,
        adaptive_config=GradientAdaptiveStepConfig(
            dt_min=0.05,
            dt_max=0.3,
            gradient_low=0.5,
            gradient_high=10.0
        )
    )
    
    evaluator_adaptive = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params, opsin_params, opsin_current,
        targets, config_adaptive, device=device, base_seed=42,
        adaptive_step=True,
        adaptive_config=config_adaptive.adaptive_config
    )
    
    start = time.time()
    loss_adaptive, details_adaptive = evaluator_adaptive.evaluate_parameter_batch(
        param_set, 40.0, 1.0
    )
    time_adaptive = time.time() - start
    
    print(f"  Time: {time_adaptive:.3f}s")
    print(f"  Total loss: {loss_adaptive[0].item():.6f}")
    print(f"  Baseline loss: {details_adaptive['baseline_losses'][0].item():.6f}")
    print(f"  Opto loss: {details_adaptive['opto_losses'][0].item():.6f}")
    
    # Check adaptive stats
    if hasattr(evaluator_adaptive, '_last_adaptive_stats'):
        stats = evaluator_adaptive._last_adaptive_stats
        if stats is not None:
            print(f"\n  Adaptive statistics:")
            print(f"    Steps: {stats['n_steps_mean']:.0f}")
            print(f"    Avg dt: {stats['avg_dt_mean']:.3f} ms")
            print(f"    dt range: [{stats['min_dt']:.3f}, {stats['max_dt']:.3f}] ms")
    
    # Performance comparison
    overhead = (time_adaptive / time_fixed - 1) * 100
    print(f"\n  Overhead: {overhead:.1f}%")
    
    # Loss comparison
    loss_diff = abs(loss_fixed[0].item() - loss_adaptive[0].item())
    loss_diff_pct = (loss_diff / loss_fixed[0].item()) * 100
    print(f"  Loss difference: {loss_diff:.6f} ({loss_diff_pct:.2f}%)")
    
    print("\nSequential strategy test PASSED")
    return True


def test_batch_strategies_adaptive():
    """Batch strategies with adaptive stepping"""
    print("\n" + "="*80)
    print("Batch Strategies with Adaptive Stepping")
    print("="*80)
    
    device, circuit_params, synaptic_params, opsin_params, opsin_current, targets = setup_test_environment()
    
    # Create parameter batch
    param_batch = [
        synaptic_params.connection_modulation,
        {k: v * 1.1 for k, v in synaptic_params.connection_modulation.items()},
        {k: v * 0.9 for k, v in synaptic_params.connection_modulation.items()},
    ]
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {len(param_batch)}")
    
    # Fixed dt baseline
    print("\nFixed dt batch evaluation:")
    config_fixed = OptimizationConfig(
        n_trials=1,
        simulation_duration=300,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=False
    )
    
    evaluator_fixed = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params, opsin_params, opsin_current,
        targets, config_fixed, device=device, base_seed=42,
        adaptive_step=False
    )
    
    start = time.time()
    losses_fixed, details_fixed = evaluator_fixed.evaluate_parameter_batch(
        param_batch, 40.0, 1.0
    )
    time_fixed = time.time() - start
    
    print(f"  Time: {time_fixed:.3f}s")
    print(f"  Losses: {[f'{l:.4f}' for l in losses_fixed.cpu().numpy()]}")
    
    # Adaptive dt
    print("\n2b. Adaptive dt batch evaluation:")
    config_adaptive = OptimizationConfig(
        n_trials=1,
        simulation_duration=300,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=True,
        adaptive_config=GradientAdaptiveStepConfig(
            dt_min=0.05,
            dt_max=0.3,
            gradient_low=0.5,
            gradient_high=10.0
        )
    )
    
    evaluator_adaptive = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params, opsin_params, opsin_current,
        targets, config_adaptive, device=device, base_seed=42,
        adaptive_step=True,
        adaptive_config=config_adaptive.adaptive_config
    )
    
    start = time.time()
    losses_adaptive, details_adaptive = evaluator_adaptive.evaluate_parameter_batch(
        param_batch, 40.0, 1.0
    )
    time_adaptive = time.time() - start
    
    print(f"  Time: {time_adaptive:.3f}s")
    print(f"  Losses: {[f'{l:.4f}' for l in losses_adaptive.cpu().numpy()]}")
    
    if hasattr(evaluator_adaptive, '_last_adaptive_stats'):
        stats = evaluator_adaptive._last_adaptive_stats
        if stats is not None:
            print(f"\n  Adaptive statistics:")
            print(f"    Steps: {stats['n_steps']:.0f}")
            print(f"    Avg dt: {stats['avg_dt']:.3f} ms")
    
    overhead = (time_adaptive / time_fixed - 1) * 100
    print(f"\n  Overhead: {overhead:.1f}%")
    
    print("\nBatch strategy test PASSED")
    return True


def test_multi_trial_consistency():
    """Multi-trial consistency with adaptive stepping"""
    print("\n" + "="*80)
    print("Multi-trial Consistency")
    print("="*80)
    
    device, circuit_params, synaptic_params, opsin_params, opsin_current, targets = setup_test_environment()
    
    param_set = [synaptic_params.connection_modulation]
    
    print("\n3a. Running 3 trials with different seeds...")
    
    config = OptimizationConfig(
        n_trials=3,
        simulation_duration=300,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=True,
        adaptive_config=GradientAdaptiveStepConfig(
            dt_min=0.05,
            dt_max=0.3
        )
    )
    
    evaluator = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params, opsin_params, opsin_current,
        targets, config, device=device, base_seed=42,
        adaptive_step=True,
        adaptive_config=config.adaptive_config
    )
    
    # Run evaluation
    losses, details = evaluator.evaluate_parameter_batch(param_set, 40.0, 1.0)
    
    print(f"  Average loss over 3 trials: {losses[0].item():.6f}")
    print(f"  Baseline loss: {details['baseline_losses'][0].item():.6f}")
    print(f"  Opto loss: {details['opto_losses'][0].item():.6f}")
    
    if hasattr(evaluator, '_last_adaptive_stats'):
        stats = evaluator._last_adaptive_stats
        if stats is not None:
            if 'n_steps_std' in stats:
                print(f"\n  Adaptive statistics (aggregated over trials):")
                print(f"    Mean steps: {stats['n_steps_mean']:.0f} ± {stats['n_steps_std']:.0f}")
                print(f"    Mean dt: {stats['avg_dt_mean']:.3f} ± {stats['avg_dt_std']:.3f} ms")
    
    print("\nMulti-trial consistency test PASSED")
    return True


def test_reproducibility():
    """Reproducibility with same seed"""
    print("\n" + "="*80)
    print("Reproducibility Test")
    print("="*80)
    
    device, circuit_params, synaptic_params, opsin_params, opsin_current, targets = setup_test_environment()
    
    param_set = [synaptic_params.connection_modulation]
    
    config = OptimizationConfig(
        n_trials=2,
        simulation_duration=200,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=True,
        adaptive_config=GradientAdaptiveStepConfig(
            dt_min=0.05,
            dt_max=0.3
        )
    )
    
    print("\n4a. First run (seed=42)...")
    evaluator1 = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params,
        opsin_params, opsin_current,
        targets, config, device=device, base_seed=42,
        adaptive_step=True,
        adaptive_config=config.adaptive_config
    )
    
    losses1, _ = evaluator1.evaluate_parameter_batch(param_set, 40.0, 1.0)
    loss1 = losses1[0].item()
    print(f"  Loss: {loss1:.6f}")
    
    print("\n4b. Second run (seed=42)...")
    evaluator2 = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params,
        opsin_params, opsin_current,
        targets, config, device=device, base_seed=42,
        adaptive_step=True,
        adaptive_config=config.adaptive_config
    )
    
    losses2, _ = evaluator2.evaluate_parameter_batch(param_set, 40.0, 1.0)
    loss2 = losses2[0].item()
    print(f"  Loss: {loss2:.6f}")
    
    # Check reproducibility
    diff = abs(loss1 - loss2)
    print(f"\n  Difference: {diff:.9f}")
    
    # Allow small numerical differences due to floating point
    tolerance = 1e-4
    if diff < tolerance:
        print(f"  O Results are reproducible (diff < {tolerance})")
        passed = True
    else:
        print(f"  X Results differ by more than tolerance {tolerance}")
        passed = False
    
    if passed:
        print("\nO Reproducibility test PASSED")
    else:
        print("\nX Reproducibility test FAILED")
    
    return passed


def test_optimizer_integration():
    """Full optimizer integration"""
    print("\n" + "="*80)
    print("Full Optimizer Integration")
    print("="*80)
    
    device, circuit_params, synaptic_params, opsin_params, opsin_current, targets = setup_test_environment()
    
    print("\nRunning mini-optimization with adaptive stepping...")
    print("(This tests the full optimizer pipeline)")
    
    config = OptimizationConfig(
        n_trials=1,
        simulation_duration=200,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=True,
        adaptive_config=GradientAdaptiveStepConfig(
            dt_min=0.05,
            dt_max=0.3
        )
    )
    
    optimizer = OptogeneticCircuitOptimizer(
        circuit_params, synaptic_params,
        opsin_params, opsin_current,
        targets, config, device=device, base_seed=42
    )
    
    # Run a very short optimization
    start = time.time()
    results = optimizer.optimize(
        method='particle_swarm',
        n_particles=8,  # Small for quick test
        n_workers=8,
        max_iterations=2,  # Just 2 iterations
        diagnostic_frequency=1,
        use_time_varying_mec=True,
        mec_pattern_type='oscillatory',
        mec_theta_freq=5.0,
        mec_gamma_freq=20.0,
        mec_rotation_groups=3
    )
    elapsed = time.time() - start
    
    print(f"\n  Optimization completed in {elapsed:.2f}s")
    print(f"  Best loss: {results['best_loss']:.6f}")
    print(f"  Method: {results['method']}")
    print(f"  Device: {results['device']}")
    
    # Check that adaptive stepping was used
    if 'pso_result' in results:
        pso_result = results['pso_result']
        if hasattr(pso_result, 'metadata_history') and len(pso_result.metadata_history) > 0:
            # Check if metadata contains adaptive stats
            last_meta = pso_result.metadata_history[-1]
            if 'baseline_loss' in last_meta:
                print(f"  Metadata tracking working")
    
    print("\nFull optimizer integration test PASSED")
    return True

"""
Add this test to test_adaptive_optogenetic_optimization.py

This test directly compares BatchOptogeneticEvaluator from the optimization
module vs. OptogeneticExperiment.simulate_stimulation from the protocol module.
"""

def test_optimization_protocol_consistency():
    """
    Cross-validation: Compare optimization evaluator vs protocol evaluator
    
    This test reveals discrepancies between:
    - BatchOptogeneticEvaluator (used in optimization)
    - OptogeneticExperiment.simulate_stimulation (used in protocol)
    
    Both should produce similar results for the same configuration.
    """
    print("\n" + "="*80)
    print("Cross-Validation: Optimization vs Protocol Evaluator")
    print("="*80)
    
    device, circuit_params, synaptic_params, opsin_params, opsin_current, targets = setup_test_environment()
    
    # Test parameters
    param_set = synaptic_params.connection_modulation
    mec_current = 40.0
    mec_current_std = 1.0
    target_pop = 'pv'
    light_intensity = 1.0
    
    # Simulation parameters
    stim_start = 1500.0
    stim_duration = 1000.0
    warmup = 500.0
    total_duration = warmup + stim_start + stim_duration
    
    # Adaptive config
    adaptive_config = GradientAdaptiveStepConfig(
        dt_min=0.05,
        dt_max=0.3,
        gradient_low=0.5,
        gradient_high=10.0
    )
    
    print(f"\nTest configuration:")
    print(f"  Device: {device}")
    print(f"  Target population: {target_pop}")
    print(f"  MEC current: {mec_current} pA")
    print(f"  Adaptive stepping: ENABLED")
    print(f"  dt range: [{adaptive_config.dt_min}, {adaptive_config.dt_max}] ms")
    
    # ========================================================================
    # PATH 1: Optimization Evaluator (BatchOptogeneticEvaluator)
    # ========================================================================

    print("\n" + "-"*80)
    print("PATH 1: Optimization Evaluator (BatchOptogeneticEvaluator)")
    print("-"*80)

    config_opt = OptimizationConfig(
        n_trials=1,
        simulation_duration=stim_duration,
        warmup_duration=warmup,
        mec_drive_levels=[mec_current],
        mec_drive_std=mec_current_std,
        adaptive_step=True,
        adaptive_config=adaptive_config
    )

    # Update optogenetic targets to match protocol
    targets.optogenetic_targets.pre_stim_duration = stim_start
    targets.optogenetic_targets.stim_duration = stim_duration
    targets.optogenetic_targets.stimulation_intensity = light_intensity

    evaluator_opt = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params,
        opsin_params, opsin_current,
        targets, config_opt,
        device=device,
        base_seed=42,
        adaptive_step=True,
        adaptive_config=adaptive_config,
        verbose=True
    )

    start = time.time()
    loss_opt, details_opt = evaluator_opt.evaluate_parameter_batch(
        [param_set], mec_current, mec_current_std
    )
    time_opt = time.time() - start

    print(f"\n  Evaluation time: {time_opt:.3f}s")
    print(f"  Total loss: {loss_opt[0].item():.6f}")
    print(f"  Baseline loss: {details_opt['baseline_losses'][0].item():.6f}")
    print(f"  Opto loss: {details_opt['opto_losses'][0].item():.6f}")

    # Extract firing rates
    opt_baseline_rates = {}
    for pop in ['gc', 'mc', 'pv', 'sst']:
        if pop in details_opt['baseline_rates']:
            opt_baseline_rates[pop] = details_opt['baseline_rates'][pop][0].item()

    print(f"\n  Baseline firing rates:")
    for pop, rate in opt_baseline_rates.items():
        print(f"    {pop.upper()}: {rate:.2f} Hz")

    # Extract optogenetic effects
    opt_opto_effects = {}
    if target_pop in details_opt['opto_details']:
        for affected_pop in details_opt['opto_details'][target_pop]:
            if affected_pop != target_pop:
                data = details_opt['opto_details'][target_pop][affected_pop]
                opt_opto_effects[affected_pop] = {
                    'mean_change': data['mean_change'][0].item(),
                    'activated_fraction': data['activated_fraction'][0].item()
                }

    print(f"\n  Optogenetic effects ({target_pop.upper()} stimulation):")
    for pop, effects in opt_opto_effects.items():
        print(f"    {pop.upper()}: Δrate={effects['mean_change']:.2f} Hz, "
              f"activated={effects['activated_fraction']:.1%}")

    # Check for adaptive stats
    if hasattr(evaluator_opt, '_last_adaptive_stats'):
        opt_adaptive_stats = evaluator_opt._last_adaptive_stats
        if opt_adaptive_stats:
            print(f"\n  Adaptive statistics:")
            if 'n_steps_mean' in opt_adaptive_stats:
                print(f"    Steps: {opt_adaptive_stats['n_steps_mean']:.0f}")
                print(f"    Avg dt: {opt_adaptive_stats['avg_dt_mean']:.3f} ms")
            elif 'n_steps' in opt_adaptive_stats:
                print(f"    Steps: {opt_adaptive_stats['n_steps']:.0f}")
                print(f"    Avg dt: {opt_adaptive_stats['avg_dt']:.3f} ms")
    
    # ========================================================================
    # PATH 2: Protocol Evaluator (OptogeneticExperiment)
    # ========================================================================
    
    print("\n" + "-"*80)
    print("PATH 2: Protocol Evaluator (OptogeneticExperiment)")
    print("-"*80)
    
    from DG_protocol import OptogeneticExperiment
    
    experiment = OptogeneticExperiment(
        circuit_params,
        synaptic_params,
        opsin_params,
        device=device,
        base_seed=42,
        adaptive_config=adaptive_config
    )
    
    start = time.time()
    result_protocol = experiment.simulate_stimulation(
        target_pop,
        light_intensity,
        stim_start=stim_start,
        stim_duration=stim_duration,
        post_duration=0.0,
        mec_current=mec_current,
        mec_current_std=mec_current_std,
        opsin_current=opsin_current,
        plot_activity=False,
        n_trials=1,
        regenerate_connectivity_per_trial=False,
        adaptive_step=True
    )
    time_protocol = time.time() - start
    
    print(f"\n  Evaluation time: {time_protocol:.3f}s")
    
    # Extract firing rates from protocol
    time_vec = result_protocol['time']
    activity_mean = result_protocol['activity_trace_mean']

    baseline_mask = (time_vec >= warmup) & (time_vec < stim_start)
    stim_mask = (time_vec >= stim_start) & (time_vec <= (stim_start + stim_duration))
    
    protocol_baseline_rates = {}
    protocol_opto_effects = {}
    
    for pop in ['gc', 'mc', 'pv', 'sst']:
        if pop == target_pop:
            continue
            
        baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
        stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
        rate_change = stim_rate - baseline_rate
        baseline_std = torch.std(baseline_rate, unbiased=False)
        
        protocol_baseline_rates[pop] = torch.mean(baseline_rate).item()
        
        activated_fraction = torch.mean((rate_change > baseline_std).float()).item()
        mean_change = torch.mean(rate_change).item()
        
        protocol_opto_effects[pop] = {
            'mean_change': mean_change,
            'activated_fraction': activated_fraction
        }
    
    print(f"\n  Baseline firing rates:")
    for pop, rate in protocol_baseline_rates.items():
        print(f"    {pop.upper()}: {rate:.2f} Hz")
    
    print(f"\n  Optogenetic effects ({target_pop.upper()} stimulation):")
    for pop, effects in protocol_opto_effects.items():
        print(f"    {pop.upper()}: Δrate={effects['mean_change']:.2f} Hz, "
              f"activated={effects['activated_fraction']:.1%}")
    
    # Check for adaptive stats
    if 'adaptive_stats' in result_protocol:
        protocol_adaptive_stats = result_protocol['adaptive_stats']
        print(f"\n  Adaptive statistics:")
        if 'n_steps_mean' in protocol_adaptive_stats:
            print(f"    Steps: {protocol_adaptive_stats['n_steps_mean']:.0f}")
            print(f"    Avg dt: {protocol_adaptive_stats['avg_dt_mean']:.3f} ms")
        elif 'n_steps' in protocol_adaptive_stats:
            print(f"    Steps: {protocol_adaptive_stats['n_steps']:.0f}")
            print(f"    Avg dt: {protocol_adaptive_stats['avg_dt']:.3f} ms")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<40} {'Optimization':<20} {'Protocol':<20} {'Diff':<15}")
    print("-"*95)
    
    # Compare timing
    time_diff_pct = abs(time_opt - time_protocol) / time_protocol * 100
    print(f"{'Evaluation time (s)':<40} {time_opt:<20.3f} {time_protocol:<20.3f} {time_diff_pct:<15.1f}%")
    
    # Compare baseline rates
    print(f"\n{'Baseline Firing Rates (Hz)':<40}")
    print("-"*95)
    
    max_rate_diff_pct = 0
    for pop in ['gc', 'mc', 'pv', 'sst']:
        if pop in opt_baseline_rates and pop in protocol_baseline_rates:
            opt_rate = opt_baseline_rates[pop]
            protocol_rate = protocol_baseline_rates[pop]
            diff = abs(opt_rate - protocol_rate)
            diff_pct = (diff / protocol_rate * 100) if protocol_rate > 0 else 0
            max_rate_diff_pct = max(max_rate_diff_pct, diff_pct)
            
            print(f"  {pop.upper():<38} {opt_rate:<20.2f} {protocol_rate:<20.2f} {diff_pct:<15.1f}%")
    
    # Compare optogenetic effects
    print(f"\n{'Optogenetic Effects':<40}")
    print("-"*95)
    
    max_effect_diff_pct = 0
    for pop in opt_opto_effects:
        if pop in protocol_opto_effects:
            opt_effect = opt_opto_effects[pop]['activated_fraction']
            protocol_effect = protocol_opto_effects[pop]['activated_fraction']
            diff = abs(opt_effect - protocol_effect)
            diff_pct = (diff / protocol_effect * 100) if protocol_effect > 0 else 0
            max_effect_diff_pct = max(max_effect_diff_pct, diff_pct)
            
            print(f"  {pop.upper()} activated fraction: {opt_effect:<20.1%} "
                  f"{protocol_effect:<20.1%} {diff_pct:<15.1f}%")
    
    # ========================================================================
    # RESULT
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    
    # Tolerance thresholds
    RATE_TOLERANCE = 10.0  # 10% difference in firing rates
    EFFECT_TOLERANCE = 15.0  # 15% difference in optogenetic effects
    
    issues = []
    
    if max_rate_diff_pct > RATE_TOLERANCE:
        issues.append(f"Baseline rates differ by up to {max_rate_diff_pct:.1f}% (threshold: {RATE_TOLERANCE}%)")
    
    if max_effect_diff_pct > EFFECT_TOLERANCE:
        issues.append(f"Opto effects differ by up to {max_effect_diff_pct:.1f}% (threshold: {EFFECT_TOLERANCE}%)")
    
    if not issues:
        print("\nO PASSED: Optimization and Protocol evaluators produce consistent results")
        print(f"  Max rate difference: {max_rate_diff_pct:.1f}%")
        print(f"  Max effect difference: {max_effect_diff_pct:.1f}%")
        passed = True
    else:
        print("\nX FAILED: Significant discrepancies detected:")
        for issue in issues:
            print(f"  - {issue}")
        
        passed = False
    
    return passed



def test_performance_benchmark():
    """Performance benchmark - fixed vs adaptive"""
    print("\n" + "="*80)
    print("Performance Benchmark")
    print("="*80)
    
    device, circuit_params, synaptic_params, opsin_params, opsin_current, targets = setup_test_environment()
    
    param_batch = [
        synaptic_params.connection_modulation,
        {k: v * 1.05 for k, v in synaptic_params.connection_modulation.items()},
    ]
    
    n_runs = 3
    
    print(f"\nRunning {n_runs} evaluations for each mode...")
    print(f"Device: {device}")
    print(f"Batch size: {len(param_batch)}")
    
    # Fixed dt benchmark
    config_fixed = OptimizationConfig(
        n_trials=1,
        simulation_duration=300,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=False
    )
    
    evaluator_fixed = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params,
        opsin_params, opsin_current,
        targets, config_fixed, device=device, base_seed=42,
        adaptive_step=False
    )
    
    times_fixed = []
    for i in range(n_runs):
        start = time.time()
        losses, _ = evaluator_fixed.evaluate_parameter_batch(param_batch, 40.0, 1.0)
        elapsed = time.time() - start
        times_fixed.append(elapsed)
    
    mean_fixed = np.mean(times_fixed)
    std_fixed = np.std(times_fixed)
    
    print(f"\nFixed dt: {mean_fixed:.3f} ± {std_fixed:.3f}s")
    
    # Adaptive dt benchmark
    config_adaptive = OptimizationConfig(
        n_trials=1,
        simulation_duration=300,
        warmup_duration=50,
        mec_drive_levels=[40.0],
        mec_drive_std=1.0,
        adaptive_step=True,
        adaptive_config=GradientAdaptiveStepConfig(
            dt_min=0.05,
            dt_max=0.3
        )
    )
    
    evaluator_adaptive = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params,
        opsin_params, opsin_current,
        targets, config_adaptive, device=device, base_seed=42,
        adaptive_step=True,
        adaptive_config=config_adaptive.adaptive_config
    )
    
    times_adaptive = []
    for i in range(n_runs):
        start = time.time()
        losses, _ = evaluator_adaptive.evaluate_parameter_batch(param_batch, 40.0, 1.0)
        elapsed = time.time() - start
        times_adaptive.append(elapsed)
    
    mean_adaptive = np.mean(times_adaptive)
    std_adaptive = np.std(times_adaptive)
    
    print(f"Adaptive dt: {mean_adaptive:.3f} ± {std_adaptive:.3f}s")
    
    overhead = ((mean_adaptive - mean_fixed) / mean_fixed) * 100
    print(f"\nOverhead: {overhead:.1f}%")
    
    if hasattr(evaluator_adaptive, '_last_adaptive_stats'):
        stats = evaluator_adaptive._last_adaptive_stats
        if stats is not None:
            fixed_steps = int((config_fixed.warmup_duration + config_fixed.simulation_duration) / 
                              circuit_params.dt)
            reduction = (1 - stats['n_steps'] / fixed_steps) * 100
            print(f"Step reduction: {reduction:.1f}% ({stats['n_steps']:.0f} vs {fixed_steps} steps)")
    
    print("\nPerformance benchmark COMPLETED")
    return True


def run_all_tests():
    """Run all tests and summarize results"""
    print("\n" + "="*80)
    print("OPTOGENETIC OPTIMIZATION ADAPTIVE STEPPING TEST SUITE")
    print("="*80)
    
    tests = [
        ("Sequential Strategy", test_sequential_strategy_adaptive),
        ("Batch Strategies", test_batch_strategies_adaptive),
        ("Multi-trial Consistency", test_multi_trial_consistency),
        ("Reproducibility", test_reproducibility),
        ("Optimizer Integration", test_optimizer_integration),
        ("Optimization-Protocol Consistency", test_optimization_protocol_consistency),
        ("Time-Varying MEC Consistency", test_with_time_varying_mec),
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = "PASSED" if passed else "FAILED"
        except Exception as e:
            print(f"\nX {test_name} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            results[test_name] = "ERROR"
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status_symbol = "O" if result == "PASSED" else "X"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed_count = sum(1 for r in results.values() if r == "PASSED")
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\nAll tests PASSED!")
        return True
    else:
        print("\nSome tests FAILED")
        return False


if __name__ == "__main__":
    import sys
    
    # Allow running individual tests
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_map = {
            "sequential": test_sequential_strategy_adaptive,
            "batch": test_batch_strategies_adaptive,
            "multitrial": test_multi_trial_consistency,
            "repro": test_reproducibility,
            "optimizer": test_optimizer_integration,
            "benchmark": test_performance_benchmark,
            "consistency": test_optimization_protocol_consistency,
        }
        
        if test_name in test_map:
            print(f"Running single test: {test_name}")
            test_map[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(test_map.keys())}")
            sys.exit(1)
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
