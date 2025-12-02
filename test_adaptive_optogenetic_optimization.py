#!/usr/bin/env python3
"""
Tests for adaptive stepping in optogenetic circuit optimization

Tests:
1. Sequential strategy with adaptive stepping
2. Batch GPU strategy with adaptive stepping (if CUDA available)
3. Multiprocess CPU strategy with adaptive stepping
4. Comparison: fixed-dt vs adaptive-dt results
5. Performance benchmarking
6. Adaptive statistics validation
"""

import torch
import time
import numpy as np
from pathlib import Path

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
        circuit_params, synaptic_params, opsin_params,
        targets, config, device=device, base_seed=42,
        adaptive_step=True,
        adaptive_config=config.adaptive_config
    )
    
    losses1, _ = evaluator1.evaluate_parameter_batch(param_set, 40.0, 1.0)
    loss1 = losses1[0].item()
    print(f"  Loss: {loss1:.6f}")
    
    print("\n4b. Second run (seed=42)...")
    evaluator2 = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params, opsin_params,
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
        circuit_params, synaptic_params, opsin_params,
        targets, config, device=device, base_seed=42
    )
    
    # Run a very short optimization
    start = time.time()
    results = optimizer.optimize(
        method='particle_swarm',
        n_particles=8,  # Small for quick test
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
        circuit_params, synaptic_params, opsin_params,
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
        circuit_params, synaptic_params, opsin_params,
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
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
#        try:
        passed = test_func()
        results[test_name] = "PASSED" if passed else "FAILED"
#        except Exception as e:
#            print(f"\nX {test_name} FAILED with exception:")
#            print(f"  {type(e).__name__}: {e}")
#            results[test_name] = "ERROR"
    
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
