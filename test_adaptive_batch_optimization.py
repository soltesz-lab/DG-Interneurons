#!/usr/bin/env python3
"""Test adaptive stepping in batch optimization with BatchCircuitEvaluator"""

import torch
import time
from DG_circuit_optimization import (
    CircuitParams, PerConnectionSynapticParams, OpsinParams,
    create_default_targets, OptimizationConfig, BatchCircuitEvaluator
)
from gradient_adaptive_stepper import GradientAdaptiveStepConfig

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on device: {device}\n")

circuit_params = CircuitParams()
synaptic_params = PerConnectionSynapticParams()
opsin_params = OpsinParams()
targets = create_default_targets()

# Create parameter batch
param_batch = [
    synaptic_params.connection_modulation,
    {k: v * 1.1 for k, v in synaptic_params.connection_modulation.items()},
    {k: v * 0.9 for k, v in synaptic_params.connection_modulation.items()},
]

print("=" * 60)
print("BatchCircuitEvaluator with Adaptive Stepping")
print("=" * 60)

# Test 1: Fixed dt
print("Fixed dt (original)")
config_fixed = OptimizationConfig(
    max_iterations=5,
    n_trials=1,
    simulation_duration=300,
    warmup_duration=50,
    adaptive_step=False,
    device=device
)

evaluator_fixed = BatchCircuitEvaluator(
    circuit_params, synaptic_params, opsin_params,
    targets, config_fixed, device=device,
    adaptive_step=False
)

start = time.time()
losses, firing_rates = evaluator_fixed.evaluate_parameter_batch(
    param_batch, 40.0, 1.0
)
time_fixed = time.time() - start

print(f"  Time: {time_fixed:.3f}s")
print(f"  Losses: {[f'{l:.4f}' for l in losses.cpu().numpy()]}")
print(f"  GC rates: {[f'{firing_rates['gc'][i]:.2f}' for i in range(len(param_batch))]}")

# Adaptive dt
print("\nAdaptive dt")
config_adaptive = OptimizationConfig(
    max_iterations=5,
    n_trials=1,
    simulation_duration=300,
    warmup_duration=50,
    adaptive_step=True,
    adaptive_config=GradientAdaptiveStepConfig(
        dt_min=0.05,
        dt_max=0.3,
        gradient_low=0.5,
        gradient_high=10.0
    ),
    device=device
)

evaluator_adaptive = BatchCircuitEvaluator(
    circuit_params, synaptic_params, opsin_params,
    targets, config_adaptive, device=device,
    adaptive_step=True,
    adaptive_config=config_adaptive.adaptive_config
)

start = time.time()
losses, firing_rates = evaluator_adaptive.evaluate_parameter_batch(
    param_batch, 40.0, 1.0
)
time_adaptive = time.time() - start

print(f"  Time: {time_adaptive:.3f}s")
print(f"  Losses: {[f'{l:.4f}' for l in losses.cpu().numpy()]}")
print(f"  GC rates: {[f'{firing_rates['gc'][i]:.2f}' for i in range(len(param_batch))]}")

if hasattr(evaluator_adaptive, '_last_adaptive_stats'):
    stats = evaluator_adaptive._last_adaptive_stats
    print(f"  Adaptive stats: {stats['n_steps']:.0f} steps, "
          f"avg dt: {stats['avg_dt']:.3f}ms")
    print(f"  dt range: [{stats['min_dt']:.3f}, {stats['max_dt']:.3f}] ms")

overhead = (time_adaptive / time_fixed - 1) * 100
print(f"\n  Overhead: {overhead:.1f}%")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)

