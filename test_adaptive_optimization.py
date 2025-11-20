#!/usr/bin/env python3
"""Test adaptive stepping in optimization"""

import torch
from DG_circuit_optimization import (
    CircuitOptimizer, CircuitParams, PerConnectionSynapticParams,
    OpsinParams, create_default_targets, OptimizationConfig
)
from gradient_adaptive_stepper import GradientAdaptiveStepConfig

# Setup
device = torch.device('cpu')
circuit_params = CircuitParams()
synaptic_params = PerConnectionSynapticParams()
opsin_params = OpsinParams()
targets = create_default_targets()

# Test Fixed dt
print("Fixed dt optimization")
config_fixed = OptimizationConfig(
    max_iterations=5,
    n_trials=1,
    simulation_duration=300,
    warmup_duration=50,
    adaptive_step=False
)

optimizer_fixed = CircuitOptimizer(
    circuit_params, synaptic_params, opsin_params,
    targets, config_fixed, device=device
)

import time
start = time.time()
# Just test evaluation, not full optimization
strategy = optimizer_fixed._select_strategy('gradient')
losses, _ = strategy.evaluate_batch(
    [synaptic_params.connection_modulation],
    40.0, 1.0, 1
)
time_fixed = time.time() - start
print(f"  Time: {time_fixed:.3f}s, Loss: {losses[0]:.6f}")

# Adaptive dt
print("\nAdaptive dt optimization")
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
    )
)

optimizer_adaptive = CircuitOptimizer(
    circuit_params, synaptic_params, opsin_params,
    targets, config_adaptive, device=device
)

start = time.time()
strategy = optimizer_adaptive._select_strategy('gradient')
losses, _ = strategy.evaluate_batch(
    [synaptic_params.connection_modulation],
    40.0, 1.0, 1
)
time_adaptive = time.time() - start
print(f"  Time: {time_adaptive:.3f}s, Loss: {losses[0]:.6f}")

# Check adaptive stats
if hasattr(strategy, '_last_adaptive_stats'):
    stats = strategy._last_adaptive_stats
    print(f"  Adaptive stats: {stats['n_steps_mean']:.0f} steps, "
          f"avg dt: {stats['avg_dt_mean']:.3f}ms")

print(f"\nOverhead: {(time_adaptive/time_fixed - 1)*100:.1f}%")
print("Tests completed successfully!")
