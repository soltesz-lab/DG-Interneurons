#!/usr/bin/env python3
"""
Gradient-Driven Adaptive Time Stepping for Neural Circuit Simulations

Automatically adjusts time step based on population activity gradients,
providing efficient simulation while maintaining accuracy.
"""

import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class GradientAdaptiveStepConfig:
    """Configuration for gradient-driven adaptive time stepping"""
    
    # Time step bounds (ms)
    dt_min: float = 0.05              # Fine resolution (2.5% of τ_AMPA = 2 ms)
    dt_max: float = 0.3               # Conservative coarse (15% of τ_AMPA)
    
    # Gradient thresholds (Hz/ms)
    gradient_low: float = 0.5         # Below this: increase dt
    gradient_high: float = 10.0       # Above this: decrease dt
    
    # Exponential moving average (EMA) smoothing
    gradient_alpha: float = 0.3       # Lower = more smoothing (0-1)
    
    # Rate limiting for stability
    max_dt_change_factor: float = 1.5  # Max dt increase/decrease per step
    
    # Option to exclude populations from gradient computation
    excluded_populations: List[str] = field(default_factory=list)


class GradientAdaptiveStepper:
    """
    Implements gradient-driven adaptive time stepping for neural circuits.
    
    Automatically detects rapid dynamics (events) via activity gradients and
    adjusts time step accordingly, providing efficient simulation with minimal
    manual configuration.
    
    Example:
        config = GradientAdaptiveStepConfig(dt_min=0.05, dt_max=0.3)
        stepper = GradientAdaptiveStepper(config)
        
        while current_time < total_duration:
            # Compute dt for this step
            if step > 0:
                dt = stepper.compute_dt(activity_current, activity_prev, dt_prev)
            else:
                dt = config.dt_max
            
            # Run circuit forward
            activity_current = circuit.forward(...)
            
            # Update for next iteration
            activity_prev = activity_current
            current_time += dt
    """
    
    def __init__(self, config: GradientAdaptiveStepConfig):
        """
        Initialize adaptive stepper
        
        Args:
            config: Configuration for adaptive stepping
        """
        self.config = config
        self.grad_smooth = 0.0  # Smoothed gradient magnitude
        self.dt_current = config.dt_max  # Start with coarse stepping
    
    def compute_dt(self, 
                   activity_current: Dict[str, Tensor],
                   activity_prev: Dict[str, Tensor],
                   dt_prev: float) -> float:
        """
        Determine next time step based on activity gradient
        
        Args:
            activity_current: Current population activities {pop: Tensor}
            activity_prev: Previous population activities {pop: Tensor}
            dt_prev: Previous time step used (ms)
            
        Returns:
            Optimal time step for next iteration (ms)
        """
        # Compute instantaneous gradient
        grad_instant = self._compute_gradient_magnitude(
            activity_current, activity_prev, dt_prev
        )
        
        # Apply exponential moving average for stability
        self.grad_smooth = (
            self.config.gradient_alpha * grad_instant +
            (1 - self.config.gradient_alpha) * self.grad_smooth
        )
        
        # Map gradient magnitude to dt
        dt_new = self._gradient_to_dt(self.grad_smooth)
        
        # Rate limit dt changes to prevent oscillations
        max_increase = self.dt_current * self.config.max_dt_change_factor
        max_decrease = self.dt_current / self.config.max_dt_change_factor
        dt_new = np.clip(dt_new, max_decrease, max_increase)
        
        # Apply absolute bounds
        dt_new = np.clip(dt_new, self.config.dt_min, self.config.dt_max)
        
        # Update current dt for next iteration
        self.dt_current = dt_new
        
        return dt_new
    
    def _compute_gradient_magnitude(self,
                                    activity_current: Dict[str, Tensor],
                                    activity_prev: Dict[str, Tensor],
                                    dt: float) -> float:
        """
        Compute gradient magnitude across all populations
        
        Uses simple average across all included populations to avoid
        population-specific weighting assumptions.
        
        Args:
            activity_current: Current activities
            activity_prev: Previous activities  
            dt: Time step between them
            
        Returns:
            Gradient magnitude (Hz/ms)
        """
        gradients = []
        
        for pop_name in activity_current.keys():
            # Skip excluded populations
            if pop_name in self.config.excluded_populations:
                continue
            
            # Skip if not in previous activities
            if pop_name not in activity_prev:
                continue
            
            # Compute population-averaged activity change
            delta = torch.mean(activity_current[pop_name] - activity_prev[pop_name])
            grad = abs(delta.item()) / dt
            
            gradients.append(grad)
        
        # Return average gradient magnitude across populations
        return np.mean(gradients) if gradients else 0.0
    
    def _gradient_to_dt(self, grad_magnitude: float) -> float:
        """
        Map gradient magnitude to time step size
        
        Uses logarithmic interpolation for smooth transition between
        fine and coarse time steps.
        
        Args:
            grad_magnitude: Smoothed gradient magnitude (Hz/ms)
            
        Returns:
            Appropriate time step (ms)
        """
        if grad_magnitude > self.config.gradient_high:
            # High gradient: use fine resolution
            return self.config.dt_min
        elif grad_magnitude < self.config.gradient_low:
            # Low gradient: use coarse resolution
            return self.config.dt_max
        else:
            # Intermediate: logarithmic interpolation
            log_dt = np.interp(
                np.log10(grad_magnitude),
                [np.log10(self.config.gradient_low), 
                 np.log10(self.config.gradient_high)],
                [np.log10(self.config.dt_max), 
                 np.log10(self.config.dt_min)]
            )
            return 10 ** log_dt
    
    def reset(self):
        """Reset stepper state (for new simulation)"""
        self.grad_smooth = 0.0
        self.dt_current = self.config.dt_max


def update_circuit_with_adaptive_dt(circuit,
                                    direct_activation: Dict[str, Tensor],
                                    external_drive: Dict[str, Tensor],
                                    dt: float) -> Dict[str, Tensor]:
    """
    Update circuit with potentially changed dt
    
    Handles recomputation of synaptic decay factors when dt changes.
    
    Args:
        circuit: DentateCircuit instance
        direct_activation: Direct activation dict
        external_drive: External drive dict
        dt: Current time step
        
    Returns:
        Updated activities dict
    """

    return circuit(direct_activation, external_drive, dt=dt)


def interpolate_to_fixed_grid(activity_current: Dict[str, Tensor],
                              current_time: float,
                              dt: float,
                              time_grid: Tensor,
                              storage: Dict[str, Tensor],
                              storage_idx: List[int]) -> None:
    """
    Interpolate variable-dt activity to fixed time grid for storage
    
    Uses linear interpolation to map activity at current_time to the
    fixed storage grid. Modifies storage in-place.
    
    Args:
        activity_current: Current activities {pop: Tensor[n_neurons]}
        current_time: Current simulation time (ms)
        dt: Time step taken (ms)
        time_grid: Fixed time grid for storage (ms)
        storage: Storage dict {pop: Tensor[n_neurons, n_steps]}
        storage_idx: List with single element [idx] tracking storage position
    """
    # Find time grid indices that fall within this simulation step
    start_time = current_time - dt
    end_time = current_time
    
    # Get indices of grid points in this interval
    mask = (time_grid >= start_time) & (time_grid <= end_time)
    grid_indices = torch.where(mask)[0]
    
    if len(grid_indices) == 0:
        return
    
    # For each grid point, interpolate linearly
    # Assume activity changed linearly from start to end of step
    for grid_idx in grid_indices:
        grid_time = time_grid[grid_idx].item()
        
        # Linear interpolation weight (0 at start_time, 1 at end_time)
        if dt > 1e-9:
            weight = (grid_time - start_time) / dt
        else:
            weight = 1.0
        
        # Store interpolated activity
        # (Here we use simple assumption that activity is constant over step,
        #  or could use linear interpolation if we stored previous activity)
        for pop_name, activity in activity_current.items():
            if pop_name in storage:
                storage[pop_name][:, grid_idx] = activity


class AdaptiveSimulationState:
    """
    Tracks state for adaptive simulation
    
    Keeps history for gradient computation and diagnostic information.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.current_time = 0.0
        self.step_index = 0
        
        # Previous activity for gradient computation
        self.activity_prev: Optional[Dict[str, Tensor]] = None
        
        # History for analysis
        self.dt_history: List[float] = []
        self.time_history: List[float] = [0.0]
        self.gradient_history: List[float] = []
    
    def update(self, 
               activity_current: Dict[str, Tensor],
               dt: float,
               gradient: float):
        """Update state after a simulation step"""
        self.activity_prev = {
            pop: activity.clone() for pop, activity in activity_current.items()
        }
        self.current_time += dt
        self.step_index += 1
        
        self.dt_history.append(dt)
        self.time_history.append(self.current_time)
        self.gradient_history.append(gradient)
    
    def get_statistics(self) -> Dict:
        """Get statistics about adaptive stepping"""
        return {
            'n_steps': self.step_index,
            'avg_dt': np.mean(self.dt_history) if self.dt_history else 0.0,
            'min_dt': np.min(self.dt_history) if self.dt_history else 0.0,
            'max_dt': np.max(self.dt_history) if self.dt_history else 0.0,
            'dt_history': np.array(self.dt_history),
            'time_history': np.array(self.time_history),
            'gradient_history': np.array(self.gradient_history),
        }


def initialize_activity_storage(circuit,
                                time_grid: Tensor) -> Dict[str, Tensor]:
    """
    Initialize storage tensors for fixed-grid activity traces
    
    Args:
        circuit: DentateCircuit instance
        time_grid: Fixed time grid tensor
        
    Returns:
        Storage dict {pop: Tensor[n_neurons, n_steps]}
    """
    n_steps = len(time_grid)
    device = time_grid.device
    
    storage = {
        'gc': torch.zeros(circuit.circuit_params.n_gc, n_steps, device=device),
        'mc': torch.zeros(circuit.circuit_params.n_mc, n_steps, device=device),
        'pv': torch.zeros(circuit.circuit_params.n_pv, n_steps, device=device),
        'sst': torch.zeros(circuit.circuit_params.n_sst, n_steps, device=device),
        'mec': torch.zeros(circuit.circuit_params.n_mec, n_steps, device=device),
    }
    
    return storage


# Example integration function for use in simulate_single_trial
def adaptive_simulation_loop(circuit,
                             total_duration: float,
                             config: GradientAdaptiveStepConfig,
                             get_activations_fn) -> Dict:
    """
    Test for integrating adaptive stepping into simulation loop
    
    Args:
        circuit: DentateCircuit instance
        total_duration: Total simulation time (ms)
        config: Adaptive stepping configuration
        get_activations_fn: Function that returns (direct_activation, external_drive)
                           given current_time
    
    Returns:
        Dict with simulation results and adaptive statistics
    """
    # Initialize
    stepper = GradientAdaptiveStepper(config)
    state = AdaptiveSimulationState(circuit.device)
    
    # Storage on fixed grid
    dt = 0.1
    time_grid = torch.arange(0, total_duration, dt, device=circuit.device)
    activity_storage = initialize_activity_storage(circuit, time_grid)
    storage_idx = [0]
    
    # Main simulation loop
    while state.current_time < total_duration:
        # Compute dt for this step
        if state.step_index > 0:
            dt = stepper.compute_dt(
                activity_current,
                state.activity_prev,
                state.dt_history[-1]
            )
        else:
            dt = config.dt_max  # Start with coarse stepping
        
        # Ensure we don't overshoot
        if state.current_time + dt > total_duration:
            dt = total_duration - state.current_time
        
        # Get inputs for current time
        direct_activation, external_drive = get_activations_fn(state.current_time)
        
        # Update circuit with adaptive dt
        activity_current = update_circuit_with_adaptive_dt(
            circuit, direct_activation, external_drive, dt
        )
        
        # Interpolate to fixed grid for storage
        interpolate_to_fixed_grid(
            activity_current, state.current_time + dt, dt,
            time_grid, activity_storage, storage_idx
        )
        
        # Update state
        state.update(activity_current, dt, stepper.grad_smooth)
    
    # Return results
    return {
        'time': time_grid,
        'activity_trace': activity_storage,
        'adaptive_stats': state.get_statistics(),
    }
