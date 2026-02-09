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
from gradient_adaptive_stepper import (
    GradientAdaptiveStepConfig,
    GradientAdaptiveStepper,
    AdaptiveSimulationState,
    update_circuit_with_adaptive_dt,
    interpolate_to_fixed_grid,
    initialize_activity_storage
)
from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit,
    CircuitParams,
    PerConnectionSynapticParams,
    OpsinParams
)


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


def _interpolate_activity_to_grid(activity_current: Dict[str, torch.Tensor],
                                  activity_prev: Dict[str, torch.Tensor],
                                  time_prev: float,
                                  time_current: float,
                                  time_grid: torch.Tensor,
                                  storage: Dict[str, torch.Tensor]) -> None:
    """
    Interpolate activity linearly between time points to fixed grid
    
    For each grid point between time_prev and time_current, linearly
    interpolate the activity.
    
    Args:
        activity_current: Current activities
        activity_prev: Previous activities
        time_prev: Previous time point (ms)
        time_current: Current time point (ms)
        time_grid: Fixed time grid
        storage: Storage dict to update in-place
    """
    # Find grid indices in this interval
    mask = (time_grid > time_prev) & (time_grid <= time_current)
    grid_indices = torch.where(mask)[0]
    
    if len(grid_indices) == 0:
        return
    
    dt_step = time_current - time_prev
    
    for grid_idx in grid_indices:
        grid_time = time_grid[grid_idx].item()
        
        # Linear interpolation weight (0 at time_prev, 1 at time_current)
        if dt_step > 1e-9:
            alpha = (grid_time - time_prev) / dt_step
        else:
            alpha = 1.0
        
        # Linearly interpolate each population
        for pop_name in activity_current.keys():
            if pop_name in storage and pop_name in activity_prev:
                interpolated = (1 - alpha) * activity_prev[pop_name] + alpha * activity_current[pop_name]
                storage[pop_name][:, grid_idx] = interpolated

                
class OpsinExpression:
    """Handle heterogeneous opsin expression"""
    
    def __init__(self, params: OpsinParams, n_cells: int, device: Optional[torch.device] = None):
        self.params = params
        self.n_cells = n_cells
        self.device = device if device is not None else get_default_device()
        self.expression_levels = self._generate_expression()
    
    def _generate_expression(self) -> Tensor:
        """Generate log-normal distribution of expression levels"""
        # Log-normal distribution
        log_mean = torch.log(torch.tensor(self.params.expression_mean, device=self.device))
        log_std = torch.tensor(self.params.expression_std, device=self.device)

        normal_samples = torch.normal(log_mean, log_std, size=(self.n_cells,), device=self.device)
        expression = torch.exp(normal_samples)

        sorted_indices = torch.argsort(expression, descending=True)

        # Calculate number of failed cells
        n_failed = int(round(self.params.failure_rate * self.n_cells))

        # Set expression to zero for cells with transduction failure
        if n_failed > 0:
            failed_cell_indices = sorted_indices[-n_failed:]
            expression[failed_cell_indices] = 0.0

        return expression

    def calculate_activation(self, positions: Tensor, light_intensity: float) -> Tensor:
        """Calculate light-induced activation probability"""
        # Distance from fiber tip (assumed at mid-point)
        fiber_position = 0.5
        distances = torch.abs(torch.norm(positions, dim=1) - fiber_position)
        # Light attenuation with tissue depth
        attenuated_intensity = light_intensity * torch.exp(-distances / self.params.light_decay)

        # Hill equation for opsin activation (avoid division by zero)
        safe_expression = torch.clamp(self.expression_levels, min=1e-6)
        half_sat_scaled = self.params.half_sat / safe_expression
        
        numerator = attenuated_intensity ** self.params.hill_coeff
        denominator = numerator + half_sat_scaled ** self.params.hill_coeff
        activation = numerator / denominator
        
        # No activation for non-expressing cells
        activation = torch.where(self.expression_levels == 0, 
                                 torch.tensor(0.0, device=self.device), activation)
        
        return activation


def get_mec_rotation_pattern(n_mec: int,
                             trial_index: int,
                             base_currents,  # Can be scalar or array
                             n_groups: int = 3,
                             device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Get spatial pattern that rotates across trials
    
    Divides MEC population into groups and rotates the group assignment across trials.
    
    Args:
        n_mec: Number of MEC neurons
        trial_index: Current trial index (0, 1, 2, ...)
        base_currents: Base current values - can be:
                      - scalar (float/int): uniform current, will be broadcast to array
                      - array (np.ndarray): spatial pattern to rotate
        n_groups: Number of groups to divide MEC population into
        device: Device to create tensor on
        
    Returns:
        group_tensor: [n_mec] tensor of rates
    """
    if device is None:
        device = get_default_device()
    
    # Handle scalar or array input
    if isinstance(base_currents, (int, float)):
        # Scalar: create uniform array
        base_currents_array = np.full(n_mec, base_currents, dtype=np.float32)
    elif isinstance(base_currents, np.ndarray):
        # Already an array
        base_currents_array = base_currents.astype(np.float32)
    elif isinstance(base_currents, torch.Tensor):
        # Convert tensor to numpy
        base_currents_array = base_currents.cpu().numpy().astype(np.float32)
    else:
        raise TypeError(f"base_currents must be scalar, numpy array, or torch tensor, got {type(base_currents)}")
    
    # Ensure correct shape
    if base_currents_array.shape != (n_mec,):
        if base_currents_array.size == n_mec:
            base_currents_array = base_currents_array.reshape(n_mec)
        else:
            raise ValueError(f"base_currents array has wrong size: {base_currents_array.shape}, expected ({n_mec},)")
    
    # Divide neurons into groups
    group_size = n_mec // n_groups
    remainder = n_mec % n_groups
    
    # Apply circular rotation based on trial index
    rotation_amount = ((trial_index + 1) * group_size) % n_mec
    rotated_array = np.roll(base_currents_array, rotation_amount)

    return torch.from_numpy(rotated_array).float().to(device)


def generate_time_varying_mec_pattern(n_mec: int,
                                      duration: float,
                                      dt: float,
                                      base_current: float,
                                      trial_index: int,
                                      pattern_type: str = 'oscillatory',
                                      theta_freq: float = 5.0,
                                      theta_amplitude: float = 0.3,
                                      gamma_freq: float = 20.0,
                                      gamma_amplitude: float = 0.15,
                                      gamma_coupling_strength: float = 0.8,
                                      gamma_preferred_phase: float = 0.0,
                                      drift_timescale: float = 200.0,
                                      drift_amplitude: float = 0.1,
                                      rotation_groups: int = 3,
                                      spatial_noise_std: float = 0.2,
                                      temporal_noise_std: float = 0.05,
                                      device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate time-varying MEC firing pattern with trial rotation
    
    Creates temporally-structured input patterns that:
    1. Vary over time to average out spatial biases within trials
    2. Rotate spatial structure across trials for ensemble balance
    
    Args:
        n_mec: Number of MEC neurons
        duration: Total duration in ms
        dt: Time step in ms
        base_current: Base current level (pA)
        trial_index: Current trial index for rotation
        pattern_type: 'oscillatory', 'drift', 'noisy', or 'constant'
        theta_freq: Theta oscillation frequency (Hz)
        theta_amplitude: Relative theta modulation depth (0-1)
        gamma_freq: Gamma oscillation frequency (Hz)  
        gamma_amplitude: Relative gamma modulation depth (0-1)
        gamma_coupling_strength: Theta-gamma coupling strength (0=independent, 1=fully coupled)
        gamma_preferred_phase: Preferred theta phase for gamma peak (radians)
        drift_timescale: Correlation time for drift pattern (ms)
        drift_amplitude: Relative drift amplitude (0-1)
        rotation_groups: Number of groups for trial rotation
        spatial_noise_std: Spatial heterogeneity (fraction of base_current, for 'noisy' pattern)
        temporal_noise_std: Temporal noise level (fraction of base_current, for 'noisy' pattern)
        device: Device to create tensors on
        
    Returns:
        mec_pattern: [n_mec, n_timesteps] tensor of MEC currents (pA)
    """
    if device is None:
        device = get_default_device()
    
    n_steps = int(duration / dt)
    time_vec = torch.arange(n_steps, device=device) * dt  # in ms

    base_currents_array = np.full(n_mec, base_current, dtype=np.float32)

    # Get spatial rotation pattern for this trial
    base_rates = get_mec_rotation_pattern(
        n_mec, trial_index, base_currents_array, rotation_groups, device
    )  # [n_mec]
    
    if pattern_type == 'constant':
        # No temporal variation, just spatial rotation
        mec_pattern = base_rates.unsqueeze(1).expand(-1, n_steps)
        
    elif pattern_type == 'oscillatory':
        # Oscillatory modulation with theta and gamma components
        
        # Generate random phase offsets for each neuron
        theta_phases = torch.rand(n_mec, device=device) * 2 * np.pi
        gamma_phases = torch.rand(n_mec, device=device) * 2 * np.pi

        # Time in seconds for frequency calculation
        time_sec = time_vec / 1000.0  # [n_steps]
        
        # Theta modulation: [n_mec, n_steps]
        theta_component = theta_amplitude * torch.sin(
            2 * np.pi * theta_freq * time_sec.unsqueeze(0) + theta_phases.unsqueeze(1)
        )

        # Theta-modulated gamma envelope
        theta_phase_matrix = 2 * np.pi * theta_freq * time_sec.unsqueeze(0) + theta_phases.unsqueeze(1)
        gamma_envelope = 1.0 + gamma_coupling_strength * torch.sin(
            theta_phase_matrix + gamma_preferred_phase
        )

        # Gamma oscillation with theta-modulated amplitude: [n_mec, n_steps]
        gamma_oscillation = torch.sin(
            2 * np.pi * gamma_freq * time_sec.unsqueeze(0) + gamma_phases.unsqueeze(1)
        )
        gamma_component = gamma_envelope * gamma_amplitude * gamma_oscillation
        
        # Combined pattern: base * (1 + theta + gamma)
        modulation = 1.0 + theta_component + gamma_component
        mec_pattern = base_rates.unsqueeze(1) * modulation
        
    elif pattern_type == 'drift':
        # Ornstein-Uhlenbeck process for slow drift
        
        tau = drift_timescale  # correlation time in ms
        sigma = drift_amplitude * base_current  # noise amplitude
        
        # Initialize with random starting points
        mec_pattern = torch.zeros(n_mec, n_steps, device=device)
        mec_pattern[:, 0] = base_rates
        
        # Generate OU process for each neuron
        sqrt_dt = np.sqrt(dt)
        for t in range(1, n_steps):
            drift_term = -(mec_pattern[:, t-1] - base_rates) / tau * dt
            noise_term = sigma * sqrt_dt * torch.randn(n_mec, device=device)
            mec_pattern[:, t] = mec_pattern[:, t-1] + drift_term + noise_term
            
            # Clip to reasonable range (prevent negative currents)
            mec_pattern[:, t] = torch.clamp(mec_pattern[:, t], 
                                           min=base_current * 0.2,
                                           max=base_current * 2.0)
    
    elif pattern_type == 'noisy':
        # Spatially heterogeneous baseline + temporal noise
        
        # Generate spatially heterogeneous baseline currents
        spatial_std = spatial_noise_std * base_current
        baseline_currents = np.random.normal(base_current, spatial_std, n_mec)
        baseline_currents = np.clip(baseline_currents, 0.0, None)  # No negative currents
        
        # Apply trial rotation to heterogeneous baselines
        base_rates_hetero = get_mec_rotation_pattern(
            n_mec, trial_index, baseline_currents, rotation_groups, device
        )  # [n_mec]
        
        # Add temporal noise at each time step
        temporal_std = temporal_noise_std * base_current
        temporal_noise = torch.randn(n_mec, n_steps, device=device) * temporal_std
        
        # Combine: heterogeneous baseline + temporal noise
        mec_pattern = base_rates_hetero.unsqueeze(1) + temporal_noise
        
        # Ensure no negative currents
        mec_pattern = torch.clamp(mec_pattern, min=0.0)
    
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}. Use 'oscillatory', 'drift', 'noisy', or 'constant'")
    
    # Ensure no negative currents (final check for all pattern types)
    mec_pattern = torch.clamp(mec_pattern, min=0.0)
    
    return mec_pattern


@dataclass
class CurrentRecordingConfig:
    """Configuration for synaptic current recording"""
    enabled: bool = False
    record_by_source: bool = True
    record_by_type: bool = True
    populations: List[str] = field(default_factory=lambda: ['gc', 'mc', 'pv', 'sst'])
    downsample_factor: int = 10  # Record every Nth timestep to save memory


class SynapticCurrentRecorder:
    """Records synaptic currents during simulation"""
    
    def __init__(self, config: CurrentRecordingConfig, 
                 circuit_params, time_grid: torch.Tensor,
                 device: Optional[torch.device] = None):
        """
        Initialize current recorder
        
        Args:
            config: Recording configuration
            circuit_params: Circuit parameters (for population sizes)
            time_grid: Time points for recording [n_timesteps]
            device: Device for tensors
        """
        self.config = config
        self.circuit_params = circuit_params
        self.device = device if device is not None else get_default_device()
        
        # Downsample time grid
        self.recording_indices = torch.arange(
            0, len(time_grid), config.downsample_factor, device=self.device
        )
        self.time_grid = time_grid[self.recording_indices]
        self.n_recorded_steps = len(self.recording_indices)
        
        # Initialize storage
        self.currents_by_source = {}
        self.currents_by_type = {}
        self.current_step = 0
        
        if config.record_by_source:
            self._init_source_storage()
        
        if config.record_by_type:
            self._init_type_storage()
    
    def _init_source_storage(self):
        """Initialize storage for currents by source"""
        for pop in self.config.populations:
            n_cells = getattr(self.circuit_params, f'n_{pop}')
            self.currents_by_source[pop] = {}
            
            # Will populate with source names as we encounter them
    
    def _init_type_storage(self):
        """Initialize storage for currents by receptor type"""
        for pop in self.config.populations:
            n_cells = getattr(self.circuit_params, f'n_{pop}')
            
            self.currents_by_type[pop] = {
                'ampa': torch.zeros(n_cells, self.n_recorded_steps, device=self.device),
                'gaba': torch.zeros(n_cells, self.n_recorded_steps, device=self.device),
                'nmda': torch.zeros(n_cells, self.n_recorded_steps, device=self.device),
                'total_exc': torch.zeros(n_cells, self.n_recorded_steps, device=self.device),
                'total_inh': torch.zeros(n_cells, self.n_recorded_steps, device=self.device),
                'net': torch.zeros(n_cells, self.n_recorded_steps, device=self.device)
            }
    
    def record_currents(self, circuit, step_index: int):
        """
        Record currents at current timestep
        
        Args:
            circuit: DentateCircuit instance
            step_index: Current simulation step (on full time grid)
        """
        # Check if this step should be recorded
        if step_index not in self.recording_indices:
            return
        
        # Get recording index
        rec_idx = self.current_step
        
        # Get all currents
        all_currents = circuit.get_all_synaptic_currents()
        
        # Store currents
        for pop in self.config.populations:
            if pop not in all_currents:
                continue
            
            pop_currents = all_currents[pop]
            
            # Record by source
            if self.config.record_by_source:
                for source, source_currents in pop_currents['by_source'].items():
                    # Initialize storage for this source if needed
                    if source not in self.currents_by_source[pop]:
                        n_cells = getattr(self.circuit_params, f'n_{pop}')
                        self.currents_by_source[pop][source] = {
                            'ampa': torch.zeros(n_cells, self.n_recorded_steps, device=self.device),
                            'gaba': torch.zeros(n_cells, self.n_recorded_steps, device=self.device),
                            'nmda': torch.zeros(n_cells, self.n_recorded_steps, device=self.device)
                        }
                    
                    # Store currents
                    for receptor_type in ['ampa', 'gaba', 'nmda']:
                        self.currents_by_source[pop][source][receptor_type][:, rec_idx] = \
                            source_currents[receptor_type]
            
            # Record by type
            if self.config.record_by_type:
                for current_type in ['ampa', 'gaba', 'nmda', 'total_exc', 'total_inh', 'net']:
                    self.currents_by_type[pop][current_type][:, rec_idx] = \
                        pop_currents['by_type'][current_type]
        
        self.current_step += 1
    
    def get_results(self) -> Dict:
        """
        Get recorded currents
        
        Returns:
            Dict with 'time', 'by_source', 'by_type' fields
        """
        return {
            'time': self.time_grid.cpu(),
            'by_source': {
                pop: {
                    source: {k: v.cpu() for k, v in currents.items()}
                    for source, currents in sources.items()
                }
                for pop, sources in self.currents_by_source.items()
            } if self.config.record_by_source else {},
            'by_type': {
                pop: {k: v.cpu() for k, v in currents.items()}
                for pop, currents in self.currents_by_type.items()
            } if self.config.record_by_type else {}
        }


def analyze_currents_by_period(recorded_currents: Dict,
                               baseline_start: float,
                               baseline_end: float,
                               stim_start: float,
                               stim_end: float) -> Dict:
    """
    Analyze synaptic currents during baseline and stimulation periods
    
    Args:
        recorded_currents: Output from SynapticCurrentRecorder.get_results()
        baseline_start: Start of baseline period (ms)
        baseline_end: End of baseline period (ms)
        stim_start: Start of stimulation period (ms)
        stim_end: End of stimulation period (ms)
    
    Returns:
        Dict with statistics for baseline and stimulation periods:
        {
            'baseline': {
                'gc': {
                    'by_source': {'mec': {'ampa_mean': ..., 'ampa_std': ..., ...}, ...},
                    'by_type': {'ampa_mean': ..., 'total_exc_mean': ..., ...}
                },
                ...
            },
            'stimulation': {...},
            'change': {...}  # stim - baseline
        }
    """
    time = recorded_currents['time']
    
    # Create masks for periods
    baseline_mask = (time >= baseline_start) & (time <= baseline_end)
    stim_mask = (time >= stim_start) & (time <= stim_end)
    
    results = {
        'baseline': {},
        'stimulation': {},
        'change': {},
        'time_info': {
            'baseline_start': baseline_start,
            'baseline_end': baseline_end,
            'stim_start': stim_start,
            'stim_end': stim_end,
            'baseline_n_points': int(torch.sum(baseline_mask)),
            'stim_n_points': int(torch.sum(stim_mask))
        }
    }
    
    # Analyze by source
    if recorded_currents['by_source']:
        for pop, sources in recorded_currents['by_source'].items():
            results['baseline'][pop] = {'by_source': {}}
            results['stimulation'][pop] = {'by_source': {}}
            results['change'][pop] = {'by_source': {}}
            
            for source, currents in sources.items():
                baseline_stats = {}
                stim_stats = {}
                change_stats = {}
                
                for receptor_type, current_trace in currents.items():
                    # [n_cells, n_timesteps]
                    baseline_values = current_trace[:, baseline_mask]
                    stim_values = current_trace[:, stim_mask]
                    
                    # Compute mean across time for each cell
                    baseline_mean_per_cell = torch.mean(baseline_values, dim=1)
                    stim_mean_per_cell = torch.mean(stim_values, dim=1)
                    change_per_cell = stim_mean_per_cell - baseline_mean_per_cell
                    
                    # Statistics across cells
                    baseline_stats[f'{receptor_type}_mean'] = float(torch.mean(baseline_mean_per_cell))
                    baseline_stats[f'{receptor_type}_std'] = float(torch.std(baseline_mean_per_cell))
                    baseline_stats[f'{receptor_type}_sem'] = float(torch.std(baseline_mean_per_cell) / 
                                                                   np.sqrt(len(baseline_mean_per_cell)))
                    
                    stim_stats[f'{receptor_type}_mean'] = float(torch.mean(stim_mean_per_cell))
                    stim_stats[f'{receptor_type}_std'] = float(torch.std(stim_mean_per_cell))
                    stim_stats[f'{receptor_type}_sem'] = float(torch.std(stim_mean_per_cell) / 
                                                               np.sqrt(len(stim_mean_per_cell)))
                    
                    change_stats[f'{receptor_type}_mean'] = float(torch.mean(change_per_cell))
                    change_stats[f'{receptor_type}_std'] = float(torch.std(change_per_cell))
                    change_stats[f'{receptor_type}_sem'] = float(torch.std(change_per_cell) / 
                                                                 np.sqrt(len(change_per_cell)))
                
                results['baseline'][pop]['by_source'][source] = baseline_stats
                results['stimulation'][pop]['by_source'][source] = stim_stats
                results['change'][pop]['by_source'][source] = change_stats
    
    # Analyze by type
    if recorded_currents['by_type']:
        for pop, currents in recorded_currents['by_type'].items():
            if pop not in results['baseline']:
                results['baseline'][pop] = {}
                results['stimulation'][pop] = {}
                results['change'][pop] = {}
            
            results['baseline'][pop]['by_type'] = {}
            results['stimulation'][pop]['by_type'] = {}
            results['change'][pop]['by_type'] = {}
            
            for current_type, current_trace in currents.items():
                baseline_values = current_trace[:, baseline_mask]
                stim_values = current_trace[:, stim_mask]
                
                baseline_mean_per_cell = torch.mean(baseline_values, dim=1)
                stim_mean_per_cell = torch.mean(stim_values, dim=1)
                change_per_cell = stim_mean_per_cell - baseline_mean_per_cell
                
                results['baseline'][pop]['by_type'][current_type] = {
                    'mean': float(torch.mean(baseline_mean_per_cell)),
                    'std': float(torch.std(baseline_mean_per_cell)),
                    'sem': float(torch.std(baseline_mean_per_cell) / np.sqrt(len(baseline_mean_per_cell)))
                }
                
                results['stimulation'][pop]['by_type'][current_type] = {
                    'mean': float(torch.mean(stim_mean_per_cell)),
                    'std': float(torch.std(stim_mean_per_cell)),
                    'sem': float(torch.std(stim_mean_per_cell) / np.sqrt(len(stim_mean_per_cell)))
                }
                
                results['change'][pop]['by_type'][current_type] = {
                    'mean': float(torch.mean(change_per_cell)),
                    'std': float(torch.std(change_per_cell)),
                    'sem': float(torch.std(change_per_cell) / np.sqrt(len(change_per_cell)))
                }
    
    return results


def aggregate_trial_results(trial_results: List[Dict], n_trials: int) -> Dict:
    """Aggregate results across multiple trials"""

    # Time is the same across trials
    time = trial_results[0]['time']

    # Stack activity traces: [n_trials, n_neurons, n_steps]
    activity_traces_all = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst', 'mec']}
    opsin_expressions_all = []

    for trial_result in trial_results:
        for pop in activity_traces_all:
            activity_traces_all[pop].append(trial_result['activity_trace'][pop])
        opsin_expressions_all.append(trial_result['opsin_expression'])

    # Calculate mean and std across trials
    activity_trace_mean = {}
    activity_trace_std = {}

    for pop in activity_traces_all:
        # Stack: [n_trials, n_neurons, n_steps]
        stacked = torch.stack(activity_traces_all[pop], dim=0)
        activity_trace_mean[pop] = torch.mean(stacked, dim=0)  # [n_neurons, n_steps]
        if n_trials == 1:
            # For single trial, std is 0 (no variance across trials)
            activity_trace_std[pop] = torch.zeros_like(activity_trace_mean[pop])
        else:
            # For multiple trials, use unbiased=False for numerical stability
            # (unbiased=True would divide by n-1, causing issues with small n)
            activity_trace_std[pop] = torch.std(stacked, dim=0, unbiased=False)


    # Opsin expression (may vary slightly per trial due to stochastic generation)
    opsin_expression_stacked = torch.stack(opsin_expressions_all, dim=0)
    opsin_expression_mean = torch.mean(opsin_expression_stacked, dim=0)
    if n_trials == 1:
        opsin_expression_std = torch.zeros_like(opsin_expression_mean)
    else:
        opsin_expression_std = torch.std(opsin_expression_stacked, dim=0, unbiased=False)

    # Aggregate adaptive stats if present
    adaptive_stats_aggregated = None
    if 'adaptive_stats' in trial_results[0]:
        adaptive_stats_aggregated = aggregate_adaptive_stats(trial_results)

    aggregated = {
        'time': time,
        'activity_trace_mean': activity_trace_mean,
        'activity_trace_std': activity_trace_std,
        'opsin_expression_mean': opsin_expression_mean,
        'opsin_expression_std': opsin_expression_std,
        'n_trials': n_trials,
        'trial_results': trial_results,  # Keep individual results for detailed analysis
        # Include layout and connectivity from last trial for visualization
        'layout': trial_results[-1]['layout'],
        'connectivity': trial_results[-1]['connectivity']
    }

    # Add adaptive stats if they were computed
    if adaptive_stats_aggregated is not None:
        aggregated['adaptive_stats'] = adaptive_stats_aggregated

    return aggregated

def aggregate_adaptive_stats(trial_results: List[Dict]) -> Dict:
    """
    Aggregate adaptive stepping statistics across trials

    Args:
        trial_results: List of trial result dicts, each containing 'adaptive_stats'

    Returns:
        Aggregated statistics with mean, std, min, max across trials
    """
    # Extract adaptive stats from each trial
    all_stats = [result['adaptive_stats'] for result in trial_results]

    # Aggregate scalar statistics
    n_steps_all = [stats['n_steps'] for stats in all_stats]
    avg_dt_all = [stats['avg_dt'] for stats in all_stats]
    min_dt_all = [stats['min_dt'] for stats in all_stats]
    max_dt_all = [stats['max_dt'] for stats in all_stats]

    aggregated = {
        # Summary statistics across trials
        'n_steps_mean': np.mean(n_steps_all),
        'n_steps_std': np.std(n_steps_all),
        'n_steps_min': np.min(n_steps_all),
        'n_steps_max': np.max(n_steps_all),

        'avg_dt_mean': np.mean(avg_dt_all),
        'avg_dt_std': np.std(avg_dt_all),
        'avg_dt_min': np.min(avg_dt_all),
        'avg_dt_max': np.max(avg_dt_all),

        'min_dt_mean': np.mean(min_dt_all),
        'min_dt_min': np.min(min_dt_all),

        'max_dt_mean': np.mean(max_dt_all),
        'max_dt_max': np.max(max_dt_all),

        # Keep individual trial statistics for detailed analysis
        'trial_n_steps': n_steps_all,
        'trial_avg_dt': avg_dt_all,
        'trial_min_dt': min_dt_all,
        'trial_max_dt': max_dt_all,

        # Optional: Store first trial's detailed history for visualization
        'dt_history_sample': all_stats[0]['dt_history'],
        'time_history_sample': all_stats[0]['time_history'],
        'gradient_history_sample': all_stats[0]['gradient_history'],
    }

    return aggregated

def aggregate_trial_currents(trial_results: List[Dict]) -> Optional[Dict]:
    """
    Aggregate recorded currents across trials

    Args:
        trial_results: List of trial result dicts with 'recorded_currents'

    Returns:
        Aggregated current statistics or None if currents not recorded
    """

    # Check if any trial has recorded currents
    if not any('recorded_currents' in tr for tr in trial_results):
        return None

    result = None

    # For now, just return the currents from the first trial
    # (full multi-trial aggregation would require more memory)
    for trial_result in trial_results:
        if 'recorded_currents' in trial_result:
            result = trial_result['recorded_currents']
            break

    return result    


class OptogeneticExperiment:
    """
    Simulate optogenetic stimulation experiments
    
    Now supports multi-trial averaging with different connectivity seeds
    """
    
    def __init__(self, circuit_params: CircuitParams,
                 synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 optimization_json_file: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 base_seed: int = 42,
                 adaptive_config: Optional[GradientAdaptiveStepConfig] = None,
                 use_time_varying_mec: bool = True,
                 mec_pattern_type: str = 'oscillatory',
                 mec_theta_freq: float = 5.0,
                 mec_theta_amplitude: float = 0.3,
                 mec_gamma_freq: float = 20.0,
                 mec_gamma_amplitude: float = 0.15,
                 mec_gamma_coupling_strength: float = 0.8,
                 mec_gamma_preferred_phase: float = 0.0,
                 mec_drift_timescale: float = 200.0,
                 mec_drift_amplitude: float = 0.1,
                 mec_rotation_groups: int = 3,
                 record_currents: bool = False,
                 current_recording_config: Optional[CurrentRecordingConfig] = None):
        """
        Initialize optogenetic experiment
        
        Args:
            circuit_params: CircuitParams instance
            synaptic_params: PerConnectionSynapticParams instance
            opsin_params: OpsinParams instance
            optimization_json_file: Optional path to optimization results JSON
            device: Device to run on
            base_seed: Base random seed for reproducibility (default: 42)
            use_time_varying_mec: Whether to use time-varying MEC input (default: True)
            mec_pattern_type: Type of temporal pattern ('oscillatory', 'drift', 'noisy', 'constant')
            mec_theta_freq: Theta oscillation frequency in Hz (default: 5.0)
            mec_theta_amplitude: Theta modulation depth 0-1 (default: 0.3)
            mec_gamma_freq: Gamma oscillation frequency in Hz (default: 20.0)
            mec_gamma_amplitude: Gamma modulation depth 0-1 (default: 0.15)
            mec_drift_timescale: Correlation time for drift pattern in ms (default: 200.0)
            mec_drift_amplitude: Drift amplitude relative to base (default: 0.1)
            mec_rotation_groups: Number of groups for spatial rotation (default: 3)
        """
        self.circuit_params = circuit_params
        self.opsin_params = opsin_params
        self.synaptic_params = synaptic_params
        self.device = device if device is not None else get_default_device()
        self.base_seed = base_seed
        # Store adaptive config
        self.adaptive_config = adaptive_config

        self.use_time_varying_mec = use_time_varying_mec
        self.mec_pattern_type = mec_pattern_type
        self.mec_theta_freq = mec_theta_freq
        self.mec_theta_amplitude = mec_theta_amplitude
        self.mec_gamma_freq = mec_gamma_freq
        self.mec_gamma_amplitude = mec_gamma_amplitude
        self.mec_gamma_coupling_strength = mec_gamma_coupling_strength
        self.mec_gamma_preferred_phase = mec_gamma_preferred_phase
        self.mec_drift_timescale = mec_drift_timescale
        self.mec_drift_amplitude = mec_drift_amplitude
        self.mec_rotation_groups = mec_rotation_groups
        
        if optimization_json_file is not None:
            success = self._load_optimization_results(optimization_json_file)
            if success:
                logger.info(f"OptogeneticExperiment initialized with optimized parameters from {optimization_json_file}")
            else:
                logger.info(f"Failed to load optimization file. Using default parameters.")

        # Create initial circuit (will be recreated for each trial if regenerate_connectivity is True)
        self.circuit = self._create_circuit(base_seed)
        self.opsin_expression = {}
        self.mec_input = None

        # Add current recording configuration
        self.record_currents = record_currents
        if current_recording_config is None:
            self.current_recording_config = CurrentRecordingConfig(enabled=record_currents)
        else:
            self.current_recording_config = current_recording_config
            self.current_recording_config.enabled = record_currents
        
        logger.info(f"OptogeneticExperiment initialized on device: {self.device}")
        logger.info(f"  Base seed: {base_seed} (trials will use base_seed + trial_index)")

        
    def _create_circuit(self, seed: int) -> DentateCircuit:
        """Create circuit with specified seed for connectivity"""
        set_random_seed(seed, self.device)
        return DentateCircuit(
            self.circuit_params, 
            self.synaptic_params, 
            self.opsin_params,
            device=self.device
        )
    
    def set_adaptive_config(self, config: GradientAdaptiveStepConfig):
        '''Update adaptive stepping configuration'''
        self.adaptive_config = config

    def _load_optimization_results(self, json_filename):
        """Load and process optimization results from JSON file"""
        import json

        try:
            with open(json_filename, 'r') as f:
                self.optimization_data = json.load(f)

            logger.info(f"Loading optimization results from {json_filename}")
            logger.info(f"Results from: {self.optimization_data['optimization_info']['timestamp']}")
            logger.info(f"Best loss achieved: {self.optimization_data['optimization_info']['best_loss']:.6f}")

            # Create modified circuit parameters
            self.circuit_params = self._create_optimized_circuit_params()

            # Create modified synaptic parameters
            self.synaptic_params = self._create_optimized_synaptic_params()
            return True

        except FileNotFoundError:
            logger.info(f"Error: Could not find optimization file {json_filename}")
            return False
        except json.JSONDecodeError:
            logger.info(f"Error: Invalid JSON format in {json_filename}")
            return False
        except KeyError as e:
            logger.info(f"Error: Missing expected key in optimization file: {e}")
            return False
        except Exception as e:
            logger.info(f"Error loading optimization file: {e}")
            return False

    def _create_optimized_synaptic_params(self):
        """Create PerConnectionSynapticParams with optimized connection modulation"""
        if ('optimized_parameters' not in self.optimization_data or 
            'connection_modulation' not in self.optimization_data['optimized_parameters']):
            logger.info("Warning: No optimized connection modulation found in JSON")
            return PerConnectionSynapticParams()

        # Extract base conductance parameters
        base_conductances = self.optimization_data['optimized_parameters'].get('base_conductances', {})
        connection_modulation = self.optimization_data['optimized_parameters']['connection_modulation']

        # Create synaptic parameters with optimized values
        optimized_params = PerConnectionSynapticParams(
            ampa_g_mean=base_conductances.get('ampa_g_mean', 0.2),
            ampa_g_std=base_conductances.get('ampa_g_std', 0.04),
            ampa_g_min=base_conductances.get('ampa_g_min', 0.01),
            ampa_g_max=base_conductances.get('ampa_g_max', 1.5),
            gaba_g_mean=base_conductances.get('gaba_g_mean', 0.25),
            gaba_g_std=base_conductances.get('gaba_g_std', 0.04),
            gaba_g_min=base_conductances.get('gaba_g_min', 0.01),
            gaba_g_max=base_conductances.get('gaba_g_max', 1.5),
            distribution=base_conductances.get('distribution', 'lognormal'),
            connection_modulation=connection_modulation
        )

        return optimized_params
    
    def _create_optimized_circuit_params(self):
        """Create CircuitParams instance using loaded optimization data"""
        # Start with default circuit parameters
        circuit_params = CircuitParams()

        # Update with any circuit configuration from optimization data
        if 'circuit_config' in self.optimization_data:
            config = self.optimization_data['circuit_config']

            # Update population sizes if they were stored in the optimization
            if 'n_gc' in config:
                circuit_params.n_gc = config['n_gc']
            if 'n_mc' in config:
                circuit_params.n_mc = config['n_mc']
            if 'n_pv' in config:
                circuit_params.n_pv = config['n_pv']
            if 'n_sst' in config:
                circuit_params.n_sst = config['n_sst']
            if 'n_mec' in config:
                circuit_params.n_mec = config['n_mec']

        return circuit_params
        
    def create_opsin_expression(self, target_population: str) -> OpsinExpression:
        """Create opsin expression for target population"""
        n_cells = getattr(self.circuit_params, f'n_{target_population}')
        return OpsinExpression(self.opsin_params, n_cells, device=self.device)

    def set_opsin_expression(self, target_population: str,
                             expression_levels: np.ndarray):
        """
        Inject externally-provided opsin expression levels.

        Bypasses stochastic generation in OpsinExpression._generate_expression,
        ensuring exact reproduction of expression patterns from a previous
        experiment (e.g., a nested experiment file).

        simulate_stimulation checks self.opsin_expression[target] and skips
        generation when it is already set, so calling this before simulation
        guarantees the injected expression is used.

        Args:
            target_population: Population name ('pv', 'sst', etc.)
            expression_levels: Array of expression levels [n_cells]
        """
        n_cells = getattr(self.circuit_params, f'n_{target_population}')
        if len(expression_levels) != n_cells:
            raise ValueError(
                f"Expression array length ({len(expression_levels)}) does not match "
                f"n_{target_population} ({n_cells})"
            )

        # Create OpsinExpression shell and replace its expression levels
        opsin = OpsinExpression(self.opsin_params, n_cells, device=self.device)
        opsin.expression_levels = (
            torch.from_numpy(np.asarray(expression_levels))
            .float()
            .to(self.device)
        )
        self.opsin_expression[target_population] = opsin

        logger.info(
            f"  Injected opsin expression for {target_population}: "
            f"{n_cells} cells, "
            f"{int(np.sum(expression_levels > 0))} expressing"
        )
    
    def simulate_stimulation(self, 
                             target_population: str,
                             light_intensity: float,
                             stim_duration: float = 1500.0,
                             stim_start: float = 500.0,
                             post_duration: float = 500.0,
                             mec_current: float = 100.0,
                             mec_current_std: float = 1.0,
                             opsin_current: float = 100.0,
                             include_dentate_spikes: bool = False,
                             ds_times: Optional[List[float]] = None,
                             plot_activity: bool = False,
                             plot_individual_trials: bool = False,
                             plot_aggregated: bool = False,
                             plot_baseline_normalize: bool = False,
                             n_trials: int = 1,
                             regenerate_connectivity_per_trial: bool = False,
                             regenerate_opsin_per_trial: bool = False,
                             adaptive_step: bool = True) -> Dict:
        """
        Simulate optogenetic stimulation experiment with MEC drive

        Supports multi-trial averaging with different connectivity.

        Args:
            target_population: Population to stimulate ('pv', 'sst', etc.)
            light_intensity: Light intensity for stimulation
            stim_start: When to start stimulation (ms)
            stim_duration: Stimulation duration (ms)
            post_duration: Period after stimulation (ms)
            mec_current: MEC drive current (pA)
            opsin_current: Optogenetic current (pA)
            include_dentate_spikes: Whether to include dentate spikes
            ds_times: Specific dentate spike times (optional)
            plot_activity: Whether to plot activity (controls both individual and aggregated)
            plot_individual_trials: If True, plot rasters for individual trials (default: False)
            plot_aggregated: If True, plot raster for trial-averaged activity (default: False)
            plot_baseline_normalize: If True, normalize activity plot relative to pre-stim baseline (default: False)
            n_trials: Number of trials to average (default: 1)
            regenerate_connectivity_per_trial: If True, create new circuit with different
              connectivity for each trial (default: False)
            regenerate_opsin_per_trial: If True, create new opsin expression
              for each trial (default: False)
        
        Returns:
            Dict with averaged results over trials
        """

        # Storage for multi-trial results
        all_trial_results = []

        # Run multiple trials
        for trial in range(n_trials):
            # Set seed for this trial
            trial_seed = self.base_seed + trial
            set_random_seed(trial_seed, self.device)

            if trial == 0:
                logger.info(f"\nRunning {n_trials} trial(s)...")

            if regenerate_connectivity_per_trial:
                logger.info(f"  Trial {trial + 1}/{n_trials}: Creating circuit with seed {trial_seed}...")
                self.circuit = self._create_circuit(trial_seed)
            else:
                logger.info(f"  Trial {trial + 1}/{n_trials}...")

            if (self.opsin_expression.get(target_population, None) is None) or regenerate_opsin_per_trial:
                logger.info(f"  Creating opsin expression...")
                self.opsin_expression[target_population] = self.create_opsin_expression(target_population)

            duration = stim_start + stim_duration + post_duration
            storage_dt = self.circuit_params.dt

            if self.use_time_varying_mec:
                mec_input = generate_time_varying_mec_pattern(
                    n_mec=self.circuit_params.n_mec,
                    duration=duration,
                    dt=storage_dt,
                    base_current=mec_current,
                    trial_index=trial,
                    pattern_type=self.mec_pattern_type,
                    theta_freq=self.mec_theta_freq,
                    theta_amplitude=self.mec_theta_amplitude,
                    gamma_freq=self.mec_gamma_freq,
                    gamma_amplitude=self.mec_gamma_amplitude,
                    gamma_coupling_strength=self.mec_gamma_coupling_strength,
                    gamma_preferred_phase=self.mec_gamma_preferred_phase,
                    drift_timescale=self.mec_drift_timescale,
                    drift_amplitude=self.mec_drift_amplitude,
                    spatial_noise_std=mec_current_std,
                    rotation_groups=self.mec_rotation_groups,
                    device=self.device
                )
            else:
                mec_input = mec_current + torch.randn(self.circuit_params.n_mec, device=self.device) * mec_current_std

            # Run single trial - only plot if requested
            plot_this_trial = plot_activity and plot_individual_trials

            if adaptive_step:
                trial_result = self._simulate_single_trial_adaptive(
                    target_population=target_population,
                    light_intensity=light_intensity,
                    stim_duration=stim_duration,
                    stim_start=stim_start,
                    post_duration=post_duration,
                    mec_input=mec_input,
                    opsin_current=opsin_current,
                    include_dentate_spikes=include_dentate_spikes,
                    ds_times=ds_times,
                    plot_activity=plot_this_trial,
                    trial_index=trial,
                    adaptive_config=self.adaptive_config
                )
            else:
                trial_result = self._simulate_single_trial(
                    target_population=target_population,
                    light_intensity=light_intensity,
                    stim_duration=stim_duration,
                    stim_start=stim_start,
                    post_duration=post_duration,
                    mec_input=mec_input,
                    opsin_current=opsin_current,
                    include_dentate_spikes=include_dentate_spikes,
                    ds_times=ds_times,
                    plot_activity=plot_this_trial,
                    trial_index=trial
                )

            all_trial_results.append(trial_result)

            # Clean up if regenerating circuits
            if regenerate_connectivity_per_trial:
                del self.circuit
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Aggregate results across trials
        aggregated_results = aggregate_trial_results(all_trial_results, n_trials)
        # Add aggregated currents to results if available
        if self.record_currents:
            aggregated_results['recorded_currents'] = aggregate_trial_currents(all_trial_results)
        
        # Plot aggregated results if requested
        if plot_activity and plot_aggregated and n_trials >= 1:
            from DG_visualization import DGCircuitVisualization

            vis = DGCircuitVisualization(self.circuit)

            # Get opsin expression levels as numpy array
            opsin_expression = self.opsin_expression[target_population].expression_levels
            if hasattr(opsin_expression, 'cpu'):
                opsin_expression_np = opsin_expression.cpu().numpy()
            else:
                opsin_expression_np = np.array(opsin_expression)

            # Generate filename suffix
            suffix = f"_aggregated_n{n_trials}"
            if plot_baseline_normalize:
                suffix += "_normalized"
            save_path = f"protocol/DG_{target_population}_stimulation_raster_{light_intensity}{suffix}.pdf"

            # Plot aggregated activity
            fig, _ = vis.plot_aggregated_activity(
                aggregated_results=aggregated_results,
                target_population=target_population,
                opsin_expression_levels=opsin_expression_np,
                light_intensity=light_intensity,
                stim_start=stim_start,
                baseline_normalize=plot_baseline_normalize,
                sort_by_activity=True,
                save_path=save_path
            )
            plt.close(fig)
        
        if regenerate_connectivity_per_trial:
            logger.info(f"Completed {n_trials} trial(s) with different connectivity")
        else:
            logger.info(f"Completed {n_trials} trial(s) with the same connectivity")

        return aggregated_results

    def _simulate_single_trial_adaptive(self,
                                        target_population: str,
                                        light_intensity: float,
                                        stim_start: float,
                                        stim_duration: float,
                                        post_duration: float,
                                        mec_input: torch.Tensor,
                                        opsin_current: float,
                                        include_dentate_spikes: bool,
                                        ds_times: Optional[List[float]],
                                        plot_activity: bool,
                                        trial_index: int,
                                        adaptive_config: Optional[GradientAdaptiveStepConfig] = None
                                        ) -> Dict:
        """
        Version of simulate_single_trial with adaptive stepping

        Changes from original:
        1. Added adaptive_config parameter
        2. Uses GradientAdaptiveStepper for dt selection
        3. Stores to fixed grid with interpolation
        4. Returns adaptive statistics

        All other logic remains the same.
        """

        # Create default adaptive config if not provided
        if adaptive_config is None:
            adaptive_config = GradientAdaptiveStepConfig(
                dt_min=0.05,
                dt_max=0.3,
                gradient_low=0.5,
                gradient_high=10.0,
                gradient_alpha=0.3,
                max_dt_change_factor=1.5
            )

        # Initialize adaptive stepping
        stepper = GradientAdaptiveStepper(adaptive_config)
        state = AdaptiveSimulationState(self.device)

        # Visualization
        vis = None
        if plot_activity:
            vis = DGCircuitVisualization(self.circuit)

        # Reset circuit state
        self.circuit.reset_state()

        # Create opsin expression
        opsin = self.opsin_expression[target_population]
        target_positions = self.circuit.layout.positions[target_population]
        activation_prob = opsin.calculate_activation(target_positions, light_intensity)

        # Identify stimulated vs non-stimulated cells (unchanged)
        activation_threshold = 1e-2
        stimulated_mask = activation_prob >= activation_threshold
        stimulated_indices = torch.where(stimulated_mask)[0].cpu().numpy()
        non_stimulated_indices = torch.where(~stimulated_mask)[0].cpu().numpy()
        activation_prob[non_stimulated_indices] = 0.0
        plot_direct_activation = {}
        logger.info(f"activation_prob: {activation_prob}")

        logger.info(f"stimulated_indices = {stimulated_indices}")
        logger.info(f"non_stimulated_indices = {non_stimulated_indices}")
        # Simulation parameters
        duration = stim_start + stim_duration + post_duration
        stim_start_step_time = stim_start  # Store as time, not step index
        stim_end_time = stim_start + stim_duration

        # Storage on fixed grid (circuit_params.dt resolution for compatibility)
        storage_dt = self.circuit_params.dt
        n_steps = int(duration / storage_dt)
        time_grid = torch.linspace(0, duration, n_steps, device=self.device)
        activity_storage = initialize_activity_storage(self.circuit, time_grid)

        
        # Generate dentate spike times (unchanged)
        if include_dentate_spikes:
            if ds_times is None:
                ds_times = self._generate_dentate_spike_times(duration, baseline_rate=0.5)
        else:
            ds_times = []

        target_opto_current = activation_prob * opsin_current

        # Initialize current recorder with fixed time grid
        current_recorder = None
        if self.record_currents:
            current_recorder = SynapticCurrentRecorder(
                self.current_recording_config,
                self.circuit_params,
                time_grid,  # fixed storage grid
                device=self.device
            )

        # Adaptive simulation loop
        step_on_fixed_grid = 0
        while state.current_time < duration:
            # Compute dt for this step
            if state.step_index > 0:
                dt_adaptive = stepper.compute_dt(
                    activity_current,
                    state.activity_prev,
                    state.dt_history[-1]
                )
            else:
                dt_adaptive = adaptive_config.dt_max  # Start with coarse stepping

            # Ensure we don't overshoot
            if state.current_time + dt_adaptive > duration:
                dt_adaptive = duration - state.current_time

            # Setup inputs
            direct_activation = {}
            if (state.current_time >= stim_start_step_time) and (state.current_time < stim_end_time):
                direct_activation[target_population] = activation_prob * opsin_current
                plot_direct_activation[target_population] = activation_prob * opsin_current

            # Calculate MEC external drive from time-varying pattern
            external_drive = {}
            if self.use_time_varying_mec:
                # Interpolate MEC pattern to current time
                time_idx = int(state.current_time / storage_dt)
                if time_idx < mec_input.shape[1]:
                    mec_drive = mec_input[:, time_idx]
                else:
                    mec_drive = mec_input[:, -1]
            else:
                # Original constant drive
                mec_drive = mec_input

            # Add dentate spike drive
            for ds_time in ds_times:
                if abs(state.current_time - ds_time) < 50:  # 50ms window
                    ds_strength = 5.0 * torch.tensor(
                        np.exp(-((state.current_time - ds_time) / 10.0) ** 2),
                        device=self.device
                    )
                    mec_drive += ds_strength

            external_drive['mec'] = mec_drive

            # Update circuit with adaptive dt
            activity_current = update_circuit_with_adaptive_dt(
                self.circuit, direct_activation, external_drive, dt_adaptive
            )

            # Interpolate to fixed grid for storage
            if state.step_index == 0:
                # First step: just store directly at t=0
                for pop in activity_storage:
                    activity_storage[pop][:, 0] = activity_current[pop]
            else:
                # Interpolate between previous and current time
                _interpolate_activity_to_grid(
                    activity_current,
                    state.activity_prev,
                    state.current_time,
                    state.current_time + dt_adaptive,
                    time_grid,
                    activity_storage
                )

            # Record currents at fixed grid points
            if current_recorder is not None:
                # Find which fixed grid point(s) we just passed
                current_grid_idx = int(state.current_time / storage_dt)
                if current_grid_idx >= step_on_fixed_grid:
                    # Record at this grid point
                    current_recorder.record_currents(self.circuit, current_grid_idx)
                    step_on_fixed_grid = current_grid_idx + 1

            # Update state for next iteration
            state.update(activity_current, dt_adaptive, stepper.grad_smooth)

        # Move results to CPU
        time_cpu = time_grid.cpu()
        activity_trace_cpu = {pop: activity.cpu() for pop, activity in activity_storage.items()}

        # Visualization
        if vis:
            split_populations = {
                target_population: {
                    'unit_ids_part1': stimulated_indices.tolist(),
                    'unit_ids_part2': non_stimulated_indices.tolist(),
                    'part1_label': f'Stimulated (n={len(stimulated_indices)})',
                    'part2_label': f'Non-stimulated (n={len(non_stimulated_indices)})'
                }
    }
            #fig, _ = vis.plot_activity_patterns(
            #    activity_trace_cpu,
            #    save_path=f"protocol/DG_{target_population}_stimulation_activity_{light_intensity}_trial{trial_index}.png"
            #)

            fig, _ = vis.plot_activity_raster(
                activity_trace_cpu,
                split_populations=split_populations,
                direct_activation=plot_direct_activation,
                sort_by_activity=True,
                save_path=f"protocol/DG_{target_population}_stimulation_raster_{light_intensity}_trial{trial_index}.png"
            )
            plt.close(fig)

        # Return results with adaptive statistics
        results = {
            'time': time_cpu,
            'activity_trace': activity_trace_cpu,
            'opsin_expression': opsin.expression_levels.cpu(),
            'target_positions': target_positions.cpu(),
            'dentate_spike_times': ds_times,
            'layout': self.circuit.layout,
            'connectivity': self.circuit.connectivity,
            'stimulated_indices': stimulated_indices,
            'non_stimulated_indices': non_stimulated_indices,
            'adaptive_stats': state.get_statistics()
        }

        # Add recorded currents to results
        if current_recorder is not None:
            results['recorded_currents'] = current_recorder.get_results()

        return results

    
    def _simulate_single_trial(self,
                              target_population: str,
                              light_intensity: float,
                              stim_start: float,
                              stim_duration: float,
                              post_duration: float,
                              mec_input: torch.Tensor,
                              opsin_current: float,
                              include_dentate_spikes: bool,
                              ds_times: Optional[List[float]],
                              plot_activity: bool,
                              trial_index: int) -> Dict:
        """Run a single trial of optogenetic stimulation"""
        
        vis = None
        if plot_activity:
            vis = DGCircuitVisualization(self.circuit)
        
        # Reset circuit state
        self.circuit.reset_state()
        
        target_positions = self.circuit.layout.positions[target_population]

        # Calculate direct optogenetic activation
        opsin = self.opsin_expression[target_population]
        activation_prob = opsin.calculate_activation(target_positions, light_intensity)

        # Identify stimulated vs non-stimulated cells
        # (cells with opsin expression above threshold)
        activation_threshold = 1e-2
        stimulated_mask = activation_prob >= activation_threshold
        stimulated_indices = torch.where(stimulated_mask)[0].cpu().numpy()
        non_stimulated_indices = torch.where(~stimulated_mask)[0].cpu().numpy()
        activation_prob[non_stimulated_indices] = 0.0
        
        # Simulation parameters
        dt = self.circuit_params.dt
        duration = stim_start + stim_duration + post_duration
        n_steps = int(duration / dt)
        stim_start_step = int(stim_start / dt)
        stim_end_step = int((stim_start + stim_duration) / dt)
        
        # Storage for results (on device initially)
        time = torch.arange(n_steps, device=self.device) * dt
        activity_trace = {
            'gc': torch.zeros(self.circuit_params.n_gc, n_steps, device=self.device),
            'mc': torch.zeros(self.circuit_params.n_mc, n_steps, device=self.device),
            'pv': torch.zeros(self.circuit_params.n_pv, n_steps, device=self.device),
            'sst': torch.zeros(self.circuit_params.n_sst, n_steps, device=self.device),
            'mec': torch.zeros(self.circuit_params.n_mec, n_steps, device=self.device)
        }

        
        # Generate dentate spike times (random occurrences)
        if include_dentate_spikes:
            if ds_times is None:
                ds_times = self._generate_dentate_spike_times(duration, baseline_rate=0.5)
        else:
            ds_times = []

        plot_direct_activation = {}
            
        # Run simulation
        for t in range(n_steps):
            current_time = t * dt

            direct_activation = {}
            if (t >= stim_start_step) and (t < stim_end_step):
                # Convert to strong current injection
                direct_activation[target_population] = activation_prob * opsin_current
                plot_direct_activation[target_population] = activation_prob * opsin_current
                
            # Calculate MEC external drive (dentate spikes + baseline)
            external_drive = {}
            # Calculate MEC external drive from time-varying pattern
            external_drive = {}
            if self.use_time_varying_mec:
                mec_drive = mec_input[:, t]
            else:
                # Original constant drive with noise
                mec_drive = mec_input

            
            # Add dentate spike drive
            for ds_time in ds_times:
                if abs(current_time - ds_time) < 50:  # 50ms window around DS
                    # Gaussian profile around DS peak
                    ds_strength = 5.0 * torch.tensor(
                        np.exp(-((current_time - ds_time) / 10.0) ** 2),
                        device=self.device
                    )
                    mec_drive += ds_strength
            
            external_drive['mec'] = mec_drive
            
            # Update circuit
            current_activity = self.circuit(direct_activation, external_drive)
            
            # Store activity
            for pop in activity_trace:
                activity_trace[pop][:, t] = current_activity[pop]

        # Move results to CPU for visualization/storage
        time_cpu = time.cpu()
        activity_trace_cpu = {pop: activity.cpu() for pop, activity in activity_trace.items()}

        if vis:
            #fig, _ = vis.plot_activity_patterns(
            #    activity_trace_cpu,
            #    save_path=f"protocol/DG_{target_population}_stimulation_activity_{light_intensity}_trial{trial_index}.png"
            #)

            split_populations = {
                target_population: {
                    'unit_ids_part1': stimulated_indices.tolist(),
                    'unit_ids_part2': non_stimulated_indices.tolist(),
                    'part1_label': f'Stimulated (n={len(stimulated_indices)})',
                    'part2_label': f'Non-stimulated (n={len(non_stimulated_indices)})'
                }
            }
            fig, _ = vis.plot_activity_raster(
                activity_trace_cpu,
                split_populations=split_populations,
                direct_activation=plot_direct_activation,
                save_path=f"protocol/DG_{target_population}_stimulation_raster_{light_intensity}_trial{trial_index}.png"
            )
            plt.close(fig)
                
        return {
            'time': time_cpu,
            'activity_trace': activity_trace_cpu,
            'opsin_expression': opsin.expression_levels.cpu(),
            'target_positions': target_positions.cpu(),
            'dentate_spike_times': ds_times,
            'layout': self.circuit.layout,
            'connectivity': self.circuit.connectivity,
            'stimulated_indices': stimulated_indices,
            'non_stimulated_indices': non_stimulated_indices
        }
    
    
    def _generate_dentate_spike_times(self, duration: float, baseline_rate: float = 0.5) -> List[float]:
        """Generate random dentate spike times (Poisson process)"""
        import random
        
        ds_times = []
        current_time = 0.0
        
        while current_time < duration:
            # Poisson process: inter-spike interval
            isi = -np.log(random.random()) / baseline_rate * 1000  # Convert to ms
            current_time += isi
            
            if current_time < duration:
                ds_times.append(current_time)
        
        return ds_times
        
