import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, NamedTuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pathlib import Path
import tqdm

from dendritic_somatic_transfer import (
    get_default_device
)
from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit,
    CircuitParams,
    PerConnectionSynapticParams,
    OpsinParams
)
from DG_visualization import (
    DGCircuitVisualization
)

from gradient_adaptive_stepper import (
    GradientAdaptiveStepConfig,
    GradientAdaptiveStepper,
    AdaptiveSimulationState,
    update_circuit_with_adaptive_dt,
    interpolate_to_fixed_grid,
    initialize_activity_storage
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

        # Some cells have no expression (transduction failure)
        sorted_indices = np.argsort(-expression)
        print(f"sorted expression = {expression[sorted_indices]}")
        n_failed = int(round(self.params.failure_rate * self.n_cells))
        print(f"failure_rate = {self.params.failure_rate} n_cells = {self.n_cells} n_failed = {n_failed}")
        print(f"sorted expression before failure = {expression[sorted_indices]}")
        expression[sorted_indices[-n_failed:]] = 0.0
        print(f"sorted expression after failure = {expression[sorted_indices]}")
        
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
                             base_currents: np.ndarray,
                             n_groups: int = 3,
                             device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Get spatial pattern that rotates across trials
    
    Divides MEC population into groups and rotates the group assignment across trials.
    
    Args:
        n_mec: Number of MEC neurons
        trial_index: Current trial index (0, 1, 2, ...)
        n_groups: Number of groups to divide MEC population into
        device: Device to create tensor on
        
    Returns:
        group_tensor: [n_mec] tensor of rates
    """
    if device is None:
        device = get_default_device()
    
    # Divide neurons into groups
    group_size = n_mec // n_groups
    remainder = n_mec % n_groups
    
    # Apply circular rotation based on trial index
    rotation_amount = ((trial_index + 1) * group_size) % n_mec
    rotated_array = np.roll(base_currents, rotation_amount)

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
                                      drift_amplitude: float = 0.4,
                                      rotation_groups: int = 3,
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
        pattern_type: 'oscillatory', 'drift', or 'constant'
        theta_freq: Theta oscillation frequency (Hz)
        theta_amplitude: Relative theta modulation depth (0-1)
        gamma_freq: Gamma oscillation frequency (Hz)  
        gamma_amplitude: Relative gamma modulation depth (0-1)
        drift_timescale: Correlation time for drift pattern (ms)
        drift_amplitude: Relative drift amplitude (0-1)
        rotation_groups: Number of groups for trial rotation
        device: Device to create tensors on
        
    Returns:
        mec_pattern: [n_mec, n_timesteps] tensor of MEC currents (pA)
    """
    if device is None:
        device = get_default_device()
    
    n_steps = int(duration / dt)
    time_vec = torch.arange(n_steps, device=device) * dt  # in ms
    
    # Get spatial rotation pattern for this trial
    base_rates = get_mec_rotation_pattern(
        n_mec, trial_index, base_current, rotation_groups, device
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
        # Gamma amplitude peaks at specific theta phase
        # gamma_coupling_strength controls coupling (0=independent, 1=fully coupled)
        # gamma_preferred_phase controls which theta phase gamma peaks at

        # Compute theta phase at each time point for each neuron
        theta_phase_matrix = 2 * np.pi * theta_freq * time_sec.unsqueeze(0) + theta_phases.unsqueeze(1)
    
        # Gamma envelope: varies from (1 - coupling_strength) to (1 + coupling_strength)
        # Peaks when theta is at preferred phase
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
        
        # OU parameters
        tau = drift_timescale  # correlation time in ms
        sigma = drift_amplitude * base_current  # noise amplitude
        
        # Initialize with random starting points
        mec_pattern = torch.zeros(n_mec, n_steps, device=device)
        mec_pattern[:, 0] = base_rates
        
        # Generate OU process for each neuron
        sqrt_dt = np.sqrt(dt)
        for t in range(1, n_steps):
            # OU dynamics: dx = -(x - mu)/tau * dt + sigma * sqrt(dt) * dW
            drift_term = -(mec_pattern[:, t-1] - base_rates) / tau * dt
            noise_term = sigma * sqrt_dt * torch.randn(n_mec, device=device)
            mec_pattern[:, t] = mec_pattern[:, t-1] + drift_term + noise_term
            
            # Clip to reasonable range (prevent negative currents)
            mec_pattern[:, t] = torch.clamp(mec_pattern[:, t], 
                                           min=base_current * 0.2,
                                           max=base_current * 2.0)
    
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}. Use 'oscillatory', 'drift', or 'constant'")
    
    # Ensure no negative currents
    mec_pattern = torch.clamp(mec_pattern, min=0.0)
    
    return mec_pattern


    
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
                 mec_drift_amplitude: float = 0.4,
                 mec_rotation_groups: int = 3):
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
            mec_pattern_type: Type of temporal pattern ('oscillatory', 'drift', 'constant')
            mec_theta_freq: Theta oscillation frequency in Hz (default: 5.0)
            mec_theta_amplitude: Theta modulation depth 0-1 (default: 0.3)
            mec_gamma_freq: Gamma oscillation frequency in Hz (default: 20.0)
            mec_gamma_amplitude: Gamma modulation depth 0-1 (default: 0.15)
            mec_drift_timescale: Correlation time for drift pattern in ms (default: 200.0)
            mec_drift_amplitude: Drift amplitude relative to base (default: 0.4)
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
                print(f"OptogeneticExperiment initialized with optimized parameters from {optimization_json_file}")
            else:
                print(f"Failed to load optimization file. Using default parameters.")

        # Create initial circuit (will be recreated for each trial if regenerate_connectivity is True)
        self.circuit = self._create_circuit(base_seed)
        self.opsin_expression = {}
 
        print(f"OptogeneticExperiment initialized on device: {self.device}")
        print(f"  Base seed: {base_seed} (trials will use base_seed + trial_index)")

        
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

            print(f"Loading optimization results from {json_filename}")
            print(f"Results from: {self.optimization_data['optimization_info']['timestamp']}")
            print(f"Best loss achieved: {self.optimization_data['optimization_info']['best_loss']:.6f}")

            # Create modified circuit parameters
            self.circuit_params = self._create_optimized_circuit_params()

            # Create modified synaptic parameters
            self.synaptic_params = self._create_optimized_synaptic_params()

            return True

        except FileNotFoundError:
            print(f"Error: Could not find optimization file {json_filename}")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_filename}")
            return False
        except KeyError as e:
            print(f"Error: Missing expected key in optimization file: {e}")
            return False
        except Exception as e:
            print(f"Error loading optimization file: {e}")
            return False

    def _create_optimized_synaptic_params(self):
        """Create PerConnectionSynapticParams with optimized connection modulation"""
        if ('optimized_parameters' not in self.optimization_data or 
            'connection_modulation' not in self.optimization_data['optimized_parameters']):
            print("Warning: No optimized connection modulation found in JSON")
            return PerConnectionSynapticParams()

        # Extract base conductance parameters
        base_conductances = self.optimization_data['optimized_parameters'].get('base_conductances', {})
        connection_modulation = self.optimization_data['optimized_parameters']['connection_modulation']

        # Create synaptic parameters with optimized values
        optimized_params = PerConnectionSynapticParams(
            ampa_g_mean=base_conductances.get('ampa_g_mean', 0.15),
            ampa_g_std=base_conductances.get('ampa_g_std', 0.05),
            ampa_g_min=base_conductances.get('ampa_g_min', 0.01),
            ampa_g_max=base_conductances.get('ampa_g_max', 1.5),
            gaba_g_mean=base_conductances.get('gaba_g_mean', 0.25),
            gaba_g_std=base_conductances.get('gaba_g_std', 0.08),
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
                             n_trials: int = 1,
                             regenerate_connectivity_per_trial: bool = False,
                             regenerate_opsin_per_trial: bool = False,
                             adaptive_step: bool = True) -> Dict:
        """
        Simulate optogenetic stimulation experiment with MEC drive
        
        Now supports multi-trial averaging with different connectivity
        
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
            plot_activity: Whether to plot activity
            n_trials: Number of trials to average (default: 1)
            regenerate_connectivity_per_trial: If True, create new circuit with different
              connectivity for each trial (default: False)
            regenerate_opsin_per_trial: If True, create new opsin expression
              for each trial (default: False)
        
        Returns:
            Dict with averaged results over trials, including:
                - time: Time vector
                - activity_trace_mean: Mean activity traces
                - activity_trace_std: Std of activity traces  
                - opsin_expression_mean: Mean opsin expression
                - trial_results: List of individual trial results
        """
        
        # Storage for multi-trial results
        all_trial_results = []
        
        # Run multiple trials
        for trial in range(n_trials):
            # Set seed for this trial
            trial_seed = self.base_seed + trial
            set_random_seed(trial_seed, self.device)
            
            if trial == 0:
                print(f"\nRunning {n_trials} trial(s)...")

            if regenerate_connectivity_per_trial:
                print(f"  Trial {trial + 1}/{n_trials}: Creating circuit with seed {trial_seed}...")
                # Create new circuit for this trial with fresh connectivity
                self.circuit = self._create_circuit(trial_seed)
            else:
                print(f"  Trial {trial + 1}/{n_trials}...")

            if (self.opsin_expression.get(target_population, None) is None) or regenerate_opsin_per_trial:
                print(f"  Creating opsin expression...")
                self.opsin_expression[target_population] = self.create_opsin_expression(target_population)

                
            # Run single trial
            if adaptive_step:
                trial_result = self._simulate_single_trial_adaptive(
                    target_population=target_population,
                    light_intensity=light_intensity,
                    stim_duration=stim_duration,
                    stim_start=stim_start,
                    post_duration=post_duration,
                    mec_current=mec_current + torch.randn(self.circuit_params.n_mec, device=self.device) * mec_current_std,
                    opsin_current=opsin_current,
                    include_dentate_spikes=include_dentate_spikes,
                    ds_times=ds_times,
                    plot_activity=plot_activity, # and trial == 0),  # Only plot first trial
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
                    mec_current=mec_current + torch.randn(self.circuit_params.n_mec, device=self.device) * mec_current_std,
                    opsin_current=opsin_current,
                    include_dentate_spikes=include_dentate_spikes,
                    ds_times=ds_times,
                    plot_activity=plot_activity, # and trial == 0),  # Only plot first trial
                    trial_index=trial
                )
            
            all_trial_results.append(trial_result)

            # Clean up to free memory (only if we created new circuits)
            if regenerate_connectivity_per_trial:
                del self.circuit
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Aggregate results across trials
        aggregated_results = self._aggregate_trial_results(all_trial_results, n_trials)

        if regenerate_connectivity_per_trial:
            print(f"Completed {n_trials} trial(s) with different connectivity")
        else:
            print(f"Completed {n_trials} trial(s) with the same connectivity")
            
        return aggregated_results


    def _simulate_single_trial_adaptive(self,
                                        target_population: str,
                                        light_intensity: float,
                                        stim_start: float,
                                        stim_duration: float,
                                        post_duration: float,
                                        mec_current: float,
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
        print(f"activation_prob: {activation_prob}")

        print(f"stimulated_indices = {stimulated_indices}")
        print(f"non_stimulated_indices = {non_stimulated_indices}")
        # Simulation parameters
        duration = stim_start + stim_duration + post_duration
        stim_start_step_time = stim_start  # Store as time, not step index
        stim_end_time = stim_start + stim_duration

        # Storage on fixed grid (circuit_params.dt resolution for compatibility)
        storage_dt = self.circuit_params.dt
        n_steps = int(duration / storage_dt)
        time_grid = torch.linspace(0, duration, n_steps, device=self.device)
        activity_storage = initialize_activity_storage(self.circuit, time_grid)

        if self.use_time_varying_mec:
            mec_pattern = generate_time_varying_mec_pattern(
                n_mec=self.circuit_params.n_mec,
                duration=duration,
                dt=storage_dt,  # Generate at storage resolution
                base_current=mec_current,
                trial_index=trial_index,
                pattern_type=self.mec_pattern_type,
                theta_freq=self.mec_theta_freq,
                theta_amplitude=self.mec_theta_amplitude,
                gamma_freq=self.mec_gamma_freq,
                gamma_amplitude=self.mec_gamma_amplitude,
                gamma_coupling_strength=self.mec_gamma_coupling_strength,
                gamma_preferred_phase=self.mec_gamma_preferred_phase,
                drift_timescale=self.mec_drift_timescale,
                drift_amplitude=self.mec_drift_amplitude,
                rotation_groups=self.mec_rotation_groups,
                device=self.device
            )  # [n_mec, n_steps]
        else:
            # Fall back to constant MEC input with noise
            mec_pattern = None
        
        # Generate dentate spike times (unchanged)
        if include_dentate_spikes:
            if ds_times is None:
                ds_times = self._generate_dentate_spike_times(duration, baseline_rate=0.5)
        else:
            ds_times = []

        # Adaptive simulation loop
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
                if time_idx < mec_pattern.shape[1]:
                    mec_drive = mec_pattern[:, time_idx]
                else:
                    mec_drive = mec_pattern[:, -1]
            else:
                # Original constant drive
                mec_drive = mec_current

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
                save_path=f"protocol/DG_{target_population}_stimulation_raster_{light_intensity}_trial{trial_index}.png"
            )
            plt.close(fig)

        # Return results with adaptive statistics
        return {
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

    
    def _simulate_single_trial(self,
                              target_population: str,
                              light_intensity: float,
                              stim_start: float,
                              stim_duration: float,
                              post_duration: float,
                              mec_current: float,
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

        print(f"opsin_expression {target_population}: {opsin.expression_levels}")

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

        # Generate time-varying MEC pattern
        if self.use_time_varying_mec:
            mec_pattern = generate_time_varying_mec_pattern(
                n_mec=self.circuit_params.n_mec,
                duration=duration,
                dt=dt,
                base_current=mec_current,
                trial_index=trial_index,
                pattern_type=self.mec_pattern_type,
                theta_freq=self.mec_theta_freq,
                theta_amplitude=self.mec_theta_amplitude,
                gamma_freq=self.mec_gamma_freq,
                gamma_amplitude=self.mec_gamma_amplitude,
                drift_timescale=self.mec_drift_timescale,
                drift_amplitude=self.mec_drift_amplitude,
                rotation_groups=self.mec_rotation_groups,
                device=self.device
            )  # [n_mec, n_steps]
        else:
            mec_pattern = None
        
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
                mec_drive = mec_pattern[:, t]
            else:
                # Original constant drive with noise
                mec_drive = mec_current

            
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
    
    def _aggregate_trial_results(self, trial_results: List[Dict], n_trials: int) -> Dict:
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
            activity_trace_std[pop] = torch.std(stacked, dim=0)    # [n_neurons, n_steps]
        
        # Opsin expression (may vary slightly per trial due to stochastic generation)
        opsin_expression_stacked = torch.stack(opsin_expressions_all, dim=0)
        opsin_expression_mean = torch.mean(opsin_expression_stacked, dim=0)
        opsin_expression_std = torch.std(opsin_expression_stacked, dim=0)

        # Aggregate adaptive stats if present
        adaptive_stats_aggregated = None
        if 'adaptive_stats' in trial_results[0]:
            adaptive_stats_aggregated = self._aggregate_adaptive_stats(trial_results)

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

    def _aggregate_adaptive_stats(self, trial_results: List[Dict]) -> Dict:
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
                               device: Optional[torch.device] = None,
                               n_trials: int = 1,
                               regenerate_connectivity_per_trial: bool = False,
                               base_seed: int = 42,
                               load_results_file: Optional[str] = None,
                               save_results_file: Optional[str] = None,
                               auto_save: bool = True,
                               save_full_activity: bool = False,
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
                               mec_drift_amplitude: float = 0.4,
                               mec_rotation_groups: int = 3,
                               validate_input_balance: bool = True):
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
        print(f"\nLoading experiment results from file: {load_results_file}")
        results, conn_analysis, conductance_analysis, metadata = load_experiment_results(load_results_file)
        
        # Print loaded configuration
        print("\nLoaded experiment configuration:")
        print(f"  Light intensities: {metadata.get('intensities', 'Unknown')}")
        print(f"  MEC current: {metadata.get('mec_current', 'Unknown')} pA")
        print(f"  Opsin current: {metadata.get('opsin_current', 'Unknown')} pA")
        print(f"  Stimulation: {metadata.get('stim_start', 'Unknown')} ms start, "
              f"{metadata.get('stim_duration', 'Unknown')} ms duration")
        
        return results, conn_analysis, conductance_analysis
    
    # Otherwise, run the experiment
    if device is None:
        device = get_default_device()
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
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
        mec_rotation_groups=mec_rotation_groups
    )
    
    print(f"\nRunning comparative experiment with {n_trials} trial(s) per condition")
    print(f"Base seed: {base_seed}\n")
    
    # Analyze connectivity patterns (from last trial's circuit)
    conn_analysis = analyze_connectivity_patterns(experiment)
    
    # Analyze conductance patterns
    conductance_analysis = analyze_conductance_patterns(experiment)
    
    print("Connectivity Analysis:")
    print(f"GC->MC distances (mean): {np.mean(conn_analysis['gc_mc_distances']):.3f} mm")
    print(f"GC->PV distances (mean): {np.mean(conn_analysis['gc_pv_distances']):.3f} mm")
    print(f"GC->SST distances (mean): {np.mean(conn_analysis['gc_sst_distances']):.3f} mm")
    print(f"MC->GC distances (mean): {np.mean(conn_analysis['mc_gc_distances']):.3f} mm")
    print(f"MC->SST distances (mean): {np.mean(conn_analysis['mc_sst_distances']):.3f} mm")
    print(f"Local radius threshold: {conn_analysis['local_radius']} mm")
    print(f"Distant minimum threshold: {conn_analysis['distant_min']} mm")

    print("\nConductance Analysis:")
    print("-" * 50)
    for conn_name, stats in conductance_analysis.items():
        if stats['n_connections'] > 0:
            print(f"{conn_name} ({stats['synapse_type']}):")
            print(f"  Connections: {stats['n_connections']}")
            print(f"  Conductance: {stats['mean_conductance']:.3f} +/- {stats['std_conductance']:.3f} nS")
            print(f"  CV: {stats['cv_conductance']:.2f}")
            print(f"  Range: [{stats['min_conductance']:.3f}, {stats['max_conductance']:.3f}] nS")
    
    # Test different stimulation intensities
    results = {}

    if stim_start < warmup:
        stim_start = warmup
    
    for target in ['pv', 'sst']:
        results[target] = {}
        print(f"\nTesting {target.upper()} stimulation...")
        
        for intensity in intensities:
            print(f"\n  Intensity: {intensity}")
            
            # Run multi-trial stimulation
            result = experiment.simulate_stimulation(
                target, intensity,
                stim_start=stim_start,
                stim_duration=stim_duration,
                plot_activity=plot_activity,
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
                    print(f"  Steps: {adaptive_stats['n_steps_mean']:.0f} +/- {adaptive_stats['n_steps_std']:.0f} "
                          f"(range: {adaptive_stats['n_steps_min']}-{adaptive_stats['n_steps_max']})")
                    print(f"  Avg dt: {adaptive_stats['avg_dt_mean']:.3f} +/- {adaptive_stats['avg_dt_std']:.3f} ms")
                    print(f"  dt range: [{adaptive_stats['min_dt_mean']:.3f}, {adaptive_stats['max_dt_mean']:.3f}] ms")
                # For single trial: show direct values
                else:
                    print(f"  Steps: {adaptive_stats['n_steps_mean']:.0f} (avg dt: {adaptive_stats['avg_dt_mean']:.3f} ms)")
                    print(f"  dt range: [{adaptive_stats['min_dt_min']:.3f}, {adaptive_stats['max_dt_max']:.3f}] ms")


                
            # Analyze network effects (using mean across trials)
            time = result['time']
            activity_mean = result['activity_trace_mean']
            activity_std = result['activity_trace_std']
            opsin_expression_mean = result['opsin_expression_mean']
            
            baseline_mask = (time >= warmup) & (time < stim_start)
            stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
            
            analysis = {}
            analysis['opsin_expression_mean'] = opsin_expression_mean
            analysis['n_trials'] = n_trials

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
                    analysis[f'{pop}_stim_rates_mean'] = stim_rate.numpy()
                    analysis[f'{pop}_baseline_rates_mean'] = baseline_rate.numpy()
                    
                    # Calculate trial-to-trial variability for target population
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
                    continue

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
    
    return results, conn_analysis, conductance_analysis


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
    import pickle
    from datetime import datetime
    
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
    
    print(f"\nExperiment results saved to: {filepath}")
    print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    if metadata:
        print(f"  Configuration: {metadata.get('n_trials', 'N/A')} trials, "
              f"seed {metadata.get('base_seed', 'N/A')}")


def load_experiment_results(filepath: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load comparative experiment results from file
    
    Args:
        filepath: Path to saved results file
        
    Returns:
        Tuple of (results, connectivity_analysis, conductance_analysis, metadata)
    """
    import pickle
    
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
    
    print(f"\nLoaded experiment results from: {filepath}")
    print(f"  Saved: {save_data.get('timestamp', 'Unknown')}")
    print(f"  Version: {save_data.get('version', 'Unknown')}")
    if metadata:
        print(f"  Configuration: {metadata.get('n_trials', 'N/A')} trials, "
              f"seed {metadata.get('base_seed', 'N/A')}")
        if 'optimization_file' in metadata and metadata['optimization_file']:
            print(f"  Optimization: {metadata['optimization_file']}")
    
    # Convert numpy arrays back to torch tensors where needed
    def convert_numpy_to_tensors(obj):
        """Recursively convert numpy arrays to torch tensors for activity traces"""
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, dict):
            # Only convert specific keys that are expected to be tensors
            tensor_keys = ['activity_trace_mean', 'activity_trace_std', 
                          'opsin_expression_mean', 'opsin_expression_std', 'time']
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
    from datetime import datetime
    
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
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Gradient magnitude over time
    ax2 = axes[1]
    ax2.plot(time_history[1:], gradient_history, 'g-', linewidth=1.5, alpha=0.7)
    ax2.axvspan(stim_start, stim_start + stim_duration, 
               alpha=0.1, color='orange', label='Stimulation')
    ax2.set_ylabel('Gradient Magnitude (Hz/ms)', fontsize=11)
    ax2.set_title('Activity Gradient Over Time', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=9)
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
    ax3.legend(loc='best', fontsize=9)
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
        print(f"Saved adaptive stepping analysis to: {save_path}")
    
    plt.show()

def plot_comparative_experiment_results(results: Dict, conn_analysis: Dict,
                                        stimulation_level: float = 1.0,
                                        save_path: Optional[str] = None) -> None:
    """
    Create visualizations from comparative experiment results
    
    Now handles multi-trial statistics with error bars
    """
    
    # Check if we have multi-trial data
    n_trials = results['pv'][stimulation_level].get('n_trials', 1)
    has_multitrial = n_trials > 1
    
    # Define colors matching the paper
    colors = {
        'pv': '#FF6B9D', # Pink 
        'sst': '#45B7D1', # Blue
        'gc': '#96CEB4', # Green 
        'mc': '#FFEAA7', # Yellow
    }
    
    # Create summary figure
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Firing ratio bar plots
    ax1 = plt.subplot(3, 4, (1, 2))
    
    targets = ['pv', 'sst']
    populations = ['gc', 'mc', 'pv', 'sst']
    
    bar_data = []
    bar_errors = []  # NEW: Error bars for multi-trial
    bar_labels = []
    bar_colors = []
    
    for target in targets:
        for pop in populations:
            if pop != target and f'{pop}_mean_change' in results[target][stimulation_level]:
                baseline_rate = results[target][stimulation_level][f'{pop}_mean_baseline_rate']
                stim_rate = results[target][stimulation_level][f'{pop}_mean_stim_rate']
                
                if baseline_rate > 0:
                    ratio = np.log2(stim_rate / baseline_rate)
                    bar_data.append(ratio)
                    bar_labels.append(f'{target.upper()}→{pop.upper()}')
                    bar_colors.append(colors[pop])
                    
                    # Add error bars if multi-trial
                    if has_multitrial and f'{pop}_mean_change_std' in results[target][stimulation_level]:
                        # Approximate error in log2 ratio using error propagation
                        std = results[target][stimulation_level][f'{pop}_mean_change_std']
                        error = std / (baseline_rate * np.log(2))  # Approximate
                        bar_errors.append(error)
                    else:
                        bar_errors.append(0)
    
    if bar_data:
        x_pos = np.arange(len(bar_labels))
        bars = ax1.bar(x_pos, bar_data, color=bar_colors, alpha=0.7, 
                      edgecolor='black', linewidth=1.5,
                      yerr=bar_errors if has_multitrial else None,
                      capsize=5)
        
        for i, (bar, value) in enumerate(zip(bars, bar_data)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel(r'Modulation Ratio ($\log_2$)', fontsize=11)
    title_suffix = f' (n={n_trials} trials)' if has_multitrial else ' (Single Trial)'
    ax1.set_title(f'Firing Rate Modulation{title_suffix}', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Panel B: Network effects summary with error bars
    ax2 = plt.subplot(3, 4, (3, 4))
    
    effect_data = []
    effect_errors = []
    effect_labels = []
    
    for target in targets:
        for pop in populations:
            if pop != target:
                excited_key = f'{pop}_excited'
                if excited_key in results[target][stimulation_level]:
                    excited_frac = results[target][stimulation_level][excited_key]
                    inhibited_frac = results[target][stimulation_level][f'{pop}_inhibited']
                    
                    effect_data.append([excited_frac, inhibited_frac])
                    effect_labels.append(f'{target.upper()}→{pop.upper()}')
                    
                    # Add error bars if available
                    if has_multitrial and f'{pop}_excited_std' in results[target][stimulation_level]:
                        excited_std = results[target][stimulation_level][f'{pop}_excited_std']
                        effect_errors.append([excited_std, excited_std])  # Same for both
                    else:
                        effect_errors.append([0, 0])
    
    if effect_data:
        effect_array = np.array(effect_data)
        effect_errors_array = np.array(effect_errors)
        x_pos = np.arange(len(effect_labels))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, effect_array[:, 0], width, 
               label='Excited', color='red', alpha=0.7, edgecolor='black',
               yerr=effect_errors_array[:, 0] if has_multitrial else None,
               capsize=5)
        bars2 = ax2.bar(x_pos + width/2, effect_array[:, 1], width, 
               label='Inhibited', color='blue', alpha=0.7, edgecolor='black',
               yerr=effect_errors_array[:, 1] if has_multitrial else None,
               capsize=5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom',
                        fontsize=8)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(effect_labels, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Fraction of Cells', fontsize=11)
        title_suffix = f' (n={n_trials} trials)' if has_multitrial else ' (Single Trial)'
        ax2.set_title(f'Network Effects{title_suffix}', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_axisbelow(True)
    
    # Panel C: Connectivity analysis (unchanged)
    ax3 = plt.subplot(3, 4, (5, 6))
    
    mec_conn = conn_analysis['mec_connectivity']
    conn_data = [
        mec_conn['mec_to_pv'],
        mec_conn['mec_to_gc'], 
        mec_conn['mec_to_sst']
    ]
    conn_labels = ['MEC -> PV', 'MEC -> GC', 'MEC -> SST']
    conn_colors = [colors['pv'], colors['gc'], colors['sst']]
    
    bars = ax3.bar(conn_labels, conn_data, color=conn_colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of Connections', fontsize=11)
    ax3.set_title('MEC Connectivity Asymmetry', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_axisbelow(True)
    
    # Add fraction labels on bars
    for i, (bar, label) in enumerate(zip(bars, conn_labels)):
        height = bar.get_height()
        if 'PV' in label:
            frac = mec_conn['pv_fraction']
        elif 'GC' in label:
            frac = mec_conn['gc_fraction']
        else:
            frac = 0.0
        
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(conn_data)*0.02,
                f'{frac:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel D: Firing rate changes bar plot with error bars
    ax4 = plt.subplot(3, 4, (7, 8))
    
    change_data = []
    change_errors = []
    change_labels = []
    change_colors = []
    
    for target in targets:
        for pop in ['gc', 'mc']:
            if f'{pop}_mean_change' in results[target][stimulation_level]:
                mean_change = results[target][stimulation_level][f'{pop}_mean_change']
                
                change_data.append(mean_change)
                change_labels.append(f'{target.upper()}→{pop.upper()}')
                change_colors.append(colors['pv'] if target == 'pv' else colors['sst'])
                
                # Add error bars if available
                if has_multitrial and f'{pop}_mean_change_std' in results[target][stimulation_level]:
                    std = results[target][stimulation_level][f'{pop}_mean_change_std']
                    change_errors.append(std)
                else:
                    change_errors.append(0)
    
    if change_data:
        bars = ax4.bar(range(len(change_labels)), change_data, 
                      color=change_colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                      yerr=change_errors if has_multitrial else None,
                      capsize=5)
        
        # Add value labels
        for bar, value in zip(bars, change_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
        
        ax4.set_xticks(range(len(change_labels)))
        ax4.set_xticklabels(change_labels, fontsize=10)
        ax4.set_ylabel(r'$\Delta$ Firing Rate (Hz)', fontsize=11)
        title_suffix = f' (n={n_trials} trials)' if has_multitrial else ' (Single Trial)'
        ax4.set_title(f'Mean Rate Changes{title_suffix}', fontsize=12, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_axisbelow(True)

    # Panel E: Scatter plots showing correlation
    for i, target in enumerate(targets):
        ax = plt.subplot(3, 4, 9 + i * 2)
        
        opsin_expression = results[target][stimulation_level]['opsin_expression_mean']
        
        if has_multitrial:
            stim_rates = results[target][stimulation_level][f'{target}_stim_rates_mean'][opsin_expression <= 0.2]
            baseline_rates = results[target][stimulation_level][f'{target}_baseline_rates_mean'][opsin_expression <= 0.2]
        else:
            stim_rates = results[target][stimulation_level][f'{target}_stim_rates'][opsin_expression <= 0.2]
            baseline_rates = results[target][stimulation_level][f'{target}_baseline_rates'][opsin_expression <= 0.2]

        ax.scatter(baseline_rates, stim_rates, c=colors[target], alpha=0.6, s=30, 
                  edgecolors='black', linewidth=0.5)
        
        # Add correlation line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(baseline_rates, stim_rates)
        
        line = slope * baseline_rates + intercept
        ax.plot(baseline_rates, line, 'r--', alpha=0.8, linewidth=2, 
               label=f'Fit (R={r_value:.2f})')

        if hasattr(baseline_rates, 'numpy'):
            baseline_rates = baseline_rates.numpy()
        if hasattr(stim_rates, 'numpy'):
            stim_rates = stim_rates.numpy()
            
        # Identity line
        max_rate = max(np.max(baseline_rates), np.max(stim_rates))
        ax.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, linewidth=1.5, 
               label='Identity')
        
        ax.set_xlabel('Baseline Rate (Hz)', fontsize=10)
        ax.set_ylabel('Stimulation Rate (Hz)', fontsize=10)
        title_suffix = f'\n(n={n_trials} trials)' if has_multitrial else '\n(Single Trial)'
        ax.set_title(f'{target.upper()} Stimulation{title_suffix}\n(Non-expressing cells)', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)

    # Panel F: Summary text
    ax6 = plt.subplot(3, 4, 12)
    ax6.axis('off')
    
    trial_text = f'{n_trials} Trial Average' if has_multitrial else 'Single Trial'
    summary_text = f"{trial_text} Summary\n"
    summary_text += "=" * 30 + "\n\n"
    
    # MEC asymmetry
    summary_text += "MEC Connectivity:\n"
    summary_text += f"  MEC -> PV: {mec_conn['mec_to_pv']} ({mec_conn['pv_fraction']:.1%})\n"
    summary_text += f"  MEC -> SST: {mec_conn['mec_to_sst']} (none)\n\n"
    
    # Network effects
    summary_text += "Optogenetic Effects:\n"
    for target in targets:
        summary_text += f"{target.upper()} stimulation:\n"
        for pop in ['gc', 'mc']:
            if f'{pop}_excited' in results[target][stimulation_level]:
                excited = results[target][stimulation_level][f'{pop}_excited']
                if has_multitrial and f'{pop}_excited_std' in results[target][stimulation_level]:
                    std = results[target][stimulation_level][f'{pop}_excited_std']
                    summary_text += f"  {pop.upper()}: {excited:.1%} ± {std:.1%} excited\n"
                else:
                    summary_text += f"  {pop.upper()}: {excited:.1%} excited\n"
        summary_text += "\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, va='top', ha='left',
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
    main_title = 'Dentate Gyrus Interneuron Effects'
    if has_multitrial:
        main_title += f' (Average of {n_trials} Trials)'
    else:
        main_title += ' (Representative Single Trial)'
        
    plt.suptitle(main_title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        trial_suffix = f'_n{n_trials}trials' if has_multitrial else '_single_trial'
        plt.savefig(f"{save_path}/DG_comparative_experiment_stim_{stimulation_level}{trial_suffix}.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/DG_comparative_experiment_stim_{stimulation_level}{trial_suffix}.png", 
                   dpi=300, bbox_inches='tight')
    
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
    
    print("Testing Disinhibition Hypothesis")
    print("=" * 40)
    
    for target in ['pv', 'sst']:
        print(f"\n{target.upper()} Stimulation:")
        print("-" * 20)
        
        analysis = analyze_disinhibition_effects(
            experiment, target, 1.0,
            mec_current=mec_current,
            opsin_current=opsin_current
        )
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if f'{pop}_paradoxical_excitation' in analysis:
                result = analysis[f'{pop}_paradoxical_excitation']
                
                print(f"{pop.upper()}:")
                print(f"  With inhibition: {result['with_inhibition']} cells excited")
                print(f"  Reduced inhibition: {result['without_inhibition']} cells excited") 
                print(f"  Disinhibition-dependent: {result['disinhibition_dependent']} cells")
                print(f"  Mean change: {result['mean_change_full']:.3f} -> {result['mean_change_no_inh']:.3f}")
                print(f"  Change variability: {result['std_change_full']:.3f} -> {result['std_change_no_inh']:.3f}")
     

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
    save_results_file: Optional[str] = None
) -> Dict:
    """
    Test role of interneuron-interneuron interactions in paradoxical excitation
    
    This test blocks connections between interneurons (PV<->SST, PV<->PV, SST<->SST)
    while keeping excitatory inputs and outputs intact. If paradoxical excitation
    depends on interneuron interactions (e.g., disinhibition via PV->SST), blocking
    these connections should reduce the effect.
    
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
    
    print("\n" + "="*80)
    print("TEST: Interneuron-Interneuron Interaction Hypothesis")
    print("="*80)
    print("\nBlocking connections:")
    print("  - PV -> PV (prevents lateral inhibition within PV population)")
    print("  - PV -> SST (prevents PV-mediated disinhibition of SST)")
    print("  - SST -> PV (prevents SST-mediated disinhibition of PV)")
    print("  - SST -> SST (prevents lateral inhibition within SST population)")
    print("\nPrediction: If paradoxical excitation relies on interneuron")
    print("           interactions (e.g., PV->SST disinhibition), this")
    print("           manipulation should reduce the effect.")
    print("="*80 + "\n")
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    
    # Create modified synaptic parameters that block interneuron interactions
    base_synaptic_params = PerConnectionSynapticParams()
    
    blocked_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params.ampa_g_mean,
        ampa_g_std=base_synaptic_params.ampa_g_std,
        ampa_g_min=base_synaptic_params.ampa_g_min,
        ampa_g_max=base_synaptic_params.ampa_g_max,
        gaba_g_mean=base_synaptic_params.gaba_g_mean,
        gaba_g_std=base_synaptic_params.gaba_g_std,
        gaba_g_min=base_synaptic_params.gaba_g_min,
        gaba_g_max=base_synaptic_params.gaba_g_max,
        distribution=base_synaptic_params.distribution,
        connection_modulation={
            **base_synaptic_params.connection_modulation,
            # Block all interneuron-interneuron connections
            'pv_pv': 0.01,
            'pv_sst': 0.01,
            'sst_pv': 0.01,
            'sst_sst': 0.01,
        }
    )
    
    results = {
        'full_network': {},
        'blocked_int_int': {}
    }
    
    for target in target_populations:
        print(f"\n{'='*60}")
        print(f"Testing {target.upper()} stimulation")
        print('='*60)
        
        # Run full network (control)
        print("\n1. Full Network (Control)")
        print("-"*60)
        experiment_full = OptogeneticExperiment(
            circuit_params, base_synaptic_params, opsin_params,
            optimization_json_file=optimization_json_file,
            device=device,
            base_seed=base_seed
        )
        
        full_results = {}
        for intensity in intensities:
            print(f"\n  Testing intensity: {intensity}")
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
                
            full_results[intensity] = analysis
        
        results['full_network'][target] = full_results
        
        # Run with blocked interneuron interactions
        print("\n2. Blocked Interneuron-Interneuron Connections")
        print("-"*60)
        experiment_blocked = OptogeneticExperiment(
            circuit_params, blocked_synaptic_params, opsin_params,
            optimization_json_file=optimization_json_file,
            device=device,
            base_seed=base_seed + 1000  # Different connectivity
        )
        
        blocked_results = {}
        for intensity in intensities:
            print(f"\n  Testing intensity: {intensity}")
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
        print(f"\n{'='*60}")
        print(f"SUMMARY: {target.upper()} stimulation at intensity {intensities[-1]}")
        print('='*60)
        print(f"{'Population':<12} {'Full Network':<20} {'Blocked Int-Int':<20} {'Change':<15}")
        print('-'*60)
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop == target:
                continue
            full_exc = full_results[intensities[-1]][f'{pop}_excited']
            blocked_exc = blocked_results[intensities[-1]][f'{pop}_excited']
            change = blocked_exc - full_exc
            print(f"{pop.upper():<12} {full_exc:>6.1%}              {blocked_exc:>6.1%}              {change:>+6.1%}")
    
    # Save results if requested
    if save_results_file:
        import pickle
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {save_results_file}")
    
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
    save_results_file: Optional[str] = None
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
    
    print("\n" + "="*80)
    print("TEST: Disinhibition Hypothesis (Clean Test)")
    print("="*80)
    print("\nBlocking connections:")
    print("  - MEC -> PV (blocks external excitation to PV)")
    print("  - GC -> PV (blocks local excitation to PV)")
    print("  - MC -> PV (blocks distant excitation to PV)")
    print("  - GC -> SST (blocks local excitation to SST)")
    print("  - MC -> SST (blocks distant excitation to SST)")
    print("\nKept intact:")
    print("  - All excitation to principal cells (GC, MC)")
    print("  - All inhibition from interneurons")
    print("  - Direct optogenetic activation of target interneurons")
    print("\nPrediction: If paradoxical excitation requires excitatory")
    print("           recruitment of the inhibitory network (disinhibition),")
    print("           this should eliminate the effect.")
    print("="*80 + "\n")
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    
    base_synaptic_params = PerConnectionSynapticParams()
    
    blocked_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params.ampa_g_mean,
        ampa_g_std=base_synaptic_params.ampa_g_std,
        ampa_g_min=base_synaptic_params.ampa_g_min,
        ampa_g_max=base_synaptic_params.ampa_g_max,
        gaba_g_mean=base_synaptic_params.gaba_g_mean,
        gaba_g_std=base_synaptic_params.gaba_g_std,
        gaba_g_min=base_synaptic_params.gaba_g_min,
        gaba_g_max=base_synaptic_params.gaba_g_max,
        distribution=base_synaptic_params.distribution,
        connection_modulation={
            **base_synaptic_params.connection_modulation,
            # Block all excitation TO interneurons
            'mec_pv': 0.01,
            'gc_pv': 0.01,
            'mc_pv': 0.01,
            'gc_sst': 0.01,
            'mc_sst': 0.01,
        }
    )
    
    results = {
        'full_network': {},
        'blocked_exc_to_int': {}
    }
    
    for target in target_populations:
        print(f"\n{'='*60}")
        print(f"Testing {target.upper()} stimulation")
        print('='*60)
        
        # Run full network (control)
        print("\n1. Full Network (Control)")
        print("-"*60)
        experiment_full = OptogeneticExperiment(
            circuit_params, base_synaptic_params, opsin_params,
            optimization_json_file=optimization_json_file,
            device=device,
            base_seed=base_seed
        )
        
        full_results = {}
        for intensity in intensities:
            print(f"\n  Testing intensity: {intensity}")
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
                    continue
                baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                rate_change = stim_rate - baseline_rate
                baseline_std = torch.std(baseline_rate)
                
                excited_fraction = torch.mean((rate_change > baseline_std).float())
                
                analysis[f'{pop}_excited'] = excited_fraction.item()
                analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()
                
            full_results[intensity] = analysis
        
        results['full_network'][target] = full_results
        
        # Run with blocked excitation to interneurons
        print("\n2. Blocked Excitation to Interneurons")
        print("-"*60)
        experiment_blocked = OptogeneticExperiment(
            circuit_params, blocked_synaptic_params, opsin_params,
            optimization_json_file=optimization_json_file,
            device=device,
            base_seed=base_seed + 1000
        )
        
        blocked_results = {}
        for intensity in intensities:
            print(f"\n  Testing intensity: {intensity}")
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
        print(f"\n{'='*60}")
        print(f"SUMMARY: {target.upper()} stimulation at intensity {intensities[-1]}")
        print('='*60)
        print(f"{'Population':<12} {'Full Network':<20} {'Blocked Exc->Int':<20} {'Change':<15}")
        print('-'*60)
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop == target:
                continue
            full_exc = full_results[intensities[-1]][f'{pop}_excited']
            blocked_exc = blocked_results[intensities[-1]][f'{pop}_excited']
            change = blocked_exc - full_exc
            
            print(f"{pop.upper():<12} {full_exc:>6.1%}              {blocked_exc:>6.1%}              {change:>+6.1%}")
            
            if abs(change) > 0.05:  # Significant reduction
                print(f"             → {'STRONG' if abs(change) > 0.15 else 'MODERATE'} reduction suggests disinhibition mechanism")
    
    if save_results_file:
        import pickle
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {save_results_file}")
    
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
    save_results_file: Optional[str] = None
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
    
    print("\n" + "="*80)
    print("TEST: Recurrent Excitation Hypothesis")
    print("="*80)
    print("\nBlocking connections:")
    print("  - GC -> MC (blocks feedforward excitation)")
    print("  - MC -> GC (blocks feedback excitation)")
    print("  - MC -> MC (blocks lateral excitation within MC)")
    print("\nKept intact:")
    print("  - All connections involving interneurons")
    print("  - External drive (MEC -> GC, MEC -> MC)")
    print("\nPrediction: If paradoxical excitation depends on amplification")
    print("           through recurrent excitatory loops among principal")
    print("           cells, blocking these connections should reduce")
    print("           the effect.")
    print("="*80 + "\n")
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    
    base_synaptic_params = PerConnectionSynapticParams()
    
    blocked_synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params.ampa_g_mean,
        ampa_g_std=base_synaptic_params.ampa_g_std,
        ampa_g_min=base_synaptic_params.ampa_g_min,
        ampa_g_max=base_synaptic_params.ampa_g_max,
        gaba_g_mean=base_synaptic_params.gaba_g_mean,
        gaba_g_std=base_synaptic_params.gaba_g_std,
        gaba_g_min=base_synaptic_params.gaba_g_min,
        gaba_g_max=base_synaptic_params.gaba_g_max,
        distribution=base_synaptic_params.distribution,
        connection_modulation={
            **base_synaptic_params.connection_modulation,
            # Block recurrent excitation among principal cells
            'gc_mc': 0.01,
            'mc_gc': 0.01,
            'mc_mc': 0.01,
        }
    )
    
    results = {
        'full_network': {},
        'blocked_recurrent': {}
    }
    
    for target in target_populations:
        print(f"\n{'='*60}")
        print(f"Testing {target.upper()} stimulation")
        print('='*60)
        
        # Run full network (control)
        print("\n1. Full Network (Control)")
        print("-"*60)
        experiment_full = OptogeneticExperiment(
            circuit_params, base_synaptic_params, opsin_params,
            optimization_json_file=optimization_json_file,
            device=device,
            base_seed=base_seed
        )
        
        full_results = {}
        for intensity in intensities:
            print(f"\n  Testing intensity: {intensity}")
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
                    continue
                baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                rate_change = stim_rate - baseline_rate
                baseline_std = torch.std(baseline_rate)
                
                excited_fraction = torch.mean((rate_change > baseline_std).float())
                
                analysis[f'{pop}_excited'] = excited_fraction.item()
                analysis[f'{pop}_mean_change'] = torch.mean(rate_change).item()
                
            full_results[intensity] = analysis
        
        results['full_network'][target] = full_results
        
        # Run with blocked recurrent excitation
        print("\n2. Blocked Recurrent Excitation")
        print("-"*60)
        experiment_blocked = OptogeneticExperiment(
            circuit_params, blocked_synaptic_params, opsin_params,
            optimization_json_file=optimization_json_file,
            device=device,
            base_seed=base_seed + 1000
        )
        
        blocked_results = {}
        for intensity in intensities:
            print(f"\n  Testing intensity: {intensity}")
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
        print(f"\n{'='*60}")
        print(f"SUMMARY: {target.upper()} stimulation at intensity {intensities[-1]}")
        print('='*60)
        print(f"{'Population':<12} {'Full Network':<20} {'Blocked Recurrent':<20} {'Change':<15}")
        print('-'*60)
        
        for pop in ['gc', 'mc']:  # Focus on principal cells for this test
            full_exc = full_results[intensities[-1]][f'{pop}_excited']
            blocked_exc = blocked_results[intensities[-1]][f'{pop}_excited']
            change = blocked_exc - full_exc
            
            print(f"{pop.upper():<12} {full_exc:>6.1%}              {blocked_exc:>6.1%}              {change:>+6.1%}")
            
            if abs(change) > 0.05:
                print(f"             → Recurrent excitation {'amplifies' if change < 0 else 'suppresses'} paradoxical response")
    
    if save_results_file:
        import pickle
        with open(save_results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {save_results_file}")
    
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
    output_dir: str = "./ablation_tests"
) -> Dict:
    """
    Run all three ablation tests and create summary comparison
    
    Args:
        optimization_json_file: Path to optimization results (optional)
        target_populations: List of populations to stimulate
        intensities: List of light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: When to start stimulation (ms)
        stim_duration: Duration of stimulation (ms)
        warmup: Pre-stimulation period (ms)
        device: Device to run on (None for auto-detect)
        n_trials: Number of trials to average
        base_seed: Base random seed
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all test results
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("ABLATION TEST SUITE")
    print("="*80)
    print(f"\nTesting {len(target_populations)} populations with {n_trials} trials each")
    print(f"Results will be saved to: {output_dir}")
    print("="*80)
    
    all_results = {}
    
    # Test 1: Interneuron-interneuron interactions
    print("\n\n" + "#"*80)
    print("# TEST 1/3: Interneuron-Interneuron Interactions")
    print("#"*80)
    results_int_int = test_interneuron_interactions(
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
        n_trials=n_trials,
        base_seed=base_seed,
        save_results_file=str(output_path / "test1_int_int.pkl")
    )
    all_results['interneuron_interactions'] = results_int_int
    
    # Test 2: Excitation to interneurons (key disinhibition test)
    print("\n\n" + "#"*80)
    print("# TEST 2/3: Excitation to Interneurons (Disinhibition Test)")
    print("#"*80)
    results_exc_int = test_excitation_to_interneurons(
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
        n_trials=n_trials,
        base_seed=base_seed + 2000,
        save_results_file=str(output_path / "test2_exc_to_int.pkl")
    )
    all_results['excitation_to_interneurons'] = results_exc_int
    
    # Test 3: Recurrent excitation
    print("\n\n" + "#"*80)
    print("# TEST 3/3: Recurrent Excitation")
    print("#"*80)
    results_recurrent = test_recurrent_excitation(
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
        n_trials=n_trials,
        base_seed=base_seed + 4000,
        save_results_file=str(output_path / "test3_recurrent.pkl")
    )
    all_results['recurrent_excitation'] = results_recurrent
    
    # Create summary comparison
    print("\n\n" + "="*80)
    print("ABLATION TEST SUMMARY")
    print("="*80)
    
    intensity = intensities[-1]
    
    for target in target_populations:
        print(f"\n{target.upper()} Stimulation at intensity {intensity}")
        print("-"*80)
        print(f"{'Manipulation':<30} {'GC Excited':<15} {'MC Excited':<15} {'Effect':<15}")
        print("-"*80)
        
        # Full network baseline
        full_gc = results_exc_int['full_network'][target][intensity]['gc_excited']
        full_mc = results_exc_int['full_network'][target][intensity]['mc_excited']
        print(f"{'Full Network (baseline)':<30} {full_gc:>6.1%}          {full_mc:>6.1%}          -")
        
        # Test 1: Block interneuron interactions
        test1_gc = results_int_int['blocked_int_int'][target][intensity]['gc_excited']
        test1_mc = results_int_int['blocked_int_int'][target][intensity]['mc_excited']
        change1_gc = (test1_gc - full_gc) / (full_gc + 1e-6)
        change1_mc = (test1_mc - full_mc) / (full_mc + 1e-6)
        print(f"{'Block Int-Int':<30} {test1_gc:>6.1%}          {test1_mc:>6.1%}          {(change1_gc+change1_mc)/2:>+6.1%}")
        
        # Test 2: Block excitation to interneurons
        test2_gc = results_exc_int['blocked_exc_to_int'][target][intensity]['gc_excited']
        test2_mc = results_exc_int['blocked_exc_to_int'][target][intensity]['mc_excited']
        change2_gc = (test2_gc - full_gc) / (full_gc + 1e-6)
        change2_mc = (test2_mc - full_mc) / (full_mc + 1e-6)
        print(f"{'Block Exc->Int (KEY TEST)':<30} {test2_gc:>6.1%}          {test2_mc:>6.1%}          {(change2_gc+change2_mc)/2:>+6.1%}")
        
        # Test 3: Block recurrent excitation
        test3_gc = results_recurrent['blocked_recurrent'][target][intensity]['gc_excited']
        test3_mc = results_recurrent['blocked_recurrent'][target][intensity]['mc_excited']
        change3_gc = (test3_gc - full_gc) / (full_gc + 1e-6)
        change3_mc = (test3_mc - full_mc) / (full_mc + 1e-6)
        print(f"{'Block Recurrent':<30} {test3_gc:>6.1%}          {test3_mc:>6.1%}          {(change3_gc+change3_mc)/2:>+6.1%}")
        
        print("\nInterpretation:")
        avg_change2 = (change2_gc + change2_mc) / 2
        if avg_change2 < -0.5:
            print("  ✓ STRONG support for disinhibition hypothesis")
            print("    (>50% reduction when blocking exc. to interneurons)")
        elif avg_change2 < -0.3:
            print("  ✓ MODERATE support for disinhibition hypothesis")
            print("    (30-50% reduction when blocking exc. to interneurons)")
        else:
            print("  ✗ WEAK support for disinhibition hypothesis")
            print("    (<30% reduction suggests other mechanisms)")
    
    # Save combined results
    combined_file = output_path / "all_ablation_tests.pkl"
    import pickle
    with open(combined_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n\nAll results saved to: {combined_file}")
    
    return all_results



if __name__ == "__main__":
    print("PyTorch Dentate Gyrus Circuit with Anatomical Connectivity")
    print("=========================================================")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='DG Optogenetic Protocol with Multi-Trial Support')
    parser.add_argument('--optimization-file', type=str, default=None,
                       help='Path to optimization JSON file')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', None],
                       help='Device to run on (default: auto-detect)')
    parser.add_argument('--n-trials', type=int, default=3,
                       help='Number of trials to average (default: 3)')
    parser.add_argument('--regenerate-connectivity', action='store_true',
                        help='Regenerate circuit connectivity for each trial (default: False, reuse same circuit)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--mec-current', type=float, default=40.0,
                        help='MEC drive current in pA (default: 40.0)')
    parser.add_argument('--opsin-current', type=float, default=200.0,
                       help='Optogenetic current in pA (default: 200.0)')
    parser.add_argument('--stim-start', type=float, default=1500.0,
                        help='Optogenetic stimulus start time [ms] (default: 1500.0)')
    parser.add_argument('--stim-duration', type=float, default=1000.0,
                        help='Optogenetic stimulus duration [ms] (default: 1000.0)')
    parser.add_argument('--load-results', type=str, default=None,
                       help='Load results from file instead of running experiment')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save results to specific file')
    parser.add_argument('--no-auto-save', action='store_true',
                       help='Disable automatic saving of results')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot results (requires --load-results)')
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
    parser.add_argument('--time-varying-mec', action='store_true',
                       help='Enable time-varying MEC input')
    parser.add_argument('--mec-pattern-type', type=str, default='oscillatory',
                       choices=['oscillatory', 'drift', 'constant'],
                       help='Type of temporal pattern for MEC input')
    parser.add_argument('--mec-theta-freq', type=float, default=5.0,
                       help='Theta oscillation frequency (Hz)')
    parser.add_argument('--mec-theta-amplitude', type=float, default=0.3,
                       help='Theta modulation depth (0-1)')
    parser.add_argument('--mec-gamma-freq', type=float, default=20.0,
                       help='Gamma oscillation frequency (Hz)')
    parser.add_argument('--mec-gamma-amplitude', type=float, default=0.15,
                       help='Gamma modulation depth (0-1)')
    parser.add_argument('--mec-rotation-groups', type=int, default=3,
                       help='Number of groups for spatial rotation')
    parser.add_argument('--mec-gamma-coupling', type=float, default=0.8,
                        help='Gamma-theta coupling strength (0=independent, 1=fully coupled)')
    parser.add_argument('--mec-gamma-phase', type=float, default=0.0,
                        help='Preferred theta phase for gamma peak (radians, 0=peak, pi=trough)')
    
    
    args = parser.parse_args()

    # Validate arguments
    if args.plot_only and args.load_results is None:
        parser.error("--plot-only requires --load-results")
    
    # Auto-detect device or use specified
    if args.device is None:
        device = get_default_device()
    else:
        device = torch.device(args.device)

    adaptive_config = None
    if args.adaptive_step:
        adaptive_config = GradientAdaptiveStepConfig(
            dt_min=args.adaptive_dt_min,
            dt_max=args.adaptive_dt_max,
            gradient_low=args.adaptive_gradient_low,
            gradient_high=args.adaptive_gradient_high,
        )

    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    print(f"Number of trials: {args.n_trials}")
    print(f"Base seed: {args.base_seed}")

    output_dir = "protocol"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Run comparative experiment with multi-trial averaging
    results, connectivity_analysis, conductance_analysis = run_comparative_experiment(
        optimization_json_file=args.optimization_file,
        intensities=[0.5, 1.0, 1.5],
        mec_current=args.mec_current,
        opsin_current=args.opsin_current,
        device=device,
        n_trials=args.n_trials,
        regenerate_connectivity_per_trial=args.regenerate_connectivity,
        base_seed=args.base_seed,
        stim_start=args.stim_start,
        stim_duration=args.stim_duration,
        load_results_file=args.load_results,
        save_results_file=args.save_results,
        auto_save=not args.no_auto_save,
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

    # Print results with multi-trial statistics
    if not args.plot_only:
        mec_conn = connectivity_analysis['mec_connectivity']
        print(f"\nMEC -> PV connections: {mec_conn['mec_to_pv']} ({mec_conn['pv_fraction']:.3f})")
        print(f"MEC -> GC connections: {mec_conn['mec_to_gc']} ({mec_conn['gc_fraction']:.3f})")
        print(f"MEC -> MC connections: {mec_conn['mec_to_mc']}")
        print(f"MEC -> SST connections: {mec_conn['mec_to_sst']}")

        for target in ['pv', 'sst']:
            print(f"\n{target.upper()} Stimulation Results (Average of {args.n_trials} trials):")
            print("-" * 50)

            for intensity in [0.5, 1.0, 1.5]:
                analysis = results[target][intensity]
                print(f"\nIntensity {intensity}:")

                for pop in ['gc', 'mc', 'pv', 'sst']:
                    if f'{pop}_excited' in analysis:
                        excited = analysis[f'{pop}_excited']
                        excited_std = analysis.get(f'{pop}_excited_std', 0.0)
                        inhibited = analysis[f'{pop}_inhibited'] 
                        mean_change = analysis[f'{pop}_mean_change']
                        mean_change_std = analysis.get(f'{pop}_mean_change_std', 0.0)
                        mean_stim_rate = analysis[f'{pop}_mean_stim_rate']
                        mean_baseline_rate = analysis[f'{pop}_mean_baseline_rate']

                        print(f"  {pop.upper()}:")
                        print(f"    Excited: {excited:.2f} +/- {excited_std:.2f}")
                        print(f"    Inhibited: {inhibited:.2f}")
                        print(f"    Rate: {mean_baseline_rate:.2f} -> {mean_stim_rate:.2f} Hz")
                        print(f"    Change: {mean_change:.3f} +/- {mean_change_std:.3f} Hz")


    # Plot adaptive stepping analysis if adaptive stepping was used
    #if args.adaptive_step and 'adaptive_stats' in results['pv'][1.0]:
    #    plot_adaptive_stepping_analysis(
    #        results['pv'][1.0]['adaptive_stats'],
    #        stim_start=args.stim_start,
    #        stim_duration=args.stim_duration,
    #        save_path="protocol/DG_adaptive_stepping_analysis.png"
    #    )

    # Plot results with multi-trial statistics
    plot_comparative_experiment_results(results, connectivity_analysis, stimulation_level=0.5, save_path="protocol")
    plot_comparative_experiment_results(results, connectivity_analysis, stimulation_level=1.0, save_path="protocol")
    plot_comparative_experiment_results(results, connectivity_analysis, stimulation_level=1.5, save_path="protocol")

    
    run_all_ablation_tests(
        optimization_json_file=args.optimization_file,
        intensities=[1.0],
        mec_current=args.mec_current,
        opsin_current=args.opsin_current,
        stim_start=args.stim_start,
        stim_duration=args.stim_duration,
        device=device,
        n_trials=args.n_trials,
        base_seed=args.base_seed,
        output_dir="./ablation_tests")
