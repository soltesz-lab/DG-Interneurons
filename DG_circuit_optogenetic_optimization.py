#!/usr/bin/env python3
"""
Optimization framework including optogenetic stimulation effects

- Circuits are recreated for each trial with different seeds
- Seed strategy: base_seed + trial_index ensures reproducibility
- Maintains batch parallelism within each trial

EvaluationStrategy pattern for automatic device-appropriate strategy selection.
"""

import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import tqdm
import json
from datetime import datetime
import multiprocessing as mp

from DG_protocol import OpsinExpression, generate_time_varying_mec_pattern


# Import existing optimization components
from DG_circuit_optimization import (
    OptimizationTargets,
    OptimizationConfig,
    evaluate_rate_ordering_constraints,
    configure_torch_threads,
    get_default_device,
    create_default_targets,
    BatchCircuitEvaluator
)

from DG_batch_circuit_dendritic_somatic_transfer import (
    BatchDentateCircuit, update_batch_circuit_with_adaptive_dt
)

from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit, CircuitParams, OpsinParams, PerConnectionSynapticParams
)

from gradient_adaptive_stepper import (
    GradientAdaptiveStepConfig,
    GradientAdaptiveStepper,
    AdaptiveSimulationState,
    update_circuit_with_adaptive_dt
)

# Store numpy.ndarray as JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    
# ============================================================================
# Random Seed Management
# ============================================================================

def set_random_seed(seed: int, device: Optional[torch.device] = None):
    """
    Set random seeds for reproducible connectivity generation
    
    Args:
        seed: Random seed value
        device: Device to set CUDA seed for (if applicable)
    
    Note:
        This affects torch and numpy random number generators, which are
        used in SpatialLayout and ConnectivityMatrix initialization.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device is not None and device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Optogenetic Targets
# ============================================================================

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

    pre_stim_duration: float = 1000.0
    stim_duration: float = 1000.0
    
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


# ============================================================================
# Batch Optogenetic Evaluator
# ============================================================================

class BatchOptogeneticEvaluator:
    """
    Evaluates multiple parameter configurations with optogenetic experiments in parallel
    
    Each trial now uses different connectivity seeds for circuit generation
    
    Seeding Strategy:
    - Base seed can be provided or defaults to 42
    - Each trial uses: base_seed + trial_index
    - This ensures reproducibility while capturing connectivity variability
    
    Extends batch circuit evaluation to include optogenetic stimulation protocols.
    Computes both baseline circuit objectives and optogenetic effect objectives.
    """
    
    def __init__(self,
                 circuit_params: CircuitParams,
                 base_synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 opsin_current: float,
                 targets: CombinedOptimizationTargets,
                 config: OptimizationConfig,
                 device: Optional[torch.device] = None,
                 base_seed: int = 42,
                 adaptive_step: bool = False,  
                 adaptive_config: Optional[GradientAdaptiveStepConfig] = None,
                 verbose: bool = False,
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
                 mec_rotation_groups: int = 3):

        """
        Initialize batch optogenetic evaluator
        
        Args:
            circuit_params: CircuitParams instance
            base_synaptic_params: PerConnectionSynapticParams instance
            opsin_params: OpsinParams instance
            opsin_current: Maximum optogenetic current (pA)
            targets: CombinedOptimizationTargets instance
            config: OptimizationConfig instance
            device: Device to run simulations on
            base_seed: Base random seed for reproducibility (default: 42)
            adaptive_step: If True, use adaptive time stepping
            adaptive_config: Configuration for adaptive stepping (optional)
            use_time_varying_mec: If True, use time-varying MEC patterns (default: False)
            mec_pattern_type: Type of temporal pattern ('oscillatory', 'drift', 'constant')
            mec_theta_freq: Theta oscillation frequency (Hz)
            mec_theta_amplitude: Theta modulation depth (0-1)
            mec_gamma_freq: Gamma oscillation frequency (Hz)
            mec_gamma_amplitude: Gamma modulation depth (0-1)
            mec_gamma_coupling_strength: Gamma-theta coupling (0=independent, 1=fully coupled)
            mec_gamma_preferred_phase: Preferred theta phase for gamma peak (radians)
            mec_drift_timescale: Correlation time for drift pattern (ms)
            mec_drift_amplitude: Drift amplitude relative to base (0-1)
            mec_rotation_groups: Number of groups for spatial rotation across trials
        """
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.opsin_current = opsin_current
        self.targets = targets
        self.config = config
        self.device = device if device is not None else get_default_device()
        self.base_seed = base_seed
        self.logger = logging.getLogger("DG_eval")
        # Create default adaptive config if needed
        self.adaptive_step = adaptive_step
        if self.adaptive_step and adaptive_config is None:
            self.adaptive_config = GradientAdaptiveStepConfig(
                dt_min=0.05,
                dt_max=0.3,
                gradient_low=0.5,
                gradient_high=10.0
            )
        else:
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
            
        if verbose:
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            self.logger.addHandler(ch)
            
        self.logger.info(f"BatchOptogeneticEvaluator initialized on device: {self.device}")
        self.logger.info(f"  Base seed: {base_seed} (trials will use base_seed + trial_index)")
        if self.use_time_varying_mec:
            self.logger.info(f"  MEC pattern: {mec_pattern_type} (theta={mec_theta_freq}Hz, gamma={mec_gamma_freq}Hz)")


    def _generate_mec_input(self,
                            batch_size: int,
                            duration: float,
                            dt: float,
                            mec_current: float,
                            mec_current_std: float,
                            trial_index: int) -> torch.Tensor:
        """
        Generate MEC input pattern for a trial

        Uses generate_time_varying_mec_pattern from DG_protocol for consistency.

        Args:
            batch_size: Number of parameter configurations in batch
            duration: Total simulation duration (ms)
            dt: Time step (ms)
            mec_current: Base MEC current (pA)
            mec_current_std: Standard deviation for noise (pA)
            trial_index: Current trial index for rotation

        Returns:
            MEC input tensor:
            - If use_time_varying_mec: [batch_size, n_mec, n_steps]
            - Otherwise: [batch_size, n_mec] constant + noise
        """
        if self.use_time_varying_mec:
            # Generate time-varying pattern using shared function
            n_steps = int(duration / dt)

            # Generate pattern for each batch element with trial rotation
            # Each batch element gets the same temporal pattern but different spatial rotation
            mec_patterns = []
            for b in range(batch_size):
                pattern = generate_time_varying_mec_pattern(
                    n_mec=self.circuit_params.n_mec,
                    duration=duration,
                    dt=dt,
                    base_current=mec_current,
                    trial_index=trial_index + b,  # Offset by batch index for variation
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
                )
                mec_patterns.append(pattern)

            # Stack: [batch_size, n_mec, n_steps]
            return torch.stack(mec_patterns, dim=0)
        else:
            # Original constant + noise approach
            mec_input = mec_current + torch.randn(
                batch_size, 
                self.circuit_params.n_mec,
                device=self.device
            ) * mec_current_std
            return mec_input

    def _create_opsin_activation(self, 
                                 target_pop: str,
                                 layout) -> torch.Tensor:
        """
        Create opsin activation currents for target population

        Uses the same activation logic as DG_protocol.OpsinExpression.calculate_activation
        to ensure consistency between optimization and experimental protocols.

        Args:
            target_pop: Target population name ('pv', 'sst', etc.)
            layout: SpatialLayout instance with cell positions

        Returns:
            Activation currents for target population [n_cells]
        """
        # Create opsin expression for target population
        n_target_cells = getattr(self.circuit_params, f'n_{target_pop}')
        opsin_expression = OpsinExpression(
            self.opsin_params,
            n_cells=n_target_cells,
            device=self.device
        )

        # Get positions for target population
        target_positions = layout.positions[target_pop]

        # Calculate activation probability using shared logic
        light_intensity = self.targets.optogenetic_targets.stimulation_intensity
        activation_prob = opsin_expression.calculate_activation(target_positions, light_intensity)

        # Convert activation probability to optogenetic current
        target_opto_current = activation_prob * self.opsin_current

        return target_opto_current
    

        
    def evaluate_parameter_batch(self,
                                 parameter_batch: List[Dict[str, float]],
                                 mec_current: float,
                                 mec_current_std: float = 1.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Evaluate a batch of parameter configurations with full optogenetic protocol
        
        Circuits are recreated for each trial with different seeds
        
        Args:
            parameter_batch: List of connection_modulation dicts, length = batch_size
            mec_current: MEC drive level (pA)
            
        Returns:
            losses: Tensor of shape [batch_size] with total loss for each configuration
            details: Dict with baseline_losses, opto_losses, and firing_rates_batch
        """
        batch_size = len(parameter_batch)
        
        
        # Optogenetic evaluation
        opto_losses, opto_details = self._evaluate_optogenetic_batch(
            parameter_batch, mec_current, mec_current_std
        )

        # Average baseline rates from PV and SST experiments
        baseline_rates = {}
        for pop in ['gc', 'mc', 'pv', 'sst']:
            pv_baseline = opto_details['pv'].get(pop, {}).get('baseline_mean', None)
            sst_baseline = opto_details['sst'].get(pop, {}).get('baseline_mean', None)
            print(f"population {pop}: pv_baseline = {pv_baseline} sst_baseline = {sst_baseline}")
            # Average the two baselines if both available
            if pv_baseline is not None and sst_baseline is not None:
                baseline_rates[pop] = (pv_baseline + sst_baseline) / 2.0
            elif pv_baseline is not None:
                baseline_rates[pop] = pv_baseline
            elif sst_baseline is not None:
                baseline_rates[pop] = sst_baseline

        # Calculate baseline loss
        baseline_losses = self._calculate_baseline_loss_from_rates(baseline_rates)

        # Combine losses
        total_losses = (self.targets.baseline_weight * baseline_losses + 
                        self.targets.optogenetic_weight * opto_losses)
        
        return total_losses, {
            'baseline_losses': baseline_losses,
            'opto_losses': opto_losses,
            'baseline_rates': baseline_rates,
            'opto_details': opto_details
        }

    def _calculate_baseline_loss_from_rates(self, 
                                            baseline_rates: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate baseline loss from firing rates

        Args:
            baseline_rates: Dict mapping pop -> rates [batch_size]

        Returns:
            losses: Tensor [batch_size]
        """
        batch_size = list(baseline_rates.values())[0].shape[0]
        losses = torch.zeros(batch_size, device=self.device)

        # Rate loss
        for pop, rates in baseline_rates.items():
            if pop in self.targets.target_rates:
                target_rate = self.targets.target_rates[pop]
                tolerance = self.targets.rate_tolerance[pop]

                errors = torch.abs(rates - target_rate)

                # Huber loss with tolerance
                rate_losses = torch.where(
                    errors <= tolerance,
                    0.5 * errors ** 2,
                    tolerance * errors - 0.5 * tolerance ** 2
                )

                # Zero rate penalty
                zero_mask = torch.isclose(rates, 
                                          torch.tensor(0.0, device=self.device),
                                          atol=1e-2, rtol=1e-2)
                rate_losses = torch.where(zero_mask, 
                                          torch.tensor(1e2, device=self.device),
                                          rate_losses)

                losses += rate_losses

        # Add constraint violations
        for b in range(batch_size):
            firing_rates_dict = {pop: baseline_rates[pop][b].item()
                               for pop in baseline_rates}
            constraint_violation, _ = evaluate_rate_ordering_constraints(
                firing_rates_dict,
                self.targets.rate_ordering_constraints
            )
            losses[b] += self.targets.constraint_violation_weight * constraint_violation

        return losses    
    
    def _evaluate_baseline_batch(self,
                                 parameter_batch: List[Dict[str, float]],
                                 mec_current: float,
                                 mec_current_std: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate baseline circuit objectives for batch of parameters
        
        Dispatches to fixed-dt or adaptive-dt implementation based on configuration.
        """
        if self.adaptive_step:
            return self._evaluate_baseline_batch_adaptive(parameter_batch, mec_current, mec_current_std)
        else:
            return self._evaluate_baseline_batch_fixed(parameter_batch, mec_current, mec_current_std)

    
    def _evaluate_baseline_batch_fixed(self,
                                       parameter_batch: List[Dict[str, float]],
                                       mec_current: float,
                                       mec_current_std: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate baseline circuit objectives for batch of parameters
        
        Creates new circuit for each trial with different seed
        
        Returns:
            losses: Tensor [batch_size]
            firing_rates_batch: Dict mapping pop -> rates [batch_size]
        """
        batch_size = len(parameter_batch)
        
        # Average over trials
        total_losses = torch.zeros(batch_size, device=self.device)
        all_firing_rates = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        duration = self.config.warmup_duration + self.config.simulation_duration
        dt = self.circuit_params.dt
        n_steps = int(duration / dt)
        for trial in range(self.config.n_trials):
            # Set seed for this trial (base_seed + trial_index)
            trial_seed = self.base_seed + trial
            set_random_seed(trial_seed, self.device)
            
            self.logger.info(f"  Baseline trial {trial + 1}/{self.config.n_trials}: Creating circuit with seed {trial_seed}...")
            
            # Create circuit for this trial with fresh connectivity
            circuit = BatchDentateCircuit(
                batch_size=batch_size,
                circuit_params=self.circuit_params,
                synaptic_params=self.base_synaptic_params,
                opsin_params=self.opsin_params,
                device=self.device
            )
            
            # Set per-batch connection modulation
            circuit.set_connection_modulation_batch(parameter_batch)

            print(f"Circuit connectivity hash: {hash(circuit.connectivity.conductance_matrices['gc_mc'].connectivity.cpu().numpy().tobytes())}")
            
            # Reset state (activities, but connectivity is already new)
            circuit.reset_state()


            # MEC input: [batch_size, n_mec]
            mec_input = self._generate_mec_input(
                batch_size=batch_size,
                duration=duration,
                dt=dt,
                mec_current=mec_current,
                mec_current_std=mec_current_std,
                trial_index=trial
            )
            
            # Collect activities over time
            activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}

            warmup_steps = int(self.config.warmup_duration / dt)
            
            for t in range(n_steps):
                # Get MEC drive for this timestep
                if self.use_time_varying_mec:
                    # Extract current timestep: [batch_size, n_mec]
                    mec_drive = mec_input[:, :, t]
                else:
                    # Constant input (already [batch_size, n_mec])
                    mec_drive = mec_input

                external_drive = {'mec': mec_drive}
                activities = circuit({}, external_drive)
                
                if t >= warmup_steps:
                    for pop in activities_over_time:
                        if pop in activities:
                            activities_over_time[pop].append(activities[pop])
            
            # Calculate losses for this trial
            trial_losses = torch.zeros(batch_size, device=self.device)
            trial_firing_rates = {}
            
            for pop in activities_over_time:
                if len(activities_over_time[pop]) > 0:
                    # Stack time series: [time, batch, neurons]
                    pop_time_series = torch.stack(activities_over_time[pop], dim=0)
                    # Mean over time: [batch, neurons]
                    mean_rates = torch.mean(pop_time_series, dim=0)
                    
                    # Population average per batch: [batch]
                    pop_firing_rates = torch.mean(mean_rates, dim=1)
                    trial_firing_rates[pop] = pop_firing_rates
                    
                    # Rate loss
                    if pop in self.targets.target_rates:
                        target_rate = self.targets.target_rates[pop]
                        tolerance = self.targets.rate_tolerance[pop]
                        
                        errors = torch.abs(pop_firing_rates - target_rate)
                        
                        # Huber loss with tolerance
                        rate_losses = torch.where(
                            errors <= tolerance,
                            0.5 * errors ** 2,
                            tolerance * errors - 0.5 * tolerance ** 2
                        )
                        
                        # Zero rate penalty
                        zero_mask = torch.isclose(pop_firing_rates, 
                                                  torch.tensor(0.0, device=self.device),
                                                  atol=1e-2, rtol=1e-2)
                        rate_losses = torch.where(zero_mask, 
                                                  torch.tensor(1e2, device=self.device),
                                                  rate_losses)
                        
                        trial_losses += rate_losses
                    
                    # Sparsity loss
                    if pop in self.targets.sparsity_targets:
                        target_sparsity = self.targets.sparsity_targets[pop]
                        actual_sparsity = torch.sum(
                            mean_rates > self.targets.activity_threshold,
                            dim=1
                        ).float() / mean_rates.shape[1]
                        
                        sparsity_errors = (actual_sparsity - target_sparsity) ** 2
                        trial_losses += sparsity_errors * self.targets.loss_weights.get('sparsity', 0.5)
            
            # Add constraint violations
            for b in range(batch_size):
                firing_rates_dict = {pop: trial_firing_rates[pop][b].item()
                                   for pop in trial_firing_rates}
                constraint_violation, _ = evaluate_rate_ordering_constraints(
                    firing_rates_dict,
                    self.targets.rate_ordering_constraints
                )
                trial_losses[b] += self.targets.constraint_violation_weight * constraint_violation
            
            total_losses += trial_losses
            
            # Store firing rates
            for pop in trial_firing_rates:
                all_firing_rates[pop].append(trial_firing_rates[pop])
            
            # Clean up circuit to free memory
            del circuit
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Average over trials
        total_losses /= self.config.n_trials
        avg_firing_rates = {pop: torch.stack(all_firing_rates[pop]).mean(dim=0)
                           for pop in all_firing_rates if len(all_firing_rates[pop]) > 0}
        
        return total_losses, avg_firing_rates


    def _evaluate_optogenetic_batch(self,
                                    parameter_batch: List[Dict[str, float]],
                                    mec_current: float,
                                    mec_current_std: float) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluate optogenetic objectives for batch of parameters
        
        Dispatches to fixed-dt or adaptive-dt implementation.
        """
        if self.adaptive_step:
            return self._evaluate_optogenetic_batch_adaptive(parameter_batch, mec_current, mec_current_std)
        else:
            return self._evaluate_optogenetic_batch_fixed(parameter_batch, mec_current, mec_current_std)
    
    def _evaluate_optogenetic_batch_fixed(self,
                                          parameter_batch: List[Dict[str, float]],
                                          mec_current: float,
                                          mec_current_std: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluate optogenetic objectives for batch of parameters
        
        Each trial creates new circuit with different seed
        
        Runs stimulation experiments for PV and SST populations and computes
        optogenetic-specific loss components. Averages over multiple trials.
        
        Returns:
            losses: Tensor [batch_size]
            details: Dict with per-population results (averaged over trials)
        """
        batch_size = len(parameter_batch)
        total_opto_losses = torch.zeros(batch_size, device=self.device)
        
        # Accumulate results over trials
        accumulated_details = {'pv': {}, 'sst': {}}

        # Run multiple trials with different connectivity
        for trial in range(self.config.n_trials):
            # Set seed for this trial
            trial_seed = self.base_seed + trial
            set_random_seed(trial_seed, self.device)
            
            trial_losses = torch.zeros(batch_size, device=self.device)

            self.logger.info(f"  Optogenetic trial {trial + 1}/{self.config.n_trials}: Creating circuit with seed {trial_seed}...")

            # Evaluate PV and SST stimulation
            for target_pop in ['pv', 'sst']:
                # Run stimulation experiment for batch (creates new circuit inside)
                stim_results = self._simulate_batch_optogenetic_stimulation(
                    parameter_batch, target_pop, mec_current, mec_current_std, trial_seed,
                    stim_start = self.targets.optogenetic_targets.pre_stim_duration,
                    stim_duration = self.targets.optogenetic_targets.stim_duration)
                
                # Calculate losses
                pop_losses = self._calculate_optogenetic_losses_batch(
                    stim_results, target_pop
                )
                
                trial_losses += pop_losses
                
                # Accumulate results for averaging
                if trial == 0:
                    # Initialize accumulators on first trial
                    accumulated_details[target_pop] = {
                        pop: {k: v.clone() if isinstance(v, torch.Tensor) else v 
                              for k, v in pop_data.items()}
                        for pop, pop_data in stim_results.items()
                    }
                else:
                    # Add to accumulators
                    for pop in stim_results:
                        for key, value in stim_results[pop].items():
                            if isinstance(value, torch.Tensor):
                                accumulated_details[target_pop][pop][key] += value
            
            total_opto_losses += trial_losses
        
        # Average over trials
        total_opto_losses /= self.config.n_trials
        
        # Average accumulated details
        for target_pop in accumulated_details:
            for pop in accumulated_details[target_pop]:
                for key in accumulated_details[target_pop][pop]:
                    if isinstance(accumulated_details[target_pop][pop][key], torch.Tensor):
                        accumulated_details[target_pop][pop][key] /= self.config.n_trials
        
        return total_opto_losses, accumulated_details

    def _evaluate_optogenetic_batch_adaptive(self,
                                             parameter_batch: List[Dict[str, float]],
                                             mec_current: float,
                                             mec_current_std: float) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluates optogenetic objectives for batch of parameters (adaptive dt)

        Uses synchronized adaptive time stepping for optogenetic experiments.
        """

        batch_size = len(parameter_batch)

        # Optogenetic protocols to test
        target_pops = ['pv', 'sst']

        # Storage for results
        total_opto_losses = torch.zeros(batch_size, device=self.device)

        # Accumulate results over trials
        accumulated_details = {'pv': {}, 'sst': {}}

        opto_adaptive_stats = []

        for trial in range(self.config.n_trials):
            
            # Set seed for this trial
            trial_seed = self.base_seed + trial
            set_random_seed(trial_seed, self.device)  # Reset seed for each target
            
            trial_losses = torch.zeros(batch_size, device=self.device)

            self.logger.info(f"  Opto trial {trial + 1}/{self.config.n_trials} (adaptive): seed {trial_seed}...")

            for target_pop in target_pops:

                set_random_seed(trial_seed, self.device)  # Reset seed for each target
                
                # Create circuit without compilation for adaptive stepping
                circuit = BatchDentateCircuit(
                    batch_size=batch_size,
                    circuit_params=self.circuit_params,
                    synaptic_params=self.base_synaptic_params,
                    opsin_params=self.opsin_params,
                    device=self.device,
                    compile_circuit=False  # Required for adaptive stepping
                )

                circuit.set_connection_modulation_batch(parameter_batch)

                circuit.reset_state()
                
                # MEC input (either constant or time-varying)
                stim_start = self.targets.optogenetic_targets.pre_stim_duration
                stim_duration = self.targets.optogenetic_targets.stim_duration
                duration = self.config.warmup_duration + stim_start + stim_duration
                dt = self.circuit_params.dt

                if self.use_time_varying_mec:
                    # Generate time-varying pattern: [batch_size, n_mec, n_steps]
                    mec_patterns = []
                    for b in range(batch_size):
                        pattern = generate_time_varying_mec_pattern(
                            n_mec=self.circuit_params.n_mec,
                            duration=duration,
                            dt=dt,
                            base_current=mec_current,
                            trial_index=trial + b,
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
                        )
                        mec_patterns.append(pattern)
                    mec_input = torch.stack(mec_patterns, dim=0)  # [batch, n_mec, n_steps]
                else:
                    # Constant + noise: [batch_size, n_mec]
                    mec_input = mec_current + torch.randn(
                        batch_size,
                        self.circuit_params.n_mec,
                        device=self.device
                    ) * mec_current_std

                # Create opsin activation using shared helper method
                target_opto_current = self._create_opsin_activation(
                    target_pop,
                    circuit.layout
                )
                
                # Run adaptive simulation with three phases:
                # 1. Warmup
                # 2. Pre-stimulation baseline
                # 3. Stimulation

                stim_results = self._run_adaptive_optogenetic_simulation(
                    circuit, mec_input, target_pop, target_opto_current
                )

                opto_adaptive_stats.append(stim_results['adaptive_stats'])

                # Calculate losses
                pop_losses = self._calculate_optogenetic_losses_batch(
                    stim_results['population_features'], target_pop
                )

                trial_losses += pop_losses

                # Accumulate results for averaging
                if trial == 0:
                    # Initialize accumulators on first trial
                    accumulated_details[target_pop] = {
                        pop: {k: v.clone() if isinstance(v, torch.Tensor) else v 
                              for k, v in pop_data.items()}
                        for pop, pop_data in stim_results['population_features'].items()
                    }
                else:
                    # Add to accumulators
                    for pop in stim_results['population_features']:
                        for key, value in stim_results['population_features'][pop].items():
                            if isinstance(value, torch.Tensor):
                                accumulated_details[target_pop][pop][key] += value
                
                # Clean up
                del circuit
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            total_opto_losses += trial_losses

        # Average over trials
        total_opto_losses /= self.config.n_trials
        
        # Average accumulated details
        for target_pop in accumulated_details:
            for pop in accumulated_details[target_pop]:
                for key in accumulated_details[target_pop][pop]:
                    if isinstance(accumulated_details[target_pop][pop][key], torch.Tensor):
                        accumulated_details[target_pop][pop][key] /= self.config.n_trials

        # Aggregate adaptive stats
        if hasattr(self, '_last_adaptive_stats') and self._last_adaptive_stats is not None:
            # Combine with baseline adaptive stats if they exist
            self._last_adaptive_stats = self._aggregate_adaptive_stats([self._last_adaptive_stats,
                                                                        opto_adaptive_stats])
        else:
            self._last_adaptive_stats = opto_adaptive_stats

        return total_opto_losses, accumulated_details


    def _run_adaptive_optogenetic_simulation(self, circuit, mec_input, target_pop, target_opto_current):
        """
        Run optogenetic simulation with adaptive stepping

        Three phases:
        1. Warmup (discard)
        2. Pre-stimulation baseline (record on fixed grid)
        3. Stimulation (record on fixed grid)

        Returns activities on fixed grid for compatibility with paradoxical excitation calculation.
        """

        batch_size = mec_input.shape[0]

        # Detect if MEC input is time-varying
        is_time_varying = (mec_input.ndim == 3)

        # Initialize adaptive stepper
        stepper = GradientAdaptiveStepper(self.adaptive_config)
        state = AdaptiveSimulationState(self.device)

        # Storage on fixed grid
        storage_dt = self.circuit_params.dt
        pre_stim_steps = int(self.targets.optogenetic_targets.pre_stim_duration / storage_dt)
        stim_steps = int(self.targets.optogenetic_targets.stim_duration / storage_dt)

        # Collect activities on the fixed grid for pre-stim and stim periods
        pre_stim_storage = {
            'gc': torch.zeros(batch_size, self.circuit_params.n_gc, pre_stim_steps, device=self.device),
            'mc': torch.zeros(batch_size, self.circuit_params.n_mc, pre_stim_steps, device=self.device),
            'pv': torch.zeros(batch_size, self.circuit_params.n_pv, pre_stim_steps, device=self.device),
            'sst': torch.zeros(batch_size, self.circuit_params.n_sst, pre_stim_steps, device=self.device),
        }

        stim_storage = {
            'gc': torch.zeros(batch_size, self.circuit_params.n_gc, stim_steps, device=self.device),
            'mc': torch.zeros(batch_size, self.circuit_params.n_mc, stim_steps, device=self.device),
            'pv': torch.zeros(batch_size, self.circuit_params.n_pv, stim_steps, device=self.device),
            'sst': torch.zeros(batch_size, self.circuit_params.n_sst, stim_steps, device=self.device),
        }

        # Phase durations
        warmup_duration = self.config.warmup_duration
        pre_stim_duration = self.targets.optogenetic_targets.pre_stim_duration
        stim_duration = self.targets.optogenetic_targets.stim_duration
        total_duration = warmup_duration + pre_stim_duration + stim_duration

        # Time grids for each phase
        warmup_end = warmup_duration
        pre_stim_end = warmup_end + pre_stim_duration
        stim_end = pre_stim_end + stim_duration

        pre_stim_grid = torch.arange(pre_stim_steps, device=self.device) * storage_dt + warmup_end
        stim_grid = torch.arange(stim_steps, device=self.device) * storage_dt + pre_stim_end


        # Adaptive simulation loop
        activity_current = None

        while state.current_time < total_duration:
            # Compute dt
            if state.step_index > 0:
                grad_magnitude = self._compute_batch_gradient(
                    activity_current, state.activity_prev, state.dt_history[-1]
                )
                stepper.grad_smooth = (
                    self.adaptive_config.gradient_alpha * grad_magnitude +
                    (1 - self.adaptive_config.gradient_alpha) * stepper.grad_smooth
                )
                dt_adaptive = stepper._gradient_to_dt(stepper.grad_smooth)

                # Rate limiting
                max_increase = stepper.dt_current * self.adaptive_config.max_dt_change_factor
                max_decrease = stepper.dt_current / self.adaptive_config.max_dt_change_factor
                dt_adaptive = np.clip(dt_adaptive, max_decrease, max_increase)
                dt_adaptive = np.clip(dt_adaptive, self.adaptive_config.dt_min, self.adaptive_config.dt_max)

                stepper.dt_current = dt_adaptive
            else:
                dt_adaptive = self.adaptive_config.dt_max

            # Don't overshoot
            if state.current_time + dt_adaptive > total_duration:
                dt_adaptive = total_duration - state.current_time

            # Determine phase and set inputs
            if state.current_time < warmup_end:
                # Warmup phase: no stimulation
                direct_activation = {}
            elif state.current_time < pre_stim_end:
                # Pre-stimulation baseline: no stimulation
                direct_activation = {}
            else:
                # Stimulation phase
                direct_activation = {target_pop: target_opto_current}

            # Extract MEC drive based on input type
            if is_time_varying:
                # Time-varying: extract current timestep from pattern
                time_idx = int(state.current_time / storage_dt)
                if time_idx < mec_input.shape[2]:
                    mec_drive = mec_input[:, :, time_idx]  # [batch, n_mec]
                else:
                    mec_drive = mec_input[:, :, -1]
            else:
                # Constant: use as-is
                mec_drive = mec_input  # [batch, n_mec]

            # Update circuit
            external_drive = {'mec': mec_drive}
            activity_current = update_batch_circuit_with_adaptive_dt(
                circuit, direct_activation, external_drive, dt_adaptive
            )

            # Calculate interval end time for overlap detection
            interval_end = state.current_time + dt_adaptive
        
            # Interpolate to fixed grids for pre-stim and stim phases
            if state.step_index > 0:
                # Check if we're in pre-stim phase
                if state.current_time >= warmup_end and state.current_time <= pre_stim_end:
                    self._interpolate_batch_to_grid(
                        activity_current, state.activity_prev,
                        state.current_time, interval_end,
                        pre_stim_grid, pre_stim_storage
                    )

                # Check if we're in stim phase
                if state.current_time >= pre_stim_end and state.current_time <= stim_end:
                    self._interpolate_batch_to_grid(
                        activity_current, state.activity_prev,
                        state.current_time, interval_end,
                        stim_grid, stim_storage
                    )
            elif state.step_index == 0:
                # First step: store initial condition if in recorded phase
                if state.current_time >= warmup_end and state.current_time < pre_stim_end:
                    for pop in pre_stim_storage:
                        if pop in activity_current:
                            pre_stim_storage[pop][:, :, 0] = activity_current[pop]
                elif state.current_time >= pre_stim_end:
                    for pop in stim_storage:
                        if pop in activity_current:
                            stim_storage[pop][:, :, 0] = activity_current[pop]

            # Update state
            state.update(activity_current, dt_adaptive, stepper.grad_smooth)

        # Extract activities as lists (for compatibility with fixed version)
        pre_stim_activities = {
            pop: [pre_stim_storage[pop][:, :, t] for t in range(pre_stim_steps)]
            for pop in pre_stim_storage
        }

        stim_activities = {
            pop: [stim_storage[pop][:, :, t] for t in range(stim_steps)]
            for pop in stim_storage
        }

        pop_features = {}
        for pop in stim_activities:
            if len(stim_activities[pop]) > 0:

                pop_pre_stim_activities = torch.stack(pre_stim_activities[pop], dim=0)
                pop_stim_activities = torch.stack(stim_activities[pop], dim=0)

                # Baseline and stimulation periods
                baseline_rates = torch.mean(pop_pre_stim_activities, dim=0)  # [batch, neurons]
                stim_rates = torch.mean(pop_stim_activities, dim=0)  # [batch, neurons]

                # Changes
                rate_changes = stim_rates - baseline_rates
                baseline_std = torch.std(baseline_rates, dim=1, keepdim=True)  # [batch, 1]

                # Fraction activated (per batch element)
                activated_fraction = torch.mean(
                    (rate_changes > baseline_std).float(), dim=1
                )  # [batch]

                baseline_mean = torch.mean(baseline_rates, dim=1)  # [batch]
                stim_mean = torch.mean(stim_rates, dim=1)  # [batch]
                mean_change = torch.mean(rate_changes, dim=1)  # [batch]

                # Gini coefficients (computed on CPU for numpy compatibility)
                baseline_gini_list = []
                stim_gini_list = []

                for b in range(batch_size):
                    baseline_gini_list.append(
                        calculate_gini_coefficient(baseline_rates[b].cpu().numpy())
                    )
                    stim_gini_list.append(
                        calculate_gini_coefficient(stim_rates[b].cpu().numpy())
                    )

                # Convert to torch tensors
                baseline_gini = torch.tensor(baseline_gini_list, device=self.device)
                stim_gini = torch.tensor(stim_gini_list, device=self.device)
                gini_change = stim_gini - baseline_gini

                pop_features[pop] = {
                    'baseline_mean': baseline_mean,
                    'stim_mean': stim_mean,
                    'mean_change': mean_change,
                    'activated_fraction': activated_fraction,
                    'baseline_gini': baseline_gini,
                    'stim_gini': stim_gini,
                    'gini_change': gini_change,
                }


        return {'population_features': pop_features,
                'adaptive_stats': state.get_statistics()}


    def _simulate_batch_optogenetic_stimulation(self,
                                                parameter_batch: List[Dict[str, float]],
                                                target_pop: str,
                                                mec_current: float,
                                                mec_current_std: float,
                                                trial_seed: int,
                                                stim_start: float = 1000.0,
                                                stim_duration: float = 2000.0,
                                                post_duration: float = 250.0,
                                                warmup: float = 500.0) -> Dict[str, torch.Tensor]:
        """
        Run optogenetic stimulation for a batch of parameter sets

        Uses provided trial_seed for circuit creation. 

        Args:
            parameter_batch: List of connection modulation dicts
            target_pop: Population to stimulate ('pv' or 'sst')
            mec_current: MEC drive level
            mec_current_std: MEC drive level stdev
            trial_seed: Random seed for this trial (already set externally)
            stim_start: Stimulation start time (ms)
            stim_duration: Stimulation duration (ms)
            post_duration: Post-stimulation period (ms)
            warmup: Pre-stimulation warmup (ms)

        Returns:
            Dict with baseline and stimulation statistics for each population
        """

        batch_size = len(parameter_batch)

        # Stimulation protocol parameters
        if stim_start < warmup:
            stim_start = warmup
        duration = stim_start + stim_duration + post_duration

        # Create new batch circuit with current random seed
        circuit = BatchDentateCircuit(
            batch_size=batch_size,
            circuit_params=self.circuit_params,
            synaptic_params=self.base_synaptic_params,
            opsin_params=self.opsin_params,
            device=self.device
        )

        circuit.set_connection_modulation_batch(parameter_batch)
        print(f"Circuit connectivity hash: {hash(circuit.connectivity.conductance_matrices['gc_mc'].connectivity.cpu().numpy().tobytes())}")

        
        # Create opsin activation using shared helper method
        target_opto_current = self._create_opsin_activation(
            target_pop, 
            circuit.layout
        )

        # Generate MEC input pattern
        dt = self.circuit_params.dt
        n_steps = int(duration / dt)

        # Get trial index from seed offset (trial_seed = base_seed + trial_index)
        trial_index = trial_seed - self.base_seed

        mec_input = self._generate_mec_input(
            batch_size=batch_size,
            duration=duration,
            dt=dt,
            mec_current=mec_current,
            mec_current_std=mec_current_std,
            trial_index=trial_index
        )

        # Storage for time series: [time, batch, neurons]
        activities_time_series = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}

        # Run simulation
        circuit.reset_state()

        stim_start_step = int(stim_start / dt)
        stim_end_step = int((stim_start + stim_duration) / dt)
        time_tensor = torch.arange(n_steps, device=self.device) * dt

        for t in range(n_steps):
            current_time = t * dt

            # Create per-population optogenetic drives
            direct_activation = {}

            if (t >= stim_start_step) and (t < stim_end_step):
                # Apply optogenetic stimulation only to target population
                for pop, n_neurons in [('gc', self.circuit_params.n_gc),
                                       ('mc', self.circuit_params.n_mc),
                                       ('pv', self.circuit_params.n_pv),
                                       ('sst', self.circuit_params.n_sst)]:
                    if pop == target_pop:
                        # Target population gets optogenetic drive
                        direct_activation[pop] = target_opto_current.unsqueeze(0).expand(batch_size, -1)
                    else:
                        # Non-target populations get zero drive
                        direct_activation[pop] = torch.zeros(batch_size, n_neurons,
                                                            device=self.device)
            else:
                # No stimulation before stim_start
                for pop, n_neurons in [('gc', self.circuit_params.n_gc),
                                       ('mc', self.circuit_params.n_mc),
                                       ('pv', self.circuit_params.n_pv),
                                       ('sst', self.circuit_params.n_sst)]:
                    direct_activation[pop] = torch.zeros(batch_size, n_neurons,
                                                        device=self.device)

            # Get MEC drive for this timestep
            if self.use_time_varying_mec:
                # Extract current timestep: [batch_size, n_mec]
                mec_drive = mec_input[:, :, t]
            else:
                # Constant input (already [batch_size, n_mec])
                mec_drive = mec_input

            external_drive = {'mec': mec_drive}

            activities = circuit(direct_activation, external_drive)

            for pop in activities_time_series:
                activities_time_series[pop].append(activities[pop])

        # Convert to tensors: [time, batch, neurons]
        for pop in activities_time_series:
            if len(activities_time_series[pop]) > 0:
                activities_time_series[pop] = torch.stack(activities_time_series[pop], dim=0)

        # Calculate statistics
        baseline_mask = (time_tensor >= warmup) & (time_tensor < stim_start)
        stim_mask = (time_tensor >= stim_start) & (time_tensor < (stim_start + stim_duration))

        results = {}
        for pop in activities_time_series:
            if len(activities_time_series[pop]) > 0:
                pop_series = activities_time_series[pop]  # [time, batch, neurons]

                # Baseline and stimulation periods
                baseline_rates = torch.mean(pop_series[baseline_mask], dim=0)  # [batch, neurons]
                stim_rates = torch.mean(pop_series[stim_mask], dim=0)  # [batch, neurons]

                # Changes
                rate_changes = stim_rates - baseline_rates
                baseline_std = torch.std(baseline_rates, dim=1, keepdim=True)  # [batch, 1]

                # Fraction activated (per batch element)
                activated_fraction = torch.mean(
                    (rate_changes > baseline_std).float(), dim=1
                )  # [batch]

                baseline_mean = torch.mean(baseline_rates, dim=1)  # [batch]
                stim_mean = torch.mean(stim_rates, dim=1)  # [batch]
                mean_change = torch.mean(rate_changes, dim=1)  # [batch]

                # Gini coefficients (computed on CPU for numpy compatibility)
                baseline_gini_list = []
                stim_gini_list = []

                for b in range(batch_size):
                    baseline_gini_list.append(
                        calculate_gini_coefficient(baseline_rates[b].cpu().numpy())
                    )
                    stim_gini_list.append(
                        calculate_gini_coefficient(stim_rates[b].cpu().numpy())
                    )

                # Convert to torch tensors
                baseline_gini = torch.tensor(baseline_gini_list, device=self.device)
                stim_gini = torch.tensor(stim_gini_list, device=self.device)
                gini_change = stim_gini - baseline_gini

                results[pop] = {
                    'baseline_mean': baseline_mean,
                    'stim_mean': stim_mean,
                    'mean_change': mean_change,
                    'activated_fraction': activated_fraction,
                    'baseline_gini': baseline_gini,
                    'stim_gini': stim_gini,
                    'gini_change': gini_change,
                }

        # Clean up
        del circuit
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return results

    
    def _calculate_optogenetic_losses_batch(self,
                                           opto_results: Dict,
                                           target_pop: str) -> torch.Tensor:
        """
        Calculate optogenetic loss for batch from stimulation results
        
        Returns:
            losses: Tensor [batch_size]
        """
        batch_size = list(opto_results.values())[0]['baseline_mean'].shape[0]
        losses = torch.zeros(batch_size, device=self.device)
        
        opto_targets = self.targets.optogenetic_targets

        # Target rate increases
        if target_pop in opto_targets.target_rate_increases:
            for affected_pop, target_fraction in opto_targets.target_rate_increases[target_pop].items():
                if affected_pop in opto_results and affected_pop != target_pop:
                    actual_fraction = opto_results[affected_pop]['activated_fraction']
                    
                    # Squared error
                    errors = (torch.clip(actual_fraction - target_fraction, max=0)) ** 2
                    
                    # Zero activation penalty
                    if abs(target_fraction) > 0:
                        zero_mask = torch.isclose(
                            torch.abs(actual_fraction),
                            torch.tensor(0.0, device=self.device),
                            atol=1e-2, rtol=1e-2
                        )
                        errors = torch.where(zero_mask,
                                             torch.tensor(1e2, device=self.device),
                                             errors)
                    
                    losses += errors
        
        # Target Gini increases
        if target_pop in opto_targets.target_gini_increase:
            for affected_pop, target_gini_change in opto_targets.target_gini_increase[target_pop].items():
                if affected_pop in opto_results and affected_pop != target_pop:
                    actual_gini_change = opto_results[affected_pop]['gini_change']
                    errors = (torch.clip(actual_gini_change - target_gini_change, max=0)) ** 2
                    losses += errors
        
        return losses

    def _evaluate_baseline_batch_adaptive(self,
                                          parameter_batch: List[Dict[str, float]],
                                          mec_current: float,
                                          mec_current_std: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate baseline circuit objectives for batch of parameters (adaptive dt)
        
        Uses synchronized adaptive stepping across the batch.
        """
        batch_size = len(parameter_batch)
        
        # Average over trials
        total_losses = torch.zeros(batch_size, device=self.device)
        all_firing_rates = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        baseline_adaptive_stats = []

        for trial in range(self.config.n_trials):
            # Set seed for this trial
            trial_seed = self.base_seed + trial
            set_random_seed(trial_seed, self.device)

            self.logger.info(f"  Baseline trial {trial + 1}/{self.config.n_trials} (adaptive): Creating circuit with seed {trial_seed}...")

            # Create circuit for this trial
            circuit = BatchDentateCircuit(
                batch_size=batch_size,
                circuit_params=self.circuit_params,
                synaptic_params=self.base_synaptic_params,
                opsin_params=self.opsin_params,
                device=self.device
            )

            circuit.set_connection_modulation_batch(parameter_batch)
            
            circuit.reset_state()

            # NEW: Consume RNG for opsin expression to match protocol path
            # This advances RNG state but we don't use the activation
            _ = self._create_opsin_activation('pv', circuit.layout) 
            
            duration = self.config.warmup_duration + self.config.simulation_duration
            dt = self.circuit_params.dt


            if self.use_time_varying_mec:
                # Time-varying: [batch_size, n_mec, n_steps]
                mec_patterns = []
                for b in range(batch_size):
                    pattern = generate_time_varying_mec_pattern(
                        n_mec=self.circuit_params.n_mec,
                        duration=duration,
                        dt=dt,
                        base_current=mec_current,
                        trial_index=trial + b,
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
                    )
                    mec_patterns.append(pattern)
                mec_input = torch.stack(mec_patterns, dim=0)
            else:
                # Constant + noise: [batch_size, n_mec]
                mec_input = mec_current + torch.randn(
                    batch_size, 
                    self.circuit_params.n_mec,
                    device=self.device
                ) * mec_current_std

            # Run adaptive simulation (handles both input shapes)
            result = self._run_batch_adaptive_baseline(circuit, mec_input)
            
            activities_over_time = result['activities_over_time']
            baseline_adaptive_stats.append(result['adaptive_stats'])

            # Calculate losses (same as fixed version)
            trial_losses = torch.zeros(batch_size, device=self.device)
            trial_firing_rates = {}

            for pop in activities_over_time:
                if len(activities_over_time[pop]) > 0:
                    pop_time_series = torch.stack(activities_over_time[pop], dim=0)
                    mean_rates = torch.mean(pop_time_series, dim=0)

                    pop_firing_rates = torch.mean(mean_rates, dim=1)
                    trial_firing_rates[pop] = pop_firing_rates

                    if pop in self.targets.target_rates:
                        target_rate = self.targets.target_rates[pop]
                        tolerance = self.targets.rate_tolerance[pop]

                        errors = torch.abs(pop_firing_rates - target_rate)

                        rate_losses = torch.where(
                            errors <= tolerance,
                            0.5 * errors ** 2,
                            tolerance * errors - 0.5 * tolerance ** 2
                        )

                        zero_mask = torch.isclose(pop_firing_rates, 
                                                  torch.tensor(0.0, device=self.device),
                                                  atol=1e-2, rtol=1e-2)
                        rate_losses = torch.where(zero_mask, 
                                                  torch.tensor(1e2, device=self.device),
                                                  rate_losses)

                        trial_losses += rate_losses

                    if pop in self.targets.sparsity_targets:
                        target_sparsity = self.targets.sparsity_targets[pop]
                        actual_sparsity = torch.sum(
                            mean_rates > self.targets.activity_threshold,
                            dim=1
                        ).float() / mean_rates.shape[1]

                        sparsity_errors = (actual_sparsity - target_sparsity) ** 2
                        trial_losses += sparsity_errors * self.targets.loss_weights.get('sparsity', 0.5)

            # Add constraints
            for b in range(batch_size):
                firing_rates_dict = {pop: trial_firing_rates[pop][b].item()
                                   for pop in trial_firing_rates}
                constraint_violation, _ = evaluate_rate_ordering_constraints(
                    firing_rates_dict,
                    self.targets.rate_ordering_constraints
                )
                trial_losses[b] += self.targets.constraint_violation_weight * constraint_violation

            total_losses += trial_losses

            for pop in trial_firing_rates:
                all_firing_rates[pop].append(trial_firing_rates[pop])

            # Clean up
            del circuit
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Average over trials
        total_losses /= self.config.n_trials
        avg_firing_rates = {pop: torch.stack(all_firing_rates[pop]).mean(dim=0)
                           for pop in all_firing_rates if len(all_firing_rates[pop]) > 0}

        # Store aggregated adaptive stats
        if hasattr(self, '_last_adaptive_stats') and self._last_adaptive_stats is not None:
            # Combine with baseline adaptive stats if they exist
            self._last_adaptive_stats = self._aggregate_adaptive_stats([self._last_adaptive_stats,
                                                                        baseline_adaptive_stats])
        else:
            self._last_adaptive_stats = baseline_adaptive_stats


        return total_losses, avg_firing_rates
    
    def _run_batch_adaptive_baseline(self, circuit, mec_input):
        """
        Run batch baseline simulation with synchronized adaptive stepping

        All batch elements use the same dt at each step (synchronized).
        """
        # Detect if MEC input is time-varying
        is_time_varying = (mec_input.ndim == 3)
        
        # Initialize adaptive stepper
        stepper = GradientAdaptiveStepper(self.adaptive_config)
        state = AdaptiveSimulationState(self.device)

        # Storage on fixed grid
        storage_dt = self.circuit_params.dt
        duration = self.config.warmup_duration + self.config.simulation_duration
        n_steps = int(duration / storage_dt)
        time_grid = torch.arange(n_steps, device=self.device) * storage_dt
        warmup_steps = int(self.config.warmup_duration / storage_dt)

        batch_size = mec_input.shape[0]

        activity_storage = {
            'gc': torch.zeros(batch_size, self.circuit_params.n_gc, n_steps, device=self.device),
            'mc': torch.zeros(batch_size, self.circuit_params.n_mc, n_steps, device=self.device),
            'pv': torch.zeros(batch_size, self.circuit_params.n_pv, n_steps, device=self.device),
            'sst': torch.zeros(batch_size, self.circuit_params.n_sst, n_steps, device=self.device),
        }

        # Adaptive loop
        while state.current_time < duration:
            # Compute dt (synchronized across batch)
            if state.step_index > 0:
                # Average gradient across batch for synchronized stepping
                grad = self._compute_batch_gradient(
                    activity_current, state.activity_prev, state.dt_history[-1]
                )
                dt_adaptive = stepper.compute_dt(
                    {'batch': torch.tensor(grad, device=self.device)},
                    {'batch': torch.tensor(0.0, device=self.device)},
                    state.dt_history[-1]
                )
            else:
                dt_adaptive = self.adaptive_config.dt_max

            # Don't overshoot
            if state.current_time + dt_adaptive > duration:
                dt_adaptive = duration - state.current_time

            # Extract MEC drive based on input type
            if is_time_varying:
                time_idx = int(state.current_time / storage_dt)
                if time_idx < mec_input.shape[2]:
                    mec_drive = mec_input[:, :, time_idx]  # [batch, n_mec]
                else:
                    mec_drive = mec_input[:, :, -1]
            else:
                mec_drive = mec_input  # [batch, n_mec]
                
            # Update circuit
            external_drive = {'mec': mec_drive}
            activity_current = update_batch_circuit_with_adaptive_dt(
                circuit, {}, external_drive, dt_adaptive
            )

            # Interpolate to grid
            if state.step_index == 0:
                for pop in activity_storage:
                    activity_storage[pop][:, :, 0] = activity_current[pop]
            else:
                self._interpolate_batch_to_grid(
                    activity_current, state.activity_prev,
                    state.current_time, state.current_time + dt_adaptive,
                    time_grid, activity_storage
                )

            # Update state
            state.update(activity_current, dt_adaptive, stepper.grad_smooth)

        # Extract post-warmup activities
        activities_over_time = {
            pop: [activity_storage[pop][:, :, t] for t in range(warmup_steps, n_steps)]
            for pop in activity_storage
        }

        return {
            'activities_over_time': activities_over_time,
            'adaptive_stats': state.get_statistics()
        }




    def _compute_batch_gradient(self, activities_current, activities_prev, dt):
        """
        Compute average gradient magnitude across batch

        For synchronized adaptive stepping.
        """
        gradients = []

        for pop_name in activities_current.keys():
            if pop_name not in activities_prev:
                continue

            # Mean across neurons and batch
            delta = torch.mean(activities_current[pop_name] - activities_prev[pop_name])
            grad = abs(delta.item()) / dt
            gradients.append(grad)

        return np.mean(gradients) if gradients else 0.0

    def _interpolate_batch_to_grid(self, activity_current, activity_prev,
                                   time_prev, time_current, time_grid, storage):
        """Linear interpolation for batch data"""
        mask = (time_grid > time_prev) & (time_grid <= time_current)
        grid_indices = torch.where(mask)[0]

        if len(grid_indices) == 0:
            return

        dt_step = time_current - time_prev

        for grid_idx in grid_indices:
            grid_time = time_grid[grid_idx].item()

            if dt_step > 1e-9:
                alpha = (grid_time - time_prev) / dt_step
            else:
                alpha = 1.0

            # Interpolate for batch: [batch, neurons]
            for pop_name in activity_current.keys():
                if pop_name in storage and pop_name in activity_prev:
                    interpolated = (1 - alpha) * activity_prev[pop_name] + alpha * activity_current[pop_name]
                    storage[pop_name][:, :, grid_idx] = interpolated

    def _aggregate_adaptive_stats(self, stats_list):
        """
        Aggregate adaptive statistics across trials

        Handles both raw stats dicts and already-aggregated stats dicts.

        Args:
            stats_list: List of stats dicts. Each can be either:
                       - Raw stats: {'n_steps': int, 'avg_dt': float, 'min_dt': float, 'max_dt': float}
                       - Aggregated stats: {'n_steps_mean': float, 'n_steps_std': float, ...}

        Returns:
            Aggregated stats dict with mean, std, min, max for each metric
        """
        if not stats_list:
            return None

        import numpy as np

        # Separate raw stats from already-aggregated stats
        raw_stats = []
        aggregated_stats = []

        for stats in stats_list:
            if 'n_steps_mean' in stats:
                # Already aggregated
                aggregated_stats.append(stats)
            elif 'n_steps' in stats:
                # Raw stats
                raw_stats.append(stats)

        # If we only have aggregated stats, combine them
        if raw_stats == [] and aggregated_stats:
            return self._combine_aggregated_stats(aggregated_stats)

        # If we only have raw stats, aggregate them normally
        if aggregated_stats == [] and raw_stats:
            return self._aggregate_raw_stats(raw_stats)

        # If we have both, aggregate raw stats then combine with aggregated
        if raw_stats and aggregated_stats:
            new_aggregated = self._aggregate_raw_stats(raw_stats)
            all_aggregated = aggregated_stats + [new_aggregated]
            return self._combine_aggregated_stats(all_aggregated)

        return None

    def _aggregate_raw_stats(self, raw_stats):
        """
        Aggregate raw adaptive statistics

        Args:
            raw_stats: List of raw stats dicts with fields like 'n_steps', 'avg_dt'

        Returns:
            Aggregated stats with mean, std, min, max
        """
        import numpy as np

        return {
            'n_steps_mean': np.mean([s['n_steps'] for s in raw_stats]),
            'n_steps_std': np.std([s['n_steps'] for s in raw_stats]),
            'avg_dt_mean': np.mean([s['avg_dt'] for s in raw_stats]),
            'avg_dt_std': np.std([s['avg_dt'] for s in raw_stats]),
            'min_dt': np.min([s['min_dt'] for s in raw_stats]),
            'max_dt': np.max([s['max_dt'] for s in raw_stats]),
        }

    def _combine_aggregated_stats(self, aggregated_stats):
        """
        Combine multiple already-aggregated statistics

        Treats each aggregated stat as a single sample at its mean value.
        This is an approximation but reasonable for combining stats from
        different evaluation phases (baseline + optogenetic).

        Args:
            aggregated_stats: List of aggregated stats dicts

        Returns:
            Combined aggregated stats
        """
        import numpy as np

        if len(aggregated_stats) == 1:
            return aggregated_stats[0]

        # Extract means as "samples" for re-aggregation
        n_steps_samples = [s['n_steps_mean'] for s in aggregated_stats]
        avg_dt_samples = [s['avg_dt_mean'] for s in aggregated_stats]

        # For min/max, take the extreme values across all aggregated stats
        all_min_dt = [s['min_dt'] for s in aggregated_stats]
        all_max_dt = [s['max_dt'] for s in aggregated_stats]

        return {
            'n_steps_mean': np.mean(n_steps_samples),
            'n_steps_std': np.std(n_steps_samples),
            'avg_dt_mean': np.mean(avg_dt_samples),
            'avg_dt_std': np.std(avg_dt_samples),
            'min_dt': np.min(all_min_dt),
            'max_dt': np.max(all_max_dt),
        }
    
# ============================================================================
# Evaluation Strategy Pattern for Optogenetics
# ============================================================================

class OptogeneticEvaluationStrategy(ABC):
    """
    Abstract base class for optogenetic parameter evaluation strategies
    
    Similar to EvaluationStrategy but specialized for combined baseline + optogenetic objectives
    """
    
    @abstractmethod
    def evaluate_batch(self,
                       parameter_sets: List[Dict[str, float]],
                       mec_current: float,
                       mec_current_std: float,
                       n_trials: int) -> Tuple[List[float], List[Dict]]:
        """
        Evaluate a batch of parameter configurations with optogenetic protocols
        
        Args:
            parameter_sets: List of connection_modulation dicts
            mec_current: MEC drive level
            mec_current_std: MEC drive level std
            n_trials: Number of trials to average
            
        Returns:
            losses: List of total loss values
            metadata_list: List of Dict with loss breakdowns and firing rates
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, any]:
        """Return information about this strategy"""
        pass


class OptogeneticSequentialStrategy(OptogeneticEvaluationStrategy):
    """Sequential evaluation for gradient-based or single evaluations"""
    
    def __init__(self, circuit_params, base_synaptic_params,
                 opsin_params, opsin_current,
                 targets, config, device, base_seed=42,
                 adaptive_step=False, adaptive_config=None,
                 verbose=False,
                 use_time_varying_mec=False,
                 mec_pattern_type='oscillatory',
                 mec_theta_freq=5.0,
                 mec_theta_amplitude=0.3,
                 mec_gamma_freq=20.0,
                 mec_gamma_amplitude=0.15,
                 mec_gamma_coupling_strength=0.8,
                 mec_gamma_preferred_phase=0.0,
                 mec_drift_timescale=200.0,
                 mec_drift_amplitude=0.4,
                 mec_rotation_groups=3):
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.opsin_current = opsin_current
        self.targets = targets
        self.config = config
        self.device = device
        self.base_seed = base_seed
        self.adaptive_step = adaptive_step
        self.adaptive_config = adaptive_config
        self.verbose = verbose
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
    
    def evaluate_batch(self, parameter_sets, mec_current, mec_current_std, n_trials):
        """Evaluate configurations one at a time"""
        losses = []
        metadata_list = []
        
        for params in parameter_sets:
            # Use batch evaluator with batch_size=1
            evaluator = BatchOptogeneticEvaluator(
                self.circuit_params, self.base_synaptic_params,
                self.opsin_params, self.opsin_current,
                self.targets, self.config, device=self.device, base_seed=self.base_seed,
                adaptive_step=self.adaptive_step,
                adaptive_config=self.adaptive_config,
                verbose=self.verbose,
                # Pass MEC pattern parameters
                use_time_varying_mec=self.use_time_varying_mec,
                mec_pattern_type=self.mec_pattern_type,
                mec_theta_freq=self.mec_theta_freq,
                mec_theta_amplitude=self.mec_theta_amplitude,
                mec_gamma_freq=self.mec_gamma_freq,
                mec_gamma_amplitude=self.mec_gamma_amplitude,
                mec_gamma_coupling_strength=self.mec_gamma_coupling_strength,
                mec_gamma_preferred_phase=self.mec_gamma_preferred_phase,
                mec_drift_timescale=self.mec_drift_timescale,
                mec_drift_amplitude=self.mec_drift_amplitude,
                mec_rotation_groups=self.mec_rotation_groups

            )
            
            loss_tensor, details = evaluator.evaluate_parameter_batch([params], mec_current, mec_current_std)
            losses.append(loss_tensor.item())

            metadata = self._extract_metadata(details, 0)  # batch index 0
            metadata_list.append(metadata)
        
        return losses, metadata_list
    
    def get_strategy_info(self):
        step_mode = "dt-adaptive" if self.adaptive_step else "fixed-dt"

        return {
            'name': 'OptogeneticSequential',
            'device': str(self.device),
            'parallelism': 'None',
            'step_strategy': step_mode,
            'description': 'Sequential evaluation with optogenetic protocols (different seeds per trial)'
        }

    def _extract_metadata(self, details: Dict, batch_idx: int) -> Dict:
        """Extract metadata for a single configuration from batch results"""
        metadata = {
            'baseline_loss': self._tensor_to_python(details['baseline_losses'][batch_idx]),
            'opto_loss': self._tensor_to_python(details['opto_losses'][batch_idx]),
            'baseline_rates': {},
            'opto_details': {}
        }
        
        # Extract baseline rates
        for pop, rates in details['baseline_rates'].items():
            metadata['baseline_rates'][pop] = self._tensor_to_python(rates[batch_idx])
        
        # Extract optogenetic details
        for target_pop in ['pv', 'sst']:
            if target_pop in details['opto_details']:
                metadata['opto_details'][target_pop] = {}
                for affected_pop, pop_data in details['opto_details'][target_pop].items():
                    metadata['opto_details'][target_pop][affected_pop] = {
                        k: self._tensor_to_python(v[batch_idx]) if hasattr(v, '__getitem__') else v
                        for k, v in pop_data.items()
                    }
                    
        return metadata

    def _tensor_to_python(obj):
        """Convert torch tensor to Python type"""
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.cpu().numpy().tolist()
        return obj

class OptogeneticBatchGPUStrategy(OptogeneticEvaluationStrategy):
    """Batch GPU evaluation for population-based optimization"""
    
    def __init__(self, circuit_params, base_synaptic_params,
                 opsin_params, opsin_current,
                 targets, config, device, base_seed=42, verbose=False,
                 use_time_varying_mec=False,
                 mec_pattern_type='oscillatory',
                 mec_theta_freq=5.0,
                 mec_theta_amplitude=0.3,
                 mec_gamma_freq=20.0,
                 mec_gamma_amplitude=0.15,
                 mec_gamma_coupling_strength=0.8,
                 mec_gamma_preferred_phase=0.0,
                 mec_drift_timescale=200.0,
                 mec_drift_amplitude=0.4,
                 mec_rotation_groups=3):

        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.opsin_currenet = opsin_current
        self.targets = targets
        self.config = config
        self.device = device
        self.base_seed = base_seed
        self.verbose = verbose
        
        # Create evaluator
        self.evaluator = BatchOptogeneticEvaluator(
            circuit_params, base_synaptic_params,
            opsin_params, opsin_current,
            targets, config, device=device, base_seed=base_seed,
            verbose=self.verbose,
            # Pass MEC pattern parameters
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

        
    def evaluate_batch(self, parameter_sets, mec_current, mec_current_std, n_trials):
        """Evaluate all configurations in parallel on GPU"""
        batch_size = len(parameter_sets)
        
        # Note: n_trials is handled internally by evaluator
        losses_tensor, details = self.evaluator.evaluate_parameter_batch(
            parameter_sets, mec_current, mec_current_std
        )
        
        # Convert to lists for consistent API
        losses_list = losses_tensor.cpu().numpy().tolist()

        metadata_list = []
        for i in range(batch_size):
            metadata = self._extract_metadata(details, i)
            metadata_list.append(metadata)
        
        return losses_list, metadata_list
    
    def get_strategy_info(self):
        return {
            'name': 'OptogeneticBatchGPU',
            'device': str(self.device),
            'parallelism': 'Data parallelism (batched optogenetic experiments)',
            'description': f'Batch optogenetic evaluation on GPU (seed {self.base_seed} + trial_idx)'
        }


    def _extract_metadata(self, details: Dict, batch_idx: int) -> Dict:
        """Extract metadata for a single configuration from batch results"""
        import torch
        
        metadata = {
            'baseline_loss': self._tensor_to_python(details['baseline_losses'][batch_idx]),
            'opto_loss': self._tensor_to_python(details['opto_losses'][batch_idx]),
            'baseline_rates': {},
            'opto_details': {}
        }
        
        # Extract baseline rates
        for pop, rates in details['baseline_rates'].items():
            metadata['baseline_rates'][pop] = self._tensor_to_python(rates[batch_idx])
        
        # Extract optogenetic details
        for target_pop in ['pv', 'sst']:
            if target_pop in details['opto_details']:
                metadata['opto_details'][target_pop] = {}
                for affected_pop, pop_data in details['opto_details'][target_pop].items():
                    metadata['opto_details'][target_pop][affected_pop] = {
                        k: self._tensor_to_python(v[batch_idx]) if hasattr(v, '__getitem__') else v
                        for k, v in pop_data.items()
                    }
        
        return metadata

    @staticmethod
    def _tensor_to_python(obj):
        """Convert torch tensor to Python type"""
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.cpu().numpy().tolist()
        return obj

    

class OptogeneticMultiprocessCPUStrategy(OptogeneticEvaluationStrategy):
    """Multiprocess CPU evaluation for population-based optimization"""
    
    def __init__(self, circuit_params, base_synaptic_params,
                 opsin_params, opsin_current,
                 targets, config, device, n_workers=None, n_threads_per_worker=1,
                 base_seed=42, adaptive_step=False, adaptive_config=False, verbose=False,
                 use_time_varying_mec=False,
                 mec_pattern_type='oscillatory',
                 mec_theta_freq=5.0,
                 mec_theta_amplitude=0.3,
                 mec_gamma_freq=20.0,
                 mec_gamma_amplitude=0.15,
                 mec_gamma_coupling_strength=0.8,
                 mec_gamma_preferred_phase=0.0,
                 mec_drift_timescale=200.0,
                 mec_drift_amplitude=0.4,
                 mec_rotation_groups=3):
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.opsin_current = opsin_current
        self.targets = targets
        self.config = config
        self.device = device
        self.base_seed = base_seed
        self.verbose = verbose

        self.adaptive_step = adaptive_step
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
        
        # Configure workers
        n_cores = mp.cpu_count()
        if n_workers is None:
            self.n_workers = max(1, n_cores // max(1, n_threads_per_worker))
        else:
            self.n_workers = n_workers
        self.n_threads_per_worker = n_threads_per_worker
        
        # Prepare data for workers
        self.worker_data = (
            circuit_params,
            {
                'ampa_g_mean': base_synaptic_params.ampa_g_mean,
                'ampa_g_std': base_synaptic_params.ampa_g_std,
                'gaba_g_mean': base_synaptic_params.gaba_g_mean,
                'gaba_g_std': base_synaptic_params.gaba_g_std,
                'distribution': base_synaptic_params.distribution,
            },
            opsin_params,
            opsin_current,
            # MEC pattern parameters
            {
                'use_time_varying_mec': use_time_varying_mec,
                'mec_pattern_type': mec_pattern_type,
                'mec_theta_freq': mec_theta_freq,
                'mec_theta_amplitude': mec_theta_amplitude,
                'mec_gamma_freq': mec_gamma_freq,
                'mec_gamma_amplitude': mec_gamma_amplitude,
                'mec_gamma_coupling_strength': mec_gamma_coupling_strength,
                'mec_gamma_preferred_phase': mec_gamma_preferred_phase,
                'mec_drift_timescale': mec_drift_timescale,
                'mec_drift_amplitude': mec_drift_amplitude,
                'mec_rotation_groups': mec_rotation_groups,
            }
            
        )
    
    def evaluate_batch(self, parameter_sets, mec_current, mec_current_std, n_trials):
        """Evaluate configurations using multiprocessing"""
        # Prepare arguments for workers
        eval_args = [
            (params, mec_current, mec_current_std, self.worker_data, self.targets,
             self.config, self.base_seed, self.adaptive_step, self.adaptive_config, self.verbose)
            for params in parameter_sets
        ]
        
        configure_torch_threads(self.n_threads_per_worker)
        
        # Use multiprocessing pool
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.n_workers) as pool:
            results = pool.map(_optogenetic_worker_evaluate, eval_args)
        
        # Unpack results
        losses = [r[0] for r in results]
        metadata_list = [r[1] for r in results]
        
        return losses, metadata_list
    
    def get_strategy_info(self):
        step_mode = "adaptive-dt" if self.adaptive_step else "fixed-dt"

        return {
            'name': 'OptogeneticMultiprocessCPU',
            'device': str(self.device),
            'parallelism': f'{self.n_workers} workers * {self.n_threads_per_worker} threads',
            'step_strategy': step_mode,
            'description': f'Multiprocess optogenetic evaluation on CPU (seed {self.base_seed} + trial_idx)'
        }


def _optogenetic_worker_evaluate(args):
    """Worker function for multiprocess optogenetic evaluation"""
    (connection_modulation, mec_current, mec_current_std, worker_data,
     targets, config, base_seed, adaptive_step, adaptive_config, verbose) = args
    
    device = torch.device('cpu')
    circuit_params, base_synaptic_params_dict, opsin_params, opsin_current, mec_pattern_params = worker_data
    
    # Create synaptic params
    synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params_dict['ampa_g_mean'],
        ampa_g_std=base_synaptic_params_dict['ampa_g_std'],
        gaba_g_mean=base_synaptic_params_dict['gaba_g_mean'],
        gaba_g_std=base_synaptic_params_dict['gaba_g_std'],
        distribution=base_synaptic_params_dict['distribution'],
        connection_modulation=connection_modulation
    )
    
    # Create evaluator with base seed
    evaluator = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params,
        opsin_params, opsin_current,
        targets, config, device=device, base_seed=base_seed,
        adaptive_step=adaptive_step,
        adaptive_config=adaptive_config,
        verbose=verbose,
        **mec_pattern_params  # Unpack MEC pattern parameters
    )
    
    # Evaluate single parameter set
    loss_tensor, details = evaluator.evaluate_parameter_batch(
        [connection_modulation], mec_current, mec_current_std
    )
    
    # Convert all tensors in details to Python types for safe multiprocessing
    def tensor_to_python(obj):
        """Recursively convert torch tensors to Python types"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
        elif isinstance(obj, dict):
            return {k: tensor_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(tensor_to_python(item) for item in obj)
        else:
            return obj
    
    details_safe = tensor_to_python(details)
    
    return loss_tensor.item(), details_safe


# ============================================================================
# Main Optogenetic Optimizer Class
# ============================================================================

class OptogeneticCircuitOptimizer:
    """
    Optogenetic circuit optimizer with automatic evaluation strategy selection
    
    Uses different connectivity seeds for each trial
    
    Extends CircuitOptimizer pattern to include optogenetic objectives.
    Automatically chooses optimal strategy based on device and method.
    Each trial regenerates circuit connectivity for proper averaging.
    """
    
    def __init__(self,
                 circuit_params: CircuitParams,
                 base_synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 opsin_current: float,
                 targets: CombinedOptimizationTargets,
                 config: OptimizationConfig,
                 device: Optional[torch.device] = None,
                 base_seed: int = 42):
        
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.opsin_current = opsin_current
        self.targets = targets
        self.config = config
        self.base_seed = base_seed
        
        # Set device
        self.device = device if device is not None else get_default_device()
        self.config.device = self.device
        
        # Storage
        self.best_loss = float('inf')
        self.best_metadata = None  # Store metadata for best configuration
        self.best_params = None
        self.best_iteration = 0
        self.history = {'loss': [], 'parameters': []}

        self.logger = logging.getLogger("DG_opto_optim")
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

        self.logger.info(f"OptogeneticCircuitOptimizer initialized on device: {self.device}")
        self.logger.info(f"  Base seed: {base_seed}")
        self.logger.info(f"  Baseline weight: {targets.baseline_weight}")
        self.logger.info(f"  Optogenetic weight: {targets.optogenetic_weight}")


    def _select_strategy(self, method: str, **kwargs) -> OptogeneticEvaluationStrategy:
        """Select optimal evaluation strategy based on method and device"""

        # Extract adaptive stepping configuration
        adaptive_step = kwargs.get('adaptive_step', self.config.adaptive_step if hasattr(self.config, 'adaptive_step') else False)
        adaptive_config = kwargs.get('adaptive_config', self.config.adaptive_config if hasattr(self.config, 'adaptive_config') else None)

        # NEW: Extract MEC pattern configuration
        use_time_varying_mec = kwargs.get('use_time_varying_mec', False)
        mec_pattern_type = kwargs.get('mec_pattern_type', 'oscillatory')
        mec_theta_freq = kwargs.get('mec_theta_freq', 5.0)
        mec_theta_amplitude = kwargs.get('mec_theta_amplitude', 0.3)
        mec_gamma_freq = kwargs.get('mec_gamma_freq', 20.0)
        mec_gamma_amplitude = kwargs.get('mec_gamma_amplitude', 0.15)
        mec_gamma_coupling_strength = kwargs.get('mec_gamma_coupling_strength', 0.8)
        mec_gamma_preferred_phase = kwargs.get('mec_gamma_preferred_phase', 0.0)
        mec_drift_timescale = kwargs.get('mec_drift_timescale', 200.0)
        mec_drift_amplitude = kwargs.get('mec_drift_amplitude', 0.4)
        mec_rotation_groups = kwargs.get('mec_rotation_groups', 3)

        # MEC parameter dict for passing to strategies
        mec_params = {
            'use_time_varying_mec': use_time_varying_mec,
            'mec_pattern_type': mec_pattern_type,
            'mec_theta_freq': mec_theta_freq,
            'mec_theta_amplitude': mec_theta_amplitude,
            'mec_gamma_freq': mec_gamma_freq,
            'mec_gamma_amplitude': mec_gamma_amplitude,
            'mec_gamma_coupling_strength': mec_gamma_coupling_strength,
            'mec_gamma_preferred_phase': mec_gamma_preferred_phase,
            'mec_drift_timescale': mec_drift_timescale,
            'mec_drift_amplitude': mec_drift_amplitude,
            'mec_rotation_groups': mec_rotation_groups,
        }

        if method == 'gradient':
            strategy = OptogeneticSequentialStrategy(
                self.circuit_params, self.base_synaptic_params,
                self.opsin_params, self.opsin_current,
                self.targets, self.config, self.device, base_seed=self.base_seed,
                adaptive_step=adaptive_step,
                adaptive_config=adaptive_config,
                verbose=True,
                **mec_params
            )

        elif method in ['particle_swarm', 'genetic_algorithm']:
            if self.device.type == 'cuda':
                strategy = OptogeneticBatchGPUStrategy(
                    self.circuit_params, self.base_synaptic_params,
                    self.opsin_params, self.opsin_current,
                    self.targets, self.config, self.device, base_seed=self.base_seed,
                    verbose=True,
                    **mec_params
                )
            else:
                strategy = OptogeneticMultiprocessCPUStrategy(
                    self.circuit_params, self.base_synaptic_params,
                    self.opsin_params, self.opsin_current,
                    self.targets, self.config, self.device,
                    n_workers=kwargs.get('n_workers', None),
                    n_threads_per_worker=kwargs.get('n_threads_per_worker', 1),
                    base_seed=self.base_seed,
                    adaptive_step=adaptive_step,
                    adaptive_config=adaptive_config,
                    verbose=True,
                    **mec_params
                )

        elif method == 'differential_evolution':
            if self.device.type == 'cuda':
                raise NotImplementedError(
                    "Differential Evolution on GPU not supported (scipy limitation). "
                    "Use CPU or try 'particle_swarm' for GPU optimization."
                )
            strategy = OptogeneticMultiprocessCPUStrategy(
                self.circuit_params, self.base_synaptic_params,
                self.opsin_params, self.opsin_current,
                self.targets, self.config, self.device,
                n_workers=kwargs.get('n_workers', None),
                n_threads_per_worker=kwargs.get('n_threads_per_worker', 1),
                base_seed=self.base_seed,
                adaptive_step=adaptive_step,
                adaptive_config=adaptive_config,
                verbose=True,
                **mec_params
            )

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Print strategy info
        info = strategy.get_strategy_info()
        self.logger.info(f"\nEvaluation Strategy: {info['name']}")
        self.logger.info(f"  Device: {info['device']}")
        self.logger.info(f"  Parallelism: {info['parallelism']}")
        self.logger.info(f"  Description: {info['description']}\n")

        return strategy


        
    def optimize(self, method: str = 'particle_swarm', **kwargs) -> Dict:
        """
        Run optimization with automatic strategy selection
        
        Args:
            method: 'gradient', 'particle_swarm', 'differential_evolution'
            **kwargs: Method-specific arguments
            
        Returns:
            Dict with optimization results
        """
        connection_names = list(self.targets.connection_bounds.keys())
        
        if method == 'particle_swarm':
            return self._optimize_particle_swarm(connection_names, **kwargs)
        elif method == 'differential_evolution':
            return self._optimize_differential_evolution(connection_names, **kwargs)
        elif method == 'gradient':
            raise NotImplementedError("Gradient-based optimization not yet implemented for optogenetics")
        else:
            raise ValueError(f"Unknown method: {method}")

    def _optimize_particle_swarm(self, connection_names, n_particles=32, max_iterations=50,
                                 n_workers=None, n_threads_per_worker=1,
                                 diagnostic_frequency=5, pso_config=None, **kwargs):
        """Particle swarm optimization using Adaptive PSO."""
        import numpy as np
        from adaptive_pso import AdaptivePSO, PSOConfig

        # Select evaluation strategy
        strategy = self._select_strategy('particle_swarm',
                                         n_workers=n_workers,
                                         n_threads_per_worker=n_threads_per_worker,
                                         **kwargs)

        # Setup bounds
        bounds = [self.targets.connection_bounds.get(name, (0.1, 5.0))
                  for name in connection_names]

        # Create objective function wrapper
        def objective_function(positions: np.ndarray) -> np.ndarray:
            """Evaluate batch of parameter configurations."""
            parameter_sets = [dict(zip(connection_names, pos)) for pos in positions]
            mec_current = self.config.mec_drive_levels[0]
            mec_current_std = self.config.mec_drive_std
            losses, metadata_list = strategy.evaluate_batch(parameter_sets, mec_current, mec_current_std, self.config.n_trials)
            return np.array(losses), metadata_list

        # Configure PSO
        pso_params = {
            'n_particles': n_particles,
            'max_iterations': max_iterations,
            'diagnostic_frequency': diagnostic_frequency,
            'verbose': True,
            'track_metadata': True
        }
        if pso_config is not None:
            pso_params.update(pso_config)

        config = PSOConfig(**pso_params)

        # Print header
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Circuit Optimization using Adaptive PSO")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Strategy: {strategy.get_strategy_info()['name']}")
        self.logger.info(f"Device: {self.device}, Seed: {self.base_seed}")
        self.logger.info(f"Particles: {config.n_particles}, Iterations: {config.max_iterations}")
        self.logger.info(f"OBL: {config.use_obl_initialization}, Adaptive: {config.use_adaptive_parameters}")
        self.logger.info(f"Multi-Swarm: {config.use_multi_swarm}")
        if config.use_multi_swarm:
            self.logger.info(f"  Sub-swarms: {config.n_sub_swarms}, Regrouping: {config.regrouping_period}")
        self.logger.info(f"{'='*80}\n")

        # Run PSO
        pso = AdaptivePSO(objective_function, bounds, config, random_seed=self.base_seed,
                          print_metadata=self._print_metadata)
        result = pso.optimize()

        # Convert results
        self.best_params = dict(zip(connection_names, result.best_position))
        self.best_loss = result.best_score
        self.best_metadata = result.best_metadata

        self.history['loss'] = result.history['best_scores']
        self.history['parameters'] = [
            dict(zip(connection_names, result.best_position))
            for _ in range(len(result.history['best_scores']))
        ]

        # Print final diagnostics
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Circuit optimization results")
        self.logger.info(f"{'='*80}")
        self._print_diagnostics(result.best_position, result.best_score, connection_names,
                                metadata=result.best_metadata)
        self.logger.info(f"\nPSO Statistics:")
        self.logger.info(f"  Total evaluations: {result.n_evaluations}")
        self.logger.info(f"  New bests found: {result.n_new_bests}")
        self.logger.info(f"  Final diversity: {result.final_diversity:.6f}")
        if result.convergence_iteration is not None:
            self.logger.info(f"  Converged at iteration: {result.convergence_iteration}")
        self.logger.info(f"  Metadata snapshots: {len(result.metadata_history)}")
        self.logger.info(f"{'='*80}\n")

        return {
            'optimized_connection_modulation': self.best_params,
            'best_loss': self.best_loss,
            'method': 'enhanced_particle_swarm',
            'n_particles': n_particles,
            'n_iterations': max_iterations,
            'device': str(self.device),
            'strategy': strategy.get_strategy_info()['name'],
            'base_seed': self.base_seed,
            'pso_config': config,
            'pso_result': result,
            'history': self.history,
            'targets': self.targets,
            'best_metadata': self.best_metadata,
            'metadata_history': result.metadata_history  # all metadata snapshots
        }
    
    def _optimize_differential_evolution(self, connection_names, max_iterations=50,
                                         n_workers=None, n_threads_per_worker=1, **kwargs):
        """Differential evolution with multiprocess CPU"""
        from scipy.optimize import differential_evolution
        
        strategy = self._select_strategy('differential_evolution',
                                         n_workers=n_workers,
                                         n_threads_per_worker=n_threads_per_worker,
                                         **kwargs)
        
        bounds = [self.targets.connection_bounds.get(name, (0.1, 5.0))
                 for name in connection_names]
        
        self.logger.info(f"Starting Differential Evolution...")
        self.logger.info(f"Iterations: {max_iterations}")
        self.logger.info(f"Trials per evaluation: {self.config.n_trials}")
        
        def objective(param_array):
            param_dict = dict(zip(connection_names, param_array))
            mec_current = self.config.mec_drive_levels[0]
            mec_current_std = self.config.mec_drive_std
            losses, _ = strategy.evaluate_batch(
                [param_dict], mec_current, mec_current_std, self.config.n_trials
            )
            return losses[0]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            popsize=15,
            workers=1,
            disp=True
        )
        
        self.best_params = dict(zip(connection_names, result.x))
        self.best_loss = result.fun
        
        return {
            'optimized_connection_modulation': self.best_params,
            'best_loss': self.best_loss,
            'method': 'differential_evolution',
            'success': result.success,
            'device': str(self.device),
            'strategy': strategy.get_strategy_info()['name'],
            'base_seed': self.base_seed
        }

    def _print_metadata(self, metadata, logger=None):
        if logger is None:
            logger = self.logger
            
        logger.info(f"\n{'='*80}")
        logger.info("Metadata from optimization evaluation")
        logger.info(f"{'='*80}")
        logger.info(f"  Baseline loss: {metadata.get('baseline_loss', 'N/A')}")
        logger.info(f"  Opto loss: {metadata.get('opto_loss', 'N/A')}")

        if 'baseline_rates' in metadata:
            logger.info("\n  Baseline rates from optimization:")
            for pop, rate in metadata['baseline_rates'].items():
                logger.info(f"    {pop.upper()}: {rate:.3f} Hz")

        if 'opto_details' in metadata:
            logger.info("\n  Optogenetic effects from optimization:")
            for target_pop in ['pv', 'sst']:
                if target_pop in metadata['opto_details']:
                    logger.info(f"\n    {target_pop.upper()} stimulation:")
                    for affected_pop in metadata['opto_details'][target_pop]:
                        data = metadata['opto_details'][target_pop][affected_pop]
                        logger.info(f"      {affected_pop.upper()}: activated_fraction = {data.get('activated_fraction', 'N/A')}")
        
    
    def _print_diagnostics(self, position, loss, connection_names, metadata=None):
        """Print detailed diagnostics for a configuration"""
        self.logger.info(f"\n{'#'*80}")
        self.logger.info("NEW BEST SOLUTION FOUND")
        self.logger.info(f"{'#'*80}")
        self.logger.info(f"Loss: {loss:.6f}\n")
        
        param_dict = dict(zip(connection_names, position))
        self.logger.info("Connection modulation parameters:")
        for name, value in param_dict.items():
            self.logger.info(f"  {name}: {value:.3f}")

        # Show metadata from optimization if available
        if metadata is not None:
            self._print_metadata(metadata)
            
        # Detailed evaluation with verbose output
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Detailed evaluation of candidate parameters")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Base seed: {self.base_seed} (trials use base_seed + trial_idx)")
        
        # Create evaluator for detailed diagnostics
        evaluator = BatchOptogeneticEvaluator(
            self.circuit_params, self.base_synaptic_params,
            self.opsin_params, self.opsin_current,
            self.targets, self.config, device=self.device, base_seed=self.base_seed,
            verbose=True
        )
        
        # Evaluate configuration
        mec_current = self.config.mec_drive_levels[0]
        mec_current_std = self.config.mec_drive_std
        total_loss_tensor, details = evaluator.evaluate_parameter_batch(
            [param_dict], mec_current, mec_current_std
        )
        recomputed_loss = total_loss_tensor.item()
        
        # Extract components
        baseline_loss = details['baseline_losses'].item()
        opto_loss = details['opto_losses'].item()
        baseline_rates = details['baseline_rates']
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Baseline circuit evaluation")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"  MEC drive: {mec_current} pA\n")
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop in baseline_rates:
                actual_rate = baseline_rates[pop].item()
                target_rate = self.targets.target_rates.get(pop, 0.0)
                tolerance = self.targets.rate_tolerance.get(pop, 0.0)
                error = abs(actual_rate - target_rate)
                
                self.logger.info(f"  {pop.upper()}:")
                self.logger.info(f"    Target: {target_rate:.3f} Hz +/- {tolerance:.3f}")
                self.logger.info(f"    Actual: {actual_rate:.3f} Hz")
                self.logger.info(f"    Error:  {error:.3f} Hz")
        
        self.logger.info(f"\n  Baseline loss: {baseline_loss:.6f}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Optogenetic stimulation evaluation")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"  Note: Averaged over {self.config.n_trials} trials with different connectivity")
        
        opto_details = details['opto_details']
        for target_pop in ['pv', 'sst']:
            if target_pop in opto_details:
                self.logger.info(f"\n--- {target_pop.upper()} Stimulation ---")
                pop_results = opto_details[target_pop]
                
                for affected_pop in ['gc', 'mc', 'pv', 'sst']:
                    if affected_pop in pop_results and affected_pop != target_pop:
                        result = pop_results[affected_pop]
                        
                        baseline_mean = result['baseline_mean'].item()
                        stim_mean = result['stim_mean'].item()
                        mean_change = result['mean_change'].item()
                        activated_fraction = result['activated_fraction'].item()
                        gini_change = result['gini_change'].item()
                        
                        self.logger.info(f"\n  {affected_pop.upper()}:")
                        self.logger.info(f"    Baseline rate: {baseline_mean:.3f} Hz")
                        self.logger.info(f"    Stim rate: {stim_mean:.3f} Hz")
                        self.logger.info(f"    Mean change: {mean_change:+.3f} Hz")
                        self.logger.info(f"    Activated fraction: {activated_fraction:.3f}")
                        
                        # Show target if available
                        opto_targets = self.targets.optogenetic_targets
                        if (target_pop in opto_targets.target_rate_increases and 
                            affected_pop in opto_targets.target_rate_increases[target_pop]):
                            target_frac = opto_targets.target_rate_increases[target_pop][affected_pop]
                            self.logger.info(f"      (target: {target_frac:.3f})")
                        
                        self.logger.info(f"    Gini change: {gini_change:+.4f}")
                        if (target_pop in opto_targets.target_gini_increase and
                            affected_pop in opto_targets.target_gini_increase[target_pop]):
                            target_gini = opto_targets.target_gini_increase[target_pop][affected_pop]
                            self.logger.info(f"      (target: {target_gini:+.4f})")
        
        self.logger.info(f"\n  Total optogenetic loss: {opto_loss:.6f}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Combined loss breakdown")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"  Baseline: {baseline_loss:.6f} x {self.targets.baseline_weight} = {baseline_loss * self.targets.baseline_weight:.6f}")
        self.logger.info(f"  Optogenetic: {opto_loss:.6f} x {self.targets.optogenetic_weight} = {opto_loss * self.targets.optogenetic_weight:.6f}")
        self.logger.info(f"  TOTAL: {recomputed_loss:.6f}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Loss verification")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"  Loss from optimizer:  {loss:.6f}")
        self.logger.info(f"  Recomputed loss:      {recomputed_loss:.6f}")
        if abs(loss - recomputed_loss) > 0.1:
            self.logger.info(f"  WARNING: Loss mismatch!")
        self.logger.info(f"{'='*80}\n")
    
    def save_results(self, filename: str):
        """Save optimization results to JSON"""
        if self.best_params is None:
            self.logger.info("No results to save")
            return
        
        results = {
            'optimization_info': {
                'timestamp': datetime.now().isoformat(),
                'best_loss': float(self.best_loss),
                'device': str(self.device),
                'base_seed': self.base_seed,
                'baseline_weight': self.targets.baseline_weight,
                'optogenetic_weight': self.targets.optogenetic_weight,
                'n_trials': self.config.n_trials,
                'mec_current': self.config.mec_drive_levels[0],
                'mec_current_std': self.config.mec_drive_std
                
            },
            'targets': {
                'baseline_rates': self.targets.target_rates,
                'sparsity_targets': self.targets.sparsity_targets,
                'optogenetic_targets': {
                    'rate_increases': self.targets.optogenetic_targets.target_rate_increases,
                    'gini_increases': self.targets.optogenetic_targets.target_gini_increase
                }
            },
            'optimized_parameters': {
                'connection_modulation': self.best_params,
                'base_conductances': {
                    'ampa_g_mean': self.base_synaptic_params.ampa_g_mean,
                    'ampa_g_std': self.base_synaptic_params.ampa_g_std,
                    'ampa_g_min': self.base_synaptic_params.ampa_g_min,
                    'ampa_g_max': self.base_synaptic_params.ampa_g_max,
                    
                    'gaba_g_mean': self.base_synaptic_params.gaba_g_mean,
                    'gaba_g_std': self.base_synaptic_params.gaba_g_std,
                    'gaba_g_min': self.base_synaptic_params.gaba_g_min,
                    'gaba_g_max': self.base_synaptic_params.gaba_g_max
                }
            },
            'best_configuration_metadata': self.best_metadata,  # Include metadata
            'history': self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"\nResults saved to {filename}")


# ============================================================================
# Convenience Functions
# ============================================================================

def run_global_optimization(optimization_config,
                            device: Optional[torch.device] = None,
                            n_workers=1,
                            n_threads_per_worker=1,
                            method='particle_swarm',
                            n_particles=20,
                            max_iterations=50,
                            diagnostic_frequency=5,
                            base_seed=42,
                            output_file="DG_optogenetic_optimization_results.json",
                            adaptive_step: bool = False,
                            adaptive_config: Optional[GradientAdaptiveStepConfig] = None,
                            opsin_current: float = 150.0,
                            # MEC pattern parameters
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
                            mec_rotation_groups: int = 3):
    """
    Run global optimization with optogenetic objectives using batch GPU interface
    
    Supports base_seed parameter for reproducible trials.
    
    Args:
        optimization_config: OptimizationConfig instance
        device: Device to run on (None for auto-detect, 'cpu', or 'cuda')
        n_workers: Number of parallel workers (for CPU)
        n_threads_per_worker: Threads per worker
        method: 'particle_swarm' or 'differential_evolution'
        n_particles: Number of particles for PSO
        max_iterations: Maximum iterations
        diagnostic_frequency: How often to print detailed diagnostics
        base_seed: Base random seed for reproducibility (default: 42)
        adaptive_step: If True, use gradient-driven adaptive time stepping
        adaptive_config: Configuration for adaptive stepping (optional)
        # MEC pattern parameters
        use_time_varying_mec: If True, use time-varying MEC input patterns
        mec_pattern_type: Type of temporal pattern ('oscillatory', 'drift', 'constant')
        mec_theta_freq: Theta oscillation frequency in Hz (default: 5.0)
        mec_theta_amplitude: Theta modulation depth 0-1 (default: 0.3)
        mec_gamma_freq: Gamma oscillation frequency in Hz (default: 20.0)
        mec_gamma_amplitude: Gamma modulation depth 0-1 (default: 0.15)
        mec_gamma_coupling_strength: Gamma-theta coupling 0-1 (default: 0.8)
        mec_gamma_preferred_phase: Preferred theta phase for gamma peak in radians (default: 0.0)
        mec_drift_timescale: Correlation time for drift pattern in ms (default: 200.0)
        mec_drift_amplitude: Drift amplitude relative to base 0-1 (default: 0.4)
        mec_rotation_groups: Number of groups for spatial rotation across trials (default: 3)

    """

    # Set adaptive stepping in config
    if hasattr(optimization_config, 'adaptive_step'):
        optimization_config.adaptive_step = adaptive_step
    if hasattr(optimization_config, 'adaptive_config'):
        optimization_config.adaptive_config = adaptive_config
        
    # Device setup
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    print(f"Optimization device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

    # Print MEC pattern configuration
    if use_time_varying_mec:
        print(f"\nMEC Input Configuration:")
        print(f"  Pattern type: {mec_pattern_type}")
        print(f"  Theta: {mec_theta_freq} Hz (amplitude: {mec_theta_amplitude})")
        print(f"  Gamma: {mec_gamma_freq} Hz (amplitude: {mec_gamma_amplitude})")
        print(f"  Gamma-theta coupling: {mec_gamma_coupling_strength}")
        print(f"  Spatial rotation: {mec_rotation_groups} groups")
    else:
        print(f"\nMEC Input: Constant + noise (traditional)")
        
    sys.stdout.flush()
        
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
        optogenetic_weight=2.0
    )
    
    # Create optimizer
    optimizer = OptogeneticCircuitOptimizer(
        circuit_params, base_synaptic_params, opsin_params, opsin_current,
        targets, optimization_config, device=device, base_seed=base_seed
    )
    
    # Run optimization
    results = optimizer.optimize(
        method=method,
        n_particles=n_particles,
        max_iterations=max_iterations,
        n_workers=n_workers,
        n_threads_per_worker=n_threads_per_worker,
        diagnostic_frequency=diagnostic_frequency,
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
    
    # Save results
    optimizer.save_results(output_file)
    
    return results, optimizer


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
    parser.add_argument('--base-seed', type=int, default=42, help='Base random seed for reproducibility')
    parser.add_argument('--output-file', type=str, default='DG_optogenetic_optimization_results.json',
                       help='Output file (default: DG_optogenetic_optimization_results.json)')
    # Adaptive stepping arguments
    parser.add_argument('--adaptive-step', action='store_true',
                       help='Use gradient-driven adaptive time stepping')
    parser.add_argument('--adaptive-dt-min', type=float, default=0.05,
                       help='Minimum time step for adaptive stepping (ms)')
    parser.add_argument('--adaptive-dt-max', type=float, default=0.3,
                       help='Maximum time step for adaptive stepping (ms)')
    parser.add_argument('--adaptive-gradient-low', type=float, default=0.5,
                       help='Low gradient threshold (Hz/ms)')
    parser.add_argument('--adaptive-gradient-high', type=float, default=10.0,
                       help='High gradient threshold (Hz/ms)')
    # MEC pattern arguments
    parser.add_argument('--time-varying-mec', action='store_true',
                       help='Enable time-varying MEC input patterns')
    parser.add_argument('--mec-pattern-type', type=str, default='oscillatory',
                       choices=['oscillatory', 'drift', 'constant'],
                       help='Type of temporal pattern for MEC input (default: oscillatory)')
    parser.add_argument('--mec-theta-freq', type=float, default=5.0,
                       help='Theta oscillation frequency in Hz (default: 5.0)')
    parser.add_argument('--mec-theta-amplitude', type=float, default=0.3,
                       help='Theta modulation depth 0-1 (default: 0.3)')
    parser.add_argument('--mec-gamma-freq', type=float, default=20.0,
                       help='Gamma oscillation frequency in Hz (default: 20.0)')
    parser.add_argument('--mec-gamma-amplitude', type=float, default=0.15,
                       help='Gamma modulation depth 0-1 (default: 0.15)')
    parser.add_argument('--mec-gamma-coupling', type=float, default=0.8,
                       help='Gamma-theta coupling strength 0-1 (default: 0.8)')
    parser.add_argument('--mec-gamma-phase', type=float, default=0.0,
                       help='Preferred theta phase for gamma peak in radians (default: 0.0)')
    parser.add_argument('--mec-drift-timescale', type=float, default=200.0,
                       help='Correlation time for drift pattern in ms (default: 200.0)')
    parser.add_argument('--mec-drift-amplitude', type=float, default=0.4,
                       help='Drift amplitude relative to base 0-1 (default: 0.4)')
    parser.add_argument('--mec-rotation-groups', type=int, default=3,
                       help='Number of groups for spatial rotation (default: 3)')

    
    args = parser.parse_args()

    # Create adaptive step config if requested
    adaptive_step_config = None
    if args.adaptive_step:
        adaptive_step_config = GradientAdaptiveStepConfig(
            dt_min=args.adaptive_dt_min,
            dt_max=args.adaptive_dt_max,
            gradient_low=args.adaptive_gradient_low,
            gradient_high=args.adaptive_gradient_high
        )
        
    config = create_default_global_opt_config()
    
    results, optimizer = run_global_optimization(
        config,
        device=args.device,
        n_workers=args.n_workers,
        n_threads_per_worker=args.n_threads,
        method=args.method,
        n_particles=args.n_particles,
        max_iterations=args.max_iterations,
        base_seed=args.base_seed,
        output_file=args.output_file,
        adaptive_step=args.adaptive_step,
        adaptive_config=adaptive_step_config,
        # MEC pattern parameters
        use_time_varying_mec=args.time_varying_mec,
        mec_pattern_type=args.mec_pattern_type,
        mec_theta_freq=args.mec_theta_freq,
        mec_theta_amplitude=args.mec_theta_amplitude,
        mec_gamma_freq=args.mec_gamma_freq,
        mec_gamma_amplitude=args.mec_gamma_amplitude,
        mec_gamma_coupling_strength=args.mec_gamma_coupling,
        mec_gamma_preferred_phase=args.mec_gamma_phase,
        mec_drift_timescale=args.mec_drift_timescale,
        mec_drift_amplitude=args.mec_drift_amplitude,
        mec_rotation_groups=args.mec_rotation_groups
    )
    
    print(f"\nOptimization Complete!")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Device: {results['device']}")
    print(f"Base seed: {results['base_seed']}")
    print(f"Output file: {args.output_file}")
