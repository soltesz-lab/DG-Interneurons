#!/usr/bin/env python3
"""
Optimization framework including optogenetic stimulation effects

Adds optogenetic stimulation objectives to the circuit optimization,
targeting the paradoxical excitation effects observed by Hainmueller et al.

Updated to use batch evaluation GPU interface consistent with DG_circuit_optimization.py.
Implements EvaluationStrategy pattern for automatic device-appropriate strategy selection.
"""
import sys
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

# Import existing optimization components
from DG_circuit_optimization import (
    OptimizationTargets,
    OptimizationConfig,
    evaluate_rate_ordering_constraints,
    configure_torch_threads,
    get_default_device,
    create_default_targets
)

from DG_batch_circuit_dendritic_somatic_transfer import (
    BatchDentateCircuit
)

from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit, CircuitParams, OpsinParams, PerConnectionSynapticParams
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


# ============================================================================
# Batch Optogenetic Evaluator
# ============================================================================

class BatchOptogeneticEvaluator:
    """
    Evaluates multiple parameter configurations with optogenetic experiments in parallel
    
    Extends batch circuit evaluation to include optogenetic stimulation protocols.
    Computes both baseline circuit objectives and optogenetic effect objectives.
    """
    
    def __init__(self,
                 circuit_params: CircuitParams,
                 base_synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 targets: CombinedOptimizationTargets,
                 config: OptimizationConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize batch optogenetic evaluator
        
        Args:
            circuit_params: CircuitParams instance
            base_synaptic_params: PerConnectionSynapticParams instance
            opsin_params: OpsinParams instance
            targets: CombinedOptimizationTargets instance
            config: OptimizationConfig instance
            device: Device to run simulations on
        """
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        self.device = device if device is not None else get_default_device()
        
        print(f"BatchOptogeneticEvaluator initialized on device: {self.device}")
    
    def evaluate_parameter_batch(self,
                                 parameter_batch: List[Dict[str, float]],
                                 mec_drive: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Evaluate a batch of parameter configurations with full optogenetic protocol
        
        Args:
            parameter_batch: List of connection_modulation dicts, length = batch_size
            mec_drive: MEC drive level (pA)
            
        Returns:
            losses: Tensor of shape [batch_size] with total loss for each configuration
            details: Dict with baseline_losses, opto_losses, and firing_rates_batch
        """
        batch_size = len(parameter_batch)
        
        # === BASELINE EVALUATION ===
        baseline_losses, baseline_rates = self._evaluate_baseline_batch(
            parameter_batch, mec_drive
        )
        
        # === OPTOGENETIC EVALUATION ===
        opto_losses, opto_details = self._evaluate_optogenetic_batch(
            parameter_batch, mec_drive
        )
        
        # === COMBINE LOSSES ===
        total_losses = (self.targets.baseline_weight * baseline_losses + 
                       self.targets.optogenetic_weight * opto_losses)
        
        return total_losses, {
            'baseline_losses': baseline_losses,
            'opto_losses': opto_losses,
            'baseline_rates': baseline_rates,
            'opto_details': opto_details
        }
    
    def _evaluate_baseline_batch(self,
                                 parameter_batch: List[Dict[str, float]],
                                 mec_drive: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate baseline circuit objectives for batch of parameters
        
        Returns:
            losses: Tensor [batch_size]
            firing_rates_batch: Dict mapping pop -> rates [batch_size]
        """
        batch_size = len(parameter_batch)
        
        # Create batch circuit
        circuit = BatchDentateCircuit(
            batch_size=batch_size,
            circuit_params=self.circuit_params,
            synaptic_params=self.base_synaptic_params,
            opsin_params=self.opsin_params,
            device=self.device
        )
        
        # Set per-batch connection modulation
        circuit.set_connection_modulation_batch(parameter_batch)
        
        # Average over trials
        total_losses = torch.zeros(batch_size, device=self.device)
        all_firing_rates = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        
        for trial in range(self.config.n_trials):
            circuit.reset_state()
            
            # MEC input: [batch_size, n_mec]
            mec_input = torch.ones(batch_size, self.circuit_params.n_mec,
                                  device=self.device) * mec_drive
            
            # Collect activities over time
            activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
            
            for t in range(self.config.simulation_duration):
                external_drive = {'mec': mec_input}
                activities = circuit({}, external_drive)
                
                if t >= self.config.warmup_duration:
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
        
        # Average over trials
        total_losses /= self.config.n_trials
        avg_firing_rates = {pop: torch.stack(all_firing_rates[pop]).mean(dim=0)
                           for pop in all_firing_rates if len(all_firing_rates[pop]) > 0}
        
        return total_losses, avg_firing_rates
    
    def _evaluate_optogenetic_batch(self,
                                    parameter_batch: List[Dict[str, float]],
                                    mec_drive: float) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluate optogenetic objectives for batch of parameters
        
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
        
        # Run multiple trials
        for trial in range(self.config.n_trials):
            trial_losses = torch.zeros(batch_size, device=self.device)
            
            # Evaluate PV and SST stimulation
            for target_pop in ['pv', 'sst']:
                # Run stimulation experiment for batch
                stim_results = self._simulate_batch_optogenetic_stimulation(
                    parameter_batch, target_pop, mec_drive
                )
                
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
    
    def _simulate_batch_optogenetic_stimulation(self,
                                               parameter_batch: List[Dict[str, float]],
                                               target_pop: str,
                                               mec_drive: float) -> Dict[str, torch.Tensor]:
        """
        Run optogenetic stimulation for a batch of parameter sets
        
        Returns dict with baseline and stimulation statistics for each population
        """
        from DG_protocol import OpsinExpression
        
        batch_size = len(parameter_batch)
        
        # Stimulation protocol parameters
        stim_start = 550
        duration = 1650
        warmup = 150
        opsin_current = 200.0
        
        # Create batch circuit
        circuit = BatchDentateCircuit(
            batch_size=batch_size,
            circuit_params=self.circuit_params,
            synaptic_params=self.base_synaptic_params,
            opsin_params=self.opsin_params,
            device=self.device
        )
        
        circuit.set_connection_modulation_batch(parameter_batch)
        
        # Setup opsin expression for target population ONLY
        from DG_protocol import OpsinExpression
        
        n_target_cells = getattr(self.circuit_params, f'n_{target_pop}')
        opsin_expression = OpsinExpression(
            self.opsin_params,
            n_cells=n_target_cells,
            device=self.device
        )
        
        # Get positions for target population
        target_positions = circuit.layout.positions[target_pop]
        
        # Calculate activation probability for target population neurons
        # This returns a tensor of shape [n_target_cells] with activation probabilities
        light_intensity = self.targets.optogenetic_targets.stimulation_intensity
        activation_prob = opsin_expression.calculate_activation(target_positions, light_intensity)
        
        # Convert activation probabilities to optogenetic current for target neurons
        # Shape: [n_target_cells]
        target_opto_current = activation_prob * opsin_current
        
        # Storage for time series: [time, batch, neurons]
        time_points = []
        activities_time_series = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        
        # Run simulation
        circuit.reset_state()
        mec_input = torch.ones(batch_size, self.circuit_params.n_mec,
                              device=self.device) * mec_drive
        
        for t in range(duration):
            # Create per-population optogenetic drives
            # Only target_pop receives non-zero drive
            direct_activation = {}
            
            if t >= stim_start:
                # Apply optogenetic stimulation only to target population
                for pop, n_neurons in [('gc', self.circuit_params.n_gc),
                                       ('mc', self.circuit_params.n_mc),
                                       ('pv', self.circuit_params.n_pv),
                                       ('sst', self.circuit_params.n_sst)]:
                    if pop == target_pop:
                        # Target population gets optogenetic drive
                        # Replicate across batch: [n_neurons] -> [batch_size, n_neurons]
                        direct_activation[pop] = target_opto_current.unsqueeze(0).expand(batch_size, -1)
                    else:
                        # Non-target populations get zero drive
                        direct_activation[pop] = torch.zeros(batch_size, n_neurons,
                                                            device=self.device)
            else:
                # No stimulation before stim_start - all populations get zero drive
                for pop, n_neurons in [('gc', self.circuit_params.n_gc),
                                       ('mc', self.circuit_params.n_mc),
                                       ('pv', self.circuit_params.n_pv),
                                       ('sst', self.circuit_params.n_sst)]:
                    direct_activation[pop] = torch.zeros(batch_size, n_neurons,
                                                        device=self.device)
            
            external_drive = {'mec': mec_input}
            activities = circuit(direct_activation, external_drive)
            
            if t >= warmup:
                time_points.append(t)
                for pop in activities_time_series:
                    if pop in activities:
                        activities_time_series[pop].append(activities[pop])
        
        # Convert to tensors: [time, batch, neurons]
        for pop in activities_time_series:
            if len(activities_time_series[pop]) > 0:
                activities_time_series[pop] = torch.stack(activities_time_series[pop], dim=0)
        
        time_tensor = torch.tensor(time_points, device=self.device)
        
        # Calculate statistics
        baseline_mask = time_tensor < stim_start
        stim_mask = time_tensor >= stim_start
        
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
                    errors = (actual_fraction - target_fraction) ** 2
                    
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
                    errors = (actual_gini_change - target_gini_change) ** 2
                    losses += errors
        
        return losses


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
                      mec_drive: float,
                      n_trials: int) -> Tuple[List[float], Dict]:
        """
        Evaluate a batch of parameter configurations with optogenetic protocols
        
        Args:
            parameter_sets: List of connection_modulation dicts
            mec_drive: MEC drive level
            n_trials: Number of trials to average
            
        Returns:
            losses: List of total loss values
            details: Dict with loss breakdowns and firing rates
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, any]:
        """Return information about this strategy"""
        pass


class OptogeneticSequentialStrategy(OptogeneticEvaluationStrategy):
    """Sequential evaluation for gradient-based or single evaluations"""
    
    def __init__(self, circuit_params, base_synaptic_params, opsin_params,
                 targets, config, device):
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        self.device = device
    
    def evaluate_batch(self, parameter_sets, mec_drive, n_trials):
        """Evaluate configurations one at a time"""
        losses = []
        all_details = []
        
        for params in parameter_sets:
            # Use batch evaluator with batch_size=1
            evaluator = BatchOptogeneticEvaluator(
                self.circuit_params, self.base_synaptic_params, self.opsin_params,
                self.targets, self.config, device=self.device
            )
            
            loss_tensor, details = evaluator.evaluate_parameter_batch([params], mec_drive)
            losses.append(loss_tensor.item())
            all_details.append(details)
        
        return losses, all_details
    
    def get_strategy_info(self):
        return {
            'name': 'OptogeneticSequential',
            'device': str(self.device),
            'parallelism': 'None',
            'description': 'Sequential evaluation with optogenetic protocols'
        }


class OptogeneticBatchGPUStrategy(OptogeneticEvaluationStrategy):
    """Batch GPU evaluation for population-based optimization"""
    
    def __init__(self, circuit_params, base_synaptic_params, opsin_params,
                 targets, config, device):
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        self.device = device
        
        # Create evaluator
        self.evaluator = BatchOptogeneticEvaluator(
            circuit_params, base_synaptic_params, opsin_params,
            targets, config, device=device
        )
    
    def evaluate_batch(self, parameter_sets, mec_drive, n_trials):
        """Evaluate all configurations in parallel on GPU"""
        batch_size = len(parameter_sets)
        
        # Note: n_trials is handled internally by evaluator
        # We just call once since BatchOptogeneticEvaluator handles trials
        losses_tensor, details = self.evaluator.evaluate_parameter_batch(
            parameter_sets, mec_drive
        )
        
        # Convert to lists for consistent API
        losses_list = losses_tensor.cpu().numpy().tolist()
        
        return losses_list, [details] * batch_size
    
    def get_strategy_info(self):
        return {
            'name': 'OptogeneticBatchGPU',
            'device': str(self.device),
            'parallelism': 'Data parallelism (batched optogenetic experiments)',
            'description': f'Batch optogenetic evaluation on GPU'
        }


class OptogeneticMultiprocessCPUStrategy(OptogeneticEvaluationStrategy):
    """Multiprocess CPU evaluation for population-based optimization"""
    
    def __init__(self, circuit_params, base_synaptic_params, opsin_params,
                 targets, config, device, n_workers=None, n_threads_per_worker=1):
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        self.device = device
        
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
            opsin_params
        )
    
    def evaluate_batch(self, parameter_sets, mec_drive, n_trials):
        """Evaluate configurations using multiprocessing"""
        # Prepare arguments for workers
        eval_args = [
            (params, mec_drive, self.worker_data, self.targets, self.config)
            for params in parameter_sets
        ]
        
        configure_torch_threads(self.n_threads_per_worker)
        
        # Use multiprocessing pool
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.n_workers) as pool:
            results = pool.map(_optogenetic_worker_evaluate, eval_args)
        
        # Unpack results
        losses = [r[0] for r in results]
        details = [r[1] for r in results]
        
        return losses, details
    
    def get_strategy_info(self):
        return {
            'name': 'OptogeneticMultiprocessCPU',
            'device': str(self.device),
            'parallelism': f'{self.n_workers} workers * {self.n_threads_per_worker} threads',
            'description': f'Multiprocess optogenetic evaluation on CPU with {self.n_workers} workers'
        }


def _optogenetic_worker_evaluate(args):
    """Worker function for multiprocess optogenetic evaluation"""
    (connection_modulation, mec_drive, worker_data, targets, config) = args
    
    device = torch.device('cpu')
    circuit_params, base_synaptic_params_dict, opsin_params = worker_data
    
    # Create synaptic params
    synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params_dict['ampa_g_mean'],
        ampa_g_std=base_synaptic_params_dict['ampa_g_std'],
        gaba_g_mean=base_synaptic_params_dict['gaba_g_mean'],
        gaba_g_std=base_synaptic_params_dict['gaba_g_std'],
        distribution=base_synaptic_params_dict['distribution'],
        connection_modulation=connection_modulation
    )
    
    # Create evaluator
    evaluator = BatchOptogeneticEvaluator(
        circuit_params, synaptic_params, opsin_params,
        targets, config, device=device
    )
    
    # Evaluate single parameter set
    loss_tensor, details = evaluator.evaluate_parameter_batch(
        [connection_modulation], mec_drive
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
    
    Extends CircuitOptimizer pattern to include optogenetic objectives.
    Automatically chooses optimal strategy based on device and method.
    """
    
    def __init__(self,
                 circuit_params: CircuitParams,
                 base_synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 targets: CombinedOptimizationTargets,
                 config: OptimizationConfig,
                 device: Optional[torch.device] = None):
        
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        
        # Set device
        self.device = device if device is not None else get_default_device()
        self.config.device = self.device
        
        # Storage
        self.best_loss = float('inf')
        self.best_params = None
        self.best_iteration = 0
        self.history = {'loss': [], 'parameters': []}
        
        print(f"OptogeneticCircuitOptimizer initialized on device: {self.device}")
        print(f"  Baseline weight: {targets.baseline_weight}")
        print(f"  Optogenetic weight: {targets.optogenetic_weight}")
    
    def _select_strategy(self, method: str, **kwargs) -> OptogeneticEvaluationStrategy:
        """Select optimal evaluation strategy based on method and device"""
        
        if method == 'gradient':
            strategy = OptogeneticSequentialStrategy(
                self.circuit_params, self.base_synaptic_params, self.opsin_params,
                self.targets, self.config, self.device
            )
        
        elif method in ['particle_swarm', 'genetic_algorithm']:
            if self.device.type == 'cuda':
                strategy = OptogeneticBatchGPUStrategy(
                    self.circuit_params, self.base_synaptic_params, self.opsin_params,
                    self.targets, self.config, self.device
                )
            else:
                strategy = OptogeneticMultiprocessCPUStrategy(
                    self.circuit_params, self.base_synaptic_params, self.opsin_params,
                    self.targets, self.config, self.device,
                    n_workers=kwargs.get('n_workers', None),
                    n_threads_per_worker=kwargs.get('n_threads_per_worker', 1)
                )
        
        elif method == 'differential_evolution':
            if self.device.type == 'cuda':
                raise NotImplementedError(
                    "Differential Evolution on GPU not supported (scipy limitation). "
                    "Use CPU or try 'particle_swarm' for GPU optimization."
                )
            strategy = OptogeneticMultiprocessCPUStrategy(
                self.circuit_params, self.base_synaptic_params, self.opsin_params,
                self.targets, self.config, self.device,
                n_workers=kwargs.get('n_workers', None),
                n_threads_per_worker=kwargs.get('n_threads_per_worker', 1)
            )
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Print strategy info
        info = strategy.get_strategy_info()
        print(f"\nEvaluation Strategy: {info['name']}")
        print(f"  Device: {info['device']}")
        print(f"  Parallelism: {info['parallelism']}")
        print(f"  Description: {info['description']}\n")
        
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
                                 diagnostic_frequency=5):
        """Particle swarm optimization with automatic strategy"""
        strategy = self._select_strategy('particle_swarm',
                                         n_workers=n_workers,
                                         n_threads_per_worker=n_threads_per_worker)
        
        n_dimensions = len(connection_names)
        bounds = [self.targets.connection_bounds.get(name, (0.1, 5.0))
                 for name in connection_names]
        
        print(f"Starting Particle Swarm Optimization...")
        print(f"Particles: {n_particles}")
        print(f"Iterations: {max_iterations}")
        
        # Initialize
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, n_dimensions))
        velocities = np.random.randn(n_particles, n_dimensions) * 0.1
        
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(n_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        
        # PSO parameters
        w_max, w_min, w = 0.9, 0.2, 0.9
        
        n_new_bests = 0
        previous_best = float('inf')
        no_improvement = 0
        
        for iteration in range(max_iterations):
            print(f"\nPSO Iteration {iteration+1}/{max_iterations}")
            c1, c2 = np.random.uniform(1.5, 2.5), np.random.uniform(1.5, 2.5)
            
            # Convert positions to parameter dicts
            parameter_sets = []
            for i in range(n_particles):
                param_dict = dict(zip(connection_names, positions[i]))
                parameter_sets.append(param_dict)
            
            # Evaluate all particles
            mec_drive = self.config.mec_drive_levels[0]
            losses, _ = strategy.evaluate_batch(
                parameter_sets, mec_drive, self.config.n_trials
            )
            losses = np.array(losses)
            
            # Update personal bests
            improved = losses < personal_best_scores
            personal_best_scores[improved] = losses[improved]
            personal_best_positions[improved] = positions[improved]
            
            # Update global best
            min_idx = np.argmin(losses)
            if losses[min_idx] < global_best_score:
                global_best_score = losses[min_idx]
                global_best_position = positions[min_idx].copy()
                self.best_params = dict(zip(connection_names, global_best_position))
                print(f"\n  New best: {global_best_score:.6f}")
                
                if n_new_bests % diagnostic_frequency == 0:
                    self._print_diagnostics(global_best_position, global_best_score, connection_names)
                
                n_new_bests += 1
            
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
                r1, r2 = np.random.random(n_dimensions), np.random.random(n_dimensions)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] = np.clip(positions[i] + velocities[i], lower_bounds, upper_bounds)
            
            self.history['loss'].append(global_best_score)
            self.history['parameters'].append(self.best_params.copy())
        
        print("\nFinal diagnostics:")
        self._print_diagnostics(global_best_position, global_best_score, connection_names)
        
        self.best_loss = global_best_score
        
        return {
            'optimized_connection_modulation': self.best_params,
            'best_loss': self.best_loss,
            'method': 'particle_swarm',
            'n_particles': n_particles,
            'device': str(self.device),
            'strategy': strategy.get_strategy_info()['name'],
            'history': self.history,
            'targets': self.targets
        }
    
    def _optimize_differential_evolution(self, connection_names, max_iterations=50,
                                         n_workers=None, n_threads_per_worker=1):
        """Differential evolution with multiprocess CPU"""
        from scipy.optimize import differential_evolution
        
        strategy = self._select_strategy('differential_evolution',
                                         n_workers=n_workers,
                                         n_threads_per_worker=n_threads_per_worker)
        
        bounds = [self.targets.connection_bounds.get(name, (0.1, 5.0))
                 for name in connection_names]
        
        print(f"Starting Differential Evolution...")
        print(f"Iterations: {max_iterations}")
        
        def objective(param_array):
            param_dict = dict(zip(connection_names, param_array))
            mec_drive = self.config.mec_drive_levels[0]
            losses, _ = strategy.evaluate_batch(
                [param_dict], mec_drive, self.config.n_trials
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
            'strategy': strategy.get_strategy_info()['name']
        }
    
    def _print_diagnostics(self, position, loss, connection_names):
        """Print detailed diagnostics for a configuration"""
        print(f"\n{'#'*80}")
        print("NEW BEST SOLUTION FOUND")
        print(f"{'#'*80}")
        print(f"Loss: {loss:.6f}\n")
        
        param_dict = dict(zip(connection_names, position))
        print("Connection modulation parameters:")
        for name, value in param_dict.items():
            print(f"  {name}: {value:.3f}")
        
        # Detailed evaluation with verbose output
        print(f"\n{'='*80}")
        print("Detailed evaluation of candidate parameters")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        
        # Create evaluator for detailed diagnostics
        evaluator = BatchOptogeneticEvaluator(
            self.circuit_params, self.base_synaptic_params, self.opsin_params,
            self.targets, self.config, device=self.device
        )
        
        # Evaluate configuration
        mec_drive = self.config.mec_drive_levels[0]
        total_loss_tensor, details = evaluator.evaluate_parameter_batch(
            [param_dict], mec_drive
        )
        recomputed_loss = total_loss_tensor.item()
        
        # Extract components
        baseline_loss = details['baseline_losses'].item()
        opto_loss = details['opto_losses'].item()
        baseline_rates = details['baseline_rates']
        
        print(f"\n{'='*80}")
        print("Baseline circuit evaluation")
        print(f"{'='*80}")
        print(f"  MEC drive: {mec_drive} pA\n")
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop in baseline_rates:
                actual_rate = baseline_rates[pop].item()
                target_rate = self.targets.target_rates.get(pop, 0.0)
                tolerance = self.targets.rate_tolerance.get(pop, 0.0)
                error = abs(actual_rate - target_rate)
                
                print(f"  {pop.upper()}:")
                print(f"    Target: {target_rate:.3f} Hz ± {tolerance:.3f}")
                print(f"    Actual: {actual_rate:.3f} Hz")
                print(f"    Error:  {error:.3f} Hz")
        
        print(f"\n  Baseline loss: {baseline_loss:.6f}")
        
        print(f"\n{'='*80}")
        print("Optogenetic stimulation evaluation")
        print(f"{'='*80}")
        
        opto_details = details['opto_details']
        for target_pop in ['pv', 'sst']:
            if target_pop in opto_details:
                print(f"\n--- {target_pop.upper()} Stimulation ---")
                pop_results = opto_details[target_pop]
                
                for affected_pop in ['gc', 'mc', 'pv', 'sst']:
                    if affected_pop in pop_results and affected_pop != target_pop:
                        result = pop_results[affected_pop]
                        
                        baseline_mean = result['baseline_mean'].item()
                        stim_mean = result['stim_mean'].item()
                        mean_change = result['mean_change'].item()
                        activated_fraction = result['activated_fraction'].item()
                        gini_change = result['gini_change'].item()
                        
                        print(f"\n  {affected_pop.upper()}:")
                        print(f"    Baseline rate: {baseline_mean:.3f} Hz")
                        print(f"    Stim rate: {stim_mean:.3f} Hz")
                        print(f"    Mean change: {mean_change:+.3f} Hz")
                        print(f"    Activated fraction: {activated_fraction:.3f}")
                        
                        # Show target if available
                        opto_targets = self.targets.optogenetic_targets
                        if (target_pop in opto_targets.target_rate_increases and 
                            affected_pop in opto_targets.target_rate_increases[target_pop]):
                            target_frac = opto_targets.target_rate_increases[target_pop][affected_pop]
                            print(f"      (target: {target_frac:.3f})")
                        
                        print(f"    Gini change: {gini_change:+.4f}")
                        if (target_pop in opto_targets.target_gini_increase and
                            affected_pop in opto_targets.target_gini_increase[target_pop]):
                            target_gini = opto_targets.target_gini_increase[target_pop][affected_pop]
                            print(f"      (target: {target_gini:+.4f})")
        
        print(f"\n  Total optogenetic loss: {opto_loss:.6f}")
        
        print(f"\n{'='*80}")
        print("Combined loss breakdown")
        print(f"{'='*80}")
        print(f"  Baseline: {baseline_loss:.6f} × {self.targets.baseline_weight} = {baseline_loss * self.targets.baseline_weight:.6f}")
        print(f"  Optogenetic: {opto_loss:.6f} × {self.targets.optogenetic_weight} = {opto_loss * self.targets.optogenetic_weight:.6f}")
        print(f"  TOTAL: {recomputed_loss:.6f}")
        
        print(f"\n{'='*80}")
        print("Loss verification")
        print(f"{'='*80}")
        print(f"  Loss from optimizer:  {loss:.6f}")
        print(f"  Recomputed loss:      {recomputed_loss:.6f}")
        if abs(loss - recomputed_loss) > 1e-4:
            print(f"  WARNING: Loss mismatch!")
        print(f"{'='*80}\n")
    
    def save_results(self, filename: str):
        """Save optimization results to JSON"""
        if self.best_params is None:
            print("No results to save")
            return
        
        results = {
            'optimization_info': {
                'timestamp': datetime.now().isoformat(),
                'best_loss': float(self.best_loss),
                'device': str(self.device),
                'baseline_weight': self.targets.baseline_weight,
                'optogenetic_weight': self.targets.optogenetic_weight
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
                'connection_modulation': self.best_params
            },
            'history': self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


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
                           diagnostic_frequency=5):
    """
    Run global optimization with optogenetic objectives using batch GPU interface
    
    Args:
        optimization_config: OptimizationConfig instance
        device: Device to run on (None for auto-detect, 'cpu', or 'cuda')
        n_workers: Number of parallel workers (for CPU)
        n_threads_per_worker: Threads per worker
        method: 'particle_swarm' or 'differential_evolution'
        n_particles: Number of particles for PSO
        max_iterations: Maximum iterations
        diagnostic_frequency: How often to print detailed diagnostics
    """
    
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
    
    # Create optimizer
    optimizer = OptogeneticCircuitOptimizer(
        circuit_params, base_synaptic_params, opsin_params,
        targets, optimization_config, device=device
    )
    
    # Run optimization
    results = optimizer.optimize(
        method=method,
        n_particles=n_particles,
        max_iterations=max_iterations,
        n_workers=n_workers,
        n_threads_per_worker=n_threads_per_worker,
        diagnostic_frequency=diagnostic_frequency
    )
    
    # Save results
    optimizer.save_results('DG_optogenetic_optimization_results.json')
    
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
    
    args = parser.parse_args()
    
    config = create_default_global_opt_config()
    
    results, optimizer = run_global_optimization(
        config,
        device=args.device,
        n_workers=args.n_workers,
        n_threads_per_worker=args.n_threads,
        method=args.method,
        n_particles=args.n_particles,
        max_iterations=args.max_iterations
    )
    
    print(f"\nOptimization Complete!")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Device: {results['device']}")
