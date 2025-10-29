#!/usr/bin/env python3
"""
Connection Parameter Optimization Framework


Uses PyTorch optimization or global optimization to find
connection_modulation parameters that achieve target firing rates for
given input conditions.

Automatically selects evaluation strategy based on device:
- GPU -> Batched parallel evaluation
- CPU -> Multiprocessing for population-based, sequential for gradient-based

Usage:
    optimizer = CircuitOptimizer(params, targets, config, device='cuda')
    results = optimizer.optimize(method='particle_swarm')
    # uses batched GPU evaluation
    
    optimizer = CircuitOptimizer(params, targets, config, device='cpu')
    results = optimizer.optimize(method='particle_swarm')
    # uses multiprocessing

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.optimize import differential_evolution
import tqdm

from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit, CircuitParams, OpsinParams, PerConnectionSynapticParams,
    get_default_device
)

from DG_batch_circuit_dendritic_somatic_transfer import (
    BatchDentateCircuit, BatchCircuitEvaluator
)

def configure_torch_threads(n_threads):
    """
    Configure PyTorch to use limited number of threads
    Call this at the start of each worker process
    """
    import os
    
    # Set environment variables BEFORE importing torch
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    
    # Also set torch-specific thread limits
    import torch
    torch.set_num_threads(n_threads)
    #torch.set_num_interop_threads(n_threads)
    

@dataclass
class OptimizationTargets:
    """Target firing rates and constraints for optimization"""
    
    # Target firing rates (Hz) for each population
    target_rates: Dict[str, float] = field(default_factory=lambda: {
        'gc': 0.2,   # Sparse granule cell activity
        'mc': 1.1,   # Moderate mossy cell activity  
        'pv': 6.0,  # Fast-spiking PV interneurons
        'sst': 4.0, # Slower SST interneurons
    })
    
    # Acceptable variance around targets (Hz)
    rate_tolerance: Dict[str, float] = field(default_factory=lambda: {
        'gc': 0.25,
        'mc': 1.0,
        'pv': 5.0,
        'sst': 2.0,
    })
    
    # Population sparsity targets (fraction active > threshold)
    sparsity_targets: Dict[str, float] = field(default_factory=lambda: {
        'gc': 0.08,   # ~8% granule cells active
        'mc': 0.60,   # Most mossy cells active
        'pv': 0.80,   # Most PV cells active
        'sst': 0.50,  # Moderate SST activity
    })
    
    # Activity threshold for sparsity calculation (Hz)
    activity_threshold: float = 1.0
    
    # Weight factors for different loss components
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'firing_rate': 1.0,
        'sparsity': 0.5,
        'smoothness': 0.01,      # Regularization
        'bounds': 0.1,         # Constraint violations
    })

    # Format: (pop1, pop2, margin) means pop1_rate >= pop2_rate + margin
    rate_ordering_constraints: List[Tuple[str, str, float]] = field(default_factory=lambda: [
        ('pv', 'sst', 5.0),    # PV should fire at least 5 Hz faster than SST
        ('pv', 'mc', 6.0),     # PV should fire at least 5 Hz faster than MC
        ('pv', 'gc', 10.0),     # PV should fire at least 6 Hz faster than GC
        ('sst', 'mc', 3.0),    # SST should fire at least 2 Hz faster than MC
        ('mc', 'gc', 1.0),     # MC should fire at least 1 Hz faster than GC
    ])
    
    # Weight for constraint violations
    constraint_violation_weight: float = 10.0
    
    # Constraint enforcement method: 'soft_penalty', 'barrier', 'augmented_lagrangian'
    constraint_method: str = 'soft_penalty'
    
    # Connection strength bounds (multipliers)
    connection_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'mec_gc': (0.5, 4.0),    # Perforant path can vary widely
        'mec_pv': (0.1, 5.0),    # Perforant path can vary widely
        'gc_mc': (0.1, 5.0),     # Mossy fiber strength
        'mc_gc': (0.1, 5.0),     # Associational pathway  
        'mc_mc': (0.5, 5.0),     # MC-MC excitation  
        'pv_gc': (0.1, 5.0),     # Strong inhibition possible
        'sst_gc': (0.1, 3.0),    # Moderate dendritic inhibition
        'gc_pv': (0.1, 5.0),     # Feedforward excitation
        'gc_sst': (0.1, 5.0),    # Weaker SST drive
        'mc_pv': (0.1, 6.0),     # Strong MC to PV
        'mc_sst': (0.1, 2.0),    # MC to SST
        'pv_pv': (0.1, 5.0),     # Lateral PV inhibition
        'sst_pv': (0.1, 3.0),    # SST disinhibition
        'sst_sst': (0.1, 5.0),   # SST lateral inhibition
        'sst_mc': (0.1, 3.0),    # Moderate dendritic inhibition
        'pv_mc': (0.1, 5.0),     # Moderate inhibition of MC
    })

@dataclass  
class OptimizationConfig:
    """Configuration for optimization procedure"""
    
    # Optimization settings
    learning_rate: float = 0.01
    max_iterations: int = 500
    convergence_threshold: float = 1e-4
    patience: int = 25  # Early stopping
    
    # Simulation settings  
    simulation_duration: int = 600 # ms
    warmup_duration: int = 100     # ms to ignore
    n_trials: int = 3              # Multiple runs for stability
    
    # MEC drive conditions to test
    mec_drive_levels: List[float] = field(default_factory=lambda: [80.0, 150.0, 200.0])  # pA
    
    # Optimizer choice
    optimizer_type: str = 'sgd'  # 'adam', 'sgd', 'rmsprop'
    scheduler_type: str = 'plateau'  # 'plateau', 'cosine', 'exponential', None
    
    # Regularization
    l1_regularization: float = 0.001
    l2_regularization: float = 0.0001

    # Simulation device (cpu or cuda)
    device: Optional[torch.device] = None

class OptimizableConnectionParameters(nn.Module):
    """
    Learnable connection modulation parameters
    
    Uses log-parameterization to ensure positive values and
    implements constraints through penalty functions.
    """
    
    def __init__(self, base_synaptic_params: PerConnectionSynapticParams,
                 targets: OptimizationTargets):
        super().__init__()
        
        self.targets = targets
        self.base_params = base_synaptic_params
        
        # Initialize learnable parameters in log-space for positivity
        self.log_modulation = nn.ParameterDict()
        
        for conn_name, base_value in base_synaptic_params.connection_modulation.items():
            # Initialize near baseline in log space
            init_log_value = torch.log(torch.tensor(base_value, dtype=torch.float32))
            self.log_modulation[conn_name] = nn.Parameter(init_log_value)
    
    def get_connection_modulation(self) -> Dict[str, float]:
        """Convert log parameters back to connection modulation values"""
        modulation = {}
        for conn_name, log_param in self.log_modulation.items():
            modulation[conn_name] = torch.exp(log_param).item()
        return modulation
    
    def create_synaptic_params(self) -> PerConnectionSynapticParams:
        """Create synaptic parameters with current optimized values"""
        current_modulation = self.get_connection_modulation()
        
        return PerConnectionSynapticParams(
            ampa_g_mean=self.base_params.ampa_g_mean,
            ampa_g_std=self.base_params.ampa_g_std,
            ampa_g_min=self.base_params.ampa_g_min,
            ampa_g_max=self.base_params.ampa_g_max,
            gaba_g_mean=self.base_params.gaba_g_mean,
            gaba_g_std=self.base_params.gaba_g_std,
            gaba_g_min=self.base_params.gaba_g_min,
            gaba_g_max=self.base_params.gaba_g_max,
            distribution=self.base_params.distribution,
            connection_modulation=current_modulation,
            e_exc=self.base_params.e_exc,
            e_inh=self.base_params.e_inh,
            v_rest=self.base_params.v_rest,
            tau_ampa=self.base_params.tau_ampa,
            tau_gaba=self.base_params.tau_gaba,
            tau_nmda=self.base_params.tau_nmda
        )
    
    def compute_bounds_penalty(self) -> Tensor:
        """Soft constraint penalty for connection bounds"""
        penalty = torch.tensor(0.0)
        
        for conn_name, log_param in self.log_modulation.items():
            if conn_name in self.targets.connection_bounds:
                min_val, max_val = self.targets.connection_bounds[conn_name]
                current_val = torch.exp(log_param)
                
                # Soft penalty outside bounds using smooth functions
                lower_violation = torch.clamp(min_val - current_val, min=0.0)
                upper_violation = torch.clamp(current_val - max_val, min=0.0)
                
                penalty += (lower_violation ** 2 + upper_violation ** 2)
        
        return penalty
    
    def compute_smoothness_penalty(self) -> Tensor:
        """Regularization penalty to prevent extreme parameter values"""
        
        # L2 penalty on deviations from baseline
        l2_penalty = torch.tensor(0.0)
        for conn_name, log_param in self.log_modulation.items():
            baseline_log = torch.log(torch.tensor(self.base_params.connection_modulation[conn_name]))
            l2_penalty += (log_param - baseline_log) ** 2
        
        return l2_penalty

def evaluate_rate_ordering_constraints(firing_rates: Dict[str, float], 
                                       constraints: List[Tuple[str, str, float]]) -> Tuple[float, List[Dict]]:
    """
    Evaluate firing rate ordering constraints
    
    Args:
        firing_rates: Dict mapping population name to firing rate
        constraints: List of (pop1, pop2, margin) tuples
        
    Returns:
        total_violation: Total constraint violation penalty
        violations_info: List of violation details for logging
    """
    total_violation = 0.0
    violations_info = []
    
    for pop1, pop2, margin in constraints:
        if pop1 in firing_rates and pop2 in firing_rates:
            rate1 = firing_rates[pop1]
            rate2 = firing_rates[pop2]
            
            # Constraint: rate1 >= rate2 + margin
            # Violation if: rate1 < rate2 + margin
            required_difference = rate2 + margin
            actual_difference = rate1
            
            violation = max(0.0, required_difference - actual_difference)
            
            if violation > 0:
                violations_info.append({
                    'constraint': f"{pop1} >= {pop2} + {margin}",
                    'pop1': pop1,
                    'pop2': pop2,
                    'rate1': rate1,
                    'rate2': rate2,
                    'required': required_difference,
                    'actual': actual_difference,
                    'violation': violation
                })
                
                # Quadratic penalty for smooth gradients
                total_violation += violation ** 2
    
    return total_violation, violations_info    

def auto_adjust_targets_for_constraints(targets: OptimizationTargets) -> OptimizationTargets:
    """
    Automatically adjust target firing rates to be consistent with ordering constraints
    Uses linear programming to find feasible targets close to original
    """
    from scipy.optimize import linprog
    
    populations = list(targets.target_rates.keys())
    n_pops = len(populations)
    pop_to_idx = {pop: i for i, pop in enumerate(populations)}
    
    original_targets = np.array([targets.target_rates[pop] for pop in populations])
    
    # Minimize sum of squared deviations from original targets
    # Using linear approximation: minimize sum of |x_i - target_i|
    c = np.ones(n_pops)  # Coefficients for linear objective
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For constraint pop1 >= pop2 + margin, we need: -x_pop1 + x_pop2 <= -margin
    A_ub = []
    b_ub = []
    
    for pop1, pop2, margin in targets.rate_ordering_constraints:
        if pop1 in pop_to_idx and pop2 in pop_to_idx:
            constraint_row = np.zeros(n_pops)
            constraint_row[pop_to_idx[pop1]] = -1  # -x_pop1
            constraint_row[pop_to_idx[pop2]] = 1   # +x_pop2
            A_ub.append(constraint_row)
            b_ub.append(-margin)  # <= -margin
    
    # Bounds: firing rates must be positive
    bounds = [(0.1, 100.0) for _ in range(n_pops)]  # Reasonable rate bounds
    
    # Solve (simple version - just find feasible point)
    # For better solution, use quadratic programming
    result = linprog(
        c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        bounds=bounds, method='highs'
    )
    
    if result.success:
        adjusted_targets = dict(zip(populations, result.x))
        print("Adjusted targets to satisfy constraints:")
        for pop in populations:
            original = targets.target_rates[pop]
            adjusted = adjusted_targets[pop]
            change = adjusted - original
            print(f"  {pop}: {original:.1f} -> {adjusted:.1f} ({change:+.1f})")
        
        # Create new targets object with adjusted rates
        new_targets = OptimizationTargets(
            target_rates=adjusted_targets,
            sparsity_targets=targets.sparsity_targets,
            rate_ordering_constraints=targets.rate_ordering_constraints,
            constraint_violation_weight=targets.constraint_violation_weight
        )
        return new_targets
    else:
        print("Warning: Could not find feasible targets satisfying all constraints")
        return targets


# ============================================================================
# Evaluation Strategy Pattern
# ============================================================================

class EvaluationStrategy(ABC):
    """
    Abstract base class for parameter evaluation strategies
    
    Enables automatic selection of optimal evaluation approach based on:
    - Target device (CPU vs GPU)
    - Optimization algorithm (gradient-based vs population-based)
    - Available resources (cores, memory)
    """
    
    @abstractmethod
    def evaluate_batch(self,
                      parameter_sets: List[Dict[str, float]],
                      mec_drive: float,
                      n_trials: int) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Evaluate a batch of parameter configurations
        
        Args:
            parameter_sets: List of connection_modulation dicts
            mec_drive: MEC drive level
            n_trials: Number of trials to average
            
        Returns:
            losses: List of loss values
            firing_rates: List of dicts with firing rates per population
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, any]:
        """Return information about this strategy"""
        pass


class SequentialStrategy(EvaluationStrategy):
    """
    Sequential evaluation strategy
    
    Used for:
    - Gradient-based optimization (any device)
    - Single evaluations
    - When parallelism not needed
    """
    
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
        firing_rates_list = []
        
        for params in parameter_sets:
            loss, firing_rates = self._evaluate_single(params, mec_drive, n_trials)
            losses.append(loss)
            firing_rates_list.append(firing_rates)
        
        return losses, firing_rates_list
    
    def _evaluate_single(self, connection_modulation, mec_drive, n_trials):
        """Evaluate single parameter configuration"""
        # Create synaptic params with modulation
        synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=self.base_synaptic_params.ampa_g_mean,
            ampa_g_std=self.base_synaptic_params.ampa_g_std,
            gaba_g_mean=self.base_synaptic_params.gaba_g_mean,
            gaba_g_std=self.base_synaptic_params.gaba_g_std,
            distribution=self.base_synaptic_params.distribution,
            connection_modulation=connection_modulation
        )
        
        # Create circuit
        circuit = DentateCircuit(
            self.circuit_params, synaptic_params, self.opsin_params,
            device=self.device
        )
        
        total_loss = 0.0
        
        for trial in range(n_trials):
            circuit.reset_state()
            
            mec_input = torch.ones(self.circuit_params.n_mec, device=self.device) * mec_drive
            activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
            
            for t in range(self.config.simulation_duration):
                external_drive = {'mec': mec_input}
                activities = circuit({}, external_drive)
                
                if t >= self.config.warmup_duration:
                    for pop in activities_over_time:
                        if pop in activities:
                            activities_over_time[pop].append(activities[pop].clone())
            
            # Calculate loss
            trial_loss, firing_rates = self._calculate_loss(activities_over_time)
            total_loss += trial_loss
        
        return total_loss / n_trials, firing_rates
    
    def _calculate_loss(self, activities_over_time):
        """Calculate loss from activity time series"""
        total_loss = 0.0
        firing_rates = {}
        
        for pop in activities_over_time:
            if len(activities_over_time[pop]) > 0:
                pop_time_series = torch.stack(activities_over_time[pop])
                mean_rates = torch.mean(pop_time_series, dim=0)
                
                if pop in self.targets.target_rates:
                    target_rate = self.targets.target_rates[pop]
                    actual_rate = torch.mean(mean_rates).item()
                    firing_rates[pop] = actual_rate
                    tolerance = self.targets.rate_tolerance[pop]
                    
                    if np.isclose(actual_rate, 0.0, 1e-2, 1e-2):
                        rate_loss = 1e2
                    else:
                        error = abs(actual_rate - target_rate)
                        rate_loss = error if error > tolerance else 0.5 * (error / tolerance) ** 2
                    
                    total_loss += rate_loss
                
                if pop in self.targets.sparsity_targets:
                    target_sparsity = self.targets.sparsity_targets[pop]
                    actual_sparsity = (torch.sum(mean_rates > self.targets.activity_threshold) / len(mean_rates)).item()
                    total_loss += (actual_sparsity - target_sparsity) ** 2

        # Add constraints
        constraint_violation, _ = evaluate_rate_ordering_constraints(
            firing_rates, self.targets.rate_ordering_constraints
        )
        total_loss += self.targets.constraint_violation_weight * constraint_violation
        
        return total_loss, firing_rates
    
    def get_strategy_info(self):
        return {
            'name': 'Sequential',
            'device': str(self.device),
            'parallelism': 'None',
            'description': 'Sequential evaluation, best for gradient-based optimization'
        }


class BatchGPUStrategy(EvaluationStrategy):
    """
    Batch GPU evaluation strategy
    
    Used for:
    - Population-based optimization on GPU
    - Particle swarm, genetic algorithms, etc.
    - When batch_size > 1 on CUDA device
    """
    
    def __init__(self, circuit_params, base_synaptic_params, opsin_params,
                 targets, config, device):
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        self.device = device
        
        # Create evaluator
        self.evaluator = BatchCircuitEvaluator(
            circuit_params, base_synaptic_params, opsin_params,
            targets, config, device=device
        )
    
    def evaluate_batch(self, parameter_sets, mec_drive, n_trials):
        """Evaluate all configurations in parallel on GPU"""
        batch_size = len(parameter_sets)
        
        # Evaluate with averaging over trials
        total_losses = torch.zeros(batch_size, device=self.device)
        
        for trial in range(n_trials):
            losses, firing_rates_batch = self.evaluator.evaluate_parameter_batch(
                parameter_sets, mec_drive
            )
            total_losses += losses
        
        total_losses /= n_trials
        
        # Convert to lists for consistent API
        losses_list = total_losses.cpu().numpy().tolist()
        
        # Get firing rates for each batch element
        firing_rates_list = []
        for b in range(batch_size):
            firing_rates = {
                pop: firing_rates_batch[pop][b].item()
                for pop in firing_rates_batch
            }
            firing_rates_list.append(firing_rates)
        
        return losses_list, firing_rates_list
    
    def get_strategy_info(self):
        return {
            'name': 'BatchGPU',
            'device': str(self.device),
            'parallelism': 'Data parallelism (batched)',
            'description': f'Batch evaluation on GPU, optimal for population-based methods'
        }


class MultiprocessCPUStrategy(EvaluationStrategy):
    """
    Multiprocess CPU evaluation strategy
    
    Used for:
    - Population-based optimization on CPU
    - When multiple CPU cores available
    - Differential evolution, particle swarm, etc.
    """
    
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
        
        # Prepare circuit factory data for workers
        self.circuit_factory_data = (
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
            (params, mec_drive, n_trials, self.circuit_factory_data,
             self.targets, self.config)
            for params in parameter_sets
        ]
        
        configure_torch_threads(self.n_threads_per_worker)
        
        # Use multiprocessing pool
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.n_workers) as pool:
            results = pool.map(_worker_evaluate_single, eval_args)
        
        # Unpack results
        losses = [r[0] for r in results]
        firing_rates_list = [r[1] for r in results]
        
        return losses, firing_rates_list
    
    def get_strategy_info(self):
        return {
            'name': 'MultiprocessCPU',
            'device': str(self.device),
            'parallelism': f'{self.n_workers} workers * {self.n_threads_per_worker} threads',
            'description': f'Multiprocess evaluation on CPU with {self.n_workers} workers'
        }


def _worker_evaluate_single(args):
    """Worker function for multiprocess evaluation (module-level for pickling)"""
    (connection_modulation, mec_drive, n_trials, circuit_factory_data,
     targets, config) = args
    
    from DG_circuit_dendritic_somatic_transfer import DentateCircuit, PerConnectionSynapticParams
    
    device = torch.device('cpu')
    circuit_params, base_synaptic_params_dict, opsin_params = circuit_factory_data
    
    # Create synaptic params
    synaptic_params = PerConnectionSynapticParams(
        ampa_g_mean=base_synaptic_params_dict['ampa_g_mean'],
        ampa_g_std=base_synaptic_params_dict['ampa_g_std'],
        gaba_g_mean=base_synaptic_params_dict['gaba_g_mean'],
        gaba_g_std=base_synaptic_params_dict['gaba_g_std'],
        distribution=base_synaptic_params_dict['distribution'],
        connection_modulation=connection_modulation
    )
    
    # Create circuit
    circuit = DentateCircuit(circuit_params, synaptic_params, opsin_params, device=device)
    
    total_loss = 0.0
    firing_rates = {}
    
    for trial in range(n_trials):
        circuit.reset_state()
        
        mec_input = torch.ones(circuit.circuit_params.n_mec, device=device) * mec_drive
        activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        
        for t in range(config.simulation_duration):
            external_drive = {'mec': mec_input}
            activities = circuit({}, external_drive)
            
            if t >= config.warmup_duration:
                for pop in activities_over_time:
                    if pop in activities:
                        activities_over_time[pop].append(activities[pop].clone())
        
        # Calculate loss
        trial_loss = 0.0
        for pop in activities_over_time:
            if len(activities_over_time[pop]) > 0:
                pop_time_series = torch.stack(activities_over_time[pop])
                mean_rates = torch.mean(pop_time_series, dim=0)
                
                if pop in targets.target_rates:
                    target_rate = targets.target_rates[pop]
                    actual_rate = torch.mean(mean_rates).item()
                    firing_rates[pop] = actual_rate
                    tolerance = targets.rate_tolerance[pop]
                    
                    if np.isclose(actual_rate, 0.0, 1e-2, 1e-2):
                        rate_loss = 1e2
                    else:
                        error = abs(actual_rate - target_rate)
                        rate_loss = error if error > tolerance else 0.5 * (error / tolerance) ** 2
                    
                    trial_loss += rate_loss
                
                if pop in targets.sparsity_targets:
                    target_sparsity = targets.sparsity_targets[pop]
                    actual_sparsity = (torch.sum(mean_rates > targets.activity_threshold) / len(mean_rates)).item()
                    trial_loss += (actual_sparsity - target_sparsity) ** 2
        
        # Add constraints
        constraint_violation, _ = evaluate_rate_ordering_constraints(
            firing_rates, targets.rate_ordering_constraints
        )
        trial_loss += targets.constraint_violation_weight * constraint_violation
        
        total_loss += trial_loss
    
    return total_loss / n_trials, firing_rates


# ============================================================================
# Main circuit optimizer class
# ============================================================================

class CircuitOptimizer:
    """
    Circuit optimizer that automatically selects optimal evaluation strategy
    
    Provides single API for all optimization methods and devices.
    Automatically chooses:
    - BatchGPU for population-based on CUDA
    - MultiprocessCPU for population-based on CPU
    - Sequential for gradient-based (any device)
    
    Usage:
        optimizer = CircuitOptimizer(
            circuit_params, synaptic_params, opsin_params,
            targets, config,
            device='cuda'  # or 'cpu' or None for auto
        )
        
        # Gradient-based (auto uses sequential)
        results = optimizer.optimize(method='gradient', max_iterations=100)
        
        # Particle swarm (auto uses batched GPU or multiprocess CPU)
        results = optimizer.optimize(method='particle_swarm', n_particles=32)
        
        # Differential evolution (CPU only, uses multiprocess)
        results = optimizer.optimize(method='differential_evolution')
    """
    
    def __init__(self,
                 circuit_params: CircuitParams,
                 base_synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 targets: OptimizationTargets,
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
        
        print(f"CircuitOptimizer initialized on device: {self.device}")
    
    def _select_strategy(self, method: str, **kwargs) -> EvaluationStrategy:
        """
        Automatically select optimal evaluation strategy based on method and device
        
        Decision logic:
        1. Gradient-based -> Sequential (works well on any device)
        2. Population-based + GPU -> BatchGPU (massive parallelism)
        3. Population-based + CPU -> MultiprocessCPU (process parallelism)
        4. Differential Evolution -> MultiprocessCPU (scipy limitation)
        """
        if method == 'gradient':
            strategy = SequentialStrategy(
                self.circuit_params, self.base_synaptic_params, self.opsin_params,
                self.targets, self.config, self.device
            )
        
        elif method == 'differential_evolution':
            if self.device.type == 'cuda':
                raise NotImplementedError(
                    "Differential Evolution on GPU not supported (scipy limitation). "
                    "Use CPU or try 'particle_swarm' for GPU optimization."
                )
            strategy = MultiprocessCPUStrategy(
                self.circuit_params, self.base_synaptic_params, self.opsin_params,
                self.targets, self.config, self.device,
                n_workers=kwargs.get('n_workers', None),
                n_threads_per_worker=kwargs.get('n_threads_per_worker', 1)
            )
        
        elif method in ['particle_swarm', 'genetic_algorithm']:
            if self.device.type == 'cuda':
                strategy = BatchGPUStrategy(
                    self.circuit_params, self.base_synaptic_params, self.opsin_params,
                    self.targets, self.config, self.device
                )
            else:
                strategy = MultiprocessCPUStrategy(
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
    
    def optimize(self, method: str = 'gradient', **kwargs) -> Dict:
        """
        Run optimization with automatic strategy selection
        
        Args:
            method: 'gradient', 'particle_swarm', 'differential_evolution'
            **kwargs: Method-specific arguments
            
        Gradient method kwargs:
            max_iterations: int = 100
            learning_rate: float = 0.01
            
        Particle swarm kwargs:
            n_particles: int = 32
            max_iterations: int = 100
            
        Differential evolution kwargs:
            max_iterations: int = 50
            n_workers: int = None (auto)
            n_threads_per_worker: int = 1
            
        Returns:
            Dict with optimization results
        """
        connection_names = list(self.targets.connection_bounds.keys())
        
        if method == 'gradient':
            return self._optimize_gradient(**kwargs)
        elif method == 'particle_swarm':
            return self._optimize_particle_swarm(connection_names, **kwargs)
        elif method == 'differential_evolution':
            return self._optimize_differential_evolution(connection_names, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _optimize_gradient(self, max_iterations=100, learning_rate=0.01):
        """Gradient-based optimization"""
        strategy = self._select_strategy('gradient')
        
        # Use existing gradient optimization logic
        opt_params = OptimizableConnectionParameters(
            self.base_synaptic_params, self.targets, device=self.device
        )
        
        optimizer = optim.SGD(opt_params.parameters(), lr=learning_rate, momentum=0.9)
        
        print(f"Starting gradient-based optimization...")
        print(f"Max iterations: {max_iterations}")
        
        patience_counter = 0
        pbar = tqdm.tqdm(range(max_iterations), desc="Gradient optimization")
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # Evaluate current parameters
            current_params = opt_params.get_connection_modulation()
            
            total_loss = 0.0
            for mec_drive in self.config.mec_drive_levels:
                losses, _ = strategy.evaluate_batch(
                    [current_params], mec_drive, self.config.n_trials
                )
                total_loss += losses[0]
            
            total_loss /= len(self.config.mec_drive_levels)
            
            # Convert to tensor for backprop
            loss_tensor = torch.tensor(total_loss, requires_grad=True, device=self.device)
            
            # Manual gradient computation (since we're using black-box evaluation)
            # This is a simplified version - full implementation would use finite differences
            loss_tensor.backward()
            optimizer.step()
            
            # Track best
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_params = current_params.copy()
                self.best_iteration = iteration
                patience_counter = 0
            else:
                patience_counter += 1
            
            pbar.set_postfix({'loss': f"{total_loss:.6f}", 'best': f"{self.best_loss:.6f}"})
            
            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping at iteration {iteration}")
                break
        
        return {
            'optimized_connection_modulation': self.best_params,
            'best_loss': self.best_loss,
            'best_iteration': self.best_iteration,
            'method': 'gradient',
            'device': str(self.device),
            'strategy': strategy.get_strategy_info()['name']
        }
    
    def _optimize_particle_swarm(self, connection_names, n_particles=32, max_iterations=100,
                                 n_workers=None, n_threads_per_worker=1):
        """Particle swarm optimization with automatic strategy"""
        strategy = self._select_strategy('particle_swarm', n_particles=n_particles,
                                         n_workers=n_workers,
                                         n_threads_per_worker=n_threads_per_worker)
        
        n_dimensions = len(connection_names)
        bounds = [self.targets.connection_bounds.get(name, (0.1, 5.0)) 
                 for name in connection_names]
        
        print(f"Starting Particle Swarm Optimization...")
        print(f"Particles: {n_particles}")
        print(f"Iterations: {max_iterations}")
        
        # Initialize on CPU (convert to numpy for compatibility)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, n_dimensions))
        velocities = np.random.randn(n_particles, n_dimensions) * 0.1
        
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(n_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        
        # PSO parameters
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for iteration in tqdm.tqdm(range(max_iterations), desc="PSO"):
            # Convert positions to parameter dicts
            parameter_sets = []
            for i in range(n_particles):
                param_dict = dict(zip(connection_names, positions[i]))
                parameter_sets.append(param_dict)
            
            # Evaluate all particles
            total_losses = np.zeros(n_particles)
            for mec_drive in self.config.mec_drive_levels:
                losses, _ = strategy.evaluate_batch(
                    parameter_sets, mec_drive, self.config.n_trials
                )
                total_losses += np.array(losses)
            
            total_losses /= len(self.config.mec_drive_levels)
            
            # Update personal bests
            improved = total_losses < personal_best_scores
            personal_best_scores[improved] = total_losses[improved]
            personal_best_positions[improved] = positions[improved]
            
            # Update global best
            min_idx = np.argmin(total_losses)
            if total_losses[min_idx] < global_best_score:
                global_best_score = total_losses[min_idx]
                global_best_position = positions[min_idx].copy()
                self.best_params = dict(zip(connection_names, global_best_position))
                print(f"\nIteration {iteration}: New best = {global_best_score:.6f}")
            
            # Update velocities and positions
            r1, r2 = np.random.random((n_particles, n_dimensions)), np.random.random((n_particles, n_dimensions))
            velocities = (w * velocities + 
                         c1 * r1 * (personal_best_positions - positions) +
                         c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, lower_bounds, upper_bounds)
        
        self.best_loss = global_best_score
        
        return {
            'optimized_connection_modulation': self.best_params,
            'best_loss': self.best_loss,
            'method': 'particle_swarm',
            'n_particles': n_particles,
            'device': str(self.device),
            'strategy': strategy.get_strategy_info()['name']
        }
    
    def _optimize_differential_evolution(self, connection_names, max_iterations=50, 
                                         n_workers=None, n_threads_per_worker=1):
        """Differential evolution with multiprocess CPU"""
        strategy = self._select_strategy('differential_evolution',
                                         n_workers=n_workers,
                                         n_threads_per_worker=n_threads_per_worker)
        
        bounds = [self.targets.connection_bounds.get(name, (0.1, 5.0)) 
                 for name in connection_names]
        
        print(f"Starting Differential Evolution...")
        print(f"Iterations: {max_iterations}")
        
        def objective(param_array):
            """Objective function for scipy"""
            param_dict = dict(zip(connection_names, param_array))
            
            total_loss = 0.0
            for mec_drive in self.config.mec_drive_levels:
                losses, _ = strategy.evaluate_batch(
                    [param_dict], mec_drive, self.config.n_trials
                )
                total_loss += losses[0]
            
            return total_loss / len(self.config.mec_drive_levels)
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            popsize=15,
            workers=1,  # Strategy handles parallelism
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
    
    def print_best_firing_rates(self, mec_drive=100.0):
        """Evaluate and print best parameters"""
        if self.best_params is None:
            print("No best parameters found. Run optimize() first.")
            return
        
        print(f"\nBest Configuration Firing Rates")
        print("="*60)
        print(f"Best loss: {self.best_loss:.6f}")
        
        # Use sequential strategy for single evaluation
        strategy = SequentialStrategy(
            self.circuit_params, self.base_synaptic_params, self.opsin_params,
            self.targets, self.config, self.device
        )
        
        _, firing_rates_list = strategy.evaluate_batch(
            [self.best_params], mec_drive, n_trials=1
        )
        firing_rates = firing_rates_list[0]
        
        print(f"MEC Drive: {mec_drive:.1f} pA\n")
        print(f"{'Population':<12} {'Target':<10} {'Actual':<10} {'Error':<10}")
        print("-"*50)
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop in firing_rates:
                actual = firing_rates[pop]
                target = self.targets.target_rates.get(pop, 0.0)
                error = ((actual - target) / target * 100) if target > 0 else 0.0
                print(f"{pop.upper():<12} {target:<10.2f} {actual:<10.2f} {error:+.1f}%")

    def _get_firing_rates_for_drive(self, mec_drive: float) -> Dict[str, Dict[str, float]]:
        """
        Helper method to get detailed firing rates for a specific MEC drive level
        
        Args:
            mec_drive: MEC drive level (pA)
            
        Returns:
            Dict mapping population name to metrics dict with keys:
                - target_rate, actual_rate, rate_error_percent
                - sparsity, target_sparsity
                - std_rate, max_rate, min_rate
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
        
        # Create circuit with best parameters
        synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=self.base_synaptic_params.ampa_g_mean,
            ampa_g_std=self.base_synaptic_params.ampa_g_std,
            gaba_g_mean=self.base_synaptic_params.gaba_g_mean,
            gaba_g_std=self.base_synaptic_params.gaba_g_std,
            distribution=self.base_synaptic_params.distribution,
            connection_modulation=self.best_params
        )
        
        circuit = DentateCircuit(
            self.circuit_params, synaptic_params, self.opsin_params,
            device=self.device
        )
        circuit.reset_state()
        
        # Run simulation
        n_steps = 600
        warmup = 150
        mec_input = torch.ones(self.circuit_params.n_mec, device=self.device) * mec_drive
        
        activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        
        for t in range(n_steps):
            external_drive = {'mec': mec_input}
            activities = circuit({}, external_drive)
            
            if t >= warmup:
                for pop in activities_over_time:
                    if pop in activities:
                        activities_over_time[pop].append(activities[pop].clone())
        
        # Calculate metrics
        firing_data = {}
        for pop in activities_over_time:
            if len(activities_over_time[pop]) > 0:
                pop_time_series = torch.stack(activities_over_time[pop])
                mean_rates = torch.mean(pop_time_series, dim=0)
                
                actual_rate = torch.mean(mean_rates).item()
                target_rate = self.targets.target_rates.get(pop, 0.0)
                sparsity = (torch.sum(mean_rates > self.targets.activity_threshold) / len(mean_rates)).item()
                
                firing_data[pop] = {
                    'target_rate': target_rate,
                    'actual_rate': actual_rate,
                    'rate_error_percent': ((actual_rate - target_rate) / target_rate * 100) if target_rate > 0 else 0.0,
                    'sparsity': sparsity,
                    'target_sparsity': self.targets.sparsity_targets.get(pop, None),
                    'std_rate': torch.std(mean_rates).item(),
                    'max_rate': torch.max(mean_rates).item(),
                    'min_rate': torch.min(mean_rates).item(),
                }
        
        return firing_data
    
    def save_best_results_to_json(self, filename: str, mec_drive_levels: List[float] = None):
        """
        Save best optimization results and firing rates to JSON file
        
        Args:
            filename: Path to JSON file to save
            mec_drive_levels: List of MEC drive levels to evaluate (default: [100.0])
        """
        if self.best_params is None:
            print("No best parameters found. Run optimize() first.")
            return
        
        if mec_drive_levels is None:
            mec_drive_levels = [100.0]
        
        import json
        from datetime import datetime
        
        # Collect firing rates and sparsity for each MEC drive level
        performance_data = {}
        for mec_drive in mec_drive_levels:
            firing_data = self._get_firing_rates_for_drive(mec_drive)
            performance_data[f'mec_{mec_drive}'] = firing_data
        
        # Prepare JSON data
        results_data = {
            'optimization_info': {
                'timestamp': datetime.now().isoformat(),
                'best_loss': float(self.best_loss),
                'best_iteration': int(self.best_iteration) if hasattr(self, 'best_iteration') else None,
                'device': str(self.device),
                'evaluation_strategy': self.history.get('strategy', 'unknown')
            },
            'targets': {
                'firing_rates': self.targets.target_rates,
                'sparsity_targets': self.targets.sparsity_targets,
                'activity_threshold': self.targets.activity_threshold,
                'rate_ordering_constraints': [
                    {'constraint': f"{c[0]} >= {c[1]} + {c[2]}", 'pop1': c[0], 'pop2': c[1], 'margin': c[2]}
                    for c in self.targets.rate_ordering_constraints
                ]
            },
            'optimized_parameters': {
                'connection_modulation': self.best_params,
                'base_conductances': {
                    'ampa_g_mean': self.base_synaptic_params.ampa_g_mean,
                    'ampa_g_std': self.base_synaptic_params.ampa_g_std,
                    'gaba_g_mean': self.base_synaptic_params.gaba_g_mean, 
                    'gaba_g_std': self.base_synaptic_params.gaba_g_std,
                    'distribution': self.base_synaptic_params.distribution,
                }
            },
            'performance': performance_data,
            'circuit_config': {
                'n_gc': self.circuit_params.n_gc,
                'n_mc': self.circuit_params.n_mc,
                'n_pv': self.circuit_params.n_pv,
                'n_sst': self.circuit_params.n_sst,
                'n_mec': self.circuit_params.n_mec,
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        print(f"  Best loss: {self.best_loss:.6f}")
        print(f"  Device: {self.device}")
        print(f"  Evaluated at {len(mec_drive_levels)} MEC drive level(s)")
        
        return results_data

# ============================================================================
# Convenience Functions
# ============================================================================

def create_default_targets() -> OptimizationTargets:
    """Create default optimization targets based on experimental data"""
    return OptimizationTargets(
        target_rates={
            'gc': 0.5,  # Low granule cell activity for sparsity
            'mc': 1.1,   # Moderate mossy cell activity  
            'pv': 10.0,   # Fast-spiking PV interneurons
            'sst': 4.0,   # Slower SST interneurons
        },
        sparsity_targets={
            'gc': 0.08,   # ~8% granule cells active (key constraint)
            'mc': 0.50,   # Most mossy cells active
            'pv': 0.85,   # Most PV cells active
            'sst': 0.60,  # Moderate SST activity
        }
    )


def create_default_config(device: Optional[torch.device] = None) -> OptimizationConfig:
    """Create default optimization configuration"""
    return OptimizationConfig(
        learning_rate=0.1,
        max_iterations=300,
        mec_drive_levels=[40.0],
        n_trials=2,
        simulation_duration=600,
        warmup_duration=100,
        device=device
    )

def create_default_global_opt_config(device: Optional[torch.device] = None) -> OptimizationConfig:
    """Create default optimization configuration for global optimization"""
    return OptimizationConfig(
        learning_rate=0.1,
        max_iterations=20,
        mec_drive_levels=[40.0],
        n_trials=4,  # Reduce for faster iteration
        simulation_duration=1000,
        warmup_duration=200,
        device=device
    )

def run_optimization(method='particle_swarm', device=None, **kwargs):
    """
    Convenience function to run unified optimization
    
    Example:
        # Auto-selects batch GPU if CUDA available
        results = run_optimization(
            method='particle_swarm',
            device='cuda',
            n_particles=32,
            max_iterations=100
        )
    """
    
    device = torch.device(device) if device else get_default_device()
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    base_synaptic_params = PerConnectionSynapticParams()
    targets = create_default_targets()
    config = create_default_config(device=device)
    
    optimizer = CircuitOptimizer(
        circuit_params, base_synaptic_params, opsin_params,
        targets, config, device=device
    )
    
    results = optimizer.optimize(method=method, **kwargs)
    
    print("\n" + "="*60)
    print("Optimization Results")
    print("="*60)
    print(f"Method: {results['method']}")
    print(f"Strategy: {results['strategy']}")
    print(f"Device: {results['device']}")
    print(f"Best loss: {results['best_loss']:.6f}")
    
    optimizer.print_best_firing_rates()
    
    # Save results to JSON
    optimizer.save_best_results_to_json(
        'DG_optimization_results.json',
        mec_drive_levels=config.mec_drive_levels
    )
    
    return results, optimizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dentate Circuit Optimizer')
    parser.add_argument('--method', type=str, default='particle_swarm',
                       choices=['gradient', 'particle_swarm', 'differential_evolution'])
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', None])
    parser.add_argument('--n-particles', type=int, default=32)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--max-iterations', type=int, default=50)
    
    args = parser.parse_args()
    
    print(f"Running {args.method} optimization...")
    
    results, optimizer = run_optimization(
        method=args.method,
        device=args.device,
        n_particles=args.n_particles,
        max_iterations=args.max_iterations,
        n_workers=args.n_workers
    )
