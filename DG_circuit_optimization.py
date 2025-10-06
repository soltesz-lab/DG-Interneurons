#!/usr/bin/env python3
"""
Connection Parameter Optimization Framework

Uses PyTorch optimization to find connection_modulation parameters that
achieve target firing rates for given input conditions.
"""

from typing import Dict, List, Tuple, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import differential_evolution, basinhopping
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import tqdm

# Import your existing classes
from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit, CircuitParams, OpsinParams, PerConnectionSynapticParams
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
    torch.set_num_interop_threads(n_threads)
    

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
        'mec_gc': (0.5, 5.0),    # Perforant path can vary widely
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
        'sst_mc': (0.1, 2.0),    # Moderate dendritic inhibition
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

class CircuitOptimizer:
    """
    Main optimization framework for tuning connection parameters
    """
    
    def __init__(self, circuit_params: CircuitParams, 
                 base_synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 targets: OptimizationTargets,
                 config: OptimizationConfig):
        
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        
        # Initialize optimizable parameters
        self.opt_params = OptimizableConnectionParameters(base_synaptic_params, targets)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Storage for optimization history
        self.history = {
            'loss': [],
            'loss_components': [],
            'firing_rates': [],
            'parameters': [],
        }
        
        # Track best parameters
        self.best_loss = float('inf')
        self.best_params = None
        self.best_synaptic_params = None
        self.best_iteration = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        
        if self.config.optimizer_type.lower() == 'adam':
            return optim.Adam(self.opt_params.parameters(), 
                            lr=self.config.learning_rate,
                            weight_decay=self.config.l2_regularization)
        elif self.config.optimizer_type.lower() == 'sgd':
            return optim.SGD(self.opt_params.parameters(),
                           lr=self.config.learning_rate,
                           momentum=0.9,
                           weight_decay=self.config.l2_regularization)
        elif self.config.optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(self.opt_params.parameters(),
                                 lr=self.config.learning_rate,
                                 momentum=0.1,
                                 weight_decay=self.config.l2_regularization)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        
        if self.config.scheduler_type is None:
            return None
        elif self.config.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        elif self.config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.max_iterations
            )
        elif self.config.scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")
    
    def simulate_circuit(self, synaptic_params: PerConnectionSynapticParams, 
                        mec_drive: float) -> Dict[str, Tensor]:
        """
        Run circuit simulation and return population statistics
        
        Returns:
            Dict with 'firing_rates' and 'sparsity' for each population
        """
        
        # Create circuit with current parameters
        circuit = DentateCircuit(self.circuit_params, synaptic_params, self.opsin_params)
        
        # Run simulation
        circuit.reset_state()
        
        # MEC drive setup
        mec_input = torch.ones(self.circuit_params.n_mec) * mec_drive
        
        # Collect activities over time
        activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        
        for t in range(self.config.simulation_duration):
            external_drive = {'mec': mec_input}
            activities = circuit({}, external_drive)
            
            # Store activities after warmup
            if t >= self.config.warmup_duration:
                for pop in activities_over_time:
                    if pop in activities:
                        activities_over_time[pop].append(activities[pop].clone())
        
        # Calculate statistics
        results = {}
        for pop in activities_over_time:
            if len(activities_over_time[pop]) > 0:
                # Stack time series and compute mean firing rates
                pop_time_series = torch.stack(activities_over_time[pop])  # (time, neurons)
                mean_rates = torch.mean(pop_time_series, dim=0)  # (neurons,)
                
                # Population average firing rate
                pop_firing_rate = torch.mean(mean_rates)
                
                # Sparsity (fraction of neurons above threshold)
                active_neurons = torch.sum(mean_rates > self.targets.activity_threshold)
                sparsity = active_neurons.float() / len(mean_rates)
                
                results[pop] = {
                    'firing_rate': pop_firing_rate,
                    'sparsity': sparsity,
                    'individual_rates': mean_rates
                }
            
        return results
    
    def compute_loss(self, simulation_results: Dict[str, Dict[str, Tensor]], 
                    mec_drive: float) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute multi-component loss function
        """
        
        loss_components = {}
        total_loss = torch.tensor(0.0)
        
        # 1. Firing rate loss
        firing_rate_loss = torch.tensor(0.0)
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop in simulation_results and pop in self.targets.target_rates:
                target_rate = self.targets.target_rates[pop]
                actual_rate = simulation_results[pop]['firing_rate']
                tolerance = self.targets.rate_tolerance[pop]
                
                # Huber loss with tolerance band
                error = torch.abs(actual_rate - target_rate)
                rate_loss = torch.where(
                    error <= tolerance,
                    0.5 * error ** 2,
                    tolerance * error - 0.5 * tolerance ** 2
                )
                
                firing_rate_loss += rate_loss
        
        loss_components['firing_rate'] = firing_rate_loss
        total_loss += self.targets.loss_weights['firing_rate'] * firing_rate_loss
        
        # 2. Sparsity loss  
        sparsity_loss = torch.tensor(0.0)
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if (pop in simulation_results and 
                pop in self.targets.sparsity_targets):
                target_sparsity = self.targets.sparsity_targets[pop]
                actual_sparsity = simulation_results[pop]['sparsity']
                
                sparsity_error = (actual_sparsity - target_sparsity) ** 2
                sparsity_loss += sparsity_error
        
        loss_components['sparsity'] = sparsity_loss
        total_loss += self.targets.loss_weights['sparsity'] * sparsity_loss
        
        # 3. Regularization losses
        bounds_penalty = self.opt_params.compute_bounds_penalty()
        smoothness_penalty = self.opt_params.compute_smoothness_penalty()
        
        loss_components['bounds'] = bounds_penalty
        loss_components['smoothness'] = smoothness_penalty
        
        total_loss += self.targets.loss_weights['bounds'] * bounds_penalty
        total_loss += self.targets.loss_weights['smoothness'] * smoothness_penalty
        
        return total_loss, loss_components
    
    def optimize_step(self) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Single optimization step across all MEC drive conditions
        """
        
        self.optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0)
        combined_loss_components = {}
        all_firing_rates = {}
        
        # Test across multiple MEC drive levels for robustness
        for mec_drive in self.config.mec_drive_levels:
            
            # Average over multiple trials for stability
            trial_loss = torch.tensor(0.0)
            trial_components = {}
            
            for trial in range(self.config.n_trials):
                
                # Create synaptic parameters with current optimization state
                current_synaptic_params = self.opt_params.create_synaptic_params()
                
                # Simulate circuit
                sim_results = self.simulate_circuit(current_synaptic_params, mec_drive)
                
                # Compute loss
                loss, loss_components = self.compute_loss(sim_results, mec_drive)
                
                trial_loss += loss
                
                # Accumulate loss components
                for comp_name, comp_value in loss_components.items():
                    if comp_name not in trial_components:
                        trial_components[comp_name] = torch.tensor(0.0)
                    trial_components[comp_name] += comp_value
                
                # Store firing rates for this trial
                trial_key = f"mec_{mec_drive}_trial_{trial}"
                all_firing_rates[trial_key] = {
                    pop: results['firing_rate'].item() 
                    for pop, results in sim_results.items()
                }
            
            # Average trial results
            trial_loss /= self.config.n_trials
            for comp_name in trial_components:
                trial_components[comp_name] /= self.config.n_trials
            
            total_loss += trial_loss
            
            # Accumulate across drive conditions
            for comp_name, comp_value in trial_components.items():
                if comp_name not in combined_loss_components:
                    combined_loss_components[comp_name] = torch.tensor(0.0)
                combined_loss_components[comp_name] += comp_value
        
        # Average across drive conditions
        total_loss /= len(self.config.mec_drive_levels)
        for comp_name in combined_loss_components:
            combined_loss_components[comp_name] /= len(self.config.mec_drive_levels)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.opt_params.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Convert to regular values for logging
        loss_values = {
            name: value.item() for name, value in combined_loss_components.items()
        }
        
        return total_loss.item(), loss_values, all_firing_rates
    
    def optimize(self) -> Dict[str, any]:
        """
        Main optimization loop - returns BEST parameters, not final parameters
        """
        
        print("Starting connection parameter optimization...")
        print(f"Target firing rates: {self.targets.target_rates}")
        print(f"MEC drive levels: {self.config.mec_drive_levels}")
        
        patience_counter = 0
        
        pbar = tqdm.tqdm(range(self.config.max_iterations), desc="Optimizing")
        
        for iteration in pbar:
            
            # Optimization step
            loss, loss_components, firing_rates = self.optimize_step()
            
            # Store history
            self.history['loss'].append(loss)
            self.history['loss_components'].append(loss_components)
            self.history['firing_rates'].append(firing_rates)
            self.history['parameters'].append(self.opt_params.get_connection_modulation())
            
            # Track best parameters
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_iteration = iteration
                # Deep copy current parameters
                self.best_params = self.opt_params.get_connection_modulation().copy()
                self.best_synaptic_params = self.opt_params.create_synaptic_params()
                patience_counter = 0
                print(f"\nNew best loss: {loss:.6f} at iteration {iteration}")
                print(f"  Parameters: {self.best_synaptic_params}")
            else:
                patience_counter += 1
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss:.6f}",
                'best': f"{self.best_loss:.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Early stopping check
            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping at iteration {iteration}")
                print(f"Best loss {self.best_loss:.6f} found at iteration {self.best_iteration}")
                break
            
            # Log every 50 iterations
            if iteration % 50 == 0:
                print(f"\nIteration {iteration}:")
                print(f"  Current loss: {loss:.6f}")
                print(f"  Best loss: {self.best_loss:.6f} (iteration {self.best_iteration})")
                for comp_name, comp_value in loss_components.items():
                    print(f"  {comp_name}: {comp_value:.6f}")
                
                current_params = self.opt_params.get_connection_modulation()
                print(f"  Current connection strengths (sample):")
                for conn_name in list(current_params.keys())[:3]:
                    print(f"    {conn_name}: {current_params[conn_name]:.3f}")
        
        print(f"\nOptimization completed!")
        print(f"Best loss: {self.best_loss:.6f} at iteration {self.best_iteration}")
        
        # Return best results
        return {
            'optimized_connection_modulation': self.best_params,
            'optimized_synaptic_params': self.best_synaptic_params,
            'best_loss': self.best_loss,
            'best_iteration': self.best_iteration,
            'final_loss': loss,  # Also include final loss for comparison
            'history': self.history,
            'n_iterations': len(self.history['loss'])
        }

    def print_best_firing_rates(self, mec_drive=100.0):
        """Print actual firing rates achieved by the best parameters"""

        if self.best_synaptic_params is None:
            print("No best parameters found. Run optimize() first.")
            return

        print("Best configuration firing rates")
        print(f"{'='*50}")
        print(f"Best loss: {self.best_loss:.6f} (iteration {self.best_iteration})")

        # Create circuit with best parameters
        circuit = DentateCircuit(self.circuit_params, self.best_synaptic_params, self.opsin_params)
        circuit.reset_state()

        # Run simulation to steady state
        n_steps = 400
        warmup = 150
        mec_input = torch.ones(self.circuit_params.n_mec) * mec_drive

        activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}

        for t in range(n_steps):
            external_drive = {'mec': mec_input}
            activities = circuit({}, external_drive)

            if t >= warmup:
                for pop in activities_over_time:
                    if pop in activities:
                        activities_over_time[pop].append(activities[pop].clone())

        # Calculate and print results
        print(f"MEC Drive: {mec_drive:.1f} pA\n")
        print(f"{'Population':<12} {'Target':<8} {'Actual':<8} {'Error':<8} {'Sparsity':<10}")
        print("-" * 50)

        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop in activities_over_time and len(activities_over_time[pop]) > 0:
                pop_time_series = torch.stack(activities_over_time[pop])
                mean_rates = torch.mean(pop_time_series, dim=0)

                actual_rate = torch.mean(mean_rates).item()
                target_rate = self.targets.target_rates.get(pop, 0.0)
                error = ((actual_rate - target_rate) / target_rate * 100) if target_rate > 0 else 0.0
                sparsity = (torch.sum(mean_rates > self.targets.activity_threshold) / len(mean_rates)).item()

                print(f"{pop.upper():<12} {target_rate:<8.1f} {actual_rate:<8.1f} {error:<+7.1f}% {sparsity:<10.3f}")
                
    def restore_best_parameters(self):
        """
        Restore the optimizer's parameters to the best state found during optimization
        Useful if you want to continue optimization from the best point
        """
        if self.best_params is None:
            print("Warning: No best parameters found. Run optimize() first.")
            return
        
        print(f"Restoring parameters to best state (loss: {self.best_loss:.6f}, iteration: {self.best_iteration})")
        
        # Restore log parameters
        for conn_name, best_value in self.best_params.items():
            if conn_name in self.opt_params.log_modulation:
                self.opt_params.log_modulation[conn_name].data = torch.log(torch.tensor(best_value))
    
    def get_parameter_trajectory(self, connection_name: str) -> List[float]:
        """
        Get the optimization trajectory for a specific connection parameter
        """
        trajectory = []
        for param_dict in self.history['parameters']:
            if connection_name in param_dict:
                trajectory.append(param_dict[connection_name])
        return trajectory

    def save_best_results_to_json(self, filename, mec_drive_levels=[100.0]):
        """Save best optimization results and firing rates to JSON file"""

        if self.best_synaptic_params is None:
            print("No best parameters found. Run optimize() first.")
            return

        import json
        from datetime import datetime

        # Collect firing rates and sparsity for each MEC drive level
        performance_data = {}

        for mec_drive in mec_drive_levels:
            # Get firing rates for this drive level (reuse simulation logic)
            firing_data = self._get_firing_rates_for_drive(mec_drive)
            performance_data[f'mec_{mec_drive}'] = firing_data

        # Prepare JSON data
        results_data = {
            'optimization_info': {
                'timestamp': datetime.now().isoformat(),
                'best_loss': float(self.best_loss),
                'best_iteration': int(self.best_iteration),
                'total_iterations': len(self.history['loss']),
                'final_loss': float(self.history['loss'][-1]) if self.history['loss'] else None,
            },
            'targets': {
                'firing_rates': self.targets.target_rates,
                'sparsity_targets': self.targets.sparsity_targets,
                'activity_threshold': self.targets.activity_threshold,
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

        print(f"Results saved to {filename}")
        return results_data

    def _get_firing_rates_for_drive(self, mec_drive):
        """Helper method to get firing rates for a specific MEC drive level"""

        circuit = DentateCircuit(self.circuit_params, self.best_synaptic_params, self.opsin_params)
        circuit.reset_state()

        # Run simulation
        n_steps = 400
        warmup = 150
        mec_input = torch.ones(self.circuit_params.n_mec) * mec_drive

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

def evaluate_particle_worker(args):
    """
    Worker function for parallel particle evaluation
    Must be at module level for multiprocessing.Pool to pickle it
    """
    position, connection_names, circuit_factory_data, targets, config = args

    try:
        # Recreate circuit factory from serialized data
        circuit_params, base_synaptic_params, opsin_params = circuit_factory_data
        
        def circuit_factory(connection_modulation):
            from DG_circuit_dendritic_somatic_transfer import (
                DentateCircuit, PerConnectionSynapticParams
            )
            synaptic_params = PerConnectionSynapticParams(
                ampa_g_mean=base_synaptic_params['ampa_g_mean'],
                ampa_g_std=base_synaptic_params['ampa_g_std'],
                gaba_g_mean=base_synaptic_params['gaba_g_mean'],
                gaba_g_std=base_synaptic_params['gaba_g_std'],
                distribution=base_synaptic_params['distribution'],
                connection_modulation=connection_modulation
            )
            return DentateCircuit(circuit_params, synaptic_params, opsin_params)
        
        # Convert position array to connection dict
        connection_modulation = dict(zip(connection_names, position))
        
        # Create circuit
        circuit = circuit_factory(connection_modulation)
        
        # Force CPU evaluation in worker processes
        device = torch.device('cpu')
        
        # Evaluate circuit
        total_loss = 0.0
        
        for mec_drive in config.mec_drive_levels:
            for trial in range(config.n_trials):
                firing_rates = {}
                circuit.reset_state()
                
                # Run simulation
                mec_input = torch.ones(circuit.circuit_params.n_mec, device=device) * mec_drive
                activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
                
                for t in range(config.simulation_duration):
                    external_drive = {'mec': mec_input}
                    activities = circuit({}, external_drive)
                    
                    if t >= config.warmup_duration:
                        for pop in activities_over_time:
                            if pop in activities:
                                activities_over_time[pop].append(activities[pop].clone().cpu())
                
                # Calculate loss
                trial_loss = 0.0
                for pop in activities_over_time:
                    if len(activities_over_time[pop]) > 0:
                        pop_time_series = torch.stack(activities_over_time[pop])
                        mean_rates = torch.mean(pop_time_series, dim=0)
                        
                        # Firing rate loss
                        if pop in targets.target_rates:
                            target_rate = targets.target_rates[pop]
                            actual_rate = torch.mean(mean_rates).item()
                            firing_rates[pop] = actual_rate
                            tolerance = targets.rate_tolerance[pop]
                            if np.isclose(actual_rate, 0.0, 1e-2, 1e-2):
                                rate_loss = 1e2 # large loss to discourage solutions with zero mean firing rates
                            else:
                                error = abs(actual_rate - target_rate)
                                if error > tolerance:
                                    rate_loss = error - tolerance
                                else:
                                    rate_loss = 0.5 * (error / tolerance) ** 2
                                    
                            trial_loss += rate_loss
                        
                        # Sparsity loss
                        if pop in targets.sparsity_targets:
                            target_sparsity = targets.sparsity_targets[pop]
                            actual_sparsity = (torch.sum(mean_rates > targets.activity_threshold) / len(mean_rates)).item()
                            sparsity_error = (actual_sparsity - target_sparsity) ** 2
                            trial_loss += sparsity_error

                # Add constraint violations
                constraint_violation, violations_info = evaluate_rate_ordering_constraints(
                    firing_rates, 
                    targets.rate_ordering_constraints
                )
                
                # Add weighted constraint penalty
                trial_loss += targets.constraint_violation_weight * constraint_violation
                            
                total_loss += trial_loss
                
                
        # Average across trials and conditions
        total_loss /= (len(config.mec_drive_levels) * config.n_trials)
        
        return position, total_loss, connection_modulation
        
    except Exception as e:
        print(f"Worker error: {e}")
        return position, 1e6, dict(zip(connection_names, position))

    
def evaluate_de_candidate_worker(param_array, connection_names, circuit_factory_data, targets, config):
    """
    Worker function for differential evolution parallel evaluation
    Must be at module level for scipy's workers parameter
    
    Note: scipy expects the objective function signature to be just (param_array,)
    so we need to use functools.partial to bind the other arguments
    """

    try:
        from DG_circuit_dendritic_somatic_transfer import (
            DentateCircuit, PerConnectionSynapticParams
        )
        
        # Unpack circuit factory data
        circuit_params, base_synaptic_params_dict, opsin_params = circuit_factory_data
        
        # Convert parameter array to connection modulation dict
        connection_modulation = dict(zip(connection_names, param_array))
        
        # Create synaptic parameters
        synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=base_synaptic_params_dict['ampa_g_mean'],
            ampa_g_std=base_synaptic_params_dict['ampa_g_std'],
            gaba_g_mean=base_synaptic_params_dict['gaba_g_mean'],
            gaba_g_std=base_synaptic_params_dict['gaba_g_std'],
            distribution=base_synaptic_params_dict['distribution'],
            connection_modulation=connection_modulation
        )
        
        # Create circuit
        circuit = DentateCircuit(circuit_params, synaptic_params, opsin_params)

        
        # Force CPU
        device = torch.device('cpu')
        
        # Evaluate circuit
        total_loss = 0.0
        
        for mec_drive in config.mec_drive_levels:
            for trial in range(config.n_trials):
                firing_rates = {}
                circuit.reset_state()
                
                # Run simulation
                mec_input = torch.ones(circuit.circuit_params.n_mec, device=device) * mec_drive
                activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
                
                for t in range(config.simulation_duration):
                    external_drive = {'mec': mec_input}
                    activities = circuit({}, external_drive)
                    
                    if t >= config.warmup_duration:
                        for pop in activities_over_time:
                            if pop in activities:
                                activities_over_time[pop].append(activities[pop].clone().cpu())
                
                # Calculate loss
                trial_loss = 0.0
                for pop in activities_over_time:
                    if len(activities_over_time[pop]) > 0:
                        pop_time_series = torch.stack(activities_over_time[pop])
                        mean_rates = torch.mean(pop_time_series, dim=0)
                        
                        # Firing rate loss
                        if pop in targets.target_rates:
                            target_rate = targets.target_rates[pop]
                            actual_rate = torch.mean(mean_rates).item()
                            firing_rates[pop] = actual_rate
                            if np.isclose(actual_rate, 0.0, 1e-2, 1e-2):
                                rate_loss = 1e2 # large loss to discourage solutions with zero mean firing rates
                            else:
                                tolerance = targets.rate_tolerance[pop]

                                error = abs(actual_rate - target_rate)
                                if error > tolerance:
                                    rate_loss = error - tolerance
                                else:
                                    rate_loss = 0.5 * (error / tolerance) ** 2
                                    
                            trial_loss += rate_loss
                        
                        # Sparsity loss
                        if pop in targets.sparsity_targets:
                            target_sparsity = targets.sparsity_targets[pop]
                            actual_sparsity = (torch.sum(mean_rates > targets.activity_threshold) / len(mean_rates)).item()
                            sparsity_error = (actual_sparsity - target_sparsity) ** 2
                            trial_loss += sparsity_error

                # Add constraint violations
                constraint_violation, violations_info = evaluate_rate_ordering_constraints(
                    firing_rates, 
                    targets.rate_ordering_constraints
                )
                
                # Add weighted constraint penalty
                trial_loss += targets.constraint_violation_weight * constraint_violation
                
                total_loss += trial_loss
        
        # Average across trials and conditions
        total_loss /= (len(config.mec_drive_levels) * config.n_trials)

        return total_loss
        
    except Exception as e:
        print(f"DE worker error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return 1e6
    
class GlobalCircuitOptimizer:
    """
    Global optimization strategies for circuit parameter tuning
    """
    
    def __init__(self, circuit_params: CircuitParams, 
                 base_synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 targets, config):
        """
        Args:

            targets: OptimizationTargets instance
            config: OptimizationConfig instance
        """
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params

        self.targets = targets
        self.config = config
        self.best_loss = float('inf')
        self.best_params = None
        self.history = {'loss': [], 'parameters': []}
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

    def _create_circuit(self, connection_modulation):
        """Create circuit with given connection modulation"""
        from DG_circuit_dendritic_somatic_transfer import (
            DentateCircuit, PerConnectionSynapticParams
        )
        
        synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=self.base_synaptic_params.ampa_g_mean,
            ampa_g_std=self.base_synaptic_params.ampa_g_std,
            gaba_g_mean=self.base_synaptic_params.gaba_g_mean,
            gaba_g_std=self.base_synaptic_params.gaba_g_std,
            distribution=self.base_synaptic_params.distribution,
            connection_modulation=connection_modulation
        )
        
        return DentateCircuit(self.circuit_params, synaptic_params, self.opsin_params)

        
    def _evaluate_parameters(self, param_array, connection_names):
        """
        Evaluate circuit performance for given parameter array
        """
        try:
            # Convert array to connection modulation dict
            connection_modulation = dict(zip(connection_names, param_array))
            
            # Create circuit with these parameters
            circuit = self._create_circuit(connection_modulation)
            
            # Evaluate circuit (similar to your existing evaluation)
            total_loss = 0.0
            
            for mec_drive in self.config.mec_drive_levels:
                for trial in range(self.config.n_trials):
                    circuit.reset_state()
                    
                    # Run simulation
                    mec_input = torch.ones(circuit.circuit_params.n_mec) * mec_drive
                    activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
                    
                    for t in range(self.config.simulation_duration):
                        external_drive = {'mec': mec_input}
                        activities = circuit({}, external_drive)
                        
                        if t >= self.config.warmup_duration:
                            for pop in activities_over_time:
                                if pop in activities:
                                    activities_over_time[pop].append(activities[pop].clone())
                    
                    # Calculate loss components
                    trial_loss = self._calculate_loss_from_activities(activities_over_time)
                    total_loss += trial_loss
            
            # Average across trials and conditions
            total_loss /= (len(self.config.mec_drive_levels) * self.config.n_trials)
            
            # Store if best
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_params = connection_modulation.copy()
                print(f"New best loss: {total_loss:.6f}")
            
            self.history['loss'].append(total_loss)
            self.history['parameters'].append(connection_modulation.copy())
            
            return total_loss
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return 1e6  # Large penalty for failed evaluations
    
    def _calculate_loss_from_activities(self, activities_over_time):
        """Calculate loss from activity time series"""
        
        total_loss = 0.0
        firing_rates = {}

        
        for pop in activities_over_time:
            if len(activities_over_time[pop]) > 0:
                pop_time_series = torch.stack(activities_over_time[pop])
                mean_rates = torch.mean(pop_time_series, dim=0)
                
                # Firing rate loss
                if pop in self.targets.target_rates:
                    target_rate = self.targets.target_rates[pop]
                    actual_rate = torch.mean(mean_rates).item()
                    firing_rates[pop] = actual_rate
                    tolerance = self.targets.rate_tolerance[pop]
                    if np.isclose(actual_rate, 0.0, 1e-2, 1e-2):
                        rate_loss = 1e2 # large loss to discourage solutions with zero mean firing rates
                    else:
                        error = abs(actual_rate - target_rate)
                        if error > tolerance:
                            rate_loss = error - tolerance
                        else:
                            rate_loss = 0.5 * (error / tolerance) ** 2
                    
                    total_loss += rate_loss
                
                # Sparsity loss
                if pop in self.targets.sparsity_targets:
                    target_sparsity = self.targets.sparsity_targets[pop]
                    actual_sparsity = (torch.sum(mean_rates > self.targets.activity_threshold) / len(mean_rates)).item()
                    sparsity_error = (actual_sparsity - target_sparsity) ** 2
                    total_loss += sparsity_error

        # Add constraint violations
        constraint_violation, violations_info = evaluate_rate_ordering_constraints(
            firing_rates, 
            self.targets.rate_ordering_constraints
        )

        # Add weighted constraint penalty
        total_loss += self.targets.constraint_violation_weight * constraint_violation
            
        # Store violation info for logging
        if hasattr(self, 'last_violations'):
            self.last_violations = violations_info

                    
        return total_loss
    
    def optimize_differential_evolution(self, connection_names, bounds_dict, n_workers=1, n_threads_per_worker=1):
        """
        Use scipy's Differential Evolution with parallel workers

        Args:
            connection_names: List of connection names to optimize
            bounds_dict: Dict of (min, max) bounds for each connection
            n_workers: Number of parallel workers (None = use all CPUs, 1 = sequential)
        """
        print("Starting Differential Evolution optimization...")

        n_cores = mp.cpu_count()

        if n_workers is None:
            # Auto-calculate: use all cores with controlled threading
            n_workers = max(1, n_cores // max(1, n_threads_per_worker))
            
        total_threads = n_workers * n_threads_per_worker
    
        print("Starting Differential Evolution optimization...")
        print(f"System configuration:")
        print(f"  Total CPU cores: {n_cores}")
        print(f"  Workers: {n_workers}")
        print(f"  Threads per worker: {n_threads_per_worker}")
        print(f"  Total threads used: {total_threads}")
        
        if total_threads > n_cores:
            print(f"  WARNING: Using {total_threads} threads on {n_cores} cores may cause contention")

        configure_torch_threads(n_threads_per_worker)
            
        # Convert bounds dict to list of tuples in correct order
        bounds = [bounds_dict.get(name, (0.1, 5.0)) for name in connection_names]

        # Create objective function with bound parameters
        # We need to use functools.partial to bind the extra arguments
        from functools import partial

        objective = partial(
            evaluate_de_candidate_worker,
            connection_names=connection_names,
            circuit_factory_data=self.circuit_factory_data,
            targets=self.targets,
            config=self.config
        )

        # Callback to track progress and store history
        def callback(xk, convergence):
            """Called after each iteration"""

            loss = evaluate_de_candidate_worker(
                xk, connection_names, self.circuit_factory_data, 
                self.targets, self.config
            )
            connection_modulation = dict(zip(connection_names, xk))

            self.history['loss'].append(loss)
            self.history['parameters'].append(connection_modulation)

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = connection_modulation.copy()
                print(f"DE generation complete - Best loss: {loss:.6f}")

            return False  # Don't stop early

        
        # Differential Evolution with parallel workers
        result = differential_evolution(
            objective,
            bounds,
            #maxiter=self.config.max_iterations, # Use fewer iterations with larger population
            maxiter=2, # Use fewer iterations with larger population
            popsize=20,           # Population size multiplier
            mutation=(0.1, 1.0),  # Mutation range
            recombination=0.7,    # Crossover probability
            seed=42,              # Reproducibility
            disp=True,            # Display progress
            polish=False,         # Don't use local polish
            workers=n_workers if n_workers > 1 else 1,  # Parallel workers
            updating='deferred',  # Evaluate entire generation before updating
            callback=callback,    # Track progress
        )

        print(f"\nDifferential Evolution completed:")
        print(f"  Success: {result.success}")
        print(f"  Best loss: {result.fun:.6f}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Iterations: {result.nit}")

        return {
            'optimized_connection_modulation': self.best_params,
            'best_loss': self.best_loss,
            'n_evaluations': result.nfev,
            'n_iterations': result.nit,
            'success': result.success,
            'history': self.history
        }

    def optimize_particle_swarm(self, connection_names, bounds_dict, n_workers=1, n_threads_per_worker=1):
        """
        Particle Swarm Optimization with multiprocessing

        Args:
            connection_names: List of connection names to optimize
            bounds_dict: Dict of (min, max) bounds for each connection
            n_workers: Number of parallel workers (None = use all CPUs)
        """
        n_cores = mp.cpu_count()

        if n_workers is None:
            # Auto-calculate: use all cores with controlled threading
            n_workers = max(1, n_cores // max(1, n_threads_per_worker))
            
        total_threads = n_workers * n_threads_per_worker
    
        print("Starting Particle Swarm optimization...")
        print(f"System configuration:")
        print(f"  Total CPU cores: {n_cores}")
        print(f"  Workers: {n_workers}")
        print(f"  Threads per worker: {n_threads_per_worker}")
        print(f"  Total threads used: {total_threads}")
        
        if total_threads > n_cores:
            print(f"  WARNING: Using {total_threads} threads on {n_cores} cores may cause contention")

        configure_torch_threads(n_threads_per_worker)

        n_particles = 20
        n_dimensions = len(connection_names)
        max_iterations = self.config.max_iterations

        # PSO parameters
        w = 0.7        # Inertia weight
        c1 = 1.5       # Cognitive parameter
        c2 = 1.5       # Social parameter

        # Initialize particles
        bounds = [bounds_dict.get(name, (0.1, 5.0)) for name in connection_names]
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])

        # Initialize positions and velocities
        positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, n_dimensions))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_dimensions))

        # Initialize best positions
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(n_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        # Create multiprocessing pool
        # Use 'spawn' method to avoid issues with CUDA/PyTorch
        ctx = mp.get_context('spawn')

        for iteration in range(max_iterations):
            print(f"PSO Iteration {iteration+1}/{max_iterations}")

            # Prepare arguments for parallel evaluation
            eval_args = [
                (positions[i], connection_names, self.circuit_factory_data, 
                 self.targets, self.config)
                for i in range(n_particles)
            ]

            # Evaluate all particles in parallel
            with ctx.Pool(processes=n_workers) as pool:
                results = pool.map(evaluate_particle_worker, eval_args)
                
            # Process results
            for i, (position, score, connection_modulation) in enumerate(results):
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = position.copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = position.copy()
                    self.best_params = connection_modulation.copy()
                    print(f"  New global best: {global_best_score:.6f}")
                    print(f"  Parameters: {self.best_params}")

                # Store in history
                self.history['loss'].append(score)
                self.history['parameters'].append(connection_modulation)

            # Update velocities and positions
            for i in range(n_particles):
                # Update velocity
                r1, r2 = np.random.random(n_dimensions), np.random.random(n_dimensions)

                cognitive_component = c1 * r1 * (personal_best_positions[i] - positions[i])
                social_component = c2 * r2 * (global_best_position - positions[i])

                velocities[i] = (w * velocities[i] + 
                               cognitive_component + 
                               social_component)

                # Update position
                positions[i] += velocities[i]

                # Enforce bounds
                positions[i] = np.clip(positions[i], lower_bounds, upper_bounds)

            self.best_loss = global_best_score

        return {
            'optimized_connection_modulation': self.best_params,
            'best_loss': global_best_score,
            'n_evaluations': max_iterations * n_particles,
            'history': self.history
        }
    
    def optimize_simulated_annealing(self, connection_names, bounds_dict, initial_params=None):
        """
        Simulated Annealing with basin hopping
        
        Good for escaping local minima with gradual cooling
        """
        print("Starting Simulated Annealing optimization...")
        
        # Initial parameters
        if initial_params is None:
            bounds = [bounds_dict.get(name, (0.1, 5.0)) for name in connection_names]
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])  # Start at midpoint
        else:
            x0 = np.array([initial_params[name] for name in connection_names])
        
        def objective(param_array):
            return self._evaluate_parameters(param_array, connection_names)
        
        # Custom step taking function
        class BoundedStep:
            def __init__(self, bounds, stepsize=0.1):
                self.bounds = bounds
                self.stepsize = stepsize
            
            def __call__(self, x):
                """Take a random step within bounds"""
                lower_bounds = np.array([b[0] for b in self.bounds])
                upper_bounds = np.array([b[1] for b in self.bounds])
                
                # Random step
                step = np.random.normal(0, self.stepsize, size=x.shape)
                x_new = x + step
                
                # Enforce bounds by reflection
                x_new = np.clip(x_new, lower_bounds, upper_bounds)
                return x_new
        
        bounds = [bounds_dict.get(name, (0.1, 5.0)) for name in connection_names]
        step_function = BoundedStep(bounds, stepsize=0.05)
        
        # Basin hopping with simulated annealing
        result = basinhopping(
            objective,
            x0,
            niter=50,              # Number of basin hops
            T=1.0,                 # Temperature for accepting worse solutions
            stepsize=0.05,         # Step size for random displacement
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'bounds': bounds,
                'options': {'maxiter': 20}  # Limited local optimization
            },
            take_step=step_function,
            disp=True
        )
        
        return {
            'optimized_connection_modulation': dict(zip(connection_names, result.x)),
            'best_loss': result.fun,
            'n_evaluations': result.nfev,
            'success': result.lowest_optimization_result.success,
            'history': self.history
        }


    def _get_firing_rates_for_drive(self, mec_drive):
        """Helper method to get firing rates for a specific MEC drive level"""

        circuit = self._create_circuit(self.best_params)
        circuit.reset_state()

        # Run simulation
        n_steps = 600
        warmup = 150
        mec_input = torch.ones(self.circuit_params.n_mec) * mec_drive

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

    def print_best_firing_rates(self, mec_drive=100.0):
        """Print actual firing rates achieved by the best parameters"""

        if self.best_params is None:
            print("No best parameters found. Run optimize() first.")
            return

        print("Best configuration firing rates")
        print(f"{'='*50}")
        print(f"Best loss: {self.best_loss:.6f}")

        # Create circuit with best parameters
        circuit = self._create_circuit(self.best_params)
        circuit.reset_state()

        # Run simulation to steady state
        n_steps = 400
        warmup = 150
        mec_input = torch.ones(self.circuit_params.n_mec) * mec_drive

        activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}

        for t in range(n_steps):
            external_drive = {'mec': mec_input}
            activities = circuit({}, external_drive)

            if t >= warmup:
                for pop in activities_over_time:
                    if pop in activities:
                        activities_over_time[pop].append(activities[pop].clone())

        # Calculate and print results
        print(f"MEC Drive: {mec_drive:.1f} pA\n")
        print(f"{'Population':<12} {'Target':<8} {'Actual':<8} {'Error':<8} {'Sparsity':<10}")
        print("-" * 50)

        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop in activities_over_time and len(activities_over_time[pop]) > 0:
                pop_time_series = torch.stack(activities_over_time[pop])
                mean_rates = torch.mean(pop_time_series, dim=0)

                actual_rate = torch.mean(mean_rates).item()
                target_rate = self.targets.target_rates.get(pop, 0.0)
                error = ((actual_rate - target_rate) / target_rate * 100) if target_rate > 0 else 0.0
                sparsity = (torch.sum(mean_rates > self.targets.activity_threshold) / len(mean_rates)).item()

                print(f"{pop.upper():<12} {target_rate:<8.1f} {actual_rate:<8.1f} {error:<+7.1f}% {sparsity:<10.3f}")

    def save_best_results_to_json(self, filename, mec_drive_levels=[100.0]):
        """Save best optimization results and firing rates to JSON file"""

        if self.best_params is None:
            print("No best parameters found. Run optimize() first.")
            return

        import json
        from datetime import datetime

        # Collect firing rates and sparsity for each MEC drive level
        performance_data = {}

        for mec_drive in mec_drive_levels:
            # Get firing rates for this drive level (reuse simulation logic)
            firing_data = self._get_firing_rates_for_drive(mec_drive)
            performance_data[f'mec_{mec_drive}'] = firing_data

        # Prepare JSON data
        results_data = {
            'optimization_info': {
                'timestamp': datetime.now().isoformat(),
                'best_loss': float(self.best_loss),
                'best_iteration': len(self.history['loss']),
                'total_iterations': len(self.history['loss']),
                'final_loss': float(self.history['loss'][-1]) if self.history['loss'] else None,
            },
            'targets': {
                'firing_rates': self.targets.target_rates,
                'sparsity_targets': self.targets.sparsity_targets,
                'activity_threshold': self.targets.activity_threshold,
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

        print(f"Results saved to {filename}")
        return results_data


    
def create_default_targets() -> OptimizationTargets:
    """Create default optimization targets based on experimental data"""
    return OptimizationTargets(
        target_rates={
            'gc': 0.5,  # Low granule cell activity for sparsity
            'mc': 1.1,   # Moderate mossy cell activity  
            'pv': 6.0,   # Fast-spiking PV interneurons
            'sst': 4.0,   # Slower SST interneurons
        },
        sparsity_targets={
            'gc': 0.08,   # ~8% granule cells active (key constraint)
            'mc': 0.50,   # Most mossy cells active
            'pv': 0.85,   # Most PV cells active
            'sst': 0.60,  # Moderate SST activity
        }
    )


def create_default_config() -> OptimizationConfig:
    """Create default optimization configuration"""
    return OptimizationConfig(
        learning_rate=0.1,
        max_iterations=300,
        mec_drive_levels=[40.0],
        n_trials=2,  # Reduce for faster iteration
        simulation_duration=600,
        warmup_duration=100,
    )

def create_default_global_opt_config() -> OptimizationConfig:
    """Create default optimization configuration for global optimization"""
    return OptimizationConfig(
        learning_rate=0.1,
        max_iterations=20,
        mec_drive_levels=[40.0],
        n_trials=2,  # Reduce for faster iteration
        simulation_duration=800,
        warmup_duration=200,
    )


def run_optimization(config: OptimizationConfig):
    """
    Example usage of the optimization framework
    """
    
    # Create circuit and optimization components
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    
    base_synaptic_params = PerConnectionSynapticParams(
        distribution='lognormal'
    )
    
    targets = create_default_targets()
    
    # Run optimization
    optimizer = CircuitOptimizer(
        circuit_params, base_synaptic_params, opsin_params, targets, config
    )
    
    results = optimizer.optimize()
    
    print(f"Final loss: {results['final_loss']:.6f}")
    print(f"Iterations: {results['n_iterations']}")

    # Print the actual firing rates for the best configuration  
    optimizer.print_best_firing_rates(mec_drive=self.config.mec_drive_levels[0])
    optimizer.save_best_results_to_json(
        'DG_optimization_results.json',
        mec_drive_levels=self.config.mec_drive_levels
    )

    # Test multiple drive levels
    for drive in [150.0, 200.0, 250.0]:
        optimizer.print_best_firing_rates(mec_drive=drive)
    
    print("\nOptimized connection modulation factors:")
    for conn_name, strength in results['optimized_connection_modulation'].items():
        baseline = base_synaptic_params.connection_modulation.get(conn_name, 1.0)
        change = (strength / baseline - 1) * 100
        print(f"  {conn_name:10s}: {strength:.3f} ({change:+3.2f}%)")

        
    return results, optimizer


def run_global_optimization(optimization_config, n_workers=1):
    """Global optimization for circuit parameter tuning"""
    
    # Setup circuit factory
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    
    def circuit_factory(connection_modulation):
        synaptic_params = PerConnectionSynapticParams(
            connection_modulation=connection_modulation
        )
        return DentateCircuit(circuit_params, synaptic_params, opsin_params)
    
    # Setup optimization targets
    targets = create_default_targets()
    targets = auto_adjust_targets_for_constraints(targets)
    
    circuit_params = CircuitParams()
    base_synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()
    
    # Create global optimizer
    global_opt = GlobalCircuitOptimizer(circuit_params,
                                        base_synaptic_params,
                                        opsin_params,
                                        targets, optimization_config)
    
    # Try different methods
    methods = {
#        'differential_evolution': global_opt.optimize_differential_evolution,
        'particle_swarm': global_opt.optimize_particle_swarm
    }
    
    results = {}
    for method_name, method_func in methods.items():
        print(f"\n{'='*50}")
        print(f"Testing {method_name.upper()}")
        print(f"{'='*50}")
        
        global_opt.best_loss = float('inf')  # Reset for fair comparison
        global_opt.history = {'loss': [], 'parameters': []}
        
        try:
            result = method_func(list(targets.connection_bounds.keys()),
                                 targets.connection_bounds,
                                 n_workers=n_workers)
            results[method_name] = result
            
            print(f"\n{method_name} Results:")
            print(f"Best loss: {result['best_loss']:.6f}")
            print(f"Evaluations: {result.get('n_evaluations', 'N/A')}")
            print("Optimized parameters:")
            optimized_parameters = result['optimized_connection_modulation']
            for conn in sorted(optimized_parameters.keys()):
                value = optimized_parameters[conn]
                print(f"  {conn}: {value:.3f}")
                
            global_opt.print_best_firing_rates(mec_drive=global_opt.config.mec_drive_levels[0])
            global_opt.save_best_results_to_json(
                'DG_optimization_results.json',
                mec_drive_levels=global_opt.config.mec_drive_levels
            )
                
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = {'error': str(e)}
    
    return results, global_opt

if __name__ == "__main__":
    
    current_threads = torch.get_num_threads()
    print(f"Current number of threads: {current_threads}")

    optimization_config = create_default_global_opt_config()
#    results, optimizer = run_optimization(optimization_config)
    results, optimizer = run_global_optimization(optimization_config, n_workers=10)

