#!/usr/bin/env python3
"""
Statistical testing framework for disinhibition hypothesis in DG circuit
Compatible with DG_circuit_dendritic_somatic_transfer.py and DG_protocol.py
Supports batch GPU evaluation for efficient parallel trial execution.
"""

import sys
import torch
import numpy as np
import scipy.stats as stats
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import pickle
import json
import pprint
from pathlib import Path
import tqdm

# Import DG circuit components
from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit,
    CircuitParams,
    PerConnectionSynapticParams,
    OpsinParams,
    get_default_device
)
from DG_protocol import (
    OptogeneticExperiment,
    OpsinExpression
)

# Import batch circuit components for GPU acceleration
from DG_batch_circuit_dendritic_somatic_transfer import (
    BatchDentateCircuit
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class ExperimentalBenchmarks:
    """Experimental data from Hainmueller et al. 2024 for validation"""
    
    # PV stimulation results
    pv_paradoxical_gc_fraction: float = 0.09  # ~8.6% of GCs show paradoxical excitation
    pv_paradoxical_mc_fraction: float = 0.08  # ~8% of MCs
    pv_paradoxical_sst_fraction: float = 0.12
    pv_gini_increase: float = 0.15  # Increase in firing rate inequality

    # SST stimulation results  
    sst_paradoxical_gc_fraction: float = 0.02
    sst_paradoxical_mc_fraction: float = 0.18
    sst_paradoxical_pv_fraction: float = 0.11


@dataclass
class StatisticalResults:
    """Statistical test results with effect sizes and confidence intervals"""
    
    mean_control: float
    mean_treatment: float
    std_control: float
    std_treatment: float
    cohens_d: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    statistical_power: float
    interpretation: str


# ============================================================================
# Module-level worker function for multiprocessing
# ============================================================================

def _worker_run_single_trial(trial_args: Tuple) -> Tuple[int, str, Optional[Dict]]:
    """
    Module-level worker function for parallel trial execution
    
    This must be at module level (not a class method) for proper multiprocessing
    pickling. Recreates the experiment from scratch in each worker process.
    
    Args:
        trial_args: Tuple of all necessary parameters
        
    Returns:
        Tuple of (trial_idx, condition_name, trial_result or None)
    """
    (trial_idx, target_population, light_intensity, condition_name,
     condition_params, trial_seed, mec_current, opsin_current,
     duration, stim_start, circuit_params_dict, synaptic_params_dict, 
     opsin_params_dict, optimization_json_file) = trial_args
    
    try:
        # Force CPU usage and prevent nested parallelism
        import torch
        torch.set_num_threads(1)
        
        # Set random seed for this trial
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        
        # Recreate circuit parameters from dicts
        circuit_params = CircuitParams()
        
        # Recreate base synaptic parameters
        base_synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=synaptic_params_dict['ampa_g_mean'],
            ampa_g_std=synaptic_params_dict['ampa_g_std'],
            ampa_g_min=synaptic_params_dict['ampa_g_min'],
            ampa_g_max=synaptic_params_dict['ampa_g_max'],
            gaba_g_mean=synaptic_params_dict['gaba_g_mean'],
            gaba_g_std=synaptic_params_dict['gaba_g_std'],
            gaba_g_min=synaptic_params_dict['gaba_g_min'],
            gaba_g_max=synaptic_params_dict['gaba_g_max'],
            distribution=synaptic_params_dict['distribution'],
            connection_modulation=synaptic_params_dict['connection_modulation'].copy()
        )
        
        # Apply condition modifications to synaptic parameters
        modified_params = PerConnectionSynapticParams(
            ampa_g_mean=base_synaptic_params.ampa_g_mean,
            ampa_g_std=base_synaptic_params.ampa_g_std,
            ampa_g_min=base_synaptic_params.ampa_g_min,
            ampa_g_max=base_synaptic_params.ampa_g_max,
            gaba_g_mean=base_synaptic_params.gaba_g_mean,
            gaba_g_std=base_synaptic_params.gaba_g_std,
            gaba_g_min=base_synaptic_params.gaba_g_min,
            gaba_g_max=base_synaptic_params.gaba_g_max,
            distribution=base_synaptic_params.distribution,
            connection_modulation=base_synaptic_params.connection_modulation.copy()
        )
        
        # Apply condition-specific modifications
        if 'inhibition_scale' in condition_params:
            scale = condition_params['inhibition_scale']
            modified_params.gaba_g_mean *= scale
            modified_params.gaba_g_std *= scale
            modified_params.gaba_g_min *= scale
            modified_params.gaba_g_max *= scale
        
        if 'excitation_scale' in condition_params:
            scale = condition_params['excitation_scale']
            modified_params.ampa_g_mean *= scale
            modified_params.ampa_g_std *= scale
            modified_params.ampa_g_min *= scale
            modified_params.ampa_g_max *= scale
        
        if 'pv_inhibition_scale' in condition_params:
            scale = condition_params['pv_inhibition_scale']
            for conn in ['pv_gc', 'pv_mc', 'pv_pv', 'pv_sst']:
                if conn in modified_params.connection_modulation:
                    modified_params.connection_modulation[conn] *= scale
        
        if 'sst_inhibition_scale' in condition_params:
            scale = condition_params['sst_inhibition_scale']
            for conn in ['sst_gc', 'sst_mc', 'sst_pv', 'sst_sst']:
                if conn in modified_params.connection_modulation:
                    modified_params.connection_modulation[conn] *= scale
        
        if 'connection_modulation' in condition_params:
            for conn, scale in condition_params['connection_modulation'].items():
                if conn in modified_params.connection_modulation:
                    modified_params.connection_modulation[conn] *= scale
        
        # Recreate opsin parameters
        opsin_params = OpsinParams()
        
        # Create experiment
        experiment = OptogeneticExperiment(
            circuit_params,
            modified_params,
            opsin_params,
            base_seed=trial_seed,
            optimization_json_file=optimization_json_file
        )
        
        # Check for no stimulation condition
        if condition_params.get('no_stimulation', False):
            light_intensity = 0.0
            opsin_current = 0.0
        
        # Run simulation
        result = experiment.simulate_stimulation(
            target_population,
            light_intensity,
            duration=duration,
            stim_start=stim_start,
            mec_current=mec_current,
            opsin_current=opsin_current,
            plot_activity=False
        )
        
        # Analyze trial results (inline to avoid method call issues)
        print(f"result keys: {list(result.keys())}")
        time = result['time']
        activity = result['activity_trace_mean']
        opsin_expression = result['opsin_expression_mean']
        
        # Define time windows
        baseline_mask = (time >= 150) & (time < stim_start)
        stim_mask = time >= stim_start
        
        trial_metrics = {
            'opsin_expression_mean': torch.mean(opsin_expression).item(),
            'opsin_expression_fraction': torch.mean((opsin_expression > 0.1).float()).item()
        }
        
        # Helper function for Gini coefficient
        def calculate_gini(rates_tensor):
            rates_np = rates_tensor.detach().cpu().numpy()
            rates_sorted = np.sort(rates_np)
            n = len(rates_sorted)
            if n == 0 or np.sum(rates_sorted) == 0:
                return 0.0
            cumsum = np.cumsum(rates_sorted)
            gini = (2 * np.sum((np.arange(1, n + 1) * rates_sorted))) / (n * np.sum(rates_sorted)) - (n + 1) / n
            return float(gini)
        
        # Analyze each population
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop == target_population:
                # For stimulated population, analyze opsin vs non-opsin cells
                expressing_mask = opsin_expression > 0.2
                non_expressing_mask = opsin_expression <= 0.2
                
                pop_activity = activity[pop]
                baseline_rate = torch.mean(pop_activity[:, baseline_mask], dim=1)
                stim_rate = torch.mean(pop_activity[:, stim_mask], dim=1)
                
                if torch.sum(expressing_mask) > 0:
                    trial_metrics[f'{pop}_opsin_baseline_mean'] = torch.mean(baseline_rate[expressing_mask]).item()
                    trial_metrics[f'{pop}_opsin_stim_mean'] = torch.mean(stim_rate[expressing_mask]).item()
                
                if torch.sum(non_expressing_mask) > 0:
                    non_expr_baseline = baseline_rate[non_expressing_mask]
                    non_expr_stim = stim_rate[non_expressing_mask]
                    rate_change = non_expr_stim - non_expr_baseline
                    
                    trial_metrics[f'{pop}_non_opsin_baseline_mean'] = torch.mean(non_expr_baseline).item()
                    trial_metrics[f'{pop}_non_opsin_stim_mean'] = torch.mean(non_expr_stim).item()
                    trial_metrics[f'{pop}_non_opsin_change_mean'] = torch.mean(rate_change).item()
                
                continue
            
            # For non-stimulated populations
            pop_activity = activity[pop]
            baseline_rate = torch.mean(pop_activity[:, baseline_mask], dim=1)
            stim_rate = torch.mean(pop_activity[:, stim_mask], dim=1)
            rate_change = stim_rate - baseline_rate
            
            baseline_mean = torch.mean(baseline_rate)
            baseline_std = torch.std(baseline_rate)
            
            paradoxically_excited = rate_change > baseline_std
            paradoxically_inhibited = rate_change < -baseline_std
            
            trial_metrics[f'{pop}_paradoxical_fraction'] = torch.mean(paradoxically_excited.float()).item()
            trial_metrics[f'{pop}_inhibited_fraction'] = torch.mean(paradoxically_inhibited.float()).item()
            trial_metrics[f'{pop}_mean_baseline'] = baseline_mean.item()
            trial_metrics[f'{pop}_std_baseline'] = baseline_std.item()
            trial_metrics[f'{pop}_mean_stim'] = torch.mean(stim_rate).item()
            trial_metrics[f'{pop}_mean_rate_change'] = torch.mean(rate_change).item()
            trial_metrics[f'{pop}_std_rate_change'] = torch.std(rate_change).item()
            
            gini_baseline = calculate_gini(baseline_rate)
            gini_stim = calculate_gini(stim_rate)
            trial_metrics[f'{pop}_gini_baseline'] = gini_baseline
            trial_metrics[f'{pop}_gini_stim'] = gini_stim
            trial_metrics[f'{pop}_gini_change'] = gini_stim - gini_baseline
        
        # Network metrics
        trial_metrics['network_total_paradoxical'] = sum(
            trial_metrics.get(f'{pop}_paradoxical_fraction', 0) 
            for pop in ['gc', 'mc', 'pv', 'sst'] 
            if pop != target_population
        )
        
        return (trial_idx, condition_name, trial_metrics)
        
    except Exception as e:
        import traceback
        print(f"    Warning: Trial {trial_idx} in {condition_name} failed: {e}")
        traceback.print_exc()
        return (trial_idx, condition_name, None)


# ============================================================================
# DisinhibitionHypothesisTester Class
# ============================================================================


class DisinhibitionHypothesisTester:
    """
    Statistical testing framework for disinhibition hypothesis
    
    Now supports batch GPU evaluation for efficient parallel execution
    """
    
    def __init__(self, 
                 circuit_params: CircuitParams = None,
                 synaptic_params: PerConnectionSynapticParams = None,
                 opsin_params: OpsinParams = None,
                 n_trials: int = 50,
                 alpha: float = 0.05, 
                 n_bootstrap: int = 1000,
                 random_seed: int = 42,
                 optimization_json_file: str = None):
        
        self.n_trials = n_trials
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.optimization_json_file = optimization_json_file
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Use provided parameters or create defaults
        self.circuit_params = circuit_params if circuit_params else CircuitParams()
        self.synaptic_params = synaptic_params if synaptic_params else PerConnectionSynapticParams()
        self.opsin_params = opsin_params if opsin_params else OpsinParams()
        
        self.benchmarks = ExperimentalBenchmarks()
        
    def create_modified_synaptic_params(self, condition_params: Dict) -> PerConnectionSynapticParams:
        """
        Create modified synaptic parameters based on pharmacological condition
        
        Args:
            condition_params: Dictionary specifying modifications
                - 'inhibition_scale': Scale all GABA conductances
                - 'excitation_scale': Scale all AMPA/NMDA conductances
                - 'pv_inhibition_scale': Scale only PV->X connections
                - 'sst_inhibition_scale': Scale only SST->X connections
                - 'connection_modulation': Modify specific connections
        """
        
        # Start with baseline parameters
        modified_params = PerConnectionSynapticParams(
            ampa_g_mean=self.synaptic_params.ampa_g_mean,
            ampa_g_std=self.synaptic_params.ampa_g_std,
            ampa_g_min=self.synaptic_params.ampa_g_min,
            ampa_g_max=self.synaptic_params.ampa_g_max,
            gaba_g_mean=self.synaptic_params.gaba_g_mean,
            gaba_g_std=self.synaptic_params.gaba_g_std,
            gaba_g_min=self.synaptic_params.gaba_g_min,
            gaba_g_max=self.synaptic_params.gaba_g_max,
            distribution=self.synaptic_params.distribution,
            connection_modulation=self.synaptic_params.connection_modulation.copy()
        )
        
        # Apply global inhibition scaling
        if 'inhibition_scale' in condition_params:
            scale = condition_params['inhibition_scale']
            modified_params.gaba_g_mean *= scale
            modified_params.gaba_g_std *= scale
            modified_params.gaba_g_min *= scale
            modified_params.gaba_g_max *= scale
        
        # Apply global excitation scaling
        if 'excitation_scale' in condition_params:
            scale = condition_params['excitation_scale']
            modified_params.ampa_g_mean *= scale
            modified_params.ampa_g_std *= scale
            modified_params.ampa_g_min *= scale
            modified_params.ampa_g_max *= scale
        
        # Apply PV-specific inhibition scaling
        if 'pv_inhibition_scale' in condition_params:
            scale = condition_params['pv_inhibition_scale']
            for conn in ['pv_gc', 'pv_mc', 'pv_pv', 'pv_sst']:
                if conn in modified_params.connection_modulation:
                    modified_params.connection_modulation[conn] *= scale
        
        # Apply SST-specific inhibition scaling
        if 'sst_inhibition_scale' in condition_params:
            scale = condition_params['sst_inhibition_scale']
            for conn in ['sst_gc', 'sst_mc', 'sst_pv', 'sst_sst']:
                if conn in modified_params.connection_modulation:
                    modified_params.connection_modulation[conn] *= scale
        
        # Apply specific connection modulations
        if 'connection_modulation' in condition_params:
            for conn, scale in condition_params['connection_modulation'].items():
                if conn in modified_params.connection_modulation:
                    modified_params.connection_modulation[conn] *= scale
        
        return modified_params
    
    def run_single_trial(self, 
                        target_population: str,
                        light_intensity: float,
                        condition_params: Dict,
                        trial_seed: int,
                        mec_current: float = 100.0,
                        opsin_current: float = 100.0,
                        duration: float = 1550.0,
                        stim_start: float = 550.0) -> Dict:
        """
        Run a single simulation trial with specified parameters
        
        Args:
            target_population: 'pv' or 'sst'
            light_intensity: Optogenetic stimulation intensity
            condition_params: Pharmacological condition parameters
            trial_seed: Random seed for this trial
            mec_current: MEC drive current (pA)
            opsin_current: Direct opsin activation current (pA)
        
        Returns:
            Dictionary with trial metrics
        """
        
        # Set random seed for this trial
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        
        # Create modified synaptic parameters
        trial_synaptic_params = self.create_modified_synaptic_params(condition_params)

        # Create experiment with modified parameters
        experiment = OptogeneticExperiment(
            self.circuit_params,
            trial_synaptic_params,
            self.opsin_params,
            base_seed=trial_seed,
            optimization_json_file=self.optimization_json_file
        )
        
        # Run simulation
        if condition_params.get('no_stimulation', False):
            light_intensity = 0.0
            opsin_current = 0.0
            
        result = experiment.simulate_stimulation(
            target_population,
            light_intensity,
            duration=duration,
            stim_start=stim_start,
            mec_current=mec_current,
            opsin_current=opsin_current,
            plot_activity=False
        )
        
        # Analyze trial results
        trial_metrics = self._analyze_single_trial(result, target_population, stim_start, duration)
        
        return trial_metrics

    def run_batch_trials_gpu(self,
                            target_population: str,
                            light_intensity: float,
                            condition_params: Dict,
                            batch_size: int,
                            starting_seed: int,
                            mec_current: float = 100.0,
                            opsin_current: float = 100.0,
                            duration: float = 1550.0,
                            stim_start: float = 550.0,
                            device: Optional[torch.device] = None) -> List[Dict]:
        """
        Run multiple trials on GPU (processed sequentially due to connectivity differences)
        
        NOTE: Despite the name, this does NOT batch trials in parallel. Each trial requires
        different connectivity (different seed), and BatchDentateCircuit shares connectivity
        across its batch dimension. The speedup comes from GPU-accelerated operations within
        each trial, not from parallel trial processing.
        
        Each batch element represents a different trial with different connectivity seed.
        Trials are processed sequentially to allow different connectivity per trial.
        
        Args:
            target_population: 'pv' or 'sst'
            light_intensity: Optogenetic stimulation intensity
            condition_params: Pharmacological condition parameters
            batch_size: Number of trials to process (processed sequentially, not batched)
            starting_seed: Starting seed for this batch
            mec_current: MEC drive current (pA)
            opsin_current: Direct opsin activation current (pA)
            duration: Total simulation duration (ms)
            stim_start: When to start stimulation (ms)
            device: Device to use (default: auto-detect)
            
        Returns:
            List of trial metric dictionaries
        """
        
        if device is None:
            device = get_default_device()
        
        # Check if no stimulation condition
        if condition_params.get('no_stimulation', False):
            light_intensity = 0.0
            opsin_current = 0.0
        
        # Create modified synaptic parameters for this condition
        trial_synaptic_params = self.create_modified_synaptic_params(condition_params)
        
        # Storage for batch results
        batch_trial_metrics = []
        
        # IMPORTANT: We must create separate circuits for each trial because each needs
        # different connectivity (different seed). The BatchDentateCircuit's batch dimension
        # is designed for different parameters with SAME connectivity, not different
        # connectivity realizations.
        #
        # GPU speedup comes from fast matrix operations within each trial, not from
        # parallel processing of trials.
        for trial_idx in range(batch_size):
            trial_seed = starting_seed + trial_idx
            
            # Set seed for this trial - this affects connectivity generation
            torch.manual_seed(trial_seed)
            np.random.seed(trial_seed)
            
            # Create circuit with batch_size=1 for this single trial
            # This uses GPU for efficient time evolution
            circuit = BatchDentateCircuit(
                batch_size=1,
                circuit_params=self.circuit_params,
                synaptic_params=trial_synaptic_params,
                opsin_params=self.opsin_params,
                device=device
            )
            
            # No connection modulation variation within this trial
            circuit.set_connection_modulation_batch([trial_synaptic_params.connection_modulation])
            
            # Setup opsin expression for target population
            n_target_cells = getattr(self.circuit_params, f'n_{target_population}')
            opsin_expression = OpsinExpression(
                self.opsin_params,
                n_cells=n_target_cells,
                device=device
            )
            
            # Get positions for target population
            target_positions = circuit.layout.positions[target_population]
            
            # Calculate activation probability for target population neurons
            activation_prob = opsin_expression.calculate_activation(target_positions, light_intensity)
            
            # Convert to optogenetic current
            target_opto_current = activation_prob * opsin_current
            
            # Storage for time series
            time_points = []
            activities_time_series = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
            
            # Run simulation
            circuit.reset_state()
            mec_input = torch.ones(1, self.circuit_params.n_mec, device=device) * mec_current
            
            for t in range(int(duration)):
                # Create per-population optogenetic drives
                direct_activation = {}
                
                if t >= stim_start:
                    # Apply optogenetic stimulation
                    for pop, n_neurons in [('gc', self.circuit_params.n_gc),
                                           ('mc', self.circuit_params.n_mc),
                                           ('pv', self.circuit_params.n_pv),
                                           ('sst', self.circuit_params.n_sst)]:
                        if pop == target_population:
                            # Target population gets optogenetic drive
                            direct_activation[pop] = target_opto_current.unsqueeze(0)  # [1, n_neurons]
                        else:
                            # Non-target populations get zero drive
                            direct_activation[pop] = torch.zeros(1, n_neurons, device=device)
                else:
                    # No stimulation before stim_start
                    for pop, n_neurons in [('gc', self.circuit_params.n_gc),
                                           ('mc', self.circuit_params.n_mc),
                                           ('pv', self.circuit_params.n_pv),
                                           ('sst', self.circuit_params.n_sst)]:
                        direct_activation[pop] = torch.zeros(1, n_neurons, device=device)
                
                external_drive = {'mec': mec_input}
                activities = circuit(direct_activation, external_drive)
                
                time_points.append(t)
                for pop in activities_time_series:
                    if pop in activities:
                        # Store activity: [1, n_neurons] -> [n_neurons]
                        activities_time_series[pop].append(activities[pop].squeeze(0))
            
            # Convert to tensors: [time, n_neurons] -> transpose to [n_neurons, time]
            for pop in activities_time_series:
                if len(activities_time_series[pop]) > 0:
                    # Stack to [time, n_neurons] then transpose to [n_neurons, time]
                    activities_time_series[pop] = torch.stack(activities_time_series[pop], dim=0).T
            
            # Create result dict similar to OptogeneticExperiment output
            result = {
                'time': torch.tensor(time_points, device=device),
                'activity_trace': activities_time_series,
                'opsin_expression': activation_prob  # [n_neurons]
            }
            
            # Analyze this trial
            trial_metrics = self._analyze_single_trial(result, target_population, stim_start, duration)
            batch_trial_metrics.append(trial_metrics)
            
            # Clean up
            del circuit
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return batch_trial_metrics

    def _analyze_single_trial(self, 
                             simulation_result: Dict,
                             target_population: str,
                             stim_start: float,
                             duration: float) -> Dict:
        """Analyze results from a single simulation trial"""
        
        time = simulation_result['time']
        activity = simulation_result['activity_trace']
        opsin_expression = simulation_result['opsin_expression']
        
        # Define time windows
        baseline_mask = (time >= 150) & (time < stim_start)
        stim_mask = time >= stim_start
        
        trial_metrics = {
            'opsin_expression_mean': torch.mean(opsin_expression).item(),
            'opsin_expression_fraction': torch.mean((opsin_expression > 0.1).float()).item()
        }
        
        # Analyze each population
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop == target_population:
                # For stimulated population, analyze opsin vs non-opsin cells
                expressing_mask = opsin_expression > 0.2
                non_expressing_mask = opsin_expression <= 0.2
                
                pop_activity = activity[pop]
                baseline_rate = torch.mean(pop_activity[:, baseline_mask], dim=1)
                stim_rate = torch.mean(pop_activity[:, stim_mask], dim=1)
                
                # Opsin-expressing cells
                if torch.sum(expressing_mask) > 0:
                    trial_metrics[f'{pop}_opsin_baseline_mean'] = torch.mean(baseline_rate[expressing_mask]).item()
                    trial_metrics[f'{pop}_opsin_stim_mean'] = torch.mean(stim_rate[expressing_mask]).item()
                
                # Non-expressing cells (for paradoxical excitation analysis)
                if torch.sum(non_expressing_mask) > 0:
                    non_expr_baseline = baseline_rate[non_expressing_mask]
                    non_expr_stim = stim_rate[non_expressing_mask]
                    rate_change = non_expr_stim - non_expr_baseline
                    
                    trial_metrics[f'{pop}_non_opsin_baseline_mean'] = torch.mean(non_expr_baseline).item()
                    trial_metrics[f'{pop}_non_opsin_stim_mean'] = torch.mean(non_expr_stim).item()
                    trial_metrics[f'{pop}_non_opsin_change_mean'] = torch.mean(rate_change).item()
                
                continue
                
            # For non-stimulated populations, analyze paradoxical effects
            pop_activity = activity[pop]
            baseline_rate = torch.mean(pop_activity[:, baseline_mask], dim=1)
            stim_rate = torch.mean(pop_activity[:, stim_mask], dim=1)
            rate_change = stim_rate - baseline_rate
            
            # Define paradoxical excitation threshold (1 SD above baseline mean)
            baseline_mean = torch.mean(baseline_rate)
            baseline_std = torch.std(baseline_rate)
            excitation_threshold = baseline_mean + baseline_std
            
            paradoxically_excited = rate_change > baseline_std
            paradoxically_inhibited = rate_change < -baseline_std
            
            total_cells = len(stim_rate)
            
            # Calculate metrics
            trial_metrics[f'{pop}_paradoxical_fraction'] = torch.mean(paradoxically_excited.float()).item()
            trial_metrics[f'{pop}_inhibited_fraction'] = torch.mean(paradoxically_inhibited.float()).item()
            trial_metrics[f'{pop}_mean_baseline'] = baseline_mean.item()
            trial_metrics[f'{pop}_std_baseline'] = baseline_std.item()
            trial_metrics[f'{pop}_mean_stim'] = torch.mean(stim_rate).item()
            trial_metrics[f'{pop}_mean_rate_change'] = torch.mean(rate_change).item()
            trial_metrics[f'{pop}_std_rate_change'] = torch.std(rate_change).item()
            
            # Calculate firing rate inequality (Gini coefficient)
            gini_baseline = self._calculate_gini_coefficient(baseline_rate)
            gini_stim = self._calculate_gini_coefficient(stim_rate)
            trial_metrics[f'{pop}_gini_baseline'] = gini_baseline
            trial_metrics[f'{pop}_gini_stim'] = gini_stim
            trial_metrics[f'{pop}_gini_change'] = gini_stim - gini_baseline
        
        # Overall network metrics
        trial_metrics['network_total_paradoxical'] = sum(
            trial_metrics.get(f'{pop}_paradoxical_fraction', 0) 
            for pop in ['gc', 'mc', 'pv', 'sst'] 
            if pop != target_population
        )
        
        return trial_metrics
    
    def _calculate_gini_coefficient(self, rates: torch.Tensor) -> float:
        """Calculate Gini coefficient for firing rate inequality"""
        rates_np = rates.detach().cpu().numpy()
        rates_sorted = np.sort(rates_np)
        n = len(rates_sorted)
        
        if n == 0 or np.sum(rates_sorted) == 0:
            return 0.0
        
        cumsum = np.cumsum(rates_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1) * rates_sorted))) / (n * np.sum(rates_sorted)) - (n + 1) / n
        return float(gini)
    
    def monte_carlo_analysis(self,
                             target_population: str,
                             light_intensity: float = 1.0,
                             mec_current: float = 100.0,
                             opsin_current: float = 100.0,
                             n_workers: Optional[int] = 1,
                             use_multiprocessing: bool = True,
                             device: Optional[torch.device] = None,
                             gpu_batch_size: int = 8) -> Dict:
        """
        Run Monte Carlo analysis with multiple independent trials
        
        Automatically selects best evaluation strategy:
        - GPU: Sequential trial processing on GPU (fast matrix ops per trial)
        - CPU with multiprocessing: Parallel process pool across trials
        - CPU sequential: Simple loop
        
        NOTE ON GPU BATCHING:
        The gpu_batch_size parameter controls how many trials are processed together
        before cleaning up GPU memory, NOT parallel batch processing. Each trial
        requires different connectivity (different random seed), and BatchDentateCircuit
        shares connectivity across its batch dimension. Therefore, trials must be
        processed sequentially.
        
        GPU speedup comes from:
        1. Fast GPU matrix operations within each trial's time evolution
        2. NOT from parallel processing of multiple trials
        
        For true parallel trial processing, use CPU multiprocessing instead.
        
        Args:
            target_population: 'pv' or 'sst'
            light_intensity: Optogenetic stimulation intensity
            mec_current: MEC drive current (pA)
            opsin_current: Direct opsin activation current (pA)
            n_workers: Number of parallel processes (for CPU multiprocessing)
            use_multiprocessing: Whether to use parallel processing (CPU only)
            device: Device to use (None = auto-detect)
            gpu_batch_size: Number of trials per memory cleanup cycle (NOT parallel batch size)
        
        Returns:
            Dictionary with raw results and statistical analysis
        """
        
        # Device selection
        if device is None:
            device = get_default_device()
        
        use_gpu = device.type == 'cuda'
        
        print(f"\nRunning Monte Carlo analysis: {self.n_trials} trials for {target_population.upper()} stimulation")
        if use_gpu:
            print(f"Using GPU for fast matrix operations (trials processed sequentially)")
            print(f"  Memory cleanup every {gpu_batch_size} trials")
            print(f"  Device: {torch.cuda.get_device_name(device)}")
            print(f"  Note: Trials require different connectivity (different seeds),")
            print(f"        so cannot be batched in parallel. Use CPU multiprocessing")
            print(f"        for true parallel trial processing.")
        elif use_multiprocessing:
            if n_workers is None:
                n_workers = max(1, mp.cpu_count() - 1)
            print(f"Using CPU multiprocessing with {n_workers} workers")
        else:
            print("Using sequential CPU evaluation")
        print("=" * 70)
        
        # Define experimental conditions
        conditions = {
            
            # Control condition
            'full_network': {
                'inhibition_scale': 1.0,
                'excitation_scale': 1.0,
            'description': 'Full network (control)'
            },
            
            # TEST 1: Block excitation TO interneurons only
            'block_exc_to_interneurons': {
                'connection_modulation': {
                    # Minimal excitatory inputs to PV
                    'mec_pv': 0.01,
                    'gc_pv': 0.01,
                    'mc_pv': 0.01,
                    
                    # Minimal excitatory inputs to SST
                    'gc_sst': 0.01,
                    'mc_sst': 0.01,
                    # Keep excitation to principal cells intact
                },
                'description': 'Block excitation to interneurons only (clean disinhibition test)'
            },
            
            # TEST 2: Block interneuron -> interneuron connections
            'block_int_to_int': {
                'connection_modulation': {
                    'pv_sst': 0.01,
                    'pv_pv': 0.01,
                    'sst_pv': 0.01,
                    'sst_sst': 0.01,
                },
                'description': 'Block interneuron-interneuron connections'
            },
            
            # TEST 3: Block MC -> interneuron specifically
            'block_mc_to_interneurons': {
                'connection_modulation': {
                    'mc_pv': 0.01,
                    'mc_sst': 0.01,
                },
                'description': 'Block MC excitation to interneurons'
            },
            
            # TEST 4: Original broad CNQX/APV
            'cnqx_apv_broad': {
                'excitation_scale': 0.1,
                'description': 'Broad glutamate blockade'
            },
            
            # TEST 5: Block principal -> principal excitation only
            'block_principal_recurrent': {
                'connection_modulation': {
                    'gc_mc': 0.01,
                    'mc_gc': 0.01,
                    'mc_mc': 0.01,
                },
                'description': 'Block recurrent excitation among principal cells'
            },
            
            # TEST 6: Gabazine (PV-specific GABA blockade)
            'gabazine': {
                'pv_inhibition_scale': 0.1,
                'description': 'Gabazine (90% blocked GABA-A/PV)'
            },
            
            # TEST 7: No optogenetic stimulation (negative control)
            'no_stimulation': {
                'no_stimulation': True,
                'description': 'No optogenetic stimulation'
            }
        }
        
        # Storage for results
        results = {condition: [] for condition in conditions.keys()}

        # GPU BATCH EVALUATION
        if use_gpu:
            print("\nRunning GPU batch evaluation...")
            
            for condition, params in conditions.items():
                print(f"\nCondition: {params['description']}")
                
                # Run trials in batches
                n_batches = (self.n_trials + gpu_batch_size - 1) // gpu_batch_size
                
                for batch_idx in tqdm.tqdm(range(n_batches), desc=f"  {condition}"):
                    # Calculate batch size for this batch
                    start_trial = batch_idx * gpu_batch_size
                    end_trial = min(start_trial + gpu_batch_size, self.n_trials)
                    current_batch_size = end_trial - start_trial
                    
                    # Calculate starting seed for this batch
                    starting_seed = self.random_seed + start_trial * 1000 + hash(condition) % 1000
                    
                    try:
                        # Run batch of trials on GPU
                        batch_results = self.run_batch_trials_gpu(
                            target_population,
                            light_intensity,
                            params,
                            current_batch_size,
                            starting_seed,
                            mec_current=mec_current,
                            opsin_current=opsin_current,
                            duration=1550.0,
                            stim_start=550.0,
                            device=device
                        )
                        
                        results[condition].extend(batch_results)
                        
                    except Exception as e:
                        print(f"    Warning: Batch {batch_idx} failed: {e}")
                        continue
            
        # CPU MULTIPROCESSING EVALUATION
        elif use_multiprocessing:
            # Parallel execution using module-level worker function
            
            # Prepare parameter dicts for serialization
            circuit_params_dict = {}  # CircuitParams uses defaults
            
            synaptic_params_dict = {
                'ampa_g_mean': self.synaptic_params.ampa_g_mean,
                'ampa_g_std': self.synaptic_params.ampa_g_std,
                'ampa_g_min': self.synaptic_params.ampa_g_min,
                'ampa_g_max': self.synaptic_params.ampa_g_max,
                'gaba_g_mean': self.synaptic_params.gaba_g_mean,
                'gaba_g_std': self.synaptic_params.gaba_g_std,
                'gaba_g_min': self.synaptic_params.gaba_g_min,
                'gaba_g_max': self.synaptic_params.gaba_g_max,
                'distribution': self.synaptic_params.distribution,
                'connection_modulation': self.synaptic_params.connection_modulation.copy()
            }
            
            opsin_params_dict = {}  # OpsinParams uses defaults
            
            all_trial_args = []
            for condition, params in conditions.items():
                for trial in range(self.n_trials):
                    trial_seed = self.random_seed + trial * 1000 + hash(condition) % 1000
                    trial_args = (
                        trial,
                        target_population,
                        light_intensity,
                        condition,
                        params,
                        trial_seed,
                        mec_current,
                        opsin_current,
                        1550.0,  # duration
                        550.0,   # stim_start
                        circuit_params_dict,
                        synaptic_params_dict,
                        opsin_params_dict,
                        self.optimization_json_file
                    )
                    all_trial_args.append(trial_args)
            
            # Run trials in parallel using module-level worker
            print(f"\nRunning {len(all_trial_args)} total trials across {len(conditions)} conditions...")
            
            with mp.Pool(processes=n_workers) as pool:
                # Use imap_unordered for progress tracking
                trial_results = list(tqdm.tqdm(
                    pool.imap_unordered(_worker_run_single_trial, all_trial_args),
                    total=len(all_trial_args),
                    desc="Processing trials"
                ))
            
            # Organize results by condition
            for trial_idx, condition, trial_result in trial_results:
                if trial_result is not None:
                    results[condition].append(trial_result)
        
        # CPU SEQUENTIAL EVALUATION
        else:
            # Serial execution
            for condition, params in conditions.items():
                print(f"\nCondition: {params['description']}")

                for trial in tqdm.tqdm(range(self.n_trials), desc=f"  {condition}"):
                    # Unique seed for each trial and condition
                    trial_seed = self.random_seed + trial * 1000 + hash(condition) % 1000

                    try:
                        trial_result = self.run_single_trial(
                            target_population,
                            light_intensity,
                            params,
                            trial_seed,
                            mec_current=mec_current,
                            opsin_current=opsin_current
                        )
                        results[condition].append(trial_result)

                    except Exception as e:
                        print(f"    Warning: Trial {trial} failed: {e}")
                        continue

        # Print summary of successful trials
        print("\nTrial completion summary:")
        for condition in conditions.keys():
            success_rate = len(results[condition]) / self.n_trials * 100
            print(f"  {condition}: {len(results[condition])}/{self.n_trials} trials ({success_rate:.1f}%)")

        # Statistical analysis
        print("\nPerforming statistical analysis...")
        statistical_analysis = self._analyze_monte_carlo_results(results, target_population)
        
        return {
            'raw_results': results,
            'statistical_analysis': statistical_analysis,
            'experimental_conditions': conditions,
            'target_population': target_population
        }
    
    def _analyze_monte_carlo_results(self, results: Dict, target_population: str) -> Dict:
        """Perform comprehensive statistical analysis on Monte Carlo results"""
        
        analysis = {}
        
        # Define metrics to analyze
        populations = [p for p in ['gc', 'mc', 'pv', 'sst'] if p != target_population]
        
        metrics_to_analyze = []
        for pop in populations:
            metrics_to_analyze.extend([
                f'{pop}_paradoxical_fraction',
                f'{pop}_mean_rate_change',
                f'{pop}_gini_change'
            ])
        metrics_to_analyze.append('network_total_paradoxical')
        
        # Primary comparison: full network vs blocking exc. input to inh. INs (tests disinhibition hypothesis)
        print("  Primary hypothesis test: Full network vs Block Exc->Int...")
        analysis['primary_disinhibition_test'] = {}
        
        for metric in metrics_to_analyze:
            full_values = [trial.get(metric, np.nan) for trial in results['full_network']]
            blocked_values = [trial.get(metric, np.nan) for trial in results['block_exc_to_interneurons']]
            
            # Remove NaN values
            full_values = [v for v in full_values if not np.isnan(v)]
            blocked_values = [v for v in blocked_values if not np.isnan(v)]
            
            if len(full_values) > 0 and len(blocked_values) > 0:
                stat_result = self._statistical_comparison(
                    full_values, blocked_values, metric
                )
                analysis['primary_disinhibition_test'][metric] = stat_result
        
        # Secondary comparisons
        secondary_tests = [
            ('full_network', 'block_int_to_int', 'Interneuron interaction test'),
            ('full_network', 'block_mc_to_interneurons', 'MC pathway test'),
            ('full_network', 'block_principal_recurrent', 'Recurrent excitation test'),
            ('full_network', 'cnqx_apv_broad', 'Broad blockade (original)'),
            ('full_network', 'no_stimulation', 'Optogenetic control')
        ]
        
        print("  Secondary hypothesis tests...")
        for condition1, condition2, desc in secondary_tests:
            comparison_key = f'{condition1}_vs_{condition2}'
            analysis[comparison_key] = {'description': desc, 'results': {}}
            
            for metric in metrics_to_analyze:
                values1 = [trial.get(metric, np.nan) for trial in results[condition1]]
                values2 = [trial.get(metric, np.nan) for trial in results[condition2]]
                
                values1 = [v for v in values1 if not np.isnan(v)]
                values2 = [v for v in values2 if not np.isnan(v)]
                
                if len(values1) > 0 and len(values2) > 0:
                    stat_result = self._statistical_comparison(values1, values2, metric)
                    analysis[comparison_key]['results'][metric] = stat_result
        
        return analysis
    
    def _statistical_comparison(self, 
                               group1: List[float],
                               group2: List[float], 
                               metric_name: str) -> StatisticalResults:
        """Perform comprehensive statistical comparison between two groups"""
        
        g1, g2 = np.array(group1), np.array(group2)
        
        # Basic statistics
        mean1, mean2 = np.mean(g1), np.mean(g2)
        std1, std2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
        n1, n2 = len(g1), len(g2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Statistical test
        if len(set(g1)) == 1 and len(set(g2)) == 1:
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
        
        # Bootstrap confidence interval for effect size
        ci = self._bootstrap_confidence_interval(g1, g2)
        
        # Statistical power calculation
        power = self._calculate_statistical_power(cohens_d, n1, n2, self.alpha)
        
        # Interpretation
        interpretation = self._interpret_results(cohens_d, p_value, power)
        
        return StatisticalResults(
            mean_control=mean2,
            mean_treatment=mean1,
            std_control=std2,
            std_treatment=std1,
            cohens_d=cohens_d,
            p_value=p_value,
            confidence_interval=ci,
            sample_size=min(n1, n2),
            statistical_power=power,
            interpretation=interpretation
        )
    
    def _bootstrap_confidence_interval(self,
                                      group1: np.ndarray,
                                      group2: np.ndarray,
                                      confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for mean difference"""
        
        bootstrap_differences = []
        
        for _ in range(self.n_bootstrap):
            boot1 = np.random.choice(group1, size=len(group1), replace=True)
            boot2 = np.random.choice(group2, size=len(group2), replace=True)
            diff = np.mean(boot1) - np.mean(boot2)
            bootstrap_differences.append(diff)
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_differences, 100 * alpha / 2)
        upper = np.percentile(bootstrap_differences, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def _calculate_statistical_power(self,
                                    effect_size: float,
                                    n1: int,
                                    n2: int,
                                    alpha: float) -> float:
        """Calculate statistical power for t-test"""
        
        df = n1 + n2 - 2
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        ncp = effect_size * np.sqrt(n_harmonic / 2)
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        
        return power
    
    def _interpret_results(self, cohens_d: float, p_value: float, power: float) -> str:
        """Interpret statistical results"""
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        # Statistical significance
        if p_value < 0.001:
            sig_interp = "highly significant"
        elif p_value < 0.01:
            sig_interp = "significant"
        elif p_value < 0.05:
            sig_interp = "marginally significant"
        else:
            sig_interp = "not significant"
        
        # Power interpretation
        if power >= 0.8:
            power_interp = "adequate"
        elif power >= 0.6:
            power_interp = "marginal"
        else:
            power_interp = "insufficient"
        
        return f"{effect_interp} effect, {sig_interp}, {power_interp} power"
    
    def validate_against_experimental_data(self, mc_results: Dict) -> Dict:
        """Validate model predictions against experimental benchmarks"""
        
        validation_results = {}
        target_pop = mc_results['target_population']
        
        # Get full network results
        full_network_trials = mc_results['raw_results']['full_network']
        
        # Define validation metrics based on target population
        if target_pop == 'pv':
            validation_metrics = {
                'gc_paradoxical_fraction': self.benchmarks.pv_paradoxical_gc_fraction,
                'mc_paradoxical_fraction': self.benchmarks.pv_paradoxical_mc_fraction,
                'sst_paradoxical_fraction': self.benchmarks.pv_paradoxical_sst_fraction,
            }
        else:  # sst
            validation_metrics = {
                'gc_paradoxical_fraction': self.benchmarks.sst_paradoxical_gc_fraction,
                'mc_paradoxical_fraction': self.benchmarks.sst_paradoxical_mc_fraction,
                'pv_paradoxical_fraction': self.benchmarks.sst_paradoxical_pv_fraction,
            }
        
        # Validate each metric
        for metric_name, experimental_value in validation_metrics.items():
            model_values = [trial.get(metric_name, np.nan) for trial in full_network_trials]
            model_values = [v for v in model_values if not np.isnan(v)]
            
            if len(model_values) > 0:
                model_mean = np.mean(model_values)
                model_std = np.std(model_values)
                model_sem = model_std / np.sqrt(len(model_values))
                
                # Calculate relative error
                relative_error = abs(model_mean - experimental_value) / (experimental_value + 1e-6)
                
                # Check if experimental value is within model confidence interval
                ci_lower = model_mean - 1.96 * model_sem
                ci_upper = model_mean + 1.96 * model_sem
                within_ci = ci_lower <= experimental_value <= ci_upper
                
                validation_results[metric_name] = {
                    'model_mean': model_mean,
                    'model_std': model_std,
                    'model_sem': model_sem,
                    'experimental_value': experimental_value,
                    'relative_error': relative_error,
                    'within_confidence_interval': within_ci,
                    'validation_score': 1.0 if relative_error < 0.2 else max(0, 1 - relative_error)
                }
        
        # Overall validation score
        individual_scores = [v['validation_score'] for v in validation_results.values()]
        validation_results['overall_validation_score'] = np.mean(individual_scores) if individual_scores else 0.0
        
        return validation_results
    
    def generate_statistical_report(self,
                                   mc_results: Dict,
                                   validation_results: Dict,
                                   output_file: str = None) -> str:
        """Generate statistical report"""
        
        report = []
        report.append("=" * 80)
        report.append(f"DG disinhibition hypothesis statistical report")
        report.append(f"Target population: {mc_results['target_population'].upper()}")
        report.append("=" * 80)
        
        analysis = mc_results['statistical_analysis']
        
        # Primary hypothesis test
        report.append("\n1. Primary hypothesis: Paradoxical excitation depends on disinhibition")
        report.append("-" * 80)
        report.append("\nComparison: Full network vs Block Exc->Int")
        
        primary = analysis['primary_disinhibition_test']
        for metric, result in primary.items():
            report.append(f"\n{metric}:")
            report.append(f"  Full network:    {result.mean_treatment:.4f} +/- {result.std_treatment:.4f}")
            report.append(f"  Block Exc->Int:  {result.mean_control:.4f} +/- {result.std_control:.4f}")
            report.append(f"  Effect size (d): {result.cohens_d:.3f}")
            report.append(f"  P-value:         {result.p_value:.6f}")
            report.append(f"  95% CI:          [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            report.append(f"  Interpretation:  {result.interpretation}")
        
        # Model validation
        report.append("\n\n2. Model validation against experimental data")
        report.append("-" * 80)
        
        if validation_results:
            overall_score = validation_results.get('overall_validation_score', 0.0)
            report.append(f"\nOverall validation score: {overall_score:.3f}")
            
            for metric, results in validation_results.items():
                if metric != 'overall_validation_score':
                    report.append(f"\n{metric}:")
                    report.append(f"  Model:        {results['model_mean']:.4f} +/- {results['model_sem']:.4f}")
                    report.append(f"  Experimental: {results['experimental_value']:.4f}")
                    report.append(f"  Relative err: {results['relative_error']:.1%}")
                    report.append(f"  Within 95% CI: {results['within_confidence_interval']}")
        
        # Statistical conclusions
        report.append("\n\n3. Statistical conclusions")
        report.append("-" * 80)
        
        # Assess primary hypothesis
        network_paradoxical = primary.get('network_total_paradoxical')
        if network_paradoxical:
            if network_paradoxical.p_value < 0.001 and abs(network_paradoxical.cohens_d) > 0.5:
                report.append("\nStrong evidence for disinhibition hypothesis:")
                report.append("  * Highly significant reduction in paradoxical excitation with Block Exc->Int")
                report.append(f"  * Large effect size (d={network_paradoxical.cohens_d:.2f}) indicates biological relevance")
                report.append(f"  * Statistical power: {network_paradoxical.statistical_power:.2f}")
            elif network_paradoxical.p_value < 0.05:
                report.append("\nModerate evidence for disinhibition hypothesis:")
                report.append("  * Significant reduction with Block Exc->Int")
                report.append(f"  * Effect size: {network_paradoxical.cohens_d:.2f}")
            else:
                report.append("\nInsufficient evidence for disinhibition hypothesis:")
                report.append("  * Statistical significance not achieved")
        
        # Model validation assessment
        if validation_results and validation_results.get('overall_validation_score', 0) > 0.7:
            report.append("\nModel validation: good match to experimental data")
            report.append("  * Model predictions closely match experimental observations")
        elif validation_results:
            report.append("\nModel validation: needs improvement")
            report.append("  * Significant discrepancies with experimental data")
            report.append("  * Model parameters may need adjustment")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")
        
        return report_text

    def _convert_to_json_serializable(self, obj):
        """
        Recursively convert objects to JSON-serializable types
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None:
            return None
        elif isinstance(obj, str):
            return obj
        else:
            # For any other type, try to convert to string as fallback
            try:
                return str(obj)
            except:
                return None
    
    def save_results(self, mc_results: Dict, validation_results: Dict, output_dir: str = "."):
        """Save all results to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        target_pop = mc_results['target_population']
        
        # Save raw results
        with open(output_path / f"mc_results_{target_pop}.pkl", 'wb') as f:
            pickle.dump(mc_results, f)
        
        # Save validation results
        with open(output_path / f"validation_{target_pop}.json", 'w') as f:
            # Convert numpy types to native Python types for JSON
            validation_json = self._convert_to_json_serializable(validation_results)
            json.dump(validation_json, f, indent=2)
        
        # Save summary statistics
        summary = self._create_summary_dataframe(mc_results)
        summary.to_csv(output_path / f"summary_{target_pop}.csv", index=False)
        
        print(f"\nResults saved to: {output_path}")
    
    def _create_summary_dataframe(self, mc_results: Dict) -> pd.DataFrame:
        """Create summary DataFrame from Monte Carlo results"""
        
        summary_data = []
        
        for condition, trials in mc_results['raw_results'].items():
            if not trials:
                continue
            
            # Get all metric names from first trial
            metrics = list(trials[0].keys())
            
            for metric in metrics:
                values = [trial.get(metric, np.nan) for trial in trials]
                values = [v for v in values if not np.isnan(v)]
                
                if len(values) > 0:
                    summary_data.append({
                        'condition': condition,
                        'metric': metric,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'sem': np.std(values) / np.sqrt(len(values)),
                        'n': len(values)
                    })
        
        return pd.DataFrame(summary_data)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_statistical_validity_summary(pv_results: Dict, 
                                      sst_results: Dict,
                                      output_dir: str = ".",
                                      show_plots: bool = True):
    """
    Create comprehensive statistical validity plot comparing PV and SST stimulation effects
    
    This function creates a publication-quality figure showing:
    1. Paradoxical excitation comparison (PV vs SST)
    2. Firing rate inequality changes (Gini coefficients)
    3. Statistical validity metrics (effect sizes, p-values, power)
    4. Comparison to experimental benchmarks
    5. Multi-trial robustness
    
    Args:
        pv_results: Results dict from monte_carlo_analysis() for PV stimulation
        sst_results: Results dict from monte_carlo_analysis() for SST stimulation
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    
    # Create figure with 3x3 grid
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Define colors
    colors = {
        'pv': '#E69F00',  # Orange
        'sst': '#56B4E9',  # Sky blue
        'gc': '#009E73',   # Green
        'mc': '#F0E442',   # Yellow
    }
    
    populations = ['gc', 'mc']
    
    # ============================================================================
    # PANEL A: Paradoxical Excitation - PV vs SST Comparison (Top Left)
    # ============================================================================
    ax_para = fig.add_subplot(gs[0, 0])
    plot_paradoxical_comparison(ax_para, pv_results, sst_results, populations, colors)
    ax_para.text(-0.15, 1.05, 'A', transform=ax_para.transAxes, 
                fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL B: Gini Coefficient Changes - PV vs SST (Top Middle)
    # ============================================================================
    ax_gini = fig.add_subplot(gs[0, 1])
    plot_gini_comparison(ax_gini, pv_results, sst_results, populations, colors)
    ax_gini.text(-0.15, 1.05, 'B', transform=ax_gini.transAxes,
                fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL C: Effect Sizes (Cohen's d) Heatmap (Top Right)
    # ============================================================================
    ax_effect = fig.add_subplot(gs[0, 2])
    plot_effect_size_heatmap(ax_effect, pv_results, sst_results, populations)
    ax_effect.text(-0.15, 1.05, 'C', transform=ax_effect.transAxes,
                  fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL D: Statistical Power Analysis (Middle Left)
    # ============================================================================
    ax_power = fig.add_subplot(gs[1, 0])
    plot_power_comparison(ax_power, pv_results, sst_results, populations)
    ax_power.text(-0.15, 1.05, 'D', transform=ax_power.transAxes,
                 fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL E: P-value Summary (Middle Center)
    # ============================================================================
    ax_pval = fig.add_subplot(gs[1, 1])
    plot_pvalue_comparison(ax_pval, pv_results, sst_results, populations)
    ax_pval.text(-0.15, 1.05, 'E', transform=ax_pval.transAxes,
                fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL F: Model Validation vs Experimental Data (Middle Right)
    # ============================================================================
    ax_valid = fig.add_subplot(gs[1, 2])
    plot_validation_comparison(ax_valid, pv_results, sst_results)
    ax_valid.text(-0.15, 1.05, 'F', transform=ax_valid.transAxes,
                 fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL G: Distribution Comparison - GC (Bottom Left)
    # ============================================================================
    ax_gc = fig.add_subplot(gs[2, 0])
    plot_population_distributions(ax_gc, pv_results, sst_results, 'gc', colors)
    ax_gc.text(-0.15, 1.05, 'G', transform=ax_gc.transAxes,
              fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL H: Distribution Comparison - MC (Bottom Middle)
    # ============================================================================
    ax_mc = fig.add_subplot(gs[2, 1])
    plot_population_distributions(ax_mc, pv_results, sst_results, 'mc', colors)
    ax_mc.text(-0.15, 1.05, 'H', transform=ax_mc.transAxes,
              fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # PANEL I: Overall Summary Statistics (Bottom Right)
    # ============================================================================
    ax_summary = fig.add_subplot(gs[2, 2])
    plot_summary_statistics(ax_summary, pv_results, sst_results)
    ax_summary.text(-0.15, 1.05, 'I', transform=ax_summary.transAxes,
                   fontsize=20, fontweight='bold', va='top')
    
    # Main title
    n_trials = len(pv_results['raw_results']['full_network'])
    fig.suptitle(f'Statistical Validity: Paradoxical Excitation and Firing Rate Inequality\n' +
                f'(n={n_trials} trials per condition per stimulation type)',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_file = output_path / "statistical_validity_summary.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    output_file_pdf = output_path / "statistical_validity_summary.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    
    print(f"Saved statistical validity plots: {output_file} and {output_file_pdf}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_disinhibition_test_results(mc_results: Dict,
                                    output_dir: str = ".",
                                    show_plots: bool = True):
    """
    Visualize mechanistic dissection of disinhibition hypothesis
    
    Shows effects of targeted ablations to identify key pathways
    
    Args:
        mc_results: Results from monte_carlo_analysis_improved()
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-paper')
    
    fig = plt.figure(figsize=(16, 12))
    target_pop = mc_results['target_population']
    
    # Define condition order and colors
    condition_order = [
        'full_network',
        'block_exc_to_interneurons',
        'block_int_to_int',
        'block_mc_to_interneurons',
        'block_principal_recurrent',
        'cnqx_apv_broad',
        'no_stimulation'
    ]
    
    condition_colors = {
        'full_network': '#2E86AB',
        'block_exc_to_interneurons': '#DC143C',  # Red - key test
        'block_int_to_int': '#FF8C00',
        'block_mc_to_interneurons': '#9370DB',
        'block_principal_recurrent': '#20B2AA',
        'cnqx_apv_broad': '#808080',
        'gabazine': '#6A994E',
        'no_stimulation': '#D3D3D3'
    }
    
    condition_labels = {
        'full_network': 'Full\nNetwork',
        'block_exc_to_interneurons': 'Block\nExc->Int',
        'block_int_to_int': 'Block\nInt→Int',
        'block_mc_to_interneurons': 'Block\nMC->Int',
        'block_principal_recurrent': 'Block\nRecurrent',
        'cnqx_apv_broad': 'CNQX/APV\nBroad',
        'no_stimulation': 'No\nStim'
    }
    
    # ============================================================================
    # Panel A: Network-wide paradoxical excitation across conditions
    # ============================================================================
    ax1 = plt.subplot(3, 3, 1)
    plot_condition_comparison(ax1, mc_results, condition_order, condition_colors, 
                              condition_labels, 'network_total_paradoxical',
                              'Network-Wide Paradoxical Excitation')
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # Panel B: GC paradoxical fraction
    # ============================================================================
    ax2 = plt.subplot(3, 3, 2)
    plot_condition_comparison(ax2, mc_results, condition_order, condition_colors,
                              condition_labels, 'gc_paradoxical_fraction',
                              'GC Paradoxical Excitation')
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # Panel C: MC paradoxical fraction
    # ============================================================================
    ax3 = plt.subplot(3, 3, 3)
    plot_condition_comparison(ax3, mc_results, condition_order, condition_colors,
                              condition_labels, 'mc_paradoxical_fraction',
                              'MC Paradoxical Excitation')
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # Panel D: Reduction from baseline (effect sizes)
    # ============================================================================
    ax4 = plt.subplot(3, 3, 4)
    plot_reduction_from_baseline(ax4, mc_results, condition_order, condition_colors,
                                 condition_labels)
    ax4.text(-0.15, 1.05, 'D', transform=ax4.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # Panel E: Statistical significance heatmap
    # ============================================================================
    ax5 = plt.subplot(3, 3, 5)
    plot_significance_heatmap(ax5, mc_results, condition_order, condition_labels)
    ax5.text(-0.15, 1.05, 'E', transform=ax5.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    
    # ============================================================================
    # Panel F: Gini coefficient changes
    # ============================================================================
    ax6 = plt.subplot(3, 3, 6)
    plot_condition_comparison(ax6, mc_results, condition_order, condition_colors,
                              condition_labels, 'gc_gini_change',
                              'GC Inequality Change')
    ax6.text(-0.15, 1.05, 'F', transform=ax6.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # Panel G: Pathway diagram
    # ============================================================================
    ax7 = plt.subplot(3, 3, 7)
    plot_pathway_diagram(ax7, target_pop)
    ax7.text(-0.15, 1.05, 'G', transform=ax7.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    # ============================================================================
    # Panel I: Conclusion summary
    # ============================================================================
    ax9 = plt.subplot(3, 3, 9)
    plot_conclusion_summary(ax9, mc_results)
    ax9.text(-0.15, 1.05, 'I', transform=ax9.transAxes,
            fontsize=20, fontweight='bold', va='top')
    
    plt.suptitle(f'Disinhibition Test: {target_pop.upper()} Stimulation\n' +
                'Targeted Ablations to Test Disinhibition Hypothesis',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    # Save figure
    output_file = output_path / f"disinhibition_test_{target_pop}.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    output_file_pdf = output_path / f"disinhibition_test_{target_pop}.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    
    print(f"Saved disinhibition test plots: {output_file}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()




def plot_condition_comparison(ax, mc_results, condition_order, condition_colors,
                              condition_labels, metric, title):
    """Plot comparison across all conditions for a specific metric"""
    
    results = mc_results['raw_results']
    
    means = []
    sems = []
    colors = []
    labels = []
    valid_conditions = []  # Track which conditions have valid data
    
    for condition in condition_order:
        if condition not in results:
            continue
            
        trials = results[condition]
        values = [t.get(metric, np.nan) for t in trials]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            means.append(np.mean(values))
            sems.append(np.std(values) / np.sqrt(len(values)))
            colors.append(condition_colors[condition])
            labels.append(condition_labels[condition])
            valid_conditions.append(condition)  # ✓ Store valid condition name
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.8,
                  capsize=4, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add significance markers vs full network
    # Get full_network data once
    if 'full_network' in results:
        full_values = [t.get(metric, np.nan) for t in results['full_network']]
        full_values = [v for v in full_values if not np.isnan(v)]
    else:
        full_values = []
    
    # iterate through valid_conditions with matching indices
    for i, condition in enumerate(valid_conditions):
        if condition == 'full_network':
            continue
            
        cond_values = [t.get(metric, np.nan) for t in results[condition]]
        cond_values = [v for v in cond_values if not np.isnan(v)]
        
        if len(full_values) > 0 and len(cond_values) > 0:
            t_stat, p_value = stats.ttest_ind(full_values, cond_values)
            
            if p_value < 0.001:
                marker = '***'
            elif p_value < 0.01:
                marker = '**'
            elif p_value < 0.05:
                marker = '*'
            else:
                marker = ''
            
            if marker:
                ax.text(i, means[i] + sems[i] * 1.2, marker,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    

def plot_reduction_from_baseline(ax, mc_results, condition_order, condition_colors,
                                 condition_labels):
    """Plot reduction in paradoxical excitation relative to full network"""
    
    results = mc_results['raw_results']
    
    # Get full network baseline
    if 'full_network' not in results:
        ax.text(0.5, 0.5, 'No full_network data available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Effect Size: Reduction in Paradoxical Excitation',
                    fontsize=11, fontweight='bold')
        return
    
    full_trials = results['full_network']
    full_values = [t.get('network_total_paradoxical', np.nan) for t in full_trials]
    full_values = [v for v in full_values if not np.isnan(v)]
    
    if len(full_values) == 0:
        ax.text(0.5, 0.5, 'No valid full_network data', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Effect Size: Reduction in Paradoxical Excitation',
                    fontsize=11, fontweight='bold')
        return
    
    full_mean = np.mean(full_values)
    
    reductions = []
    reduction_sems = []
    colors = []
    labels = []
    valid_conditions = []  # Track which conditions have valid data
    
    for condition in condition_order:
        if condition == 'full_network' or condition not in results:
            continue
            
        trials = results[condition]
        values = [t.get('network_total_paradoxical', np.nan) for t in trials]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            cond_mean = np.mean(values)
            reduction = (full_mean - cond_mean) / full_mean * 100  # Percent reduction
            
            # Bootstrap SEM for reduction
            bootstrap_reductions = []
            for _ in range(1000):
                boot_full = np.random.choice(full_values, len(full_values), replace=True)
                boot_cond = np.random.choice(values, len(values), replace=True)
                boot_reduction = (np.mean(boot_full) - np.mean(boot_cond)) / np.mean(boot_full) * 100
                bootstrap_reductions.append(boot_reduction)
            
            reductions.append(reduction)
            reduction_sems.append(np.std(bootstrap_reductions))
            colors.append(condition_colors[condition])
            labels.append(condition_labels[condition])
            valid_conditions.append(condition)
    
    if len(reductions) == 0:
        ax.text(0.5, 0.5, 'No valid comparison data', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Effect Size: Reduction in Paradoxical Excitation',
                    fontsize=11, fontweight='bold')
        return
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, reductions, yerr=reduction_sems, color=colors, alpha=0.8,
                  capsize=4, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, reduction in zip(bars, reductions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{reduction:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # Highlight the key test (block_exc_to_interneurons)
    if 'block_exc_to_interneurons' in valid_conditions:
        idx = valid_conditions.index('block_exc_to_interneurons')
        bars[idx].set_linewidth(3)
        bars[idx].set_edgecolor('red')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('% Reduction from\nFull Network', fontsize=10, fontweight='bold')
    ax.set_title('Effect Size: Reduction in Paradoxical Excitation',
                fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)

    
def plot_significance_heatmap(ax, mc_results, condition_order, condition_labels):
    """Create heatmap of p-values for all condition comparisons"""
    
    results = mc_results['raw_results']
    metrics = ['gc_paradoxical_fraction', 'mc_paradoxical_fraction', 
               'network_total_paradoxical']
    
    # Build matrix
    conditions = [c for c in condition_order if c in results and c != 'full_network']
    pvalue_matrix = []
    
    for metric in metrics:
        row = []
        full_values = [t.get(metric, np.nan) for t in results['full_network']]
        full_values = [v for v in full_values if not np.isnan(v)]
        
        for condition in conditions:
            cond_values = [t.get(metric, np.nan) for t in results[condition]]
            cond_values = [v for v in cond_values if not np.isnan(v)]
            
            if len(full_values) > 0 and len(cond_values) > 0:
                t_stat, p_value = stats.ttest_ind(full_values, cond_values)
                row.append(-np.log10(p_value + 1e-10))
            else:
                row.append(0)
        
        pvalue_matrix.append(row)
    
    pvalue_matrix = np.array(pvalue_matrix)
    
    # Create heatmap
    im = ax.imshow(pvalue_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('-log₁₀(p-value)', rotation=270, labelpad=15, fontsize=9)
    
    # Labels
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([condition_labels[c] for c in conditions], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=8)
    
    # Annotations
    for i in range(len(metrics)):
        for j in range(len(conditions)):
            text = ax.text(j, i, f'{pvalue_matrix[i, j]:.1f}',
                          ha="center", va="center", 
                          color="white" if pvalue_matrix[i, j] > 2.5 else "black",
                          fontsize=9)
    
    ax.set_title('Statistical Significance vs Full Network\n(-log₁₀ p-value)',
                     fontsize=11, fontweight='bold')


def plot_pathway_diagram(ax, target_pop):
    """Draw simplified pathway diagram showing tested connections"""
    
    ax.axis('off')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    
    # Draw nodes
    nodes = {
        'MEC': (2, 8),
        'GC': (2, 5),
        'MC': (5, 5),
        'PV': (8, 7),
        'SST': (8, 3)
    }
    
    for name, (x, y) in nodes.items():
        if name == target_pop.upper():
            color = 'red'
            size = 1500
        elif name in ['PV', 'SST']:
            color = 'orange'
            size = 1000
        else:
            color = 'lightblue'
            size = 1000
        
        ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw connections tested
    arrows = [
        ('MEC', 'PV', 'green', 'solid', 2),
        ('GC', 'PV', 'green', 'solid', 2),
        ('MC', 'PV', 'green', 'solid', 2),
        ('GC', 'SST', 'green', 'dashed', 1),
        ('MC', 'SST', 'green', 'solid', 2),
        ('PV', 'SST', 'red', 'solid', 2),
        ('SST', 'PV', 'red', 'dashed', 1),
    ]
    
    for src, tgt, color, style, width in arrows:
        x1, y1 = nodes[src]
        x2, y2 = nodes[tgt]
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        # Shorten to avoid overlap with nodes
        x1 += dx / length * 0.5
        y1 += dy / length * 0.5
        x2 -= dx / length * 0.5
        y2 -= dy / length * 0.5
        
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=width, color=color, 
                                 linestyle=style, alpha=0.6))
    
    ax.set_title('Circuit Connectivity\n(Green=Exc, Red=Inh)', 
                fontsize=11, fontweight='bold')
    
    # Legend
    ax.text(0.5, 0.5, 'Solid = strong\nDashed = weak',
           fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    
def plot_conclusion_summary(ax, mc_results):
    """Display overall disinhibition conclusion"""
    
    ax.axis('off')
    
    # Analyze key comparisons
    results = mc_results['raw_results']
    
    # Full network value
    full_trials = results['full_network']
    full_vals = [t.get('network_total_paradoxical', np.nan) for t in full_trials]
    full_vals = [v for v in full_vals if not np.isnan(v)]
    full_mean = np.mean(full_vals)
    
    # Key test: block_exc_to_interneurons
    if 'block_exc_to_interneurons' in results:
        block_trials = results['block_exc_to_interneurons']
        block_vals = [t.get('network_total_paradoxical', np.nan) for t in block_trials]
        block_vals = [v for v in block_vals if not np.isnan(v)]
        block_mean = np.mean(block_vals)
        
        reduction = (full_mean - block_mean) / full_mean
        t_stat, p_value = stats.ttest_ind(full_vals, block_vals)
        
        summary_text = "Primary test:\n"
        summary_text += "Block Exc -> Interneurons\n\n"
        summary_text += f"Reduction: {reduction:.1%}\n"
        summary_text += f"P-value: {p_value:.2e}\n\n"
        
        if p_value < 0.001 and reduction > 0.5:
            summary_text += "Strong support for\n"
            summary_text += "disinhibition hypothesis.\n\n"
            summary_text += "Paradoxical excitation\n"
            summary_text += "Requires excitation of\n"
            summary_text += "the inhibitory network.\n"
        elif p_value < 0.05 and reduction > 0.3:
            summary_text += "Moderate support for\n"
            summary_text += "disinhibition hypothesis.\n"
        else:
            summary_text += "Weak/no support for\n"
            summary_text += "simple disinhibition.\n"
    
    # Compare to broad blockade
    if 'cnqx_apv_broad' in results:
        broad_trials = results['cnqx_apv_broad']
        broad_vals = [t.get('network_total_paradoxical', np.nan) for t in broad_trials]
        broad_vals = [v for v in broad_vals if not np.isnan(v)]
        broad_mean = np.mean(broad_vals)
        
        broad_reduction = (full_mean - broad_mean) / full_mean
        
        summary_text += f"\nBroad CNQX/APV:\n"
        summary_text += f"Reduction: {broad_reduction:.1%}\n"
        summary_text += "(Confounded by direct\n"
        summary_text += "suppression of principal\n"
        summary_text += "cell excitation)\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
def plot_paradoxical_comparison(ax, pv_results, sst_results, populations, colors):
    """Validate paradoxical excitation: fraction of excited cells"""
    
    # Show fraction of cells exhibiting paradoxical excitation
    # Compare: no stimulation baseline vs full network stimulation
    
    width = 0.25
    labels = []
    nostim_fracs = []
    stim_fracs = []
    nostim_sems = []
    stim_sems = []
    bar_colors = []
    
    for pop in populations:
        # PV stimulation
        # No stimulation control (should be near zero)
        nostim_trials = pv_results['raw_results'].get('no_stimulation', [])
        pv_nostim = []
        for trial in nostim_trials:
            frac = trial.get(f'{pop}_paradoxical_fraction', np.nan)
            if not np.isnan(frac):
                pv_nostim.append(frac)
        
        # Full network with stimulation
        stim_trials = pv_results['raw_results']['full_network']
        pv_stim = []
        for trial in stim_trials:
            frac = trial.get(f'{pop}_paradoxical_fraction', np.nan)
            if not np.isnan(frac):
                pv_stim.append(frac)
        
        if len(pv_stim) > 0:
            nostim_fracs.append(np.mean(pv_nostim) if len(pv_nostim) > 0 else 0)
            stim_fracs.append(np.mean(pv_stim))
            nostim_sems.append(np.std(pv_nostim) / np.sqrt(len(pv_nostim)) if len(pv_nostim) > 0 else 0)
            stim_sems.append(np.std(pv_stim) / np.sqrt(len(pv_stim)))
            labels.append(f'{pop.upper()}\nPV')
            bar_colors.append(colors['pv'])
        
        # SST stimulation
        nostim_trials = sst_results['raw_results'].get('no_stimulation', [])
        sst_nostim = []
        for trial in nostim_trials:
            frac = trial.get(f'{pop}_paradoxical_fraction', np.nan)
            if not np.isnan(frac):
                sst_nostim.append(frac)
        
        stim_trials = sst_results['raw_results']['full_network']
        sst_stim = []
        for trial in stim_trials:
            frac = trial.get(f'{pop}_paradoxical_fraction', np.nan)
            if not np.isnan(frac):
                sst_stim.append(frac)
        
        if len(sst_stim) > 0:
            nostim_fracs.append(np.mean(sst_nostim) if len(sst_nostim) > 0 else 0)
            stim_fracs.append(np.mean(sst_stim))
            nostim_sems.append(np.std(sst_nostim) / np.sqrt(len(sst_nostim)) if len(sst_nostim) > 0 else 0)
            stim_sems.append(np.std(sst_stim) / np.sqrt(len(sst_stim)))
            labels.append(f'{pop.upper()}\nSST')
            bar_colors.append(colors['sst'])
    
    x_pos = np.arange(len(labels))
    
    # Create grouped bars
    bars1 = ax.bar(x_pos - width/2, nostim_fracs, width, yerr=nostim_sems,
                   label='No Stimulation', color='lightgray', alpha=0.7,
                   capsize=4, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, stim_fracs, width, yerr=stim_sems,
                   label='With Stimulation', color=bar_colors, alpha=0.8,
                   capsize=4, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add statistical significance
    for i, label in enumerate(labels):
        # Determine which dataset
        if 'PV' in label:
            pop = label.split('\n')[0].lower()
            nostim_trials = pv_results['raw_results'].get('no_stimulation', [])
            stim_trials = pv_results['raw_results']['full_network']
        else:
            pop = label.split('\n')[0].lower()
            nostim_trials = sst_results['raw_results'].get('no_stimulation', [])
            stim_trials = sst_results['raw_results']['full_network']
        
        nostim_vals = [t.get(f'{pop}_paradoxical_fraction', np.nan) 
                       for t in nostim_trials]
        stim_vals = [t.get(f'{pop}_paradoxical_fraction', np.nan) 
                     for t in stim_trials]
        
        nostim_vals = [v for v in nostim_vals if not np.isnan(v)]
        stim_vals = [v for v in stim_vals if not np.isnan(v)]
        
        if len(stim_vals) > 0:
            # Independent t-test (different conditions)
            if len(nostim_vals) > 0:
                t_stat, p_value = stats.ttest_ind(nostim_vals, stim_vals)
            else:
                # One-sample t-test against zero if no nostim data
                t_stat, p_value = stats.ttest_1samp(stim_vals, 0)
            
            y_max = max(nostim_fracs[i] + nostim_sems[i], 
                       stim_fracs[i] + stim_sems[i])
            y_pos = y_max * 1.15
            
            if p_value < 0.001:
                marker = '***'
            elif p_value < 0.01:
                marker = '**'
            elif p_value < 0.05:
                marker = '*'
            else:
                marker = 'ns'
            
            # Draw significance bracket
            x1 = i - width/2
            x2 = i + width/2
            ax.plot([x1, x1, x2, x2], [y_pos*0.96, y_pos, y_pos, y_pos*0.96], 
                   'k-', linewidth=1.2)
            ax.text(i, y_pos*1.02, marker, ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Population & Stimulation Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fraction of Cells with\nParadoxical Excitation', fontsize=11, fontweight='bold')
    ax.set_title('Validation: Paradoxical Excitation\n(Excited Cell Fraction)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim([0, None])
    ax.legend(fontsize=10, frameon=True, fancybox=True, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)


def plot_gini_comparison(ax, pv_results, sst_results, populations, colors):
    """Validate inequality increase: baseline vs stimulation Gini coefficients"""
    
    width = 0.2
    labels = []
    baseline_ginis = []
    stim_ginis = []
    baseline_sems = []
    stim_sems = []
    bar_colors = []
    
    for pop in populations:
        # PV stimulation
        pv_trials = pv_results['raw_results']['full_network']
        pv_baseline = []
        pv_stim = []
        
        for trial in pv_trials:
            baseline = trial.get(f'{pop}_gini_baseline', np.nan)
            stim = trial.get(f'{pop}_gini_stim', np.nan)
            if not np.isnan(baseline) and not np.isnan(stim):
                pv_baseline.append(baseline)
                pv_stim.append(stim)
        
        if len(pv_baseline) > 0:
            baseline_ginis.append(np.mean(pv_baseline))
            stim_ginis.append(np.mean(pv_stim))
            baseline_sems.append(np.std(pv_baseline) / np.sqrt(len(pv_baseline)))
            stim_sems.append(np.std(pv_stim) / np.sqrt(len(pv_stim)))
            labels.append(f'{pop.upper()}\nPV')
            bar_colors.append(colors['pv'])
        
        # SST stimulation
        sst_trials = sst_results['raw_results']['full_network']
        sst_baseline = []
        sst_stim = []
        
        for trial in sst_trials:
            baseline = trial.get(f'{pop}_gini_baseline', np.nan)
            stim = trial.get(f'{pop}_gini_stim', np.nan)
            if not np.isnan(baseline) and not np.isnan(stim):
                sst_baseline.append(baseline)
                sst_stim.append(stim)
        
        if len(sst_baseline) > 0:
            baseline_ginis.append(np.mean(sst_baseline))
            stim_ginis.append(np.mean(sst_stim))
            baseline_sems.append(np.std(sst_baseline) / np.sqrt(len(sst_baseline)))
            stim_sems.append(np.std(sst_stim) / np.sqrt(len(sst_stim)))
            labels.append(f'{pop.upper()}\nSST')
            bar_colors.append(colors['sst'])
    
    x_pos = np.arange(len(labels))
    
    # Create grouped bars
    bars1 = ax.bar(x_pos - width/2, baseline_ginis, width, yerr=baseline_sems,
                   label='Baseline', color='gray', alpha=0.6,
                   capsize=4, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, stim_ginis, width, yerr=stim_sems,
                   label='Stimulation', color=bar_colors, alpha=0.8,
                   capsize=4, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add statistical significance for paired comparisons
    for i, label in enumerate(labels):
        # Determine which dataset this is
        if 'PV' in label:
            pop = label.split('\n')[0].lower()
            trials = pv_results['raw_results']['full_network']
        else:
            pop = label.split('\n')[0].lower()
            trials = sst_results['raw_results']['full_network']
        
        baseline_vals = []
        stim_vals = []
        for trial in trials:
            baseline = trial.get(f'{pop}_gini_baseline', np.nan)
            stim = trial.get(f'{pop}_gini_stim', np.nan)
            if not np.isnan(baseline) and not np.isnan(stim):
                baseline_vals.append(baseline)
                stim_vals.append(stim)
        
        if len(baseline_vals) > 0:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(baseline_vals, stim_vals)
            
            y_max = max(baseline_ginis[i] + baseline_sems[i], 
                       stim_ginis[i] + stim_sems[i])
            y_pos = y_max * 1.08
            
            if p_value < 0.001:
                marker = '***'
            elif p_value < 0.01:
                marker = '**'
            elif p_value < 0.05:
                marker = '*'
            else:
                marker = 'ns'
            
            # Draw significance bracket
            x1 = i - width/2
            x2 = i + width/2
            ax.plot([x1, x1, x2, x2], [y_pos*0.97, y_pos, y_pos, y_pos*0.97], 
                   'k-', linewidth=1.2)
            ax.text(i, y_pos*1.02, marker, ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Population & Stimulation Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gini Coefficient', fontsize=11, fontweight='bold')
    ax.set_title('Validation: Inequality Increase\n(Baseline vs Stimulation)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, frameon=True, fancybox=True, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)

        

def plot_effect_size_heatmap(ax, pv_results, sst_results, populations):
    """Create heatmap of effect sizes (Cohen's d) for primary hypothesis"""
    
    # Collect effect sizes
    effect_matrix = []
    row_labels = []
    
    for target, results in [('PV', pv_results), ('SST', sst_results)]:
        primary = results['statistical_analysis']['primary_disinhibition_test']
        row_effects = []
        
        for pop in populations:
            metric = f'{pop}_paradoxical_fraction'
            if metric in primary:
                d = primary[metric].cohens_d
                row_effects.append(d)
            else:
                row_effects.append(0)
        
        effect_matrix.append(row_effects)
        row_labels.append(f'{target} Stim')
    
    effect_matrix = np.array(effect_matrix)
    
    # Create heatmap
    im = ax.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', 
                   vmin=-1.5, vmax=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d", rotation=270, labelpad=20, fontsize=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(populations)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([p.upper() for p in populations], fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(populations)):
            text = ax.text(j, i, f'{effect_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11,
                          fontweight='bold')
    
    ax.set_title("Effect Sizes (Cohen's d)\nFull Network vs Disinhibition",
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Population', fontsize=11, fontweight='bold')
    
    # Add reference lines for effect size interpretation
    ax.text(1.15, -0.15, 'Small: |d|<0.5\nMedium: 0.5≤|d|<0.8\nLarge: |d|≥0.8',
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_power_comparison(ax, pv_results, sst_results, populations):
    """Compare statistical power between PV and SST experiments"""
    
    x_pos = np.arange(len(populations))
    width = 0.35
    
    pv_powers = []
    sst_powers = []
    
    for pop in populations:
        # PV power
        pv_primary = pv_results['statistical_analysis']['primary_disinhibition_test']
        metric = f'{pop}_paradoxical_fraction'
        if metric in pv_primary:
            pv_powers.append(pv_primary[metric].statistical_power)
        else:
            pv_powers.append(0)
        
        # SST power
        sst_primary = sst_results['statistical_analysis']['primary_disinhibition_test']
        if metric in sst_primary:
            sst_powers.append(sst_primary[metric].statistical_power)
        else:
            sst_powers.append(0)
    
    # Create bars
    bars1 = ax.bar(x_pos - width/2, pv_powers, width,
                   label='PV Stimulation', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, sst_powers, width,
                   label='SST Stimulation', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    # Color bars by power level
    for bars, powers in [(bars1, pv_powers), (bars2, sst_powers)]:
        for bar, power in zip(bars, powers):
            if power >= 0.8:
                bar.set_facecolor('#2E8B57')  # Green
            elif power >= 0.6:
                bar.set_facecolor('#FFA500')  # Orange
            else:
                bar.set_facecolor('#DC143C')  # Red
    
    # Add value labels
    for bars, powers in [(bars1, pv_powers), (bars2, sst_powers)]:
        for bar, power in zip(bars, powers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{power:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Population', fontsize=11, fontweight='bold')
    ax.set_ylabel('Statistical Power', fontsize=11, fontweight='bold')
    ax.set_title('Statistical Power Analysis\n(Primary Hypothesis Test)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.upper() for p in populations], fontsize=11)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.6, 
              linewidth=2, label='Adequate (0.8)')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.6,
              linewidth=2, label='Marginal (0.6)')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9, frameon=True, fancybox=True, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)


def plot_pvalue_comparison(ax, pv_results, sst_results, populations):
    """Compare p-values between PV and SST experiments"""
    
    # Collect p-values
    pvalue_matrix = []
    row_labels = []
    
    for target, results in [('PV', pv_results), ('SST', sst_results)]:
        primary = results['statistical_analysis']['primary_disinhibition_test']
        row_pvals = []
        
        for pop in populations:
            metric = f'{pop}_paradoxical_fraction'
            if metric in primary:
                p = primary[metric].p_value
                # Convert to -log10 scale for visualization
                row_pvals.append(-np.log10(p + 1e-10))
            else:
                row_pvals.append(0)
        
        pvalue_matrix.append(row_pvals)
        row_labels.append(f'{target} Stim')
    
    pvalue_matrix = np.array(pvalue_matrix)
    
    # Create heatmap
    im = ax.imshow(pvalue_matrix, cmap='Reds', aspect='auto', 
                   vmin=0, vmax=5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('-log10(p-value)', rotation=270, labelpad=20, fontsize=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(populations)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([p.upper() for p in populations], fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    
    # Add text annotations with significance markers
    for i in range(len(row_labels)):
        for j in range(len(populations)):
            # Get actual p-value
            target = 'PV' if i == 0 else 'SST'
            results = pv_results if i == 0 else sst_results
            primary = results['statistical_analysis']['primary_disinhibition_test']
            metric = f'{populations[j]}_paradoxical_fraction'
            
            if metric in primary:
                p = primary[metric].p_value
                
                if p < 0.001:
                    marker = '***'
                elif p < 0.01:
                    marker = '**'
                elif p < 0.05:
                    marker = '*'
                else:
                    marker = 'ns'
                
                color = 'white' if pvalue_matrix[i, j] > 2 else 'black'
                ax.text(j, i, f'{pvalue_matrix[i, j]:.1f}\n{marker}',
                       ha="center", va="center", color=color, fontsize=10,
                       fontweight='bold')
    
    ax.set_title('Statistical Significance\n(Primary Hypothesis Test)',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Population', fontsize=11, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=-0.5, color='black', linewidth=2)
    ax.text(1.15, -0.15, 'p<0.001: ***\np<0.01: **\np<0.05: *\np≥0.05: ns',
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_validation_comparison(ax, pv_results, sst_results):
    """Compare model predictions to experimental benchmarks"""
    
    benchmarks = ExperimentalBenchmarks()
    
    # Define metrics and their experimental values
    metrics_data = [
        ('GC\n(PV stim)', 
         'gc_paradoxical_fraction', 
         benchmarks.pv_paradoxical_gc_fraction,
         pv_results),
        ('MC\n(PV stim)', 
         'mc_paradoxical_fraction',
         benchmarks.pv_paradoxical_mc_fraction,
         pv_results),
        ('GC\n(SST stim)',
         'gc_paradoxical_fraction',
         benchmarks.sst_paradoxical_gc_fraction,
         sst_results),
        ('MC\n(SST stim)',
         'mc_paradoxical_fraction',
         benchmarks.sst_paradoxical_mc_fraction,
         sst_results),
    ]
    
    x_pos = np.arange(len(metrics_data))
    width = 0.35
    
    model_means = []
    model_sems = []
    exp_values = []
    
    for label, metric, exp_val, results in metrics_data:
        trials = results['raw_results']['full_network']
        values = [t.get(metric, np.nan) for t in trials]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            model_means.append(np.mean(values))
            model_sems.append(np.std(values) / np.sqrt(len(values)))
        else:
            model_means.append(0)
            model_sems.append(0)
        
        exp_values.append(exp_val)
    
    # Create bars for model
    bars = ax.bar(x_pos, model_means, width, yerr=model_sems,
                  label='Model', color='#2E86AB', alpha=0.7,
                  capsize=5, edgecolor='black', linewidth=1.5)
    
    # Add experimental data as scatter
    ax.scatter(x_pos, exp_values, color='red', s=150,
              marker='D', label='Experimental', zorder=10,
              edgecolors='black', linewidth=2)
    
    # Add connecting lines showing agreement
    for i, (m, e) in enumerate(zip(model_means, exp_values)):
        ax.plot([i, i], [m, e], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Paradoxical Excitation Fraction', fontsize=11, fontweight='bold')
    ax.set_title('Model Validation vs\nExperimental Benchmarks',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m[0] for m in metrics_data], fontsize=9)
    ax.legend(fontsize=10, frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Calculate and display RMSE
    rmse = np.sqrt(np.mean([(m - e)**2 for m, e in zip(model_means, exp_values)]))
    ax.text(0.98, 0.98, f'RMSE: {rmse:.4f}', transform=ax.transAxes,
           ha='right', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))


def plot_population_distributions(ax, pv_results, sst_results, pop, colors):
    """Plot detailed distributions for a specific population"""
    
    # Get full network data for both conditions
    pv_full = pv_results['raw_results']['full_network']
    pv_cnqx = pv_results['raw_results']['cnqx_apv_broad']
    sst_full = sst_results['raw_results']['full_network']
    sst_cnqx = sst_results['raw_results']['cnqx_apv_broad']
    
    metric = f'{pop}_paradoxical_fraction'
    
    # Extract values
    pv_full_vals = [t.get(metric, np.nan) for t in pv_full]
    pv_cnqx_vals = [t.get(metric, np.nan) for t in pv_cnqx]
    sst_full_vals = [t.get(metric, np.nan) for t in sst_full]
    sst_cnqx_vals = [t.get(metric, np.nan) for t in sst_cnqx]
    
    # Remove NaNs
    pv_full_vals = [v for v in pv_full_vals if not np.isnan(v)]
    pv_cnqx_vals = [v for v in pv_cnqx_vals if not np.isnan(v)]
    sst_full_vals = [v for v in sst_full_vals if not np.isnan(v)]
    sst_cnqx_vals = [v for v in sst_cnqx_vals if not np.isnan(v)]
    
    # Create violin plots
    data = [pv_full_vals, pv_cnqx_vals, sst_full_vals, sst_cnqx_vals]
    positions = [1, 2, 3.5, 4.5]
    plot_colors = [colors['pv'], '#CCCCCC', colors['sst'], '#CCCCCC']
    
    parts = ax.violinplot(data, positions=positions, showmeans=True,
                         showextrema=True, widths=0.7)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plot_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add box plots overlay
    bp = ax.boxplot(data, positions=positions, widths=0.3, showfliers=False,
                   boxprops=dict(linewidth=1.5, color='black'),
                   whiskerprops=dict(linewidth=1.5, color='black'),
                   capprops=dict(linewidth=1.5, color='black'),
                   medianprops=dict(linewidth=2, color='red'))
    
    # Add significance brackets
    # PV: Full vs CNQX
    t1, p1 = stats.ttest_ind(pv_full_vals, pv_cnqx_vals)
    y1 = max(max(pv_full_vals), max(pv_cnqx_vals)) * 1.1
    ax.plot([1, 1, 2, 2], [y1*0.98, y1, y1, y1*0.98], 'k-', linewidth=1.5)
    marker1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else 'ns'
    ax.text(1.5, y1*1.02, marker1, ha='center', va='bottom',
           fontsize=11, fontweight='bold')
    
    # SST: Full vs CNQX
    t2, p2 = stats.ttest_ind(sst_full_vals, sst_cnqx_vals)
    y2 = max(max(sst_full_vals), max(sst_cnqx_vals)) * 1.1
    ax.plot([3.5, 3.5, 4.5, 4.5], [y2*0.98, y2, y2, y2*0.98], 'k-', linewidth=1.5)
    marker2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
    ax.text(4, y2*1.02, marker2, ha='center', va='bottom',
           fontsize=11, fontweight='bold')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['PV\nFull', 'PV\nCNQX', 'SST\nFull', 'SST\nCNQX'],
                      fontsize=9)
    ax.set_ylabel('Paradoxical Excitation\nFraction', fontsize=10, fontweight='bold')
    ax.set_title(f'{pop.upper()} Response Distributions\n(Multi-trial)',
                fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)


def plot_summary_statistics(ax, pv_results, sst_results):
    """Display comprehensive summary statistics"""
    
    ax.axis('off')
    
    pv_primary = pv_results['statistical_analysis']['primary_disinhibition_test']
    sst_primary = sst_results['statistical_analysis']['primary_disinhibition_test']
    
    n_trials = len(pv_results['raw_results']['full_network'])
    
    summary_text = "Summary Statistics\n"
    summary_text += "=" * 45 + "\n\n"
    summary_text += f"Trials per condition: {n_trials}\n"
    summary_text += f"Total simulations: {n_trials * 6 * 2}\n\n"
    
    # PV Summary
    summary_text += "PV stimulation:\n"
    summary_text += "-" * 45 + "\n"
    
    pv_network = pv_primary.get('network_total_paradoxical')
    if pv_network:
        summary_text += f"Network-wide paradoxical excitation:\n"
        summary_text += f"  Effect size (d): {pv_network.cohens_d:+.3f}\n"
        summary_text += f"  P-value: {pv_network.p_value:.2e}\n"
        summary_text += f"  Power: {pv_network.statistical_power:.2f}\n"
        summary_text += f"  Interpretation: {pv_network.interpretation}\n\n"
    
    # SST Summary
    summary_text += "SST stimulation:\n"
    summary_text += "-" * 45 + "\n"
    
    sst_network = sst_primary.get('network_total_paradoxical')
    if sst_network:
        summary_text += f"Network-wide paradoxical excitation:\n"
        summary_text += f"  Effect size (d): {sst_network.cohens_d:+.3f}\n"
        summary_text += f"  P-value: {sst_network.p_value:.2e}\n"
        summary_text += f"  Power: {sst_network.statistical_power:.2f}\n"
        summary_text += f"  Interpretation: {sst_network.interpretation}\n\n"
    
    # Overall Conclusion
    summary_text += "OVERALL CONCLUSION:\n"
    summary_text += "-" * 45 + "\n"
    
    pv_strong = (pv_network and pv_network.p_value < 0.001 and 
                abs(pv_network.cohens_d) > 0.5)
    sst_strong = (sst_network and sst_network.p_value < 0.001 and 
                 abs(sst_network.cohens_d) > 0.5)
    
    if pv_strong and sst_strong:
        summary_text += "strong evidence for disinhibition\n"
        summary_text += "hypothesis in both PV and SST\n"
        summary_text += "stimulation conditions.\n\n"
        summary_text += "Both show:\n"
        summary_text += "- Highly significant effects\n"
        summary_text += "- Large effect sizes\n"
        summary_text += "- Adequate statistical power\n"
    elif pv_strong or sst_strong:
        summary_text += "Moderate evidence for\n"
        summary_text += "disinhibition hypothesis.\n\n"
        if pv_strong:
            summary_text += "Strong support for PV,\n"
            summary_text += "weaker for SST.\n"
        else:
            summary_text += "Strong support for SST,\n"
            summary_text += "weaker for PV.\n"
    else:
        summary_text += "Insufficient evidence for\n"
        summary_text += "strong conclusions about\n"
        summary_text += "disinhibition hypothesis.\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))


        
def plot_monte_carlo_results(mc_results: Dict,
                            validation_results: Dict = None,
                            output_dir: str = ".",
                            show_plots: bool = True):
    """
    Create comprehensive visualizations of Monte Carlo analysis results
    
    Args:
        mc_results: Results from monte_carlo_analysis()
        validation_results: Optional validation results
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    target_pop = mc_results['target_population']
    analysis = mc_results['statistical_analysis']
    raw_results = mc_results['raw_results']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define colors for conditions
    condition_colors = {
        'full_network': '#2E86AB',
        'block_exc_to_interneurons': '#DC143C',  # Red - key test
        'block_int_to_int': '#FF8C00',
        'block_mc_to_interneurons': '#9370DB',
        'block_principal_recurrent': '#20B2AA',
        'cnqx_apv_broad': '#808080',
        'gabazine': '#6A994E',
        'no_stimulation': '#D3D3D3'
    }
    
    populations = [p for p in ['gc', 'mc', 'pv', 'sst'] if p != target_pop]
    
    # Panel 1: Paradoxical excitation fractions
    ax1 = plt.subplot(3, 4, 1)
    plot_paradoxical_fractions(ax1, raw_results, populations, condition_colors, target_pop)
    
    # Panel 2: Mean rate changes
    ax2 = plt.subplot(3, 4, 2)
    plot_mean_rate_changes(ax2, raw_results, populations, condition_colors, target_pop)
    
    # Panel 3: Gini coefficient changes
    ax3 = plt.subplot(3, 4, 3)
    plot_gini_changes(ax3, raw_results, populations, condition_colors, target_pop)
    
    # Panel 4: Effect sizes (primary hypothesis)
    ax4 = plt.subplot(3, 4, 4)
    plot_effect_sizes(ax4, analysis['primary_disinhibition_test'], populations)
    
    # Panel 5-7: Distribution comparisons for each population
    for idx, pop in enumerate(populations):
        ax = plt.subplot(3, 4, 5 + idx)
        plot_distribution_comparison(ax, raw_results, pop, condition_colors)
    
    # Panel 8: Network-wide paradoxical excitation
    ax8 = plt.subplot(3, 4, 8)
    plot_network_paradoxical(ax8, raw_results, condition_colors)
    
    # Panel 9: P-values heatmap
    ax9 = plt.subplot(3, 4, 9)
    plot_pvalue_heatmap(ax9, analysis)
    
    # Panel 10: Power analysis
    ax10 = plt.subplot(3, 4, 10)
    plot_power_analysis(ax10, analysis['primary_disinhibition_test'], populations)
    
    # Panel 11: Model validation (if available)
    if validation_results:
        ax11 = plt.subplot(3, 4, 11)
        plot_model_validation(ax11, validation_results)
    
    # Panel 12: Summary statistics table
    ax12 = plt.subplot(3, 4, 12)
    plot_summary_table(ax12, analysis, validation_results)
    
    plt.suptitle(f'Monte Carlo Analysis: {target_pop.upper()} Stimulation',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f"monte_carlo_analysis_{target_pop}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_paradoxical_fractions(ax, raw_results, populations, condition_colors, target_pop):
    """Plot paradoxical excitation fractions"""
    
    conditions = list(raw_results.keys())
    
    # Check if we have any data
    has_data = False
    for condition in conditions:
        if condition in raw_results and len(raw_results[condition]) > 0:
            has_data = True
            break
    
    if not has_data:
        ax.text(0.5, 0.5, 'No data available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Paradoxical Excitation Fractions', fontsize=11, fontweight='bold')
        return
    
    x_pos = np.arange(len(conditions))
    width = 0.8 / len(populations) if len(populations) > 0 else 0.8
    
    for i, pop in enumerate(populations):
        means = []
        sems = []
        
        for condition in conditions:
            if condition not in raw_results:
                means.append(0)
                sems.append(0)
                continue
                
            trials = raw_results[condition]
            values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in trials]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) > 0:
                means.append(np.mean(values))
                sems.append(np.std(values) / np.sqrt(len(values)))
            else:
                means.append(0)
                sems.append(0)
        
        # Plot bars for this population
        bar_positions = [x + i * width for x in x_pos]
        ax.bar(bar_positions, means, width, yerr=sems,
              label=pop.upper(), alpha=0.7, capsize=3,
              edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Condition', fontsize=10, fontweight='bold')
    ax.set_ylabel('Paradoxical Fraction', fontsize=10, fontweight='bold')
    ax.set_title('Paradoxical Excitation Fractions', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos + width * (len(populations) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8, loc='best', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)        

def plot_mean_rate_changes(ax, raw_results, populations, condition_colors, target_pop):
    """Plot mean firing rate changes"""
    
    conditions = list(raw_results.keys())
    x_pos = np.arange(len(conditions))
    width = 0.8 / len(populations)
    
    for i, pop in enumerate(populations):
        means = []
        sems = []
        
        for condition in conditions:
            trials = raw_results[condition]
            values = [t.get(f'{pop}_mean_rate_change', np.nan) for t in trials]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) > 0:
                means.append(np.mean(values))
                sems.append(np.std(values) / np.sqrt(len(values)))
            else:
                means.append(0)
                sems.append(0)
        
        ax.bar(x_pos + i * width, means, width, yerr=sems,
              label=pop.upper(), alpha=0.7, capsize=3)
    
    ax.set_xlabel('Condition')
    ax.set_ylabel('Mean Rate Change (Hz)')
    ax.set_title('Mean Firing Rate Changes')
    ax.set_xticks(x_pos + width * (len(populations) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_gini_changes(ax, raw_results, populations, condition_colors, target_pop):
    """Plot Gini coefficient changes"""
    
    conditions = list(raw_results.keys())
    x_pos = np.arange(len(conditions))
    width = 0.8 / len(populations)
    
    for i, pop in enumerate(populations):
        means = []
        sems = []
        
        for condition in conditions:
            trials = raw_results[condition]
            values = [t.get(f'{pop}_gini_change', np.nan) for t in trials]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) > 0:
                means.append(np.mean(values))
                sems.append(np.std(values) / np.sqrt(len(values)))
            else:
                means.append(0)
                sems.append(0)
        
        ax.bar(x_pos + i * width, means, width, yerr=sems,
              label=pop.upper(), alpha=0.7, capsize=3)
    
    ax.set_xlabel('Condition')
    ax.set_ylabel('$\\Delta$ Gini Coefficient')
    ax.set_title('Firing Rate Inequality Changes')
    ax.set_xticks(x_pos + width * (len(populations) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_effect_sizes(ax, primary_hypothesis, populations):
    """Plot effect sizes (Cohen's d) for primary hypothesis"""
    
    metrics = [f'{pop}_paradoxical_fraction' for pop in populations]
    effect_sizes = []
    errors = []
    labels = []
    
    for metric in metrics:
        if metric in primary_hypothesis:
            result = primary_hypothesis[metric]
            effect_sizes.append(result.cohens_d)
            ci_width = (result.confidence_interval[1] - result.confidence_interval[0]) / 2
            errors.append(ci_width)
            labels.append(metric.split('_')[0].upper())
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, effect_sizes, yerr=errors, alpha=0.7, capsize=5)
    
    # Color bars by effect size magnitude
    for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
        if abs(d) < 0.2:
            bar.set_color('#999999')  # Gray for negligible
        elif abs(d) < 0.5:
            bar.set_color('#FFA500')  # Orange for small
        elif abs(d) < 0.8:
            bar.set_color('#FF6347')  # Tomato for medium
        else:
            bar.set_color('#DC143C')  # Crimson for large
    
    ax.set_xlabel('Population')
    ax.set_ylabel("Cohen's d")
    ax.set_title('Effect Sizes (Full vs Disinhibition)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3, label='Small')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Medium')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, label='Large')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_distribution_comparison(ax, raw_results, pop, condition_colors):
    """Plot distribution comparison for a specific population"""
    
    # Compare full network vs disinhibition
    full_trials = raw_results['full_network']
    disinhibition_trials = raw_results['block_exc_to_interneurons']
    
    full_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in full_trials]
    disinhibition_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in disinhibition_trials]
    
    full_values = [v for v in full_values if not np.isnan(v)]
    disinhibition_values = [v for v in disinhibition_values if not np.isnan(v)]
    
    if len(full_values) > 0 and len(disinhibition_values) > 0:
        # Violin plots
        parts = ax.violinplot([full_values, disinhibition_values], positions=[1, 2],
                             showmeans=True, showextrema=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['#2E86AB', '#C73E1D'][i])
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Full\nNetwork', 'DISINHIBITION/\nAPV'], fontsize=8)
        ax.set_ylabel('Paradoxical Fraction')
        ax.set_title(f'{pop.upper()} Distribution')
        ax.grid(True, alpha=0.3)


def plot_network_paradoxical(ax, raw_results, condition_colors):
    """Plot network-wide paradoxical excitation"""
    
    conditions = list(raw_results.keys())
    means = []
    sems = []
    colors = []
    valid_conditions = []  # Track conditions with valid data
    
    for condition in conditions:
        trials = raw_results[condition]
        values = [t.get('network_total_paradoxical', np.nan) for t in trials]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            means.append(np.mean(values))
            sems.append(np.std(values) / np.sqrt(len(values)))
            colors.append(condition_colors.get(condition, '#999999'))
            valid_conditions.append(condition)
    
    if len(means) == 0:
        ax.text(0.5, 0.5, 'No valid data', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Network-Wide Paradoxical Excitation', fontsize=11, fontweight='bold')
        return
    
    x_pos = np.arange(len(valid_conditions))
    bars = ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7, capsize=5,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Condition', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Paradoxical Fraction', fontsize=10, fontweight='bold')
    ax.set_title('Network-Wide Paradoxical Excitation', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_conditions, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance stars comparing to full_network
    if 'full_network' in valid_conditions:
        full_idx = valid_conditions.index('full_network')
        full_mean = means[full_idx]
        full_sem = sems[full_idx]
        
        # Get full_network values for statistical test
        full_values = [t.get('network_total_paradoxical', np.nan) 
                      for t in raw_results['full_network']]
        full_values = [v for v in full_values if not np.isnan(v)]
        
        for i, condition in enumerate(valid_conditions):
            if condition == 'full_network':
                continue
            
            # Get condition values for statistical test
            cond_values = [t.get('network_total_paradoxical', np.nan) 
                          for t in raw_results[condition]]
            cond_values = [v for v in cond_values if not np.isnan(v)]
            
            if len(cond_values) > 0:
                # Perform t-test
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(full_values, cond_values)
                
                # Add significance marker
                if p_value < 0.001:
                    marker = '***'
                elif p_value < 0.01:
                    marker = '**'
                elif p_value < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    y_pos = means[i] + sems[i] * 1.2
                    ax.text(i, y_pos, marker, ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')

                    

def plot_pvalue_heatmap(ax, analysis):
    """Plot p-value heatmap for all comparisons"""
    
    primary = analysis['primary_disinhibition_test']
    
    metrics = list(primary.keys())
    p_values = [-np.log10(primary[m].p_value + 1e-10) for m in metrics]
    
    # Create heatmap data
    heatmap_data = np.array(p_values).reshape(-1, 1)
    
    im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=0, vmax=4)
    
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([m.replace('_', ' ') for m in metrics], fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['Full vs\nDisinhibition'], fontsize=8)
    ax.set_title('-log10(p-value)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add significance thresholds
    ax.axhline(y=-0.5, color='black', linewidth=2)
    for i in range(len(metrics)):
        pval = primary[metrics[i]].p_value
        if pval < 0.001:
            marker = '***'
        elif pval < 0.01:
            marker = '**'
        elif pval < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        ax.text(0, i, marker, ha='center', va='center', fontsize=10, 
               color='white' if p_values[i] > 2 else 'black')


def plot_power_analysis(ax, primary_hypothesis, populations):
    """Plot statistical power analysis"""
    
    metrics = [f'{pop}_paradoxical_fraction' for pop in populations]
    powers = []
    labels = []
    
    for metric in metrics:
        if metric in primary_hypothesis:
            result = primary_hypothesis[metric]
            powers.append(result.statistical_power)
            labels.append(metric.split('_')[0].upper())
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, powers, alpha=0.7)
    
    # Color bars by power level
    for bar, power in zip(bars, powers):
        if power >= 0.8:
            bar.set_color('#2E8B57')  # Green for adequate
        elif power >= 0.6:
            bar.set_color('#FFA500')  # Orange for marginal
        else:
            bar.set_color('#DC143C')  # Red for insufficient
    
    ax.set_xlabel('Population')
    ax.set_ylabel('Statistical Power')
    ax.set_title('Statistical Power Analysis')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Adequate (0.8)')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Marginal (0.6)')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_model_validation(ax, validation_results):
    """Plot model validation against experimental data"""
    
    metrics = [k for k in validation_results.keys() if k != 'overall_validation_score']
    model_means = [validation_results[m]['model_mean'] for m in metrics]
    model_sems = [validation_results[m]['model_sem'] for m in metrics]
    exp_values = [validation_results[m]['experimental_value'] for m in metrics]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x_pos - width/2, model_means, width, yerr=model_sems,
          label='Model', alpha=0.7, capsize=3)
    ax.scatter(x_pos + width/2, exp_values, color='red', s=100,
              marker='D', label='Experimental', zorder=10)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Paradoxical Fraction')
    ax.set_title(f"Model Validation (Score: {validation_results['overall_validation_score']:.2f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ') for m in metrics], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_summary_table(ax, analysis, validation_results):
    """Plot summary statistics table"""
    
    ax.axis('off')
    
    # Create summary text
    primary = analysis['primary_disinhibition_test']
    network_result = primary.get('network_total_paradoxical')
    
    if network_result:
        summary_text = f"Primary Hypothesis Test:\n"
        summary_text += f"  Effect size: {network_result.cohens_d:.3f}\n"
        summary_text += f"  P-value: {network_result.p_value:.6f}\n"
        summary_text += f"  Power: {network_result.statistical_power:.3f}\n"
        summary_text += f"  {network_result.interpretation}\n\n"
        
        if validation_results:
            summary_text += f"Model Validation:\n"
            summary_text += f"  Overall score: {validation_results['overall_validation_score']:.3f}\n"
            
            if validation_results['overall_validation_score'] > 0.7:
                summary_text += "  Status: good match\n"
            else:
                summary_text += "  Status: needs improvement\n"
        
        # Interpretation
        summary_text += "\nConclusion:\n"
        if network_result.p_value < 0.001 and abs(network_result.cohens_d) > 0.5:
            summary_text += "  Strong support for\n  disinhibition hypothesis"
        elif network_result.p_value < 0.05:
            summary_text += "  Moderate support for\n  disinhibition hypothesis"
        else:
            summary_text += "  Insufficient evidence for\n  disinhibition hypothesis"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        
def plot_paradoxical_excitation_violins(mc_results: Dict,
                                        output_dir: str = ".",
                                        show_plots: bool = True):
    """
    Create violin plots showing paradoxical excitation across multiple trials
    
    Args:
        mc_results: Results from monte_carlo_analysis()
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    target_pop = mc_results['target_population']
    raw_results = mc_results['raw_results']
    
    # Define colors
    colors = {
        'pv': '#FF6B9D',
        'sst': '#45B7D1',
        'gc': '#96CEB4',
        'mc': '#FFEAA7',
    }
    
    condition_colors = {
        'full_network': '#2E86AB',
        'block_exc_to_interneurons': '#DC143C',  # Red - key test
        'block_int_to_int': '#FF8C00',
        'block_mc_to_interneurons': '#9370DB',
        'block_principal_recurrent': '#20B2AA',
        'cnqx_apv_broad': '#808080',
        'gabazine': '#6A994E',
        'no_stimulation': '#D3D3D3'
    }
    
    populations = [p for p in ['gc', 'mc', 'pv', 'sst'] if p != target_pop]
    
    # Create figure with multiple violin plot panels
    fig = plt.figure(figsize=(20, 14))
    
    # Add title emphasizing this is multi-trial data
    fig.text(0.5, 0.98, f'Multi-trial analysis (n={len(raw_results["full_network"])} trials)', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Panel 1: Paradoxical excitation fractions - violin plots across conditions
    ax1 = plt.subplot(3, 3, 1)
    plot_violin_paradoxical_by_condition(ax1, raw_results, populations, condition_colors, target_pop)
    
    # Panel 2: Paradoxical excitation - comparison of full network vs disinhibition
    ax2 = plt.subplot(3, 3, 2)
    plot_violin_primary_comparison(ax2, raw_results, populations, colors, target_pop)
    
    # Panel 3: Firing rate modulation ratios - violin plots
    ax3 = plt.subplot(3, 3, 3)
    plot_violin_rate_modulation(ax3, raw_results, populations, colors, target_pop)
    
    # Panels 4-6: Individual population violin plots
    for idx, pop in enumerate(populations):
        ax = plt.subplot(3, 3, 4 + idx)
        plot_violin_single_population(ax, raw_results, pop, condition_colors, target_pop)
    
    # Panel 7: Network-wide paradoxical excitation
    ax7 = plt.subplot(3, 3, 7)
    plot_violin_network_wide(ax7, raw_results, condition_colors)
    
    # Panel 8: Gini coefficient changes
    ax8 = plt.subplot(3, 3, 8)
    plot_violin_gini_changes(ax8, raw_results, populations, colors, target_pop)
    
    # Panel 9: Statistical summary
    ax9 = plt.subplot(3, 3, 9)
    plot_statistical_summary_box(ax9, mc_results)
    
    plt.suptitle(f'Paradoxical Excitation Analysis: {target_pop.upper()} Stimulation (Multi-Trial)', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_file = output_path / f"paradoxical_excitation_violins_{target_pop}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved violin plot: {output_file}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_violin_paradoxical_by_condition(ax, raw_results, populations, condition_colors, target_pop):
    """Violin plots of paradoxical excitation fractions across all conditions"""
    
    conditions = ['full_network', 'block_exc_to_interneurons', 'block_int_to_int',
                  'block_mc_to_interneurons', 'block_principal_recurrent',
                  'cnqx_apv_broad', 'gabazine', 'no_stimulation']
    
    # Collect data for the most responsive population
    pop = populations[0]  # Use first non-target population
    
    violin_data = []
    labels = []
    plot_colors = []
    
    for condition in conditions:
        if condition in raw_results:
            trials = raw_results[condition]
            values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in trials]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) > 0:
                violin_data.append(values)
                # Short labels for readability
                label_map = {
                    'full_network': 'Full',
                    'disinhibition_test': 'Disinh. test',
                    'cnqx_apv_broad': 'CNQX/APV',
                    'gabazine': 'Gabazine',
                    'no_stimulation': 'No Stim'
                }
                labels.append(label_map.get(condition, condition))
                plot_colors.append(condition_colors[condition])
    
    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(labels)),
                             showmeans=True, showextrema=True, widths=0.7)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Customize means and extrema
        for partname in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(2)
        
        # Add scatter points for individual trials
        for i, values in enumerate(violin_data):
            y = values
            x = np.random.normal(i, 0.04, size=len(y))  # Add jitter
            ax.scatter(x, y, alpha=0.3, s=20, color=plot_colors[i], edgecolors='black', linewidth=0.5)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Paradoxical Excitation\nFraction', fontsize=10)
    ax.set_title(f'{pop.upper()} Response Across Conditions\n(n={len(violin_data[0])} trials per condition)', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_violin_primary_comparison(ax, raw_results, populations, colors, target_pop):
    """Violin plots comparing full network vs disinhibition test for all populations"""
    
    violin_data = []
    labels = []
    plot_colors = []
    positions = []
    
    # Check if required conditions exist
    if 'full_network' not in raw_results:
        ax.text(0.5, 0.5, 'No full_network data available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Primary Hypothesis Test:\nFull Network vs Blocked Exc->Int', 
                    fontsize=11, fontweight='bold')
        return
    
    if 'block_exc_to_interneurons' not in raw_results:
        ax.text(0.5, 0.5, 'No block_exc_to_interneurons data available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Primary Hypothesis Test:\nFull Network vs Blocked Exc->Int', 
                    fontsize=11, fontweight='bold')
        return
    
    for pop in populations:
        # Full network data
        full_trials = raw_results['full_network']
        full_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in full_trials]
        full_values = [v for v in full_values if not np.isnan(v)]
        
        # Disinhibition test data
        disinhibition_trials = raw_results['block_exc_to_interneurons']
        disinhibition_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) 
                               for t in disinhibition_trials]
        disinhibition_values = [v for v in disinhibition_values if not np.isnan(v)]
        
        if len(full_values) > 0 and len(disinhibition_values) > 0:
            violin_data.extend([full_values, disinhibition_values])
            labels.extend([f'{pop.upper()}\nFull', f'{pop.upper()}\nBlocked\nExc->Int'])
            plot_colors.extend([colors[pop], '#CCCCCC'])
    
    # Check if we have any valid data
    if not violin_data or len(violin_data) == 0:
        ax.text(0.5, 0.5, 'No valid data for comparison', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Primary Hypothesis Test:\nFull Network vs Blocked Exc->Int', 
                    fontsize=11, fontweight='bold')
        return
    
    current_pos = 0
    for i in range(len(violin_data)):
        if i % 2 == 0:
            positions.append(current_pos)
        else:
            positions.append(current_pos + 0.5)
            current_pos += 1.5
    
    # Create violin plots
    parts = ax.violinplot(violin_data, positions=positions,
                         showmeans=True, showextrema=True, widths=0.4)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plot_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Customize mean markers
    for partname in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(2)
    
    # Add significance markers
    for i in range(0, len(violin_data), 2):
        if i + 1 >= len(violin_data):  # ✓ Safety check
            break
            
        full_data = violin_data[i]
        blocked_data = violin_data[i+1]
        
        if len(full_data) == 0 or len(blocked_data) == 0:  # ✓ Safety check
            continue
        
        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(full_data, blocked_data)
        
        # Determine significance marker
        if p_value < 0.001:
            marker = '***'
        elif p_value < 0.01:
            marker = '**'
        elif p_value < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        
        # Add significance bracket and marker
        y_max = max(max(full_data), max(blocked_data))
        x_pos = (positions[i] + positions[i+1]) / 2
        y_bracket = y_max * 1.1
        
        # Draw bracket
        ax.plot([positions[i], positions[i], positions[i+1], positions[i+1]], 
               [y_bracket*0.98, y_bracket, y_bracket, y_bracket*0.98], 
               'k-', linewidth=1.5)
        
        # Add marker text
        ax.text(x_pos, y_bracket * 1.02, marker, ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    # Set x-axis properties
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Paradoxical Excitation\nFraction', fontsize=10, fontweight='bold')
    ax.set_title('Primary Hypothesis Test:\nFull Network vs Blocked Exc->Int', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)    

def plot_violin_rate_modulation(ax, raw_results, populations, colors, target_pop):
    """Violin plots of firing rate modulation ratios"""
    
    full_trials = raw_results['full_network']
    
    violin_data = []
    labels = []
    plot_colors = []
    
    for pop in populations:
        ratios = []
        for trial in full_trials:
            baseline = trial.get(f'{pop}_mean_baseline', np.nan)
            stim = trial.get(f'{pop}_mean_stim', np.nan)
            
            if not np.isnan(baseline) and not np.isnan(stim) and baseline > 0:
                ratio = np.log2(stim / baseline)
                ratios.append(ratio)
        
        if len(ratios) > 0:
            violin_data.append(ratios)
            labels.append(pop.upper())
            plot_colors.append(colors[pop])
    
    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(labels)),
                             showmeans=True, showextrema=True, widths=0.7)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add scatter points
        for i, values in enumerate(violin_data):
            y = values
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, color=plot_colors[i], 
                      edgecolors='black', linewidth=0.5)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'Rate Modulation ($\log_2$)', fontsize=10)
    ax.set_title(f'Firing Rate Modulation\n(Full Network, n={len(violin_data[0])} trials)', 
                fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_violin_single_population(ax, raw_results, pop, condition_colors, target_pop):
    """Detailed violin plot for a single population across all conditions"""
    
    conditions = ['full_network', 'block_exc_to_interneurons', 'block_int_to_int',
                  'block_mc_to_interneurons', 'block_principal_recurrent',
                  'cnqx_apv_broad', 'gabazine', 'no_stimulation']

    
    violin_data = []
    labels = []
    plot_colors = []
    
    for condition in conditions:
        if condition in raw_results:
            trials = raw_results[condition]
            values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in trials]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) > 0:
                violin_data.append(values)
                label_map = {
                    'full_network': 'Full',
                    'block_exc_to_interneurons': 'Block Exc Int',
                    'block_int_to_int': 'Block Int-Int',
                    'cnqx_apv_broad': 'CNQX',
                    'gabazine': 'Gaba',
                    'no_stimulation': 'None'
                }
                labels.append(label_map.get(condition, condition))
                plot_colors.append(condition_colors[condition])
    
    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(labels)),
                             showmeans=True, showextrema=True, widths=0.7)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Calculate and show means
        means = [np.mean(data) for data in violin_data]
        ax.plot(range(len(means)), means, 'ko-', linewidth=2, markersize=8,
               label='Mean', zorder=10)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Paradoxical Fraction', fontsize=10)
    ax.set_title(f'{pop.upper()} Paradoxical Excitation', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_violin_network_wide(ax, raw_results, condition_colors):
    """Violin plot of network-wide paradoxical excitation"""
    
    conditions = ['full_network', 'block_exc_to_interneurons', 'block_int_to_int',
                  'block_mc_to_interneurons', 'block_principal_recurrent',
                  'cnqx_apv_broad', 'gabazine', 'no_stimulation']

    
    violin_data = []
    labels = []
    plot_colors = []
    
    for condition in conditions:
        if condition in raw_results:
            trials = raw_results[condition]
            values = [t.get('network_total_paradoxical', np.nan) for t in trials]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) > 0:
                violin_data.append(values)
                label_map = {
                    'full_network': 'Full\nNetwork',
                    'block_exc_to_interneurons': 'Block\nExc-IN',
                    'block_int_to_int': 'Block\nIN-IN',
                    'cnqx_apv_broad': 'CNQX/\nAPV',
                    'gabazine': 'Gabazine',
                    'no_stimulation': 'No\nStim'
                }
                labels.append(label_map.get(condition, condition))
                plot_colors.append(condition_colors[condition])
    
    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(labels)),
                             showmeans=True, showextrema=True, widths=0.7)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add box plot overlay for quartiles
        for i, data in enumerate(violin_data):
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            ax.plot([i-0.1, i+0.1], [median, median], 'k-', linewidth=3)
            ax.plot([i, i], [q1, q3], 'k-', linewidth=2)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Total Paradoxical\nExcitation', fontsize=10)
    ax.set_title('Network-Wide Paradoxical Excitation', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_violin_gini_changes(ax, raw_results, populations, colors, target_pop):
    """Violin plots of Gini coefficient changes"""
    
    full_trials = raw_results['full_network']
    
    violin_data = []
    labels = []
    plot_colors = []
    
    for pop in populations:
        gini_changes = []
        for trial in full_trials:
            gini_change = trial.get(f'{pop}_gini_change', np.nan)
            if not np.isnan(gini_change):
                gini_changes.append(gini_change)
        
        if len(gini_changes) > 0:
            violin_data.append(gini_changes)
            labels.append(pop.upper())
            plot_colors.append(colors[pop])
    
    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(labels)),
                             showmeans=True, showextrema=True, widths=0.7)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add scatter points
        for i, values in enumerate(violin_data):
            y = values
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, color=plot_colors[i],
                      edgecolors='black', linewidth=0.5)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'$\Delta$ Gini Coefficient', fontsize=10)
    ax.set_title(f'Firing Rate Inequality Changes\n(n={len(violin_data[0])} trials)', 
                fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_statistical_summary_box(ax, mc_results):
    """Display statistical summary of multi-trial analysis"""
    
    ax.axis('off')
    
    analysis = mc_results['statistical_analysis']
    primary = analysis['primary_disinhibition_test']
    target_pop = mc_results['target_population']
    n_trials = len(mc_results['raw_results']['full_network'])
    
    summary_text = "Multi-Trial Statistical Summary\n"
    summary_text += "=" * 35 + "\n\n"
    summary_text += f"Target: {target_pop.upper()} stimulation\n"
    summary_text += f"Trials per condition: {n_trials}\n"
    summary_text += f"Total simulations: {n_trials * 6}\n\n"
    
    # Primary hypothesis test
    network_result = primary.get('network_total_paradoxical')
    if network_result:
        summary_text += "Primary Hypothesis:\n"
        summary_text += f"  Effect size (d): {network_result.cohens_d:.3f}\n"
        summary_text += f"  P-value: {network_result.p_value:.2e}\n"
        summary_text += f"  Power: {network_result.statistical_power:.3f}\n"
        summary_text += f"  95% CI: [{network_result.confidence_interval[0]:.3f},\n"
        summary_text += f"           {network_result.confidence_interval[1]:.3f}]\n\n"
        
        # Interpretation
        summary_text += "Interpretation:\n"
        if network_result.p_value < 0.001 and abs(network_result.cohens_d) > 0.5:
            summary_text += "  - strong support for\n"
            summary_text += "    disinhibition hypothesis\n"
        elif network_result.p_value < 0.05:
            summary_text += "  - moderate support for\n"
            summary_text += "    disinhibition hypothesis\n"
        else:
            summary_text += "  - insufficient evidence\n"
            summary_text += "    for disinhibition\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, va='top', ha='left',
           fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))


def create_violin_plots(mc_results: Dict,
                        validation_results: Dict = None,
                        output_dir: str = ".",
                        show_plots: bool = True):
    """
    Create violin plots with proper styling
    
    Args:
        mc_results: Results from monte_carlo_analysis()
        validation_results: Optional validation results
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    target_pop = mc_results['target_population']
    raw_results = mc_results['raw_results']
    analysis = mc_results['statistical_analysis']
    
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with proper dimensions for publication (2-column format)
    fig = plt.figure(figsize=(7, 9))
    
    # Define colors (colorblind-friendly palette)
    colors = {
        'pv': '#E69F00',  # Orange
        'sst': '#56B4E9',  # Sky blue
        'gc': '#009E73',  # Green
        'mc': '#F0E442',  # Yellow
    }
    
    populations = [p for p in ['gc', 'mc', 'pv', 'sst'] if p != target_pop]
    
    # Panel A: Primary comparison (Full vs disinhibition) - 2x2 grid
    for idx, pop in enumerate(populations):
        ax = plt.subplot(3, 2, idx + 1)
        
        # Get data
        full_trials = raw_results['full_network']
        disinhibition_trials = raw_results['block_exc_to_interneurons']
        
        full_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in full_trials]
        disinhibition_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in disinhibition_trials]
        
        full_values = [v for v in full_values if not np.isnan(v)]
        disinhibition_values = [v for v in disinhibition_values if not np.isnan(v)]
        
        if len(full_values) > 0 and len(disinhibition_values) > 0:
            # Create violin plot
            parts = ax.violinplot([full_values, disinhibition_values], positions=[0, 1],
                                 showmeans=False, showextrema=False, widths=0.7)
            
            # Color violins
            parts['bodies'][0].set_facecolor('#2E86AB')
            parts['bodies'][1].set_facecolor('#C73E1D')
            for pc in parts['bodies']:
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Add boxplot overlay for quartiles
            bp = ax.boxplot([full_values, disinhibition_values], positions=[0, 1],
                          widths=0.3, showfliers=False,
                          boxprops=dict(linewidth=1.5, color='black'),
                          whiskerprops=dict(linewidth=1.5, color='black'),
                          capprops=dict(linewidth=1.5, color='black'),
                          medianprops=dict(linewidth=2, color='red'))
            
            # Add individual data points with jitter
            for i, values in enumerate([full_values, disinhibition_values]):
                y = values
                x = np.random.normal(i, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.4, s=15, color='black', zorder=10)
            
            # Statistical test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(full_values, disinhibition_values)
            
            # Add significance marker
            y_max = max(max(full_values), max(disinhibition_values))
            if p_value < 0.001:
                marker = '***'
            elif p_value < 0.01:
                marker = '**'
            elif p_value < 0.05:
                marker = '*'
            else:
                marker = 'ns'
            
            # Draw significance bracket
            x1, x2 = 0, 1
            y = y_max * 1.15
            ax.plot([x1, x1, x2, x2], [y*0.98, y, y, y*0.98], 'k-', linewidth=1.5)
            ax.text(0.5, y*1.02, marker, ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
            
            # Effect size
            cohens_d = (np.mean(full_values) - np.mean(disinhibition_values)) / \
                      np.sqrt((np.var(full_values) + np.var(disinhibition_values)) / 2)
            
            # Add text with statistics
            stats_text = f"d={cohens_d:.2f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Full\nNetwork', 'DISINHIBITION/\nAPV'], fontsize=9)
        ax.set_ylabel('Paradoxical\nExcitation Fraction', fontsize=9)
        ax.set_title(f'{pop.upper()}', fontsize=11, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
    
    # Panel E: Network-wide effect
    ax5 = plt.subplot(3, 2, 5)
    
    conditions = ['full_network', 'block_exc_to_interneurons', 'gabazine', 'no_stimulation']
    condition_labels = ['Full\nNetwork', 'Disinhibition', 'Gabazine', 'No\nStim']
    condition_colors_list = ['#2E86AB', '#C73E1D', '#6A994E', '#999999']
    
    violin_data = []
    for condition in conditions:
        if condition in raw_results:
            trials = raw_results[condition]
            values = [t.get('network_total_paradoxical', np.nan) for t in trials]
            values = [v for v in values if not np.isnan(v)]
            if len(values) > 0:
                violin_data.append(values)
    
    if violin_data:
        parts = ax5.violinplot(violin_data, positions=range(len(violin_data)),
                              showmeans=False, showextrema=False, widths=0.7)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(condition_colors_list[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add boxplot overlay
        bp = ax5.boxplot(violin_data, positions=range(len(violin_data)),
                        widths=0.3, showfliers=False,
                        boxprops=dict(linewidth=1.5, color='black'),
                        whiskerprops=dict(linewidth=1.5, color='black'),
                        capprops=dict(linewidth=1.5, color='black'),
                        medianprops=dict(linewidth=2, color='red'))
    
    ax5.set_xticks(range(len(condition_labels)))
    ax5.set_xticklabels(condition_labels, fontsize=9)
    ax5.set_ylabel('Network-Wide\nParadoxical Excitation', fontsize=9)
    ax5.set_title('Network Effect Across Conditions', fontsize=11, fontweight='bold', pad=10)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.set_ylim(bottom=0)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax5.set_axisbelow(True)
    
    # Panel F: Summary statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Create summary table
    primary = analysis['primary_disinhibition_test']
    network_result = primary.get('network_total_paradoxical')
    
    summary_text = f"Statistical Summary\n"
    summary_text += "─" * 30 + "\n"
    summary_text += f"Target: {target_pop.upper()} stimulation\n"
    summary_text += f"n = {len(raw_results['full_network'])} trials\n\n"
    
    if network_result:
        summary_text += "Primary Hypothesis Test:\n"
        summary_text += f"  Effect size: d = {network_result.cohens_d:.3f}\n"
        summary_text += f"  P-value: {network_result.p_value:.2e}\n"
        summary_text += f"  Power: {network_result.statistical_power:.2f}\n"
        summary_text += f"  95% CI: [{network_result.confidence_interval[0]:.3f},\n"
        summary_text += f"          {network_result.confidence_interval[1]:.3f}]\n\n"
        
        # Conclusion
        if network_result.p_value < 0.001 and abs(network_result.cohens_d) > 0.5:
            conclusion = "STRONG support"
        elif network_result.p_value < 0.05:
            conclusion = "MODERATE support"
        else:
            conclusion = "INSUFFICIENT evidence"
        
        summary_text += f"Conclusion:\n  {conclusion}\n  for disinhibition hypothesis"
    
    # Add validation score if available
    if validation_results:
        summary_text += f"\n\nModel Validation:\n"
        summary_text += f"  Score: {validation_results['overall_validation_score']:.2f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Main title
    plt.suptitle(f'Disinhibition Hypothesis: {target_pop.upper()} Stimulation (n={len(raw_results["full_network"])} trials)', 
                fontsize=12, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    # Save figure
    output_file = output_path / f"detailed_violins_{target_pop}.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    # Also save as PDF for publication
    output_file_pdf = output_path / f"detailed_violins_{target_pop}.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    
    print(f"Saved detailed plots: {output_file} and {output_file_pdf}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()



# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_statistical_analysis(
    target_populations: List[str] = ['pv', 'sst'],
    n_trials: int = 50,
    light_intensity: float = 1.0,
    mec_current: float = 100.0,
    opsin_current: float = 200.0,
    n_workers: int = 1,
    device: Optional[str] = None,
    gpu_batch_size: int = 8,
    optimization_json_file: str = None,
    output_dir: str = "./statistical_results",
    show_plots: bool = False):
    """
    Run statistical analysis for both PV and SST stimulation
    
    Args:
        target_populations: List of populations to test
        n_trials: Number of Monte Carlo trials per condition
        light_intensity: Optogenetic stimulation intensity
        mec_current: MEC drive current (pA)
        opsin_current: Direct opsin activation current (pA)
        n_workers: Number of parallel workers (for CPU multiprocessing)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        gpu_batch_size: Batch size for GPU evaluation (trials per batch)
        optimization_json_file: Path to optimization results (optional)
        output_dir: Directory for saving results
        show_plots: Whether to display plots (default: False, only save to files)
    """
    
    print("=" * 80)
    print("Statistical analysis of DG disinhibition hypothesis")
    print(f"Multi-trial Monte Carlo approach (n={n_trials} trials per condition)")
    print("=" * 80)
    
    # Device setup
    if device is None:
        eval_device = get_default_device()
    else:
        eval_device = torch.device(device)
    
    print(f"\nDevice: {eval_device}")
    if eval_device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(eval_device)}")
        print(f"Memory: {torch.cuda.get_device_properties(eval_device).total_memory / 1024**3:.2f} GB")
        print(f"Batch size: {gpu_batch_size} trials per batch")
    else:
        print(f"CPU: {mp.cpu_count()} cores available")
        print(f"Workers: {n_workers}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    
    for target in target_populations:
        print(f"\n\nAnalyzing {target.upper()} stimulation...")
        print("-" * 80)
        
        # Initialize tester
        tester = DisinhibitionHypothesisTester(
            n_trials=n_trials,
            optimization_json_file=optimization_json_file
        )
        
        # Determine evaluation strategy
        use_multiprocessing = (eval_device.type == 'cpu' and n_workers > 1)
        
        # Run Monte Carlo analysis with device specification
        mc_results = tester.monte_carlo_analysis(
            target,
            light_intensity=light_intensity,
            mec_current=mec_current,
            opsin_current=opsin_current,
            n_workers=n_workers,
            use_multiprocessing=use_multiprocessing,
            device=eval_device,
            gpu_batch_size=gpu_batch_size
        )
        
        # Validate against experimental data
        validation_results = tester.validate_against_experimental_data(mc_results)
        
        # Generate report
        report = tester.generate_statistical_report(
            mc_results,
            validation_results,
            output_file=str(output_path / f"report_{target}.txt")
        )
        
        print("\n" + report)
        
        # Save results
        tester.save_results(mc_results, validation_results, output_dir=str(output_path))
        
        # Create plots
        print("\nGenerating statistical analysis plots...")
        
        # Generate comprehensive Monte Carlo results visualization
        print(f"  - Monte Carlo results plot...")
        plot_monte_carlo_results(
            mc_results, 
            validation_results, 
            output_dir=str(output_path),
            show_plots=show_plots
        )
        
        # Generate mechanistic dissection plots
        print(f"  - Disinhibition test results plot...")
        plot_disinhibition_test_results(
            mc_results, 
            output_dir=str(output_path),
            show_plots=show_plots
        )
        
        # Generate detailed violin plots
        print(f"  - Detailed violin plots...")
        create_violin_plots(
            mc_results, 
            validation_results, 
            output_dir=str(output_path),
            show_plots=show_plots
        )
        
        all_results[target] = {
            'mc_results': mc_results,
            'validation': validation_results,
            'report': report
        }

    if 'pv' in all_results and 'sst' in all_results:
        print("\nGenerating cross-comparison plot (PV vs SST)...")
        plot_statistical_validity_summary(
            all_results['pv']['mc_results'], 
            all_results['sst']['mc_results'],
            output_dir=str(output_path),
            show_plots=show_plots
        )
        
    print("\n" + "=" * 80)
    print(f"Analysis complete. Results saved to: {output_path}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Statistical Testing Framework for DG Circuit')
    parser.add_argument('--optimization-file', type=str, default=None,
                       help='Path to optimization results JSON file')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials per condition (default: 50)')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', None],
                       help='Device to use: cpu, cuda, or None for auto-detect')
    parser.add_argument('--gpu-batch-size', type=int, default=8,
                       help='Batch size for GPU evaluation (default: 8)')
    parser.add_argument('--n-workers', type=int, default=4,
                       help='Number of CPU workers for multiprocessing (default: 4)')
    parser.add_argument('--light-intensity', type=float, default=1.0,
                       help='Optogenetic light intensity (default: 1.0)')
    parser.add_argument('--mec-current', type=float, default=40.0,
                       help='MEC drive current in pA (default: 40.0)')
    parser.add_argument('--opsin-current', type=float, default=200.0,
                       help='Opsin activation current in pA (default: 200.0)')
    parser.add_argument('--output-dir', type=str, default='./statistical_results',
                       help='Output directory (default: ./statistical_results)')
    parser.add_argument('--targets', nargs='+', default=['sst', 'pv'],
                       help='Target populations to test (default: sst pv)')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively (default: False, only save to files)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Statistical Testing Framework for DG Circuit")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Trials per condition: {args.n_trials}")
    print(f"  Device: {args.device if args.device else 'auto-detect'}")
    print(f"  GPU batch size: {args.gpu_batch_size}")
    print(f"  CPU workers: {args.n_workers}")
    print(f"  Light intensity: {args.light_intensity}")
    print(f"  MEC current: {args.mec_current} pA")
    print(f"  Opsin current: {args.opsin_current} pA")
    print(f"  Target populations: {args.targets}")
    print(f"  Show plots: {args.show_plots}")
    
    if args.optimization_file:
        print(f"  Using optimization results: {args.optimization_file}")
    
    # Run analysis
    results = run_statistical_analysis(
        target_populations=args.targets,
        n_trials=args.n_trials,
        light_intensity=args.light_intensity,
        mec_current=args.mec_current,
        opsin_current=args.opsin_current,
        n_workers=args.n_workers,
        device=args.device,
        gpu_batch_size=args.gpu_batch_size,
        optimization_json_file=args.optimization_file,
        output_dir=args.output_dir,
        show_plots=args.show_plots
    )
