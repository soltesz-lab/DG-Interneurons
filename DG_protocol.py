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
    dendritic_somatic_transfer,
    get_cell_type_parameters,
    DendriticParameters,
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
        no_expression = torch.rand(self.n_cells, device=self.device) < self.params.failure_rate
        expression[no_expression] = 0.0
        
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
                 base_seed: int = 42):
        """
        Initialize optogenetic experiment
        
        Args:
            circuit_params: CircuitParams instance
            synaptic_params: PerConnectionSynapticParams instance
            opsin_params: OpsinParams instance
            optimization_json_file: Optional path to optimization results JSON
            device: Device to run on
            base_seed: Base random seed for reproducibility (default: 42)
        """
        self.circuit_params = circuit_params
        self.opsin_params = opsin_params
        self.synaptic_params = synaptic_params
        self.device = device if device is not None else get_default_device()
        self.base_seed = base_seed
        
        if optimization_json_file is not None:
            success = self._load_optimization_results(optimization_json_file)
            if success:
                print(f"OptogeneticExperiment initialized with optimized parameters from {optimization_json_file}")
            else:
                print(f"Failed to load optimization file. Using default parameters.")

        # Create initial circuit (will be recreated for each trial)
        self.circuit = self._create_circuit(base_seed)
        
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
                             post_duration: float = 250.0,
                             mec_current: float = 100.0,
                             mec_current_std: float = 1.0,
                             opsin_current: float = 100.0,
                             include_dentate_spikes: bool = False,
                             ds_times: Optional[List[float]] = None,
                             plot_activity: bool = False,
                             n_trials: int = 1) -> Dict:
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
        
        # Run multiple trials with different connectivity
        for trial in range(n_trials):
            # Set seed for this trial
            trial_seed = self.base_seed + trial
            set_random_seed(trial_seed, self.device)
            
            if trial == 0:
                print(f"\nRunning {n_trials} trial(s)...")
            print(f"  Trial {trial + 1}/{n_trials}: Creating circuit with seed {trial_seed}...")
            
            # Create NEW circuit for this trial with fresh connectivity
            self.circuit = self._create_circuit(trial_seed)
            
            # Run single trial
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
                plot_activity=(plot_activity and trial == 0),  # Only plot first trial
                trial_index=trial
            )
            
            all_trial_results.append(trial_result)
            
            # Clean up to free memory
            del self.circuit
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Aggregate results across trials
        aggregated_results = self._aggregate_trial_results(all_trial_results, n_trials)
        
        print(f"Completed {n_trials} trial(s) with different connectivity")
        
        return aggregated_results
    
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
        
        # Create opsin expression
        opsin = self.create_opsin_expression(target_population)
        target_positions = self.circuit.layout.positions[target_population]

        # Calculate direct optogenetic activation
        activation_prob = opsin.calculate_activation(target_positions, light_intensity)
        
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
        
        # Run simulation
        for t in range(n_steps):
            current_time = t * dt

            direct_activation = {}
            if (t >= stim_start_step) and (t < stim_end_step):
                # Convert to strong current injection
                direct_activation[target_population] = activation_prob * opsin_current
                
            # Calculate MEC external drive (dentate spikes + baseline)
            external_drive = {}
            mec_drive = torch.ones(self.circuit_params.n_mec, device=self.device) * mec_current
            
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
            fig, _ = vis.plot_activity_patterns(
                activity_trace_cpu,
                save_path=f"protocol/DG_{target_population}_stimulation_activity_{light_intensity}_trial{trial_index}.png"
            )
            plt.close(fig)
                
        return {
            'time': time_cpu,
            'activity_trace': activity_trace_cpu,
            'opsin_expression': opsin.expression_levels.cpu(),
            'target_positions': target_positions.cpu(),
            'dentate_spike_times': ds_times,
            'layout': self.circuit.layout,
            'connectivity': self.circuit.connectivity
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
        
        return {
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
                               stim_start: float = 1000.0,
                               stim_duration: float = 2000.0,
                               plot_activity: bool = True,
                               device: Optional[torch.device] = None,
                               n_trials: int = 1,
                               base_seed: int = 42):
    """
    Compare PV vs SST stimulation with anatomical connectivity
    
    UPDATED: Now supports multi-trial averaging
    
    Args:
        optimization_json_file: Path to optimization results (optional)
        intensities: List of light intensities to test
        mec_current: MEC drive current (pA)
        opsin_current: Optogenetic current (pA)
        stim_start: When to start stimulation (ms)
        plot_activity: Whether to plot activity traces
        device: Device to run on (None for auto-detect)
        n_trials: Number of trials to average (default: 1)
        base_seed: Base random seed for reproducibility (default: 42)
        
    Returns:
        results: Dict with averaged results
        connectivity_analysis: Connectivity pattern analysis
        conductance_analysis: Conductance pattern analysis
    """
    
    if device is None:
        device = get_default_device()
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    experiment = OptogeneticExperiment(
        circuit_params, synaptic_params, opsin_params, 
        optimization_json_file=optimization_json_file,
        device=device,
        base_seed=base_seed
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
                n_trials=n_trials
            )
            
            # Analyze network effects (using mean across trials)
            time = result['time']
            activity_mean = result['activity_trace_mean']
            activity_std = result['activity_trace_std']
            opsin_expression_mean = result['opsin_expression_mean']
            
            baseline_mask = (time >= 150) & (time < stim_start)
            stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
            
            analysis = {}
            analysis['opsin_expression_mean'] = opsin_expression_mean
            analysis['n_trials'] = n_trials
            
            for pop in ['gc', 'mc', 'pv', 'sst']:
                baseline_rate = torch.mean(activity_mean[pop][:, baseline_mask], dim=1)
                stim_rate = torch.mean(activity_mean[pop][:, stim_mask], dim=1)
                
                # Also calculate std across time for each neuron
                baseline_std_time = torch.std(activity_mean[pop][:, baseline_mask], dim=1)
                stim_std_time = torch.std(activity_mean[pop][:, stim_mask], dim=1)
                
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
    
    return results, conn_analysis, conductance_analysis


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
                              expression_threshold: float = 0.1) -> Dict:
    """Extract mask for non-opsin and opsin expressing cells."""
    expressing_mask = opsin_expression >= expression_threshold
    return expressing_mask


def plot_comparative_experiment_results(results: Dict, conn_analysis: Dict,
                                        stimulation_level: float = 1.0,
                                        save_path: Optional[str] = None) -> None:
    """
    Create visualizations from comparative experiment results
    
    UPDATED: Now handles multi-trial statistics with error bars
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
        print(f"opsin_expression {target} {stimulation_level}: {opsin_expression}")
        
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
        print(f"baseline_rates = {baseline_rates}")
        print(f"intercept = {intercept}")
        line = slope * baseline_rates + intercept
        ax.plot(baseline_rates, line, 'r--', alpha=0.8, linewidth=2, 
               label=f'Fit (R={r_value:.2f})')
        
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
        main_title += f' (Average of {n_trials} Trials with Different Connectivity)'
    else:
        main_title += ' (Representative Single Trial)'
        
    plt.suptitle(main_title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        trial_suffix = f'_n{n_trials}trials' if has_multitrial else '_single_trial'
        plt.savefig(f"{save_path}/DG_comparative_experiment{trial_suffix}.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/DG_comparative_experiment{trial_suffix}.png", 
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
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--mec-current', type=float, default=40.0,
                        help='MEC drive current in pA (default: 40.0)')
    parser.add_argument('--opsin-current', type=float, default=200.0,
                       help='Optogenetic current in pA (default: 200.0)')
    
    args = parser.parse_args()
    
    # Auto-detect device or use specified
    if args.device is None:
        device = get_default_device()
    else:
        device = torch.device(args.device)
        
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
        intensities=[0.5, 1.0, 2.0],
        mec_current=args.mec_current,
        opsin_current=args.opsin_current,
        device=device,
        n_trials=args.n_trials,
        base_seed=args.base_seed
    )

    # Print results with multi-trial statistics
    mec_conn = connectivity_analysis['mec_connectivity']
    print(f"\nMEC -> PV connections: {mec_conn['mec_to_pv']} ({mec_conn['pv_fraction']:.3f})")
    print(f"MEC -> GC connections: {mec_conn['mec_to_gc']} ({mec_conn['gc_fraction']:.3f})")
    print(f"MEC -> MC connections: {mec_conn['mec_to_mc']}")
    print(f"MEC -> SST connections: {mec_conn['mec_to_sst']}")
    
    for target in ['pv', 'sst']:
        print(f"\n{target.upper()} Stimulation Results (Average of {args.n_trials} trials):")
        print("-" * 50)
        
        for intensity in [0.5, 1.0, 2.0]:
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
                    print(f"    Excited: {excited:.2f} ± {excited_std:.2f}")
                    print(f"    Inhibited: {inhibited:.2f}")
                    print(f"    Rate: {mean_baseline_rate:.2f} -> {mean_stim_rate:.2f} Hz")
                    print(f"    Change: {mean_change:.3f} +/- {mean_change_std:.3f} Hz")

    # Plot results with multi-trial statistics
    plot_comparative_experiment_results(results, connectivity_analysis, save_path="protocol")
