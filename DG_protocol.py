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
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dendritic_somatic_transfer import (
    dendritic_somatic_transfer,
    get_cell_type_parameters,
    DendriticParameters
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

class OpsinExpression:
    """Handle heterogeneous opsin expression"""
    
    def __init__(self, params: OpsinParams, n_cells: int):
        self.params = params
        self.n_cells = n_cells
        self.expression_levels = self._generate_expression()
    
    def _generate_expression(self) -> Tensor:
        """Generate log-normal distribution of expression levels"""
        # Log-normal distribution
        log_mean = torch.log(torch.tensor(self.params.expression_mean))
        log_std = self.params.expression_std

        normal_samples = torch.normal(log_mean, log_std, size=(self.n_cells,))
        expression = torch.exp(normal_samples)

        # Some cells have no expression (transduction failure)
        no_expression = torch.rand(self.n_cells) < self.params.failure_rate
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
        activation = torch.where(self.expression_levels == 0, torch.tensor(0.0), activation)
        
        return activation


class OptogeneticExperiment:
    """Simulate optogenetic stimulation experiments"""
    
    def __init__(self, circuit_params: CircuitParams,
                 synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 optimization_json_file=None
                 ):
        self.circuit_params = circuit_params
        self.opsin_params = opsin_params
        self.synaptic_params = synaptic_params
        if optimization_json_file is not None:
            success = self._load_optimization_results(optimization_json_file)
            if success:
                print(f"OptogeneticExperiment initialized with optimized parameters from {optimization_json_file}")
            else:
                print(f"Failed to load optimization file. Using default parameters.")

        self.circuit = DentateCircuit(self.circuit_params, self.synaptic_params, self.opsin_params).to(device)


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
            connection_modulation=connection_modulation  # Use optimized values
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
        return OpsinExpression(self.opsin_params, n_cells)
    
    def simulate_stimulation(self, 
                             target_population: str,
                             light_intensity: float,
                             duration: float = 1550.0,
                             stim_start: float = 550.0,
                             mec_current: float = 100.0, # MEC drive current in pA
                             opsin_current: float = 100.0,
                             include_dentate_spikes: bool = False,
                             ds_times: Optional[List[float]] = None,
                             plot_activity: bool = False) -> Dict:
        """Simulate optogenetic stimulation experiment with MEC drive"""

        vis = None
        if plot_activity:
            vis = DGCircuitVisualization(self.circuit)
        
        # Reset circuit state
        self.circuit.reset_state()
        self.circuit.reset_state()
        
        # Create opsin expression
        opsin = self.create_opsin_expression(target_population)
        target_positions = self.circuit.layout.positions[target_population]

        # Calculate direct optogenetic activation
        activation_prob = opsin.calculate_activation(target_positions, light_intensity)
        
        # Simulation parameters
        dt = self.circuit_params.dt
        n_steps = int(duration / dt)
        stim_start_step = int(stim_start / dt)
        
        # Storage for results
        time = torch.arange(n_steps) * dt
        activity_trace = {
            'gc': torch.zeros(self.circuit_params.n_gc, n_steps),
            'mc': torch.zeros(self.circuit_params.n_mc, n_steps),
            'pv': torch.zeros(self.circuit_params.n_pv, n_steps),
            'sst': torch.zeros(self.circuit_params.n_sst, n_steps),
            'mec': torch.zeros(self.circuit_params.n_mec, n_steps)
        }
        
        # Generate dentate spike times (random occurrences)
        if include_dentate_spikes:
            if ds_times is None:
                ds_times = self._generate_dentate_spike_times(duration, baseline_rate=0.5)  # 0.5 Hz
        else:
            ds_times = []


        print(f"opsin_current = {opsin_current}")
        
        # Run simulation
        for t in tqdm.tqdm(range(n_steps)):
            current_time = t * dt

            direct_activation = {}
            if t >= stim_start_step:
                # Convert to strong current injection
                direct_activation[target_population] = activation_prob * opsin_current
                
            # Calculate MEC external drive (dentate spikes + baseline)
            external_drive = {}
            mec_drive = torch.ones(self.circuit_params.n_mec) * mec_current
            
            # Add dentate spike drive
            for ds_time in ds_times:
                if abs(current_time - ds_time) < 50:  # 50ms window around DS
                    # Gaussian profile around DS peak
                    ds_strength = 5.0 * torch.tensor(np.exp(-((current_time - ds_time) / 10.0) ** 2))

                    mec_drive += ds_strength
            
            external_drive['mec'] = mec_drive
            
            # Update circuit
            current_activity = self.circuit(direct_activation, external_drive)
            
            # Store activity
            for pop in activity_trace:
                #print(f"{pop} activity: {current_activity[pop].cpu()}")
                activity_trace[pop][:, t] = current_activity[pop].cpu()

        if vis:
            fig, _ = vis.plot_activity_patterns(activity_trace,
                                                save_path = f"DG_{target_population}_stimulation_activity_{light_intensity}.png")
            plt.close(fig)
                
        return {
            'time': time.cpu(),
            'activity_trace': activity_trace,
            'opsin_expression': opsin.expression_levels.cpu(),
            'target_positions': target_positions.cpu(),
            'dentate_spike_times': ds_times,
            'layout': self.circuit.layout,
            'connectivity': self.circuit.connectivity
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
    """Analyze the anatomical connectivity patterns including MEC"""
    layout = experiment.circuit.layout
    # Access conductance matrices through the new structure
    conductance_matrices = experiment.circuit.connectivity.conductance_matrices
    
    analysis = {}
    
    # Helper function to get connectivity matrix by name
    def get_connectivity(conn_name: str) -> torch.Tensor:
        """Get connectivity matrix for a specific connection"""
        if conn_name in conductance_matrices:
            return conductance_matrices[conn_name].connectivity
        else:
            return torch.zeros((1, 1))  # Return empty matrix if connection doesn't exist
    
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
                gc_mc_distances.extend(distances.tolist())
        
        # Connected PVs
        if i < gc_pv_conn.size(0):
            connected_pv = torch.where(gc_pv_conn[i] > 0)[0]
            if len(connected_pv) > 0:
                gc_pos = layout.positions['gc'][i:i+1]
                pv_pos = layout.positions['pv'][connected_pv]
                distances = torch.norm(gc_pos - pv_pos, dim=1)
                gc_pv_distances.extend(distances.tolist())
            
        # Connected SSTs
        if i < gc_sst_conn.size(0):
            connected_sst = torch.where(gc_sst_conn[i] > 0)[0]
            if len(connected_sst) > 0:
                gc_pos = layout.positions['gc'][i:i+1]
                sst_pos = layout.positions['sst'][connected_sst]
                distances = torch.norm(gc_pos - sst_pos, dim=1)
                gc_sst_distances.extend(distances.tolist())
    
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
                mc_gc_distances.extend(distances.tolist())
        
        # Connected SSTs
        if i < mc_sst_conn.size(0):
            connected_sst = torch.where(mc_sst_conn[i] > 0)[0]
            if len(connected_sst) > 0:
                mc_pos = layout.positions['mc'][i:i+1]
                sst_pos = layout.positions['sst'][connected_sst]
                distances = torch.norm(mc_pos - sst_pos, dim=1)
                mc_sst_distances.extend(distances.tolist())
    
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


def analyze_mec_asymmetry_effects(experiment: OptogeneticExperiment) -> Dict:
    """Analyze how MEC -> PV (but not SST) asymmetry affects circuit dynamics"""
    
    # Reset circuit state
    experiment.circuit.reset_state()

    # Simulation parameters
    n_steps = 2000
    dt = 0.1
    time = torch.arange(n_steps) * dt
    mec_drive_time = 100.0  # Start MEC drive at 100ms
    
    activity_trace = {
        'gc': torch.zeros(experiment.circuit_params.n_gc, n_steps),
        'mc': torch.zeros(experiment.circuit_params.n_mc, n_steps), 
        'pv': torch.zeros(experiment.circuit_params.n_pv, n_steps),
        'sst': torch.zeros(experiment.circuit_params.n_sst, n_steps),
        'mec': torch.zeros(experiment.circuit_params.n_mec, n_steps)
    }
    
    # Store conductance information
    conductance_stats = analyze_conductance_patterns(experiment)
    
    for t in range(n_steps):
        current_time = t * dt
        
        # MEC drive (simulating dentate spike)
        external_drive = {}
        if mec_drive_time <= current_time <= mec_drive_time + 50.0:  # 50ms MEC drive
            mec_drive = torch.ones(experiment.circuit_params.n_mec) * 500.0  # Strong drive (pA)
        else:
            mec_drive = torch.ones(experiment.circuit_params.n_mec) * 50.0   # Baseline (pA)
        
        external_drive['mec'] = mec_drive
        
        # No optogenetic stimulation
        direct_activation = {}
        
        # Update circuit
        current_activity = experiment.circuit(direct_activation, external_drive)
        
        # Store activity
        for pop in activity_trace:
            activity_trace[pop][:, t] = current_activity[pop].cpu()
    
    # Analyze temporal dynamics
    baseline_mask = time < mec_drive_time
    response_mask = (time >= mec_drive_time) & (time <= (mec_drive_time + 50.0))
    
    analysis = {'conductance_stats': conductance_stats}
    
    for pop in ['gc', 'mc', 'pv', 'sst']:
        baseline_rate = torch.mean(activity_trace[pop][:, baseline_mask], dim=1)
        response_rate = torch.mean(activity_trace[pop][:, response_mask], dim=1)
        
        # Calculate response latency (time to peak)
        mec_drive_start_idx = int(mec_drive_time / dt)
        mec_drive_end_idx = int((mec_drive_time + 50.0) / dt)
        pop_trace = torch.mean(activity_trace[pop], dim=0)
        baseline_mean = torch.mean(pop_trace[:mec_drive_start_idx])
        baseline_std = torch.std(pop_trace[:mec_drive_start_idx])
        
        if torch.max(pop_trace[mec_drive_start_idx:mec_drive_end_idx]) > baseline_mean + 2*baseline_std:
            peak_idx = torch.argmax(pop_trace[mec_drive_start_idx:mec_drive_end_idx]) + mec_drive_start_idx
            latency = (peak_idx * dt) - mec_drive_time  # Relative to MEC drive onset
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
    mec_baseline = torch.mean(activity_trace['mec'][:, baseline_mask], dim=1)
    mec_response = torch.mean(activity_trace['mec'][:, response_mask], dim=1)
    
    analysis['mec_response'] = {
        'baseline_mean': torch.mean(mec_baseline).item(),
        'response_mean': torch.mean(mec_response).item(),
        'drive_effectiveness': torch.mean(mec_response - mec_baseline).item() / 450.0  # Relative to drive increase
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
    
def analyze_mec_asymmetry_effects(experiment: OptogeneticExperiment) -> Dict:
    """Analyze how MEC -> PV (but not SST) asymmetry affects circuit dynamics"""
    
    # Simulate with strong MEC drive (dentate spike)
    circuit_params = experiment.circuit_params
    opsin_params = experiment.opsin_params
    
    # Reset and set strong MEC drive
    experiment.circuit.reset_state()

    # Simulate 200ms with strong MEC drive at t=100ms
    n_steps = 2000
    dt = 0.1
    
    time = torch.arange(n_steps) * dt
    mec_drive_time = 100.0  # Start MEC drive at 100ms
    
    activity_trace = {
        'gc': torch.zeros(circuit_params.n_gc, n_steps),
        'mc': torch.zeros(circuit_params.n_mc, n_steps), 
        'pv': torch.zeros(circuit_params.n_pv, n_steps),
        'sst': torch.zeros(circuit_params.n_sst, n_steps),
        'mec': torch.zeros(circuit_params.n_mec, n_steps)
    }
    
    for t in range(n_steps):
        current_time = t * dt
        
        # MEC drive (simulating dentate spike)
        external_drive = {}
        if mec_drive_time <= current_time <= mec_drive_time + 50.0:  # 50ms MEC drive
            mec_drive = torch.ones(circuit_params.n_mec) * 10.0  # Strong drive
        else:
            mec_drive = torch.ones(circuit_params.n_mec) * 0.1   # Baseline
        
        external_drive['mec'] = mec_drive
        
        # No optogenetic stimulation
        direct_activation = {}
        
        # Update circuit
        current_activity = experiment.circuit(direct_activation, external_drive)
        
        # Store activity
        for pop in activity_trace:
            activity_trace[pop][:, t] = current_activity[pop].cpu()
    
    # Analyze temporal dynamics
    baseline_mask = time < mec_drive_time
    response_mask = (time >= mec_drive_time) & (time <= (mec_drive_time + 50.0))
    
    analysis = {}
    
    for pop in ['gc', 'mc', 'pv', 'sst']:
        baseline_rate = torch.mean(activity_trace[pop][:, baseline_mask], dim=1)
        response_rate = torch.mean(activity_trace[pop][:, response_mask], dim=1)
        
        # Calculate response latency (time to peak)
        mec_drive_start_idx = int(mec_drive_time / dt)
        mec_drive_end_idx = int((mec_drive_time + 50.0) / dt)
        pop_trace = torch.mean(activity_trace[pop], dim=0)
        if torch.max(pop_trace[mec_drive_start_idx:mec_drive_end_idx]) > torch.mean(pop_trace[:mec_drive_start_idx]) + 2*torch.std(pop_trace[:mec_drive_start_idx]):
            peak_idx = torch.argmax(pop_trace[mec_drive_start_idx:mec_drive_end_idx]) + mec_drive_start_idx
            latency = (peak_idx * 0.1) - mec_drive_time  # Relative to MEC drive onset
        else:
            latency = float('nan')
        
        analysis[f'{pop}_response'] = {
            'baseline_mean': torch.mean(baseline_rate).item(),
            'response_mean': torch.mean(response_rate).item(),
            'response_latency': latency,
            'activated_fraction': torch.mean(response_rate > baseline_rate + torch.std(baseline_rate)).item()
        }
    
    # MEC analysis
    mec_baseline = torch.mean(activity_trace['mec'][:, baseline_mask], dim=1)
    mec_response = torch.mean(activity_trace['mec'][:, response_mask], dim=1)
    
    analysis['mec_response'] = {
        'baseline_mean': torch.mean(mec_baseline).item(),
        'response_mean': torch.mean(mec_response).item(),
        'drive_effectiveness': torch.mean(mec_response).item() / 10.0  # Relative to drive strength
    }
    
    # Key asymmetry analysis
    pv_response_strength = analysis['pv_response']['response_mean'] - analysis['pv_response']['baseline_mean']
    sst_response_strength = analysis['sst_response']['response_mean'] - analysis['sst_response']['baseline_mean']
    
    analysis['asymmetry_effect'] = {
        'pv_direct_response': pv_response_strength,
        'sst_indirect_response': sst_response_strength,
        'asymmetry_ratio': pv_response_strength / (sst_response_strength + 1e-6),
        'pv_latency': analysis['pv_response']['response_latency'],
        'sst_latency': analysis['sst_response']['response_latency']
    }
    
    return analysis

def run_comparative_experiment(optimization_json_file=None,
                               intensities = [0.5, 1.0, 2.0],
                               mec_current = 100.0,
                               opsin_current = 100.0,
                               stim_start = 550,
                               plot_activity=True):
    """Compare PV vs SST stimulation with anatomical connectivity"""
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    experiment = OptogeneticExperiment(circuit_params, synaptic_params, opsin_params, optimization_json_file=optimization_json_file)
    
    # Analyze connectivity patterns
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
            print(f"  Intensity: {intensity}")
            result = experiment.simulate_stimulation(target, intensity,
                                                     stim_start=stim_start,
                                                     duration = stim_start + 1000.0,
                                                     plot_activity = plot_activity,
                                                     mec_current = mec_current,
                                                     opsin_current = opsin_current)
            
            # Analyze network effects
            time = result['time']
            activity = result['activity_trace']
            opsin_expression = result['opsin_expression']
            
            baseline_mask = (time >= 150) & (time < stim_start)  # Pre-stimulation
            stim_mask = time >= stim_start     # During stimulation
            
            analysis = {}
            analysis['opsin_expression'] = opsin_expression
            for pop in ['gc', 'mc', 'pv', 'sst']:

                baseline_rate = torch.mean(activity[pop][:, baseline_mask], dim=1)
                stim_rate = torch.mean(activity[pop][:, stim_mask], dim=1)
                if pop == target:
                    #baseline_rates = activity[pop][:, baseline_mask].view(-1)
                    #stim_rates = activity[pop][:, stim_mask].view(-1)
                    analysis[f'{pop}_stim_rates'] = stim_rate.numpy()
                    analysis[f'{pop}_baseline_rates'] = baseline_rate.numpy()
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
    """
    Extract mask for non-opsin and opson expressing cells.
    
    Args:
        target_pop: Target population name
        opsin_expression: Opsin expression levels
        expression_threshold: Threshold for considering cells as expressing
        
    Returns:
        Array with opsin/non-opsin flag values.
    """
    expressing_mask = opsin_expression >= expression_threshold
    
    return expressing_mask


def plot_comparative_experiment_results(results: Dict, conn_analysis: Dict,
                                        stimulation_level: float = 1.0,
                                        save_path: Optional[str] = None) -> None:
    """
    Create visualizations from comparative experiment results.
    NOTE: This shows results from a single trial
    
    Args:
        results: Results from run_comparative_experiment()
        conn_analysis: Connectivity analysis results
        save_path: Optional path to save figures
    """
    
    # Define colors matching the paper
    colors = {
        'pv': '#FF6B9D',   # Pink
        'sst': '#45B7D1',  # Blue  
        'gc': '#96CEB4',   # Green
        'mc': '#FFEAA7',   # Yellow
    }
    
    # Create summary figure
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Firing ratio bar plots
    ax1 = plt.subplot(3, 4, (1, 2))
    
    targets = ['pv', 'sst']
    populations = ['gc', 'mc', 'pv', 'sst']
    
    bar_data = []
    bar_labels = []
    bar_colors = []
    
    for target in targets:
        for pop in populations:
            if pop != target and f'{pop}_mean_change' in results[target][stimulation_level]:
                baseline_rate = results[target][stimulation_level][f'{pop}_mean_baseline_rate']
                stim_rate = results[target][stimulation_level][f'{pop}_mean_stim_rate']
                
                # Calculate modulation ratio
                if baseline_rate > 0:
                    ratio = np.log2(stim_rate / baseline_rate)
                    bar_data.append(ratio)
                    bar_labels.append(f'{target.upper()}→{pop.upper()}')
                    bar_colors.append(colors[pop])
    
    if bar_data:
        x_pos = np.arange(len(bar_labels))
        bars = ax1.bar(x_pos, bar_data, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, bar_data)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel(r'Modulation Ratio ($\log_2$)', fontsize=11)
    ax1.set_title('Firing Rate Modulation (Single Trial)', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Panel B: Network effects summary
    ax2 = plt.subplot(3, 4, (3, 4))
    
    effect_data = []
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
    
    if effect_data:
        effect_array = np.array(effect_data)
        x_pos = np.arange(len(effect_labels))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, effect_array[:, 0], width, 
               label='Excited', color='red', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x_pos + width/2, effect_array[:, 1], width, 
               label='Inhibited', color='blue', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom',
                        fontsize=8)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(effect_labels, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Fraction of Cells', fontsize=11)
        ax2.set_title('Network Effects (Single Trial)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_axisbelow(True)
    
    # Panel C: Connectivity analysis
    ax3 = plt.subplot(3, 4, (5, 6))
    
    mec_conn = conn_analysis['mec_connectivity']
    conn_data = [
        mec_conn['mec_to_pv'],
        mec_conn['mec_to_gc'], 
        mec_conn['mec_to_sst']
    ]
    conn_labels = ['MEC -> PV', 'MEC -> GC', 'MEC -> SST']
    conn_colors = [colors['pv'], colors['gc'], colors['sst']]
    
    bars = ax3.bar(conn_labels, conn_data, color=conn_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
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
    
    # Panel D: Firing rate changes bar plot
    ax4 = plt.subplot(3, 4, (7, 8))
    
    change_data = []
    change_labels = []
    change_colors = []
    
    for target in targets:
        for pop in ['gc', 'mc']:
            if f'{pop}_mean_change' in results[target][stimulation_level]:
                mean_change = results[target][stimulation_level][f'{pop}_mean_change']
                
                change_data.append(mean_change)
                change_labels.append(f'{target.upper()}→{pop.upper()}')
                change_colors.append(colors['pv'] if target == 'pv' else colors['sst'])
    
    if change_data:
        bars = ax4.bar(range(len(change_labels)), change_data, 
                      color=change_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
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
        ax4.set_title('Mean Rate Changes (Single Trial)', fontsize=12, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_axisbelow(True)
    
    # Panel E: Scatter plots showing correlation (single trial)
    for i, target in enumerate(targets):
        ax = plt.subplot(3, 4, 9 + i * 2)
        
        opsin_expression = results[target][stimulation_level]['opsin_expression']
        stim_rates = results[target][stimulation_level][f'{target}_stim_rates'][opsin_expression <= 0.2]
        baseline_rates = results[target][stimulation_level][f'{target}_baseline_rates'][opsin_expression <= 0.2]
        
        ax.scatter(baseline_rates, stim_rates, c=colors[target], alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Add correlation line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(baseline_rates, stim_rates)
        line = slope * baseline_rates + intercept
        ax.plot(baseline_rates, line, 'r--', alpha=0.8, linewidth=2, label=f'Fit (R={r_value:.2f})')
        
        # Identity line
        max_rate = max(np.max(baseline_rates), np.max(stim_rates))
        ax.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, linewidth=1.5, label='Identity')
        
        ax.set_xlabel('Baseline Rate (Hz)', fontsize=10)
        ax.set_ylabel('Stimulation Rate (Hz)', fontsize=10)
        ax.set_title(f'{target.upper()} Stimulation\n(Non-expressing cells)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)
    
    # Panel F: Summary statistics
    ax6 = plt.subplot(3, 4, 12)
    ax6.axis('off')
    
    summary_text = "Single Trial Summary\n"
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
                summary_text += f"  {pop.upper()}: {excited:.1%} excited\n"
        summary_text += "\n"
    
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, va='top', ha='left',
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Dentate Gyrus Interneuron Effects (Representative Single Trial)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(f"{save_path}/DG_comparative_experiment_single_trial.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/DG_comparative_experiment_single_trial.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Single trial results (use statistical_testing.py for robust analysis)")
    print("=" * 70)
    print(f"MEC -> PV connections: {mec_conn['mec_to_pv']} ({mec_conn['pv_fraction']:.3f})")
    print(f"MEC -> SST connections: {mec_conn['mec_to_sst']} ")
    
    for target in targets:
        print(f"\n{target.upper()} stimulation effects:")
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if pop != target and f'{pop}_excited' in results[target][stimulation_level]:
                excited = results[target][stimulation_level][f'{pop}_excited']
                inhibited = results[target][stimulation_level][f'{pop}_inhibited']
                print(f"  {pop.upper()}: {excited:.1%} excited, {inhibited:.1%} inhibited")
    
    print("\n" + "=" * 70)
    print("NOTE: This is a single trial. Run statistical_testing.py for")
    print("multi-trial analysis with statistical inference.")
    print("=" * 70)

def analyze_disinhibition_effects(experiment: OptogeneticExperiment,
                                  target_population: str, 
                                  light_intensity: float,
                                  mec_current = 40.0,
                                  opsin_current = 100.0) -> Dict:
    """Analyze disinhibition mechanisms with fixed connectivity access"""
    
    # Store original synaptic parameters
    original_synaptic_params = experiment.synaptic_params
    
    # Run simulation with full network
    result_full = experiment.simulate_stimulation(target_population, light_intensity,
                                                  mec_current = mec_current,
                                                  opsin_current = opsin_current)
    
    # Create modified synaptic parameters with reduced inhibition
    modified_synaptic_params = PerConnectionSynapticParams(
        # Keep excitatory conductances the same
        ampa_g_mean=original_synaptic_params.ampa_g_mean,
        ampa_g_std=original_synaptic_params.ampa_g_std,
        ampa_g_min=original_synaptic_params.ampa_g_min,
        ampa_g_max=original_synaptic_params.ampa_g_max,
        
        # Reduce inhibitory conductances
        gaba_g_mean=original_synaptic_params.gaba_g_mean * 0.1,  # 90% reduction
        gaba_g_std=original_synaptic_params.gaba_g_std * 0.1,
        gaba_g_min=original_synaptic_params.gaba_g_min * 0.1,
        gaba_g_max=original_synaptic_params.gaba_g_max * 0.1,
        
        # Keep other parameters
        distribution=original_synaptic_params.distribution,
        connection_modulation=original_synaptic_params.connection_modulation,
        e_exc=original_synaptic_params.e_exc,
        e_inh=original_synaptic_params.e_inh,
        v_rest=original_synaptic_params.v_rest,
        tau_ampa=original_synaptic_params.tau_ampa,
        tau_gaba=original_synaptic_params.tau_gaba,
        tau_nmda=original_synaptic_params.tau_nmda
    )
    
    # Create new experiment with reduced inhibition
    experiment_no_inh = OptogeneticExperiment(
        experiment.circuit_params, 
        modified_synaptic_params,
        experiment.opsin_params
    )
    
    result_no_inhibition = experiment_no_inh.simulate_stimulation(target_population, light_intensity)
    
    time = result_full['time']
    baseline_mask = time < 100
    stim_mask = time >= 100
    
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

def test_disinhibition_hypothesis(optimization_json_file=None,
                                  mec_current = 100.0,
                                  opsin_current = 100.0):
    """Test whether disinhibition mechanisms explain paradoxical excitation"""
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    experiment = OptogeneticExperiment(circuit_params, synaptic_params, opsin_params,
                                       optimization_json_file=optimization_json_file)
    
    print("Testing Disinhibition Hypothesis")
    print("=" * 40)
    
    for target in ['pv', 'sst']:
        print(f"\n{target.upper()} Stimulation:")
        print("-" * 20)
        
        analysis = analyze_disinhibition_effects(experiment, target, 1.0,
                                                 mec_current = mec_current,
                                                 opsin_current = opsin_current)
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            if f'{pop}_paradoxical_excitation' in analysis:
                result = analysis[f'{pop}_paradoxical_excitation']
                
                print(f"{pop.upper()}:")
                print(f"  With inhibition: {result['with_inhibition']} cells excited")
                print(f"  Reduced inhibition: {result['without_inhibition']} cells excited") 
                print(f"  Disinhibition-dependent: {result['disinhibition_dependent']} cells")
                print(f"  Mean change: {result['mean_change_full']:.3f} -> {result['mean_change_no_inh']:.3f}")
                print(f"  Change variability: {result['std_change_full']:.3f} -> {result['std_change_no_inh']:.3f}")

                
    
def run_protocol():
    """Main experimental protocol"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create experiment
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    experiment = OptogeneticExperiment(circuit_params, synaptic_params, opsin_params)
    
    # Analyze MEC asymmetry effects with analysis
    print("\nMEC asymmetry analysis")
    print("=" * 60)
    
    mec_analysis = analyze_mec_asymmetry_effects(experiment)
    
    print("MEC Drive Response (simulating dentate spike):")
    print(f"  MEC activation: {mec_analysis['mec_response']['response_mean']:.2f} Hz")
    print(f"  PV response: {mec_analysis['pv_response']['response_mean']:.2f} Hz "
          f"(latency: {mec_analysis['pv_response']['response_latency']:.1f} ms)")
    print(f"  SST response: {mec_analysis['sst_response']['response_mean']:.2f} Hz "
          f"(latency: {mec_analysis['sst_response']['response_latency']:.1f} ms)")
    print(f"  GC response: {mec_analysis['gc_response']['response_mean']:.2f} Hz")
    
    asymmetry = mec_analysis['asymmetry_effect']
    print(f"\nKey Asymmetry:")
    print(f"  PV direct response: {asymmetry['pv_direct_response']:.3f}")
    print(f"  SST indirect response: {asymmetry['sst_indirect_response']:.3f}")
    print(f"  Asymmetry ratio: {asymmetry['asymmetry_ratio']:.1f}")
    
    if asymmetry['asymmetry_ratio'] > 2.0:
        print("  - MEC -> PV asymmetry creates differential activation")
    else:
        print("  - Asymmetry effect may be too weak")
    
    # Print conductance statistics
    print("\nConductance Statistics:")
    print("-" * 30)
    for conn_name, stats in mec_analysis['conductance_stats'].items():
        if stats['n_connections'] > 0:
            print(f"{conn_name}: {stats['mean_conductance']:.3f} +/- {stats['std_conductance']:.3f} nS "
                  f"(CV={stats['cv_conductance']:.2f}, n={stats['n_connections']})")
    
    # Run comparative experiment
    results, connectivity_analysis, conductance_analysis = run_comparative_experiment(mec_current = 40.0,
                                                                                      opsin_current = 100.0)
    
    print("\n" + "="*60)
    print("Experimental results")
    print("="*60)
    
    mec_conn = connectivity_analysis['mec_connectivity']
    print(f"MEC -> PV connections: {mec_conn['mec_to_pv']} ({mec_conn['pv_fraction']:.3f})")
    print(f"MEC -> GC connections: {mec_conn['mec_to_gc']} ({mec_conn['gc_fraction']:.3f})")
    print(f"MEC -> MC connections: {mec_conn['mec_to_mc']}")
    print(f"MEC -> SST connections: {mec_conn['mec_to_sst']}")
    
    for target in ['pv', 'sst']:
        print(f"\n{target.upper()} Stimulation Results:")
        print("-" * 30)
        
        for intensity in [0.5, 1.0, 2.0]:
            analysis = results[target][intensity]
            print(f"\nIntensity {intensity}:")
            
            for pop in ['gc', 'mc', 'pv', 'sst']:
                if f'{pop}_excited' in analysis:
                    excited = analysis[f'{pop}_excited']
                    inhibited = analysis[f'{pop}_inhibited'] 
                    mean_change = analysis[f'{pop}_mean_change']
                    mean_stim_rate = analysis[f'{pop}_mean_stim_rate']
                    mean_baseline_rate = analysis[f'{pop}_mean_baseline_rate']
                    print(f"  {pop.upper()}: {excited:.2f} excited, {inhibited:.2f} inhibited")
                    print(f"    Change: {mean_change:.3f} Hz "
                          f"({mean_baseline_rate:.2f} -> {mean_stim_rate:.2f} Hz)")
    
    plot_comparative_experiment_results(results, connectivity_analysis, save_path=".")
    
    print("\n" + "="*60)
    print("Disinhibition analysis")
    print("="*60)
    test_disinhibition_hypothesis()
    
    return results, connectivity_analysis, conductance_analysis, mec_analysis    

if __name__ == "__main__":
    print("PyTorch Dentate Gyrus Circuit with Anatomical Connectivity")
    print("=========================================================")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if len(sys.argv) > 1:
        optimization_json_file=sys.argv[1]
    else:
        optimization_json_file=None


    # Run comparative experiment
    results, connectivity_analysis, conductance_analysis = run_comparative_experiment(optimization_json_file=optimization_json_file,
                                                                                      intensities = [0.5, 1.0, 2.0],
                                                                                      mec_current=40.0,
                                                                                      opsin_current=200.0)


    mec_conn = connectivity_analysis['mec_connectivity']
    print(f"MEC -> PV connections: {mec_conn['mec_to_pv']} ({mec_conn['pv_fraction']:.3f})")
    print(f"MEC -> GC connections: {mec_conn['mec_to_gc']} ({mec_conn['gc_fraction']:.3f})")
    print(f"MEC -> MC connections: {mec_conn['mec_to_mc']}")
    print(f"MEC -> SST connections: {mec_conn['mec_to_sst']}")
    
    for target in ['pv', 'sst']:
        print(f"\n{target.upper()} Stimulation Results:")
        print("-" * 30)
        
        for intensity in [0.5, 1.0, 2.0]:
            analysis = results[target][intensity]
            print(f"\nIntensity {intensity}:")
            opsin_expression = analysis['opsin_expression']
            for pop in ['gc', 'mc', 'pv', 'sst']:
                if f'{pop}_excited' in analysis:
                    excited = analysis[f'{pop}_excited']
                    inhibited = analysis[f'{pop}_inhibited'] 
                    mean_change = analysis[f'{pop}_mean_change']
                    mean_stim_rate = analysis[f'{pop}_mean_stim_rate']
                    mean_baseline_rate = analysis[f'{pop}_mean_baseline_rate']
                    print(f"  {pop.upper()}: {excited:.2f} excited, {inhibited:.2f} inhibited, "
                          f"mean change from baseline: {mean_change:.3f}")
                    print(f"  {pop.upper()}: mean stim rate: {mean_stim_rate:.2f}, "
                          f"mean baseline rate: {mean_baseline_rate:.2f}")
                else:
                    opsin_stim_rates = analysis[f'{pop}_stim_rates'][opsin_expression > 0.2]
                    opsin_baseline_rates = analysis[f'{pop}_baseline_rates'][opsin_expression > 0.2]
                    non_opsin_stim_rates = analysis[f'{pop}_stim_rates'][opsin_expression <= 0.2]
                    non_opsin_baseline_rates = analysis[f'{pop}_baseline_rates'][opsin_expression <= 0.2]
                    print(f"  {pop.upper()}: {np.mean(opsin_stim_rates)} mean opsin stimulated rates, {np.mean(opsin_baseline_rates)} mean opsin baseline rates, ")
                    print(f"  {pop.upper()}: {np.mean(non_opsin_stim_rates)} mean non-opsin stimulated rates, {np.mean(non_opsin_baseline_rates)} mean non-opsin baseline rates, ")

    plot_comparative_experiment_results(results, connectivity_analysis, save_path="figures")
    
                    
    # Test disinhibition hypothesis
    #test_disinhibition_hypothesis(optimization_json_file=optimization_json_file,
    #                              mec_current=40.0,
    #                              opsin_current=200.0)
