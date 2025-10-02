#!/usr/bin/env python33
"""
Dentate Gyrus Circuit with Dendritic-Somatic Transfer Function

Integrates the biophysically realistic two-stage dendritic-somatic transfer 
function into the main dentate gyrus circuit model.
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, NamedTuple, List
import numpy as np
import matplotlib.pyplot as plt

from dendritic_somatic_transfer import (
    dendritic_somatic_transfer,
    get_cell_type_parameters,
    DendriticParameters
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SynapticStateManager:
    """Manages synaptic state variables with proper time constants"""
    
    def __init__(self, circuit_params, synaptic_params, connectivity_matrices):
        self.circuit_params = circuit_params
        self.synaptic_params = synaptic_params
        self.dt = circuit_params.dt
        
        # Initialize synaptic state variables for each connection type
        self.synaptic_states = {}
        
        for conn_name, cond_matrix in connectivity_matrices.items():
            n_pre, n_post = cond_matrix.connectivity.shape
            synapse_type = cond_matrix.synapse_type
            
            # Create state variables for this connection
            states = {
                'ampa_conductance': torch.zeros(n_pre, n_post),
                'gaba_conductance': torch.zeros(n_pre, n_post),
                'nmda_conductance': torch.zeros(n_pre, n_post),
            }
            
            self.synaptic_states[conn_name] = {
                'states': states,
                'synapse_type': synapse_type,
                'max_conductances': cond_matrix.conductances,
                'connectivity': cond_matrix.connectivity
            }
    
    def update_synaptic_states(self, activities: Dict[str, torch.Tensor]):
        """Update all synaptic state variables based on presynaptic firing rates"""
        
        # Calculate decay factors for each synapse type
        tensor_dt = torch.tensor(self.dt)
        ampa_decay = torch.exp(-tensor_dt / self.synaptic_params.tau_ampa)
        gaba_decay = torch.exp(-tensor_dt / self.synaptic_params.tau_gaba) 
        nmda_decay = torch.exp(-tensor_dt / self.synaptic_params.tau_nmda)
        
        for conn_name, syn_data in self.synaptic_states.items():
            parts = conn_name.split('_')
            pre_pop, post_pop = parts[0], parts[1]
            
            if pre_pop not in activities:
                continue
                
            pre_rates = activities[pre_pop]
            states = syn_data['states']
            max_conductances = syn_data['max_conductances']
            connectivity = syn_data['connectivity']
            synapse_type = syn_data['synapse_type']
            
            # Calculate synaptic input (firing rate * max conductance * connectivity)
            synaptic_input = pre_rates.unsqueeze(1) * max_conductances * connectivity
            
            if synapse_type == 'excitatory':
                # Update AMPA conductances
                states['ampa_conductance'] = (
                    ampa_decay * states['ampa_conductance'] + 
                    (1 - ampa_decay) * synaptic_input * (1 - self.synaptic_params.nmda_fraction)
                )
                
                # Update NMDA conductances 
                states['nmda_conductance'] = (
                    nmda_decay * states['nmda_conductance'] + 
                    (1 - nmda_decay) * synaptic_input * self.synaptic_params.nmda_fraction
                )
                
            elif synapse_type == 'inhibitory':
                # Update GABA conductances
                states['gaba_conductance'] = (
                    gaba_decay * states['gaba_conductance'] + 
                    (1 - gaba_decay) * synaptic_input
                )
    
    def get_total_conductances(self, conn_name: str) -> Dict[str, torch.Tensor]:
        """Get total conductances for a specific connection"""
        if conn_name not in self.synaptic_states:
            return {'ampa': torch.tensor(0.), 'gaba': torch.tensor(0.), 'nmda': torch.tensor(0.)}
            
        states = self.synaptic_states[conn_name]['states']
        
        return {
            'ampa': torch.sum(states['ampa_conductance'], dim=0),
            'gaba': torch.sum(states['gaba_conductance'], dim=0), 
            'nmda': torch.sum(states['nmda_conductance'], dim=0)
        }
    
    def reset_states(self):
        """Reset all synaptic states to zero"""
        for syn_data in self.synaptic_states.values():
            for state_var in syn_data['states'].values():
                state_var.zero_()


@dataclass
class PerConnectionSynapticParams:
    """Parameters for per-connection conductances"""
    
    # Base conductance parameters (mean values)
    ampa_g_mean: float = 0.2      # AMPA conductance mean (nS per connection)
    ampa_g_std: float = 0.04      # AMPA conductance standard deviation
    ampa_g_min: float = 0.01      # Minimum AMPA conductance
    ampa_g_max: float = 1.5       # Maximum AMPA conductance
    
    gaba_g_mean: float = 0.25     # GABA conductance mean (nS per connection)
    gaba_g_std: float = 0.04       # GABA conductance standard deviation  
    gaba_g_min: float = 0.01      # Minimum GABA conductance
    gaba_g_max: float = 1.5       # Maximum GABA conductance
    
    # Distribution type for conductance heterogeneity
    distribution: str = 'lognormal'  # 'normal', 'lognormal', 'gamma'
    
    # Connection-type specific modulation factors
    connection_modulation: Dict[str, float] = field(default_factory=lambda: {
        # Excitatory connections
        'mec_gc': 1.381,      # Strong perforant path
        'mec_pv': 0.5,      # Perforant path to PV
        'mc_gc': 4.549,       # Strong associational
        'mc_mc': 0.537,       # Moderate recurrent
        'gc_mc': 0.623,        # Strong mossy fiber collaterals
        'gc_pv': 1.767,       # Moderate excitation to PV
        'gc_sst': 1.366,      # Weaker excitation to SST
        'mc_pv': 2.119,       # Strong excitation to PV
        'mc_sst': 0.382,      # Standard excitation to SST
        
        # Inhibitory connections  
        'pv_gc': 0.928,       # Strong perisomatic inhibition
        'pv_mc': 1.574,       # Moderate inhibition of MC
        'pv_pv': 1.216,       # Very strong PV lateral inhibition
        'sst_gc': 0.626,      # Moderate dendritic inhibition
        'sst_mc': 0.463,      # Weak inhibition of MC
        'sst_pv': 1.204,      # Strong disinhibition
        'sst_sst': 2.647     # SST lateral inhibition
    })
    
    # Reversal potentials (mV)
    e_exc: float = 0.0              # Excitatory reversal
    e_inh: float = -70.0            # Inhibitory reversal
    v_rest: float = -70.0           # Resting potential
    
    # Synaptic time constants (ms)
    tau_ampa: float = 2.0
    tau_gaba: float = 10.0
    tau_nmda: float = 40.0

@dataclass
class ConductanceMatrix:
    """Container for per-connection conductance matrices"""
    
    # Connectivity matrices (binary)
    connectivity: torch.Tensor
    
    # Per-connection conductances (nS)  
    conductances: torch.Tensor
    
    # Connection type identifier
    connection_name: str
    synapse_type: str  # 'excitatory' or 'inhibitory'
    
    def __post_init__(self):
        """Validate that matrices have compatible shapes"""
        assert self.connectivity.shape == self.conductances.shape, \
            f"Connectivity {self.connectivity.shape} and conductances {self.conductances.shape} must have same shape"
        
        # Zero out conductances where there are no connections
        self.conductances = self.conductances * self.connectivity

def generate_conductance_distribution(n_connections: int, 
                                    mean_conductance: float,
                                    std_conductance: float,
                                    min_conductance: float,
                                    max_conductance: float,
                                    distribution: str = 'lognormal',
                                    connection_modulation: float = 1.0) -> torch.Tensor:
    """
    Generate heterogeneous conductance values for connections
    
    Args:
        n_connections: Number of connections
        mean_conductance: Mean conductance value
        std_conductance: Standard deviation
        min_conductance: Minimum allowed conductance
        max_conductance: Maximum allowed conductance  
        distribution: Distribution type ('normal', 'lognormal', 'gamma')
        connection_modulation: Connection-type specific scaling factor
        
    Returns:
        conductances: Tensor of conductance values
    """
    
    # Apply connection-specific modulation to mean
    modulated_mean = mean_conductance * connection_modulation
    modulated_std = std_conductance * connection_modulation
    
    if distribution == 'normal':
        # Normal distribution (can have negative values, so clip)
        conductances = torch.normal(modulated_mean, modulated_std, (n_connections,))
        
    elif distribution == 'lognormal':
        # Log-normal distribution (always positive, realistic for conductances)
        # Convert mean/std to log-normal parameters
        mean_log = math.log(modulated_mean**2 / math.sqrt(modulated_mean**2 + modulated_std**2))
        std_log = math.sqrt(math.log(1 + (modulated_std/modulated_mean)**2))
        
        log_conductances = torch.normal(mean_log, std_log, (n_connections,))
        conductances = torch.exp(log_conductances)
        
    elif distribution == 'gamma':
        # Gamma distribution (positive, skewed - realistic for biological parameters)
        # Convert mean/std to shape/scale parameters
        scale = modulated_std**2 / modulated_mean
        shape = modulated_mean / scale
        
        # Use numpy for gamma sampling, then convert to torch
        gamma_samples = np.random.gamma(shape, scale, n_connections)
        conductances = torch.from_numpy(gamma_samples).float()
        
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")
    
    # Clamp to physiologically reasonable bounds
    conductances = torch.clamp(conductances, min_conductance, max_conductance)
    
    return conductances



@dataclass
class CircuitParams:
    """Circuit parameters with anatomical connectivity"""
    # Population sizes (scaled for simulation)
    n_gc: int = 1000
    n_mc: int = 30
    n_pv: int = 30
    n_sst: int = 20
    n_mec: int = 60  # MEC layer 2 principal cells
    
    # Local connection probabilities (GC -> local targets)
    p_gc_mc_local: float = 0.3   # GC to nearby MC
    p_gc_pv_local: float = 0.05    # GC to nearby PV
    p_gc_sst_local: float = 0.06   # GC to nearby SST
    
    # MC connection probabilities
    p_mc_gc_local: float = 0.01      # MC to local GC
    p_mc_gc_distant: float = 0.02    # MC to distant GC
    p_mc_pv_distant: float = 0.08    # MC to distant PV
    p_mc_sst: float = 0.25           # MC to local SST (molecular layer)
    p_mc_mc_distant: float = 0.02    # MC to distant MC
    
    # MEC connection probabilities (from Hainmueller et al.)
    p_mec_gc: float = 0.025         # Perforant path to GC
    p_mec_mc: float = 0.0          # Assuming no direct MEC → MC
    p_mec_pv: float = 0.019        # Direct MEC → PV (1.91% from paper)
    p_mec_sst: float = 0.0         # NO direct MEC → SST (key asymmetry!)

    # Interneuron feedback connections  
    p_pv_gc: float = 0.33           # PV feedback inhibition
    p_pv_mc: float = 0.1           # PV to MC inhibition
    p_pv_pv: float = 0.11          # PV lateral inhibition
    p_pv_sst: float = 0.0          
    p_sst_gc: float = 0.2          # SST dendritic inhibition
    p_sst_mc: float = 0.3          # SST to MC inhibition
    p_sst_pv: float = 0.15         # SST to PV feedback
    p_sst_sst: float = 0.05        # SST to SST feedback
    
    # Population heterogeneity for competition
    pv_subpop_ratio: float = 0.6   # Fraction in "fast" PV subpopulation
    sst_subpop_ratio: float = 0.5  # Fraction in "proximal" SST subpopulation
    
    # Spatial parameters (mm)
    gc_layer_thickness: float = 0.05   # Thin granule cell layer
    hilar_thickness: float = 0.3       # Hilar region thickness
    ml_thickness: float = 0.4          # Molecular layer thickness
    local_radius: float = 0.5          # Local connection radius
    distant_min: float = 0.9           # Minimum distance for "distant"
    mec_distance: float = 2.0          # Distance from MEC to DG
    
    # Synaptic parameters
    dt: float = 0.1                    # ms
    tau_syn: float = 5.0               # synaptic time constant
    tau_mec: float = 8.0               # slower MEC integration

@dataclass  
class OpsinParams:
    """Opsin expression and light activation parameters"""
    expression_mean: float = 0.8
    expression_std: float = 0.2
    failure_rate: float = 0.4
    light_decay: float = 0.3          # mm^-1
    hill_coeff: float = 2.5
    half_sat: float = 0.4

class SpatialLayout:
    """Handle 3D spatial organization of DG populations"""
    
    def __init__(self, params: CircuitParams):
        self.params = params
        self.positions = self._create_anatomical_layout()
        self.scale = 1.0
        
    def _create_anatomical_layout(self) -> Dict[str, Tensor]:
        """Create anatomically realistic 3D layout"""
        positions = {}
        
        # Granule cell layer (tightly packed, z ≈ 0)
        gc_x = torch.randn(self.params.n_gc) * 0.8
        gc_y = torch.randn(self.params.n_gc) * 0.8
        gc_z = torch.randn(self.params.n_gc) * self.params.gc_layer_thickness
        positions['gc'] = torch.stack([gc_x, gc_y, gc_z], dim=1)
        
        # Mossy cells (hilar region, z > 0)
        mc_x = torch.randn(self.params.n_mc) * 0.6
        mc_y = torch.randn(self.params.n_mc) * 0.6
        mc_z = torch.abs(torch.randn(self.params.n_mc)) * (self.params.hilar_thickness) - 0.05
        positions['mc'] = torch.stack([mc_x, mc_y, mc_z], dim=1)
        
        # PV interneurons (distributed across layers)
        pv_x = torch.randn(self.params.n_pv) * 0.7
        pv_y = torch.randn(self.params.n_pv) * 0.7
        pv_z = torch.randn(self.params.n_pv) * (self.params.hilar_thickness + self.params.gc_layer_thickness) * 0.3
        positions['pv'] = torch.stack([pv_x, pv_y, pv_z], dim=1)
        
        # SST interneurons (hilar region bias, z > 0)
        sst_x = torch.randn(self.params.n_sst) * 0.8
        sst_y = torch.randn(self.params.n_sst) * 0.8
        sst_z = torch.abs(torch.randn(self.params.n_sst)) * self.params.hilar_thickness + 0.05
        positions['sst'] = torch.stack([sst_x, sst_y, sst_z], dim=1)
        
        # MEC layer 2 (distant from DG, separate coordinate system)
        mec_x = torch.randn(self.params.n_mec) * 1.0 + self.params.mec_distance  # Offset in x
        mec_y = torch.randn(self.params.n_mec) * 0.6
        mec_z = -torch.rand(self.params.n_mec) * self.params.ml_thickness
        positions['mec'] = torch.stack([mec_x, mec_y, mec_z], dim=1)
        
        return positions
    
    def distance_matrix(self, pop1: str, pop2: str) -> Tensor:
        """Calculate distance matrix between two populations"""
        pos1 = self.positions[pop1]
        pos2 = self.positions[pop2]
        
        # Broadcasting for efficient distance calculation
        diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        return distances
    
    def create_local_mask(self, pop1: str, pop2: str) -> Tensor:
        """Create mask for local connections"""
        distances = self.distance_matrix(pop1, pop2)
        return distances < self.params.local_radius
    
    def create_distant_mask(self, pop1: str, pop2: str) -> Tensor:
        """Create mask for distant connections"""
        distances = self.distance_matrix(pop1, pop2)
        return distances > self.params.distant_min

class ConnectivityMatrix:
    """Connectivity matrix generator with per-connection conductances"""
    
    def __init__(self, circuit_params, layout, synaptic_params: PerConnectionSynapticParams):
        self.circuit_params = circuit_params
        self.layout = layout
        self.synaptic_params = synaptic_params
        self.conductance_matrices = self._build_conductance_matrices()
    
    def _build_conductance_matrices(self) -> Dict[str, ConductanceMatrix]:
        """Build connectivity and conductance matrices together"""
        
        conductance_matrices = {}
        
        # Define all connections with their types
        connections = {
            # Excitatory connections
            'gc_mc': ('gc', 'mc', 'excitatory', self.circuit_params.p_gc_mc_local, 'local'),
            'gc_pv': ('gc', 'pv', 'excitatory', self.circuit_params.p_gc_pv_local, 'local'),
            'gc_sst': ('gc', 'sst', 'excitatory', self.circuit_params.p_gc_sst_local, 'local'),
            'mc_gc': ('mc', 'gc', 'excitatory', (self.circuit_params.p_mc_gc_local, self.circuit_params.p_mc_gc_distant), 'mixed'),
            'mc_pv': ('mc', 'pv', 'excitatory', self.circuit_params.p_mc_pv_distant, 'distant'),
            'mc_sst': ('mc', 'sst', 'excitatory', self.circuit_params.p_mc_sst, 'local'),
            'mc_mc': ('mc', 'mc', 'excitatory', self.circuit_params.p_mc_mc_distant, 'distant'),
            'mec_gc': ('mec', 'gc', 'excitatory', self.circuit_params.p_mec_gc, 'random'),
            'mec_pv': ('mec', 'pv', 'excitatory', self.circuit_params.p_mec_pv, 'random'),
            
            # Inhibitory connections
            'pv_gc': ('pv', 'gc', 'inhibitory', self.circuit_params.p_pv_gc, 'random'),
            'pv_mc': ('pv', 'mc', 'inhibitory', self.circuit_params.p_pv_mc, 'random'),
            'pv_pv': ('pv', 'pv', 'inhibitory', self.circuit_params.p_pv_pv, 'competitive'),
            'pv_sst': ('pv', 'sst', 'inhibitory', self.circuit_params.p_pv_sst, 'local'),
            'sst_gc': ('sst', 'gc', 'inhibitory', self.circuit_params.p_sst_gc, 'random'),
            'sst_mc': ('sst', 'mc', 'inhibitory', self.circuit_params.p_sst_mc, 'random'),
            'sst_pv': ('sst', 'pv', 'inhibitory', self.circuit_params.p_sst_pv, 'random'),
            'sst_sst': ('sst', 'sst', 'inhibitory', self.circuit_params.p_sst_sst, 'random'),
        }
        
        for conn_name, (pre_pop, post_pop, syn_type, prob, conn_type) in connections.items():
            # Generate connectivity matrix
            connectivity = self._generate_connectivity(pre_pop, post_pop, prob, conn_type)
            
            # Get actual connection indices
            connection_indices = torch.nonzero(connectivity, as_tuple=True)
            n_actual_connections = len(connection_indices[0])
            
            if n_actual_connections == 0:
                print(f"Warning: No connections found for {conn_name}")
                conductances = torch.zeros_like(connectivity)
            else:
                # Generate conductances for existing connections
                if syn_type == 'excitatory':
                    base_conductances = generate_conductance_distribution(
                        int(n_actual_connections),
                        self.synaptic_params.ampa_g_mean,
                        self.synaptic_params.ampa_g_std,
                        self.synaptic_params.ampa_g_min,
                        self.synaptic_params.ampa_g_max,
                        self.synaptic_params.distribution,
                        self.synaptic_params.connection_modulation.get(conn_name, 1.0)
                    )
                else:  # inhibitory
                    base_conductances = generate_conductance_distribution(
                        int(n_actual_connections),
                        self.synaptic_params.gaba_g_mean,
                        self.synaptic_params.gaba_g_std,
                        self.synaptic_params.gaba_g_min,
                        self.synaptic_params.gaba_g_max,
                        self.synaptic_params.distribution,
                        self.synaptic_params.connection_modulation.get(conn_name, 1.0)
                    )
                
                # Map conductances back to full matrix
                conductances = torch.zeros_like(connectivity)
                conductances[connection_indices] = base_conductances
            
            # Store as ConductanceMatrix
            conductance_matrices[conn_name] = ConductanceMatrix(
                connectivity=connectivity,
                conductances=conductances,
                connection_name=conn_name,
                synapse_type=syn_type
            )
        
        return conductance_matrices
    
    def _generate_connectivity(self, pre_pop: str, post_pop: str, 
                             prob, conn_type: str) -> torch.Tensor:
        """Generate connectivity matrix based on connection type"""
        
        pre_size = getattr(self.circuit_params, f'n_{pre_pop}')
        post_size = getattr(self.circuit_params, f'n_{post_pop}')
        
        if conn_type == 'local':
            return self._local_connectivity(pre_pop, post_pop, prob)
        elif conn_type == 'distant':
            return self._distant_connectivity(pre_pop, post_pop, prob)
        elif conn_type == 'mixed':
            local_prob, distant_prob = prob
            return self._mixed_connectivity(pre_pop, post_pop, local_prob, distant_prob)
        elif conn_type == 'competitive':
            return self._competitive_connectivity(pre_size, prob)
        else:  # random
            return self._random_connectivity(pre_size, post_size, prob)
    
    def _local_connectivity(self, pre_pop: str, post_pop: str, prob: float) -> torch.Tensor:
        """Create connections only between nearby cells"""
        if not hasattr(self.layout, 'create_local_mask'):
            # Fallback to random if no spatial layout
            pre_size = getattr(self.circuit_params, f'n_{pre_pop}')
            post_size = getattr(self.circuit_params, f'n_{post_pop}')
            return self._random_connectivity(pre_size, post_size, prob)
            
        local_mask = self.layout.create_local_mask(pre_pop, post_pop)
        random_mask = torch.rand_like(local_mask.float()) < prob
        return (local_mask & random_mask).float()
    
    def _distant_connectivity(self, pre_pop: str, post_pop: str, prob: float) -> torch.Tensor:
        """Create connections only between distant cells"""
        if not hasattr(self.layout, 'create_distant_mask'):
            # Fallback to random if no spatial layout
            pre_size = getattr(self.circuit_params, f'n_{pre_pop}')
            post_size = getattr(self.circuit_params, f'n_{post_pop}')
            return self._random_connectivity(pre_size, post_size, prob)
            
        distant_mask = self.layout.create_distant_mask(pre_pop, post_pop)
        random_mask = torch.rand_like(distant_mask.float()) < prob
        return (distant_mask & random_mask).float()
    
    def _mixed_connectivity(self, pre_pop: str, post_pop: str, 
                          local_prob: float, distant_prob: float) -> torch.Tensor:
        """Create both local and distant connections"""
        local_conn = self._local_connectivity(pre_pop, post_pop, local_prob)
        distant_conn = self._distant_connectivity(pre_pop, post_pop, distant_prob)
        return local_conn + distant_conn
    
    def _competitive_connectivity(self, n_cells: int, prob: float) -> torch.Tensor:
        """Create competitive lateral inhibition matrix"""
        base_conn = self._random_connectivity(n_cells, n_cells, prob)
        
        # Enhanced competition for PV cells (based on subpopulations)
        n_fast = int(n_cells * self.circuit_params.pv_subpop_ratio)
        competition_matrix = torch.ones(n_cells, n_cells)
        competition_matrix[:n_fast, n_fast:] *= 2.0    # Fast -> slow stronger
        competition_matrix[n_fast:, :n_fast] *= 0.5    # Slow -> fast weaker
        
        return base_conn * competition_matrix
    
    def _random_connectivity(self, n_pre: int, n_post: int, prob: float) -> torch.Tensor:
        """Create random connectivity matrix"""
        connectivity = (torch.rand(n_pre, n_post) < prob).float()
        
        # Remove self-connections if pre and post are the same population
        if n_pre == n_post:
            connectivity = connectivity * (1 - torch.eye(n_pre))
            
        return connectivity
    
class DentateCircuit(nn.Module):
    """Dentate gyrus circuit model with dendritic-somatic transfer"""
    
    def __init__(self,
                 circuit_params: CircuitParams,
                 synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams):
        super().__init__()
        self.circuit_params = circuit_params
        self.synaptic_params = synaptic_params
        self.opsin_params = opsin_params

        # Create spatial layout and connectivity
        self.layout = SpatialLayout(circuit_params)
        self.connectivity = ConnectivityMatrix(circuit_params, self.layout, self.synaptic_params)
        
        # Register conductance matrices as buffers (non-trainable)
        for name, cond_matrix in self.connectivity.conductance_matrices.items():
            self.register_buffer(f'conn_{name}', cond_matrix.connectivity)
            self.register_buffer(f'cond_{name}', cond_matrix.conductances)
            
        # Get cell-type specific dendritic parameters
        self.dendritic_params = get_cell_type_parameters()
        
        
        # Population activities (state variables)
        self.register_buffer('gc_activity', torch.zeros(circuit_params.n_gc))
        self.register_buffer('mc_activity', torch.zeros(circuit_params.n_mc))
        self.register_buffer('pv_activity', torch.zeros(circuit_params.n_pv))
        self.register_buffer('sst_activity', torch.zeros(circuit_params.n_sst))
        self.register_buffer('mec_activity', torch.zeros(circuit_params.n_mec))
        
        # Dendritic-somatic state variables
        self.register_buffer('gc_adaptation', torch.zeros(circuit_params.n_gc))
        self.register_buffer('mc_adaptation', torch.zeros(circuit_params.n_mc))
        self.register_buffer('pv_adaptation', torch.zeros(circuit_params.n_pv))
        self.register_buffer('sst_adaptation', torch.zeros(circuit_params.n_sst))
        self.register_buffer('mec_adaptation', torch.zeros(circuit_params.n_mec))
        
        # Store dendritic and somatic voltages for analysis
        self.register_buffer('gc_v_dendrite', torch.zeros(circuit_params.n_gc))
        self.register_buffer('mc_v_dendrite', torch.zeros(circuit_params.n_mc))
        self.register_buffer('pv_v_dendrite', torch.zeros(circuit_params.n_pv))
        self.register_buffer('sst_v_dendrite', torch.zeros(circuit_params.n_sst))
        
        self.register_buffer('gc_v_soma', torch.zeros(circuit_params.n_gc))
        self.register_buffer('mc_v_soma', torch.zeros(circuit_params.n_mc))
        self.register_buffer('pv_v_soma', torch.zeros(circuit_params.n_pv))
        self.register_buffer('sst_v_soma', torch.zeros(circuit_params.n_sst))

        self.add_synaptic_state_manager()

        
    def add_synaptic_state_manager(self):
        """Add synaptic state manager to the circuit (call this in __init__)"""
        if not hasattr(self, 'synaptic_state_manager'):
            self.synaptic_state_manager = SynapticStateManager(
                self.circuit_params,
                self.synaptic_params, 
                self.connectivity.conductance_matrices
            )

        # Add NMDA fraction parameter if not present
        if not hasattr(self.synaptic_params, 'nmda_fraction'):
            self.synaptic_params.nmda_fraction = 0.3  # Default NMDA fraction
        
    def reset_state(self):
        """Reset circuit to baseline state"""
        # Reset activities
        self.gc_activity.zero_()
        self.mc_activity.zero_()
        self.pv_activity.zero_()
        self.sst_activity.zero_()
        self.mec_activity.zero_()
        
        # Reset state variables
        self.gc_adaptation.zero_()
        self.mc_adaptation.zero_()
        self.pv_adaptation.zero_()
        self.sst_adaptation.zero_()
        self.mec_adaptation.zero_()
        
        self.gc_v_dendrite.zero_()
        self.mc_v_dendrite.zero_()
        self.pv_v_dendrite.zero_()
        self.sst_v_dendrite.zero_()
        
        self.gc_v_soma.zero_()
        self.mc_v_soma.zero_()
        self.pv_v_soma.zero_()
        self.sst_v_soma.zero_()

        # Reset synaptic states
        if hasattr(self, 'synaptic_state_manager'):
            self.synaptic_state_manager.reset_states()


    def get_synaptic_state_info(self) -> Dict[str, Dict]:
        """Get information about current synaptic states"""
        if not hasattr(self, 'synaptic_state_manager'):
            return {}

        state_info = {}

        for conn_name in self.synaptic_state_manager.synaptic_states.keys():
            conductances = self.synaptic_state_manager.get_total_conductances(conn_name)

            state_info[conn_name] = {
                'ampa_total': torch.sum(conductances['ampa']).item(),
                'gaba_total': torch.sum(conductances['gaba']).item(), 
                'nmda_total': torch.sum(conductances['nmda']).item(),
                'ampa_mean': torch.mean(conductances['ampa']).item(),
                'gaba_mean': torch.mean(conductances['gaba']).item(),
                'nmda_mean': torch.mean(conductances['nmda']).item()
            }

        return state_info
        
    def firing_rate_to_membrane_potential(self, post_rates: torch.Tensor, post_type: str) -> torch.Tensor:
        """
        Convert postsynaptic firing rates to estimated membrane potential

        Uses inverse of the cell's f-I relationship to estimate what membrane potential
        would produce the observed firing rates.

        Args:
            post_rates: Firing rates of postsynaptic neurons (Hz)
            post_type: Cell type ('gc', 'mc', 'pv', 'sst', 'mec')

        Returns:
            estimated_potential: Estimated membrane potential (mV)
        """
        dendritic_params = self.dendritic_params[post_type]

        v_rest = dendritic_params.v_rest
        v_thresh = dendritic_params.axon_hill_thresh  # Axon hillock threshold
        max_rate = dendritic_params.max_firing_rate
        rate_gain = dendritic_params.rate_gain

        # Clamp firing rates to valid range
        safe_rates = torch.clamp(post_rates, 0.0, max_rate)

        # For cells with zero firing rate, potential is at resting
        # For active cells, estimate potential above threshold

        # The transfer function uses: firing_rate = max_rate * expm1(rate_gain * (V_soma - V_thresh))
        # Inverse: V_soma = V_thresh + ln(firing_rate/max_rate + 1) / rate_gain

        # Handle zero rates specially
        zero_mask = safe_rates <= 1e-3  # Essentially zero firing
        active_mask = ~zero_mask

        estimated_potential = torch.full_like(safe_rates, v_rest)

        if torch.any(active_mask) and rate_gain > 0:
            active_rates = safe_rates[active_mask]
            normalized_rates = active_rates / max_rate

            # Inverse of expm1: if y = expm1(x), then x = ln(y + 1)
            v_above_thresh = torch.log(normalized_rates + 1.0) / rate_gain
            v_soma_active = v_thresh + v_above_thresh

            # For very low rates, interpolate between resting and threshold
            # This handles the transition region more smoothly
            rate_fraction = torch.clamp(normalized_rates, 0.0, 0.1) / 0.1  # 0-10% of max rate
            smooth_transition = rate_fraction < 1.0

            if torch.any(smooth_transition):
                transition_mask = active_mask.clone()
                transition_mask[active_mask] = smooth_transition

                # Smooth interpolation from rest to threshold for low rates
                interp_factor = rate_fraction[smooth_transition]
                v_interp = v_rest + interp_factor * (v_thresh - v_rest)
                estimated_potential[transition_mask] = v_interp

                # Higher rates use the inverse exponential relationship
                high_rate_mask = active_mask.clone()
                high_rate_mask[active_mask] = ~smooth_transition
                estimated_potential[high_rate_mask] = v_soma_active[~smooth_transition]
            else:
                estimated_potential[active_mask] = v_soma_active

        elif torch.any(active_mask):
            # Fallback for cells with rate_gain = 0: linear relationship
            active_rates = safe_rates[active_mask]
            normalized_rates = active_rates / max_rate
            # Linear interpolation from rest to threshold + some headroom
            v_max = v_thresh + 15.0  # 15mV above threshold for max rate
            estimated_potential[active_mask] = v_rest + normalized_rates * (v_max - v_rest)

        return estimated_potential

    


    def firing_rate_to_current(self, pre_rates: torch.Tensor,
                               conn_name: str,
                               post_type: str = 'gc',
                               post_rates: torch.Tensor = None) -> torch.Tensor:
        """
        Conductance-based synaptic conversion using dynamic synaptic states

        Args:
            pre_rates: Presynaptic firing rates (Hz) - not used directly, synaptic states are pre-computed
            conn_name: Name of the connection (e.g., 'gc_mc', 'pv_gc')
            post_type: Postsynaptic cell type ('gc', 'mc', 'pv', 'sst', 'mec')
            post_rates: Postsynaptic firing rates (Hz) for voltage estimation

        Returns:
            synaptic_current: Total synaptic current to each postsynaptic cell (pA)
        """

        # Get current synaptic conductances from state manager
        conductances = self.synaptic_state_manager.get_total_conductances(conn_name)

        total_ampa = conductances['ampa']
        total_gaba = conductances['gaba'] 
        total_nmda = conductances['nmda']

        # Estimate postsynaptic membrane potential from firing rates
        if post_rates is not None:
            post_potential = self.firing_rate_to_membrane_potential(post_rates, post_type)
        else:
            # Default potentials by cell type
            potential_defaults = {
                'gc': -70.0,
                'mc': -60.0,
                'pv': -55.0, 
                'sst': -60.0,
                'mec': -65.0
            }
            post_potential = torch.full_like(total_ampa, potential_defaults.get(post_type, -65.0))

        # Calculate currents for each synapse type
        # AMPA current
        ampa_driving_force = post_potential - self.synaptic_params.e_exc
        ampa_current = total_ampa * ampa_driving_force

        # GABA current  
        gaba_driving_force = post_potential - self.synaptic_params.e_inh
        gaba_current = total_gaba * gaba_driving_force

        # NMDA current (with voltage dependence)
        if torch.sum(total_nmda) > 0:
            # Apply NMDA voltage dependence (Mg2+ block)
            mg_concentration = 1.0  # mM
            nmda_voltage_factor = 1.0 / (1.0 + (mg_concentration / 3.57) * torch.exp(-0.062 * post_potential))
            effective_nmda_conductance = total_nmda * nmda_voltage_factor

            nmda_driving_force = post_potential - self.synaptic_params.e_exc
            nmda_current = effective_nmda_conductance * nmda_driving_force
        else:
            nmda_current = torch.zeros_like(ampa_current)

        # Total synaptic current
        total_current = ampa_current + gaba_current + nmda_current

        return total_current

    
    def calculate_synaptic_currents(self) -> Dict[str, torch.Tensor]:
        """
        Convert all firing rates to proper synaptic currents using dynamic postsynaptic potentials
        """

        # Define current activities
        activities = {
            'gc': self.gc_activity,
            'mc': self.mc_activity, 
            'pv': self.pv_activity,
            'sst': self.sst_activity,
            'mec': self.mec_activity
        }

        self.synaptic_state_manager.update_synaptic_states(activities)

        # Initialize current accumulators
        currents = {
            'gc': torch.zeros(self.circuit_params.n_gc),
            'mc': torch.zeros(self.circuit_params.n_mc),
            'pv': torch.zeros(self.circuit_params.n_pv),
            'sst': torch.zeros(self.circuit_params.n_sst),
            'mec': torch.zeros(self.circuit_params.n_mec)
        }

        # Process each connection type
        for conn_name, cond_matrix in self.connectivity.conductance_matrices.items():
            parts = conn_name.split('_')
            pre_pop, post_pop = parts[0], parts[1]
            
            if pre_pop not in activities or post_pop not in activities:
                continue
                
            pre_activity = activities[pre_pop]
            post_activity = activities[post_pop]
            
            # Calculate current using per-connection conductances
            current = self.firing_rate_to_current(
                pre_activity, 
                conn_name,
                post_pop,
                post_activity
            )
            
            # Accumulate currents (subtract because we're using negative current convention)
            currents[post_pop] -= current

        return currents


    def update_activity_with_dendritic_somatic(self, direct_activation: Dict[str, torch.Tensor], 
                                               external_drive: Dict[str, torch.Tensor] = None):
        """
        Updated activity update using conductance-based dendritic-somatic transfer
        """
        if external_drive is None:
            external_drive = {}

        # Step 1: Update synaptic states and get conductances for each population
        activities = {
            'gc': self.gc_activity,
            'mc': self.mc_activity,
            'pv': self.pv_activity,
            'sst': self.sst_activity,
            'mec': self.mec_activity
        }

        self.synaptic_state_manager.update_synaptic_states(activities)

        # Step 2: Update each population using conductance-based transfer
        for pop in ['gc', 'mc', 'pv', 'sst']:

            # Get current states
            if pop == 'gc':
                current_activity = self.gc_activity
                adaptation_state = self.gc_adaptation
            elif pop == 'mc':
                current_activity = self.mc_activity
                adaptation_state = self.mc_adaptation
            elif pop == 'pv':
                current_activity = self.pv_activity
                adaptation_state = self.pv_adaptation
            elif pop == 'sst':
                current_activity = self.sst_activity
                adaptation_state = self.sst_adaptation

            # Aggregate conductances from all presynaptic sources
            total_ampa = torch.zeros_like(current_activity)
            total_gaba = torch.zeros_like(current_activity)  
            total_nmda = torch.zeros_like(current_activity)

            # Sum conductances from all incoming connections
            for conn_name in self.synaptic_state_manager.synaptic_states.keys():
                parts = conn_name.split('_')
                if len(parts) >= 2 and parts[1] == pop:  # This connection targets current population
                    conductances = self.synaptic_state_manager.get_total_conductances(conn_name)
                    total_ampa += conductances['ampa']
                    total_gaba += conductances['gaba']
                    total_nmda += conductances['nmda']

            # Add direct activation as additional AMPA-like conductance
            # Convert current (pA) to conductance (nS) assuming some driving force
            direct_current = direct_activation.get(pop, torch.zeros_like(current_activity))
            # Assume 70mV driving force for conversion: I = g * (V - E), so g = I / (V - E)
            direct_conductance = torch.clamp(direct_current / 70.0, min=0.0)  # Only positive conductances
            total_ampa += direct_conductance
            #if torch.sum(direct_conductance) > 0.0:
            #    print(f"{pop}: direct conductance: {torch.sum(direct_conductance)} total_ampa: {torch.sum(total_ampa)}")
            
            # Apply conductance-based dendritic-somatic transfer
            params = self.dendritic_params[pop]
            new_activity_tensor, new_adaptation_tensor, new_v_dendrite_tensor, new_v_soma_tensor = \
                self._dendritic_somatic_transfer_updated(
                    total_ampa, total_gaba, total_nmda, params, adaptation_state
                )

            # Update state variables
            if pop == 'gc':
                self.gc_activity = new_activity_tensor
                self.gc_adaptation = new_adaptation_tensor
                self.gc_v_dendrite = new_v_dendrite_tensor
                self.gc_v_soma = new_v_soma_tensor
            elif pop == 'mc':
                self.mc_activity = new_activity_tensor
                self.mc_adaptation = new_adaptation_tensor
                self.mc_v_dendrite = new_v_dendrite_tensor
                self.mc_v_soma = new_v_soma_tensor
            elif pop == 'pv':
                self.pv_activity = new_activity_tensor
                self.pv_adaptation = new_adaptation_tensor
                self.pv_v_dendrite = new_v_dendrite_tensor
                self.pv_v_soma = new_v_soma_tensor
            elif pop == 'sst':
                self.sst_activity = new_activity_tensor
                self.sst_adaptation = new_adaptation_tensor
                self.sst_v_dendrite = new_v_dendrite_tensor
                self.sst_v_soma = new_v_soma_tensor

        # Handle MEC separately (external drive only, no synaptic input in this model)
        mec_drive = external_drive.get('mec', torch.zeros_like(self.mec_activity))
        mec_direct = direct_activation.get('mec', torch.zeros_like(self.mec_activity))
        mec_total_ampa = torch.clamp((mec_drive + mec_direct) / 70.0, min=0.0)
        mec_gaba = torch.zeros_like(mec_total_ampa)
        mec_nmda = torch.zeros_like(mec_total_ampa)

        mec_activity, mec_adaptation, _, _ = self._dendritic_somatic_transfer_updated(
            mec_total_ampa, mec_gaba, mec_nmda, 
            self.dendritic_params['mec'], self.mec_adaptation
        )

        self.mec_activity = mec_activity
        self.mec_adaptation = mec_adaptation

    def _dendritic_somatic_transfer_updated(self,
                                            ampa_conductances: torch.Tensor,
                                            gaba_conductances: torch.Tensor, 
                                            nmda_conductances: torch.Tensor,
                                            params: DendriticParameters, 
                                            adaptation_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Vectorized dendritic-somatic transfer function using conductances

            Args:
                ampa_conductances: AMPA conductances for each cell (nS)
                gaba_conductances: GABA conductances for each cell (nS) 
                nmda_conductances: NMDA conductances for each cell (nS)
                params: Cell-type specific parameters
                adaptation_state: Current adaptation state

            Returns:
                firing_rate, new_adaptation, v_dendrite, v_soma: Updated states
            """

            firing_rate, states = dendritic_somatic_transfer(
                ampa_conductances, gaba_conductances, nmda_conductances,
                params, adaptation_state
            )

            return (firing_rate, 
                    states['adaptation'], 
                    states['v_dendrite'], 
                    states['v_soma'])

        
    def forward(self, direct_activation: Dict[str, Tensor], external_drive: Dict[str, Tensor] = None) -> Dict[str, Tensor]:
        """Single time step forward pass with dendritic-somatic processing"""
        
        self.update_activity_with_dendritic_somatic(direct_activation, external_drive)
        
        return {
            'gc': self.gc_activity.clone(),
            'mc': self.mc_activity.clone(), 
            'pv': self.pv_activity.clone(),
            'sst': self.sst_activity.clone(),
            'mec': self.mec_activity.clone()
        }

    def get_conductance_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about conductance distributions"""
        stats = {}
        
        for conn_name, cond_matrix in self.connectivity.conductance_matrices.items():
            # Get conductances for existing connections only
            existing_connections = cond_matrix.connectivity > 0
            active_conductances = cond_matrix.conductances[existing_connections]
            
            if len(active_conductances) > 0:
                stats[conn_name] = {
                    'mean': float(torch.mean(active_conductances)),
                    'std': float(torch.std(active_conductances)),
                    'min': float(torch.min(active_conductances)),
                    'max': float(torch.max(active_conductances)),
                    'n_connections': len(active_conductances),
                    'synapse_type': cond_matrix.synapse_type
                }
            else:
                stats[conn_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'n_connections': 0,
                    'synapse_type': cond_matrix.synapse_type
                }
        
        return stats
    
    def get_dendritic_somatic_states(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get detailed dendritic and somatic state information"""
        return {
            'gc': {
                'activity': self.gc_activity.clone(),
                'adaptation': self.gc_adaptation.clone(),
                'v_dendrite': self.gc_v_dendrite.clone(),
                'v_soma': self.gc_v_soma.clone(),
            },
            'mc': {
                'activity': self.mc_activity.clone(),
                'adaptation': self.mc_adaptation.clone(),
                'v_dendrite': self.mc_v_dendrite.clone(),
                'v_soma': self.mc_v_soma.clone(),
            },
            'pv': {
                'activity': self.pv_activity.clone(),
                'adaptation': self.pv_adaptation.clone(),
                'v_dendrite': self.pv_v_dendrite.clone(),
                'v_soma': self.pv_v_soma.clone(),
            },
            'sst': {
                'activity': self.sst_activity.clone(),
                'adaptation': self.sst_adaptation.clone(),
                'v_dendrite': self.sst_v_dendrite.clone(),
                'v_soma': self.sst_v_soma.clone(),
            },
        }

    def update_connection_modulation(self, new_connection_modulation: Dict[str, float]):
        """
        Update connection weights based on new connection modulation values

        Args:
            new_connection_modulation: Dictionary with connection names as keys
                                     and modulation factors as values
                                     (e.g., from optimized JSON results)
        """

        print("Updating connection modulation parameters...")

        # Create new synaptic parameters with updated connection modulation
        updated_synaptic_params = PerConnectionSynapticParams(
            ampa_g_mean=self.synaptic_params.ampa_g_mean,
            ampa_g_std=self.synaptic_params.ampa_g_std,
            ampa_g_min=self.synaptic_params.ampa_g_min,
            ampa_g_max=self.synaptic_params.ampa_g_max,
            gaba_g_mean=self.synaptic_params.gaba_g_mean,
            gaba_g_std=self.synaptic_params.gaba_g_std,
            gaba_g_min=self.synaptic_params.gaba_g_min,
            gaba_g_max=self.synaptic_params.gaba_g_max,
            distribution=self.synaptic_params.distribution,
            connection_modulation=new_connection_modulation,  # Use new values
            e_exc=self.synaptic_params.e_exc,
            e_inh=self.synaptic_params.e_inh,
            v_rest=self.synaptic_params.v_rest,
            tau_ampa=self.synaptic_params.tau_ampa,
            tau_gaba=self.synaptic_params.tau_gaba,
            tau_nmda=self.synaptic_params.tau_nmda
        )

        # Store updated synaptic parameters
        self.synaptic_params = updated_synaptic_params

        # Rebuild connectivity and conductance matrices
        new_connectivity = ConnectivityMatrix(
            self.circuit_params, 
            self.layout, 
            updated_synaptic_params
        )

        # Update stored connectivity
        self.connectivity = new_connectivity

        # Update all registered buffers with new conductance matrices
        changes_summary = {}

        for name, cond_matrix in new_connectivity.conductance_matrices.items():
            # Update connectivity matrix buffer
            conn_buffer_name = f'conn_{name}'
            cond_buffer_name = f'cond_{name}'

            if hasattr(self, conn_buffer_name):
                old_conn = getattr(self, conn_buffer_name)
                new_conn = cond_matrix.connectivity
                setattr(self, conn_buffer_name, new_conn)

                # Calculate change statistics for conductances
                old_cond = getattr(self, cond_buffer_name) if hasattr(self, cond_buffer_name) else None
                new_cond = cond_matrix.conductances
                setattr(self, cond_buffer_name, new_cond)

                if old_cond is not None:
                    # Calculate mean conductance change for existing connections
                    existing_mask = (old_conn > 0) & (new_conn > 0)
                    if torch.sum(existing_mask) > 0:
                        old_mean = torch.mean(old_cond[existing_mask])
                        new_mean = torch.mean(new_cond[existing_mask])
                        percent_change = ((new_mean - old_mean) / old_mean * 100).item()
                        changes_summary[name] = {
                            'old_mean': old_mean.item(),
                            'new_mean': new_mean.item(),
                            'percent_change': percent_change,
                            'synapse_type': cond_matrix.synapse_type
                        }
            else:
                # Register new buffers if they don't exist
                self.register_buffer(conn_buffer_name, cond_matrix.connectivity)
                self.register_buffer(cond_buffer_name, cond_matrix.conductances)

                changes_summary[name] = {
                    'old_mean': 0.0,
                    'new_mean': torch.mean(cond_matrix.conductances[cond_matrix.connectivity > 0]).item(),
                    'percent_change': float('inf'),
                    'synapse_type': cond_matrix.synapse_type
                }

        # Print summary of changes
        print(f"\nConnection modulation updates applied:")
        print(f"{'Connection':<12} {'Type':<12} {'Old Mean':<10} {'New Mean':<10} {'Change':<10}")
        print("-" * 65)

        for conn_name, stats in changes_summary.items():
            change_str = f"{stats['percent_change']:+.1f}%" if stats['percent_change'] != float('inf') else "NEW"
            print(f"{conn_name:<12} {stats['synapse_type']:<12} {stats['old_mean']:<10.3f} "
                  f"{stats['new_mean']:<10.3f} {change_str:<10}")

        print(f"\nSuccessfully updated {len(changes_summary)} connection types.")

        return changes_summary

    def load_and_apply_optimization_results(self, json_filename: str):
        """
        Load optimization results from JSON file and apply to circuit

        Args:
            json_filename: Path to JSON file containing optimization results
        """

        import json

        try:
            with open(json_filename, 'r') as f:
                data = json.load(f)

            print(f"Loading optimization results from {json_filename}")
            print(f"Results from: {data['optimization_info']['timestamp']}")
            print(f"Best loss achieved: {data['optimization_info']['best_loss']:.6f}")

            # Extract connection modulation parameters
            if 'optimized_parameters' in data and 'connection_modulation' in data['optimized_parameters']:
                connection_modulation = data['optimized_parameters']['connection_modulation']

                # Apply the parameters
                changes = self.update_connection_modulation(connection_modulation)

                # Print performance summary if available
                if 'performance' in data:
                    print(f"\nExpected performance (from optimization):")
                    for drive_key, perf_data in data['performance'].items():
                        print(f"\n{drive_key.upper()}:")
                        for pop, metrics in perf_data.items():
                            print(f"  {pop.upper()}: {metrics['actual_rate']:.1f} Hz "
                                  f"(target: {metrics['target_rate']:.1f} Hz), "
                                  f"sparsity: {metrics['sparsity']:.3f}")

                return changes

            else:
                print("Error: JSON file does not contain expected 'connection_modulation' data")
                return None

        except FileNotFoundError:
            print(f"Error: Could not find file {json_filename}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_filename}")
            return None
        except KeyError as e:
            print(f"Error: Missing expected key in JSON file: {e}")
            return None
    
def load_optimization_results(filename):
    """Load optimization results from JSON file"""
    
    import json
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded results from {filename}")
    print(f"Optimization completed: {data['optimization_info']['timestamp']}")
    print(f"Best loss: {data['optimization_info']['best_loss']:.6f}")
    print(f"Best iteration: {data['optimization_info']['best_iteration']}")
    
    return data
    
def test_circuit():
    """Test the dentate circuit"""
    print("Testing Dentate Circuit with Dendritic-Somatic Transfer...")
    
    # Create circuit parameters
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    # Create circuit
    circuit = DentateCircuit(circuit_params, synaptic_params, opsin_params)
    
    # Test basic operation
    direct_activation = {
        'gc': torch.randn(circuit_params.n_gc) * 50.0,  # pA
        'pv': torch.randn(circuit_params.n_pv) * 30.0,
    }
    
    external_drive = {
        'mec': torch.randn(circuit_params.n_mec) * 100.0
    }
    
    print("Running forward pass...")
    activities = circuit(direct_activation, external_drive)
    
    print("Activity levels:")
    for pop, activity in activities.items():
        print(f"  {pop.upper()}: {torch.mean(activity):.2f} ± {torch.std(activity):.2f} Hz")
    
    # Get detailed states
    states = circuit.get_dendritic_somatic_states()
    
    print("\nDendritic-somatic states:")
    for pop, pop_states in states.items():
        print(f"  {pop.upper()}:")
        print(f"    V_dendrite: {torch.mean(pop_states['v_dendrite']):.1f} ± {torch.std(pop_states['v_dendrite']):.1f} mV")
        print(f"    V_soma: {torch.mean(pop_states['v_soma']):.1f} ± {torch.std(pop_states['v_soma']):.1f} mV")
        print(f"    Adaptation: {torch.mean(pop_states['adaptation']):.3f} ± {torch.std(pop_states['adaptation']):.3f}")

    conductance_stats = circuit.get_conductance_statistics()

    print("\nConductance statistics:")
    for conn_name, stats in conductance_stats.items():
        if stats['n_connections'] > 0:
            print(f"  {conn_name} ({stats['synapse_type']}):")
            print(f"    Connections: {stats['n_connections']}")
            print(f"    Conductance: {stats['mean']:.3f} ± {stats['std']:.3f} nS")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}] nS")

        
    return circuit, activities, states

if __name__ == "__main__":
    # Test the enhanced implementation
    circuit, activities, states = test_circuit()
