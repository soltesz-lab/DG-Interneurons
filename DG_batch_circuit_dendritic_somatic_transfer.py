#!/usr/bin/env python3
"""
Batch Dentate Gyrus Circuit for Efficient GPU Parallel Evaluation

Extends the single-circuit model to support batch dimension for evaluating
multiple parameter configurations simultaneously on GPU.

Design principles:
- Connectivity matrices are shared across batch (same circuit topology)
- Connection modulation parameters vary across batch
- All state variables have shape [batch_size, n_neurons]
- Backward compatible with single-circuit evaluation (batch_size=1)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from DG_circuit_dendritic_somatic_transfer import (
    CircuitParams, OpsinParams, PerConnectionSynapticParams,
    SpatialLayout, ConnectivityMatrix, get_default_device
)

from dendritic_somatic_transfer import (
    dendritic_somatic_transfer,
    get_cell_type_parameters,
    DendriticParameters
)


class BatchSynapticStateManager:
    """
    Manages synaptic state variables with batch dimension
    
    All states have shape [batch_size, n_pre, n_post]
    Connectivity matrices are shared across batch
    """
    
    def __init__(self, batch_size: int, circuit_params, synaptic_params,
                 connectivity_matrices, device: Optional[torch.device] = None):
        self.batch_size = batch_size
        self.circuit_params = circuit_params
        self.synaptic_params = synaptic_params
        self.dt = circuit_params.dt
        self.device = device if device is not None else get_default_device()
        
        # Initialize batch synaptic state variables
        self.synaptic_states = {}
        
        for conn_name, cond_matrix in connectivity_matrices.items():
            n_pre, n_post = cond_matrix.connectivity.shape
            synapse_type = cond_matrix.synapse_type
            
            # Create batch state variables: [batch_size, n_pre, n_post]
            states = {
                'ampa_conductance': torch.zeros(batch_size, n_pre, n_post, device=self.device),
                'gaba_conductance': torch.zeros(batch_size, n_pre, n_post, device=self.device),
                'nmda_conductance': torch.zeros(batch_size, n_pre, n_post, device=self.device),
            }
            
            self.synaptic_states[conn_name] = {
                'states': states,
                'synapse_type': synapse_type,
                'max_conductances': cond_matrix.conductances.to(self.device),  # [n_pre, n_post]
                'connectivity': cond_matrix.connectivity.to(self.device)  # [n_pre, n_post]
            }
    
    def update_synaptic_states(self, activities: Dict[str, torch.Tensor],
                               connection_modulation_batch: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update all synaptic state variables based on presynaptic firing rates
        
        Args:
            activities: Dict mapping pop_name -> activity tensor [batch_size, n_neurons]
            connection_modulation_batch: Optional dict mapping conn_name -> modulation [batch_size]
                                        If None, uses base conductances without modulation
        """
        # Calculate decay factors (shared across batch)
        tensor_dt = torch.tensor(self.dt, device=self.device)
        ampa_decay = torch.exp(-tensor_dt / self.synaptic_params.tau_ampa)
        gaba_decay = torch.exp(-tensor_dt / self.synaptic_params.tau_gaba) 
        nmda_decay = torch.exp(-tensor_dt / self.synaptic_params.tau_nmda)
        
        for conn_name, syn_data in self.synaptic_states.items():
            parts = conn_name.split('_')
            pre_pop, post_pop = parts[0], parts[1]
            
            if pre_pop not in activities:
                continue
                
            pre_rates = activities[pre_pop]  # [batch_size, n_pre]
            states = syn_data['states']
            max_conductances = syn_data['max_conductances']  # [n_pre, n_post]
            connectivity = syn_data['connectivity']  # [n_pre, n_post]
            synapse_type = syn_data['synapse_type']
            
            # Apply per-batch connection modulation if provided
            if connection_modulation_batch is not None and conn_name in connection_modulation_batch:
                # modulation: [batch_size] -> [batch_size, 1, 1] for broadcasting
                modulation = connection_modulation_batch[conn_name].view(self.batch_size, 1, 1)
                effective_conductances = max_conductances.unsqueeze(0) * modulation
            else:
                # No modulation, broadcast base conductances
                effective_conductances = max_conductances.unsqueeze(0)  # [1, n_pre, n_post]
            
            # Calculate synaptic input: [batch, n_pre, 1] * [batch, n_pre, n_post] * [1, n_pre, n_post]
            # pre_rates: [batch, n_pre] -> [batch, n_pre, 1]
            # effective_conductances: [batch, n_pre, n_post]
            # connectivity: [n_pre, n_post] -> broadcast to [batch, n_pre, n_post]
            synaptic_input = (pre_rates.unsqueeze(2) * 
                            effective_conductances * 
                            connectivity.unsqueeze(0))
            
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
        """
        Get total conductances for a specific connection
        
        Returns:
            Dict with keys 'ampa', 'gaba', 'nmda', each tensor shape [batch_size, n_post]
        """
        if conn_name not in self.synaptic_states:
            zero = torch.zeros(self.batch_size, 1, device=self.device)
            return {'ampa': zero, 'gaba': zero, 'nmda': zero}
            
        states = self.synaptic_states[conn_name]['states']
        
        # Sum over presynaptic neurons: [batch, n_pre, n_post] -> [batch, n_post]
        return {
            'ampa': torch.sum(states['ampa_conductance'], dim=1),
            'gaba': torch.sum(states['gaba_conductance'], dim=1), 
            'nmda': torch.sum(states['nmda_conductance'], dim=1)
        }
    
    def reset_states(self):
        """Reset all synaptic states to zero"""
        for syn_data in self.synaptic_states.values():
            for state_var in syn_data['states'].values():
                state_var.zero_()


class BatchDentateCircuit(nn.Module):
    """
    Batch dentate gyrus circuit for parallel evaluation of multiple parameter sets
    
    Key differences from single circuit:
    - All state variables have shape [batch_size, n_neurons]
    - Connectivity matrices are shared (same topology across batch)
    - Connection modulation parameters can vary per batch element
    - Forward pass processes all batch elements simultaneously
    
    Usage:
        circuit = BatchDentateCircuit(
            batch_size=32,
            circuit_params=params,
            synaptic_params=base_synaptic_params,
            opsin_params=opsin_params,
            device=torch.device('cuda')
        )
        
        # Set per-batch connection modulation
        circuit.set_connection_modulation_batch(modulation_dict)
        
        # Run simulation
        activities = circuit(direct_activation, external_drive)
        # Returns: {'gc': [32, 1000], 'mc': [32, 30], ...}
    """
    
    def __init__(self,
                 batch_size: int,
                 circuit_params: CircuitParams,
                 synaptic_params: PerConnectionSynapticParams,
                 opsin_params: OpsinParams,
                 device: Optional[torch.device] = None,
                 inference_mode: bool = True,
                 compile_circuit: bool = True):
        super().__init__()
        
        self.batch_size = batch_size
        self.device = device if device is not None else get_default_device()
        print(f"Initializing BatchDentateCircuit (batch_size={batch_size}) on device: {self.device}")
        
        self.circuit_params = circuit_params
        self.synaptic_params = synaptic_params
        self.opsin_params = opsin_params

        # Create spatial layout and connectivity (shared across batch)
        self.layout = SpatialLayout(circuit_params, device=self.device)
        self.connectivity = ConnectivityMatrix(
            circuit_params, self.layout, self.synaptic_params, device=self.device
        )
        
        # Register connectivity matrices as buffers (shared across batch)
        for name, cond_matrix in self.connectivity.conductance_matrices.items():
            self.register_buffer(f'conn_{name}', cond_matrix.connectivity)
            self.register_buffer(f'cond_{name}', cond_matrix.conductances)
            
        # Get cell-type specific dendritic parameters
        self.dendritic_params = get_cell_type_parameters()
        
        # Batch population activities: [batch_size, n_neurons]
        self.register_buffer('gc_activity', torch.zeros(batch_size, circuit_params.n_gc, device=self.device))
        self.register_buffer('mc_activity', torch.zeros(batch_size, circuit_params.n_mc, device=self.device))
        self.register_buffer('pv_activity', torch.zeros(batch_size, circuit_params.n_pv, device=self.device))
        self.register_buffer('sst_activity', torch.zeros(batch_size, circuit_params.n_sst, device=self.device))
        self.register_buffer('mec_activity', torch.zeros(batch_size, circuit_params.n_mec, device=self.device))
        
        # Batch dendritic-somatic state variables: [batch_size, n_neurons]
        self.register_buffer('gc_adaptation', torch.zeros(batch_size, circuit_params.n_gc, device=self.device))
        self.register_buffer('mc_adaptation', torch.zeros(batch_size, circuit_params.n_mc, device=self.device))
        self.register_buffer('pv_adaptation', torch.zeros(batch_size, circuit_params.n_pv, device=self.device))
        self.register_buffer('sst_adaptation', torch.zeros(batch_size, circuit_params.n_sst, device=self.device))
        self.register_buffer('mec_adaptation', torch.zeros(batch_size, circuit_params.n_mec, device=self.device))
        
        self.register_buffer('gc_v_dendrite', torch.zeros(batch_size, circuit_params.n_gc, device=self.device))
        self.register_buffer('mc_v_dendrite', torch.zeros(batch_size, circuit_params.n_mc, device=self.device))
        self.register_buffer('pv_v_dendrite', torch.zeros(batch_size, circuit_params.n_pv, device=self.device))
        self.register_buffer('sst_v_dendrite', torch.zeros(batch_size, circuit_params.n_sst, device=self.device))
        
        self.register_buffer('gc_v_soma', torch.zeros(batch_size, circuit_params.n_gc, device=self.device))
        self.register_buffer('mc_v_soma', torch.zeros(batch_size, circuit_params.n_mc, device=self.device))
        self.register_buffer('pv_v_soma', torch.zeros(batch_size, circuit_params.n_pv, device=self.device))
        self.register_buffer('sst_v_soma', torch.zeros(batch_size, circuit_params.n_sst, device=self.device))

        # Initialize batch synaptic state manager
        self.synaptic_state_manager = BatchSynapticStateManager(
            batch_size,
            self.circuit_params,
            self.synaptic_params, 
            self.connectivity.conductance_matrices,
            device=self.device
        )
        
        # Add NMDA fraction parameter if not present
        if not hasattr(self.synaptic_params, 'nmda_fraction'):
            self.synaptic_params.nmda_fraction = 0.3
        
        # Storage for per-batch connection modulation
        # Dict mapping conn_name -> tensor of shape [batch_size]
        self.connection_modulation_batch = None

        self._inference_mode = inference_mode
        self.set_inference_mode(inference_mode)

        if compile_circuit:
            
            # Compile batch dendritic-somatic transfer
            self._batch_dendritic_somatic_transfer = torch.compile(
                self._batch_dendritic_somatic_transfer,
                mode='reduce-overhead',
            #    fullgraph=True  # This method is pure tensor operations
            )
            
            # Optionally compile synaptic state manager
            if hasattr(self, 'synaptic_state_manager'):
                self.synaptic_state_manager.update_synaptic_states = torch.compile(
                    self.synaptic_state_manager.update_synaptic_states,
                    mode='reduce-overhead'
                )

    def set_inference_mode(self, enabled: bool = True):
        # Disable or enable gradients for all parameters and buffers
        for param in self.parameters():
            param.requires_grad_(not enabled)
    
        for buf in self.buffers():
            buf.requires_grad_(not enabled)

        # Set eval/train mode
        if enabled:
            self.eval()
        else:
            self.train()
    
        self._inference_mode = enabled
        
                
    def set_connection_modulation_batch(self, modulation_list: List[Dict[str, float]]):
        """
        Set connection modulation parameters for each batch element
        
        Args:
            modulation_list: List of length batch_size, each element is a dict
                           mapping connection_name -> modulation_factor
                           
        Example:
            modulation_list = [
                {'gc_mc': 1.5, 'pv_gc': 2.0, ...},  # Batch element 0
                {'gc_mc': 1.2, 'pv_gc': 1.8, ...},  # Batch element 1
                ...
            ]
        """
        assert len(modulation_list) == self.batch_size, \
            f"Expected {self.batch_size} modulation dicts, got {len(modulation_list)}"
        
        # Convert list of dicts to dict of tensors
        self.connection_modulation_batch = {}
        
        # Get all connection names from first dict
        conn_names = modulation_list[0].keys()
        
        for conn_name in conn_names:
            # Stack modulation values across batch: [batch_size]
            values = torch.tensor(
                [mod_dict[conn_name] for mod_dict in modulation_list],
                device=self.device,
                dtype=torch.float32
            )
            self.connection_modulation_batch[conn_name] = values
    
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
        self.synaptic_state_manager.reset_states()

    def update_activity_with_dendritic_somatic(self, 
                                               direct_activation: Dict[str, torch.Tensor], 
                                               external_drive: Dict[str, torch.Tensor] = None):
        """
        Updated activity update using conductance-based dendritic-somatic transfer
        
        All inputs should have shape [batch_size, n_neurons]
        """
        if external_drive is None:
            external_drive = {}

        # Step 1: Update synaptic states with per-batch modulation
        activities = {
            'gc': self.gc_activity,
            'mc': self.mc_activity,
            'pv': self.pv_activity,
            'sst': self.sst_activity,
            'mec': self.mec_activity
        }

        self.synaptic_state_manager.update_synaptic_states(
            activities,
            connection_modulation_batch=self.connection_modulation_batch
        )

        # Step 2: Update each population using batch dendritic-somatic transfer
        for pop in ['gc', 'mc', 'pv', 'sst']:

            # Get current states [batch_size, n_neurons]
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
            # Each will be [batch_size, n_post]
            total_ampa = torch.zeros_like(current_activity)
            total_gaba = torch.zeros_like(current_activity)  
            total_nmda = torch.zeros_like(current_activity)

            # Sum conductances from all incoming connections
            for conn_name in self.synaptic_state_manager.synaptic_states.keys():
                parts = conn_name.split('_')
                if len(parts) >= 2 and parts[1] == pop:
                    conductances = self.synaptic_state_manager.get_total_conductances(conn_name)
                    total_ampa += conductances['ampa']
                    total_gaba += conductances['gaba']
                    total_nmda += conductances['nmda']

            # Add direct activation as additional AMPA-like conductance
            direct_current = direct_activation.get(pop, torch.zeros_like(current_activity))
            direct_conductance = torch.clamp(direct_current / 70.0, min=0.0)
            total_ampa += direct_conductance
            
            # Apply batch dendritic-somatic transfer
            params = self.dendritic_params[pop]
            new_activity, new_adaptation, new_v_dendrite, new_v_soma = \
                self._batch_dendritic_somatic_transfer(
                    total_ampa, total_gaba, total_nmda, params, adaptation_state
                )

            # Update state variables
            if pop == 'gc':
                self.gc_activity = new_activity
                self.gc_adaptation = new_adaptation
                self.gc_v_dendrite = new_v_dendrite
                self.gc_v_soma = new_v_soma
            elif pop == 'mc':
                self.mc_activity = new_activity
                self.mc_adaptation = new_adaptation
                self.mc_v_dendrite = new_v_dendrite
                self.mc_v_soma = new_v_soma
            elif pop == 'pv':
                self.pv_activity = new_activity
                self.pv_adaptation = new_adaptation
                self.pv_v_dendrite = new_v_dendrite
                self.pv_v_soma = new_v_soma
            elif pop == 'sst':
                self.sst_activity = new_activity
                self.sst_adaptation = new_adaptation
                self.sst_v_dendrite = new_v_dendrite
                self.sst_v_soma = new_v_soma

        # Handle MEC separately
        mec_drive = external_drive.get('mec', torch.zeros_like(self.mec_activity))
        mec_direct = direct_activation.get('mec', torch.zeros_like(self.mec_activity))
        mec_total_ampa = torch.clamp((mec_drive + mec_direct) / 70.0, min=0.0)
        mec_gaba = torch.zeros_like(mec_total_ampa)
        mec_nmda = torch.zeros_like(mec_total_ampa)

        mec_activity, mec_adaptation, _, _ = self._batch_dendritic_somatic_transfer(
            mec_total_ampa, mec_gaba, mec_nmda, 
            self.dendritic_params['mec'], self.mec_adaptation
        )

        self.mec_activity = mec_activity
        self.mec_adaptation = mec_adaptation

    def _batch_dendritic_somatic_transfer(self,
                                           ampa_conductances: torch.Tensor,
                                           gaba_conductances: torch.Tensor, 
                                           nmda_conductances: torch.Tensor,
                                           params: DendriticParameters, 
                                           adaptation_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Vectorized batch dendritic-somatic transfer function
        
        All inputs have shape [batch_size, n_neurons]
        Process each neuron across all batches simultaneously
        
        Returns:
            Tuple of (firing_rate, adaptation, v_dendrite, v_soma)
            Each with shape [batch_size, n_neurons]
        """
        batch_size, n_neurons = ampa_conductances.shape
        
        # Flatten batch and neurons for vectorized processing
        # [batch_size, n_neurons] -> [batch_size * n_neurons]
        ampa_flat = ampa_conductances.reshape(-1)
        gaba_flat = gaba_conductances.reshape(-1)
        nmda_flat = nmda_conductances.reshape(-1)
        adapt_flat = adaptation_state.reshape(-1)
        
        # Call existing dendritic_somatic_transfer
        firing_rate_flat, states = dendritic_somatic_transfer(
            ampa_flat, gaba_flat, nmda_flat,
            params, adapt_flat, device=self.device
        )
        
        # Reshape back to [batch_size, n_neurons]
        firing_rate = firing_rate_flat.reshape(batch_size, n_neurons)
        adaptation = states['adaptation'].reshape(batch_size, n_neurons)
        v_dendrite = states['v_dendrite'].reshape(batch_size, n_neurons)
        v_soma = states['v_soma'].reshape(batch_size, n_neurons)
        
        return firing_rate, adaptation, v_dendrite, v_soma

    def forward(self, direct_activation: Dict[str, Tensor], 
                external_drive: Dict[str, Tensor] = None) -> Dict[str, Tensor]:
        """
        Single time step forward pass with batch dendritic-somatic processing
        
        Args:
            direct_activation: Dict mapping pop_name -> activation [batch_size, n_neurons]
            external_drive: Dict mapping pop_name -> drive [batch_size, n_neurons]
            
        Returns:
            Dict mapping pop_name -> activity [batch_size, n_neurons]
        """
        self.update_activity_with_dendritic_somatic(direct_activation, external_drive)
        
        return {
            'gc': self.gc_activity.clone(),
            'mc': self.mc_activity.clone(), 
            'pv': self.pv_activity.clone(),
            'sst': self.sst_activity.clone(),
            'mec': self.mec_activity.clone()
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'device': str(self.device),
                'batch_size': self.batch_size
            }
        else:
            # Estimate CPU memory
            total_params = sum(p.numel() * p.element_size() for p in self.parameters())
            total_buffers = sum(b.numel() * b.element_size() for b in self.buffers())
            total_mb = (total_params + total_buffers) / 1024**2
            return {
                'estimated_mb': total_mb,
                'device': str(self.device),
                'batch_size': self.batch_size
            }


class BatchCircuitEvaluator:
    """
    Evaluates multiple parameter configurations in parallel using batched simulation
    
    Provides high-level interface for optimization algorithms to evaluate
    batches of connection modulation parameters efficiently on GPU.
    """
    
    def __init__(self,
                 circuit_params,
                 base_synaptic_params,
                 opsin_params,
                 targets,
                 config,
                 device: Optional[torch.device] = None):
        """
        Initialize batched circuit evaluator
        
        Args:
            circuit_params: CircuitParams instance
            base_synaptic_params: PerConnectionSynapticParams instance
            opsin_params: OpsinParams instance
            targets: OptimizationTargets instance (from DG_circuit_optimization)
            config: OptimizationConfig instance (from DG_circuit_optimization)
            device: Device to run simulations on
        """
        self.circuit_params = circuit_params
        self.base_synaptic_params = base_synaptic_params
        self.opsin_params = opsin_params
        self.targets = targets
        self.config = config
        self.device = device if device is not None else get_default_device()
        
        print(f"BatchCircuitEvaluator initialized on device: {self.device}")
    
    def evaluate_parameter_batch(self,
                                 parameter_batch: List[Dict[str, float]],
                                 mec_drive: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate a batch of parameter configurations in parallel
        
        Args:
            parameter_batch: List of connection_modulation dicts, length = batch_size
                           Each dict maps connection_name -> modulation_factor
            mec_drive: MEC drive level (pA)
            
        Returns:
            losses: Tensor of shape [batch_size] with loss for each configuration
            firing_rates_batch: Dict mapping pop_name -> firing rates [batch_size]
        """
        batch_size = len(parameter_batch)
        
        # Create batched circuit
        circuit = BatchDentateCircuit(
            batch_size=batch_size,
            circuit_params=self.circuit_params,
            synaptic_params=self.base_synaptic_params,
            opsin_params=self.opsin_params,
            device=self.device
        )
        
        # Set per-batch connection modulation
        circuit.set_connection_modulation_batch(parameter_batch)
        
        # Run simulation
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
                        # activities[pop] has shape [batch_size, n_neurons]
                        activities_over_time[pop].append(activities[pop])
        
        # Calculate statistics and losses
        # Stack time series: [time_steps, batch_size, n_neurons]
        losses = torch.zeros(batch_size, device=self.device)
        firing_rates_batch = {pop: torch.zeros(batch_size, device=self.device) 
                             for pop in ['gc', 'mc', 'pv', 'sst']}
        
        for pop in activities_over_time:
            if len(activities_over_time[pop]) > 0:
                # Stack: [time, batch, neurons] -> mean over time: [batch, neurons]
                pop_time_series = torch.stack(activities_over_time[pop], dim=0)
                mean_rates = torch.mean(pop_time_series, dim=0)  # [batch, neurons]
                
                # Population average firing rate per batch element: [batch]
                pop_firing_rates = torch.mean(mean_rates, dim=1)
                firing_rates_batch[pop] = pop_firing_rates
                
                # Calculate loss components for this population
                if pop in self.targets.target_rates:
                    target_rate = self.targets.target_rates[pop]
                    tolerance = self.targets.rate_tolerance[pop]
                    
                    # Vectorized loss calculation across batch
                    errors = torch.abs(pop_firing_rates - target_rate)
                    
                    # Huber loss with tolerance
                    rate_losses = torch.where(
                        errors <= tolerance,
                        0.5 * errors ** 2,
                        tolerance * errors - 0.5 * tolerance ** 2
                    )
                    
                    losses += rate_losses
                
                # Sparsity loss
                if pop in self.targets.sparsity_targets:
                    target_sparsity = self.targets.sparsity_targets[pop]
                    # Sparsity per batch element: [batch]
                    actual_sparsity = torch.sum(
                        mean_rates > self.targets.activity_threshold, 
                        dim=1
                    ).float() / mean_rates.shape[1]
                    
                    sparsity_errors = (actual_sparsity - target_sparsity) ** 2
                    losses += sparsity_errors * self.targets.loss_weights['sparsity']
        
        # Add constraint violations (requires converting to CPU for evaluation)
        # Import here to avoid circular dependency
        from DG_circuit_optimization import evaluate_rate_ordering_constraints
        
        for b in range(batch_size):
            firing_rates_dict = {pop: firing_rates_batch[pop][b].item() 
                                for pop in firing_rates_batch}
            constraint_violation, _ = evaluate_rate_ordering_constraints(
                firing_rates_dict,
                self.targets.rate_ordering_constraints
            )
            losses[b] += self.targets.constraint_violation_weight * constraint_violation
        
        return losses, firing_rates_batch

    
def test_batch_circuit(batch_size: int = 8,
                        n_steps: int = 100,
                        device: Optional[torch.device] = None,
                        verbose: bool = True):
    """
    Test batch circuit simulation
    
    Args:
        batch_size: Number of circuits to simulate in parallel
        n_steps: Number of simulation timesteps
        device: Device to run on (None for auto-detect)
        verbose: Whether to print detailed output
    
    Returns:
        circuit: The batch circuit
        activities: Final activities [batch_size, n_neurons]
    """
    if device is None:
        device = get_default_device()
    
    if verbose:
        print("="*60)
        print("Testing Batch Dentate Circuit")
        print("="*60)
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print()
    
    # Create circuit parameters
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    # Create batch circuit
    import time
    start = time.time()
    circuit = BatchDentateCircuit(
        batch_size=batch_size,
        circuit_params=circuit_params,
        synaptic_params=synaptic_params,
        opsin_params=opsin_params,
        device=device
    )
    init_time = time.time() - start
    
    if verbose:
        print(f"Batch circuit initialized in {init_time:.3f}s")
        mem_usage = circuit.get_memory_usage()
        if 'allocated_mb' in mem_usage:
            print(f"GPU memory: {mem_usage['allocated_mb']:.1f} MB allocated")
        else:
            print(f"Estimated memory: {mem_usage['estimated_mb']:.1f} MB")
        print()
    
    # Create different connection modulation for each batch element
    base_modulation = synaptic_params.connection_modulation
    modulation_list = []
    for b in range(batch_size):
        # Vary modulation slightly for each batch element
        batch_modulation = {
            name: value * (1.0 + 0.2 * (b / batch_size - 0.5))
            for name, value in base_modulation.items()
        }
        modulation_list.append(batch_modulation)
    
    circuit.set_connection_modulation_batch(modulation_list)
    
    # Create batch inputs [batch_size, n_neurons]
    direct_activation = {
        'gc': torch.randn(batch_size, circuit_params.n_gc, device=device) * 50.0,
        'pv': torch.randn(batch_size, circuit_params.n_pv, device=device) * 30.0,
    }
    
    external_drive = {
        'mec': torch.randn(batch_size, circuit_params.n_mec, device=device) * 100.0
    }
    
    # Run simulation
    if verbose:
        print(f"Running {n_steps} simulation steps...")
    
    start = time.time()
    for step in range(n_steps):
        activities = circuit(direct_activation, external_drive)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    sim_time = time.time() - start
    
    if verbose:
        print(f"Simulation completed in {sim_time:.3f}s")
        print(f"  Time per step: {sim_time/n_steps*1000:.2f} ms")
        print(f"  Time per circuit per step: {sim_time/n_steps/batch_size*1000:.2f} ms")
        print(f"  Throughput: {n_steps*batch_size/sim_time:.1f} circuit-steps/sec")
        print()
        
        print("Final activity levels (mean across batch):")
        for pop, activity in activities.items():
            # activity shape: [batch_size, n_neurons]
            mean_across_batch = torch.mean(activity, dim=0)  # [n_neurons]
            print(f"  {pop.upper()}: {torch.mean(mean_across_batch):.2f} +/- {torch.std(mean_across_batch):.2f} Hz")
        
        print("\nActivity variance across batch:")
        for pop, activity in activities.items():
            # Variance across batch dimension
            batch_means = torch.mean(activity, dim=1)  # [batch_size]
            print(f"  {pop.upper()}: {torch.mean(batch_means):.2f} (std={torch.std(batch_means):.2f})")
    
    return circuit, activities


def benchmark_batch_vs_sequential(batch_sizes=[1, 4, 8, 16, 32],
                                  n_steps=100,
                                  device: Optional[torch.device] = None):
    """
    Benchmark batch circuit against sequential evaluation
    
    Shows speedup from batching on GPU
    """
    if device is None:
        device = get_default_device()
    
    print("="*60)
    print("Batch vs Sequential Benchmark")
    print("="*60)
    print(f"Device: {device}")
    print(f"Steps per circuit: {n_steps}")
    print()
    
    circuit_params = CircuitParams()
    opsin_params = OpsinParams()
    synaptic_params = PerConnectionSynapticParams()
    
    # Baseline: sequential evaluation (batch_size=1)
    print("Baseline: Sequential evaluation (batch_size=1)")
    circuit_seq = BatchDentateCircuit(1, circuit_params, synaptic_params, 
                                      opsin_params, device=device)
    
    modulation_list = [synaptic_params.connection_modulation]
    circuit_seq.set_connection_modulation_batch(modulation_list)
    
    direct_activation = {
        'gc': torch.randn(1, circuit_params.n_gc, device=device) * 50.0,
    }
    external_drive = {
        'mec': torch.randn(1, circuit_params.n_mec, device=device) * 100.0
    }
    
    import time
    
    # Warmup
    for _ in range(10):
        _ = circuit_seq(direct_activation, external_drive)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_steps):
        _ = circuit_seq(direct_activation, external_drive)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    baseline_time = time.time() - start
    print(f"  Time: {baseline_time:.3f}s")
    print(f"  Throughput: {n_steps/baseline_time:.1f} circuit-steps/sec")
    print()
    
    # Test different batch sizes
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'Throughput':<20} {'Speedup':<12}")
    print("-"*60)
    
    for batch_size in batch_sizes:
        if batch_size == 1:
            # Already computed
            speedup = 1.0
            throughput = n_steps / baseline_time
            print(f"{batch_size:<12} {baseline_time:<12.3f} {throughput:<20.1f} {speedup:<12.2f}x")
            continue
        
        circuit_batch = BatchDentateCircuit(batch_size, circuit_params, 
                                            synaptic_params, opsin_params, device=device)
        
        modulation_list = [synaptic_params.connection_modulation] * batch_size
        circuit_batch.set_connection_modulation_batch(modulation_list)
        
        direct_activation_batch = {
            'gc': torch.randn(batch_size, circuit_params.n_gc, device=device) * 50.0,
        }
        external_drive_batch = {
            'mec': torch.randn(batch_size, circuit_params.n_mec, device=device) * 100.0
        }
        
        # Warmup
        for _ in range(10):
            _ = circuit_batch(direct_activation_batch, external_drive_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_steps):
            _ = circuit_batch(direct_activation_batch, external_drive_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        batch_time = time.time() - start
        
        # Calculate speedup relative to sequential
        # Sequential would take: baseline_time * batch_size for same total circuits
        sequential_equivalent_time = baseline_time * batch_size
        speedup = sequential_equivalent_time / batch_time
        throughput = (n_steps * batch_size) / batch_time
        
        print(f"{batch_size:<12} {batch_time:<12.3f} {throughput:<20.1f} {speedup:<12.2f}x")
        
        # Clean up
        del circuit_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print()
    print("Note: Speedup is relative to sequential evaluation of the same")
    print("      total number of circuits (batch_size * n_steps)")


if __name__ == "__main__":
    # Test batch circuit
    print("Testing batch circuit implementation...")
    circuit, activities = test_batch_circuit(batch_size=16, n_steps=100)
    
    # Run benchmark if GPU available
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("Running benchmark on GPU...")
        benchmark_batch_vs_sequential(
            batch_sizes=[1, 4, 8, 16, 32, 64],
            n_steps=100,
            device=torch.device('cuda')
        )
    else:
        print("\n" + "="*60)
        print("Running benchmark on CPU...")
        benchmark_batch_vs_sequential(
            batch_sizes=[1, 4, 8, 16, 32, 64],
            n_steps=100,
            device=torch.device('cpu')
        )
        
