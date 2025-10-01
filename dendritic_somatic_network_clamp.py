#!/usr/bin/env python3
"""
Network Clamp Testing Framework - Conductance-Based Version

Uses biophysically realistic conductance-based dendritic-somatic transfer
with comprehensive testing of dendritic integration mechanisms.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm

# Import the dendritic-somatic transfer function
from dendritic_somatic_transfer import (
    dendritic_somatic_transfer, 
    get_cell_type_parameters,
    DendriticParameters,
    nmda_voltage_dependence
)

from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit, CircuitParams, OpsinParams,
    PerConnectionSynapticParams
)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class ConductanceAnalysisParams:
    """Parameters for conductance analysis experiments"""
    
    # Test ranges (now in nS for conductances)
    conductance_range: Tuple[float, float] = (0.0, 10.0)  # nS
    heterogeneity_levels: List[float] = None  # CV values
    n_test_points: int = 50
    n_trials: int = 10
    
    # Circuit test parameters
    stimulus_duration: int = 500  # ms
    baseline_duration: int = 100  # ms
    recovery_duration: int = 200  # ms
    
    def __post_init__(self):
        if self.heterogeneity_levels is None:
            self.heterogeneity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class NetworkClamp:
    """
    Network clamp testing framework for conductance-based dendritic-somatic transfer
    """
    
    def __init__(self, figsize=(15, 12), optimization_json_file=None):
        self.figsize = figsize
        self.cell_params = get_cell_type_parameters()
        self.analysis_params = ConductanceAnalysisParams()
        
        # Color schemes
        self.excitatory_color = '#e74c3c'
        self.inhibitory_color = '#3498db'
        self.connection_colors = {
            'mec_gc': '#e67e22',
            'gc_mc': '#27ae60',
            'mc_gc': '#f39c12',
            'pv_gc': '#3498db',
            'sst_gc': '#9b59b6',
        }

        # Load optimized parameters if provided
        self.optimization_data = None
        self.optimized_circuit_params = CircuitParams()
        self.optimized_synaptic_params = PerConnectionSynapticParams()
    
        if optimization_json_file is not None:
            success = self._load_optimization_results(optimization_json_file)
            if success:
                print(f"NetworkClamp initialized with optimized parameters")
            else:
                print("Using default parameters")

    def _load_optimization_results(self, json_filename):
        """Load and process optimization results from JSON file"""
        import json

        try:
            with open(json_filename, 'r') as f:
                self.optimization_data = json.load(f)

            print(f"Loading from {json_filename}")
            print(f"Best loss: {self.optimization_data['optimization_info']['best_loss']:.6f}")

            self.optimized_circuit_params = self._create_optimized_circuit_params()
            self.optimized_synaptic_params = self._create_optimized_synaptic_params()

            return True

        except Exception as e:
            print(f"Error loading optimization file: {e}")
            return False

    def _create_optimized_circuit_params(self):
        """Create CircuitParams using loaded optimization data"""
        circuit_params = CircuitParams()

        if 'circuit_config' in self.optimization_data:
            config = self.optimization_data['circuit_config']
            for key in ['n_gc', 'n_mc', 'n_pv', 'n_sst', 'n_mec']:
                if key in config:
                    setattr(circuit_params, key, config[key])

        return circuit_params

    def _create_optimized_synaptic_params(self):
        """Create PerConnectionSynapticParams with optimized values"""
        if ('optimized_parameters' not in self.optimization_data or 
            'connection_modulation' not in self.optimization_data['optimized_parameters']):
            return PerConnectionSynapticParams()

        base_cond = self.optimization_data['optimized_parameters'].get('base_conductances', {})
        conn_mod = self.optimization_data['optimized_parameters']['connection_modulation']

        return PerConnectionSynapticParams(
            ampa_g_mean=base_cond.get('ampa_g_mean', 0.15),
            ampa_g_std=base_cond.get('ampa_g_std', 0.05),
            gaba_g_mean=base_cond.get('gaba_g_mean', 0.25),
            gaba_g_std=base_cond.get('gaba_g_std', 0.08),
            connection_modulation=conn_mod
        )

    def create_circuit(self, opsin_params=None):
        """Create DentateCircuit with synaptic state manager"""
        if opsin_params is None:
            opsin_params = OpsinParams()

        if self.optimized_circuit_params and self.optimized_synaptic_params:
            print("Creating circuit with optimized parameters")
            circuit = DentateCircuit(
                self.optimized_circuit_params,
                self.optimized_synaptic_params, 
                opsin_params
            )
        else:
            print("Creating circuit with default parameters")
            circuit = DentateCircuit(
                CircuitParams(), 
                PerConnectionSynapticParams(), 
                opsin_params
            )
        
        return circuit

    def test_static_transfer_curves(self, conductance_range=(0, 10), n_points=300):
        """
        Test static conductance-based transfer curves
        Converts conductance inputs (nS) to firing rates (Hz)
        """
        print("Testing conductance-based transfer curves...")
        
        # Create conductance range (nS)
        conductance_input = torch.linspace(conductance_range[0], conductance_range[1], n_points)
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()
        
        # Plot transfer curves for each cell type
        for i, (cell_type, params) in enumerate(self.cell_params.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Test pure AMPA
            firing_rates_ampa = []
            v_dendrites_ampa = []
            v_somas_ampa = []
            
            for conductance in conductance_input:
                g = torch.tensor([conductance.item()])
                fr, states = dendritic_somatic_transfer(
                    ampa_conductance=g,
                    gaba_conductance=torch.zeros_like(g),
                    nmda_conductance=torch.zeros_like(g),
                    params=params
                )
                firing_rates_ampa.append(fr.item())
                v_dendrites_ampa.append(states['v_dendrite'].item())
                v_somas_ampa.append(states['v_soma'].item())
            
            # Test AMPA + NMDA mix (70% AMPA, 30% NMDA)
            firing_rates_mixed = []
            v_dendrites_mixed = []
            nmda_factors = []
            
            for conductance in conductance_input:
                g_total = torch.tensor([conductance.item()])
                g_ampa = g_total * 0.7
                g_nmda = g_total * 0.3
                
                fr, states = dendritic_somatic_transfer(
                    ampa_conductance=g_ampa,
                    gaba_conductance=torch.zeros_like(g_total),
                    nmda_conductance=g_nmda,
                    params=params
                )
                firing_rates_mixed.append(fr.item())
                v_dendrites_mixed.append(states['v_dendrite'].item())
                nmda_factors.append(states['nmda_factor'].item())
            
            # Plot firing rate curves
            ax.plot(conductance_input.numpy(), firing_rates_ampa, 'b-', 
                   linewidth=2.5, label='AMPA only')
            ax.plot(conductance_input.numpy(), firing_rates_mixed, 'r-', 
                   linewidth=2.5, label='AMPA + NMDA')
            ax.set_xlabel('Excitatory Conductance (nS)', fontsize=10)
            ax.set_ylabel('Firing Rate (Hz)', color='b', fontsize=10)
            ax.set_title(f'{cell_type.upper()} Transfer Function', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=8)
            
            # Add voltage traces on secondary axis
            ax2 = ax.twinx()
            ax2.plot(conductance_input.numpy(), v_dendrites_mixed, 'g--', 
                    alpha=0.5, linewidth=1.5, label='V_dendrite')
            ax2.set_ylabel('Voltage (mV)', color='g', fontsize=10)
            ax2.legend(loc='upper right', fontsize=7)
        
        plt.tight_layout()
        plt.savefig('conductance_transfer_curves.png', dpi=150)
        plt.show()
        
        return conductance_input, firing_rates_ampa, firing_rates_mixed

    def test_nmda_voltage_dependence(self, cell_type='gc'):
        """Test NMDA voltage dependence with conductance inputs"""
        print(f"Testing NMDA voltage dependence for {cell_type} cells...")
        
        params = self.cell_params[cell_type]
        conductance_range = torch.linspace(0, 25, 200)  # nS
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Test different NMDA fractions
        ax1 = axes[0, 0]
        nmda_fractions = [0.0, 0.2, 0.4, 0.6]
        
        for nmda_frac in nmda_fractions:
            firing_rates = []
            for g_total in conductance_range:
                g_ampa = g_total * (1 - nmda_frac)
                g_nmda = g_total * nmda_frac
                
                fr, _ = dendritic_somatic_transfer(
                    ampa_conductance=g_ampa,
                    gaba_conductance=torch.zeros_like(g_total),
                    nmda_conductance=g_nmda,
                    params=params
                )
                firing_rates.append(fr.item())
            
            ax1.plot(conductance_range.numpy(), firing_rates, 
                    linewidth=2, label=f'NMDA={nmda_frac:.1f}')
        
        ax1.set_xlabel('Total Excitatory Conductance (nS)')
        ax1.set_ylabel('Firing Rate (Hz)')
        ax1.set_title('Effect of NMDA Fraction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # NMDA unblock vs voltage
        ax2 = axes[0, 1]
        voltage_range = torch.linspace(-80, -20, 100)
        mg_concentrations = [0.5, 1.0, 2.0, 4.0]
        
        for mg_conc in mg_concentrations:
            nmda_factors = nmda_voltage_dependence(voltage_range, mg_conc)
            ax2.plot(voltage_range.numpy(), nmda_factors.numpy(), 
                    linewidth=2, label=f'[Mg2+]={mg_conc:.1f}mM')
        
        ax2.set_xlabel('Membrane Voltage (mV)')
        ax2.set_ylabel('NMDA Unblock Factor')
        ax2.set_title('NMDA Mg2+ Block vs Voltage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Compare effective NMDA conductance
        ax3 = axes[1, 0]
        g_nmda_max = torch.tensor([10.0])  # nS
        
        for mg_conc in [0.5, 1.0, 2.0]:
            effective_nmda = []
            voltages = []
            
            for g_ampa_val in torch.linspace(0, 15, 50):
                g_ampa = torch.tensor([g_ampa_val])
                
                fr, states = dendritic_somatic_transfer(
                    ampa_conductance=g_ampa,
                    gaba_conductance=torch.zeros_like(g_ampa),
                    nmda_conductance=g_nmda_max,
                    params=params
                )
                
                effective_nmda.append(states['effective_nmda_conductance'].item())
                voltages.append(states['v_dendrite'].item())
            
            ax3.plot(voltages, effective_nmda, linewidth=2, 
                    label=f'[Mg2+]={mg_conc:.1f}mM')
        
        ax3.set_xlabel('Dendritic Voltage (mV)')
        ax3.set_ylabel('Effective NMDA Conductance (nS)')
        ax3.set_title('NMDA Conductance vs Dendritic Voltage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Temporal dynamics
        ax4 = axes[1, 1]
        time_steps = 100
        dt = 1.0  # ms
        
        # Step conductance input
        g_ampa_step = torch.zeros(time_steps)
        g_nmda_step = torch.zeros(time_steps)
        g_ampa_step[20:80] = 7.0  # nS
        g_nmda_step[20:80] = 3.0  # nS
        
        firing_rates = []
        adaptation_state = torch.tensor([0.0])
        
        for t in range(time_steps):
            fr, states = dendritic_somatic_transfer(
                ampa_conductance=g_ampa_step[t:t+1],
                gaba_conductance=torch.zeros(1),
                nmda_conductance=g_nmda_step[t:t+1],
                params=params,
                adaptation_state=adaptation_state
            )
            firing_rates.append(fr.item())
            adaptation_state = states['adaptation']
        
        time_axis = np.arange(time_steps) * dt
        ax4.plot(time_axis, firing_rates, 'b-', linewidth=2, label='Firing Rate')
        ax4.plot(time_axis, g_ampa_step.numpy(), 'r:', alpha=0.5, label='AMPA conductance')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Firing Rate (Hz) / Conductance (nS)')
        ax4.set_title('Temporal Response')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nmda_voltage_dependence_conductance.png', dpi=150)
        plt.show()

    def test_inhibition_effects(self, cell_type='gc'):
        """Test inhibitory conductance effects"""
        print(f"Testing inhibitory conductance effects for {cell_type} cells...")
        
        params = self.cell_params[cell_type]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Fixed excitation, varying inhibition
        ax1 = axes[0, 0]
        g_exc_fixed = torch.tensor([5.0])  # nS
        g_inh_range = torch.linspace(0, 20, 100)
        
        firing_rates = []
        for g_inh in g_inh_range:
            fr, _ = dendritic_somatic_transfer(
                ampa_conductance=g_exc_fixed * 0.7,
                gaba_conductance=torch.tensor([g_inh.item()]),
                nmda_conductance=g_exc_fixed * 0.3,
                params=params
            )
            firing_rates.append(fr.item())
        
        ax1.plot(g_inh_range.numpy(), firing_rates, 'b-', linewidth=2)
        ax1.set_xlabel('Inhibitory Conductance (nS)')
        ax1.set_ylabel('Firing Rate (Hz)')
        ax1.set_title(f'Inhibition Effect (Fixed Exc={g_exc_fixed.item():.1f}nS)')
        ax1.grid(True, alpha=0.3)
        
        # E/I balance
        ax2 = axes[0, 1]
        ei_ratios = [0.5, 1.0, 2.0, 4.0]
        total_conductance = torch.linspace(0, 25, 100)
        
        for ei_ratio in ei_ratios:
            firing_rates = []
            for g_total in total_conductance:
                g_exc = g_total * ei_ratio / (1 + ei_ratio)
                g_inh = g_total / (1 + ei_ratio)
                
                fr, _ = dendritic_somatic_transfer(
                    ampa_conductance=g_exc * 0.7,
                    gaba_conductance=g_inh,
                    nmda_conductance=g_exc * 0.3,
                    params=params
                )
                firing_rates.append(fr.item())
            
            ax2.plot(total_conductance.numpy(), firing_rates, 
                    linewidth=2, label=f'E/I={ei_ratio:.1f}')
        
        ax2.set_xlabel('Total Conductance (nS)')
        ax2.set_ylabel('Firing Rate (Hz)')
        ax2.set_title('E/I Balance Effects')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Divisive vs subtractive inhibition
        ax3 = axes[1, 0]
        g_exc_range = torch.linspace(0, 20, 100)
        g_inh_levels = [0, 2, 5, 10]
        
        for g_inh_val in g_inh_levels:
            firing_rates = []
            for g_exc in g_exc_range:
                fr, _ = dendritic_somatic_transfer(
                    ampa_conductance=g_exc * 0.7,
                    gaba_conductance=torch.tensor([float(g_inh_val)]),
                    nmda_conductance=g_exc * 0.3,
                    params=params
                )
                firing_rates.append(fr.item())
            
            ax3.plot(g_exc_range.numpy(), firing_rates, 
                    linewidth=2, label=f'Inhib={g_inh_val}nS')
        
        ax3.set_xlabel('Excitatory Conductance (nS)')
        ax3.set_ylabel('Firing Rate (Hz)')
        ax3.set_title('Inhibition Type Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('inhibition_effects_conductance.png', dpi=150)
        plt.show()

    def compare_cell_types(self):
        """Compare conductance-based processing across cell types"""
        print("Comparing cell types with conductance inputs...")
        
        conductance_range = torch.linspace(0, 10, 300)
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()
        
        # Main comparison
        ax1 = axes[0]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (cell_type, params) in enumerate(self.cell_params.items()):
            firing_rates = []
            
            for g_total in conductance_range:
                fr, _ = dendritic_somatic_transfer(
                    ampa_conductance=g_total * 0.7,
                    gaba_conductance=torch.zeros_like(g_total),
                    nmda_conductance=g_total * 0.3,
                    params=params
                )
                firing_rates.append(fr.item())
            
            ax1.plot(conductance_range.numpy(), firing_rates, 
                    linewidth=2.5, color=colors[i], label=cell_type.upper())
        
        ax1.set_xlabel('Excitatory Conductance (nS)')
        ax1.set_ylabel('Firing Rate (Hz)')
        ax1.set_title('Cell Type Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Max firing rates
        ax2 = axes[1]
        cell_types = list(self.cell_params.keys())
        max_rates = [self.cell_params[ct].max_firing_rate for ct in cell_types]
        ax2.bar(cell_types, max_rates, alpha=0.7, color=colors[:len(cell_types)])
        ax2.set_ylabel('Max Firing Rate (Hz)')
        ax2.set_title('Maximum Firing Rates')
        ax2.set_xticklabels(cell_types, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Somatic coupling
        ax3 = axes[2]
        couplings = [self.cell_params[ct].somatic_coupling for ct in cell_types]
        ax3.bar(cell_types, couplings, alpha=0.7, color=colors[:len(cell_types)])
        ax3.set_ylabel('Somatic Coupling')
        ax3.set_title('Dendrite-Soma Coupling')
        ax3.set_xticklabels(cell_types, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Ca2+ spikes
        ax4 = axes[3]
        ca_spikes = [1 if self.cell_params[ct].ca_spike_enabled else 0 
                     for ct in cell_types]
        ax4.bar(cell_types, ca_spikes, alpha=0.7, color=colors[:len(cell_types)])
        ax4.set_ylabel('Ca2+ Spikes Enabled')
        ax4.set_title('Dendritic Ca2+ Spikes')
        ax4.set_xticklabels(cell_types, rotation=45)
        ax4.set_yticks([0, 1])
        ax4.grid(True, alpha=0.3)
        
        # Adaptation
        ax5 = axes[4]
        adaptation = [1 if self.cell_params[ct].adaptation_enabled else 0 
                      for ct in cell_types]
        ax5.bar(cell_types, adaptation, alpha=0.7, color=colors[:len(cell_types)])
        ax5.set_ylabel('Adaptation Enabled')
        ax5.set_title('Spike Frequency Adaptation')
        ax5.set_xticklabels(cell_types, rotation=45)
        ax5.set_yticks([0, 1])
        ax5.grid(True, alpha=0.3)
        
        axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('cell_type_comparison_conductance.png', dpi=150)
        plt.show()

    def analyze_conductance_distributions(self, circuit, save_fig=True):
        """Analyze conductance distributions in the circuit"""
        print("Analyzing conductance distributions...")
        
        conductance_stats = circuit.get_conductance_statistics()
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.ravel()
        
        plot_idx = 0
        for conn_name, stats in conductance_stats.items():
            if stats['n_connections'] == 0 or plot_idx >= len(axes):
                continue
                
            cond_matrix = circuit.connectivity.conductance_matrices[conn_name]
            existing = cond_matrix.connectivity > 0
            conductances = cond_matrix.conductances[existing].numpy()
            
            if len(conductances) == 0:
                continue
            
            color = (self.excitatory_color if cond_matrix.synapse_type == 'excitatory' 
                    else self.inhibitory_color)
            
            ax = axes[plot_idx]
            ax.hist(conductances, bins=30, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(stats['mean'], color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {stats["mean"]:.3f}')
            ax.set_xlabel('Conductance (nS)')
            ax.set_ylabel('Count')
            ax.set_title(f'{conn_name} ({cond_matrix.synapse_type})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('conductance_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return conductance_stats

    def test_conductance_heterogeneity_effects(self, mec_current=200.0):
        """Test effects of different conductance heterogeneity levels"""
        print("Testing conductance heterogeneity effects...")
        
        heterogeneity_levels = self.analysis_params.heterogeneity_levels
        n_trials = self.analysis_params.n_trials
        
        results = {
            'heterogeneity': [],
            'gc_activity_mean': [],
            'gc_activity_std': [],
            'mc_activity_mean': [],
            'mc_activity_std': [],
            'pv_activity_mean': [],
            'pv_activity_std': [],
            'sst_activity_mean': [],
            'sst_activity_std': [],
            'network_correlation': [],
            'trial': []
        }
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.ravel()
        
        for het_level in heterogeneity_levels:
            print(f"  Testing heterogeneity CV={het_level:.1f}")
            
            for trial in tqdm(range(n_trials), desc=f"CV={het_level:.1f}"):
                # Create synaptic params with specified heterogeneity
                synaptic_params = self._create_heterogeneous_params(het_level)
                
                # Create circuit with synaptic state manager
                circuit = DentateCircuit(
                    self.optimized_circuit_params,
                    synaptic_params,
                    OpsinParams()
                )
                circuit.add_synaptic_state_manager()
                
                # Run test
                activities, correlation = self._run_heterogeneity_test(circuit, mec_current)
                
                # Store results
                results['heterogeneity'].append(het_level)
                results['gc_activity_mean'].append(torch.mean(activities['gc']).item())
                results['gc_activity_std'].append(torch.std(activities['gc']).item())
                results['mc_activity_mean'].append(torch.mean(activities['mc']).item())
                results['mc_activity_std'].append(torch.std(activities['mc']).item())
                results['pv_activity_mean'].append(torch.mean(activities['pv']).item())
                results['pv_activity_std'].append(torch.std(activities['pv']).item())
                results['sst_activity_mean'].append(torch.mean(activities['sst']).item())
                results['sst_activity_std'].append(torch.std(activities['sst']).item())
                results['network_correlation'].append(correlation)
                results['trial'].append(trial)
        
        self._plot_heterogeneity_results(results, axes)
        return results

    def _create_heterogeneous_params(self, heterogeneity_level):
        """Create synaptic parameters with specified heterogeneity (CV)"""
        base = self.optimized_synaptic_params
        
        ampa_std = base.ampa_g_mean * heterogeneity_level
        gaba_std = base.gaba_g_mean * heterogeneity_level
        
        return PerConnectionSynapticParams(
            ampa_g_mean=base.ampa_g_mean,
            ampa_g_std=ampa_std,
            ampa_g_min=base.ampa_g_min,
            ampa_g_max=base.ampa_g_max,
            gaba_g_mean=base.gaba_g_mean,
            gaba_g_std=gaba_std,
            gaba_g_min=base.gaba_g_min,
            gaba_g_max=base.gaba_g_max,
            distribution=base.distribution,
            connection_modulation=base.connection_modulation.copy()
        )

    def _run_heterogeneity_test(self, circuit, mec_current):
        """Run test to measure circuit activity"""
        circuit.reset_state()
        
        n_steps = self.analysis_params.stimulus_duration
        mec_drive = torch.ones(circuit.circuit_params.n_mec) * mec_current
        
        activities = []
        for t in range(n_steps):
            activity = circuit({}, {'mec': mec_drive})
            activities.append({pop: act.clone() for pop, act in activity.items()})
        
        # Average over steady state (second half)
        steady_start = n_steps // 2
        avg_activities = {}
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            pop_acts = torch.stack([activities[t][pop] for t in range(steady_start, n_steps)])
            avg_activities[pop] = torch.mean(pop_acts, dim=0)
        
        # Calculate network correlation
        pop_means = torch.tensor([
            torch.mean(avg_activities['gc']).item(),
            torch.mean(avg_activities['mc']).item(),
            torch.mean(avg_activities['pv']).item(),
            torch.mean(avg_activities['sst']).item()
        ])
        
        correlation = (torch.std(pop_means) / torch.mean(pop_means)).item() if torch.mean(pop_means) > 0 else 0.0
        
        return avg_activities, correlation

    def _plot_heterogeneity_results(self, results, axes):
        """Plot heterogeneity analysis results"""
        df = pd.DataFrame(results)
        
        # Population activity vs heterogeneity
        ax1 = axes[0]
        for pop, color in zip(['gc', 'mc', 'pv', 'sst'], ['blue', 'red', 'green', 'purple']):
            mean_col = f'{pop}_activity_mean'
            grouped = df.groupby('heterogeneity')[mean_col]
            means = grouped.mean()
            stds = grouped.std()
            
            ax1.errorbar(means.index, means.values, yerr=stds.values,
                        label=pop.upper(), color=color, marker='o', linewidth=2)
        
        ax1.set_xlabel('Conductance Heterogeneity (CV)')
        ax1.set_ylabel('Population Activity (Hz)')
        ax1.set_title('Activity vs Heterogeneity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Activity variability
        ax2 = axes[1]
        for pop, color in zip(['gc', 'mc', 'pv', 'sst'], ['blue', 'red', 'green', 'purple']):
            std_col = f'{pop}_activity_std'
            grouped = df.groupby('heterogeneity')[std_col]
            means = grouped.mean()
            stds = grouped.std()
            
            ax2.errorbar(means.index, means.values, yerr=stds.values,
                        label=pop.upper(), color=color, marker='s', linewidth=2)
        
        ax2.set_xlabel('Conductance Heterogeneity (CV)')
        ax2.set_ylabel('Activity Variability (Hz)')
        ax2.set_title('Variability vs Heterogeneity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Network correlation
        ax3 = axes[2]
        grouped = df.groupby('heterogeneity')['network_correlation']
        means = grouped.mean()
        stds = grouped.std()
        
        ax3.errorbar(means.index, means.values, yerr=stds.values,
                    color='black', marker='d', linewidth=2)
        ax3.set_xlabel('Conductance Heterogeneity (CV)')
        ax3.set_ylabel('Network Correlation')
        ax3.set_title('Network Correlation vs Heterogeneity')
        ax3.grid(True, alpha=0.3)
        
        # GC distribution for select heterogeneity levels
        ax4 = axes[3]
        het_levels = [0.0, 0.4, 0.8]
        for i, het_level in enumerate(het_levels):
            subset = df[df['heterogeneity'] == het_level]['gc_activity_mean']
            ax4.hist(subset, bins=20, alpha=0.6, label=f'CV={het_level:.1f}',
                    color=plt.cm.viridis(i/len(het_levels)))
        
        ax4.set_xlabel('GC Activity (Hz)')
        ax4.set_ylabel('Count')
        ax4.set_title('GC Activity Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # E/I ratio
        ax5 = axes[4]
        df['ei_ratio'] = ((df['gc_activity_mean'] + df['mc_activity_mean']) / 
                          (df['pv_activity_mean'] + df['sst_activity_mean'] + 1e-6))
        
        grouped = df.groupby('heterogeneity')['ei_ratio']
        means = grouped.mean()
        stds = grouped.std()
        
        ax5.errorbar(means.index, means.values, yerr=stds.values,
                    color='orange', marker='^', linewidth=2)
        ax5.set_xlabel('Conductance Heterogeneity (CV)')
        ax5.set_ylabel('E/I Activity Ratio')
        ax5.set_title('E/I Balance vs Heterogeneity')
        ax5.grid(True, alpha=0.3)
        
        axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('heterogeneity_effects.png', dpi=300, bbox_inches='tight')
        plt.show()

    def test_connection_specific_effects(self, mec_current=200.0):
        """Test effects of modifying specific connections"""
        print("Testing connection-specific effects...")
        
        test_connections = {
            'mec_gc': 'Perforant Path',
            'gc_pv': 'Feedforward Inhibition',
            'pv_gc': 'Feedback Inhibition',
            'mc_gc': 'Associational',
            'sst_gc': 'Dendritic Inhibition'
        }
        
        modulation_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.ravel()
        
        results = {}
        
        for i, (conn_name, conn_label) in enumerate(test_connections.items()):
            if i >= len(axes):
                break
                
            print(f"  Testing {conn_name}...")
            
            conn_results = {
                'modulation': [],
                'gc_activity': [],
                'gc_sparsity': [],
                'network_sync': []
            }
            
            for mod_level in modulation_levels:
                # Create modified parameters
                modified_params = self._create_connection_modified_params(conn_name, mod_level)
                
                # Test circuit
                circuit = DentateCircuit(
                    self.optimized_circuit_params,
                    modified_params,
                    OpsinParams()
                )
                circuit.add_synaptic_state_manager()
                
                test_results = self._run_connection_test(circuit, mec_current)
                
                conn_results['modulation'].append(mod_level)
                conn_results['gc_activity'].append(test_results['gc_activity'])
                conn_results['gc_sparsity'].append(test_results['gc_sparsity'])
                conn_results['network_sync'].append(test_results['network_sync'])
            
            # Plot
            ax = axes[i]
            ax2 = ax.twinx()
            
            line1 = ax.plot(conn_results['modulation'], conn_results['gc_activity'],
                           'b-o', linewidth=2, label='GC Activity')
            ax.set_xlabel('Connection Strength')
            ax.set_ylabel('GC Activity (Hz)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            
            line2 = ax2.plot(conn_results['modulation'], conn_results['gc_sparsity'],
                            'r-s', linewidth=2, label='GC Sparsity')
            ax2.set_ylabel('GC Sparsity', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f'{conn_label}\n({conn_name})')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            
            results[conn_name] = conn_results
        
        if len(test_connections) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('connection_specific_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

    def _create_connection_modified_params(self, target_connection, modulation):
        """Create parameters with modified specific connection"""
        base = self.optimized_synaptic_params
        modified_modulation = base.connection_modulation.copy()
        
        if target_connection in modified_modulation:
            modified_modulation[target_connection] *= modulation
        else:
            modified_modulation[target_connection] = modulation
        
        return PerConnectionSynapticParams(
            ampa_g_mean=base.ampa_g_mean,
            ampa_g_std=base.ampa_g_std,
            ampa_g_min=base.ampa_g_min,
            ampa_g_max=base.ampa_g_max,
            gaba_g_mean=base.gaba_g_mean,
            gaba_g_std=base.gaba_g_std,
            gaba_g_min=base.gaba_g_min,
            gaba_g_max=base.gaba_g_max,
            distribution=base.distribution,
            connection_modulation=modified_modulation
        )

    def _run_connection_test(self, circuit, mec_current):
        """Test circuit response for connection analysis"""
        circuit.reset_state()
        
        n_steps = 400
        mec_drive = torch.ones(circuit.circuit_params.n_mec) * mec_current
        
        for t in range(n_steps):
            activities = circuit({}, {'mec': mec_drive})
        
        gc_activity = torch.mean(activities['gc']).item()
        gc_sparsity = (torch.sum(activities['gc'] > 1.0) / len(activities['gc'])).item()
        
        all_activities = torch.cat([activities[pop] for pop in ['gc', 'mc', 'pv', 'sst']])
        network_sync = torch.std(all_activities).item() if len(all_activities) > 1 else 0.0
        
        return {
            'gc_activity': gc_activity,
            'gc_sparsity': gc_sparsity,
            'network_sync': network_sync
        }

    def run_conductance_analysis(self, mec_current=200.0):
        """Run complete conductance analysis framework"""
        print("="*60)
        print("CONDUCTANCE HETEROGENEITY ANALYSIS")
        print("="*60)
        
        # Create baseline circuit
        baseline_circuit = self.create_circuit()
        
        # 1. Analyze conductance distributions
        print("\n1. CONDUCTANCE DISTRIBUTION ANALYSIS")
        print("-"*40)
        conductance_stats = self.analyze_conductance_distributions(baseline_circuit)
        
        # 2. Test heterogeneity effects
        print("\n2. HETEROGENEITY EFFECTS ANALYSIS")
        print("-"*40)
        heterogeneity_results = self.test_conductance_heterogeneity_effects(mec_current)
        
        # 3. Test connection-specific effects
        print("\n3. CONNECTION-SPECIFIC EFFECTS")
        print("-"*40)
        connection_results = self.test_connection_specific_effects(mec_current)
        
        return {
            'statistics': conductance_stats,
            'heterogeneity': heterogeneity_results,
            'connections': connection_results
        }

    def run_network_clamp_analysis(self):
        """Run complete conductance-based network clamp analysis"""
        
        print("="*60)
        print("CONDUCTANCE-BASED NETWORK CLAMP ANALYSIS")
        print("="*60)
        
        # Test 1: Static transfer curves
        print("\n1. Conductance Transfer Curves")
        self.test_static_transfer_curves()
        
        # Test 2: NMDA voltage dependence
        print("\n2. NMDA Voltage Dependence")
        self.test_nmda_voltage_dependence('gc')
        
        # Test 3: Inhibition effects
        print("\n3. Inhibitory Conductance Effects")
        self.test_inhibition_effects('gc')
        
        # Test 4: Cell type comparison
        print("\n4. Cell Type Comparison")
        self.compare_cell_types()
        
        # Test 5: Circuit-level analysis
        print("\n5. Circuit-Level Conductance Analysis")
        circuit = self.create_circuit()
        self.analyze_conductance_distributions(circuit)
        
        # Test 6: Full conductance analysis
        print("\n6. Comprehensive Conductance Analysis")
        self.run_conductance_analysis()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tester = NetworkClamp(optimization_json_file=sys.argv[1])
    else:
        tester = NetworkClamp()
    
    tester.run_network_clamp_analysis()
