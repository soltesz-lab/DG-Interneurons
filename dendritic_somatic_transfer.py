#!/usr/bin/env python3
"""
Dendritic-Somatic Transfer Function for Dentate Gyrus Modeling

Implements biophysically realistic two-stage dendritic integration and somatic 
firing rate conversion with cell-type specific parameters.

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

def get_default_device() -> torch.device:
    """Get default device (CUDA if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class DendriticParameters:
    """Cell-type specific dendritic integration parameters"""
    # Membrane properties
    v_rest: float = -70.0          # Resting potential (mV)
    v_thresh: float = -50.0        # Firing threshold (mV)
    input_resistance: float = 200.0 # Input resistance (MOhm)
    membrane_tau: float = 20.0     # Membrane time constant (ms)

    # Synaptic reversal potentials
    e_exc: float = 0.0             # Excitatory reversal (mV)
    e_inh: float = -70.0           # Inhibitory reversal (mV)
    
    # NMDA properties
    mg_concentration: float = 1.0  # Extracellular Mg2+ (mM)
    
    # Dendritic nonlinearities
    dendritic_spike_thresh: float = -40.0  # Dendritic spike threshold (mV)
    dendritic_spike_amp: float = 20.0      # Dendritic spike amplitude (mV)
    ca_spike_enabled: bool = True          # Enable Ca2+ spikes
    
    # Somatic integration
    somatic_coupling: float = 0.7          # Dendrite-soma coupling strength
    axon_hill_thresh: float = -50.0        # Axon hillock threshold (mV)
    max_firing_rate: float = 100.0         # Maximum firing rate (Hz)
    rate_gain: float = 0.1                 # Firing rate gain (Hz/mV)
    
    # Adaptation
    adaptation_enabled: bool = True
    adaptation_tau: float = 100.0          # Adaptation time constant (ms)
    adaptation_strength: float = 10.0       # Adaptation strength

def nmda_voltage_dependence(voltage: torch.Tensor, mg_conc: float = 1.0) -> torch.Tensor:
    """
    Calculate NMDA receptor Mg2+ block removal factor
    
    Based on Jahr & Stevens (1990) and Nowak et al. (1984)
    
    Args:
        voltage: Membrane voltage (mV)
        mg_conc: Extracellular Mg2+ concentration (mM)
        
    Returns:
        mg_block: NMDA Mg2+ block removal factor (0-1)
    """
    # Standard NMDA Mg2+ block equation
    mg_block = 1.0 / (1.0 + (mg_conc / 3.57) * torch.exp(-0.062 * voltage))
    return mg_block

def dendritic_spike_nonlinearity(voltage: torch.Tensor, 
                                 thresh: float = -40.0, 
                                 amplitude: float = 20.0) -> torch.Tensor:
    """
    Model dendritic Ca2+ spike generation
    
    Implements sigmoid-like dendritic spike when threshold is crossed
    
    Args:
        voltage: Dendritic voltage (mV)
        thresh: Threshold for dendritic spike (mV)
        amplitude: Amplitude of dendritic spike (mV)
        
    Returns:
        dendritic_boost: Additional voltage from dendritic spike (mV)
    """
    # Sigmoid activation for dendritic spikes
    above_thresh = voltage - thresh
    spike_factor = torch.sigmoid(above_thresh / 5.0)  # 5mV slope
    dendritic_boost = amplitude * spike_factor
    
    return dendritic_boost


def dendritic_somatic_transfer(ampa_conductance: torch.Tensor,
                               gaba_conductance: torch.Tensor,
                               nmda_conductance: torch.Tensor,
                               params: DendriticParameters,
                               adaptation_state: Optional[torch.Tensor] = None,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict]:
    """
    Dendritic-somatic transfer function using conductance inputs from dynamic synapses
    
    Stage 1: Iteratively solve for dendritic potential with NMDA voltage dependence
    Stage 2: Somatic integration and firing rate conversion
    
    Args:
        ampa_conductance: Total AMPA conductance (nS)
        gaba_conductance: Total GABA conductance (nS)  
        nmda_conductance: Total NMDA conductance before Mg2+ block (nS)
        params: Cell-type specific parameters
        adaptation_state: Previous adaptation level (optional)
        device: Device to create new tensors on (optional, inferred from inputs)
        
    Returns:
        firing_rate: Output firing rate (Hz)
        states: Dictionary containing internal states
    """
    
    # Infer device from input tensors if not specified
    if device is None:
        device = ampa_conductance.device
    
    # Stage 1: Iterative Dendritic Integration
    # =======================================
    
    # Initialize dendritic potential estimate
    v_dendrite = torch.full_like(ampa_conductance, params.v_rest)
    
    # Iteratively solve for dendritic potential due to NMDA voltage dependence
    # This creates a feedback loop: V_dendrite affects NMDA conductance, which affects V_dendrite
    max_iterations = 5  # Expect to converge in 2-3 iterations
    tolerance = 0.1  # mV
    
    for iteration in range(max_iterations):
        v_dendrite_prev = v_dendrite.clone()
        
        # Apply NMDA Mg2+ block based on current voltage estimate
        nmda_voltage_factor = nmda_voltage_dependence(v_dendrite, params.mg_concentration)
        effective_nmda_conductance = nmda_conductance * nmda_voltage_factor
        
        # Calculate total conductance
        total_conductance = ampa_conductance + gaba_conductance + effective_nmda_conductance
        
        # Calculate dendritic potential using conductance-based equation
        # V = (g_ampa*E_exc + g_gaba*E_inh + g_nmda*E_exc + g_leak*E_rest) / (g_total + g_leak)
        
        # Assume leak conductance = 1nS (can be made parameter if needed)
        g_leak = torch.ones_like(total_conductance, device=device)  # nS
        
        numerator = (ampa_conductance * params.e_exc + 
                    gaba_conductance * params.e_inh +
                    effective_nmda_conductance * params.e_exc + 
                    g_leak * params.v_rest)
        
        denominator = total_conductance + g_leak
        
        v_dendrite = numerator / denominator
        
        # Check for convergence
        voltage_change = torch.abs(v_dendrite - v_dendrite_prev)
        if torch.max(voltage_change) < tolerance:
            break
    
    # Apply dendritic nonlinearities (Ca2+ spikes)
    dendritic_boost = torch.zeros_like(v_dendrite, device=device)
    if params.ca_spike_enabled:
        dendritic_boost = dendritic_spike_nonlinearity(
            v_dendrite, params.dendritic_spike_thresh, params.dendritic_spike_amp
        )
        v_dendrite = v_dendrite + dendritic_boost
    
    # Stage 2: Somatic Integration and Firing Rate
    # ============================================
    
    # Dendrite-soma coupling
    v_soma = params.v_rest + params.somatic_coupling * (v_dendrite - params.v_rest)
    
    # Apply adaptation (previous activity reduces current response)
    if params.adaptation_enabled and adaptation_state is not None:
        adaptation_effect = adaptation_state * params.adaptation_strength
        v_soma = v_soma - adaptation_effect
    
    # Convert somatic voltage to firing rate
    above_threshold = torch.clamp(v_soma - params.axon_hill_thresh, min=0.0)
    
    # Exponential firing rate relationship
    if params.rate_gain > 0:
        firing_rate = params.max_firing_rate * torch.expm1(above_threshold * params.rate_gain)
        firing_rate = torch.clamp(firing_rate, max=params.max_firing_rate)
    else:
        firing_rate = torch.zeros_like(above_threshold, device=device)
    
    # Update adaptation state for next timestep
    new_adaptation = adaptation_state if adaptation_state is not None else torch.zeros_like(firing_rate, device=device)
    if params.adaptation_enabled:
        # Simple exponential adaptation
        adaptation_decay = torch.exp(torch.tensor(-1.0 / params.adaptation_tau, device=device))
        new_adaptation = adaptation_decay * new_adaptation + (1 - adaptation_decay) * firing_rate / params.max_firing_rate
    
    # Calculate final NMDA factor for output
    final_nmda_factor = nmda_voltage_dependence(v_dendrite, params.mg_concentration)
    
    # Return states for analysis
    states = {
        'v_dendrite': v_dendrite,
        'v_soma': v_soma,
        'adaptation': new_adaptation,
        'nmda_factor': final_nmda_factor,
        'effective_nmda_conductance': effective_nmda_conductance,
        'dendritic_boost': dendritic_boost,
        'above_threshold': above_threshold,
        'total_conductance': total_conductance,
        'convergence_iterations': iteration + 1
    }
    
    return firing_rate, states


def get_cell_type_parameters() -> Dict[str, DendriticParameters]:
    """
    Cell-type specific dendritic parameters for conductance-based integration
    """
    
    cell_params = {
        'gc': DendriticParameters(
            v_rest=-70.0,
            v_thresh=-42.0,
            e_exc=0.0,
            e_inh=-70.0,
            mg_concentration=1.0,
            dendritic_spike_thresh=-35.0,
            ca_spike_enabled=True,
            somatic_coupling=0.9,
            max_firing_rate=40.0,
            rate_gain=0.006,
            adaptation_enabled=True,
            adaptation_strength=20.0,
            axon_hill_thresh=-55
        ),
        
        'mc': DendriticParameters(
            v_rest=-65.0,
            v_thresh=-45.0,
            e_exc=0.0,
            e_inh=-70.0,
            mg_concentration=1.0,
            dendritic_spike_thresh=-38.0,
            ca_spike_enabled=True,
            somatic_coupling=0.5,
            axon_hill_thresh=-63,
            max_firing_rate=80.0,
            rate_gain=0.02,
            adaptation_enabled=False,
            adaptation_strength=1.0
        ),
        
        'pv': DendriticParameters(
            v_rest=-65.0,
            v_thresh=-45.0,
            e_exc=0.0,
            e_inh=-70.0,
            mg_concentration=1.0,
            dendritic_spike_thresh=-35.0,
            ca_spike_enabled=False,
            somatic_coupling=0.8,
            max_firing_rate=100.0,
            rate_gain=0.025,
            adaptation_enabled=False,
            adaptation_strength=0.05,
            axon_hill_thresh=-55,
        ),
        
        'sst': DendriticParameters(
            v_rest=-68.0,
            v_thresh=-47.0,
            e_exc=0.0,
            e_inh=-70.0,
            mg_concentration=1.0,
            dendritic_spike_thresh=-40.0,
            ca_spike_enabled=True,
            somatic_coupling=0.6,
            max_firing_rate=80.0,
            rate_gain=0.02,
            adaptation_enabled=False,
            adaptation_strength=1.0
        ),
        
        'mec': DendriticParameters(
            v_rest=-67.0,
            v_thresh=-48.0,
            e_exc=0.0,
            e_inh=-70.0,
            mg_concentration=1.0,
            dendritic_spike_thresh=-38.0,
            ca_spike_enabled=True,
            somatic_coupling=0.75,
            max_firing_rate=40.0,
            rate_gain=0.013,
            adaptation_enabled=False,
            adaptation_strength=0.1,
            axon_hill_thresh=-55,
        )
    }
    
    return cell_params


def test_dendritic_somatic_transfer(device: Optional[torch.device] = None,
                                    save_figures: bool = True):
    """
    Test the conductance-based dendritic-somatic transfer function
    
    Args:
        device: Device to run tests on (None for auto-detect)
        save_figures: Whether to save figure files
    """
    if device is None:
        device = get_default_device()
    
    print("="*60)
    print("Testing conductance-based dendritic-somatic transfer function")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("="*60)
    
    # Get parameters for different cell types
    cell_params = get_cell_type_parameters()
    
    # Test conductance range (nS)
    conductance_range = torch.linspace(0, 30, 300, device=device)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    # Test each cell type
    for i, (cell_type, params) in enumerate(cell_params.items()):
        ax = axes[i]
        
        # Test 1: Pure AMPA response
        firing_rates_ampa = []
        v_dendrites_ampa = []
        v_somas_ampa = []
        
        for conductance in conductance_range:
            cond = torch.tensor([conductance.item()], device=device)
            fr, states = dendritic_somatic_transfer(
                ampa_conductance=cond,
                gaba_conductance=torch.zeros_like(cond),
                nmda_conductance=torch.zeros_like(cond),
                params=params,
                device=device
            )
            
            firing_rates_ampa.append(fr.item())
            v_dendrites_ampa.append(states['v_dendrite'].item())
            v_somas_ampa.append(states['v_soma'].item())
        
        # Test 2: Mixed AMPA + NMDA (30% NMDA)
        firing_rates_mixed = []
        v_dendrites_mixed = []
        v_somas_mixed = []
        nmda_factors = []
        convergence_iters = []
        
        for conductance in conductance_range:
            cond = torch.tensor([conductance.item()], device=device)
            ampa_cond = cond * 0.7
            nmda_cond = cond * 0.3
            
            fr, states = dendritic_somatic_transfer(
                ampa_conductance=ampa_cond,
                gaba_conductance=torch.zeros_like(cond),
                nmda_conductance=nmda_cond,
                params=params,
                device=device
            )
            
            firing_rates_mixed.append(fr.item())
            v_dendrites_mixed.append(states['v_dendrite'].item())
            v_somas_mixed.append(states['v_soma'].item())
            nmda_factors.append(states['nmda_factor'].item())
            convergence_iters.append(states['convergence_iterations'])
        
        # Test 3: Inhibition effect
        firing_rates_inhib = []
        fixed_excitation = 2.5  # nS fixed excitatory conductance
        
        for inhib_cond in conductance_range:
            cond_exc = torch.tensor([fixed_excitation], device=device)
            cond_inh = torch.tensor([inhib_cond.item()], device=device)
            
            fr, states = dendritic_somatic_transfer(
                ampa_conductance=cond_exc * 0.7,
                gaba_conductance=cond_inh,
                nmda_conductance=cond_exc * 0.3,
                params=params,
                device=device
            )
            
            firing_rates_inhib.append(fr.item())
        
        # Plot firing rate curves
        ax.plot(conductance_range.cpu().numpy(), firing_rates_ampa, 'b-', 
                linewidth=2.5, label='AMPA only')
        ax.plot(conductance_range.cpu().numpy(), firing_rates_mixed, 'r-', 
                linewidth=2.5, label='AMPA + NMDA (30%)')
        ax.set_xlabel('Excitatory Conductance (nS)', fontsize=10)
        ax.set_ylabel('Firing Rate (Hz)', color='b', fontsize=10)
        ax.set_title(f'{cell_type.upper()} Transfer Function', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        # Add voltage traces on secondary axis
        ax2 = ax.twinx()
        ax2.plot(conductance_range.cpu().numpy(), v_dendrites_mixed, 'g--', 
                alpha=0.5, linewidth=1.5, label='V_dendrite')
        ax2.plot(conductance_range.cpu().numpy(), v_somas_mixed, 'm--', 
                alpha=0.5, linewidth=1.5, label='V_soma')
        ax2.axhline(y=params.v_thresh, color='k', linestyle=':', 
                   alpha=0.5, linewidth=1)
        ax2.set_ylabel('Voltage (mV)', color='g', fontsize=10)
        ax2.legend(loc='upper right', fontsize=7)
        
        # Add parameter info text box
        param_text = (f"R_in={params.input_resistance:.0f}M$\\Omega$\n"
                     f"Mg$^{{2+}}$={params.mg_concentration:.1f}mM\n"
                     f"Max rate={params.max_firing_rate:.0f}Hz\n"
                     f"Gain={params.rate_gain:.3f}\n"
                     f"Ca$^{{2+}}$ spikes={'Yes' if params.ca_spike_enabled else 'No'}\n"
                     f"Adapt={'Yes' if params.adaptation_enabled else 'No'}")
        ax.text(0.65, 0.05, param_text, transform=ax.transAxes, 
               verticalalignment='bottom', fontsize=7,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Use the last subplot for inhibition comparison across cell types
    ax_inhib = axes[-1]
    ax_inhib.clear()
    
    for cell_type, params in cell_params.items():
        firing_rates_inhib_cell = []
        fixed_exc = 2.5
        
        for inhib_cond in torch.linspace(0, 25, 100, device=device):
            cond_exc = torch.tensor([fixed_exc], device=device)
            cond_inh = torch.tensor([inhib_cond.item()], device=device)
            
            fr, _ = dendritic_somatic_transfer(
                ampa_conductance=cond_exc * 0.7,
                gaba_conductance=cond_inh,
                nmda_conductance=cond_exc * 0.3,
                params=params,
                device=device
            )
            firing_rates_inhib_cell.append(fr.item())
        
        ax_inhib.plot(torch.linspace(0, 25, 100).cpu().numpy(), 
                     firing_rates_inhib_cell, linewidth=2, 
                     label=cell_type.upper())
    
    ax_inhib.set_xlabel('Inhibitory Conductance (nS)', fontsize=10)
    ax_inhib.set_ylabel('Firing Rate (Hz)', fontsize=10)
    ax_inhib.set_title('Inhibition Effect (Fixed Excitation = 2.5 nS)', fontsize=11)
    ax_inhib.grid(True, alpha=0.3)
    ax_inhib.legend(fontsize=8)
    
    plt.tight_layout()
    if save_figures:
        plt.savefig('conductance_based_transfer_functions.png', dpi=150)
        print("\nSaved figure: conductance_based_transfer_functions.png")
    plt.show()
    
    # Additional analysis: NMDA voltage dependence
    print("\n" + "="*60)
    print("NMDA Voltage Dependence Analysis")
    print("="*60)
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    for i, (cell_type, params) in enumerate(cell_params.items()):
        ax = axes2[i]
        
        # Test NMDA contribution at different conductance levels
        test_conductances = [2.5, 5.0, 10.0, 15.0, 20.0]  # nS
        
        for total_cond in test_conductances:
            ampa_fractions = torch.linspace(0, 1, 50, device=device)
            firing_rates = []
            nmda_contributions = []
            
            for ampa_frac in ampa_fractions:
                nmda_frac = 1 - ampa_frac
                
                ampa_cond = torch.tensor([total_cond * ampa_frac], device=device)
                nmda_cond = torch.tensor([total_cond * nmda_frac], device=device)
                
                fr, states = dendritic_somatic_transfer(
                    ampa_conductance=ampa_cond,
                    gaba_conductance=torch.zeros_like(ampa_cond),
                    nmda_conductance=nmda_cond,
                    params=params,
                    device=device
                )
                
                firing_rates.append(fr.item())
                effective_nmda = states['effective_nmda_conductance'].item()
                nmda_contributions.append(effective_nmda / (total_cond + 1e-6))
            
            ax.plot(ampa_fractions.cpu().numpy() * 100, firing_rates, 
                   linewidth=2, label=f'{total_cond}nS total')
        
        ax.set_xlabel('AMPA Fraction (%)', fontsize=10)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
        ax.set_title(f'{cell_type.upper()} - NMDA vs AMPA Mix', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim([0, 100])
    
    # Summary plot: convergence iterations
    ax_conv = axes2[-1]
    
    for cell_type, params in cell_params.items():
        conductance_test = torch.linspace(0, 30, 100, device=device)
        iterations_needed = []
        
        for cond in conductance_test:
            ampa_cond = torch.tensor([cond.item() * 0.5], device=device)
            nmda_cond = torch.tensor([cond.item() * 0.5], device=device)
            
            _, states = dendritic_somatic_transfer(
                ampa_conductance=ampa_cond,
                gaba_conductance=torch.zeros_like(ampa_cond),
                nmda_conductance=nmda_cond,
                params=params,
                device=device
            )
            iterations_needed.append(states['convergence_iterations'])
        
        ax_conv.plot(conductance_test.cpu().numpy(), iterations_needed, 
                    linewidth=2, label=cell_type.upper(), marker='o', 
                    markersize=2, alpha=0.7)
    
    ax_conv.set_xlabel('Total Conductance (nS)', fontsize=10)
    ax_conv.set_ylabel('Iterations to Converge', fontsize=10)
    ax_conv.set_title('Voltage Solution Convergence', fontsize=11)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend(fontsize=8)
    ax_conv.set_ylim([0, 6])
    
    plt.tight_layout()
    if save_figures:
        plt.savefig('nmda_voltage_dependence_analysis.png', dpi=150)
        print("Saved figure: nmda_voltage_dependence_analysis.png")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Cell Type Parameters Summary")
    print("="*60)
    for cell_type, params in cell_params.items():
        print(f"\n{cell_type.upper()}:")
        print(f"  V_rest: {params.v_rest:.1f} mV")
        print(f"  E_exc: {params.e_exc:.1f} mV")
        print(f"  E_inh: {params.e_inh:.1f} mV")
        print(f"  Mg2+ concentration: {params.mg_concentration:.1f} mM")
        print(f"  Max firing rate: {params.max_firing_rate:.0f} Hz")
        print(f"  Rate gain: {params.rate_gain:.4f}")
        print(f"  Ca2+ spikes: {params.ca_spike_enabled}")
        print(f"  Adaptation: {params.adaptation_enabled}")
        print(f"  Somatic coupling: {params.somatic_coupling:.2f}")
    
    return cell_params


def benchmark_devices(n_tests: int = 1000):
    """
    Benchmark transfer function performance on CPU vs GPU
    
    Args:
        n_tests: Number of transfer function calls to benchmark
    """
    print("="*60)
    print("Device Benchmark for Dendritic-Somatic Transfer")
    print("="*60)
    
    cell_params = get_cell_type_parameters()
    gc_params = cell_params['gc']
    
    # Create test inputs
    test_conductances = {
        'ampa': 5.0,  # nS
        'gaba': 2.0,  # nS
        'nmda': 3.0,  # nS
    }
    
    import time
    
    # CPU benchmark
    print("\nCPU Benchmark:")
    print("-"*60)
    device_cpu = torch.device('cpu')
    
    ampa_cond_cpu = torch.tensor([test_conductances['ampa']], device=device_cpu)
    gaba_cond_cpu = torch.tensor([test_conductances['gaba']], device=device_cpu)
    nmda_cond_cpu = torch.tensor([test_conductances['nmda']], device=device_cpu)
    
    start = time.time()
    for _ in range(n_tests):
        fr, states = dendritic_somatic_transfer(
            ampa_cond_cpu, gaba_cond_cpu, nmda_cond_cpu,
            gc_params, device=device_cpu
        )
    cpu_time = time.time() - start
    
    print(f"Completed {n_tests} iterations in {cpu_time:.3f}s")
    print(f"Average time per call: {cpu_time/n_tests*1000:.3f} ms")
    
    # GPU benchmark if available
    if torch.cuda.is_available():
        print("\nGPU Benchmark:")
        print("-"*60)
        device_gpu = torch.device('cuda')
        
        ampa_cond_gpu = torch.tensor([test_conductances['ampa']], device=device_gpu)
        gaba_cond_gpu = torch.tensor([test_conductances['gaba']], device=device_gpu)
        nmda_cond_gpu = torch.tensor([test_conductances['nmda']], device=device_gpu)
        
        # Warmup
        for _ in range(10):
            fr, states = dendritic_somatic_transfer(
                ampa_cond_gpu, gaba_cond_gpu, nmda_cond_gpu,
                gc_params, device=device_gpu
            )
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_tests):
            fr, states = dendritic_somatic_transfer(
                ampa_cond_gpu, gaba_cond_gpu, nmda_cond_gpu,
                gc_params, device=device_gpu
            )
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"Completed {n_tests} iterations in {gpu_time:.3f}s")
        print(f"Average time per call: {gpu_time/n_tests*1000:.3f} ms")
        print(f"\nSpeedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("\nGPU not available for comparison")


if __name__ == "__main__":
    # Test with default device (auto-detect)
    print("\nRunning transfer function tests with auto-detected device...")
    cell_params = test_dendritic_somatic_transfer()
    
    # Print parameter summary
    print("\nCell Type Parameters Summary:")
    print("=" * 50)
    for cell_type, params in cell_params.items():
        print(f"{cell_type.upper()}:")
        print(f"  Input Resistance: {params.input_resistance:.0f} MOhm")
        print(f"  Max Rate: {params.max_firing_rate:.0f} Hz")
        print(f"  Rate Gain: {params.rate_gain:.3f}")
        print(f"  Ca2+ Spikes: {params.ca_spike_enabled}")
        print(f"  Adaptation: {params.adaptation_enabled}")
        print()
    
    # Run benchmark if GPU available
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("Running device benchmark...")
        print("="*60)
        benchmark_devices(n_tests=1000)
