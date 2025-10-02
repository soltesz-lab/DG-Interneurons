#!/usr/bin/env python3
"""
Enhanced Analysis and Visualization Tools for Optogenetic Optimization

Extends the baseline optimization analysis to support optogenetic stimulation
objectives, including rate increases, Gini coefficient changes, and population-
specific effects during PV and SST stimulation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import scipy.stats as stats
import json

# Import optogenetic optimization components
from DG_circuit_optogenetic_optimization import (
    CombinedOptimizationTargets,
    OptogeneticTargets,
    simulate_optogenetic_stimulation,
    evaluate_optogenetic_objectives,
    evaluate_best_parameters
)

from DG_circuit_optimization import (
    OptimizationTargets,
    OptimizationConfig,
    CircuitParams,
    OpsinParams,
)

from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit,
    PerConnectionSynapticParams
)


class OptimizationAnalyzer:
    """
    Analysis of optimization results
    """
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_convergence_analysis(self, history: Dict, save_fig: bool = True):
        """
        Plot convergence analysis
        """
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.ravel()
        
        iterations = range(len(history['loss']))
        
        # 1. Total loss convergence
        ax1 = axes[0]
        ax1.plot(iterations, history['loss'], 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Loss Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add moving average
        window_size = min(20, len(history['loss']) // 10)
        if window_size > 1:
            moving_avg = pd.Series(history['loss']).rolling(window=window_size).mean()
            ax1.plot(iterations, moving_avg, 'r--', linewidth=2, alpha=0.7, label='Moving Avg')
            ax1.legend()
        
        # 2. Loss components over time
        ax2 = axes[1]
        if history['loss_components']:
            # Extract component names from first entry
            component_names = list(history['loss_components'][0].keys())
            
            for comp_name in component_names:
                comp_values = [entry[comp_name] for entry in history['loss_components']]
                ax2.plot(iterations, comp_values, label=comp_name, linewidth=1.5, alpha=0.8)
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss Component Value')
            ax2.set_title('Loss Component Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # 3. Parameter evolution (sample key connections)
        ax3 = axes[2]
        if history['parameters']:
            key_connections = ['mec_gc', 'pv_gc', 'mc_gc', 'sst_gc']
            
            for conn_name in key_connections:
                if conn_name in history['parameters'][0]:
                    param_values = [entry[conn_name] for entry in history['parameters']]
                    ax3.plot(iterations, param_values, label=conn_name, linewidth=2, alpha=0.8)
            
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Connection Strength Multiplier')
            ax3.set_title('Key Connection Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Firing rate evolution for key populations
        ax4 = axes[3]
        if history['firing_rates']:
            # Extract firing rates for first MEC drive condition and first trial
            first_condition = list(history['firing_rates'][0].keys())[0]
            populations = ['gc', 'mc', 'pv', 'sst']
            
            for pop in populations:
                if first_condition in history['firing_rates'][0]:
                    rates = []
                    for entry in history['firing_rates']:
                        if first_condition in entry and pop in entry[first_condition]:
                            rates.append(entry[first_condition][pop])
                        else:
                            rates.append(np.nan)
                    
                    ax4.plot(iterations, rates, label=f'{pop.upper()}', linewidth=2, alpha=0.8)
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Firing Rate (Hz)')
            ax4.set_title('Population Firing Rates Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Loss gradient analysis (approximate)
        ax5 = axes[4]
        if len(history['loss']) > 10:
            # Compute approximate gradients
            gradients = np.gradient(np.array(history['loss']))
            ax5.plot(iterations, np.abs(gradients), 'g-', linewidth=1.5, alpha=0.7)
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('|Loss Gradient|')
            ax5.set_title('Convergence Rate')
            ax5.grid(True, alpha=0.3)
            ax5.set_yscale('log')
        
        # 6. Parameter variance over time
        ax6 = axes[5]
        if history['parameters']:
            param_vars = []
            for entry in history['parameters']:
                param_values = list(entry.values())
                param_vars.append(np.var(param_values))
            
            ax6.plot(iterations, param_vars, 'purple', linewidth=2, alpha=0.8)
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Parameter Variance')
            ax6.set_title('Parameter Diversity')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('optimization_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parameter_analysis(self, initial_params: Dict, final_params: Dict, 
                               targets: 'OptimizationTargets', save_fig: bool = True):
        """
        Analyze parameter changes and constraint satisfaction
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Parameter changes bar chart
        ax1 = axes[0, 0]
        connections = list(final_params.keys())
        initial_values = [initial_params.get(conn, 1.0) for conn in connections]
        final_values = [final_params[conn] for conn in connections]
        changes = [(final - initial) / initial * 100 
                  for initial, final in zip(initial_values, final_values)]
        
        colors = ['red' if change < 0 else 'blue' for change in changes]
        bars = ax1.bar(range(len(connections)), changes, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(connections)))
        ax1.set_xticklabels(connections, rotation=45, ha='right')
        ax1.set_ylabel('Parameter Change (%)')
        ax1.set_title('Connection Strength Changes')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # Add value labels on bars
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{change:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=8)
        
        # 2. Before vs after scatter plot
        ax2 = axes[0, 1]
        ax2.scatter(initial_values, final_values, alpha=0.7, s=60)
        
        # Add diagonal line (no change)
        min_val = min(min(initial_values), min(final_values))
        max_val = max(max(initial_values), max(final_values))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax2.set_xlabel('Initial Connection Strength')
        ax2.set_ylabel('Final Connection Strength')
        ax2.set_title('Parameter Changes (Initial vs Final)')
        ax2.grid(True, alpha=0.3)
        
        # Add connection labels
        for i, conn in enumerate(connections):
            ax2.annotate(conn, (initial_values[i], final_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Constraint satisfaction
        ax3 = axes[1, 0]
        constraint_violations = []
        constraint_labels = []
        
        for conn in connections:
            if conn in targets.connection_bounds:
                min_bound, max_bound = targets.connection_bounds[conn]
                final_val = final_params[conn]
                
                if final_val < min_bound:
                    violation = (min_bound - final_val) / min_bound * 100
                    constraint_violations.append(-violation)  # Negative for below
                    constraint_labels.append(f'{conn}\n(below)')
                elif final_val > max_bound:
                    violation = (final_val - max_bound) / max_bound * 100
                    constraint_violations.append(violation)   # Positive for above
                    constraint_labels.append(f'{conn}\n(above)')
                else:
                    constraint_violations.append(0.0)
                    constraint_labels.append(conn)
        
        colors = ['red' if v != 0 else 'green' for v in constraint_violations]
        bars = ax3.bar(range(len(constraint_labels)), constraint_violations, 
                      color=colors, alpha=0.7)
        ax3.set_xticks(range(len(constraint_labels)))
        ax3.set_xticklabels(constraint_labels, rotation=45, ha='right')
        ax3.set_ylabel('Constraint Violation (%)')
        ax3.set_title('Constraint Satisfaction')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # 4. Parameter distribution
        ax4 = axes[1, 1]
        ax4.hist(initial_values, bins=15, alpha=0.5, color='blue', 
                label='Initial', density=True)
        ax4.hist(final_values, bins=15, alpha=0.5, color='red', 
                label='Final', density=True)
        ax4.set_xlabel('Connection Strength')
        ax4.set_ylabel('Density')
        ax4.set_title('Parameter Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def validate_optimized_circuit(self, optimized_synaptic_params, 
                                  circuit_params, opsin_params, targets, 
                                  mec_drive_levels=[200.0], save_fig=True):
        """
        Validation of optimized circuit
        """
        
        from DG_circuit_dendritic_somatic_transfer import DentateCircuit
        
        print("Validating optimized circuit...")
        
        results = {}
        
        for mec_drive in mec_drive_levels:
            print(f"Testing MEC drive: {mec_drive} pA")
            
            # Create optimized circuit
            circuit = DentateCircuit(circuit_params, optimized_synaptic_params, opsin_params)
            circuit.reset_state()
            
            # Run extended simulation
            n_steps = 800
            warmup = 200
            mec_input = torch.ones(circuit_params.n_mec) * mec_drive
            
            activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
            
            for t in range(n_steps):
                external_drive = {'mec': mec_input}
                activities = circuit({}, external_drive)
                
                if t >= warmup:
                    for pop in activities_over_time:
                        if pop in activities:
                            activities_over_time[pop].append(activities[pop].clone())
            
            # Calculate detailed statistics
            drive_results = {}
            for pop in activities_over_time:
                if len(activities_over_time[pop]) > 0:
                    pop_time_series = torch.stack(activities_over_time[pop])
                    mean_rates = torch.mean(pop_time_series, dim=0)
                    
                    drive_results[pop] = {
                        'mean_firing_rate': torch.mean(mean_rates).item(),
                        'std_firing_rate': torch.std(mean_rates).item(),
                        'sparsity': (torch.sum(mean_rates > targets.activity_threshold) / len(mean_rates)).item(),
                        'cv': torch.std(mean_rates).item() / (torch.mean(mean_rates).item() + 1e-6),
                        'max_rate': torch.max(mean_rates).item(),
                        'min_rate': torch.min(mean_rates).item(),
                        'individual_rates': mean_rates.numpy(),
                        'time_series': pop_time_series.numpy()
                    }
            
            results[f'mec_{mec_drive}'] = drive_results
        
        # Create validation plots
        self._plot_validation_results(results, targets, mec_drive_levels, save_fig)
        
        return results
    
    def _plot_validation_results(self, results, targets, mec_drive_levels, save_fig):
        """Plot validation results"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Target vs actual firing rates
        ax1 = plt.subplot(2, 3, 1)
        populations = ['gc', 'mc', 'pv', 'sst']
        
        for pop in populations:
            target_rate = targets.target_rates[pop]
            actual_rates = []
            
            for drive in mec_drive_levels:
                key = f'mec_{drive}'
                if key in results and pop in results[key]:
                    actual_rates.append(results[key][pop]['mean_firing_rate'])
            
            if actual_rates:
                mean_actual = np.mean(actual_rates)
                std_actual = np.std(actual_rates) if len(actual_rates) > 1 else 0
                
                ax1.errorbar(target_rate, mean_actual, yerr=std_actual,
                           fmt='o', capsize=5, capthick=2, label=pop.upper())
        
        # Perfect agreement line
        max_rate = max([targets.target_rates[pop] for pop in populations])
        ax1.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, label='Perfect Agreement')
        
        ax1.set_xlabel('Target Firing Rate (Hz)')
        ax1.set_ylabel('Actual Firing Rate (Hz)')
        ax1.set_title('Target vs Actual Firing Rates')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sparsity validation
        ax2 = plt.subplot(2, 3, 2)
        for pop in populations:
            target_sparsity = targets.sparsity_targets.get(pop, None)
            if target_sparsity is not None:
                actual_sparsities = []
                
                for drive in mec_drive_levels:
                    key = f'mec_{drive}'
                    if key in results and pop in results[key]:
                        actual_sparsities.append(results[key][pop]['sparsity'])
                
                if actual_sparsities:
                    mean_actual = np.mean(actual_sparsities)
                    std_actual = np.std(actual_sparsities) if len(actual_sparsities) > 1 else 0
                    
                    ax2.errorbar(target_sparsity, mean_actual, yerr=std_actual,
                               fmt='s', capsize=5, capthick=2, label=pop.upper())
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Agreement')
        ax2.set_xlabel('Target Sparsity')
        ax2.set_ylabel('Actual Sparsity')
        ax2.set_title('Target vs Actual Sparsity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Population activity distributions
        ax3 = plt.subplot(2, 3, 3)
        first_result = results[list(results.keys())[0]]
        
        for i, pop in enumerate(populations):
            if pop in first_result:
                rates = first_result[pop]['individual_rates']
                ax3.hist(rates, bins=20, alpha=0.6, label=f'{pop.upper()}', 
                        density=True, histtype='step', linewidth=2)
        
        ax3.set_xlabel('Individual Cell Firing Rate (Hz)')
        ax3.set_ylabel('Density')
        ax3.set_title('Individual Cell Rate Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Time series for key populations
        ax4 = plt.subplot(2, 3, 4)
        first_result = results[list(results.keys())[0]]
        
        for pop in ['gc', 'pv']:  # Show key populations
            if pop in first_result:
                time_series = first_result[pop]['time_series']
                mean_time_series = np.mean(time_series, axis=1)
                time_axis = np.arange(len(mean_time_series))
                
                ax4.plot(time_axis, mean_time_series, linewidth=2, 
                        label=f'{pop.upper()} population')
        
        ax4.set_xlabel('Time (simulation steps)')
        ax4.set_ylabel('Population Firing Rate (Hz)')
        ax4.set_title('Population Activity Time Series')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. E/I balance analysis
        ax5 = plt.subplot(2, 3, 5)
        drive_labels = []
        ei_ratios = []
        
        for drive in mec_drive_levels:
            key = f'mec_{drive}'
            if key in results:
                exc_activity = (results[key].get('gc', {}).get('mean_firing_rate', 0) +
                               results[key].get('mc', {}).get('mean_firing_rate', 0))
                inh_activity = (results[key].get('pv', {}).get('mean_firing_rate', 0) +
                               results[key].get('sst', {}).get('mean_firing_rate', 0))
                
                if inh_activity > 0:
                    ei_ratio = exc_activity / inh_activity
                    ei_ratios.append(ei_ratio)
                    drive_labels.append(f'{drive:.0f}')
        
        if ei_ratios:
            bars = ax5.bar(drive_labels, ei_ratios, alpha=0.7, color='orange')
            ax5.set_xlabel('MEC Drive (pA)')
            ax5.set_ylabel('E/I Activity Ratio')
            ax5.set_title('Excitation-Inhibition Balance')
            ax5.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, ratio in zip(bars, ei_ratios):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{ratio:.2f}', ha='center', va='bottom')
        
        # 6. Summary metrics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for pop in populations:
            target_rate = targets.target_rates[pop]
            target_sparsity = targets.sparsity_targets.get(pop, 'N/A')
            
            # Average across drives
            actual_rates = []
            actual_sparsities = []
            
            for drive in mec_drive_levels:
                key = f'mec_{drive}'
                if key in results and pop in results[key]:
                    actual_rates.append(results[key][pop]['mean_firing_rate'])
                    actual_sparsities.append(results[key][pop]['sparsity'])
            
            if actual_rates:
                mean_rate = np.mean(actual_rates)
                mean_sparsity = np.mean(actual_sparsities) if actual_sparsities else 'N/A'
                
                rate_error = abs(mean_rate - target_rate) / target_rate * 100
                sparsity_error = (abs(mean_sparsity - target_sparsity) / target_sparsity * 100 
                                 if isinstance(target_sparsity, float) and isinstance(mean_sparsity, float) 
                                 else 'N/A')
                
                summary_data.append([
                    pop.upper(),
                    f'{target_rate:.1f}',
                    f'{mean_rate:.1f}',
                    f'{rate_error:.1f}%',
                    f'{target_sparsity:.2f}' if isinstance(target_sparsity, float) else 'N/A',
                    f'{mean_sparsity:.2f}' if isinstance(mean_sparsity, float) else 'N/A',
                    f'{sparsity_error:.1f}%' if isinstance(sparsity_error, float) else 'N/A'
                ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Pop', 'Target\nRate', 'Actual\nRate', 'Rate\nError',
                                   'Target\nSparsity', 'Actual\nSparsity', 'Sparsity\nError'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        ax6.set_title('Validation Summary', pad=20)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('circuit_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def sensitivity_analysis(self, optimized_params, circuit_params, opsin_params, 
                           targets, perturbation_size=0.1):
        """
        Analyze sensitivity of optimized parameters to perturbations
        """
        
        from DG_circuit_dendritic_somatic_transfer import DentateCircuit
        
        print("Running sensitivity analysis...")
        
        baseline_circuit = DentateCircuit(circuit_params, optimized_params, opsin_params)
        
        # Get baseline performance
        baseline_performance = self._evaluate_circuit_performance(
            baseline_circuit, targets, mec_drive=200.0
        )
        
        sensitivity_results = {}
        
        # Test each connection parameter
        for conn_name in optimized_params.connection_modulation.keys():
            print(f"  Testing {conn_name}...")
            
            conn_sensitivities = {}
            
            # Test positive and negative perturbations
            for direction, multiplier in [('increase', 1 + perturbation_size), 
                                        ('decrease', 1 - perturbation_size)]:
                
                # Create perturbed parameters
                perturbed_modulation = optimized_params.connection_modulation.copy()
                perturbed_modulation[conn_name] *= multiplier
                
                perturbed_params = PerConnectionSynapticParams(
                    **{k: getattr(optimized_params, k) for k in optimized_params.__dict__ if k != 'connection_modulation'},
                    connection_modulation=perturbed_modulation
                )
                
                # Test perturbed circuit
                perturbed_circuit = DentateCircuit(circuit_params, perturbed_params, opsin_params)
                perturbed_performance = self._evaluate_circuit_performance(
                    perturbed_circuit, targets, mec_drive=200.0
                )
                
                # Calculate sensitivity metrics
                sensitivity_metrics = {}
                for metric in ['total_loss', 'rate_error', 'sparsity_error']:
                    if metric in baseline_performance and metric in perturbed_performance:
                        baseline_val = baseline_performance[metric]
                        perturbed_val = perturbed_performance[metric]
                        
                        if baseline_val != 0:
                            sensitivity = (perturbed_val - baseline_val) / baseline_val / perturbation_size
                        else:
                            sensitivity = float('inf') if perturbed_val != 0 else 0.0
                        
                        sensitivity_metrics[metric] = sensitivity
                
                conn_sensitivities[direction] = sensitivity_metrics
            
            sensitivity_results[conn_name] = conn_sensitivities
        
        # Plot sensitivity results
        self._plot_sensitivity_results(sensitivity_results)
        
        return sensitivity_results
    
    def _evaluate_circuit_performance(self, circuit, targets, mec_drive):
        """Evaluate circuit performance metrics"""
        
        circuit.reset_state()
        
        # Run simulation
        n_steps = 400
        warmup = 100
        mec_input = torch.ones(circuit.circuit_params.n_mec) * mec_drive
        
        activities_over_time = {pop: [] for pop in ['gc', 'mc', 'pv', 'sst']}
        
        for t in range(n_steps):
            external_drive = {'mec': mec_input}
            activities = circuit({}, external_drive)
            
            if t >= warmup:
                for pop in activities_over_time:
                    if pop in activities:
                        activities_over_time[pop].append(activities[pop].clone())
        
        # Calculate performance metrics
        total_loss = 0.0
        rate_errors = []
        sparsity_errors = []
        
        for pop in activities_over_time:
            if len(activities_over_time[pop]) > 0:
                pop_time_series = torch.stack(activities_over_time[pop])
                mean_rates = torch.mean(pop_time_series, dim=0)
                
                # Firing rate error
                if pop in targets.target_rates:
                    target_rate = targets.target_rates[pop]
                    actual_rate = torch.mean(mean_rates).item()
                    rate_error = abs(actual_rate - target_rate) / target_rate
                    rate_errors.append(rate_error)
                    total_loss += rate_error
                
                # Sparsity error
                if pop in targets.sparsity_targets:
                    target_sparsity = targets.sparsity_targets[pop]
                    actual_sparsity = (torch.sum(mean_rates > targets.activity_threshold) / len(mean_rates)).item()
                    sparsity_error = abs(actual_sparsity - target_sparsity) / target_sparsity
                    sparsity_errors.append(sparsity_error)
                    total_loss += sparsity_error
        
        return {
            'total_loss': total_loss,
            'rate_error': np.mean(rate_errors) if rate_errors else 0.0,
            'sparsity_error': np.mean(sparsity_errors) if sparsity_errors else 0.0
        }
    
    def _plot_sensitivity_results(self, sensitivity_results):
        """Plot sensitivity analysis results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        connections = list(sensitivity_results.keys())
        metrics = ['total_loss', 'rate_error', 'sparsity_error']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            increase_sens = []
            decrease_sens = []
            
            for conn in connections:
                if (conn in sensitivity_results and 
                    'increase' in sensitivity_results[conn] and
                    'decrease' in sensitivity_results[conn]):
                    
                    inc_val = sensitivity_results[conn]['increase'].get(metric, 0)
                    dec_val = sensitivity_results[conn]['decrease'].get(metric, 0)
                    
                    increase_sens.append(inc_val)
                    decrease_sens.append(dec_val)
                else:
                    increase_sens.append(0)
                    decrease_sens.append(0)
            
            x_pos = np.arange(len(connections))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, increase_sens, width, 
                          label='Increase', alpha=0.7, color='red')
            bars2 = ax.bar(x_pos + width/2, decrease_sens, width,
                          label='Decrease', alpha=0.7, color='blue')
            
            ax.set_xlabel('Connection')
            ax.set_ylabel(f'{metric.replace("_", " ").title()} Sensitivity')
            ax.set_title(f'Sensitivity: {metric.replace("_", " ").title()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(connections, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_optimization_analysis(results, optimizer, targets, 
                                      circuit_params, opsin_params):
    """
    Run complete analysis of optimization results
    """
    
    analyzer = OptimizationAnalyzer()
    
    print("Optimization analysis")
    print("="*60)
    
    # 1. Convergence analysis
    print("\n1. Analyzing convergence...")
    analyzer.plot_convergence_analysis(results['history'])
    
    # 2. Parameter analysis
    print("\n2. Analyzing parameter changes...")
    initial_params = optimizer.base_synaptic_params.connection_modulation
    final_params = results['optimized_connection_modulation']
    analyzer.plot_parameter_analysis(initial_params, final_params, targets)
    
    # 3. Circuit validation
    print("\n3. Validating optimized circuit...")
    validation_results = analyzer.validate_optimized_circuit(
        results['optimized_synaptic_params'],
        circuit_params, opsin_params, targets,
        mec_drive_levels=[150.0, 200.0, 250.0]
    )
    
    # 4. Sensitivity analysis
    print("\n4. Running sensitivity analysis...")
    from DG_circuit_dendritic_somatic_transfer import PerConnectionSynapticParams
    sensitivity_results = analyzer.sensitivity_analysis(
        results['optimized_synaptic_params'],
        circuit_params, opsin_params, targets,
        perturbation_size=0.15
    )
    
    return {
        'validation': validation_results,
        'sensitivity': sensitivity_results,
        'analyzer': analyzer
    }



class OptogeneticOptimizationAnalyzer(OptimizationAnalyzer):
    """
    Enhanced analyzer supporting both baseline and optogenetic optimization
    """
    
    def __init__(self, figsize=(15, 10)):
        super().__init__(figsize)
        self.optogenetic_colors = {
            'pv': '#E74C3C',  # Red for PV
            'sst': '#3498DB',  # Blue for SST
            'gc': '#2ECC71',  # Green for GC
            'mc': '#F39C12'   # Orange for MC
        }
    
    def plot_optogenetic_convergence(self, history: Dict, 
                                     targets: CombinedOptimizationTargets,
                                     save_fig: bool = True):
        """
        Plot convergence analysis including optogenetic objectives
        """
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.ravel()
        
        iterations = range(len(history['loss']))
        
        # 1. Total loss convergence
        ax1 = axes[0]
        ax1.plot(iterations, history['loss'], 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Combined Loss Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add moving average
        window_size = min(20, len(history['loss']) // 10)
        if window_size > 1:
            moving_avg = pd.Series(history['loss']).rolling(window=window_size).mean()
            ax1.plot(iterations, moving_avg, 'r--', linewidth=2, 
                    alpha=0.7, label='Moving Avg')
            ax1.legend()
        
        # 2. Baseline vs Optogenetic loss components
        ax2 = axes[1]
        if history['loss_components']:
            # Extract baseline and optogenetic components
            baseline_losses = []
            opto_losses = []
            
            for entry in history['loss_components']:
                baseline = 0.0
                opto = 0.0
                
                for key, value in entry.items():
                    if 'rate_increase' in key or 'gini' in key or 'activated' in key:
                        opto += value
                    else:
                        baseline += value
                
                baseline_losses.append(baseline * targets.baseline_weight)
                opto_losses.append(opto * targets.optogenetic_weight)
            
            ax2.plot(iterations, baseline_losses, label='Baseline', 
                    linewidth=2, alpha=0.8, color='#2ECC71')
            ax2.plot(iterations, opto_losses, label='Optogenetic', 
                    linewidth=2, alpha=0.8, color='#E74C3C')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Weighted Loss Component')
            ax2.set_title('Baseline vs Optogenetic Loss Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # 3. Detailed loss components
        ax3 = axes[2]
        if history['loss_components']:
            component_names = list(history['loss_components'][0].keys())
            
            for comp_name in component_names[:5]:  # Show top 5
                comp_values = [entry[comp_name] for entry in history['loss_components']]
                ax3.plot(iterations, comp_values, label=comp_name, 
                        linewidth=1.5, alpha=0.7)
            
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Loss Component Value')
            ax3.set_title('Detailed Loss Components')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # 4. Firing rate evolution
        ax4 = axes[3]
        if history['firing_rates']:
            first_condition = list(history['firing_rates'][0].keys())[0]
            populations = ['gc', 'mc', 'pv', 'sst']
            
            for pop in populations:
                rates = []
                for entry in history['firing_rates']:
                    if first_condition in entry and pop in entry[first_condition]:
                        rates.append(entry[first_condition][pop])
                    else:
                        rates.append(np.nan)
                
                ax4.plot(iterations, rates, label=f'{pop.upper()}', 
                        linewidth=2, alpha=0.8, 
                        color=self.optogenetic_colors.get(pop, 'gray'))
            
            # Add target lines
            for pop in populations:
                if pop in targets.target_rates:
                    ax4.axhline(y=targets.target_rates[pop], 
                              linestyle='--', alpha=0.5,
                              color=self.optogenetic_colors.get(pop, 'gray'))
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Firing Rate (Hz)')
            ax4.set_title('Population Firing Rates vs Targets')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Parameter evolution (key connections)
        ax5 = axes[4]
        if history['parameters']:
            key_connections = ['mec_gc', 'pv_gc', 'mc_gc', 'sst_gc']
            
            for conn_name in key_connections:
                if conn_name in history['parameters'][0]:
                    param_values = [entry[conn_name] for entry in history['parameters']]
                    ax5.plot(iterations, param_values, label=conn_name, 
                            linewidth=2, alpha=0.8)
            
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Connection Strength Multiplier')
            ax5.set_title('Key Connection Evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Loss gradient analysis
        ax6 = axes[5]
        if len(history['loss']) > 10:
            gradients = np.gradient(np.array(history['loss']))
            ax6.plot(iterations, np.abs(gradients), 'purple', 
                    linewidth=1.5, alpha=0.7)
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('|Loss Gradient|')
            ax6.set_title('Convergence Rate')
            ax6.grid(True, alpha=0.3)
            ax6.set_yscale('log')
        
        # 7. PV stimulation effects (if available in history)
        ax7 = axes[6]
        ax7.set_title('PV Stimulation Effects')
        ax7.text(0.5, 0.5, 'Run validation for\nstimulation analysis', 
                ha='center', va='center', fontsize=12)
        ax7.axis('off')
        
        # 8. SST stimulation effects
        ax8 = axes[7]
        ax8.set_title('SST Stimulation Effects')
        ax8.text(0.5, 0.5, 'Run validation for\nstimulation analysis', 
                ha='center', va='center', fontsize=12)
        ax8.axis('off')
        
        # 9. Parameter variance
        ax9 = axes[8]
        if history['parameters']:
            param_vars = []
            for entry in history['parameters']:
                param_values = list(entry.values())
                param_vars.append(np.var(param_values))
            
            ax9.plot(iterations, param_vars, 'purple', linewidth=2, alpha=0.8)
            ax9.set_xlabel('Iteration')
            ax9.set_ylabel('Parameter Variance')
            ax9.set_title('Parameter Diversity')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('optogenetic_optimization_convergence.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_optogenetic_validation(self, performance_data: Dict,
                                    targets: CombinedOptimizationTargets,
                                    save_fig: bool = True):
        """
        Comprehensive validation plots including optogenetic effects
        """
        
        fig = plt.figure(figsize=(20, 14))
        
        # Use middle MEC drive level for main analysis
        drive_keys = list(performance_data.keys())
        middle_drive = drive_keys[len(drive_keys) // 2]
        data = performance_data[middle_drive]
        
        # 1. Baseline firing rates
        ax1 = plt.subplot(3, 4, 1)
        populations = ['gc', 'mc', 'pv', 'sst']
        
        for pop in populations:
            if pop in data['baseline']:
                target_rate = targets.target_rates[pop]
                actual_rate = data['baseline'][pop]['mean_rate']
                
                ax1.scatter(target_rate, actual_rate, s=100, 
                          color=self.optogenetic_colors.get(pop, 'gray'),
                          label=pop.upper(), alpha=0.7)
        
        max_rate = max([targets.target_rates[pop] for pop in populations])
        ax1.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5)
        ax1.set_xlabel('Target Firing Rate (Hz)')
        ax1.set_ylabel('Actual Firing Rate (Hz)')
        ax1.set_title('Baseline: Target vs Actual Rates')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Baseline sparsity
        ax2 = plt.subplot(3, 4, 2)
        for pop in populations:
            if pop in data['baseline'] and pop in targets.sparsity_targets:
                target_sparsity = targets.sparsity_targets[pop]
                actual_sparsity = data['baseline'][pop]['sparsity']
                
                ax2.scatter(target_sparsity, actual_sparsity, s=100,
                          color=self.optogenetic_colors.get(pop, 'gray'),
                          label=pop.upper(), alpha=0.7)
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('Target Sparsity')
        ax2.set_ylabel('Actual Sparsity')
        ax2.set_title('Baseline: Target vs Actual Sparsity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PV stimulation - activated fractions
        ax3 = plt.subplot(3, 4, 3)
        pv_targets = targets.optogenetic_targets.target_rate_increases.get('pv', {})
        
        affected_pops = []
        target_fractions = []
        actual_fractions = []
        
        for pop in ['gc', 'mc', 'sst']:
            if pop in data['pv_stimulation']:
                affected_pops.append(pop.upper())
                actual_fractions.append(
                    data['pv_stimulation'][pop]['activated_fraction']
                )
                target_fractions.append(pv_targets.get(pop, 0.0))
        
        x_pos = np.arange(len(affected_pops))
        width = 0.35
        
        ax3.bar(x_pos - width/2, target_fractions, width, 
               label='Target', alpha=0.7, color='gray')
        ax3.bar(x_pos + width/2, actual_fractions, width,
               label='Actual', alpha=0.7, color='#E74C3C')
        
        ax3.set_xlabel('Population')
        ax3.set_ylabel('Activated Fraction')
        ax3.set_title('PV Stimulation: Activated Fractions')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(affected_pops)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. SST stimulation - activated fractions
        ax4 = plt.subplot(3, 4, 4)
        sst_targets = targets.optogenetic_targets.target_rate_increases.get('sst', {})
        
        affected_pops = []
        target_fractions = []
        actual_fractions = []
        
        for pop in ['gc', 'mc', 'pv']:
            if pop in data['sst_stimulation']:
                affected_pops.append(pop.upper())
                actual_fractions.append(
                    data['sst_stimulation'][pop]['activated_fraction']
                )
                target_fractions.append(sst_targets.get(pop, 0.0))
        
        x_pos = np.arange(len(affected_pops))
        
        ax4.bar(x_pos - width/2, target_fractions, width,
               label='Target', alpha=0.7, color='gray')
        ax4.bar(x_pos + width/2, actual_fractions, width,
               label='Actual', alpha=0.7, color='#3498DB')
        
        ax4.set_xlabel('Population')
        ax4.set_ylabel('Activated Fraction')
        ax4.set_title('SST Stimulation: Activated Fractions')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(affected_pops)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. PV stimulation - rate changes
        ax5 = plt.subplot(3, 4, 5)
        
        pops = []
        baseline_rates = []
        stim_rates = []
        
        for pop in ['gc', 'mc', 'sst']:
            if pop in data['pv_stimulation']:
                pops.append(pop.upper())
                baseline_rates.append(data['pv_stimulation'][pop]['baseline_mean'])
                stim_rates.append(data['pv_stimulation'][pop]['stim_mean'])
        
        x_pos = np.arange(len(pops))
        
        ax5.bar(x_pos - width/2, baseline_rates, width,
               label='Baseline', alpha=0.7, color='gray')
        ax5.bar(x_pos + width/2, stim_rates, width,
               label='Stimulation', alpha=0.7, color='#E74C3C')
        
        ax5.set_xlabel('Population')
        ax5.set_ylabel('Firing Rate (Hz)')
        ax5.set_title('PV Stimulation: Rate Changes')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(pops)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. SST stimulation - rate changes
        ax6 = plt.subplot(3, 4, 6)
        
        pops = []
        baseline_rates = []
        stim_rates = []
        
        for pop in ['gc', 'mc', 'pv']:
            if pop in data['sst_stimulation']:
                pops.append(pop.upper())
                baseline_rates.append(data['sst_stimulation'][pop]['baseline_mean'])
                stim_rates.append(data['sst_stimulation'][pop]['stim_mean'])
        
        x_pos = np.arange(len(pops))
        
        ax6.bar(x_pos - width/2, baseline_rates, width,
               label='Baseline', alpha=0.7, color='gray')
        ax6.bar(x_pos + width/2, stim_rates, width,
               label='Stimulation', alpha=0.7, color='#3498DB')
        
        ax6.set_xlabel('Population')
        ax6.set_ylabel('Firing Rate (Hz)')
        ax6.set_title('SST Stimulation: Rate Changes')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(pops)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. PV stimulation - Gini changes
        ax7 = plt.subplot(3, 4, 7)
        pv_gini_targets = targets.optogenetic_targets.target_gini_increase.get('pv', {})
        
        pops = []
        target_gini = []
        actual_gini = []
        
        for pop in ['gc', 'mc']:
            if pop in data['pv_stimulation']:
                pops.append(pop.upper())
                actual_gini.append(data['pv_stimulation'][pop]['gini_change'])
                target_gini.append(pv_gini_targets.get(pop, 0.0))
        
        x_pos = np.arange(len(pops))
        
        ax7.bar(x_pos - width/2, target_gini, width,
               label='Target', alpha=0.7, color='gray')
        ax7.bar(x_pos + width/2, actual_gini, width,
               label='Actual', alpha=0.7, color='#E74C3C')
        
        ax7.set_xlabel('Population')
        ax7.set_ylabel('Gini Coefficient Change')
        ax7.set_title('PV Stimulation: Inequality Changes')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(pops)
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 8. SST stimulation - Gini changes
        ax8 = plt.subplot(3, 4, 8)
        sst_gini_targets = targets.optogenetic_targets.target_gini_increase.get('sst', {})
        
        pops = []
        target_gini = []
        actual_gini = []
        
        for pop in ['gc', 'mc']:
            if pop in data['sst_stimulation']:
                pops.append(pop.upper())
                actual_gini.append(data['sst_stimulation'][pop]['gini_change'])
                target_gini.append(sst_gini_targets.get(pop, 0.0))
        
        x_pos = np.arange(len(pops))
        
        ax8.bar(x_pos - width/2, target_gini, width,
               label='Target', alpha=0.7, color='gray')
        ax8.bar(x_pos + width/2, actual_gini, width,
               label='Actual', alpha=0.7, color='#3498DB')
        
        ax8.set_xlabel('Population')
        ax8.set_ylabel('Gini Coefficient Change')
        ax8.set_title('SST Stimulation: Inequality Changes')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(pops)
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 9. Baseline rate distributions
        ax9 = plt.subplot(3, 4, 9)
        # This would require individual cell data - placeholder
        ax9.set_title('Rate Distributions')
        ax9.text(0.5, 0.5, 'Individual cell data\nnot available in summary',
                ha='center', va='center')
        ax9.axis('off')
        
        # 10. Performance across MEC drives
        ax10 = plt.subplot(3, 4, 10)
        
        mec_drives = []
        gc_rates = []
        
        for drive_key in performance_data.keys():
            drive_val = float(drive_key.split('_')[1])
            mec_drives.append(drive_val)
            
            if 'gc' in performance_data[drive_key]['baseline']:
                gc_rates.append(
                    performance_data[drive_key]['baseline']['gc']['mean_rate']
                )
        
        ax10.plot(mec_drives, gc_rates, 'o-', linewidth=2, 
                 markersize=8, color='#2ECC71')
        ax10.axhline(y=targets.target_rates['gc'], linestyle='--', 
                    color='gray', alpha=0.5, label='Target')
        ax10.set_xlabel('MEC Drive (pA)')
        ax10.set_ylabel('GC Firing Rate (Hz)')
        ax10.set_title('GC Activity vs Input Drive')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # 11. E/I balance
        ax11 = plt.subplot(3, 4, 11)
        
        ei_ratios = []
        drive_labels = []
        
        for drive_key in performance_data.keys():
            drive_val = float(drive_key.split('_')[1])
            drive_labels.append(f'{drive_val:.0f}')
            
            data_point = performance_data[drive_key]['baseline']
            exc = (data_point.get('gc', {}).get('mean_rate', 0) +
                  data_point.get('mc', {}).get('mean_rate', 0))
            inh = (data_point.get('pv', {}).get('mean_rate', 0) +
                  data_point.get('sst', {}).get('mean_rate', 0))
            
            if inh > 0:
                ei_ratios.append(exc / inh)
        
        if ei_ratios:
            bars = ax11.bar(drive_labels, ei_ratios, alpha=0.7, color='orange')
            ax11.set_xlabel('MEC Drive (pA)')
            ax11.set_ylabel('E/I Activity Ratio')
            ax11.set_title('Excitation-Inhibition Balance')
            ax11.grid(True, alpha=0.3, axis='y')
            
            for bar, ratio in zip(bars, ei_ratios):
                ax11.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{ratio:.2f}', ha='center', va='bottom')
        
        # 12. Summary table
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        summary_data = []
        middle_data = performance_data[middle_drive]
        
        # Baseline summary
        summary_data.append(['Baseline', '', '', ''])
        for pop in populations:
            if pop in middle_data['baseline']:
                target = targets.target_rates[pop]
                actual = middle_data['baseline'][pop]['mean_rate']
                error = abs(actual - target) / target * 100
                summary_data.append([
                    f'{pop.upper()}',
                    f'{target:.1f}',
                    f'{actual:.1f}',
                    f'{error:.1f}%'
                ])
        
        # PV stim summary
        summary_data.append(['', '', '', ''])
        summary_data.append(['PV STIM', 'Activated', 'Rate Change', 'Gini Change'])
        for pop in ['gc', 'mc', 'sst']:
            if pop in middle_data['pv_stimulation']:
                data_point = middle_data['pv_stimulation'][pop]
                summary_data.append([
                    f'{pop.upper()}',
                    f"{data_point['activated_fraction']:.1%}",
                    f"{data_point['mean_change']:+.1f}",
                    f"{data_point['gini_change']:+.3f}"
                ])
        
        table = ax12.table(cellText=summary_data,
                          cellLoc='center', loc='center',
                          colWidths=[0.2, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.8)
        ax12.set_title('Performance Summary', pad=20, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('optogenetic_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_baseline_vs_combined_optimization(self, 
                                                  baseline_results: Dict,
                                                  combined_results: Dict,
                                                  save_fig: bool = True):
        """
        Compare results from baseline-only vs combined optimization
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        # 1. Loss convergence comparison
        ax1 = axes[0]
        baseline_loss = baseline_results['history']['loss']
        combined_loss = combined_results['history']['loss']
        
        ax1.plot(baseline_loss, label='Baseline Only', 
                linewidth=2, alpha=0.7)
        ax1.plot(combined_loss, label='Combined', 
                linewidth=2, alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Convergence Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Final parameter comparison
        ax2 = axes[1]
        baseline_params = baseline_results['optimized_connection_modulation']
        combined_params = combined_results['optimized_connection_modulation']
        
        connections = list(baseline_params.keys())
        baseline_vals = [baseline_params[c] for c in connections]
        combined_vals = [combined_params[c] for c in connections]
        
        x_pos = np.arange(len(connections))
        width = 0.35
        
        ax2.bar(x_pos - width/2, baseline_vals, width, 
               label='Baseline', alpha=0.7)
        ax2.bar(x_pos + width/2, combined_vals, width,
               label='Combined', alpha=0.7)
        
        ax2.set_xlabel('Connection')
        ax2.set_ylabel('Strength Multiplier')
        ax2.set_title('Final Parameters Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(connections, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Parameter differences
        ax3 = axes[2]
        param_diffs = [(combined_vals[i] - baseline_vals[i]) / baseline_vals[i] * 100
                      for i in range(len(connections))]
        
        colors = ['red' if d < 0 else 'blue' for d in param_diffs]
        bars = ax3.bar(connections, param_diffs, color=colors, alpha=0.7)
        
        ax3.set_xlabel('Connection')
        ax3.set_ylabel('Parameter Change (%)')
        ax3.set_title('Combined vs Baseline Parameter Changes')
        ax3.set_xticklabels(connections, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # 4-6: Population-specific comparisons
        for idx, pop in enumerate(['gc', 'pv', 'sst']):
            ax = axes[3 + idx]
            
            # Extract firing rates over optimization
            baseline_rates = []
            combined_rates = []
            
            for entry in baseline_results['history']['firing_rates']:
                first_cond = list(entry.keys())[0]
                if pop in entry[first_cond]:
                    baseline_rates.append(entry[first_cond][pop])
            
            for entry in combined_results['history']['firing_rates']:
                first_cond = list(entry.keys())[0]
                if pop in entry[first_cond]:
                    combined_rates.append(entry[first_cond][pop])
            
            if baseline_rates and combined_rates:
                ax.plot(baseline_rates, label='Baseline Only', 
                       linewidth=2, alpha=0.7)
                ax.plot(combined_rates, label='Combined',
                       linewidth=2, alpha=0.7)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Firing Rate (Hz)')
                ax.set_title(f'{pop.upper()} Rate Evolution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('baseline_vs_combined_comparison.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_optogenetic_sensitivity(self, 
                                       circuit_factory_data: tuple,
                                       optimized_params: Dict,
                                       targets: CombinedOptimizationTargets,
                                       perturbation_size: float = 0.15,
                                       save_fig: bool = True):
        """
        Analyze sensitivity of optogenetic effects to parameter changes
        """
        
        print("Analyzing optogenetic sensitivity...")
        
        circuit_params, base_synaptic_params_dict, opsin_params = circuit_factory_data
        
        # Get baseline optogenetic performance
        baseline_pv = simulate_optogenetic_stimulation(
            circuit_factory_data, optimized_params, 'pv', 1.0, mec_current=200.0
        )
        baseline_sst = simulate_optogenetic_stimulation(
            circuit_factory_data, optimized_params, 'sst', 1.0, mec_current=200.0
        )
        
        sensitivity_results = {'pv': {}, 'sst': {}}
        
        # Test key connections
        key_connections = ['pv_gc', 'sst_gc', 'pv_mc', 'sst_mc', 
                           'mc_gc', 'gc_pv', 'gc_sst']
        
        for conn_name in key_connections:
            if conn_name not in optimized_params:
                continue
            
            print(f"  Testing {conn_name}...")
            
            # Test perturbations
            for direction, multiplier in [('increase', 1 + perturbation_size),
                                         ('decrease', 1 - perturbation_size)]:
                
                perturbed_params = optimized_params.copy()
                perturbed_params[conn_name] *= multiplier
                
                # Test PV stimulation
                pv_result = simulate_optogenetic_stimulation(
                    circuit_factory_data, perturbed_params, 'pv', 
                    1.0, mec_current=200.0
                )
                
                # Test SST stimulation
                sst_result = simulate_optogenetic_stimulation(
                    circuit_factory_data, perturbed_params, 'sst',
                    1.0, mec_current=200.0
                )
                
                # Calculate sensitivity metrics
                pv_sensitivity = self._calculate_opto_sensitivity(
                    baseline_pv, pv_result, perturbation_size
                )
                sst_sensitivity = self._calculate_opto_sensitivity(
                    baseline_sst, sst_result, perturbation_size
                )
                
                if conn_name not in sensitivity_results['pv']:
                    sensitivity_results['pv'][conn_name] = {}
                if conn_name not in sensitivity_results['sst']:
                    sensitivity_results['sst'][conn_name] = {}
                
                sensitivity_results['pv'][conn_name][direction] = pv_sensitivity
                sensitivity_results['sst'][conn_name][direction] = sst_sensitivity
        
        # Plot results
        self._plot_optogenetic_sensitivity(sensitivity_results, save_fig)
        
        return sensitivity_results
    
    def _calculate_opto_sensitivity(self, baseline: Dict, 
                                    perturbed: Dict,
                                    perturbation_size: float) -> Dict:
        """Calculate sensitivity metrics for optogenetic effects"""
        
        sensitivity = {}
        
        for pop in baseline.keys():
            if pop in perturbed:
                # Activated fraction sensitivity
                baseline_act = baseline[pop]['activated_fraction']
                perturbed_act = perturbed[pop]['activated_fraction']
                
                if baseline_act > 0:
                    act_sens = ((perturbed_act - baseline_act) / baseline_act / 
                               perturbation_size)
                else:
                    act_sens = 0.0
                
                # Gini change sensitivity
                baseline_gini = baseline[pop]['gini_change']
                perturbed_gini = perturbed[pop]['gini_change']
                
                gini_sens = ((perturbed_gini - baseline_gini) / 
                            (abs(baseline_gini) + 1e-6) / perturbation_size)
                
                sensitivity[pop] = {
                    'activated_fraction': act_sens,
                    'gini_change': gini_sens
                }
        
        return sensitivity
    
    def _plot_optogenetic_sensitivity(self, sensitivity_results: Dict,
                                     save_fig: bool):
        """Plot optogenetic sensitivity analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # PV stimulation - activated fraction sensitivity
        ax1 = axes[0, 0]
        self._plot_sensitivity_bars(
            ax1, sensitivity_results['pv'], 'activated_fraction',
            'PV Stimulation: Activated Fraction Sensitivity'
        )
        
        # PV stimulation - Gini sensitivity
        ax2 = axes[0, 1]
        self._plot_sensitivity_bars(
            ax2, sensitivity_results['pv'], 'gini_change',
            'PV Stimulation: Gini Change Sensitivity'
        )
        
        # SST stimulation - activated fraction sensitivity
        ax3 = axes[1, 0]
        self._plot_sensitivity_bars(
            ax3, sensitivity_results['sst'], 'activated_fraction',
            'SST Stimulation: Activated Fraction Sensitivity'
        )
        
        # SST stimulation - Gini sensitivity
        ax4 = axes[1, 1]
        self._plot_sensitivity_bars(
            ax4, sensitivity_results['sst'], 'gini_change',
            'SST Stimulation: Gini Change Sensitivity'
        )
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('optogenetic_sensitivity.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_sensitivity_bars(self, ax, sensitivity_data: Dict,
                               metric: str, title: str):
        """Helper to plot sensitivity bars"""
        
        connections = list(sensitivity_data.keys())
        populations = ['gc', 'mc', 'pv', 'sst']
        
        # Extract data
        increase_vals = {pop: [] for pop in populations}
        decrease_vals = {pop: [] for pop in populations}
        
        for conn in connections:
            for pop in populations:
                if ('increase' in sensitivity_data[conn] and
                    pop in sensitivity_data[conn]['increase']):
                    inc_val = sensitivity_data[conn]['increase'][pop].get(metric, 0)
                    dec_val = sensitivity_data[conn]['decrease'][pop].get(metric, 0)
                    
                    increase_vals[pop].append(inc_val)
                    decrease_vals[pop].append(dec_val)
                else:
                    increase_vals[pop].append(0)
                    decrease_vals[pop].append(0)
        
        # Plot grouped bars
        x_pos = np.arange(len(connections))
        width = 0.15
        
        for i, pop in enumerate(populations):
            offset = (i - 1.5) * width
            ax.bar(x_pos + offset, increase_vals[pop], width,
                  label=f'{pop.upper()} +', alpha=0.7,
                  color=self.optogenetic_colors.get(pop, 'gray'))
        
        ax.set_xlabel('Connection')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} Sensitivity')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(connections, rotation=45, ha='right')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)


def load_and_analyze_optogenetic_results(json_filename: str):
    """
    Load optimization results from JSON and create comprehensive analysis
    """
    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    print(f"\nLoaded optimization results from: {json_filename}")
    print(f"Method: {data['optimization_info']['method']}")
    print(f"Best loss: {data['optimization_info']['best_loss']:.6f}")
    
    # Create analyzer
    analyzer = OptogeneticOptimizationAnalyzer()
    
    # Reconstruct targets
    opto_targets = OptogeneticTargets(
        target_rate_increases=data['optogenetic_targets']['target_rate_increases'],
        target_gini_increase=data['optogenetic_targets']['target_gini_increases'],
        stimulation_intensity=data['optogenetic_targets']['stimulation_intensity']
    )
    
    base_targets = OptimizationTargets(
        target_rates=data['baseline_targets']['firing_rates'],
        sparsity_targets=data['baseline_targets']['sparsity_targets']
    )
    
    combined_targets = CombinedOptimizationTargets(
        target_rates=base_targets.target_rates,
        sparsity_targets=base_targets.sparsity_targets,
        optogenetic_targets=opto_targets,
        baseline_weight=data['optimization_weights']['baseline_weight'],
        optogenetic_weight=data['optimization_weights']['optogenetic_weight']
    )
    
    # Plot validation results
    print("\nCreating validation plots...")
    analyzer.plot_optogenetic_validation(
        data['performance'],
        combined_targets,
        save_fig=True
    )
    
    # Print summary
    if 'performance_summary' in data:
        print("\n" + "="*70)
        print("Optimization summary")
        print("="*70)
        print(f"\nMethod: {data['optimization_info']['method']}")
        print(f"Baseline weight: {data['optimization_weights']['baseline_weight']:.1%}")
        print(f"Optogenetic weight: {data['optimization_weights']['optogenetic_weight']:.1%}")
        print(f"Final loss: {data['optimization_info']['best_loss']:.6f}")
    
    return analyzer, data, combined_targets


if __name__ == "__main__":
    
    # Load and analyze results from JSON file
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'DG_optogenetic_optimization_results.json'
    
    try:
        analyzer, data, targets = load_and_analyze_optogenetic_results(json_file)
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Please run optogenetic optimization first to generate results.")
