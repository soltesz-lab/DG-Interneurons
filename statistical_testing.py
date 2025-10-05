#!/usr/bin/env python3
"""
Statistical testing framework for disinhibition hypothesis in DG circuit
Compatible with DG_circuit_dendritic_somatic_transfer.py and DG_protocol.py
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
from pathlib import Path
import tqdm

# Import DG circuit components
from DG_circuit_dendritic_somatic_transfer import (
    DentateCircuit,
    CircuitParams,
    PerConnectionSynapticParams,
    OpsinParams
)
from DG_protocol import (
    OptogeneticExperiment,
    OpsinExpression
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
    pv_paradoxical_gc_fraction: float = 0.15  # ~15% of GCs show paradoxical excitation
    pv_paradoxical_mc_fraction: float = 0.08  # ~8% of MCs
    pv_paradoxical_sst_fraction: float = 0.12
    pv_response_latency_ms: float = 5.0
    pv_gini_increase: float = 0.15  # Increase in firing rate inequality
    


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


class DisinhibitionHypothesisTester:
    """
    Statistical testing framework for disinhibition hypothesis
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
            optimization_json_file=self.optimization_json_file
        )
        
        # Run simulation
        if condition_params.get('no_stimulation', False):
            light_intensity = 0.0
        
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

    def _run_trial_worker(self, trial_args: Tuple) -> Tuple[int, str, Optional[Dict]]:
        """
        Worker function for parallel trial execution
        
        Args:
            trial_args: Tuple of (trial_idx, target_population, light_intensity, 
                                  condition_params, trial_seed, mec_current, opsin_current,
                                  duration, stim_start)
        
        Returns:
            Tuple of (trial_idx, condition_name, trial_result or None)
        """
        (trial_idx, target_population, light_intensity, condition_name,
         condition_params, trial_seed, mec_current, opsin_current,
         duration, stim_start) = trial_args
        
        try:
            # Force CPU usage for multiprocessing (GPU doesn't work well with fork)
            import torch
            torch.set_num_threads(1)  # Prevent nested parallelism
            
            trial_result = self.run_single_trial(
                target_population,
                light_intensity,
                condition_params,
                trial_seed,
                mec_current=mec_current,
                opsin_current=opsin_current,
                duration=duration,
                stim_start=stim_start
            )
            return (trial_idx, condition_name, trial_result)
            
        except Exception as e:
            print(f"    Warning: Trial {trial_idx} failed: {e}")
            return (trial_idx, condition_name, None)

    
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
            
            paradoxically_excited = stim_rate > excitation_threshold
            paradoxically_inhibited = stim_rate < (baseline_mean - baseline_std)
            
            total_cells = len(stim_rate)
            
            # Calculate metrics
            trial_metrics[f'{pop}_paradoxical_fraction'] = torch.mean(paradoxically_excited.float()).item()
            trial_metrics[f'{pop}_inhibited_fraction'] = torch.mean(paradoxically_inhibited.float()).item()
            trial_metrics[f'{pop}_mean_baseline'] = baseline_mean.item()
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
                             use_multiprocessing: bool = True) -> Dict:
        """
        Run Monte Carlo analysis with multiple independent trials
        
        Args:
            target_population: 'pv' or 'sst'
            light_intensity: Optogenetic stimulation intensity
            mec_current: MEC drive current (pA)
            opsin_current: Direct opsin activation current (pA)
            n_workers: Number of parallel processes (None = use CPU count - 1)
            use_multiprocessing: Whether to use parallel processing
        
        Returns:
            Dictionary with raw results and statistical analysis
        """
        
        print(f"\nRunning Monte Carlo analysis: {self.n_trials} trials for {target_population.upper()} stimulation")
        if use_multiprocessing:
            if n_workers is None:
                n_workers = max(1, mp.cpu_count() - 1)
            print(f"Using {n_workers} parallel processes")
        print("=" * 70)
        
        # Define experimental conditions
        conditions = {
            'full_network': {
                'inhibition_scale': 1.0,
                'description': 'Full network (control)'
            },
            'reduced_inhibition_50': {
                'inhibition_scale': 0.5,
                'description': '50% reduced inhibition'
            },
            'reduced_inhibition_10': {
                'inhibition_scale': 0.1,
                'description': '90% reduced inhibition'
            },
            'cnqx_apv': {
                'excitation_scale': 0.1,
                'description': 'CNQX/APV (90% blocked glutamate)'
            },
            'gabazine': {
                'pv_inhibition_scale': 0.1,
                'description': 'Gabazine (90% blocked GABA-A/PV)'
            },
            'no_stimulation': {
                'no_stimulation': True,
                'description': 'No optogenetic stimulation'
            }
        }
        
        # Storage for results
        results = {condition: [] for condition in conditions.keys()}

        if use_multiprocessing:
            # Parallel execution
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
                        550.0    # stim_start
                    )
                    all_trial_args.append(trial_args)
            
            # Run trials in parallel
            print(f"\nRunning {len(all_trial_args)} total trials across {len(conditions)} conditions...")
            
            with mp.Pool(processes=n_workers) as pool:
                # Use imap_unordered for progress tracking
                trial_results = list(tqdm.tqdm(
                    pool.imap_unordered(self._run_trial_worker, all_trial_args),
                    total=len(all_trial_args),
                    desc="Processing trials"
                ))
            
            # Organize results by condition
            for trial_idx, condition, trial_result in trial_results:
                if trial_result is not None:
                    results[condition].append(trial_result)
        else:
            # Serial execution
            # Run trials for each condition
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
        
        # Primary comparison: full network vs CNQX/APV (tests disinhibition hypothesis)
        print("  Primary hypothesis test: Full network vs CNQX/APV...")
        analysis['primary_hypothesis'] = {}
        
        for metric in metrics_to_analyze:
            full_values = [trial.get(metric, np.nan) for trial in results['full_network']]
            cnqx_values = [trial.get(metric, np.nan) for trial in results['cnqx_apv']]
            
            # Remove NaN values
            full_values = [v for v in full_values if not np.isnan(v)]
            cnqx_values = [v for v in cnqx_values if not np.isnan(v)]
            
            if len(full_values) > 0 and len(cnqx_values) > 0:
                stat_result = self._statistical_comparison(
                    full_values, cnqx_values, metric
                )
                analysis['primary_hypothesis'][metric] = stat_result
        
        # Secondary comparisons
        secondary_comparisons = [
            ('full_network', 'reduced_inhibition_50', 'Disinhibition mechanism'),
            ('full_network', 'reduced_inhibition_10', 'Strong disinhibition'),
            ('full_network', 'gabazine', 'PV-specific blockade'),
            ('full_network', 'no_stimulation', 'Optogenetic control')
        ]
        
        print("  Secondary hypothesis tests...")
        for condition1, condition2, desc in secondary_comparisons:
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
        report.append("\nComparison: Full network vs CNQX/APV (glutamate blockade)")
        
        primary = analysis['primary_hypothesis']
        for metric, result in primary.items():
            report.append(f"\n{metric}:")
            report.append(f"  Full network:    {result.mean_treatment:.4f} +/- {result.std_treatment:.4f}")
            report.append(f"  CNQX/APV:        {result.mean_control:.4f} +/- {result.std_control:.4f}")
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
                report.append("  * Highly significant reduction in paradoxical excitation with CNQX/APV")
                report.append(f"  * Large effect size (d={network_paradoxical.cohens_d:.2f}) indicates biological relevance")
                report.append(f"  * Statistical power: {network_paradoxical.statistical_power:.2f}")
            elif network_paradoxical.p_value < 0.05:
                report.append("\nModerate evidence for disinhibition hypothesis:")
                report.append("  * Significant reduction with CNQX/APV")
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
        'reduced_inhibition_50': '#A23B72',
        'reduced_inhibition_10': '#F18F01',
        'cnqx_apv': '#C73E1D',
        'gabazine': '#6A994E',
        'no_stimulation': '#999999'
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
    plot_effect_sizes(ax4, analysis['primary_hypothesis'], populations)
    
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
    plot_power_analysis(ax10, analysis['primary_hypothesis'], populations)
    
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
    x_pos = np.arange(len(conditions))
    width = 0.8 / len(populations)
    
    for i, pop in enumerate(populations):
        means = []
        sems = []
        
        for condition in conditions:
            trials = raw_results[condition]
            values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in trials]
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
    ax.set_ylabel('Paradoxical Fraction')
    ax.set_title('Paradoxical Excitation Fractions')
    ax.set_xticks(x_pos + width * (len(populations) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


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
    ax.set_title('Effect Sizes (Full vs CNQX/APV)')
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
    
    # Compare full network vs CNQX/APV
    full_trials = raw_results['full_network']
    cnqx_trials = raw_results['cnqx_apv']
    
    full_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in full_trials]
    cnqx_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in cnqx_trials]
    
    full_values = [v for v in full_values if not np.isnan(v)]
    cnqx_values = [v for v in cnqx_values if not np.isnan(v)]
    
    if len(full_values) > 0 and len(cnqx_values) > 0:
        # Violin plots
        parts = ax.violinplot([full_values, cnqx_values], positions=[1, 2],
                             showmeans=True, showextrema=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['#2E86AB', '#C73E1D'][i])
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Full\nNetwork', 'CNQX/\nAPV'], fontsize=8)
        ax.set_ylabel('Paradoxical Fraction')
        ax.set_title(f'{pop.upper()} Distribution')
        ax.grid(True, alpha=0.3)


def plot_network_paradoxical(ax, raw_results, condition_colors):
    """Plot network-wide paradoxical excitation"""
    
    conditions = list(raw_results.keys())
    means = []
    sems = []
    colors = []
    
    for condition in conditions:
        trials = raw_results[condition]
        values = [t.get('network_total_paradoxical', np.nan) for t in trials]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            means.append(np.mean(values))
            sems.append(np.std(values) / np.sqrt(len(values)))
            colors.append(condition_colors.get(condition, '#999999'))
        else:
            means.append(0)
            sems.append(0)
            colors.append('#999999')
    
    x_pos = np.arange(len(conditions))
    bars = ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7, capsize=5)
    
    ax.set_xlabel('Condition')
    ax.set_ylabel('Total Paradoxical Fraction')
    ax.set_title('Network-Wide Paradoxical Excitation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add significance stars
    full_mean = means[0]  # Assuming full_network is first
    for i, (mean, sem) in enumerate(zip(means, sems)):
        if i > 0 and abs(mean - full_mean) > 2 * (sem + sems[0]):
            ax.text(i, mean + sem, '***', ha='center', va='bottom', fontsize=12)


def plot_pvalue_heatmap(ax, analysis):
    """Plot p-value heatmap for all comparisons"""
    
    primary = analysis['primary_hypothesis']
    
    metrics = list(primary.keys())
    p_values = [-np.log10(primary[m].p_value + 1e-10) for m in metrics]
    
    # Create heatmap data
    heatmap_data = np.array(p_values).reshape(-1, 1)
    
    im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=0, vmax=4)
    
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([m.replace('_', ' ') for m in metrics], fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['Full vs\nCNQX/APV'], fontsize=8)
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
    primary = analysis['primary_hypothesis']
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
        'reduced_inhibition_50': '#A23B72',
        'reduced_inhibition_10': '#F18F01',
        'cnqx_apv': '#C73E1D',
        'gabazine': '#6A994E',
        'no_stimulation': '#999999'
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
    
    # Panel 2: Paradoxical excitation - comparison of full network vs CNQX/APV
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
    
    conditions = ['full_network', 'reduced_inhibition_50', 'reduced_inhibition_10', 
                 'cnqx_apv', 'gabazine', 'no_stimulation']
    
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
                    'reduced_inhibition_50': 'Inh-50%',
                    'reduced_inhibition_10': 'Inh-10%',
                    'cnqx_apv': 'CNQX/APV',
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
    """Violin plots comparing full network vs CNQX/APV for all populations"""
    
    violin_data = []
    labels = []
    plot_colors = []
    
    for pop in populations:
        # Full network data
        full_trials = raw_results['full_network']
        full_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in full_trials]
        full_values = [v for v in full_values if not np.isnan(v)]
        
        # CNQX/APV data
        cnqx_trials = raw_results['cnqx_apv']
        cnqx_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in cnqx_trials]
        cnqx_values = [v for v in cnqx_values if not np.isnan(v)]
        
        if len(full_values) > 0 and len(cnqx_values) > 0:
            violin_data.extend([full_values, cnqx_values])
            labels.extend([f'{pop.upper()}\nFull', f'{pop.upper()}\nCNQX'])
            plot_colors.extend([colors[pop], '#CCCCCC'])
    
    if violin_data:
        positions = []
        current_pos = 0
        for i in range(len(violin_data)):
            if i % 2 == 0:
                positions.append(current_pos)
            else:
                positions.append(current_pos + 0.5)
                current_pos += 1.5
        
        parts = ax.violinplot(violin_data, positions=positions,
                             showmeans=True, showextrema=True, widths=0.4)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add significance markers
        for i in range(0, len(violin_data), 2):
            full_data = violin_data[i]
            cnqx_data = violin_data[i+1]
            
            # Simple t-test for significance
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(full_data, cnqx_data)
            
            # Add significance stars
            y_max = max(max(full_data), max(cnqx_data))
            x_pos = (positions[i] + positions[i+1]) / 2
            
            if p_value < 0.001:
                marker = '***'
            elif p_value < 0.01:
                marker = '**'
            elif p_value < 0.05:
                marker = '*'
            else:
                marker = 'ns'
            
            ax.text(x_pos, y_max * 1.1, marker, ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Paradoxical Excitation\nFraction', fontsize=10)
    ax.set_title('Primary Hypothesis Test:\nFull Network vs CNQX/APV', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


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
    
    conditions = ['full_network', 'reduced_inhibition_50', 'reduced_inhibition_10',
                 'cnqx_apv', 'gabazine', 'no_stimulation']
    
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
                    'reduced_inhibition_50': 'Inh-50',
                    'reduced_inhibition_10': 'Inh-10',
                    'cnqx_apv': 'CNQX',
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
    
    conditions = ['full_network', 'reduced_inhibition_50', 'reduced_inhibition_10',
                 'cnqx_apv', 'gabazine', 'no_stimulation']
    
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
                    'reduced_inhibition_50': 'Reduced\nInh-50%',
                    'reduced_inhibition_10': 'Reduced\nInh-10%',
                    'cnqx_apv': 'CNQX/\nAPV',
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
    primary = analysis['primary_hypothesis']
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
    

def run_statistical_analysis(
    target_populations: List[str] = ['pv', 'sst'],
    n_trials: int = 50,
    light_intensity: float = 1.0,
    mec_current: float = 100.0,
    opsin_current: float = 100.0,
    n_workers: int = 1,
    optimization_json_file: str = None,
    output_dir: str = "./statistical_results"):
    """
    Run statistical analysis for both PV and SST stimulation
    
    Args:
        target_populations: List of populations to test
        n_trials: Number of Monte Carlo trials per condition
        light_intensity: Optogenetic stimulation intensity
        mec_current: MEC drive current (pA)
        opsin_current: Direct opsin activation current (pA)
        optimization_json_file: Path to optimization results (optional)
        output_dir: Directory for saving results
    """
    
    print("=" * 80)
    print("Statistical analysis of dg disinhibition hypothesis")
    print(f"Multi-trial Monte Carlo approach (n={n_trials} trials per condition)")
    print("=" * 80)
    
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
        
        # Run Monte Carlo analysis
        mc_results = tester.monte_carlo_analysis(
            target,
            light_intensity=light_intensity,
            mec_current=mec_current,
            opsin_current=opsin_current,
            n_workers = n_workers,
            use_multiprocessing = True)
        
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
        
        # Create comprehensive plots (existing function)
        print("\nGenerating statistical analysis plots...")
        plot_monte_carlo_results(
            mc_results,
            validation_results,
            output_dir=str(output_path),
            show_plots=False
        )
        
        # Create violin plots showing multi-trial distributions
        print("Generating multi-trial violin plots...")
        plot_paradoxical_excitation_violins(
            mc_results,
            output_dir=str(output_path),
            show_plots=False
        )
        
        # Additional detailed violin plot
        print("Generating detailed violin plots...")
        create_violin_plots(
            mc_results,
            validation_results,
            output_dir=str(output_path),
            show_plots=False
        )
        
        all_results[target] = {
            'mc_results': mc_results,
            'validation': validation_results,
            'report': report
        }
    
    print("\n" + "=" * 80)
    print(f"Analysis complete. Results saved to: {output_path}")
    print("\nGenerated files:")
    print("  - monte_carlo_analysis_[target].png (overall analysis)")
    print("  - paradoxical_excitation_violins_[target].png (multi-trial distributions)")
    print("  - detailed_violins_[target].png (detailed violin plots)")
    print("  - report_[target].txt (statistical report)")
    print("  - mc_results_[target].pkl (raw data)")
    print("=" * 80)
    
    return all_results


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
    
    # Panel A: Primary comparison (Full vs CNQX/APV) - 2x2 grid
    for idx, pop in enumerate(populations):
        ax = plt.subplot(3, 2, idx + 1)
        
        # Get data
        full_trials = raw_results['full_network']
        cnqx_trials = raw_results['cnqx_apv']
        
        full_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in full_trials]
        cnqx_values = [t.get(f'{pop}_paradoxical_fraction', np.nan) for t in cnqx_trials]
        
        full_values = [v for v in full_values if not np.isnan(v)]
        cnqx_values = [v for v in cnqx_values if not np.isnan(v)]
        
        if len(full_values) > 0 and len(cnqx_values) > 0:
            # Create violin plot
            parts = ax.violinplot([full_values, cnqx_values], positions=[0, 1],
                                 showmeans=False, showextrema=False, widths=0.7)
            
            # Color violins
            parts['bodies'][0].set_facecolor('#2E86AB')
            parts['bodies'][1].set_facecolor('#C73E1D')
            for pc in parts['bodies']:
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Add boxplot overlay for quartiles
            bp = ax.boxplot([full_values, cnqx_values], positions=[0, 1],
                          widths=0.3, showfliers=False,
                          boxprops=dict(linewidth=1.5, color='black'),
                          whiskerprops=dict(linewidth=1.5, color='black'),
                          capprops=dict(linewidth=1.5, color='black'),
                          medianprops=dict(linewidth=2, color='red'))
            
            # Add individual data points with jitter
            for i, values in enumerate([full_values, cnqx_values]):
                y = values
                x = np.random.normal(i, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.4, s=15, color='black', zorder=10)
            
            # Statistical test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(full_values, cnqx_values)
            
            # Add significance marker
            y_max = max(max(full_values), max(cnqx_values))
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
            cohens_d = (np.mean(full_values) - np.mean(cnqx_values)) / \
                      np.sqrt((np.var(full_values) + np.var(cnqx_values)) / 2)
            
            # Add text with statistics
            stats_text = f"d={cohens_d:.2f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Full\nNetwork', 'CNQX/\nAPV'], fontsize=9)
        ax.set_ylabel('Paradoxical\nExcitation Fraction', fontsize=9)
        ax.set_title(f'{pop.upper()}', fontsize=11, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
    
    # Panel E: Network-wide effect
    ax5 = plt.subplot(3, 2, 5)
    
    conditions = ['full_network', 'cnqx_apv', 'gabazine', 'no_stimulation']
    condition_labels = ['Full\nNetwork', 'CNQX/\nAPV', 'Gabazine', 'No\nStim']
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
    primary = analysis['primary_hypothesis']
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


if __name__ == "__main__":
    # Example usage
    print("Statistical Testing Framework for DG Circuit")
    print("=" * 70)
    
    # Check for optimization file
    optimization_file = None
    if len(sys.argv) > 1:
        optimization_file = sys.argv[1]
        print(f"Using optimization results: {optimization_file}")
    
    # Run analysis
    results = run_statistical_analysis(
        target_populations=['pv', 'sst'],
        n_trials=2,
        light_intensity=1.0,
        mec_current=40.0,
        opsin_current=200.0,
        n_workers = 4,
        optimization_json_file=optimization_file,
        output_dir="./statistical_results"
    )
    
