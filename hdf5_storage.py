"""
HDF5 Storage Utilities for Nested Experiments

Provides incremental saving and loading of nested experimental results
to avoid memory issues with large experiments.

"""

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger('hdf5_storage')
logger.setLevel(logging.INFO)


def create_hdf5_experiment_file(filepath: str, metadata: Dict) -> h5py.File:
    """
    Initialize HDF5 file with metadata and structure
    
    Args:
        filepath: Path to HDF5 file
        metadata: Experiment metadata dict
        
    Returns:
        Open h5py.File object (caller must close)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    f = h5py.File(filepath, 'w')
    
    # Store metadata as attributes
    meta_grp = f.create_group('metadata')
    for key, value in metadata.items():
        if key == 'seed_structure':
            # Handle seed structure separately
            seed_grp = meta_grp.create_group('seed_structure')
            seed_grp.create_dataset('connectivity_seeds', 
                                   data=np.array(value['connectivity_seeds']))
            seed_grp.create_dataset('mec_pattern_seeds',
                                   data=np.array(value['mec_pattern_seeds']))
        elif isinstance(value, (list, tuple)):
            meta_grp.attrs[key] = np.array(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            meta_grp.attrs[key] = value if value is not None else 'None'
    
    # Create target population groups
    f.create_group('pv')
    f.create_group('sst')
    
    logger.info(f"Created HDF5 experiment file: {filepath}")
    
    return f


def save_trial_to_hdf5(f: h5py.File,
                       target: str,
                       intensity: float,
                       connectivity_idx: int,
                       mec_pattern_idx: int,
                       trial_result: Dict,
                       save_full_activity: bool = False,
                       stim_start: float = None,
                       stim_duration: float = None,
                       warmup: float = None,
                       expression_threshold: float = 0.2) -> None:
    """
    Save a single trial result to HDF5 file
    
    Args:
        f: Open h5py.File object
        target: Target population ('pv' or 'sst')
        intensity: Light intensity
        connectivity_idx: Connectivity instance index
        mec_pattern_idx: MEC pattern index
        trial_result: Trial result dict from simulate_stimulation
        save_full_activity: Whether to save complete activity traces
        stim_start: Stimulation start time (needed for period stats)
        stim_duration: Stimulation duration (needed for period stats)
        warmup: Baseline period start (needed for period stats)
        expression_threshold: Opsin expression threshold for non-expressing cells
    """
    # Create group hierarchy
    intensity_key = f"intensity_{intensity}"
    conn_key = f"connectivity_{connectivity_idx}"
    pattern_key = f"pattern_{mec_pattern_idx}"
    
    # Navigate/create group structure
    if intensity_key not in f[target]:
        f[target].create_group(intensity_key)
    
    if conn_key not in f[target][intensity_key]:
        f[target][intensity_key].create_group(conn_key)
    
    trial_grp = f[target][intensity_key][conn_key].create_group(pattern_key)
    
    # Convert tensors to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x)
    
    # Save essential data
    time = to_numpy(trial_result['time'])
    trial_grp.create_dataset('time', data=time)
    
    # Save activity traces (use aggregated if available, otherwise raw)
    if 'activity_trace_mean' in trial_result:
        activity_data = trial_result['activity_trace_mean']
    else:
        activity_data = trial_result['activity_trace']
    
    # Get opsin expression for target population analysis
    if 'opsin_expression_mean' in trial_result:
        opsin_expression = to_numpy(trial_result['opsin_expression_mean'])
    elif 'opsin_expression' in trial_result:
        opsin_expression = to_numpy(trial_result['opsin_expression'])
    else:
        opsin_expression = None
    
    # Compute time masks if period info provided
    if stim_start is not None and stim_duration is not None and warmup is not None:
        baseline_mask = (time >= warmup) & (time < stim_start)
        stim_mask = (time >= stim_start) & (time <= (stim_start + stim_duration))
        compute_period_stats = True
    else:
        compute_period_stats = False
    
    for pop in ['gc', 'mc', 'pv', 'sst', 'mec']:
        if pop in activity_data:
            dataset_name = f'activity_{pop}'
            pop_activity = to_numpy(activity_data[pop])
            
            if save_full_activity:
                # Save full traces
                trial_grp.create_dataset(dataset_name, 
                                         data=pop_activity,
                                         compression='gzip',
                                         compression_opts=9)
            else:
                # Save summary statistics only
                trial_grp.attrs[f'{dataset_name}_mean'] = np.mean(pop_activity)
                trial_grp.attrs[f'{dataset_name}_std'] = np.std(pop_activity)
                
                # Save period-specific statistics if period info available
                if compute_period_stats:
                    # Compute statistics per cell for baseline and stim periods
                    baseline_rates = np.mean(pop_activity[:, baseline_mask], axis=1)
                    stim_rates = np.mean(pop_activity[:, stim_mask], axis=1)
                    rate_changes = stim_rates - baseline_rates
                    baseline_std = np.std(baseline_rates)
                    
                    # Check if this is target population
                    is_target = (pop == target)
                    
                    if is_target and opsin_expression is not None:
                        # For target population, also save non-expressing cell stats
                        non_expressing_mask = opsin_expression < expression_threshold
                        n_nonexpr = np.sum(non_expressing_mask)
                        
                        if n_nonexpr > 0:
                            baseline_rates_nonexpr = baseline_rates[non_expressing_mask]
                            stim_rates_nonexpr = stim_rates[non_expressing_mask]
                            rate_changes_nonexpr = rate_changes[non_expressing_mask]
                            baseline_std_nonexpr = np.std(baseline_rates_nonexpr)
                            
                            excited_frac_nonexpr = np.mean(rate_changes_nonexpr > baseline_std_nonexpr)
                            
                            # Save non-expressing stats
                            trial_grp.attrs[f'{pop}_excited_nonexpr'] = excited_frac_nonexpr
                            trial_grp.attrs[f'{pop}_mean_change_nonexpr'] = np.mean(rate_changes_nonexpr)
                            trial_grp.attrs[f'{pop}_baseline_rate_nonexpr'] = np.mean(baseline_rates_nonexpr)
                            trial_grp.attrs[f'{pop}_stim_rate_nonexpr'] = np.mean(stim_rates_nonexpr)
                            trial_grp.attrs[f'{pop}_n_nonexpr'] = n_nonexpr
                    
                    # Standard statistics (all cells)
                    excited_frac = np.mean(rate_changes > baseline_std)
                    
                    trial_grp.attrs[f'{pop}_excited'] = excited_frac
                    trial_grp.attrs[f'{pop}_mean_change'] = np.mean(rate_changes)
                    trial_grp.attrs[f'{pop}_baseline_rate'] = np.mean(baseline_rates)
                    trial_grp.attrs[f'{pop}_stim_rate'] = np.mean(stim_rates)
    
    # Save opsin expression
    if opsin_expression is not None:
        trial_grp.create_dataset('opsin_expression', data=opsin_expression)
    
    # Save indices and metadata as attributes
    trial_grp.attrs['connectivity_idx'] = connectivity_idx
    trial_grp.attrs['mec_pattern_idx'] = mec_pattern_idx
    trial_grp.attrs['target_population'] = target
    
    if 'adaptive_stats' in trial_result:
        stats = trial_result['adaptive_stats']
        trial_grp.attrs['n_steps'] = stats.get('n_steps_mean', stats.get('n_steps', 0))
        trial_grp.attrs['avg_dt'] = stats.get('avg_dt_mean', stats.get('avg_dt', 0))


def load_trial_from_hdf5(f: h5py.File,
                         target: str,
                         intensity: float,
                         connectivity_idx: int,
                         mec_pattern_idx: int) -> Dict:
    """
    Load a single trial result from HDF5
    
    Returns:
        Trial result dict compatible with NestedTrialResult
    """
    intensity_key = f"intensity_{intensity}"
    conn_key = f"connectivity_{connectivity_idx}"
    pattern_key = f"pattern_{mec_pattern_idx}"
    
    trial_grp = f[target][intensity_key][conn_key][pattern_key]
    
    # Reconstruct activity traces
    activity_trace = {}
    for pop in ['gc', 'mc', 'pv', 'sst', 'mec']:
        dataset_name = f'activity_{pop}'
        if dataset_name in trial_grp:
            activity_trace[pop] = torch.from_numpy(trial_grp[dataset_name][:])
        elif f'{dataset_name}_mean' in trial_grp.attrs:
            # Only summary stats available - create placeholder
            activity_trace[pop] = None
    
    result = {
        'time': torch.from_numpy(trial_grp['time'][:]),
        'activity_trace': activity_trace,
        'opsin_expression': trial_grp['opsin_expression'][:],
        'connectivity_idx': trial_grp.attrs['connectivity_idx'],
        'mec_pattern_idx': trial_grp.attrs['mec_pattern_idx'],
        'target_population': trial_grp.attrs['target_population']
    }
    
    return result


def load_nested_trials_from_hdf5(
    f: h5py.File,
    target: str,
    intensity: float,
    n_connectivity: int,
    n_mec_patterns: int,
    require_full_activity: bool = True
) -> List:
    """
    Load all trials for a given condition as NestedTrialResult objects
    
    Args:
        f: Open h5py.File object
        target: Target population ('pv' or 'sst')
        intensity: Light intensity
        n_connectivity: Number of connectivity instances
        n_mec_patterns: Number of MEC patterns per connectivity
        require_full_activity: If True, raise error if full activity traces not available
        
    Returns:
        List of NestedTrialResult-like objects (dicts with required fields)
        
    Raises:
        ValueError: If require_full_activity=True and traces not available
    """
    from collections import namedtuple
    
    # Define minimal NestedTrialResult structure
    NestedTrialResult = namedtuple('NestedTrialResult', 
                                   ['connectivity_idx', 'mec_pattern_idx', 'seed',
                                    'results', 'target_population', 'opsin_expression'])
    
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in f[target]:
        return []
    
    trials = []
    has_full_activity = None  # Will be determined from first trial
    
    for conn_idx in range(n_connectivity):
        conn_key = f"connectivity_{conn_idx}"
        
        if conn_key not in f[target][intensity_key]:
            continue
        
        for pattern_idx in range(n_mec_patterns):
            pattern_key = f"pattern_{pattern_idx}"
            
            if pattern_key not in f[target][intensity_key][conn_key]:
                continue
            
            trial_grp = f[target][intensity_key][conn_key][pattern_key]
            
            # Check if this trial has full activity (check first population)
            if has_full_activity is None:
                has_full_activity = 'activity_gc' in trial_grp
                
                if require_full_activity and not has_full_activity:
                    raise ValueError(
                        f"HDF5 file does not contain full activity traces for {target} "
                        f"at intensity {intensity}. Nested weights analysis requires "
                        f"full activity traces. Please re-run the experiment with "
                        f"save_full_activity=True in the nested configuration."
                    )
            
            # Load activity traces
            activity_trace = {}
            for pop in ['gc', 'mc', 'pv', 'sst', 'mec']:
                dataset_name = f'activity_{pop}'
                if dataset_name in trial_grp:
                    activity_trace[pop] = torch.from_numpy(trial_grp[dataset_name][:])
                elif require_full_activity:
                    raise ValueError(
                        f"Missing activity trace for {pop} in trial "
                        f"(conn={conn_idx}, pattern={pattern_idx})"
                    )
                else:
                    activity_trace[pop] = None
            
            # Load time array
            time = torch.from_numpy(trial_grp['time'][:])
            
            # Load opsin expression
            if 'opsin_expression' in trial_grp:
                opsin_expression = trial_grp['opsin_expression'][:]
            else:
                opsin_expression = np.array([])
            
            # Create result dict
            results = {
                'time': time,
                'activity_trace': activity_trace,
            }
            
            # Create NestedTrialResult
            trial = NestedTrialResult(
                connectivity_idx=conn_idx,
                mec_pattern_idx=pattern_idx,
                seed=0,  # Not stored in HDF5
                results=results,
                target_population=trial_grp.attrs['target_population'],
                opsin_expression=opsin_expression
            )
            
            trials.append(trial)
    
    return trials

def get_trial_indices(f: h5py.File, 
                      target: str,
                      intensity: float) -> List[tuple]:
    """
    Get all (connectivity_idx, mec_pattern_idx) pairs for a condition
    
    Returns:
        List of (conn_idx, pattern_idx) tuples
    """
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in f[target]:
        return []
    
    indices = []
    for conn_key in f[target][intensity_key].keys():
        conn_idx = int(conn_key.split('_')[1])
        for pattern_key in f[target][intensity_key][conn_key].keys():
            pattern_idx = int(pattern_key.split('_')[1])
            indices.append((conn_idx, pattern_idx))
    
    return indices


def extract_trial_statistics_from_hdf5(f: h5py.File,
                                       target: str,
                                       intensity: float,
                                       n_connectivity: int,
                                       n_mec_patterns: int) -> Dict:
    """
    Extract per-trial statistics directly from HDF5 attributes
    
    Pure I/O function - reads pre-computed statistics without any analysis.
    
    Args:
        f: Open h5py.File object
        target: Target population
        intensity: Light intensity
        n_connectivity: Number of connectivity instances
        n_mec_patterns: Number of MEC patterns per connectivity
        
    Returns:
        Dict mapping conn_idx to list of trial statistics
    """
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in f[target]:
        return {}
    
    # Organize by connectivity
    conn_groups = {}
    
    for conn_idx in range(n_connectivity):
        conn_key = f"connectivity_{conn_idx}"
        
        if conn_key not in f[target][intensity_key]:
            continue
        
        conn_groups[conn_idx] = []
        
        for pattern_idx in range(n_mec_patterns):
            pattern_key = f"pattern_{pattern_idx}"
            
            if pattern_key not in f[target][intensity_key][conn_key]:
                continue
            
            trial_grp = f[target][intensity_key][conn_key][pattern_key]
            
            # Extract statistics for all populations
            trial_stats = {
                'connectivity_idx': conn_idx,
                'mec_pattern_idx': pattern_idx,
                'target_population': trial_grp.attrs['target_population'],
                'populations': {}
            }
            
            for pop in ['gc', 'mc', 'pv', 'sst']:
                # Check if this population has data
                excited_key = f'{pop}_excited'
                if excited_key not in trial_grp.attrs:
                    continue
                
                pop_stats = {
                    'excited': trial_grp.attrs[f'{pop}_excited'],
                    'mean_change': trial_grp.attrs[f'{pop}_mean_change'],
                    'baseline_rate': trial_grp.attrs[f'{pop}_baseline_rate'],
                    'stim_rate': trial_grp.attrs[f'{pop}_stim_rate']
                }
                
                # Add non-expressing cell stats if this is target population
                if pop == target and f'{pop}_excited_nonexpr' in trial_grp.attrs:
                    pop_stats['excited_nonexpr'] = trial_grp.attrs[f'{pop}_excited_nonexpr']
                    pop_stats['mean_change_nonexpr'] = trial_grp.attrs[f'{pop}_mean_change_nonexpr']
                    pop_stats['baseline_rate_nonexpr'] = trial_grp.attrs[f'{pop}_baseline_rate_nonexpr']
                    pop_stats['stim_rate_nonexpr'] = trial_grp.attrs[f'{pop}_stim_rate_nonexpr']
                    pop_stats['n_nonexpr'] = trial_grp.attrs.get(f'{pop}_n_nonexpr', 0)
                
                trial_stats['populations'][pop] = pop_stats
            
            conn_groups[conn_idx].append(trial_stats)
    
    return conn_groups


def load_metadata_from_hdf5(filepath: str) -> Dict:
    """Load metadata from HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        metadata = {}
        
        # Load attributes
        for key, value in f['metadata'].attrs.items():
            # Convert numpy arrays to lists
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            # Handle scalar 'None' strings
            elif isinstance(value, (str, bytes)):
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                metadata[key] = None if value == 'None' else value
            else:
                metadata[key] = value
        
        # Load seed structure
        if 'seed_structure' in f['metadata']:
            seed_grp = f['metadata']['seed_structure']
            metadata['seed_structure'] = {
                'connectivity_seeds': seed_grp['connectivity_seeds'][:].tolist(),
                'mec_pattern_seeds': seed_grp['mec_pattern_seeds'][:].tolist()
            }
    
    return metadata
