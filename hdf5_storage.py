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
                       save_full_activity: bool = False) -> None:
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
    trial_grp.create_dataset('time', data=to_numpy(trial_result['time']))
    
    # Save activity traces (use aggregated if available, otherwise raw)
    if 'activity_trace_mean' in trial_result:
        activity_data = trial_result['activity_trace_mean']
    else:
        activity_data = trial_result['activity_trace']
    
    for pop in ['gc', 'mc', 'pv', 'sst', 'mec']:
        if pop in activity_data:
            dataset_name = f'activity_{pop}'
            if save_full_activity:
                trial_grp.create_dataset(dataset_name, 
                                        data=to_numpy(activity_data[pop]),
                                        compression='gzip', compression_opts=4)
            else:
                # Save only summary statistics to save space
                pop_activity = to_numpy(activity_data[pop])
                trial_grp.attrs[f'{dataset_name}_mean'] = np.mean(pop_activity)
                trial_grp.attrs[f'{dataset_name}_std'] = np.std(pop_activity)
    
    # Save opsin expression
    if 'opsin_expression_mean' in trial_result:
        trial_grp.create_dataset('opsin_expression',
                                data=to_numpy(trial_result['opsin_expression_mean']))
    elif 'opsin_expression' in trial_result:
        trial_grp.create_dataset('opsin_expression',
                                data=to_numpy(trial_result['opsin_expression']))
    
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


def aggregate_from_hdf5(f: h5py.File,
                        target: str,
                        intensity: float,
                        stim_start: float,
                        stim_duration: float,
                        warmup: float,
                        expression_threshold: float = 0.2) -> Dict:
    """
    Compute aggregated statistics directly from HDF5 file
    
    This enables analysis without loading all data into memory.
    """
    from nested_experiment import _aggregate_trial_group
    
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in f[target]:
        return {}
    
    # Group by connectivity
    conn_groups = {}
    
    for conn_key in f[target][intensity_key].keys():
        conn_idx = int(conn_key.split('_')[1])
        conn_groups[conn_idx] = []
        
        for pattern_key in f[target][intensity_key][conn_key].keys():
            pattern_idx = int(pattern_key.split('_')[1])
            
            # Load this trial
            trial_data = load_trial_from_hdf5(f, target, intensity, 
                                             conn_idx, pattern_idx)
            
            # Create NestedTrialResult-like object
            from nested_experiment import NestedTrialResult
            trial_result = NestedTrialResult(
                connectivity_idx=conn_idx,
                mec_pattern_idx=pattern_idx,
                seed=0,  # Not stored in HDF5
                results=trial_data,
                target_population=trial_data['target_population'],
                opsin_expression=trial_data['opsin_expression']
            )
            
            conn_groups[conn_idx].append(trial_result)
    
    # Aggregate within each connectivity instance
    by_conn = {}
    for conn_idx, trials in conn_groups.items():
        by_conn[conn_idx] = _aggregate_trial_group(
            trials, stim_start, stim_duration, warmup, expression_threshold
        )
    
    return by_conn


def compute_variance_from_hdf5(f: h5py.File,
                               target: str,
                               intensity: float,
                               stim_start: float,
                               stim_duration: float,
                               warmup: float) -> Dict:
    """
    Compute variance decomposition directly from HDF5
    
    Memory-efficient version that processes one connectivity at a time.
    """
    from nested_experiment import _decompose_population_variance
    
    intensity_key = f"intensity_{intensity}"
    
    if intensity_key not in f[target]:
        return {}
    
    # Load trials grouped by connectivity
    conn_groups = {}
    
    for conn_key in f[target][intensity_key].keys():
        conn_idx = int(conn_key.split('_')[1])
        conn_groups[conn_idx] = []
        
        for pattern_key in f[target][intensity_key][conn_key].keys():
            pattern_idx = int(pattern_key.split('_')[1])
            
            trial_data = load_trial_from_hdf5(f, target, intensity,
                                             conn_idx, pattern_idx)
            
            from nested_experiment import NestedTrialResult
            trial_result = NestedTrialResult(
                connectivity_idx=conn_idx,
                mec_pattern_idx=pattern_idx,
                seed=0,
                results=trial_data,
                target_population=trial_data['target_population'],
                opsin_expression=trial_data['opsin_expression']
            )
            
            conn_groups[conn_idx].append(trial_result)
    
    # Compute variance for each population
    pop_variance = {}
    for pop in ['gc', 'mc', 'pv', 'sst']:
        pop_variance[pop] = _decompose_population_variance(
            conn_groups, pop, stim_start, stim_duration, warmup
        )
    
    return pop_variance


def load_metadata_from_hdf5(filepath: str) -> Dict:
    """Load metadata from HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        
        # Load seed structure
        if 'seed_structure' in f['metadata']:
            seed_grp = f['metadata']['seed_structure']
            metadata['seed_structure'] = {
                'connectivity_seeds': seed_grp['connectivity_seeds'][:].tolist(),
                'mec_pattern_seeds': seed_grp['mec_pattern_seeds'][:].tolist()
            }
        
        # Convert 'None' strings back to None
        for key, value in metadata.items():
            if value == 'None':
                metadata[key] = None
        
    return metadata
