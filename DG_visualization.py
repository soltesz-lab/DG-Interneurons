#!/usr/bin/env python3
"""
Dentate Gyrus Circuit Visualization Module

Visualization tools for spatial organization, connectivity patterns,
and circuit dynamics in the dentate gyrus model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from typing import Dict, Tuple, List, Optional, Union
import torch
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set style parameters
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    # Figure parameters
    figsize_large: Tuple[int, int] = (15, 10)
    figsize_medium: Tuple[int, int] = (12, 8)
    figsize_small: Tuple[int, int] = (10, 6)
    dpi: int = 300
    
    # Cell type colors (anatomically meaningful)
    colors: Dict[str, str] = None
    
    # Marker sizes
    marker_sizes: Dict[str, float] = None
    
    # 3D view parameters
    elevation: float = 20
    azimuth: float = 45
    
    # Connectivity visualization
    connection_alpha: float = 0.3
    connection_width: float = 0.5
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'gc': '#2E8B57',      # Sea green (granule cells - dense, sparse activity)
                'mc': '#FF6347',      # Tomato (mossy cells - excitatory)
                'pv': '#4169E1',      # Royal blue (PV+ interneurons - fast spiking)
                'sst': '#9370DB',     # Medium purple (SST+ interneurons - dendritic)
                'mec': '#FFD700',     # Gold (MEC input)
            }
            
        if self.marker_sizes is None:
            self.marker_sizes = {
                'gc': 25,    # Small (numerous granule cells)
                'mc': 60,    # Medium (mossy cells)
                'pv': 50,    # Medium (PV+ interneurons)
                'sst': 45,   # Medium (SST+ interneurons)
                'mec': 70,   # Larger (external input)
            }

class DGCircuitVisualization:
    """Visualization toolkit for dentate gyrus circuits"""
    
    def __init__(self, circuit, config: VisualizationConfig = None):
        """
        Initialize visualization object with a DentateCircuit instance
        
        Args:
            circuit: DentateCircuit model instance
            config: VisualizationConfig for styling
        """
        self.circuit = circuit
        self.config = config or VisualizationConfig()
        self.layout = circuit.layout
        self.connectivity = circuit.connectivity
        
        # Extract population sizes
        self.pop_sizes = {
            'gc': circuit.circuit_params.n_gc,
            'mc': circuit.circuit_params.n_mc,
            'pv': circuit.circuit_params.n_pv,
            'sst': circuit.circuit_params.n_sst,
            'mec': circuit.circuit_params.n_mec,
        }
        
        # Cache positions as numpy arrays for efficiency
        self._position_cache = {}
        for pop, positions in self.layout.positions.items():
            self._position_cache[pop] = positions.cpu().numpy()
    
    def plot_spatial_organization(self, view='3d', populations=None, show_layers=True,
                                  alpha=0.7, save_path=None):
        """
        Plot 3D spatial organization of all cell populations
        
        Args:
            view: '3d' or '2d' projection view
            populations: List of populations to show (None for all)
            show_layers: Whether to show anatomical layer boundaries
            alpha: Transparency of points
            save_path: Path to save figure
        """
        if populations is None:
            populations = ['gc', 'mc', 'pv', 'sst', 'mec']
        
        if view == '3d':
            fig = plt.figure(figsize=self.config.figsize_large, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot each population
            for pop in populations:
                if pop in self._position_cache:
                    pos = self._position_cache[pop]
                    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                             c=self.config.colors[pop], 
                             s=self.config.marker_sizes[pop],
                             alpha=alpha, label=f'{pop.upper()} (n={len(pos)})')
            
            if show_layers:
                self._add_anatomical_layers_3d(ax)
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.view_init(elev=self.config.elevation, azim=self.config.azimuth)
            
        else:  # 2D projection
            fig, axes = plt.subplots(1, 3, figsize=self.config.figsize_large, dpi=self.config.dpi)
            projections = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
            
            for ax, (xlabel, ylabel, dim1, dim2) in zip(axes, projections):
                for pop in populations:
                    if pop in self._position_cache:
                        pos = self._position_cache[pop]
                        ax.scatter(pos[:, dim1], pos[:, dim2], 
                                 c=self.config.colors[pop],
                                 s=self.config.marker_sizes[pop] * 0.5,
                                 alpha=alpha, label=f'{pop.upper()}')
                
                ax.set_xlabel(f'{xlabel} (mm)')
                ax.set_ylabel(f'{ylabel} (mm)')
                ax.grid(True, alpha=0.3)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Dentate Gyrus Spatial Organization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _add_anatomical_layers_3d(self, ax):
        """Add anatomical layer boundaries to 3D plot"""
        params = self.circuit.circuit_params
        
        # Define layer boundaries
        layers = [
            ('Molecular Layer', -params.ml_thickness, 0, 'lightcoral'),
            ('Granule Cell Layer', -params.gc_layer_thickness/2, params.gc_layer_thickness/2, 'lightgreen'),
            ('Hilar Region', 0, params.hilar_thickness, 'lightyellow'),
        ]
        
        # Create semi-transparent layer indicators
        x_range = [-2, 2]
        y_range = [-2, 2]
        
        for name, z_bottom, z_top, color in layers:
            # Create wireframe box for each layer
            xx, yy = np.meshgrid(x_range, y_range)
            ax.plot_surface(xx, yy, np.full_like(xx, z_bottom), alpha=0.1, color=color)
            ax.plot_surface(xx, yy, np.full_like(xx, z_top), alpha=0.1, color=color)
    
    def plot_connectivity_matrix(self, connection_type=None, populations=None, 
                                normalize=True, save_path=None):
        """
        Plot connectivity matrices as heatmaps
        
        Args:
            connection_type: Specific connection to plot (e.g., 'gc_mc')
            populations: List of populations for full connectivity matrix
            normalize: Whether to normalize connection strengths
            save_path: Path to save figure
        """
        if connection_type:
            # Plot single connection type
            fig, ax = plt.subplots(figsize=self.config.figsize_medium, dpi=self.config.dpi)
            
            conn_name = f'conn_{connection_type}'
            if hasattr(self.circuit, conn_name):
                matrix = getattr(self.circuit, conn_name).cpu().numpy()
                
                if normalize:
                    matrix = matrix / (matrix.max() + 1e-8)
                
                im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
                
                # Add labels
                pre_pop, post_pop = connection_type.split('_')
                ax.set_title(f'{pre_pop.upper()} → {post_pop.upper()} Connectivity', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel(f'Post-synaptic {post_pop.upper()} cells')
                ax.set_ylabel(f'Pre-synaptic {pre_pop.upper()} cells')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Connection Strength' if normalize else 'Connection Weight')
                
                # Add connection statistics
                n_connections = np.sum(matrix > 0)
                connection_prob = n_connections / (matrix.shape[0] * matrix.shape[1])
                ax.text(0.02, 0.98, f'Connections: {n_connections}\nProbability: {connection_prob:.3f}',
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        else:
            # Plot full connectivity matrix
            if populations is None:
                populations = ['gc', 'mc', 'pv', 'sst']
            
            # Build full connectivity matrix
            total_size = sum(self.pop_sizes[pop] for pop in populations)
            full_matrix = np.zeros((total_size, total_size))
            
            # Create population boundaries for visualization
            boundaries = [0]
            for pop in populations:
                boundaries.append(boundaries[-1] + self.pop_sizes[pop])
            
            # Fill in connectivity blocks
            for i, pre_pop in enumerate(populations):
                for j, post_pop in enumerate(populations):
                    conn_name = f'conn_{pre_pop}_{post_pop}'
                    if hasattr(self.circuit, conn_name):
                        matrix = getattr(self.circuit, conn_name).cpu().numpy()
                        
                        pre_start, pre_end = boundaries[i], boundaries[i+1]
                        post_start, post_end = boundaries[j], boundaries[j+1]
                        
                        full_matrix[pre_start:pre_end, post_start:post_end] = matrix
            
            # Plot full matrix
            fig, ax = plt.subplots(figsize=self.config.figsize_large, dpi=self.config.dpi)
            
            if normalize:
                full_matrix = full_matrix / (full_matrix.max() + 1e-8)
            
            im = ax.imshow(full_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            
            # Add population boundaries
            for boundary in boundaries[1:-1]:
                ax.axhline(boundary, color='black', linewidth=2)
                ax.axvline(boundary, color='black', linewidth=2)
            
            # Add population labels
            for i, pop in enumerate(populations):
                center = (boundaries[i] + boundaries[i+1]) / 2
                ax.text(center, -total_size * 0.05, pop.upper(), 
                       ha='center', va='top', fontsize=12, fontweight='bold')
                ax.text(-total_size * 0.05, center, pop.upper(), 
                       ha='right', va='center', fontsize=12, fontweight='bold', rotation=90)
            
            ax.set_title('Full Dentate Gyrus Connectivity Matrix', fontsize=16, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Connection Strength' if normalize else 'Connection Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_connectivity(self, connection_type, max_connections=1000, 
                                 connection_threshold=0.1, save_path=None):
        """
        Plot connectivity patterns overlaid on spatial positions
        
        Args:
            connection_type: Connection to visualize (e.g., 'gc_mc')
            max_connections: Maximum number of connections to draw
            connection_threshold: Minimum connection strength to show
            save_path: Path to save figure
        """
        pre_pop, post_pop = connection_type.split('_')
        
        # Get connection matrix
        conn_name = f'conn_{connection_type}'
        if not hasattr(self.circuit, conn_name):
            print(f"Connection type {connection_type} not found")
            return None
        
        matrix = getattr(self.circuit, conn_name).cpu().numpy()
        pre_pos = self._position_cache[pre_pop]
        post_pos = self._position_cache[post_pop]
        
        # Find connections above threshold
        pre_indices, post_indices = np.where(matrix > connection_threshold)
        connection_strengths = matrix[pre_indices, post_indices]
        
        # Subsample if too many connections
        if len(pre_indices) > max_connections:
            # Preferentially sample stronger connections
            sample_probs = connection_strengths / np.sum(connection_strengths)
            selected_indices = np.random.choice(len(pre_indices), max_connections, 
                                              replace=False, p=sample_probs)
            pre_indices = pre_indices[selected_indices]
            post_indices = post_indices[selected_indices]
            connection_strengths = connection_strengths[selected_indices]
        
        # Create 3D plot
        fig = plt.figure(figsize=self.config.figsize_large, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot cell positions
        ax.scatter(pre_pos[:, 0], pre_pos[:, 1], pre_pos[:, 2],
                  c=self.config.colors[pre_pop], s=self.config.marker_sizes[pre_pop],
                  alpha=0.6, label=f'{pre_pop.upper()} (n={len(pre_pos)})')
        
        ax.scatter(post_pos[:, 0], post_pos[:, 1], post_pos[:, 2],
                  c=self.config.colors[post_pop], s=self.config.marker_sizes[post_pop],
                  alpha=0.6, label=f'{post_pop.upper()} (n={len(post_pos)})')
        
        # Draw connections
        for i in range(len(pre_indices)):
            pre_idx, post_idx = pre_indices[i], post_indices[i]
            strength = connection_strengths[i]
            
            # Line from pre to post
            ax.plot([pre_pos[pre_idx, 0], post_pos[post_idx, 0]],
                   [pre_pos[pre_idx, 1], post_pos[post_idx, 1]],
                   [pre_pos[pre_idx, 2], post_pos[post_idx, 2]],
                   color='gray', alpha=self.config.connection_alpha * strength,
                   linewidth=self.config.connection_width)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.view_init(elev=self.config.elevation, azim=self.config.azimuth)
        
        plt.legend()
        plt.title(f'{pre_pop.upper()} → {post_pop.upper()} Spatial Connectivity\n'
                 f'Showing {len(pre_indices)} connections > {connection_threshold}',
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_distance_distribution(self, connection_types=None, bins=50, save_path=None):
        """
        Plot distribution of connection distances for different connection types
        
        Args:
            connection_types: List of connections to analyze
            bins: Number of histogram bins
            save_path: Path to save figure
        """
        if connection_types is None:
            connection_types = ['gc_mc', 'mc_gc',
                                'gc_pv', 'mc_pv',
                                'pv_gc', 'mc_pv',
                                'mc_sst', 'sst_pv']
        
        fig, axes = plt.subplots(4, 2, figsize=self.config.figsize_large, dpi=self.config.dpi)
        axes = axes.flatten()
        
        distance_stats = {}
        
        for idx, connection_type in enumerate(connection_types):
            if idx >= len(axes):
                break
                
            pre_pop, post_pop = connection_type.split('_')
            
            # Check if connection exists
            conn_name = f'conn_{connection_type}'
            if not hasattr(self.circuit, conn_name):
                continue
            
            # Calculate distances for all possible connections
            distance_matrix = self.layout.distance_matrix(pre_pop, post_pop).cpu().numpy()
            connection_matrix = getattr(self.circuit, conn_name).cpu().numpy()
            
            # Get distances for actual connections
            connected_distances = distance_matrix[connection_matrix > 0]
            all_distances = distance_matrix.flatten()
            
            ax = axes[idx]
            
            # Plot histograms
            ax.hist(all_distances, bins=bins, alpha=0.5, color='gray', 
                   label='All possible', density=True)
            ax.hist(connected_distances, bins=bins, alpha=0.7, 
                   color=self.config.colors[pre_pop], label='Connected', density=True)
            
            ax.set_xlabel('Distance (mm)')
            ax.set_ylabel('Density')
            ax.set_title(f'{pre_pop.upper()} → {post_pop.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate statistics
            distance_stats[connection_type] = {
                'mean_connected': np.mean(connected_distances),
                'std_connected': np.std(connected_distances),
                'mean_all': np.mean(all_distances),
                'n_connections': len(connected_distances)
            }
        
        # Remove unused subplots
        for idx in range(len(connection_types), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Connection Distance Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig, distance_stats
    
    def plot_mean_activity(self, activity_trace, save_path=None):
        """
        Plot circuit activity over time
        
        Args:

            save_path: Path to save figure
        """
        # Initialize storage
        activity_history = {pop: [] for pop in self.pop_sizes.keys()}

        timesteps = None
        for pop in activity_trace.keys():
            timesteps = activity_trace[pop].shape[1]
            break
        
        # Compute mean activity
        for t in range(timesteps):
            for pop, activity in activity_trace.items():
                activity = activity[:, t]
                activity_history[pop].append(activity.mean().item())
        
        # Plot results
        fig, axes = plt.subplots(2, 3, figsize=self.config.figsize_large, dpi=self.config.dpi)
        axes = axes.flatten()
        
        time_axis = np.arange(timesteps) * self.circuit.circuit_params.dt
        
        for idx, (pop, history) in enumerate(activity_history.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            ax.plot(time_axis, history, color=self.config.colors[pop], linewidth=2)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Mean Firing Rate (Hz)')
            ax.set_title(f'{pop.upper()} Population Activity')
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplot
        if len(activity_history) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.suptitle('DG Circuit Mean Activity', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig, activity_history

    def plot_activity_patterns(self, activity_trace, max_neurons_per_plot=50, 
                               individual_alpha=0.3, mean_linewidth=3, save_path=None):
        """
        Plot circuit activity over time with individual neuron traces and population mean

        Args:
            activity_trace: Dictionary with population keys, each containing array of shape (n_cells, timesteps)
            max_neurons_per_plot: Maximum number of individual neurons to plot per population (shows most active)
            individual_alpha: Transparency for individual neuron traces
            mean_linewidth: Line width for population mean trace
            save_path: Path to save figure
        """
        # Determine number of timesteps
        timesteps = None
        for pop in activity_trace.keys():
            timesteps = activity_trace[pop].shape[1]
            break

        # Create time axis
        time_axis = np.arange(timesteps) * self.circuit.circuit_params.dt

        # Plot results
        fig, axes = plt.subplots(2, 3, figsize=self.config.figsize_large, dpi=self.config.dpi)
        axes = axes.flatten()

        # Storage for mean activity
        activity_history = {pop: [] for pop in self.pop_sizes.keys()}

        for idx, (pop, activity) in enumerate(activity_trace.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get activity data (convert from torch tensor if needed)
            if hasattr(activity, 'cpu'):
                activity_data = activity.cpu().numpy()
            else:
                activity_data = np.array(activity)

            n_cells = activity_data.shape[0]

            # Determine which neurons to plot - select most active neurons
            if n_cells > max_neurons_per_plot:
                # Calculate mean activity for each neuron across time
                mean_activity_per_neuron = np.mean(activity_data, axis=1)

                # Get indices of top max_neurons_per_plot most active neurons
                neuron_indices = np.argsort(mean_activity_per_neuron)[-max_neurons_per_plot:]

                # Sort indices for consistent plotting order
                neuron_indices = np.sort(neuron_indices)
            else:
                neuron_indices = np.arange(n_cells)

            # Plot individual neuron traces with transparency
            for neuron_idx in neuron_indices:
                ax.plot(time_axis, np.round(activity_data[neuron_idx, :], 3),
                       color=self.config.colors[pop], 
                       alpha=individual_alpha, 
                       linewidth=0.5,
                       zorder=1)

            # Compute and plot mean activity
            mean_activity = np.round(np.mean(activity_data, axis=0), 3)
            activity_history[pop] = mean_activity.tolist()

            ax.plot(time_axis, mean_activity, 
                   color=self.config.colors[pop], 
                   linewidth=mean_linewidth, 
                   label=f'Mean (n={n_cells})',
                   zorder=2)

            # Add shaded region for standard deviation
            std_activity = np.round(np.std(activity_data, axis=0), 2)
            ax.fill_between(time_axis, 
                           mean_activity - std_activity, 
                           mean_activity + std_activity,
                           color=self.config.colors[pop], 
                           alpha=0.15,
                           zorder=0)

            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_title(f'{pop.upper()} Population Activity\n({n_cells} cells, showing {len(neuron_indices)} most active)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        # Remove unused subplot
        if len(activity_trace) < len(axes):
            fig.delaxes(axes[-1])

        plt.suptitle('DG Circuit Activity: Most Active Neurons and Population Mean', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig, activity_history


    def plot_activity_raster(self, activity_trace, vmin=0, vmax=None, 
                             mean_linewidth=2.5, cmap='seismic', save_path=None,
                             sort_by_activity=False, split_populations=None, direct_activation=None,
                             activation_cmap='plasma', activation_bar_width=None,
                             baseline_window=None, normalize_to_baseline=False,
                             activity_std=None, show_std_shading=True,
                             sorting_window=None,
                             use_global_colormap=True):
        """
        Plot circuit activity as raster plots with neurons sorted by mean firing rate

        Each neuron's activity is shown as a row in a heatmap, with color indicating
        firing rate. Neurons are ordered by mean firing rate (highest at top).
        Population mean firing rate is superimposed as a line trace.
        Optionally shows direct current injection as color-coded rectangles.

        Args:
            activity_trace: Dictionary with population keys, each containing array of shape (n_cells, timesteps)
            vmin: Minimum value for colormap (default: 0 Hz)
            vmax: Maximum value for colormap (default: auto-computed per population)
            mean_linewidth: Line width for population mean trace overlay
            cmap: Colormap name for activity (default: 'coolwarm')
            save_path: Path to save figure
            sort_by_activity: Whether to sort neurons by mean activity
            split_populations: Dict mapping population name to dict with 'unit_ids_part1' and 'unit_ids_part2'
            direct_activation: Dict mapping population names to arrays of current injection values (n_cells,)
            activation_cmap: Colormap name for direct activation visualization (default: 'plasma')
            activation_bar_width: Width of activation bar in time units (default: 5% of time range)
            baseline_window: Tuple (start_time, end_time) in ms for baseline calculation. If provided,
                            activity can be normalized relative to this baseline period.
            normalize_to_baseline: If True and baseline_window is provided, normalize activity by
                                  subtracting baseline mean and dividing by baseline std
            activity_std: Dictionary with population keys containing std across trials (n_cells, timesteps).
                         If provided, will show shaded regions for trial-to-trial variability
            show_std_shading: Whether to show standard deviation shading (only if activity_std provided)
            use_global_colormap: If True and normalize_to_baseline=True, use same colormap 
                            range for all populations for fair comparison
        """
        # Determine number of timesteps
        timesteps = None
        for pop in activity_trace.keys():
            timesteps = activity_trace[pop].shape[1]
            break

        # Create time axis
        time_axis = np.arange(timesteps) * self.circuit.circuit_params.dt

        # Calculate global colormap limits if requested
        global_vmin, global_vmax = None, None
        if normalize_to_baseline and use_global_colormap:
            all_data = []
            for pop, activity in activity_trace.items():
                if hasattr(activity, 'cpu'):
                    activity_data = activity.cpu().numpy()
                else:
                    activity_data = np.array(activity)
                all_data.append(activity_data.flatten())

            all_data = np.concatenate(all_data)
            all_data = all_data[np.isfinite(all_data)]

            if len(all_data) > 0:
                global_vmax = np.percentile(np.abs(all_data), 99)
                if global_vmax < 0.1:
                    global_vmax = 1.0
                global_vmin = -global_vmax
        
        # Calculate baseline statistics if requested
        baseline_stats = {}
        if baseline_window is not None and normalize_to_baseline:
            baseline_start, baseline_end = baseline_window
            baseline_mask = (time_axis >= baseline_start) & (time_axis < baseline_end)

            for pop, activity in activity_trace.items():
                if hasattr(activity, 'cpu'):
                    activity_data = activity.cpu().numpy()
                else:
                    activity_data = np.array(activity)

                # Calculate baseline mean and std for each neuron
                baseline_activity = activity_data[:, baseline_mask]
                baseline_mean = np.mean(baseline_activity, axis=1, keepdims=True)
                baseline_std = np.std(baseline_activity, axis=1, keepdims=True)
                baseline_std = np.maximum(baseline_std, 0.1)  # Prevent division by zero

                baseline_stats[pop] = {
                    'mean': baseline_mean,
                    'std': baseline_std
                }

        n_populations = len(activity_trace)
        if split_populations:
            for pop in split_populations:
                if pop in activity_trace:
                    n_populations += 1

        fig, axes = plt.subplots(2, 3, figsize=self.config.figsize_large,
                                 dpi=self.config.dpi, constrained_layout=True)
        axes = axes.flatten()

        # Adjust subplot spacing to make room for colorbars
        fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.07, 
                            wspace=0.4, hspace=0.35)

        # Storage for mean activity and raster image objects
        activity_history = {pop: [] for pop in self.pop_sizes.keys()}
        image_objects = []
        
        panel_idx = 0
        for idx, (pop, activity) in enumerate(activity_trace.items()):
            if panel_idx >= len(axes):
                break

            # Get activity data
            if hasattr(activity, 'cpu'):
                activity_data = activity.cpu().numpy()
            else:
                activity_data = np.array(activity)

            # Get std data if provided
            std_data = None
            if activity_std is not None and pop in activity_std:
                if hasattr(activity_std[pop], 'cpu'):
                    std_data = activity_std[pop].cpu().numpy()
                else:
                    std_data = np.array(activity_std[pop])

            # Apply baseline normalization if requested
            if normalize_to_baseline and pop in baseline_stats:
                activity_data = (activity_data - baseline_stats[pop]['mean']) / baseline_stats[pop]['std']
                if std_data is not None:
                    std_data = std_data / baseline_stats[pop]['std']

            n_cells = activity_data.shape[0]

            # Get direct activation data
            activation_data = None
            if direct_activation is not None and pop in direct_activation:
                activation_data = direct_activation[pop]
                if hasattr(activation_data, 'cpu'):
                    activation_data = activation_data.cpu().numpy()
                else:
                    activation_data = np.array(activation_data)

            # Check if this population should be split
            should_split = split_populations and pop in split_populations
            if should_split:
                split_info = split_populations[pop]
                unit_ids_part1 = np.array(split_info['unit_ids_part1'])
                unit_ids_part2 = np.array(split_info['unit_ids_part2'])
                part1_label = split_info.get('part1_label', 'Part 1')
                part2_label = split_info.get('part2_label', 'Part 2')

                # Plot Part 1
                if len(unit_ids_part1) > 0:
                    ax = axes[panel_idx]
                    activity_part1 = activity_data[unit_ids_part1, :]
                    std_part1 = std_data[unit_ids_part1, :] if std_data is not None else None
                    activation_part1 = activation_data[unit_ids_part1] if activation_data is not None else None

                    im = self._plot_single_raster(ax, activity_part1, time_axis, pop, 
                                                  global_vmin if use_global_colormap else vmin,
                                                  global_vmax if use_global_colormap else vmax,
                                                  cmap, mean_linewidth,
                                                  title_suffix=f' - {part1_label}',
                                                  direct_activation=activation_part1,
                                                  activation_cmap=activation_cmap,
                                                  activation_bar_width=activation_bar_width,
                                                  normalize_to_baseline=normalize_to_baseline,
                                                  baseline_window=baseline_window,
                                                  activity_std=std_part1,
                                                  show_std_shading=show_std_shading,
                                                  sort_by_activity=sort_by_activity,
                                                  sorting_window=sorting_window)
                    if im is not None:
                        image_objects.append(im)
                    panel_idx += 1

                # Plot Part 2
                if len(unit_ids_part2) > 0:
                    ax = axes[panel_idx]
                    activity_part2 = activity_data[unit_ids_part2, :]
                    std_part2 = std_data[unit_ids_part2, :] if std_data is not None else None
                    activation_part2 = activation_data[unit_ids_part2] if activation_data is not None else None

                    im = self._plot_single_raster(ax, activity_part2, time_axis, pop, 
                                                  global_vmin if use_global_colormap else vmin,
                                                  global_vmax if use_global_colormap else vmax,
                                                  cmap, mean_linewidth,
                                                  title_suffix=f' - {part2_label}',
                                                  direct_activation=activation_part2,
                                                  activation_cmap=activation_cmap,
                                                  activation_bar_width=activation_bar_width,
                                                  normalize_to_baseline=normalize_to_baseline,
                                                  baseline_window=baseline_window,
                                                  activity_std=std_part2,
                                                  show_std_shading=show_std_shading,
                                                  sort_by_activity=sort_by_activity,
                                                  sorting_window=sorting_window)
                    if im is not None:
                        image_objects.append(im)
                    panel_idx += 1

                activity_history[pop] = np.mean(activity_data, axis=0).tolist()
            else:
                # Plot normally (no split)
                ax = axes[panel_idx]
                im = self._plot_single_raster(ax, activity_data, time_axis, pop, 
                                              global_vmin if use_global_colormap else vmin,
                                              global_vmax if use_global_colormap else vmax,
                                              cmap, mean_linewidth,
                                              direct_activation=activation_data,
                                              activation_cmap=activation_cmap,
                                              activation_bar_width=activation_bar_width,
                                              sort_by_activity=sort_by_activity,
                                              normalize_to_baseline=normalize_to_baseline,
                                              baseline_window=baseline_window,
                                              activity_std=std_data,
                                              show_std_shading=show_std_shading,
                                              sorting_window=sorting_window)
                activity_history[pop] = np.mean(activity_data, axis=0).tolist()
                panel_idx += 1
                if im is not None:
                    image_objects.append(im)

        # Remove unused subplots
        for idx in range(panel_idx, len(axes)):
            fig.delaxes(axes[idx])

        # Update title based on normalization
        title = 'DG Circuit Activity'
        if normalize_to_baseline:
            title += ' (Normalized to Baseline)'
        if activity_std is not None:
            title += f' - Mean ± Std Across Trials'
        else:
            title += ': Neurons Sorted by Mean Firing Rate'

        plt.suptitle(title, fontsize=16, fontweight='bold')

        # Store image objects as figure attribute to prevent garbage collection
        fig._activity_images = image_objects

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)

        return fig, activity_history

    def _plot_single_raster(self, ax, activity_data, time_axis, pop, 
                            vmin, vmax, cmap, mean_linewidth, title_suffix='',
                            direct_activation=None, activation_cmap='plasma',
                            activation_bar_width=None, sort_by_activity=False,
                            normalize_to_baseline=False, baseline_window=None,
                            activity_std=None, show_std_shading=True,
                            sorting_window=None):
        """
        Helper method to plot a single raster panel with Rectangle patches for activation
        sorting_window: Optional tuple (start_time, end_time) in ms. If provided,
                       neurons are sorted by mean activity during this window only.
                       If None, sorts by mean activity across entire trace.
        Returns:
            im: The image object from imshow (to prevent garbage collection)
        """
        n_cells = activity_data.shape[0]

        # Calculate mean activity per neuron across time
        #mean_activity_per_neuron = np.mean(activity_data, axis=1)

        
        
        # Calculate mean activity per neuron for sorting
        if sorting_window is not None and sort_by_activity:
            # Sort based on activity during specified window
            sort_start, sort_end = sorting_window

            # Convert to numpy if tensors
            if hasattr(sort_start, 'cpu'):
                sort_start = sort_start.cpu().numpy() if sort_start.dim() == 0 else sort_start.cpu().item()
            if hasattr(sort_end, 'cpu'):
                sort_end = sort_end.cpu().numpy() if sort_end.dim() == 0 else sort_end.cpu().item()

            # Ensure scalar values
            if isinstance(sort_start, np.ndarray):
                sort_start = float(sort_start)
            if isinstance(sort_end, np.ndarray):
                sort_end = float(sort_end)

            sort_mask = (time_axis >= sort_start) & (time_axis < sort_end)
            mean_activity_per_neuron = np.mean(activity_data[:, sort_mask], axis=1)
        else:
            # Sort based on activity across entire trace
            mean_activity_per_neuron = np.mean(activity_data, axis=1)
        
        # Sort neurons by mean firing rate (descending)
        sorted_indices = np.argsort(mean_activity_per_neuron)[::-1]
        if sort_by_activity:
            sorted_activity = activity_data[sorted_indices, :]
            if activity_std is not None:
                sorted_std = activity_std[sorted_indices, :]
            else:
                sorted_std = None
        else:
            sorted_activity = activity_data
            sorted_std = activity_std
            
        # Sort direct activation data if provided
        sorted_activation = None
        if direct_activation is not None and np.sum(direct_activation) > 0.0:
            if sort_by_activity:
                sorted_activation = direct_activation[sorted_indices]
            else:
                sorted_activation = direct_activation

        # Determine colormap range with robust handling
        if normalize_to_baseline:
            if vmin is not None and vmax is not None:
                # Use provided global limits
                vmin_pop = vmin
                vmax_pop = vmax
            else:
                # Calculate per-population limits
                activity_abs_max = np.abs(sorted_activity)
                vmax_pop = np.percentile(activity_abs_max[np.isfinite(activity_abs_max)], 99)
                if vmax_pop < 0.1:
                    vmax_pop = 1.0
                vmin_pop = -vmax_pop
        
        # Calculate activation bar parameters
        time_range = time_axis[-1] - time_axis[0]
        if activation_bar_width is None:
            bar_width = time_range * 0.05
        else:
            bar_width = activation_bar_width

        bar_gap = time_range * 0.01
        bar_x_start = time_axis[0] - bar_width - bar_gap

        # Draw activation rectangles if provided
        if sorted_activation is not None:
            activation_vmin = np.min(sorted_activation)
            activation_vmax = np.max(sorted_activation)

            if activation_vmax > activation_vmin:
                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable
                from matplotlib.patches import Rectangle
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                norm = Normalize(vmin=activation_vmin, vmax=activation_vmax)
                cmap_obj = plt.get_cmap(activation_cmap)

                for i in range(n_cells):
                    activation_value = sorted_activation[i]
                    color = cmap_obj(norm(activation_value))

                    rect = Rectangle(xy=(bar_x_start, i), 
                                     width=bar_width, 
                                     height=1.0,
                                     facecolor=color, 
                                     edgecolor='none',
                                     zorder=10)
                    ax.add_patch(rect)

                # Add colorbar for activation
                sm = ScalarMappable(cmap=cmap_obj, norm=norm)
                sm.set_array([])

                cbar_ax = inset_axes(ax, 
                                     width="2%",  
                                     height="30%",
                                     loc='upper left',
                                     bbox_to_anchor=(-0.25, 0.05, 1, 1),
                                     bbox_transform=ax.transAxes,
                                     borderpad=0)

                cbar_activation = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
                cbar_activation.set_label('Input\nCurrent\n(nA)', 
                                          fontsize=8, 
                                          rotation=0,
                                          ha='center',
                                          va='bottom',
                                          labelpad=10)
                cbar_activation.ax.tick_params(labelsize=7)

        # Plot raster as heatmap
        im = ax.imshow(sorted_activity, 
                       aspect='auto',
                       cmap=cmap,
                       vmin=vmin_pop,
                       vmax=vmax_pop,
                       interpolation='nearest',
                       extent=[time_axis[0], time_axis[-1], n_cells, 0],
                       zorder=1)

        # Store image as axis attribute to prevent garbage collection
        ax._raster_image = im

        # Mark baseline window if provided
        if baseline_window is not None:
            baseline_start, baseline_end = baseline_window
            ax.axvspan(baseline_start, baseline_end, 
                       alpha=0.1, color='green', zorder=0,
                       label='Baseline')

        # Adjust x-limits
        if sorted_activation is not None:
            ax.set_xlim([bar_x_start, time_axis[-1]])
        else:
            ax.set_xlim([time_axis[0], time_axis[-1]])

        # Configure primary axis
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Neuron Index\n(sorted by activity)', fontsize=10)

        # Title
        title_prefix = f'{pop.upper()} Activity Raster'
        if sorted_activation is not None:
            title_prefix += ' [with input current]'
        if normalize_to_baseline:
            title_prefix += '\n(Baseline Normalized)'
        ax.set_title(f'{title_prefix}{title_suffix}\n({n_cells} cells)', fontsize=11)

        # Compute population mean activity
        mean_activity = np.mean(activity_data, axis=0)

        # Compute std across neurons if std across trials was provided
        if sorted_std is not None and show_std_shading:
            mean_std = np.mean(sorted_std, axis=0)

        # Create twin axis for mean firing rate overlay
        ax2 = ax.twinx()

        # Normalize mean activity for overlay
        if normalize_to_baseline:
            mean_normalized = ((mean_activity + vmax_pop) / (2 * vmax_pop)) * n_cells
        else:
            mean_normalized = (mean_activity / vmax_pop) * n_cells

        mean_normalized = np.clip(mean_normalized, 0, n_cells)

        # Plot mean firing rate
        ax2.plot(time_axis, mean_normalized, 
                 color='black',
                 linewidth=mean_linewidth,
                 linestyle='-',
                 alpha=0.8,
                 label=f'Mean={np.mean(mean_activity):.1f}',
                 zorder=2)

        # Add shaded region for trial-to-trial variability if available
        if sorted_std is not None and show_std_shading:
            if normalize_to_baseline:
                std_normalized_upper = ((mean_activity + mean_std + vmax_pop) / (2 * vmax_pop)) * n_cells
                std_normalized_lower = ((mean_activity - mean_std + vmax_pop) / (2 * vmax_pop)) * n_cells
            else:
                std_normalized_upper = ((mean_activity + mean_std) / vmax_pop) * n_cells
                std_normalized_lower = ((mean_activity - mean_std) / vmax_pop) * n_cells

            std_normalized_upper = np.clip(std_normalized_upper, 0, n_cells)
            std_normalized_lower = np.clip(std_normalized_lower, 0, n_cells)

            ax2.fill_between(time_axis,
                            std_normalized_lower,
                            std_normalized_upper,
                            color='black',
                            alpha=0.2,
                            zorder=1,
                            label='± Std (trials)')

        # Configure secondary axis
        ax2.set_ylabel('Normalized Mean Activity', rotation=270, labelpad=20, fontsize=10)
        ax2.set_ylim([0, n_cells])
        ax2.legend(loc='upper right', framealpha=0.7, fontsize=10)
        ax2.set_yticks([])

        # Add grid
        ax.grid(False)

        # Create raster colorbar last and attach to both axes
        cbar_raster = plt.colorbar(im, ax=[ax, ax2], fraction=0.046, pad=0.04)
        if normalize_to_baseline:
            cbar_raster.set_label('Normalized Firing Rate', rotation=270, labelpad=15, fontsize=10)
        else:
            cbar_raster.set_label('Firing Rate (Hz)', rotation=270, labelpad=15, fontsize=10)
        cbar_raster.ax.tick_params(labelsize=10)

        # Store colorbar reference to prevent garbage collection
        ax._activity_colorbar = cbar_raster

        # Return the image object so caller can keep reference
        return im    

    
    def plot_aggregated_activity(self, aggregated_results: Dict,
                                 target_population: str,
                                 opsin_expression_levels: np.ndarray,
                                 light_intensity: float,
                                 stim_start: float = 500.0,
                                 warmup: float = 100.0,
                                 baseline_normalize: bool = False,
                                 sort_by_activity: bool = True,
                                 sort_by_stim_period: bool = True,
                                 save_path: Optional[str] = None):
        """
        Plot trial-averaged activity with standard deviation shading

        Args:
            aggregated_results: Results dictionary from multi-trial simulation containing:
                - 'time': Time vector
                - 'activity_trace_mean': Mean activity across trials
                - 'activity_trace_std': Std of activity across trials
                - 'n_trials': Number of trials averaged
                - 'trial_results': List of individual trial results
            target_population: Population that was stimulated ('pv', 'sst', etc.)
            opsin_expression_levels: Array of opsin expression levels (n_cells,)
            light_intensity: Light intensity used for stimulation
            stim_start: Stimulation start time in ms (for baseline window)
            baseline_normalize: If True, normalize activity relative to pre-stim baseline
            sort_by_activity: Whether to sort neurons by mean activity
            save_path: Path to save figure (optional)

        Returns:
            fig: Matplotlib figure object
            activity_history: Dictionary with mean activity traces
        """
        # Extract aggregated data
        time_cpu = aggregated_results['time']
        activity_mean = aggregated_results['activity_trace_mean']
        activity_std = aggregated_results['activity_trace_std']
        n_trials = aggregated_results['n_trials']

        # Get stimulated vs non-stimulated indices from first trial
        first_trial = aggregated_results['trial_results'][0]
        stimulated_indices = first_trial['stimulated_indices']
        non_stimulated_indices = first_trial['non_stimulated_indices']

        # Setup split populations for visualization
        split_populations = {
            target_population: {
                'unit_ids_part1': stimulated_indices.tolist(),
                'unit_ids_part2': non_stimulated_indices.tolist(),
                'part1_label': f'Stimulated (n={len(stimulated_indices)})',
                'part2_label': f'Non-stimulated (n={len(non_stimulated_indices)})'
            }
        }

        # Calculate direct activation from opsin expression
        # Threshold for showing activation (cells with expression >= threshold)
        activation_threshold = 0.2
        expressing_mask = opsin_expression_levels >= activation_threshold

        # Create activation array (only for expressing cells)
        plot_direct_activation = {}
        if np.any(expressing_mask):
            # Scale by expression level for visualization
            activation_values = opsin_expression_levels * light_intensity
            plot_direct_activation[target_population] = activation_values

        # Calculate baseline window (before stimulation)
        baseline_window = (warmup, stim_start) if baseline_normalize else None

        # Generate save path if not provided
        if save_path is None:
            suffix = f"_aggregated_n{n_trials}"
            if baseline_normalize:
                suffix += "_normalized"
            save_path = f"protocol/DG_{target_population}_stimulation_raster_{light_intensity}{suffix}.png"

        # Calculate sorting window based on stim_start
        sorting_window = (stim_start, time_cpu[-1]) if sort_by_stim_period else None

        # Plot with trial-averaged data
        fig, activity_history = self.plot_activity_raster(
            activity_mean,
            split_populations=split_populations,
            direct_activation=plot_direct_activation if plot_direct_activation else None,
            sort_by_activity=sort_by_activity,
            sorting_window=sorting_window,
            baseline_window=baseline_window,
            normalize_to_baseline=baseline_normalize,
            activity_std=activity_std,
            show_std_shading=True,
            save_path=save_path
        )

        return fig, activity_history

        
    def plot_network_graph(self, connection_types=None, layout_type='spring',
                          node_size_scale=1.0, save_path=None):
        """
        Plot circuit as a network graph
        
        Args:
            connection_types: Connections to include in graph
            layout_type: NetworkX layout algorithm
            node_size_scale: Scaling factor for node sizes
            save_path: Path to save figure
        """
        if connection_types is None:
            connection_types = ['gc_mc', 'mc_gc', 'pv_gc', 'sst_gc']
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes for each cell
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        node_id = 0
        pop_node_ranges = {}
        
        for pop in ['gc', 'mc', 'pv', 'sst']:
            start_id = node_id
            n_cells = min(self.pop_sizes[pop], 50)  # Limit for visualization
            
            for i in range(n_cells):
                G.add_node(node_id, population=pop, cell_id=i)
                node_colors.append(self.config.colors[pop])
                node_sizes.append(self.config.marker_sizes[pop] * node_size_scale)
                node_labels[node_id] = f'{pop}_{i}'
                node_id += 1
            
            pop_node_ranges[pop] = (start_id, node_id)
        
        # Add edges based on connectivity
        for connection_type in connection_types:
            pre_pop, post_pop = connection_type.split('_')
            
            if pre_pop not in pop_node_ranges or post_pop not in pop_node_ranges:
                continue
            
            conn_name = f'conn_{connection_type}'
            if not hasattr(self.circuit, conn_name):
                continue
            
            matrix = getattr(self.circuit, conn_name).cpu().numpy()
            pre_start, pre_end = pop_node_ranges[pre_pop]
            post_start, post_end = pop_node_ranges[post_pop]
            
            # Subsample connections for visualization
            pre_range = min(pre_end - pre_start, matrix.shape[0])
            post_range = min(post_end - post_start, matrix.shape[1])
            
            for i in range(pre_range):
                for j in range(post_range):
                    if matrix[i, j] > 0.1:  # Threshold for visualization
                        G.add_edge(pre_start + i, post_start + j, 
                                 weight=matrix[i, j], connection_type=connection_type)
        
        # Create layout
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.config.figsize_large, dpi=self.config.dpi)
        
        # Draw edges with different colors for different connection types
        edge_colors = {'gc_mc': 'red', 'mc_gc': 'blue', 'pv_gc': 'green', 'sst_gc': 'purple'}
        
        for connection_type in connection_types:
            edges_of_type = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('connection_type') == connection_type]
            if edges_of_type:
                nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                     edge_color=edge_colors.get(connection_type, 'gray'),
                                     alpha=0.6, arrows=True, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=node_sizes, alpha=0.8, ax=ax)
        
        ax.set_title('Dentate Gyrus Circuit Network Graph', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=self.config.colors[pop], 
                                    markersize=10, label=pop.upper()) 
                         for pop in ['gc', 'mc', 'pv', 'sst']]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig, G


    def plot_current_traces(self, recorded_currents: Dict,
                            population: str,
                            stim_start: float,
                            stim_end: float,
                            baseline_start: Optional[float] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time traces of synaptic currents for a population

        Integrates with existing DG visualization style and uses class config.

        Args:
            recorded_currents: Output from SynapticCurrentRecorder.get_results()
            population: Population name ('gc', 'mc', 'pv', 'sst')
            stim_start: Stimulation start time (ms)
            stim_end: Stimulation end time (ms)
            baseline_start: Optional baseline period start (ms)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        time = recorded_currents['time']
        if hasattr(time, 'cpu'):
            time = time.cpu().numpy()
        else:
            time = np.array(time)

        fig = plt.figure(figsize=self.config.figsize_large, dpi=self.config.dpi)
        gs = gridspec.GridSpec(3, 1, hspace=0.3)

        # Panel 1: Currents by receptor type
        ax1 = fig.add_subplot(gs[0])

        currents_by_type = recorded_currents['by_type'][population]

        # Define colors for receptor types
        receptor_colors = {'ampa': '#FF6B6B', 'gaba': '#4ECDC4', 'nmda': '#FFA07A'}

        # Plot mean across cells
        for current_type, color in receptor_colors.items():
            current = currents_by_type[current_type]  # [n_cells, n_time]
            if hasattr(current, 'cpu'):
                current = current.cpu().numpy()
            else:
                current = np.array(current)

            mean_current = np.mean(current, axis=0)
            std_current = np.std(current, axis=0)

            ax1.plot(time, mean_current, color=color, 
                    label=current_type.upper(), linewidth=2)
            ax1.fill_between(time, mean_current - std_current, 
                            mean_current + std_current,
                            color=color, alpha=0.2)

        # Mark stimulation period
        ax1.axvspan(stim_start, stim_end, alpha=0.1, color='orange', 
                    label='Stimulation', zorder=0)
        if baseline_start is not None:
            ax1.axvspan(baseline_start, stim_start, alpha=0.1, color='green', 
                        label='Baseline', zorder=0)
            
        ax1.set_xlabel('Time (ms)', fontsize=11)
        ax1.set_ylabel('Current (pA)', fontsize=11)
        ax1.set_title(f'{population.upper()}: Currents by Receptor Type', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_axisbelow(True)

        # Panel 2: Total excitatory vs inhibitory
        ax2 = fig.add_subplot(gs[1])

        total_exc = currents_by_type['total_exc']
        total_inh = currents_by_type['total_inh']

        if hasattr(total_exc, 'cpu'):
            total_exc = total_exc.cpu().numpy()
            total_inh = total_inh.cpu().numpy()
        else:
            total_exc = np.array(total_exc)
            total_inh = np.array(total_inh)

        mean_exc = np.mean(total_exc, axis=0)
        std_exc = np.std(total_exc, axis=0)
        mean_inh = np.mean(total_inh, axis=0)
        std_inh = np.std(total_inh, axis=0)

        ax2.plot(time, mean_exc, color='#E74C3C', label='Total Excitatory', linewidth=2)
        ax2.fill_between(time, mean_exc - std_exc, mean_exc + std_exc, 
                        color='#E74C3C', alpha=0.2)

        ax2.plot(time, mean_inh, color='#3498DB', label='Total Inhibitory', linewidth=2)
        ax2.fill_between(time, mean_inh - std_inh, mean_inh + std_inh,
                        color='#3498DB', alpha=0.2)

        # Add zero line
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Mark periods
        ax2.axvspan(stim_start, stim_end, alpha=0.1, color='orange', zorder=0)
        if baseline_start is not None:
            ax2.axvspan(baseline_start, stim_start, alpha=0.1, color='green', zorder=0)

        ax2.set_xlabel('Time (ms)', fontsize=11)
        ax2.set_ylabel('Current (pA)', fontsize=11)
        ax2.set_title(f'{population.upper()}: Total Excitatory vs Inhibitory', 
                     fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)

        # Panel 3: Net current and E/I ratio
        ax3 = fig.add_subplot(gs[2])

        net_current = currents_by_type['net']
        if hasattr(net_current, 'cpu'):
            net_current = net_current.cpu().numpy()
        else:
            net_current = np.array(net_current)

        mean_net = np.mean(net_current, axis=0)
        std_net = np.std(net_current, axis=0)

        # Plot net current
        ax3_twin = ax3.twinx()

        ax3.plot(time, mean_net, color='#9B59B6', label='Net Current', linewidth=2)
        ax3.fill_between(time, mean_net - std_net, mean_net + std_net,
                        color='#9B59B6', alpha=0.2)
        ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Calculate and plot E/I ratio
        ei_ratio = np.abs(mean_exc) / (np.abs(mean_inh) + 1e-6)
        ax3_twin.plot(time, ei_ratio, color='#27AE60', label='|E|/|I| Ratio', 
                     linewidth=2, linestyle=':')
        ax3_twin.axhline(1, color='#27AE60', linestyle='--', linewidth=1, alpha=0.5)

        # Mark periods
        ax3.axvspan(stim_start, stim_end, alpha=0.1, color='orange', zorder=0)
        if baseline_start is not None:
            ax3.axvspan(baseline_start, stim_start, alpha=0.1, color='green', zorder=0)

        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_ylabel('Net Current (pA)', fontsize=11, color='#9B59B6')
        ax3_twin.set_ylabel('E/I Ratio', fontsize=11, color='#27AE60')
        ax3.set_title(f'{population.upper()}: Net Current and E/I Balance', 
                     fontsize=12, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='#9B59B6')
        ax3_twin.tick_params(axis='y', labelcolor='#27AE60')
        ax3.grid(True, alpha=0.3)
        ax3.set_axisbelow(True)

        # Combined legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')

        plt.suptitle(f'Synaptic Currents: {population.upper()}', 
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved current traces to {save_path}")

        return fig


    def plot_current_sources(self, recorded_currents: Dict,
                            population: str,
                            stim_start: float,
                            stim_end: float,
                            baseline_start: Optional[float] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot contributions from different presynaptic sources

        Args:
            recorded_currents: Output from SynapticCurrentRecorder.get_results()
            population: Target population name
            stim_start: Stimulation start time (ms)
            stim_end: Stimulation end time (ms)
            baseline_start: Optional baseline period start (ms)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        time = recorded_currents['time']
        if hasattr(time, 'cpu'):
            time = time.cpu().numpy()
        else:
            time = np.array(time)

        currents_by_source = recorded_currents['by_source'][population]

        # Get list of sources
        sources = list(currents_by_source.keys())
        n_sources = len(sources)

        if n_sources == 0:
            print(f"No current sources found for {population}")
            return None

        fig, axes = plt.subplots(n_sources, 1, 
                                figsize=(self.config.figsize_large[0], 4*n_sources), 
                                sharex=True, squeeze=False, dpi=self.config.dpi)
        axes = axes.flatten()

        # Define colors for receptor types
        receptor_colors = {'ampa': '#FF6B6B', 'gaba': '#4ECDC4', 'nmda': '#FFA07A'}

        for idx, source in enumerate(sources):
            ax = axes[idx]
            source_currents = currents_by_source[source]

            # Plot AMPA (red), GABA (cyan), NMDA (coral)
            for receptor_type, color in receptor_colors.items():
                if receptor_type in source_currents:
                    current = source_currents[receptor_type]  # [n_cells, n_time]
                    if hasattr(current, 'cpu'):
                        current = current.cpu().numpy()
                    else:
                        current = np.array(current)

                    mean_current = np.mean(current, axis=0)
                    std_current = np.std(current, axis=0)

                    ax.plot(time, mean_current, color=color, 
                           label=f'{receptor_type.upper()}', linewidth=2)
                    ax.fill_between(time, mean_current - std_current, 
                                   mean_current + std_current,
                                   color=color, alpha=0.2)

            # Mark periods
            ax.axvspan(stim_start, stim_end, alpha=0.1, color='orange', 
                      label='Stimulation' if idx == 0 else '', zorder=0)
            if baseline_start is not None:
                ax.axvspan(baseline_start, stim_start, alpha=0.1, color='green',
                          label='Baseline' if idx == 0 else '', zorder=0)

            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylabel('Current (pA)', fontsize=11)

            # Use class config colors for source if available
            source_color = self.config.colors.get(source, 'black')
            ax.set_title(f'{source.upper()} → {population.upper()}', 
                        fontsize=11, fontweight='bold', color=source_color)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

        axes[-1].set_xlabel('Time (ms)', fontsize=11)

        plt.suptitle(f'Synaptic Currents by Source: {population.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved current sources plot to {save_path}")

        return fig


    def plot_current_comparison_bar(self, current_analysis: Dict,
                                    target_population: str,
                                    populations: Optional[List[str]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Bar plot comparing baseline vs stimulation currents

        Args:
            current_analysis: Output from analyze_currents_by_period()
            target_population: Population that was optogenetically stimulated
            populations: List of populations to plot (default: ['gc', 'mc', 'pv', 'sst'])
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        if populations is None:
            populations = ['gc', 'mc', 'pv', 'sst']

        # Filter to only populations present in analysis
        populations = [p for p in populations if p in current_analysis['baseline']]

        if len(populations) == 0:
            print("No populations found in current analysis")
            return None

        fig, axes = plt.subplots(2, len(populations), 
                                figsize=(4*len(populations), 8),
                                dpi=self.config.dpi)

        # Handle single population case
        if len(populations) == 1:
            axes = axes.reshape(-1, 1)

        for pop_idx, pop in enumerate(populations):
            if pop not in current_analysis['baseline']:
                continue

            # Top row: Total excitatory and inhibitory
            ax_top = axes[0, pop_idx]

            baseline = current_analysis['baseline'][pop]['by_type']
            stim = current_analysis['stimulation'][pop]['by_type']

            categories = ['Excitatory', 'Inhibitory']
            baseline_vals = [baseline['total_exc']['mean'], baseline['total_inh']['mean']]
            stim_vals = [stim['total_exc']['mean'], stim['total_inh']['mean']]
            baseline_errs = [baseline['total_exc']['sem'], baseline['total_inh']['sem']]
            stim_errs = [stim['total_exc']['sem'], stim['total_inh']['sem']]

            x = np.arange(len(categories))
            width = 0.35

            ax_top.bar(x - width/2, baseline_vals, width, label='Baseline',
                      color='#27AE60', alpha=0.7, yerr=baseline_errs, capsize=5)
            ax_top.bar(x + width/2, stim_vals, width, label='Stimulation',
                      color='#E67E22', alpha=0.7, yerr=stim_errs, capsize=5)

            ax_top.set_xticks(x)
            ax_top.set_xticklabels(categories)
            ax_top.set_ylabel('Current (pA)', fontsize=10)

            # Use class config color for population
            pop_color = self.config.colors.get(pop, 'black')
            ax_top.set_title(f'{pop.upper()}: Exc vs Inh', 
                            fontsize=11, fontweight='bold', color=pop_color)
            if pop_idx == 0:
                ax_top.legend(fontsize=10)
            ax_top.grid(True, alpha=0.3, axis='y')
            ax_top.set_axisbelow(True)
            ax_top.axhline(0, color='black', linestyle='-', linewidth=1)

            # Bottom row: By receptor type
            ax_bot = axes[1, pop_idx]

            receptor_types = ['AMPA', 'GABA', 'NMDA']
            baseline_vals = [baseline['ampa']['mean'], baseline['gaba']['mean'], 
                            baseline['nmda']['mean']]
            stim_vals = [stim['ampa']['mean'], stim['gaba']['mean'], 
                        stim['nmda']['mean']]
            baseline_errs = [baseline['ampa']['sem'], baseline['gaba']['sem'],
                            baseline['nmda']['sem']]
            stim_errs = [stim['ampa']['sem'], stim['gaba']['sem'],
                        stim['nmda']['sem']]

            x = np.arange(len(receptor_types))

            ax_bot.bar(x - width/2, baseline_vals, width, label='Baseline',
                      color='#27AE60', alpha=0.7, yerr=baseline_errs, capsize=5)
            ax_bot.bar(x + width/2, stim_vals, width, label='Stimulation',
                      color='#E67E22', alpha=0.7, yerr=stim_errs, capsize=5)

            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels(receptor_types)
            ax_bot.set_ylabel('Current (pA)', fontsize=10)
            ax_bot.set_title(f'{pop.upper()}: By Receptor', 
                            fontsize=11, fontweight='bold', color=pop_color)
            ax_bot.grid(True, alpha=0.3, axis='y')
            ax_bot.set_axisbelow(True)
            ax_bot.axhline(0, color='black', linestyle='-', linewidth=1)

        plt.suptitle(f'Synaptic Currents: {target_population.upper()} Stimulation',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved current comparison to {save_path}")

        return fig


    def plot_current_heatmap(self, recorded_currents: Dict,
                            population: str,
                            current_type: str = 'net',
                            stim_start: Optional[float] = None,
                            stim_end: Optional[float] = None,
                            baseline_start: Optional[float] = None,
                            sort_by_mean: bool = True,
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of currents across cells and time

        Similar to activity raster but for currents.

        Args:
            recorded_currents: Output from SynapticCurrentRecorder.get_results()
            population: Population name
            current_type: Type of current to plot ('ampa', 'gaba', 'nmda', 
                         'total_exc', 'total_inh', 'net')
            stim_start: Stimulation start time (ms)
            stim_end: Stimulation end time (ms)
            baseline_start: Baseline period start (ms)
            sort_by_mean: Sort cells by mean current
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        time = recorded_currents['time']
        if hasattr(time, 'cpu'):
            time = time.cpu().numpy()
        else:
            time = np.array(time)

        currents_by_type = recorded_currents['by_type'][population]

        if current_type not in currents_by_type:
            print(f"Current type {current_type} not found for {population}")
            return None

        current = currents_by_type[current_type]
        if hasattr(current, 'cpu'):
            current = current.cpu().numpy()
        else:
            current = np.array(current)

        n_cells = current.shape[0]

        # Sort by mean current if requested
        if sort_by_mean:
            mean_current_per_cell = np.mean(current, axis=1)
            sorted_indices = np.argsort(mean_current_per_cell)[::-1]
            current = current[sorted_indices, :]

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize_large, dpi=self.config.dpi)

        # Determine colormap range
        if vmin is None:
            vmin = np.percentile(current, 5)
        if vmax is None:
            vmax = np.percentile(current, 95)

        # Plot heatmap
        im = ax.imshow(current, aspect='auto', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, interpolation='nearest',
                       extent=[time[0], time[-1], n_cells, 0])

        # Mark periods
        if stim_start is not None and stim_end is not None:
            ax.axvspan(stim_start, stim_end, alpha=0.1, color='orange',
                      label='Stimulation', zorder=0)
        if baseline_start is not None and stim_start is not None:
            ax.axvspan(baseline_start, stim_start, alpha=0.1, color='green',
                      label='Baseline', zorder=0)

        # Plot mean current overlay
        ax2 = ax.twinx()
        mean_current = np.mean(current, axis=0)
        # Normalize to cell index range for overlay
        mean_normalized = ((mean_current - vmin) / (vmax - vmin)) * n_cells
        mean_normalized = np.clip(mean_normalized, 0, n_cells)

        ax2.plot(time, mean_normalized, color='black', linewidth=2,
                label=f'Mean={np.mean(mean_current):.1f} pA')
        ax2.set_ylabel('Normalized Mean Current', rotation=270, labelpad=20)
        ax2.set_ylim([0, n_cells])
        ax2.legend(loc='upper right', framealpha=0.7)

        # Labels and title
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Cell Index', fontsize=11)

        # Use class config color for title
        pop_color = self.config.colors.get(population, 'black')
        title = f'{population.upper()}: {current_type.replace("_", " ").title()} Current'
        if sort_by_mean:
            title += '\n(Sorted by Mean Current)'
        ax.set_title(title, fontsize=12, fontweight='bold', color=pop_color)

        # Colorbar
        cbar = plt.colorbar(im, ax=[ax, ax2], fraction=0.046, pad=0.04)
        cbar.set_label('Current (pA)', rotation=270, labelpad=15)

        ax.grid(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved current heatmap to {save_path}")

        return fig

    def compute_input_weights_by_source(self, 
                                        target_population: str,
                                        opsin_expression: Optional[Dict[str, np.ndarray]] = None
                                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute total synaptic weights from each source population to target cells

        For each cell in the target population, sums all synaptic conductances from
        each presynaptic population. Optionally separates contributions from
        opsin-expressing vs non-expressing cells.

        Args:
            target_population: Target population name ('gc', 'mc', 'pv', 'sst')
            opsin_expression: Optional dict mapping source population names to boolean
                              arrays indicating opsin expression (True = expressing)
                              Shape: {source_pop: [n_source_cells]}

        Returns:
            Dictionary with structure:
            {
                'source_pop': {
                    'total_weights': np.ndarray [n_target_cells],
                    'mean_weight': float,
                    'std_weight': float,
                    'n_connections': int,
                    'opsin_expressing': np.ndarray [n_target_cells] (optional),
                    'non_expressing': np.ndarray [n_target_cells] (optional),
                    'mean_opsin_expressing': float (optional),
                    'mean_non_expressing': float (optional)
                },
                ...
            }
        """
        n_target = self.pop_sizes[target_population]
        weight_data = {}

        # Find all connections targeting this population
        for conn_name, cond_matrix in self.connectivity.conductance_matrices.items():
            parts = conn_name.split('_')
            if len(parts) >= 2 and parts[1] == target_population:
                source_pop = parts[0]

                # Get conductance matrix [n_source, n_target]
                if hasattr(cond_matrix.conductances, 'cpu'):
                    conductances = cond_matrix.conductances.cpu().numpy()
                    connectivity = cond_matrix.connectivity.cpu().numpy()
                else:
                    conductances = np.array(cond_matrix.conductances)
                    connectivity = np.array(cond_matrix.connectivity)

                # Sum conductances from all source cells to each target cell
                total_weights = np.sum(conductances, axis=0)  # [n_target]

                # Count connections
                n_connections = np.sum(connectivity > 0)

                source_data = {
                    'total_weights': total_weights,
                    'mean_weight': float(np.mean(total_weights)),
                    'std_weight': float(np.std(total_weights)),
                    'n_connections': int(n_connections),
                    'synapse_type': cond_matrix.synapse_type
                }

                # Separate by opsin expression if provided
                if opsin_expression is not None and source_pop in opsin_expression:
                    opsin_mask = opsin_expression[source_pop]

                    if hasattr(opsin_mask, 'cpu'):
                        opsin_mask = opsin_mask.cpu().numpy()
                    else:
                        opsin_mask = np.array(opsin_mask).astype(bool)

                    # Ensure mask is boolean
                    opsin_mask = opsin_mask.astype(bool)

                    # Sum weights from opsin-expressing sources
                    opsin_weights = np.sum(conductances[opsin_mask, :], axis=0)

                    # Sum weights from non-expressing sources
                    non_opsin_weights = np.sum(conductances[~opsin_mask, :], axis=0)

                    source_data['opsin_expressing'] = opsin_weights
                    source_data['non_expressing'] = non_opsin_weights
                    source_data['mean_opsin_expressing'] = float(np.mean(opsin_weights))
                    source_data['mean_non_expressing'] = float(np.mean(non_opsin_weights))
                    source_data['n_opsin_expressing'] = int(np.sum(opsin_mask))
                    source_data['n_non_expressing'] = int(np.sum(~opsin_mask))

                weight_data[source_pop] = source_data

        return weight_data


    def plot_input_weight_distribution(self,
                                       post_population: str,
                                       opsin_expression: Optional[Dict[str, np.ndarray]] = None,
                                       sources_to_plot: Optional[List[str]] = None,
                                       plot_type: str = 'both',
                                       stimulated_population: Optional[str] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of synaptic input weights from different source populations

        Creates visualizations showing how synaptic weights are distributed across
        cells in the target population, with separate plots for each source.

        Args:
            post_population: Post-synaptic population name ('gc', 'mc', 'pv', 'sst')
            opsin_expression: Optional dict with opsin expression masks
            sources_to_plot: List of source populations to include (None = all)
            plot_type: 'histogram', 'violin', or 'both'
            stimulated_population: Optional name of optogenetically stimulated population.
                                   If provided and matches post_population, only plots
                                   weights to opsin-non-expressing cells.
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        # Compute weight data
        weight_data = self.compute_input_weights_by_source(
            post_population, opsin_expression
        )

        # Filter to non-expressing target cells if this is the stimulated population
        # Filter to non-expressing target cells if this is the stimulated population
        filter_to_non_expressing = (
            stimulated_population is not None and
            stimulated_population == post_population and
            opsin_expression is not None and
            post_population in opsin_expression
        )

        if filter_to_non_expressing:
            # Get mask for non-expressing cells in target population
            opsin_mask = opsin_expression[post_population]
            if hasattr(opsin_mask, 'cpu'):
                opsin_mask = opsin_mask.cpu().numpy()
            else:
                opsin_mask = np.array(opsin_mask).astype(bool)

            non_expressing_mask = ~opsin_mask
            non_expressing_indices = np.where(non_expressing_mask)[0]

            # Filter weight data to only non-expressing target cells
            for source in list(weight_data.keys()):
                data = weight_data[source]

                # Filter total weights to non-expressing cells
                data['total_weights'] = data['total_weights'][non_expressing_indices]

                # Recalculate statistics for filtered cells
                data['mean_weight'] = float(np.mean(data['total_weights']))
                data['std_weight'] = float(np.std(data['total_weights']))

                # Filter both opsin+ and opsin- components to non-expressing targets
                if 'opsin_expressing' in data:
                    # Filter opsin-expressing presynaptic weights to non-expressing targets
                    data['opsin_expressing'] = data['opsin_expressing'][non_expressing_indices]
                    data['mean_opsin_expressing'] = float(np.mean(data['opsin_expressing']))

                    # Filter non-expressing presynaptic weights to non-expressing targets
                    if 'non_expressing' in data:
                        data['non_expressing'] = data['non_expressing'][non_expressing_indices]
                        data['mean_non_expressing'] = float(np.mean(data['non_expressing']))

                weight_data[source] = data
        
        # Filter populations
        if sources_to_plot is not None:
            weight_data = {k: v for k, v in weight_data.items() 
                           if k in sources_to_plot}

        if len(weight_data) == 0:
            print(f"No input weights found for {post_population}")
            return None

        sources = list(weight_data.keys())
        n_sources = len(sources)

        # Determine subplot layout
        if plot_type == 'both':
            fig = plt.figure(figsize=(self.config.figsize_large[0], 4*n_sources),
                            dpi=self.config.dpi)
            gs = gridspec.GridSpec(n_sources, 2, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(self.config.figsize_large[0], 3*n_sources),
                            dpi=self.config.dpi)
            gs = gridspec.GridSpec(n_sources, 1, hspace=0.3)

        for idx, source in enumerate(sources):
            data = weight_data[source]
            source_color = self.config.colors.get(source, '#7F8C8D')

            if plot_type in ['histogram', 'both']:
                if plot_type == 'both':
                    ax_hist = fig.add_subplot(gs[idx, 0])
                else:
                    ax_hist = fig.add_subplot(gs[idx, 0])

                # Plot histogram
                weights = data['total_weights']
                ax_hist.hist(weights, bins=50, color=source_color, alpha=0.7,
                            edgecolor='black', linewidth=0.5)

                # Add statistics
                ax_hist.axvline(data['mean_weight'], color='red', linestyle='--',
                              linewidth=2, label=f'Mean={data["mean_weight"]:.2f}')
                ax_hist.axvline(data['mean_weight'] + data['std_weight'],
                              color='orange', linestyle=':', linewidth=1.5,
                              label=f'Std={data["std_weight"]:.2f}')
                ax_hist.axvline(data['mean_weight'] - data['std_weight'],
                              color='orange', linestyle=':', linewidth=1.5)

                ax_hist.set_xlabel('Total Input Weight (nS)', fontsize=10)
                ax_hist.set_ylabel('Number of Cells', fontsize=10)
                ax_hist.set_title(f'{source.upper()} $\\rightarrow$ {post_population.upper()}\n'
                                f'({data["synapse_type"]}, {data["n_connections"]} connections)',
                                fontsize=11, fontweight='bold', color=source_color)
                ax_hist.legend(fontsize=9, loc='upper right')
                ax_hist.grid(True, alpha=0.3, axis='y')
                ax_hist.set_axisbelow(True)

            if plot_type in ['violin', 'both']:
                if plot_type == 'both':
                    ax_violin = fig.add_subplot(gs[idx, 1])
                else:
                    ax_violin = fig.add_subplot(gs[idx, 0])

                # Prepare data for violin plot
                if opsin_expression is not None and source in opsin_expression:
                    # Two groups: opsin-expressing and non-expressing
                    plot_data = []
                    labels = []

                    if 'opsin_expressing' in data:
                        plot_data.append(data['opsin_expressing'])
                        labels.append(f'Opsin+ (n={data["n_opsin_expressing"]})')

                    if 'non_expressing' in data:
                        plot_data.append(data['non_expressing'])
                        labels.append(f'Opsin- (n={data["n_non_expressing"]})')

                    # Create violin plot
                    parts = ax_violin.violinplot(plot_data, positions=range(len(plot_data)),
                                                showmeans=True, showmedians=True)

                    # Color the violins
                    colors = ['#E74C3C', '#3498DB']
                    for i, pc in enumerate(parts['bodies']):
                        pc.set_facecolor(colors[i % len(colors)])
                        pc.set_alpha(0.7)

                    ax_violin.set_xticks(range(len(labels)))
                    ax_violin.set_xticklabels(labels, rotation=15, ha='right')
                    ax_violin.set_ylabel('Input Weight (nS)', fontsize=10)
                    ax_violin.set_title(f'{source.upper()} $\\rightarrow$ {post_population.upper()}\n'
                                      'Opsin-Expressing vs Non-Expressing',
                                      fontsize=11, fontweight='bold', color=source_color)
                    ax_violin.grid(True, alpha=0.3, axis='y')
                    ax_violin.set_axisbelow(True)

                    # Add statistics annotations
                    if len(plot_data) == 2:
                        mean_diff = data['mean_opsin_expressing'] - data['mean_non_expressing']
                        ax_violin.text(0.98, 0.98,
                                     f'$\\Delta$ Mean = {mean_diff:.2f} nS',
                                     transform=ax_violin.transAxes,
                                     ha='right', va='top',
                                     bbox=dict(boxstyle='round', facecolor='white',
                                             alpha=0.8, edgecolor='gray'))
                else:
                    # Single violin for total weights
                    ax_violin.violinplot([data['total_weights']], positions=[0],
                                       showmeans=True, showmedians=True)
                    ax_violin.set_xticks([0])
                    ax_violin.set_xticklabels(['All Sources'])
                    ax_violin.set_ylabel('Input Weight (nS)', fontsize=10)
                    ax_violin.set_title(f'{source.upper()} $\\rightarrow$ {post_population.upper()}',
                                      fontsize=11, fontweight='bold', color=source_color)
                    ax_violin.grid(True, alpha=0.3, axis='y')
                    ax_violin.set_axisbelow(True)

        # Overall title
        title = f'Synaptic Input Weights: {post_population.upper()}'
        if filter_to_non_expressing:
            title += f' (Opsin-Non-Expressing Cells Only, n={len(non_expressing_indices)})'
        elif opsin_expression is not None:
            title += ' (Separated by Opsin Expression)'
        plt.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved input weight distribution to {save_path}")

        return fig


    def plot_weight_heatmap_by_cell(self,
                                    post_population: str,
                                    opsin_expression: Optional[Dict[str, np.ndarray]] = None,
                                    sort_by: str = 'total',
                                    normalize: bool = False,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap showing input weights from each source to each target cell

        Each row is a target cell, each column group represents a source population.
        Cells can be sorted by total input, specific source, or left unsorted.

        Args:
            post_population: Post-synaptic population name
            opsin_expression: Optional dict with opsin expression masks
            sort_by: How to sort cells ('total', 'source_name', or 'none')
            normalize: Whether to normalize weights to [0, 1] range per source
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        weight_data = self.compute_input_weights_by_source(
            post_population, opsin_expression
        )

        if len(weight_data) == 0:
            print(f"No input weights found for {post_population}")
            return None

        sources = list(weight_data.keys())
        n_target = self.pop_sizes[post_population]

        # Build weight matrix [n_target, n_sources (or more if split by opsin)]
        weight_matrix = []
        column_labels = []
        column_colors = []

        for source in sources:
            data = weight_data[source]
            source_color = self.config.colors.get(source, '#7F8C8D')

            if opsin_expression is not None and source in opsin_expression:
                # Split into opsin+ and opsin-
                if 'opsin_expressing' in data:
                    weight_matrix.append(data['opsin_expressing'])
                    column_labels.append(f'{source.upper()}\nOpsin+')
                    column_colors.append(source_color)

                if 'non_expressing' in data:
                    weight_matrix.append(data['non_expressing'])
                    column_labels.append(f'{source.upper()}\nOpsin-')
                    # Lighter shade for non-expressing
                    column_colors.append(source_color)
            else:
                weight_matrix.append(data['total_weights'])
                column_labels.append(source.upper())
                column_colors.append(source_color)

        # Convert to array [n_target, n_columns]
        weight_matrix = np.column_stack(weight_matrix)

        # Sort cells if requested
        if sort_by == 'total':
            total_input = np.sum(weight_matrix, axis=1)
            sort_indices = np.argsort(total_input)[::-1]
            weight_matrix = weight_matrix[sort_indices, :]
        elif sort_by in sources:
            source_idx = sources.index(sort_by)
            if opsin_expression is not None and sort_by in opsin_expression:
                # Sort by total from this source (opsin+ and opsin- combined)
                source_total = weight_matrix[:, source_idx] + weight_matrix[:, source_idx + 1]
                sort_indices = np.argsort(source_total)[::-1]
            else:
                sort_indices = np.argsort(weight_matrix[:, source_idx])[::-1]
            weight_matrix = weight_matrix[sort_indices, :]

        # Normalize if requested
        if normalize:
            weight_matrix = weight_matrix / (np.max(weight_matrix, axis=0, keepdims=True) + 1e-8)

        # Create figure
        fig = plt.figure(figsize=(max(len(column_labels) * 1.5, 10), 10),
                        dpi=self.config.dpi)
        ax = fig.add_subplot(111)

        # Plot heatmap
        im = ax.imshow(weight_matrix, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest')

        # Configure axes
        ax.set_xticks(range(len(column_labels)))
        ax.set_xticklabels(column_labels, rotation=45, ha='right', fontsize=10)

        # Color x-axis labels by source
        for i, (label, color) in enumerate(zip(ax.get_xticklabels(), column_colors)):
            label.set_color(color)
            label.set_fontweight('bold')

        ax.set_ylabel(f'{post_population.upper()} Cell Index', fontsize=11)
        ax.set_xlabel('Source Population', fontsize=11)

        # Title
        title = f'Synaptic Input Weights: {post_population.upper()}'
        if normalize:
            title += ' (Normalized)'
        if sort_by == 'total':
            title += '\n(Sorted by Total Input)'
        elif sort_by in sources:
            title += f'\n(Sorted by {sort_by.upper()} Input)'
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if normalize:
            cbar.set_label('Normalized Weight', rotation=270, labelpad=15, fontsize=10)
        else:
            cbar.set_label('Weight (nS)', rotation=270, labelpad=15, fontsize=10)

        # Add vertical lines to separate source groups
        for i in range(1, len(column_labels)):
            ax.axvline(i - 0.5, color='white', linewidth=2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved weight heatmap to {save_path}")

        return fig


    def plot_weight_summary_bars(self,
                                 post_population: str,
                                 opsin_expression: Optional[Dict[str, np.ndarray]] = None,
                                 show_error_bars: bool = True,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Bar plot summarizing mean input weights from each source

        Shows mean +/- SEM, with separate bars for opsin-expressing vs non-expressing
        if opsin_expression is provided.

        Args:
            post_population: Post-synaptic population name
            opsin_expression: Optional dict with opsin expression masks
            show_error_bars: Whether to show error bars (SEM)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        weight_data = self.compute_input_weights_by_source(
            post_population, opsin_expression
        )

        if len(weight_data) == 0:
            print(f"No input weights found for {post_population}")
            return None

        sources = list(weight_data.keys())

        # Determine if we're splitting by opsin
        has_opsin_split = any('opsin_expressing' in weight_data[s] 
                             for s in sources if s in (opsin_expression or {}))

        if has_opsin_split:
            # Create grouped bar plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize_large,
                                           dpi=self.config.dpi)

            # Left panel: Total weights
            means_total = [weight_data[s]['mean_weight'] for s in sources]
            stds_total = [weight_data[s]['std_weight'] for s in sources]
            colors_total = [self.config.colors.get(s, '#7F8C8D') for s in sources]

            x_pos = np.arange(len(sources))
            bars1 = ax1.bar(x_pos, means_total, yerr=stds_total if show_error_bars else None,
                           color=colors_total, alpha=0.7, capsize=5, edgecolor='black',
                           linewidth=1.5)

            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([s.upper() for s in sources], rotation=45, ha='right')
            ax1.set_ylabel('Mean Input Weight (nS)', fontsize=11)
            ax1.set_title('Total Input Weights', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_axisbelow(True)

            # Right panel: Opsin+ vs Opsin- comparison
            # Only include sources that have opsin split
            sources_with_opsin = [s for s in sources 
                                 if 'opsin_expressing' in weight_data[s]]

            if len(sources_with_opsin) > 0:
                x_pos2 = np.arange(len(sources_with_opsin))
                width = 0.35

                means_opsin = [weight_data[s]['mean_opsin_expressing'] 
                              for s in sources_with_opsin]
                means_non = [weight_data[s]['mean_non_expressing'] 
                            for s in sources_with_opsin]

                bars2a = ax2.bar(x_pos2 - width/2, means_opsin, width,
                               label='Opsin+', color='#E74C3C', alpha=0.7,
                               edgecolor='black', linewidth=1.5)
                bars2b = ax2.bar(x_pos2 + width/2, means_non, width,
                               label='Opsin-', color='#3498DB', alpha=0.7,
                               edgecolor='black', linewidth=1.5)

                ax2.set_xticks(x_pos2)
                ax2.set_xticklabels([s.upper() for s in sources_with_opsin],
                                   rotation=45, ha='right')
                ax2.set_ylabel('Mean Input Weight (nS)', fontsize=11)
                ax2.set_title('Opsin-Expressing vs Non-Expressing', 
                             fontsize=12, fontweight='bold')
                ax2.legend(fontsize=10, loc='upper right')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.set_axisbelow(True)

            plt.suptitle(f'Input Weight Summary: {post_population.upper()}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        else:
            # Simple bar plot without opsin split
            fig, ax = plt.subplots(figsize=self.config.figsize_medium,
                                  dpi=self.config.dpi)

            means = [weight_data[s]['mean_weight'] for s in sources]
            stds = [weight_data[s]['std_weight'] for s in sources]
            colors = [self.config.colors.get(s, '#7F8C8D') for s in sources]

            x_pos = np.arange(len(sources))
            bars = ax.bar(x_pos, means, yerr=stds if show_error_bars else None,
                         color=colors, alpha=0.7, capsize=5, edgecolor='black',
                         linewidth=1.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([s.upper() for s in sources], rotation=45, ha='right')
            ax.set_ylabel('Mean Input Weight (nS)', fontsize=11)
            ax.set_title(f'Input Weight Summary: {post_population.upper()}',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)

            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved weight summary to {save_path}")

        return fig


    def plot_weight_correlation_matrix(self,
                                       post_population: str,
                                       opsin_expression: Optional[Dict[str, np.ndarray]] = None,
                                       method: str = 'pearson',
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix showing how inputs from different sources co-vary

        Useful for understanding whether cells receiving strong input from one source
        also receive strong input from other sources.

        Args:
            post_population: Post-synaptic population name
            opsin_expression: Optional dict with opsin expression masks
            method: Correlation method ('pearson' or 'spearman')
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        weight_data = self.compute_input_weights_by_source(
            post_population, opsin_expression
        )

        if len(weight_data) == 0:
            print(f"No input weights found for {post_population}")
            return None

        sources = list(weight_data.keys())

        # Build weight matrix
        weight_matrix = []
        column_labels = []

        for source in sources:
            data = weight_data[source]

            if opsin_expression is not None and source in opsin_expression:
                if 'opsin_expressing' in data:
                    weight_matrix.append(data['opsin_expressing'])
                    column_labels.append(f'{source.upper()}\nOpsin+')

                if 'non_expressing' in data:
                    weight_matrix.append(data['non_expressing'])
                    column_labels.append(f'{source.upper()}\nOpsin-')
            else:
                weight_matrix.append(data['total_weights'])
                column_labels.append(source.upper())

        weight_matrix = np.column_stack(weight_matrix)

        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = np.corrcoef(weight_matrix.T)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(weight_matrix, axis=0)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Create figure
        fig, ax = plt.subplots(figsize=(max(len(column_labels) * 0.8, 8),
                                       max(len(column_labels) * 0.8, 8)),
                              dpi=self.config.dpi)

        # Plot correlation matrix
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                       aspect='auto', interpolation='nearest')

        # Configure axes
        ax.set_xticks(range(len(column_labels)))
        ax.set_yticks(range(len(column_labels)))
        ax.set_xticklabels(column_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(column_labels, fontsize=9)

        # Add correlation values as text
        for i in range(len(column_labels)):
            for j in range(len(column_labels)):
                text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center', color=text_color, fontsize=8)

        ax.set_title(f'Input Weight Correlations: {post_population.upper()}\n'
                    f'({method.capitalize()} correlation)',
                    fontsize=12, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15, fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved correlation matrix to {save_path}")

        return fig


    def analyze_weights_by_response_type(self,
                                         target_population: str,
                                         post_population: str,
                                         activity_trace: Dict[str, torch.Tensor],
                                         baseline_mask: torch.Tensor,
                                         stim_mask: torch.Tensor,
                                         opsin_expression: Optional[Dict[str, np.ndarray]] = None,
                                         threshold_std: float = 1.0) -> Dict:
            """
            Analyze synaptic weights to cells that increased vs decreased firing

            Classifies post-synaptic cells based on firing rate changes during 
            optogenetic stimulation, then computes weight distributions from each
            presynaptic source to excited vs suppressed cells.

            Args:
                target_population: Optogenetically stimulated population (e.g., 'pv')
                post_population: Post-synaptic population to analyze (e.g., 'gc')
                activity_trace: Activity traces (from simulation results)
                baseline_mask: Boolean mask for baseline period
                stim_mask: Boolean mask for stimulation period
                opsin_expression: Optional dict with opsin expression arrays
                threshold_std: Threshold in standard deviations for classification

            Returns:
                Dictionary with weight distributions and statistics:
                {
                    'post_population': str,
                    'target_population': str,
                    'n_excited': int,
                    'n_suppressed': int,
                    'n_unchanged': int,
                    'excited_indices': np.ndarray,
                    'suppressed_indices': np.ndarray,
                    'unchanged_indices': np.ndarray,
                    'weights_by_source_and_response': {
                        'source_pop': {
                            'to_excited': np.ndarray,  # or opsin_to_excited, etc.
                            'to_suppressed': np.ndarray,
                            'synapse_type': str,
                            'split_by_opsin': bool,
                            ...
                        },
                        ...
                    }
                }
            """

                    
            # Filter to non-expressing cells if analyzing the stimulated population
            filter_to_non_expressing = (
                target_population == post_population and
                opsin_expression is not None and
                post_population in opsin_expression
            )
            
            # Classify post-synaptic cells by response
            post_activity = activity_trace[post_population]

            # Apply filter to non-expressing cells if needed
            cell_indices_to_analyze = None
            if filter_to_non_expressing:
                opsin_mask = opsin_expression[post_population]
                if hasattr(opsin_mask, 'cpu'):
                    opsin_mask = opsin_mask.cpu().numpy()
                else:
                    opsin_mask = np.array(opsin_mask).astype(bool)

                # Get indices of non-expressing cells
                non_expressing_mask = ~opsin_mask
                cell_indices_to_analyze = torch.from_numpy(np.where(non_expressing_mask)[0])

                # Filter activity to only non-expressing cells
                post_activity = post_activity[cell_indices_to_analyze, :]
            
            baseline_rate = torch.mean(post_activity[:, baseline_mask], dim=1)
            stim_rate = torch.mean(post_activity[:, stim_mask], dim=1)
            rate_change = stim_rate - baseline_rate
            baseline_std = torch.std(baseline_rate)

            # Classify cells
            excited_mask = rate_change > threshold_std * baseline_std
            suppressed_mask = rate_change < -threshold_std * baseline_std
            unchanged_mask = ~(excited_mask | suppressed_mask)

            excited_indices_filtered = torch.where(excited_mask)[0].cpu().numpy()
            suppressed_indices_filtered = torch.where(suppressed_mask)[0].cpu().numpy()
            unchanged_indices_filtered = torch.where(unchanged_mask)[0].cpu().numpy()

            # Map back to original cell indices if we filtered
            if cell_indices_to_analyze is not None:
                cell_indices_np = cell_indices_to_analyze.cpu().numpy()
                excited_indices = cell_indices_np[excited_indices_filtered]
                suppressed_indices = cell_indices_np[suppressed_indices_filtered]
                unchanged_indices = cell_indices_np[unchanged_indices_filtered]
            else:
                excited_indices = excited_indices_filtered
                suppressed_indices = suppressed_indices_filtered
                unchanged_indices = unchanged_indices_filtered
        
            n_excited = len(excited_indices)
            n_suppressed = len(suppressed_indices)
            n_unchanged = len(unchanged_indices)

            # Get weight data for each source
            weight_data = self.compute_input_weights_by_source(
                post_population, opsin_expression
            )

            # Organize results
            results = {
                'post_population': post_population,
                'target_population': target_population,
                'n_excited': n_excited,
                'n_suppressed': n_suppressed,
                'n_unchanged': n_unchanged,
                'excited_indices': excited_indices,
                'suppressed_indices': suppressed_indices,
                'unchanged_indices': unchanged_indices,
                'filtered_to_non_expressing': filter_to_non_expressing,
                'n_analyzed': len(cell_indices_to_analyze) if cell_indices_to_analyze is not None else self.pop_sizes[post_population],
                'weights_by_source_and_response': {}
            }

            # For each source population, get weights to excited vs suppressed cells
            for source, source_data in weight_data.items():
                source_results = {}

                # Check if this source should be split by opsin expression
                split_by_opsin = (source == target_population and 
                                 opsin_expression is not None and 
                                 source in opsin_expression)

                if split_by_opsin:
                    # Split by opsin expression AND response type
                    opsin_weights = source_data['opsin_expressing']
                    non_opsin_weights = source_data['non_expressing']

                    source_results['opsin_to_excited'] = opsin_weights[excited_indices] if n_excited > 0 else np.array([])
                    source_results['opsin_to_suppressed'] = opsin_weights[suppressed_indices] if n_suppressed > 0 else np.array([])
                    source_results['opsin_to_unchanged'] = opsin_weights[unchanged_indices] if n_unchanged > 0 else np.array([])

                    source_results['non_opsin_to_excited'] = non_opsin_weights[excited_indices] if n_excited > 0 else np.array([])
                    source_results['non_opsin_to_suppressed'] = non_opsin_weights[suppressed_indices] if n_suppressed > 0 else np.array([])
                    source_results['non_opsin_to_unchanged'] = non_opsin_weights[unchanged_indices] if n_unchanged > 0 else np.array([])

                    source_results['n_opsin'] = source_data['n_opsin_expressing']
                    source_results['n_non_opsin'] = source_data['n_non_expressing']
                else:
                    # Just split by response type
                    total_weights = source_data['total_weights']

                    source_results['to_excited'] = total_weights[excited_indices] if n_excited > 0 else np.array([])
                    source_results['to_suppressed'] = total_weights[suppressed_indices] if n_suppressed > 0 else np.array([])
                    source_results['to_unchanged'] = total_weights[unchanged_indices] if n_unchanged > 0 else np.array([])

                source_results['synapse_type'] = source_data['synapse_type']
                source_results['split_by_opsin'] = split_by_opsin

                results['weights_by_source_and_response'][source] = source_results

            return results

    def plot_weights_by_response_type(self,
                                      analysis_results: Dict,
                                      sources_to_plot: Optional[List[str]] = None,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create violin plots of synaptic weights split by post-synaptic response

        Shows distribution of weights from each source to excited vs suppressed cells.
        If source is the optogenetically stimulated population, further splits by
        opsin expression.

        Args:
            analysis_results: Output from analyze_weights_by_response_type()
            sources_to_plot: List of sources to include (None = all)
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object
        """
        weights_data = analysis_results['weights_by_source_and_response']
        post_pop = analysis_results['post_population']
        target_pop = analysis_results['target_population']

        n_excited = analysis_results['n_excited']
        n_suppressed = analysis_results['n_suppressed']

        # Filter sources
        if sources_to_plot is not None:
            weights_data = {k: v for k, v in weights_data.items() if k in sources_to_plot}

        if len(weights_data) == 0:
            print("No weight data to plot")
            return None

        sources = list(weights_data.keys())

        # Create figure
        n_sources = len(sources)
        # Create figure with 2-column vertical layout
        n_sources = len(sources)
        n_rows = (n_sources + 1) // 2  # Ceiling division
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(12, 5*n_rows),
                                 sharey=True,
                                 squeeze=False)
        
        axes = axes.flatten()

        # Color scheme
        excited_color = '#E74C3C'  # Red
        suppressed_color = '#3498DB'  # Blue

        for idx, source in enumerate(sources):
            ax = axes[idx]
            source_data = weights_data[source]

            plot_data = []
            labels = []
            colors = []

            if source_data['split_by_opsin']:
                # Four violin groups: opsin+/excited, opsin+/suppressed, opsin-/excited, opsin-/suppressed

                # Opsin+ to excited
                if len(source_data['opsin_to_excited']) > 0:
                    plot_data.append(source_data['opsin_to_excited'])
                    labels.append(f'Opsin+\n$\\rightarrow${post_pop.upper()}+')
                    colors.append(excited_color)

                # Opsin+ to suppressed
                if len(source_data['opsin_to_suppressed']) > 0:
                    plot_data.append(source_data['opsin_to_suppressed'])
                    labels.append(f'Opsin+\n$\\rightarrow${post_pop.upper()}$-$')
                    colors.append(suppressed_color)

                # Opsin- to excited
                if len(source_data['non_opsin_to_excited']) > 0:
                    plot_data.append(source_data['non_opsin_to_excited'])
                    labels.append(f'Opsin$-$\n$\\rightarrow${post_pop.upper()}+')
                    colors.append(excited_color)

                # Opsin- to suppressed
                if len(source_data['non_opsin_to_suppressed']) > 0:
                    plot_data.append(source_data['non_opsin_to_suppressed'])
                    labels.append(f'Opsin$-$\n$\\rightarrow${post_pop.upper()}$-$')
                    colors.append(suppressed_color)
            else:
                # Two violin groups: excited, suppressed

                if len(source_data['to_excited']) > 0:
                    plot_data.append(source_data['to_excited'])
                    labels.append(f'$\\rightarrow${post_pop.upper()}+')
                    colors.append(excited_color)

                if len(source_data['to_suppressed']) > 0:
                    plot_data.append(source_data['to_suppressed'])
                    labels.append(f'$\\rightarrow${post_pop.upper()}$-$')
                    colors.append(suppressed_color)

            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{source.upper()}', fontsize=12, fontweight='bold')
                continue

            # Create violin plot
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                                 showmeans=True, showmedians=True)

            # Color violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)

            # Style other elements
            for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1.5)

            # Set labels
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylabel('Input Weight (nS)', fontsize=11)

            # Title with source info
            synapse_type = source_data['synapse_type']
            title = f'{source.upper()}\n({synapse_type})'
            if source_data['split_by_opsin']:
                title += f"\nOpsin+: {source_data['n_opsin']}, Opsin$-$: {source_data['n_non_opsin']}"

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)

            # Add statistics text
            stats_text = []
            for i, data in enumerate(plot_data):
                if len(data) > 0:
                    mean = np.mean(data)
                    text = labels[i].replace("$", "").replace("\\rightarrow", "->")
                    stats_text.append(f'{text}: {mean:.2f} nS')

            if stats_text:
                ax.text(0.98, 0.98, '\n'.join(stats_text),
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white',
                                            alpha=0.8, edgecolor='gray'))
                
        # Remove unused subplot if odd number of sources
        if n_sources % 2 == 1:
            fig.delaxes(axes[-1])

        # Overall title
        title = f'Synaptic Weights by Stimulation Response\n'
        title += f'{target_pop.upper()} Stimulation $\\rightarrow$ {post_pop.upper()} Population\n'
        # Add info about filtering if applicable
        if analysis_results.get('filtered_to_non_expressing', False):
            n_analyzed = analysis_results.get('n_analyzed', 0)
            title += f' (Non-Opsin-Expressing Only)'
        title += f'\n({n_excited} excited, {n_suppressed} suppressed cells)'

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved weights by response plot to: {save_path}")

        return fig
    
    def create_summary_report(self, save_dir='./DG_visualization_report'):
        """
        Create a comprehensive visualization report
        
        Args:
            save_dir: Directory to save all visualization files
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive dentate gyrus visualization report...")
        
        # 1. Spatial organization
        print("  - Spatial organization plots...")
        fig1 = self.plot_spatial_organization(save_path=f'{save_dir}/spatial_organization_3d.png')
        plt.close(fig1)
        
        fig2 = self.plot_spatial_organization(view='2d', save_path=f'{save_dir}/spatial_organization_2d.png')
        plt.close(fig2)
        
        # 2. Connectivity matrices
        print("  - Connectivity matrices...")
        connection_types = ['gc_mc', 'mc_gc', 'pv_gc', 'sst_gc', 'gc_pv']
        for conn_type in connection_types:
            try:
                fig = self.plot_connectivity_matrix(conn_type, save_path=f'{save_dir}/connectivity_{conn_type}.png')
                plt.close(fig)
            except:
                continue
        
        # Full connectivity matrix
        fig3 = self.plot_connectivity_matrix(save_path=f'{save_dir}/connectivity_full_matrix.png')
        plt.close(fig3)
        
        # 3. Spatial connectivity
        print("  - Spatial connectivity plots...")
        for conn_type in ['gc_mc', 'mc_gc'][:2]:  # Limit for speed
            try:
                fig = self.plot_spatial_connectivity(conn_type, save_path=f'{save_dir}/spatial_connectivity_{conn_type}.png')
                plt.close(fig)
            except:
                continue
        
        # 4. Distance distributions
        print("  - Distance distribution analysis...")
        fig4, stats = self.plot_distance_distribution(save_path=f'{save_dir}/distance_distributions.png')
        plt.close(fig4)
        
        # 5. Network graph
        print("  - Network graph...")
        fig5, G = self.plot_network_graph(save_path=f'{save_dir}/network_graph.png')
        plt.close(fig5)
        
        # 6. Activity patterns
        #print("  - Activity patterns...")
        #fig6, activity = self.plot_activity_patterns(save_path=f'{save_dir}/activity_patterns.png')
        #plt.close(fig6)
        
        print(f"Report generated in {save_dir}/")
        print("Files created:")
        for file in os.listdir(save_dir):
            print(f"  - {file}")
        
        return stats, activity

if __name__ == "__main__":
    from DG_circuit_dendritic_somatic_transfer import (
        DentateCircuit, CircuitParams, PerConnectionSynapticParams, OpsinParams
    )

        
    # Create circuit parameters
    circuit_params = CircuitParams()
    synaptic_params = PerConnectionSynapticParams()
    opsin_params = OpsinParams()
    
    # Create circuit
    circuit = DentateCircuit(circuit_params, synaptic_params, opsin_params)

    save_plots = True
    
    vis = DGCircuitVisualization(circuit)
    
    # 3D spatial organization
    print("Generating 3D spatial organization plot...")
    fig1 = vis.plot_spatial_organization(view='3d')
    if save_plots:
        plt.savefig('DG_spatial_3d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2D projections
    print("Generating 2D projection plots...")
    fig2 = vis.plot_spatial_organization(view='2d')
    if save_plots:
        plt.savefig('DG_spatial_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Layer-specific views
    print("Generating layer-specific views...")
    # Show only granule cells and mossy cells (main circuit)
    fig3 = vis.plot_spatial_organization(populations=['gc', 'mc'], view='3d')
    plt.title('Granule Cells and Mossy Cells Only')
    if save_plots:
        plt.savefig('DG_gc_mc_spatial.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Show only interneurons
    fig4 = vis.plot_spatial_organization(populations=['pv', 'sst'], view='3d')
    plt.title('Interneuron Populations Only')
    if save_plots:
        plt.savefig('DG_interneuron_spatial.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Individual connectivity matrices
    connection_types = ['mec_pv', 'mec_gc', 'gc_mc', 'mc_gc', 'pv_gc',
                        'sst_gc', 'gc_pv', 'gc_sst', 'mc_pv', 'mc_sst',
                        'pv_sst', 'sst_pv']
    
    for conn_type in connection_types:
        print(f"Plotting {conn_type} connectivity matrix...")
        try:
            fig = vis.plot_connectivity_matrix(conn_type)
            if save_plots:
                plt.savefig(f'DG_connectivity_{conn_type}.png', dpi=150, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"  Could not plot {conn_type}: {e}")
            continue
    
    # Full connectivity matrix
    print("Plotting full connectivity matrix...")
    fig = vis.plot_connectivity_matrix()
    if save_plots:
        plt.savefig('DG_connectivity_full.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Spatial connectivity patterns
    print("Plotting spatial connectivity patterns...")
    try:
        fig = vis.plot_spatial_connectivity('gc_mc', max_connections=200)
        if save_plots:
            plt.savefig('DG_spatial_connectivity_gc_mc.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"  Could not plot spatial connectivity: {e}")

    print("Plotting distance distributions...")
    connection_types = ['gc_mc', 'mc_gc', 'pv_gc', 'sst_gc',
                        'mc_pv', 'mc_sst', 'pv_sst', 'pv_pv']
    fig, stats = vis.plot_distance_distribution(connection_types)
    
    if save_plots:
        plt.savefig('DG_distance_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
