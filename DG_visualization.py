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
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from typing import Dict, Tuple, List, Optional, Union
import torch
from dataclasses import dataclass
#import colorcet as cc
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
                             mean_linewidth=2.5, cmap='coolwarm', save_path=None):
        """
        Plot circuit activity as raster plots with neurons sorted by mean firing rate

        Each neuron's activity is shown as a row in a heatmap, with color indicating
        firing rate. Neurons are ordered by mean firing rate (highest at top).
        Population mean firing rate is superimposed as a line trace.

        Args:
            activity_trace: Dictionary with population keys, each containing array of shape (n_cells, timesteps)
            vmin: Minimum value for colormap (default: 0 Hz)
            vmax: Maximum value for colormap (default: auto-computed per population)
            mean_linewidth: Line width for population mean trace overlay
            cmap: Colormap name (default: 'coolwarm')
            save_path: Path to save figure
        """
        # Determine number of timesteps
        timesteps = None
        for pop in activity_trace.keys():
            timesteps = activity_trace[pop].shape[1]
            break

        # Create time axis
        time_axis = np.arange(timesteps) * self.circuit.circuit_params.dt

        # Create figure with subplots
        n_populations = len(activity_trace)
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

            # Calculate mean activity per neuron across time
            mean_activity_per_neuron = np.mean(activity_data, axis=1)

            # Sort neurons by mean firing rate (descending - highest at top)
            sorted_indices = np.argsort(mean_activity_per_neuron)[::-1]
            sorted_activity = activity_data[sorted_indices, :]

            # Determine colormap range
            vmax_pop = vmax if vmax is not None else np.percentile(sorted_activity, 99)

            # Plot raster as heatmap
            im = ax.imshow(sorted_activity, 
                           aspect='auto',
                           cmap=cmap,
                           vmin=vmin,
                           vmax=vmax_pop,
                           interpolation='nearest',
                           extent=[time_axis[0], time_axis[-1], n_cells, 0],
                           alpha=0.6,
                           zorder=1)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Firing Rate (Hz)', rotation=270, labelpad=15)

            # Compute population mean activity
            mean_activity = np.mean(activity_data, axis=0)
            activity_history[pop] = mean_activity.tolist()

            # Create twin axis for mean firing rate overlay
            ax2 = ax.twinx()

            # Normalize mean activity to neuron index range for overlay
            # Map mean activity to span the neuron index range
            mean_normalized = (mean_activity / vmax_pop) * n_cells

            # Plot mean firing rate as overlay
            ax2.plot(time_axis, mean_normalized, 
                     color='black',
                     linewidth=mean_linewidth,
                     linestyle='-',
                     alpha=0.8,
                     label=f'Mean={np.mean(mean_activity):.1f} Hz',
                     zorder=2)

            # Configure primary axis (raster)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Neuron Index\n(sorted by activity)')
            ax.set_title(f'{pop.upper()} Activity Raster\n({n_cells} cells)')

            # Configure secondary axis (mean trace)
            ax2.set_ylabel('Normalized Mean Activity', rotation=270, labelpad=20)
            ax2.set_ylim([0, n_cells])
            ax2.legend(loc='upper right', framealpha=0.7)

            # Add grid on primary axis
            ax.grid(False)

        # Remove unused subplots
        for idx in range(len(activity_trace), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('DG Circuit Activity Rasters: Neurons Sorted by Mean Firing Rate', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

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
    connection_types = ['gc_mc', 'mc_gc', 'pv_gc', 'sst_gc']
    
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
                        'mc_pv', 'mc_sst', 'sst_pv', 'pv_pv']
    fig, stats = vis.plot_distance_distribution(connection_types)
    
    if save_plots:
        plt.savefig('DG_distance_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
