"""
Axis Scaling Utilities for Barnes-Hut Simulations
=================================================

This module provides utilities to automatically scale plot axes based on particle positions
to ensure all particles remain visible throughout the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_dynamic_limits(position_history, padding=0.15, min_range=2.0):
    """
    Calculate dynamic axis limits based on all particle positions in simulation.
    
    Parameters:
    position_history: list of position arrays [(N, 2), ...]
    padding: fraction of range to add as padding (default 0.15 = 15%)
    min_range: minimum axis range to ensure plots aren't too zoomed in
    
    Returns:
    x_lim, y_lim: tuples of (min, max) for x and y axes
    """
    # Stack all positions from all time steps
    all_positions = np.vstack(position_history)
    
    # Calculate bounds
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    
    # Calculate ranges
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Ensure minimum range
    if x_range < min_range:
        center_x = (x_min + x_max) / 2
        x_min = center_x - min_range / 2
        x_max = center_x + min_range / 2
        x_range = min_range
        
    if y_range < min_range:
        center_y = (y_min + y_max) / 2
        y_min = center_y - min_range / 2
        y_max = center_y + min_range / 2
        y_range = min_range
    
    # Add padding
    x_pad = padding * x_range
    y_pad = padding * y_range
    
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)
    
    return x_lim, y_lim

def calculate_frame_limits(positions, padding=0.1, min_range=1.0):
    """
    Calculate axis limits for a single frame of positions.
    
    Parameters:
    positions: array of shape (N, 2) with particle positions
    padding: fraction of range to add as padding
    min_range: minimum axis range
    
    Returns:
    x_lim, y_lim: tuples of (min, max) for x and y axes
    """
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Ensure minimum range
    if x_range < min_range:
        center_x = (x_min + x_max) / 2
        x_min = center_x - min_range / 2
        x_max = center_x + min_range / 2
        x_range = min_range
        
    if y_range < min_range:
        center_y = (y_min + y_max) / 2
        y_min = center_y - min_range / 2
        y_max = center_y + min_range / 2
        y_range = min_range
    
    x_pad = padding * x_range
    y_pad = padding * y_range
    
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)
    
    return x_lim, y_lim

def set_equal_aspect_with_limits(ax, x_lim, y_lim):
    """
    Set equal aspect ratio while respecting axis limits.
    
    Parameters:
    ax: matplotlib axis object
    x_lim, y_lim: axis limits tuples
    """
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')

def plot_simulation_timeline(position_history, time_points, masses, 
                           cluster_sizes=None, n_frames=5, 
                           figsize=(20, 4), title="Simulation Timeline"):
    """
    Create a timeline plot showing key frames of the simulation with proper axis scaling.
    
    Parameters:
    position_history: list of position arrays
    time_points: corresponding time points
    masses: particle masses for sizing
    cluster_sizes: list of cluster sizes for coloring (optional)
    n_frames: number of frames to show
    figsize: figure size
    title: plot title
    
    Returns:
    fig, axes: matplotlib figure and axes objects
    """
    # Select frames evenly spaced through simulation
    frame_indices = np.linspace(0, len(position_history) - 1, n_frames, dtype=int)
    
    # Calculate global axis limits
    x_lim, y_lim = calculate_dynamic_limits(position_history, padding=0.1)
    
    fig, axes = plt.subplots(1, n_frames, figsize=figsize)
    if n_frames == 1:
        axes = [axes]
    
    # Color scheme
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, frame_idx in enumerate(frame_indices):
        pos = position_history[frame_idx]
        t = time_points[frame_idx]
        
        if cluster_sizes is None:
            # Single color for all particles
            axes[i].scatter(pos[:, 0], pos[:, 1], s=masses*10, alpha=0.7, c='blue')
        else:
            # Multiple clusters with different colors
            start_idx = 0
            for j, cluster_size in enumerate(cluster_sizes):
                end_idx = start_idx + cluster_size
                color = colors[j % len(colors)]
                axes[i].scatter(pos[start_idx:end_idx, 0], pos[start_idx:end_idx, 1],
                              s=masses[start_idx:end_idx]*10, alpha=0.7, c=color,
                              label=f'Cluster {j+1}' if i == 0 else "")
                
                # Highlight central particle
                if start_idx == 0:  # First particle of first cluster
                    axes[i].scatter(pos[start_idx, 0], pos[start_idx, 1], 
                                  c='darkblue', s=50, marker='*')
                elif j == 1:  # First particle of second cluster
                    axes[i].scatter(pos[start_idx, 0], pos[start_idx, 1], 
                                  c='darkred', s=50, marker='*')
                
                start_idx = end_idx
        
        set_equal_aspect_with_limits(axes[i], x_lim, y_lim)
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_title(f't = {t:.1f}')
        axes[i].grid(True, alpha=0.3)
        
        if i == 0 and cluster_sizes is not None:
            axes[i].legend(loc='upper right', fontsize=8)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig, axes

def plot_before_after_comparison(initial_positions, final_positions, masses,
                                cluster_sizes=None, figsize=(12, 5)):
    """
    Create before/after comparison plot with proper axis scaling.
    
    Parameters:
    initial_positions: initial particle positions
    final_positions: final particle positions  
    masses: particle masses
    cluster_sizes: list of cluster sizes for coloring (optional)
    figsize: figure size
    
    Returns:
    fig, axes: matplotlib figure and axes objects
    """
    # Calculate global limits
    all_pos = np.vstack([initial_positions, final_positions])
    x_lim, y_lim = calculate_frame_limits(all_pos, padding=0.15)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    positions = [initial_positions, final_positions]
    titles = ['Initial Configuration', 'Final Configuration']
    
    for i in range(2):
        pos = positions[i]
        
        if cluster_sizes is None:
            axes[i].scatter(pos[:, 0], pos[:, 1], s=masses*15, alpha=0.7, c='blue')
        else:
            colors = ['blue', 'red', 'green', 'orange']
            start_idx = 0
            for j, cluster_size in enumerate(cluster_sizes):
                end_idx = start_idx + cluster_size
                color = colors[j % len(colors)]
                axes[i].scatter(pos[start_idx:end_idx, 0], pos[start_idx:end_idx, 1],
                              s=masses[start_idx:end_idx]*15, alpha=0.7, c=color,
                              label=f'Cluster {j+1}' if i == 0 else "")
                start_idx = end_idx
        
        set_equal_aspect_with_limits(axes[i], x_lim, y_lim)
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position') 
        axes[i].set_title(titles[i])
        axes[i].grid(True, alpha=0.3)
        
        if i == 0 and cluster_sizes is not None:
            axes[i].legend()
    
    plt.tight_layout()
    return fig, axes

def create_adaptive_animation_frame(ax, positions, masses, x_lim, y_lim, 
                                  time, cluster_sizes=None, trail_positions=None):
    """
    Create a single animation frame with adaptive scaling.
    
    Parameters:
    ax: matplotlib axis object
    positions: current particle positions
    masses: particle masses
    x_lim, y_lim: axis limits
    time: current time
    cluster_sizes: list of cluster sizes for coloring (optional)
    trail_positions: previous positions for trails (optional)
    """
    ax.clear()
    
    # Draw trails if provided
    if trail_positions is not None:
        for trail_pos in trail_positions:
            if cluster_sizes is None:
                ax.scatter(trail_pos[:, 0], trail_pos[:, 1], s=1, alpha=0.3, c='gray')
            else:
                colors = ['lightblue', 'lightcoral', 'lightgreen', 'moccasin']
                start_idx = 0
                for j, cluster_size in enumerate(cluster_sizes):
                    end_idx = start_idx + cluster_size
                    color = colors[j % len(colors)]
                    ax.scatter(trail_pos[start_idx:end_idx, 0], trail_pos[start_idx:end_idx, 1],
                             s=1, alpha=0.3, c=color)
                    start_idx = end_idx
    
    # Draw current positions
    if cluster_sizes is None:
        ax.scatter(positions[:, 0], positions[:, 1], s=masses*10, alpha=0.8, c='blue')
    else:
        colors = ['blue', 'red', 'green', 'orange']
        start_idx = 0
        for j, cluster_size in enumerate(cluster_sizes):
            end_idx = start_idx + cluster_size
            color = colors[j % len(colors)]
            ax.scatter(positions[start_idx:end_idx, 0], positions[start_idx:end_idx, 1],
                      s=masses[start_idx:end_idx]*10, alpha=0.8, c=color)
            start_idx = end_idx
    
    set_equal_aspect_with_limits(ax, x_lim, y_lim)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Barnes-Hut N-Body Simulation (t = {time:.2f})')
    ax.grid(True, alpha=0.3)

# Example usage function
def demo_axis_scaling():
    """Demonstrate the axis scaling utilities."""
    print("Axis Scaling Utilities Demo")
    print("=" * 30)
    
    # Create some sample data
    import matplotlib.pyplot as plt
    
    # Simulate expanding particle system
    n_particles = 50
    n_frames = 5
    
    position_history = []
    time_points = np.linspace(0, 10, n_frames)
    
    for i, t in enumerate(time_points):
        # Particles expand over time
        expansion_factor = 1 + t * 0.5
        angles = np.random.random(n_particles) * 2 * np.pi
        radii = np.random.random(n_particles) * expansion_factor
        
        positions = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        position_history.append(positions)
    
    masses = np.ones(n_particles)
    cluster_sizes = [25, 25]  # Two clusters
    
    # Demonstrate timeline plot
    fig, axes = plot_simulation_timeline(position_history, time_points, masses, 
                                       cluster_sizes, n_frames=5)
    plt.savefig('axis_scaling_demo_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Demonstrate before/after comparison
    fig, axes = plot_before_after_comparison(position_history[0], position_history[-1], 
                                           masses, cluster_sizes)
    plt.savefig('axis_scaling_demo_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Axis scaling demo completed!")
    print("Key features:")
    print("• Dynamic axis limits based on all particle positions")
    print("• Consistent scaling across all frames")
    print("• Proper aspect ratios maintained")
    print("• Minimum range enforcement for readability")

if __name__ == "__main__":
    demo_axis_scaling()
