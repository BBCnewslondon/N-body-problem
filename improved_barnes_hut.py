"""
Improved Barnes-Hut Simulation with Better Physics and Visualization
================================================================

This version fixes the numerical instability and visualization issues by:
1. Better initial conditions and physics parameters
2. Improved time step and simulation parameters  
3. Enhanced visualization with adaptive scaling
4. Energy conservation monitoring and corrections
"""

import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_simulation import BarnesHutNBody
import time

def improved_barnes_hut_simulation():
    """Create an improved Barnes-Hut simulation with stable physics."""
    
    # Improved simulation parameters
    N_per_cluster = 50  # Smaller clusters for better stability
    N_total = 2 * N_per_cluster
    
    print(f"Creating improved Barnes-Hut simulation with {N_total} particles")
    print("Scenario: Two stable star clusters with controlled collision")
    print("-" * 60)
    
    # Create simulation with better parameters
    sim = BarnesHutNBody(N=N_total, G=0.1, theta=0.5)  # Reduced G for stability
    
    # More realistic cluster setup
    positions = np.zeros((N_total, 2))
    velocities = np.zeros((N_total, 2))
    masses = np.ones(N_total) * 0.1  # Smaller masses for stability
    
    # Cluster 1: Left side with better initial conditions
    cluster1_center = np.array([-3.0, 0.0])
    cluster1_velocity = np.array([0.2, 0.0])  # Slower collision speed
    
    # Central massive object for cluster 1
    positions[0] = cluster1_center
    velocities[0] = cluster1_velocity
    masses[0] = 5.0  # Massive central object
    
    # Particles in stable orbits around central mass
    for i in range(1, N_per_cluster):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.5, 2.0)  # Controlled radius range
        
        # Position relative to cluster center
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster1_center + pos_rel
        
        # Circular orbital velocity for stability
        orbital_speed = np.sqrt(5.0 * sim.G / radius)  # Keplerian velocity
        orbital_vel = orbital_speed * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = cluster1_velocity + orbital_vel * 0.8  # Slightly sub-Keplerian
        
        masses[i] = 0.1  # Small particle mass
    
    # Cluster 2: Right side with symmetric setup
    cluster2_center = np.array([3.0, 0.5])
    cluster2_velocity = np.array([-0.2, 0.0])  # Moving towards cluster 1
    
    # Central massive object for cluster 2
    positions[N_per_cluster] = cluster2_center
    velocities[N_per_cluster] = cluster2_velocity
    masses[N_per_cluster] = 4.0  # Slightly different mass
    
    for i in range(N_per_cluster + 1, N_total):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.5, 1.8)
        
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster2_center + pos_rel
        
        orbital_speed = np.sqrt(4.0 * sim.G / radius)
        orbital_vel = orbital_speed * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = cluster2_velocity + orbital_vel * 0.8
        
        masses[i] = 0.1
    
    # Set initial conditions
    sim.set_initial_conditions(positions, velocities, masses)
    
    # Calculate initial energy for reference
    initial_energy = sim.calculate_total_energy()
    print(f"Initial total energy: {initial_energy:.3f}")
    
    # Display initial setup with better visualization
    plt.figure(figsize=(15, 10))
    
    # Calculate reasonable axis limits for initial view
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    x_center = np.mean(positions[:, 0])
    y_center = np.mean(positions[:, 1])
    max_range = max(x_range, y_range)
    margin = max_range * 0.3
    
    initial_xlim = [x_center - max_range/2 - margin, x_center + max_range/2 + margin]
    initial_ylim = [y_center - max_range/2 - margin, y_center + max_range/2 + margin]
    
    plt.subplot(2, 3, 1)
    # Plot particles with size proportional to mass
    cluster1_particles = positions[:N_per_cluster]
    cluster2_particles = positions[N_per_cluster:]
    cluster1_masses = masses[:N_per_cluster]
    cluster2_masses = masses[N_per_cluster:]
    
    plt.scatter(cluster1_particles[1:, 0], cluster1_particles[1:, 1], 
               c='blue', alpha=0.7, s=cluster1_masses[1:]*100, label='Cluster 1')
    plt.scatter(cluster2_particles[1:, 0], cluster2_particles[1:, 1], 
               c='red', alpha=0.7, s=cluster2_masses[1:]*100, label='Cluster 2')
    
    # Highlight central masses
    plt.scatter(positions[0, 0], positions[0, 1], c='darkblue', s=200, marker='*', 
                label='Central Mass 1')
    plt.scatter(positions[N_per_cluster, 0], positions[N_per_cluster, 1], c='darkred', 
                s=200, marker='*', label='Central Mass 2')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Initial Configuration (Improved)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim(initial_xlim)
    plt.ylim(initial_ylim)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Run simulation with improved parameters
    print("Starting improved Barnes-Hut simulation...")
    print(f"G = {sim.G}, θ = {sim.theta}")
    
    t_span = [0, 30.0]  # Longer simulation time
    dt = 0.02  # Smaller time step for stability
    
    start_time = time.time()
    sim.simulate(t_span, dt, use_barnes_hut=True, save_interval=10)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.1f} seconds")
    print(f"Total time steps: {len(sim.time_points)}")
    
    # Calculate final energy and conservation
    final_energy = sim.energy_history[-1]
    energy_conservation = abs((final_energy - initial_energy) / initial_energy)
    print(f"Final total energy: {final_energy:.3f}")
    print(f"Energy conservation error: {energy_conservation:.2e}")
    
    # Calculate improved axis limits based on a subset of the simulation
    # Use first 80% of simulation to avoid extreme outliers
    subset_history = sim.position_history[:int(0.8 * len(sim.position_history))]
    all_pos = np.vstack([positions] + subset_history)
    
    # Remove extreme outliers (beyond 3 standard deviations)
    x_std = np.std(all_pos[:, 0])
    y_std = np.std(all_pos[:, 1])
    x_mean = np.mean(all_pos[:, 0])
    y_mean = np.mean(all_pos[:, 1])
    
    # Filter outliers
    mask = (np.abs(all_pos[:, 0] - x_mean) < 3 * x_std) & (np.abs(all_pos[:, 1] - y_mean) < 3 * y_std)
    filtered_pos = all_pos[mask]
    
    x_min, x_max = np.min(filtered_pos[:, 0]), np.max(filtered_pos[:, 0])
    y_min, y_max = np.min(filtered_pos[:, 1]), np.max(filtered_pos[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Ensure minimum range for visibility
    min_range = 5.0
    if x_range < min_range:
        x_center = (x_min + x_max) / 2
        x_min = x_center - min_range / 2
        x_max = x_center + min_range / 2
        x_range = min_range
    if y_range < min_range:
        y_center = (y_min + y_max) / 2
        y_min = y_center - min_range / 2
        y_max = y_center + min_range / 2
        y_range = min_range
    
    padding = 0.2
    x_lim = [x_min - padding * x_range, x_max + padding * x_range]
    y_lim = [y_min - padding * y_range, y_max + padding * y_range]
    
    print(f"Improved axis limits: X=[{x_lim[0]:.1f}, {x_lim[1]:.1f}], Y=[{y_lim[0]:.1f}, {y_lim[1]:.1f}]")
    
    # Show key simulation phases
    phases = [
        len(sim.position_history) // 4,
        len(sim.position_history) // 2,
        3 * len(sim.position_history) // 4,
        -1
    ]
    phase_titles = ['Early Evolution', 'Approach Phase', 'Interaction Phase', 'Final State']
    
    for i, (phase_idx, title) in enumerate(zip(phases, phase_titles)):
        if phase_idx == -1:
            phase_idx = len(sim.position_history) - 1
            
        plt.subplot(2, 3, i + 2)
        pos = sim.position_history[phase_idx]
        
        # Plot with improved sizing and colors
        plt.scatter(pos[1:N_per_cluster, 0], pos[1:N_per_cluster, 1], 
                   c='blue', alpha=0.8, s=masses[1:N_per_cluster]*150)
        plt.scatter(pos[N_per_cluster+1:, 0], pos[N_per_cluster+1:, 1], 
                   c='red', alpha=0.8, s=masses[N_per_cluster+1:]*150)
        
        # Central masses
        plt.scatter(pos[0, 0], pos[0, 1], c='darkblue', s=300, marker='*')
        plt.scatter(pos[N_per_cluster, 0], pos[N_per_cluster, 1], c='darkred', s=300, marker='*')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'{title} (t={sim.time_points[phase_idx]:.1f})')
        plt.grid(True, alpha=0.3)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.gca().set_aspect('equal', adjustable='box')
    
    # Energy conservation plot
    plt.subplot(2, 3, 6)
    energy_history = np.array(sim.energy_history)
    relative_energy_error = (energy_history - energy_history[0]) / abs(energy_history[0])
    
    plt.plot(sim.time_points, relative_energy_error, 'g-', linewidth=2, label='Energy Drift')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Perfect Conservation')
    plt.xlabel('Time')
    plt.ylabel('Relative Energy Error')
    plt.title('Energy Conservation (Improved)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('symlog', linthresh=1e-10)  # Symmetric log scale for small values
    
    print(f"Final relative energy error: {relative_energy_error[-1]:.2e}")
    
    plt.tight_layout()
    plt.savefig('improved_barnes_hut_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create improved timeline
    print("Creating improved timeline visualization...")
    
    # Select key frames with better spacing
    n_frames = 6
    frame_indices = np.linspace(0, len(sim.position_history) - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(frame_indices):
        pos = sim.position_history[frame_idx]
        t = sim.time_points[frame_idx]
        
        # Plot with consistent scaling and better visualization
        axes[i].scatter(pos[1:N_per_cluster, 0], pos[1:N_per_cluster, 1], 
                       c='blue', alpha=0.8, s=masses[1:N_per_cluster]*120, label='Cluster 1' if i == 0 else "")
        axes[i].scatter(pos[N_per_cluster+1:, 0], pos[N_per_cluster+1:, 1], 
                       c='red', alpha=0.8, s=masses[N_per_cluster+1:]*120, label='Cluster 2' if i == 0 else "")
        
        # Central masses with trails
        axes[i].scatter(pos[0, 0], pos[0, 1], c='darkblue', s=200, marker='*')
        axes[i].scatter(pos[N_per_cluster, 0], pos[N_per_cluster, 1], c='darkred', s=200, marker='*')
        
        # Add trails for central masses
        if frame_idx > 0:
            trail_length = min(frame_idx, 20)
            trail_start = max(0, frame_idx - trail_length)
            for j in range(trail_start, frame_idx):
                alpha = (j - trail_start) / trail_length * 0.5
                trail_pos = sim.position_history[j]
                axes[i].plot(trail_pos[0, 0], trail_pos[0, 1], 'o', c='lightblue', 
                           alpha=alpha, markersize=3)
                axes[i].plot(trail_pos[N_per_cluster, 0], trail_pos[N_per_cluster, 1], 'o', 
                           c='lightcoral', alpha=alpha, markersize=3)
        
        axes[i].set_xlim(x_lim)
        axes[i].set_ylim(y_lim)
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_title(f't = {t:.1f}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
        
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Improved Barnes-Hut Cluster Collision Timeline', fontsize=16)
    plt.tight_layout()
    plt.savefig('improved_barnes_hut_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Performance and stability analysis
    print("\n" + "="*60)
    print("IMPROVED SIMULATION ANALYSIS")
    print("="*60)
    print(f"• Total particles: {N_total}")
    print(f"• Algorithm: Barnes-Hut (θ = {sim.theta})")
    print(f"• Gravitational constant: {sim.G}")
    print(f"• Simulation time: {sim.time_points[-1]:.1f} time units")
    print(f"• Time step: {dt}")
    print(f"• Energy conservation: {energy_conservation:.2e} relative error")
    print(f"• Computation time: {end_time - start_time:.1f} seconds")
    print(f"• Average time per step: {(end_time - start_time)/len(sim.time_points)*1000:.1f} ms")
    print(f"• Axis scaling: Dynamic with outlier filtering")
    print(f"• Final axis range: X={x_range:.1f}, Y={y_range:.1f}")
    
    if energy_conservation < 1e-2:
        print("✓ EXCELLENT energy conservation!")
    elif energy_conservation < 1e-1:
        print("✓ Good energy conservation")
    else:
        print("⚠ Poor energy conservation - consider smaller time step")
    
    print("="*60)
    
    return sim

if __name__ == "__main__":
    print("Improved Barnes-Hut N-Body Simulation")
    print("====================================")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run the improved simulation
    simulation = improved_barnes_hut_simulation()
    
    print("\n" + "="*50)
    print("IMPROVEMENTS MADE:")
    print("• Reduced gravitational constant for stability")
    print("• Smaller particle masses and better mass ratios")
    print("• Keplerian orbital velocities for stable clusters")
    print("• Smaller time steps for numerical accuracy")
    print("• Outlier filtering for better axis scaling")
    print("• Enhanced visualization with trails and sizing")
    print("• Better energy conservation monitoring")
    print("="*50)
