"""
Custom Barnes-Hut Simulation Example
===================================

This script shows how to create your own Barnes-Hut simulation from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_simulation import BarnesHutNBody

def my_custom_simulation():
    """Create a custom Barnes-Hut simulation of colliding star clusters."""
    
    # Simulation parameters
    N_per_cluster = 75  # particles per cluster
    N_total = 2 * N_per_cluster
    
    print(f"Creating custom Barnes-Hut simulation with {N_total} particles")
    print("Scenario: Two star clusters on collision course")
    print("-" * 50)
    
    # Create the simulation
    sim = BarnesHutNBody(N=N_total, G=1.0, theta=0.5)
    
    # Manually set up two clusters
    positions = np.zeros((N_total, 2))
    velocities = np.zeros((N_total, 2))
    masses = np.ones(N_total)
    
    # Cluster 1: Left side
    cluster1_center = np.array([-4.0, 0.0])
    cluster1_velocity = np.array([0.5, 0.0])  # Moving right
    
    for i in range(N_per_cluster):
        # Random position around cluster center
        angle = np.random.random() * 2 * np.pi
        radius = np.random.rayleigh(1.0)  # Rayleigh distribution for realistic cluster
        
        positions[i] = cluster1_center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        # Add orbital velocity + cluster velocity
        orbital_speed = 0.3 * np.sqrt(1.0 / max(radius, 0.1))
        orbital_vel = orbital_speed * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = cluster1_velocity + orbital_vel
        
        # Central mass is heavier
        masses[i] = 2.0 if i == 0 else 1.0
    
    # Cluster 2: Right side
    cluster2_center = np.array([4.0, 1.0])
    cluster2_velocity = np.array([-0.5, 0.0])  # Moving left
    
    for i in range(N_per_cluster, N_total):
        angle = np.random.random() * 2 * np.pi
        radius = np.random.rayleigh(1.2)
        
        positions[i] = cluster2_center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        orbital_speed = 0.3 * np.sqrt(1.0 / max(radius, 0.1))
        orbital_vel = orbital_speed * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = cluster2_velocity + orbital_vel
        
        masses[i] = 1.5 if i == N_per_cluster else 1.0  # Different central mass
    
    # Set initial conditions
    sim.set_initial_conditions(positions, velocities, masses)    # Display initial setup
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(positions[:N_per_cluster, 0], positions[:N_per_cluster, 1], 
               c='blue', alpha=0.7, s=masses[:N_per_cluster]*20, label='Cluster 1')
    plt.scatter(positions[N_per_cluster:, 0], positions[N_per_cluster:, 1], 
               c='red', alpha=0.7, s=masses[N_per_cluster:]*20, label='Cluster 2')
    plt.scatter(positions[0, 0], positions[0, 1], c='darkblue', s=100, marker='*')
    plt.scatter(positions[N_per_cluster, 0], positions[N_per_cluster, 1], c='darkred', s=100, marker='*')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Initial Configuration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Run the simulation
    print("Starting Barnes-Hut simulation...")
    print(f"θ = {sim.theta} (accuracy parameter)")
      t_span = [0, 12.0]  # Longer time to see collision
    dt = 0.05
    
    start_time = time.time()
    sim.simulate(t_span, dt, use_barnes_hut=True, save_interval=3)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.1f} seconds")
    print(f"Total time steps: {len(sim.time_points)}")
    
    # Calculate dynamic axis limits after simulation
    all_pos = np.vstack([positions] + sim.position_history)
    x_min, x_max = np.min(all_pos[:, 0]), np.max(all_pos[:, 0])
    y_min, y_max = np.min(all_pos[:, 1]), np.max(all_pos[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.15
    x_lim = [x_min - padding * x_range, x_max + padding * x_range]
    y_lim = [y_min - padding * y_range, y_max + padding * y_range]
    
    # Update the initial plot with proper limits
    plt.subplot(2, 2, 1)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Show intermediate and final states
    mid_idx = len(sim.position_history) // 2
    final_idx = -1
    
    plt.subplot(2, 2, 2)
    mid_pos = sim.position_history[mid_idx]
    plt.scatter(mid_pos[:N_per_cluster, 0], mid_pos[:N_per_cluster, 1], 
               c='blue', alpha=0.7, s=masses[:N_per_cluster]*20)
    plt.scatter(mid_pos[N_per_cluster:, 0], mid_pos[N_per_cluster:, 1], 
               c='red', alpha=0.7, s=masses[N_per_cluster:]*20)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Mid-Collision (t={sim.time_points[mid_idx]:.1f})')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(2, 2, 3)
    final_pos = sim.position_history[final_idx]
    plt.scatter(final_pos[:N_per_cluster, 0], final_pos[:N_per_cluster, 1], 
               c='blue', alpha=0.7, s=masses[:N_per_cluster]*20)
    plt.scatter(final_pos[N_per_cluster:, 0], final_pos[N_per_cluster:, 1], 
               c='red', alpha=0.7, s=masses[N_per_cluster:]*20)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Final State (t={sim.time_points[final_idx]:.1f})')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Energy analysis
    plt.subplot(2, 2, 4)
    energy_history = np.array(sim.energy_history)
    energy_drift = (energy_history - energy_history[0]) / abs(energy_history[0])
    plt.plot(sim.time_points, energy_drift, 'g-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Relative Energy Drift')
    plt.title('Energy Conservation')
    plt.grid(True, alpha=0.3)
    print(f"Final energy drift: {energy_drift[-1]:.2e}")
    
    plt.tight_layout()
    plt.savefig('custom_barnes_hut_collision.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create a simple animation of the collision
    print("Creating collision animation...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate_frame(frame):
        ax.clear()
        pos = sim.position_history[frame]
        
        ax.scatter(pos[:N_per_cluster, 0], pos[:N_per_cluster, 1], 
                  c='blue', alpha=0.7, s=masses[:N_per_cluster]*15, label='Cluster 1')
        ax.scatter(pos[N_per_cluster:, 0], pos[N_per_cluster:, 1], 
                  c='red', alpha=0.7, s=masses[N_per_cluster:]*15, label='Cluster 2')
        
        # Highlight central masses
        ax.scatter(pos[0, 0], pos[0, 1], c='darkblue', s=50, marker='*')
        ax.scatter(pos[N_per_cluster, 0], pos[N_per_cluster, 1], c='darkred', s=50, marker='*')
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Cluster Collision - Barnes-Hut Simulation (t={sim.time_points[frame]:.1f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
      # Create a few key frames instead of full animation
    key_frames = [0, len(sim.position_history)//4, len(sim.position_history)//2, 
                  3*len(sim.position_history)//4, -1]
    
    fig, axes = plt.subplots(1, len(key_frames), figsize=(20, 4))
    
    # Calculate dynamic axis limits based on all frames
    all_positions = np.vstack(sim.position_history)
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    
    for i, frame in enumerate(key_frames):
        if frame == -1:
            frame = len(sim.position_history) - 1
            
        pos = sim.position_history[frame]
        
        axes[i].scatter(pos[:N_per_cluster, 0], pos[:N_per_cluster, 1], 
                       c='blue', alpha=0.7, s=masses[:N_per_cluster]*15)
        axes[i].scatter(pos[N_per_cluster:, 0], pos[N_per_cluster:, 1], 
                       c='red', alpha=0.7, s=masses[N_per_cluster:]*15)
        axes[i].scatter(pos[0, 0], pos[0, 1], c='darkblue', s=50, marker='*')
        axes[i].scatter(pos[N_per_cluster, 0], pos[N_per_cluster, 1], c='darkred', s=50, marker='*')
        
        # Use dynamic axis limits
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_title(f't = {sim.time_points[frame]:.1f}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
    
    plt.suptitle('Barnes-Hut Cluster Collision Timeline', fontsize=16)
    plt.tight_layout()
    plt.savefig('custom_barnes_hut_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nSimulation Analysis:")
    print(f"• Total particles: {N_total}")
    print(f"• Algorithm: Barnes-Hut (θ = {sim.theta})")
    print(f"• Simulation time: {sim.time_points[-1]:.1f} time units")
    print(f"• Energy conservation: {energy_drift[-1]:.2e} relative drift")
    print(f"• Computation time: {end_time - start_time:.1f} seconds")
    print(f"• Average time per step: {(end_time - start_time)/len(sim.time_points)*1000:.1f} ms")
    
    return sim

if __name__ == "__main__":
    import time
    
    print("Custom Barnes-Hut Simulation")
    print("=" * 40)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run the custom simulation
    simulation = my_custom_simulation()
    
    print("\n" + "="*50)
    print("Barnes-Hut Algorithm successfully demonstrated!")
    print("Key advantages observed:")
    print("• Efficient O(N log N) force calculation")
    print("• Reasonable energy conservation")
    print("• Enables large-scale simulations")
    print("• Tunable accuracy with θ parameter")
    print("="*50)
