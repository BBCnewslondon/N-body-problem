"""
Highly Stable Barnes-Hut Simulation
==================================

This version focuses on numerical stability and realistic physics with:
1. Much smaller gravitational constant and masses
2. Very small time steps
3. Proper velocity scaling
4. Conservative collision parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_simulation import BarnesHutNBody
import time

def stable_barnes_hut_simulation():
    """Create a highly stable Barnes-Hut simulation."""
    
    # Conservative parameters for maximum stability
    N_per_cluster = 30  # Small clusters
    N_total = 2 * N_per_cluster
    
    print(f"Creating stable Barnes-Hut simulation with {N_total} particles")
    print("Focus: Numerical stability and realistic physics")
    print("-" * 50)
    
    # Very conservative physics parameters
    sim = BarnesHutNBody(N=N_total, G=0.01, theta=0.5)  # Very small G
    
    # Initialize arrays
    positions = np.zeros((N_total, 2))
    velocities = np.zeros((N_total, 2))
    masses = np.ones(N_total) * 0.01  # Very small masses
    
    # Cluster 1: Stable configuration
    cluster1_center = np.array([-2.0, 0.0])
    cluster1_velocity = np.array([0.1, 0.0])  # Very slow approach
    
    # Central mass for cluster 1
    positions[0] = cluster1_center
    velocities[0] = cluster1_velocity  
    masses[0] = 1.0  # Central mass
    
    # Stable satellite particles
    for i in range(1, N_per_cluster):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, 1.0)  # Close, stable orbits
        
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster1_center + pos_rel
        
        # Proper circular velocity for stability
        v_circular = np.sqrt(sim.G * masses[0] / radius)
        vel_rel = v_circular * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = cluster1_velocity + vel_rel * 0.7  # Slightly elliptical
        
        masses[i] = 0.01
    
    # Cluster 2: Mirror configuration
    cluster2_center = np.array([2.0, 0.0])
    cluster2_velocity = np.array([-0.1, 0.0])
    
    positions[N_per_cluster] = cluster2_center
    velocities[N_per_cluster] = cluster2_velocity
    masses[N_per_cluster] = 1.0
    
    for i in range(N_per_cluster + 1, N_total):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, 1.0)
        
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster2_center + pos_rel
        
        v_circular = np.sqrt(sim.G * masses[N_per_cluster] / radius)
        vel_rel = v_circular * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = cluster2_velocity + vel_rel * 0.7
        
        masses[i] = 0.01
    
    # Set initial conditions
    sim.set_initial_conditions(positions, velocities, masses)
    
    # Calculate and display initial energy
    initial_energy = sim.calculate_total_energy()
    print(f"Initial total energy: {initial_energy:.6f}")
    
    # Display setup
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Initial configuration
    axes[0, 0].scatter(positions[1:N_per_cluster, 0], positions[1:N_per_cluster, 1], 
                      c='blue', alpha=0.8, s=50, label='Cluster 1')
    axes[0, 0].scatter(positions[N_per_cluster+1:, 0], positions[N_per_cluster+1:, 1], 
                      c='red', alpha=0.8, s=50, label='Cluster 2')
    axes[0, 0].scatter(positions[0, 0], positions[0, 1], c='darkblue', s=200, marker='*')
    axes[0, 0].scatter(positions[N_per_cluster, 0], positions[N_per_cluster, 1], c='darkred', s=200, marker='*')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].set_title('Initial Configuration (Stable)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-2, 2)
    axes[0, 0].set_aspect('equal')
    
    # Run stable simulation
    print("Starting stable Barnes-Hut simulation...")
    print(f"G = {sim.G}, θ = {sim.theta}")
    
    t_span = [0, 20.0]  # Moderate time span
    dt = 0.005  # Very small time step for stability
    
    start_time = time.time()
    sim.simulate(t_span, dt, use_barnes_hut=True, save_interval=20)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.1f} seconds")
    print(f"Total time steps: {len(sim.time_points)}")
    
    # Check energy conservation
    final_energy = sim.energy_history[-1]
    energy_conservation = abs((final_energy - initial_energy) / initial_energy)
    print(f"Final total energy: {final_energy:.6f}")
    print(f"Energy conservation error: {energy_conservation:.2e}")
    
    # Calculate reasonable axis limits (no extreme outliers)
    all_positions = np.vstack(sim.position_history)
    
    # Use percentiles to avoid extreme outliers
    x_min, x_max = np.percentile(all_positions[:, 0], [1, 99])
    y_min, y_max = np.percentile(all_positions[:, 1], [1, 99])
    
    # Ensure reasonable range
    x_range = max(x_max - x_min, 4.0)  # Minimum 4 unit range
    y_range = max(y_max - y_min, 4.0)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    padding = 0.3
    x_lim = [x_center - x_range/2 - padding*x_range, x_center + x_range/2 + padding*x_range]
    y_lim = [y_center - y_range/2 - padding*y_range, y_center + y_range/2 + padding*y_range]
    
    print(f"Stable axis limits: X=[{x_lim[0]:.1f}, {x_lim[1]:.1f}], Y=[{y_lim[0]:.1f}, {y_lim[1]:.1f}]")
    
    # Show evolution phases
    phase_indices = [
        len(sim.position_history) // 4,
        len(sim.position_history) // 2,
        3 * len(sim.position_history) // 4,
        -1
    ]
    
    phase_titles = ['Early Phase', 'Mid Phase', 'Late Phase', 'Final State']
    
    for i, (phase_idx, title) in enumerate(zip(phase_indices, phase_titles)):
        if phase_idx == -1:
            phase_idx = len(sim.position_history) - 1
            
        ax = axes[0, i+1] if i < 2 else axes[1, i-1]
        pos = sim.position_history[phase_idx]
        
        ax.scatter(pos[1:N_per_cluster, 0], pos[1:N_per_cluster, 1], 
                  c='blue', alpha=0.8, s=50)
        ax.scatter(pos[N_per_cluster+1:, 0], pos[N_per_cluster+1:, 1], 
                  c='red', alpha=0.8, s=50)
        ax.scatter(pos[0, 0], pos[0, 1], c='darkblue', s=200, marker='*')
        ax.scatter(pos[N_per_cluster, 0], pos[N_per_cluster, 1], c='darkred', s=200, marker='*')
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{title} (t={sim.time_points[phase_idx]:.1f})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Energy conservation plot
    axes[1, 2].plot(sim.time_points, sim.energy_history, 'g-', linewidth=2, label='Total Energy')
    axes[1, 2].axhline(y=initial_energy, color='r', linestyle='--', alpha=0.7, label='Initial Energy')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Total Energy')
    axes[1, 2].set_title('Energy Conservation (Stable)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stable_barnes_hut_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create clean timeline
    print("Creating stable timeline visualization...")
    
    n_frames = 5
    frame_indices = np.linspace(0, len(sim.position_history) - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(1, n_frames, figsize=(20, 4))
    
    for i, frame_idx in enumerate(frame_indices):
        pos = sim.position_history[frame_idx]
        t = sim.time_points[frame_idx]
        
        axes[i].scatter(pos[1:N_per_cluster, 0], pos[1:N_per_cluster, 1], 
                       c='blue', alpha=0.8, s=80, label='Cluster 1' if i == 0 else "")
        axes[i].scatter(pos[N_per_cluster+1:, 0], pos[N_per_cluster+1:, 1], 
                       c='red', alpha=0.8, s=80, label='Cluster 2' if i == 0 else "")
        axes[i].scatter(pos[0, 0], pos[0, 1], c='darkblue', s=250, marker='*')
        axes[i].scatter(pos[N_per_cluster, 0], pos[N_per_cluster, 1], c='darkred', s=250, marker='*')
        
        axes[i].set_xlim(x_lim)
        axes[i].set_ylim(y_lim)
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_title(f't = {t:.1f}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
        
        if i == 0:
            axes[i].legend()
    
    plt.suptitle('Stable Barnes-Hut Cluster Interaction Timeline', fontsize=16)
    plt.tight_layout()
    plt.savefig('stable_barnes_hut_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final analysis
    print("\n" + "="*60)
    print("STABLE SIMULATION ANALYSIS")
    print("="*60)
    print(f"• Total particles: {N_total}")
    print(f"• Algorithm: Barnes-Hut (θ = {sim.theta})")
    print(f"• Gravitational constant: {sim.G}")
    print(f"• Simulation time: {sim.time_points[-1]:.1f} time units")
    print(f"• Time step: {dt}")
    print(f"• Energy conservation: {energy_conservation:.2e} relative error")
    print(f"• Computation time: {end_time - start_time:.1f} seconds")
    print(f"• Average time per step: {(end_time - start_time)/len(sim.time_points)*1000:.1f} ms")
    print(f"• Axis range: X={x_range:.1f}, Y={y_range:.1f}")
    
    if energy_conservation < 1e-3:
        print("✓ EXCELLENT energy conservation!")
        stability = "EXCELLENT"
    elif energy_conservation < 1e-2:
        print("✓ Very good energy conservation")
        stability = "VERY GOOD"
    elif energy_conservation < 1e-1:
        print("✓ Good energy conservation")
        stability = "GOOD"
    else:
        print("⚠ Poor energy conservation")
        stability = "POOR"
    
    print(f"• Overall stability: {stability}")
    print("="*60)
    
    return sim

if __name__ == "__main__":
    print("Stable Barnes-Hut N-Body Simulation")
    print("===================================")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run the stable simulation
    simulation = stable_barnes_hut_simulation()
    
    print("\n" + "="*50)
    print("STABILITY IMPROVEMENTS:")
    print("• Very small gravitational constant (G=0.01)")
    print("• Small particle masses (0.01 units)")
    print("• Tiny time step (dt=0.005)")
    print("• Proper Keplerian orbital velocities")
    print("• Outlier-resistant axis calculation")
    print("• Conservative collision parameters")
    print("• Enhanced energy monitoring")
    print("="*50)
