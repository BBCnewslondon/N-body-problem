"""
Barnes-Hut Algorithm Examples
============================

This file demonstrates various applications of the Barnes-Hut algorithm
for efficient N-body simulations, showing different configurations and
use cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_simulation import BarnesHutNBody, benchmark_comparison

def example_galaxy_simulation():
    """
    Example 1: Galaxy disk simulation with central supermassive black hole.
    """
    print("Example 1: Galaxy Disk Simulation")
    print("-" * 40)
    
    # Create galaxy with 150 particles
    N = 150
    system = BarnesHutNBody(N, G=1.0, theta=0.5)
    system.create_galaxy_disk(center=(0, 0), radius=5.0, central_mass=100.0)
    
    print(f"Simulating {N} particles forming a galaxy disk...")
    print("Central supermassive object mass: 100.0")
    print("Disk radius: 5.0 units")
    
    # Run simulation
    system.simulate((0, 3), dt=0.01, use_barnes_hut=True, save_interval=3)
    
    # Show energy conservation
    initial_energy = system.energy_history[0]
    final_energy = system.energy_history[-1]
    energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
    print(f"Energy conservation: {energy_drift:.2e} relative drift")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Initial configuration
    initial_pos = system.position_history[0]
    ax1.scatter(initial_pos[:, 0], initial_pos[:, 1], c=system.masses, 
                s=30, cmap='viridis', alpha=0.7)
    ax1.scatter(initial_pos[0, 0], initial_pos[0, 1], c='red', s=200, 
                marker='*', label='Central Black Hole')
    ax1.set_title('Initial Galaxy Configuration')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Final configuration
    final_pos = system.position_history[-1]
    ax2.scatter(final_pos[:, 0], final_pos[:, 1], c=system.masses, 
                s=30, cmap='viridis', alpha=0.7)
    ax2.scatter(final_pos[0, 0], final_pos[0, 1], c='red', s=200, 
                marker='*', label='Central Black Hole')
    ax2.set_title('Final Galaxy Configuration')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Energy evolution
    ax3.plot(system.time_points, system.energy_history, 'b-', linewidth=2)
    ax3.set_title('Energy Conservation')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Total Energy')
    ax3.grid(True, alpha=0.3)
    
    # Radial distribution evolution
    center = system.position_history[0][0]  # Central black hole position
    initial_radii = np.linalg.norm(initial_pos[1:] - center, axis=1)
    final_radii = np.linalg.norm(final_pos[1:] - center, axis=1)
    
    ax4.hist(initial_radii, bins=20, alpha=0.5, label='Initial', density=True)
    ax4.hist(final_radii, bins=20, alpha=0.5, label='Final', density=True)
    ax4.set_title('Radial Distribution')
    ax4.set_xlabel('Distance from Center')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create animation
    print("Creating animation...")
    fig_anim, anim = system.animate(interval=25, trail_length=20)
    plt.show()
    
    return system

def example_cluster_collision():
    """
    Example 2: Two colliding star clusters.
    """
    print("\nExample 2: Cluster Collision")
    print("-" * 40)
    
    # Create two clusters
    N = 100  # 50 particles per cluster
    system = BarnesHutNBody(N, G=1.0, theta=0.7)  # Slightly less accurate for speed
    
    # Initialize positions and velocities manually for two clusters
    positions = np.zeros((N, 2))
    velocities = np.zeros((N, 2))
    masses = np.ones(N)
    
    # Cluster 1: left side, moving right
    for i in range(N//2):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, 1.5)
        positions[i] = [-4 + radius*np.cos(angle), radius*np.sin(angle)]
        velocities[i] = [1.0 + np.random.normal(0, 0.1), np.random.normal(0, 0.1)]
        masses[i] = 1.0
    
    # Cluster 2: right side, moving left
    for i in range(N//2, N):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, 1.5)
        positions[i] = [4 + radius*np.cos(angle), radius*np.sin(angle)]
        velocities[i] = [-1.0 + np.random.normal(0, 0.1), np.random.normal(0, 0.1)]
        masses[i] = 1.0
    
    system.set_initial_conditions(positions, velocities, masses)
    
    print(f"Simulating collision of two {N//2}-particle clusters...")
    
    # Run simulation
    system.simulate((0, 8), dt=0.02, use_barnes_hut=True, save_interval=2)
    
    # Create animation focused on collision
    print("Creating collision animation...")
    fig_anim, anim = system.animate(interval=30, trail_length=25, 
                                   xlim=(-8, 8), ylim=(-6, 6))
    plt.show()
    
    return system

def example_accuracy_comparison():
    """
    Example 3: Compare different theta values for accuracy vs speed.
    """
    print("\nExample 3: Accuracy vs Speed Comparison")
    print("-" * 40)
    
    # Create reference system with direct calculation
    N = 50
    reference_system = BarnesHutNBody(N, G=1.0)
    reference_system.create_cluster(radius=2.0, velocity_dispersion=0.3)
    
    # Store initial conditions
    initial_pos = reference_system.positions.copy()
    initial_vel = reference_system.velocities.copy()
    initial_mass = reference_system.masses.copy()
    
    # Test different theta values
    theta_values = [0.1, 0.5, 1.0, 2.0]
    colors = ['red', 'blue', 'green', 'orange']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    print("Computing force accuracy for different theta values...")
    
    # Calculate reference forces (direct method)
    reference_system.positions = initial_pos.copy()
    reference_forces = reference_system.calculate_forces_direct()
    
    force_errors = []
    computation_times = []
    
    for i, theta in enumerate(theta_values):
        print(f"Testing theta = {theta}...")
        
        # Create system with specific theta
        system = BarnesHutNBody(N, G=1.0, theta=theta)
        system.set_initial_conditions(initial_pos, initial_vel, initial_mass)
        
        # Time force calculation
        import time
        start_time = time.time()
        bh_forces = system.calculate_forces_barnes_hut()
        computation_time = time.time() - start_time
        
        # Calculate force error
        force_error = np.mean(np.linalg.norm(bh_forces - reference_forces, axis=1))
        force_errors.append(force_error)
        computation_times.append(computation_time * 1000)  # Convert to ms
        
        # Plot force comparison for one particle
        particle_idx = 0
        ax1.quiver(initial_pos[particle_idx, 0], initial_pos[particle_idx, 1],
                  bh_forces[particle_idx, 0], bh_forces[particle_idx, 1],
                  color=colors[i], label=f'θ={theta}', scale=10)
    
    # Plot reference force
    ax1.quiver(initial_pos[0, 0], initial_pos[0, 1],
              reference_forces[0, 0], reference_forces[0, 1],
              color='black', label='Direct (exact)', scale=10, width=0.005)
    
    ax1.scatter(initial_pos[:, 0], initial_pos[:, 1], c='gray', s=20, alpha=0.5)
    ax1.set_title('Force Vectors on Particle 0')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error vs theta
    ax2.semilogy(theta_values, force_errors, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Force Error vs Theta')
    ax2.set_xlabel('Theta Parameter')
    ax2.set_ylabel('Average Force Error')
    ax2.grid(True, alpha=0.3)
    
    # Computation time vs theta
    ax3.plot(theta_values, computation_times, 'bo-', linewidth=2, markersize=8)
    ax3.set_title('Computation Time vs Theta')
    ax3.set_xlabel('Theta Parameter')
    ax3.set_ylabel('Time (ms)')
    ax3.grid(True, alpha=0.3)
    
    # Trade-off plot
    ax4.loglog(force_errors, computation_times, 'go-', linewidth=2, markersize=8)
    for i, theta in enumerate(theta_values):
        ax4.annotate(f'θ={theta}', (force_errors[i], computation_times[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_title('Accuracy vs Speed Trade-off')
    ax4.set_xlabel('Force Error')
    ax4.set_ylabel('Computation Time (ms)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nSummary:")
    for i, theta in enumerate(theta_values):
        print(f"θ={theta:3.1f}: Error={force_errors[i]:.2e}, Time={computation_times[i]:.2f}ms")

def example_large_scale_simulation():
    """
    Example 4: Large-scale simulation demonstrating Barnes-Hut efficiency.
    """
    print("\nExample 4: Large-Scale Simulation (500+ particles)")
    print("-" * 40)
    
    # Create large system
    N = 500
    system = BarnesHutNBody(N, G=0.1, theta=0.8)  # Reduced G for stability
    
    # Create multiple clusters
    cluster_centers = [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, 0)]
    cluster_masses = [50, 40, 60, 45, 80]
    
    positions = []
    velocities = []
    masses = []
    
    particles_per_cluster = N // len(cluster_centers)
    
    for i, (center, cluster_mass) in enumerate(zip(cluster_centers, cluster_masses)):
        # Add central massive object
        positions.append(center)
        velocities.append([0, 0])
        masses.append(cluster_mass)
        
        # Add surrounding particles
        for j in range(particles_per_cluster - 1):
            # Random position around cluster center
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0.5, 1.5)
            pos = [center[0] + radius*np.cos(angle), 
                   center[1] + radius*np.sin(angle)]
            positions.append(pos)
            
            # Orbital velocity around cluster center
            v_mag = np.sqrt(cluster_mass * 0.1 / radius) * 0.5
            vel = [-v_mag*np.sin(angle), v_mag*np.cos(angle)]
            # Add small random motion towards other clusters
            other_centers = [c for k, c in enumerate(cluster_centers) if k != i]
            if other_centers:
                target = other_centers[np.random.randint(len(other_centers))]
                direction = np.array(target) - np.array(center)
                direction = direction / np.linalg.norm(direction)
                vel += 0.1 * direction
            velocities.append(vel)
            masses.append(1.0)
    
    # Fill remaining particles randomly
    while len(positions) < N:
        positions.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        velocities.append([np.random.normal(0, 0.1), np.random.normal(0, 0.1)])
        masses.append(0.5)
    
    system.set_initial_conditions(np.array(positions[:N]), 
                                 np.array(velocities[:N]), 
                                 np.array(masses[:N]))
    
    print(f"Simulating {N} particles in multiple interacting clusters...")
    print("This would be very slow with direct O(N²) method!")
    
    # Run simulation
    system.simulate((0, 5), dt=0.01, use_barnes_hut=True, save_interval=5)
    
    # Create animation
    print("Creating large-scale animation...")
    fig_anim, anim = system.animate(interval=40, trail_length=10, 
                                   xlim=(-8, 8), ylim=(-8, 8))
    plt.show()
    
    return system

def run_all_examples():
    """
    Run all Barnes-Hut examples.
    """
    print("Barnes-Hut N-Body Algorithm Examples")
    print("=" * 50)
    
    try:
        # Run examples
        galaxy_system = example_galaxy_simulation()
        collision_system = example_cluster_collision()
        example_accuracy_comparison()
        large_system = example_large_scale_simulation()
        
        # Run performance benchmark
        print("\nPerformance Benchmark")
        print("-" * 40)
        benchmark_results = benchmark_comparison([25, 50, 100, 200])
        
        print("\n" + "=" * 50)
        print("All Barnes-Hut examples completed!")
        print("\nKey advantages of Barnes-Hut algorithm:")
        print("• Reduces complexity from O(N²) to O(N log N)")
        print("• Enables simulation of thousands of particles")
        print("• Tunable accuracy with theta parameter")
        print("• Maintains reasonable accuracy for most applications")
        print("• Essential for large-scale astrophysical simulations")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples()
