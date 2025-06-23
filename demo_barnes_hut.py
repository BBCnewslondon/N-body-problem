"""
Simple Barnes-Hut Algorithm Demonstration
=========================================

This script shows how to use the Barnes-Hut algorithm for efficient N-body simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_simulation import BarnesHutNBody

def demo_galaxy_formation():
    """Demonstrate galaxy formation with 200 particles."""
    print("Demo: Galaxy Formation with Barnes-Hut Algorithm")
    print("=" * 50)
    
    # Create system with 200 particles
    N = 200
    sim = BarnesHutNBody(N, G=1.0, theta=0.5)
      # Initialize as galaxy disk
    print(f"Setting up {N} particles in galaxy disk configuration...")
    sim.create_galaxy_disk(center=(0, 0), radius=8.0, central_mass=50.0)    # Show initial configuration
    plt.figure(figsize=(12, 5))
    
    # Calculate proper axis limits
    all_pos = np.vstack([sim.positions, sim.position_history[-1]])
    x_min, x_max = np.min(all_pos[:, 0]), np.max(all_pos[:, 0])
    y_min, y_max = np.min(all_pos[:, 1]), np.max(all_pos[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    x_lim = [x_min - padding * x_range, x_max + padding * x_range]
    y_lim = [y_min - padding * y_range, y_max + padding * y_range]
    
    plt.subplot(1, 2, 1)
    plt.scatter(sim.positions[:, 0], sim.positions[:, 1], s=sim.masses*5, alpha=0.7, c='blue')
    plt.scatter(sim.positions[0, 0], sim.positions[0, 1], s=100, c='red', marker='*', 
                label='Central Object')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Initial Galaxy Configuration')
    plt.legend()
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    
    # Run simulation
    print("Running Barnes-Hut simulation...")
    t_span = [0, 3.0]
    dt = 0.02
    
    start_time = time.time()
    sim.simulate(t_span, dt, use_barnes_hut=True, save_interval=5)
    simulation_time = time.time() - start_time
    
    print(f"Simulation completed in {simulation_time:.2f} seconds")
    print(f"Simulated {len(sim.time_points)} time steps")
    print(f"Average: {simulation_time/len(sim.time_points)*1000:.1f} ms per step")
      # Show final configuration
    plt.subplot(1, 2, 2)
    final_positions = sim.position_history[-1]
    plt.scatter(final_positions[:, 0], final_positions[:, 1], 
                s=sim.masses*5, alpha=0.7, c='blue')
    plt.scatter(final_positions[0, 0], final_positions[0, 1], 
                s=100, c='red', marker='*', label='Central Object')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Final Configuration (t={t_span[1]})')
    plt.legend()
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('barnes_hut_galaxy_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Energy conservation analysis
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    energy_history = np.array(sim.energy_history)
    energy_drift = (energy_history - energy_history[0]) / abs(energy_history[0])
    plt.plot(sim.time_points, energy_drift)
    plt.xlabel('Time')
    plt.ylabel('Relative Energy Drift')
    plt.title('Energy Conservation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sim.time_points, energy_history)
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Total System Energy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('barnes_hut_energy_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Final energy drift: {energy_drift[-1]:.2e}")
    return sim, sim.time_points, sim.position_history

def demo_accuracy_comparison():
    """Compare Barnes-Hut accuracy for different theta values."""
    print("\nDemo: Barnes-Hut Accuracy Comparison")
    print("=" * 40)
    
    # Create small test system for accuracy comparison
    N = 50
    theta_values = [0.1, 0.5, 1.0, 2.0]
    
    # Reference system (direct calculation)
    ref_sim = BarnesHutNBody(N, G=1.0, theta=0.5)
    ref_sim.create_galaxy_disk(center=(0, 0), radius=3.0, central_mass=10.0)
    ref_forces = ref_sim.calculate_forces_direct()
    
    plt.figure(figsize=(12, 8))
    
    for i, theta in enumerate(theta_values):        # Calculate forces with Barnes-Hut for this theta
        bh_sim = BarnesHutNBody(N, G=1.0, theta=theta)
        bh_sim.positions = ref_sim.positions.copy()
        bh_sim.velocities = ref_sim.velocities.copy()
        bh_sim.masses = ref_sim.masses.copy()
        
        start_time = time.time()
        bh_forces = bh_sim.calculate_forces_barnes_hut()
        calc_time = time.time() - start_time
        
        # Calculate force error
        force_error = np.linalg.norm(bh_forces - ref_forces, axis=1)
        mean_error = np.mean(force_error)
        max_error = np.max(force_error)
        
        print(f"θ = {theta:3.1f}: Mean error = {mean_error:.2e}, "
              f"Max error = {max_error:.2e}, Time = {calc_time*1000:.1f} ms")
        
        # Plot force comparison
        plt.subplot(2, 2, i+1)
        plt.scatter(range(N), force_error, alpha=0.7)
        plt.xlabel('Particle Index')
        plt.ylabel('Force Error')
        plt.title(f'θ = {theta}, Mean Error = {mean_error:.2e}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('barnes_hut_accuracy_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

def demo_performance_scaling():
    """Demonstrate performance scaling with particle number."""
    print("\nDemo: Performance Scaling")
    print("=" * 30)
    
    N_values = [25, 50, 100, 200, 400]
    direct_times = []
    barnes_hut_times = []
    
    for N in N_values:
        print(f"Testing N = {N} particles...")
        
        # Create test system
        sim = BarnesHutNBody(N, G=1.0, theta=0.5)
        sim.create_galaxy_disk(center=(0, 0), radius=2.0, central_mass=5.0)
        
        # Time direct method
        start_time = time.time()
        sim.calculate_forces_direct()
        direct_time = time.time() - start_time
        direct_times.append(direct_time)
        
        # Time Barnes-Hut method
        start_time = time.time()
        sim.calculate_forces_barnes_hut()
        bh_time = time.time() - start_time
        barnes_hut_times.append(bh_time)
        
        speedup = direct_time / bh_time if bh_time > 0 else 1.0
        print(f"  Direct: {direct_time*1000:.1f} ms, "
              f"Barnes-Hut: {bh_time*1000:.1f} ms, "
              f"Speedup: {speedup:.1f}x")
    
    # Plot scaling comparison
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.loglog(N_values, direct_times, 'o-', label='Direct O(N²)', linewidth=2)
    plt.loglog(N_values, barnes_hut_times, 's-', label='Barnes-Hut O(N log N)', linewidth=2)
    
    # Add theoretical scaling lines
    N_theory = np.array(N_values)
    plt.loglog(N_theory, direct_times[0] * (N_theory/N_values[0])**2, 
               '--', alpha=0.5, label='Theoretical N²')
    plt.loglog(N_theory, barnes_hut_times[0] * (N_theory/N_values[0]) * np.log2(N_theory/N_values[0]), 
               '--', alpha=0.5, label='Theoretical N log N')
    
    plt.xlabel('Number of Particles (N)')
    plt.ylabel('Computation Time (s)')
    plt.title('Performance Scaling Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    speedups = [d/b if b > 0 else 1.0 for d, b in zip(direct_times, barnes_hut_times)]
    plt.semilogx(N_values, speedups, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Particles (N)')
    plt.ylabel('Speedup Factor')
    plt.title('Barnes-Hut Speedup vs Direct Method')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('barnes_hut_performance_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import time
    
    print("Barnes-Hut Algorithm Demonstration")
    print("==================================")
    print()
    
    # Run demonstrations
    sim, t_eval, r_history = demo_galaxy_formation()
    demo_accuracy_comparison()
    demo_performance_scaling()
    
    print("\n" + "="*50)
    print("Barnes-Hut Algorithm Key Benefits:")
    print("• Reduces O(N²) complexity to O(N log N)")
    print("• Enables simulations with thousands of particles")
    print("• Tunable accuracy with theta parameter")
    print("• Essential for large-scale astrophysical simulations")
    print("• Maintains reasonable physics accuracy")
    print("="*50)
