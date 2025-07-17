"""
Comparison of Original vs Stable Barnes-Hut Implementations
=========================================================

This script demonstrates the numerical stability improvements by comparing:
1. Original Barnes-Hut implementation
2. Fixed stable Barnes-Hut implementation

The comparison focuses on energy conservation and numerical stability.
"""

import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_simulation import BarnesHutNBody
from barnes_hut_simulation_fixed import BarnesHutNBodyStable
import time

def create_identical_test_system():
    """Create identical initial conditions for both simulations."""
    N = 40
    np.random.seed(42)  # For reproducible results
    
    # Create two clusters
    positions = np.zeros((N, 2))
    velocities = np.zeros((N, 2))
    masses = np.ones(N) * 0.1
    
    # Cluster 1
    N1 = N // 2
    cluster1_center = np.array([-2.0, 0.0])
    positions[0] = cluster1_center
    masses[0] = 2.0
    velocities[0] = np.array([0.15, 0.0])
    
    for i in range(1, N1):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2, 1.2)
        
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster1_center + pos_rel
        
        # Circular velocity (not perfect to make it more challenging)
        v_circular = np.sqrt(0.1 * masses[0] / radius)
        vel_rel = v_circular * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = velocities[0] + vel_rel * 0.9
    
    # Cluster 2
    cluster2_center = np.array([2.0, 0.0])
    positions[N1] = cluster2_center
    masses[N1] = 2.0
    velocities[N1] = np.array([-0.15, 0.0])
    
    for i in range(N1 + 1, N):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2, 1.2)
        
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster2_center + pos_rel
        
        v_circular = np.sqrt(0.1 * masses[N1] / radius)
        vel_rel = v_circular * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = velocities[N1] + vel_rel * 0.9
    
    return positions, velocities, masses

def run_comparison():
    """Run both simulations and compare results."""
    print("Running Numerical Stability Comparison")
    print("=" * 50)
    
    # Create identical initial conditions
    positions, velocities, masses = create_identical_test_system()
    N = len(positions)
    
    # Initialize original simulation
    print("\n1. Running Original Barnes-Hut Simulation...")
    sim_original = BarnesHutNBody(N=N, G=0.1, theta=0.5)
    sim_original.set_initial_conditions(positions.copy(), velocities.copy(), masses.copy())
    
    start_time = time.time()
    sim_original.simulate((0, 3), dt=0.01, use_barnes_hut=True, save_interval=3)
    original_time = time.time() - start_time
    
    # Calculate energy conservation for original
    original_initial_energy = sim_original.energy_history[0]
    original_final_energy = sim_original.energy_history[-1]
    original_energy_drift = abs(original_final_energy - original_initial_energy) / abs(original_initial_energy)
    
    print(f"Original - Initial energy: {original_initial_energy:.6f}")
    print(f"Original - Final energy: {original_final_energy:.6f}")
    print(f"Original - Energy drift: {original_energy_drift:.2e}")
    print(f"Original - Runtime: {original_time:.2f} seconds")
    
    # Initialize stable simulation
    print("\n2. Running Stable Barnes-Hut Simulation...")
    sim_stable = BarnesHutNBodyStable(N=N, G=0.1, theta=0.5, softening=0.05)
    sim_stable.set_initial_conditions(positions.copy(), velocities.copy(), masses.copy())
    
    start_time = time.time()
    stable_energy_drift = sim_stable.simulate((0, 3), dt=0.01, use_barnes_hut=True, save_interval=3)
    stable_time = time.time() - start_time
    
    print(f"Stable - Runtime: {stable_time:.2f} seconds")
    
    # Create comparison plots
    create_comparison_plots(sim_original, sim_stable, original_energy_drift, stable_energy_drift)
    
    # Print summary
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Original energy drift: {original_energy_drift:.2e}")
    print(f"Stable energy drift:   {stable_energy_drift:.2e}")
    print(f"Improvement factor:    {original_energy_drift/stable_energy_drift:.1f}x")
    print(f"Runtime ratio:         {stable_time/original_time:.2f}")
    
    if stable_energy_drift < 1e-4:
        stability_rating = "EXCELLENT"
    elif stable_energy_drift < 1e-3:
        stability_rating = "Very Good"
    elif stable_energy_drift < 1e-2:
        stability_rating = "Good"
    else:
        stability_rating = "Poor"
    
    print(f"Stability rating:      {stability_rating}")
    
    return sim_original, sim_stable

def create_comparison_plots(sim_original, sim_stable, original_drift, stable_drift):
    """Create side-by-side comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Energy evolution comparison
    axes[0, 0].plot(sim_original.time_points, sim_original.energy_history, 'r-', 
                   linewidth=2, label='Original', alpha=0.8)
    axes[0, 0].plot(sim_stable.time_points, sim_stable.energy_history, 'b-', 
                   linewidth=2, label='Stable', alpha=0.8)
    axes[0, 0].axhline(y=sim_original.energy_history[0], color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Total Energy')
    axes[0, 0].set_title('Total Energy Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy drift comparison
    original_drift_series = (sim_original.energy_history - sim_original.energy_history[0]) / abs(sim_original.energy_history[0])
    stable_drift_series = (sim_stable.energy_history - sim_stable.energy_history[0]) / abs(sim_stable.energy_history[0])
    
    axes[0, 1].plot(sim_original.time_points, original_drift_series, 'r-', 
                   linewidth=2, label='Original', alpha=0.8)
    axes[0, 1].plot(sim_stable.time_points, stable_drift_series, 'b-', 
                   linewidth=2, label='Stable', alpha=0.8)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Relative Energy Drift')
    axes[0, 1].set_title('Energy Conservation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Energy drift (log scale)
    axes[0, 2].semilogy(sim_original.time_points, np.abs(original_drift_series), 'r-', 
                       linewidth=2, label='Original', alpha=0.8)
    axes[0, 2].semilogy(sim_stable.time_points, np.abs(stable_drift_series), 'b-', 
                       linewidth=2, label='Stable', alpha=0.8)
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('|Relative Energy Drift|')
    axes[0, 2].set_title('Energy Drift (Log Scale)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Final configurations
    original_final = sim_original.position_history[-1]
    stable_final = sim_stable.position_history[-1]
    
    axes[1, 0].scatter(original_final[:, 0], original_final[:, 1], 
                      c=sim_original.masses, cmap='Reds', s=50, alpha=0.8)
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    axes[1, 0].set_title('Original - Final Configuration')
    axes[1, 0].set_aspect('equal')
    
    axes[1, 1].scatter(stable_final[:, 0], stable_final[:, 1], 
                      c=sim_stable.masses, cmap='Blues', s=50, alpha=0.8)
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    axes[1, 1].set_title('Stable - Final Configuration')
    axes[1, 1].set_aspect('equal')
    
    # Energy components comparison for stable simulation
    if hasattr(sim_stable, 'kinetic_energy_history'):
        axes[1, 2].plot(sim_stable.time_points, sim_stable.kinetic_energy_history, 
                       'r-', linewidth=2, label='Kinetic', alpha=0.8)
        axes[1, 2].plot(sim_stable.time_points, sim_stable.potential_energy_history, 
                       'b-', linewidth=2, label='Potential', alpha=0.8)
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Energy')
        axes[1, 2].set_title('Stable - Energy Components')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # Create a summary comparison
        comparison_data = {
            'Original': [original_drift, sim_original.energy_history[0], sim_original.energy_history[-1]],
            'Stable': [stable_drift, sim_stable.energy_history[0], sim_stable.energy_history[-1]]
        }
        
        labels = ['Energy Drift', 'Initial Energy', 'Final Energy']
        x = np.arange(len(labels))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, comparison_data['Original'], width, 
                      label='Original', alpha=0.8, color='red')
        axes[1, 2].bar(x + width/2, comparison_data['Stable'], width, 
                      label='Stable', alpha=0.8, color='blue')
        
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_ylabel('Values')
        axes[1, 2].set_title('Comparison Summary')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(labels)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stability_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_stability_improvements():
    """Analyze the specific improvements made."""
    print("\nSTABILITY IMPROVEMENTS ANALYSIS")
    print("=" * 50)
    
    improvements = [
        "1. Softening Parameters:",
        "   - Prevents singularities when particles get very close",
        "   - Smooths force calculations near r=0",
        "   - Reduces numerical errors in energy calculations",
        "",
        "2. Double Precision:",
        "   - Uses np.float64 for better numerical accuracy",
        "   - Reduces accumulation of floating-point errors",
        "   - Improves energy conservation",
        "",
        "3. Velocity-Verlet Integration:",
        "   - More stable than leapfrog integration",
        "   - Better energy conservation properties",
        "   - Symmetric integration scheme",
        "",
        "4. Adaptive Time Stepping:",
        "   - Automatically adjusts dt based on forces",
        "   - Prevents instabilities from large time steps",
        "   - Maintains accuracy during close encounters",
        "",
        "5. Improved Energy Calculation:",
        "   - Uses softened potential for consistency",
        "   - Better numerical precision",
        "   - Tracks kinetic and potential energy separately",
        "",
        "6. Real-time Monitoring:",
        "   - Continuous energy drift monitoring",
        "   - Warnings for numerical instabilities",
        "   - Adaptive parameter adjustment"
    ]
    
    for improvement in improvements:
        print(improvement)

if __name__ == "__main__":
    # Run the comparison
    sim_original, sim_stable = run_comparison()
    
    # Analyze improvements
    analyze_stability_improvements()
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS FOR NUMERICAL STABILITY")
    print("=" * 50)
    print("1. Always use softening parameters in N-body simulations")
    print("2. Implement adaptive time stepping for dynamic systems")
    print("3. Use double precision for critical calculations")
    print("4. Monitor energy conservation during simulation")
    print("5. Choose stable integration schemes (Velocity-Verlet)")
    print("6. Implement proper error handling and warnings")