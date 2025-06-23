"""
N-Body Problem Examples
=======================

This file demonstrates how to use the two-body and three-body simulation classes
for custom physics simulations and educational demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from two_body_simulation import TwoBodyProblem
from three_body_simulation import ThreeBodyProblem

def example_custom_two_body():
    """
    Example: Create a custom two-body simulation with specific parameters.
    """
    print("Example 1: Custom Two-Body Orbit")
    print("-" * 40)
    
    # Create system with custom masses and gravitational constant
    system = TwoBodyProblem(m1=2.0, m2=1.0, G=1.5)
    
    # Set up a highly eccentric orbit
    system.set_initial_conditions(
        r1_init=[-1.0, 0], r2_init=[2.0, 0],
        v1_init=[0, -0.3], v2_init=[0, 0.6]
    )
    
    # Run simulation
    system.simulate((0, 15), num_points=1500)
    
    # Check energy conservation
    energy_drift, energies = system.check_conservation()
    print(f"Energy conservation: {energy_drift:.2e} relative drift")
    
    # Create plots
    fig = system.plot_orbits()
    plt.suptitle("Custom Two-Body System (Eccentric Orbit)")
    plt.show()
    
    return system

def example_figure_eight():
    """
    Example: Demonstrate the famous Figure-8 solution to the three-body problem.
    """
    print("\nExample 2: Three-Body Figure-8 Orbit")
    print("-" * 40)
    
    # Create three-body system with equal masses
    system = ThreeBodyProblem(m1=1.0, m2=1.0, m3=1.0, G=1.0)
    
    # Use the precise Chenciner-Montgomery initial conditions
    system.set_initial_conditions(
        r1_init=[-0.97000436, 0.24308753], 
        r2_init=[0.97000436, -0.24308753],
        r3_init=[0.0, 0.0],
        v1_init=[0.4662036850, 0.4323657300], 
        v2_init=[0.4662036850, 0.4323657300],
        v3_init=[-0.93240737, -0.86473146]
    )
    
    # Simulate one complete period
    system.simulate((0, 6.32), num_points=2000)
    
    # Analyze conservation laws
    energy_drift, energies = system.check_conservation()
    print(f"Energy conservation: {energy_drift:.2e} relative drift")
    print(f"Initial energy: {energies[0]:.6f}")
    print(f"Final energy: {energies[-1]:.6f}")
    
    # Create comprehensive plots
    fig = system.plot_orbits()
    plt.suptitle("Three-Body Figure-8 Orbit (Chenciner-Montgomery Solution)")
    plt.show()    # Create animation focused on the complete orbit with proper axis limits (2x speed)
    custom_limits = {'xlim': (-1.5, 1.5), 'ylim': (-1.0, 1.0)}
    fig_anim, anim = system.animate_orbits(interval=15, trail_length=150, custom_limits=custom_limits)
    plt.show()
    
    return system

def example_chaotic_dynamics():
    """
    Example: Demonstrate chaotic behavior by comparing slightly different initial conditions.
    """
    print("\nExample 3: Sensitivity to Initial Conditions (Chaos)")
    print("-" * 40)
    
    # Create two identical systems with tiny difference in initial conditions
    epsilon = 1e-6  # Tiny perturbation
    
    systems = []
    for i, perturbation in enumerate([0, epsilon]):
        system = ThreeBodyProblem(m1=1.0, m2=1.0, m3=1.0, G=1.0)
        system.set_initial_conditions(
            r1_init=[-1.2, 0], 
            r2_init=[1.2, 0],
            r3_init=[0, 1.5 + perturbation],  # Tiny difference here
            v1_init=[0.1, 0.3], 
            v2_init=[-0.1, 0.3],
            v3_init=[0, -0.6]
        )
        system.simulate((0, 8), num_points=1500)
        systems.append(system)
    
    # Compare the trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['b', 'r', 'g']
    for i, system in enumerate(systems):
        for j, body in enumerate(['body1', 'body2', 'body3']):
            x, y = system.positions[body]
            ax1.plot(x, y, colors[j], alpha=0.7 if i == 0 else 0.5, 
                    linewidth=2 if i == 0 else 1,
                    label=f'Body {j+1}' + ('' if i == 0 else ' (perturbed)'))
    
    ax1.set_title('Orbital Trajectories')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot separation between body 3 in both systems
    x1_orig, y1_orig = systems[0].positions['body3']
    x1_pert, y1_pert = systems[1].positions['body3']
    separation = np.sqrt((x1_pert - x1_orig)**2 + (y1_pert - y1_orig)**2)
    
    ax2.semilogy(systems[0].time_points, separation, 'purple', linewidth=2)
    ax2.set_title('Separation of Body 3 Trajectories')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Distance (log scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Chaotic Sensitivity (Initial perturbation: {epsilon:.0e})')
    plt.tight_layout()
    plt.show()
    
    print(f"Maximum separation reached: {np.max(separation):.6f}")
    print("This demonstrates the butterfly effect in three-body dynamics!")
    
    return systems

def example_energy_analysis():
    """
    Example: Detailed analysis of energy conservation in different scenarios.
    """
    print("\nExample 4: Energy Conservation Analysis")
    print("-" * 40)
    
    scenarios = [
        ("Two-Body Circular", lambda: TwoBodyProblem(1.0, 1.0, 1.0)),
        ("Three-Body Figure-8", lambda: ThreeBodyProblem(1.0, 1.0, 1.0, 1.0))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (name, create_system) in enumerate(scenarios):
        system = create_system()
        
        if "Two-Body" in name:
            # Circular orbit for two-body
            system.set_initial_conditions(
                r1_init=[-0.5, 0], r2_init=[0.5, 0],
                v1_init=[0, -0.5], v2_init=[0, 0.5]
            )
            system.simulate((0, 12), num_points=1500)
        else:
            # Figure-8 for three-body
            system.set_initial_conditions(
                r1_init=[-0.97000436, 0.24308753], 
                r2_init=[0.97000436, -0.24308753],
                r3_init=[0.0, 0.0],
                v1_init=[0.4662036850, 0.4323657300], 
                v2_init=[0.4662036850, 0.4323657300],
                v3_init=[-0.93240737, -0.86473146]
            )
            system.simulate((0, 6.32), num_points=1500)
        
        # Energy analysis
        energy_drift, energies = system.check_conservation()
        
        # Plot energy over time
        axes[i, 0].plot(system.time_points, energies, 'b-', linewidth=2)
        axes[i, 0].set_title(f'{name}: Total Energy')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Energy')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot energy drift
        energy_change = np.array(energies) - energies[0]
        axes[i, 1].plot(system.time_points, energy_change, 'r-', linewidth=2)
        axes[i, 1].set_title(f'{name}: Energy Drift')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('ΔE')
        axes[i, 1].grid(True, alpha=0.3)
        
        print(f"{name}: Energy drift = {energy_drift:.2e}")
    
    plt.tight_layout()
    plt.suptitle('Energy Conservation in N-Body Systems', y=1.02)
    plt.show()

def run_all_examples():
    """
    Run all examples in sequence.
    """
    print("N-Body Problem Examples")
    print("=" * 50)
    
    try:
        example_custom_two_body()
        example_figure_eight()
        example_chaotic_dynamics()
        example_energy_analysis()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("These examples demonstrate:")
        print("• Custom parameter setup")
        print("• Famous solutions (Figure-8)")
        print("• Chaotic dynamics and sensitivity")
        print("• Energy conservation analysis")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure both simulation files are in the same directory.")

if __name__ == "__main__":
    run_all_examples()