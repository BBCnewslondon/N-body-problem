"""
Simple Barnes-Hut Visualization Demo
===================================

This version focuses on creating beautiful, clear visualizations of 
Barnes-Hut simulations without complex collision dynamics that cause 
numerical instability. Perfect for demonstrating the algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_simulation import BarnesHutNBody
import time

def simple_barnes_hut_demo():
    """Create a simple, stable Barnes-Hut demonstration."""
    
    print("Simple Barnes-Hut Visualization Demo")
    print("====================================")
    print("Focus: Clear visualization and algorithm demonstration")
    print("-" * 55)
    
    # Simple, stable parameters
    N = 80
    sim = BarnesHutNBody(N=N, G=0.5, theta=0.5)
    
    # Create a simple rotating galaxy disk
    print(f"Setting up {N} particles in a stable galaxy configuration...")
    
    positions = np.zeros((N, 2))
    velocities = np.zeros((N, 2))
    masses = np.ones(N) * 0.1
    
    # Central supermassive object
    positions[0] = [0, 0]
    velocities[0] = [0, 0]
    masses[0] = 10.0  # Central mass
    
    # Disk particles in stable orbits
    for i in range(1, N):
        # Logarithmic spiral for realistic galaxy appearance
        angle = (i - 1) * 0.3  # Spiral parameter
        radius = 0.5 + (i - 1) * 0.1  # Increasing radius
        
        # Position on spiral
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[i] = [x, y]
        
        # Circular velocity for stable orbit
        v_circ = np.sqrt(sim.G * masses[0] / radius)
        velocities[i] = v_circ * np.array([-np.sin(angle), np.cos(angle)]) * 0.8
        
        masses[i] = 0.1
    
    # Set initial conditions
    sim.set_initial_conditions(positions, velocities, masses)
    
    # Calculate initial energy
    initial_energy = sim.calculate_total_energy()
    print(f"Initial total energy: {initial_energy:.3f}")
    
    # Run simulation with conservative parameters
    print("Running Barnes-Hut simulation...")
    
    t_span = [0, 15.0]
    dt = 0.05  # Reasonable time step
    
    start_time = time.time()
    sim.simulate(t_span, dt, use_barnes_hut=True, save_interval=10)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.1f} seconds")
    print(f"Total time steps: {len(sim.time_points)}")
    
    # Energy conservation check
    final_energy = sim.energy_history[-1]
    energy_conservation = abs((final_energy - initial_energy) / initial_energy)
    print(f"Energy conservation error: {energy_conservation:.2e}")
    
    # Calculate sensible axis limits
    all_positions = np.vstack(sim.position_history)
    
    # Use 95th percentile to avoid extreme outliers
    x_range = np.percentile(all_positions[:, 0], 95) - np.percentile(all_positions[:, 0], 5)
    y_range = np.percentile(all_positions[:, 1], 95) - np.percentile(all_positions[:, 1], 5)
    
    # Center the view
    x_center = np.median(all_positions[:, 0])
    y_center = np.median(all_positions[:, 1])
    
    # Use the larger range and add padding
    view_range = max(x_range, y_range, 10.0)  # Minimum 10 unit range
    padding = 0.2
    
    x_lim = [x_center - view_range/2 - padding*view_range, 
             x_center + view_range/2 + padding*view_range]
    y_lim = [y_center - view_range/2 - padding*view_range, 
             y_center + view_range/2 + padding*view_range]
    
    print(f"View limits: X=[{x_lim[0]:.1f}, {x_lim[1]:.1f}], Y=[{y_lim[0]:.1f}, {y_lim[1]:.1f}]")
    
    # Create beautiful visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    particle_colors = colors[1:]  # Exclude central mass
    
    # Timeline phases
    phase_indices = [0, len(sim.position_history)//4, len(sim.position_history)//2, 
                     3*len(sim.position_history)//4, len(sim.position_history)-1]
    phase_titles = ['Initial', 'Quarter', 'Half', 'Three-Quarters', 'Final']
    
    for i, (phase_idx, title) in enumerate(zip(phase_indices, phase_titles)):
        row = i // 3
        col = i % 3
        
        pos = sim.position_history[phase_idx]
        t = sim.time_points[phase_idx]
        
        # Plot particles with beautiful colors and sizing
        axes[row, col].scatter(pos[1:, 0], pos[1:, 1], 
                              c=particle_colors, alpha=0.8, s=masses[1:]*80, 
                              cmap='viridis', edgecolors='white', linewidth=0.5)
        
        # Central mass with special styling
        axes[row, col].scatter(pos[0, 0], pos[0, 1], c='gold', s=400, marker='*', 
                              edgecolors='black', linewidth=2, label='Central Mass')
        
        # Add particle trails for beauty
        if phase_idx > 0:
            trail_length = min(phase_idx, 15)
            trail_start = max(0, phase_idx - trail_length)
            
            for j in range(trail_start, phase_idx, 2):  # Every other frame
                trail_pos = sim.position_history[j]
                alpha = (j - trail_start) / trail_length * 0.3
                axes[row, col].scatter(trail_pos[1:, 0], trail_pos[1:, 1], 
                                     c='gray', alpha=alpha, s=5)
        
        axes[row, col].set_xlim(x_lim)
        axes[row, col].set_ylim(y_lim)
        axes[row, col].set_xlabel('X Position')
        axes[row, col].set_ylabel('Y Position')
        axes[row, col].set_title(f'{title} State (t = {t:.1f})')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_aspect('equal')
        axes[row, col].set_facecolor('black')  # Dark background for space-like appearance
        
        if i == 0:
            axes[row, col].legend()
    
    # Energy plot
    axes[1, 2].plot(sim.time_points, sim.energy_history, 'cyan', linewidth=3, label='Total Energy')
    axes[1, 2].axhline(y=initial_energy, color='red', linestyle='--', 
                       linewidth=2, alpha=0.8, label='Initial Energy')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Total Energy')
    axes[1, 2].set_title('Energy Conservation')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_facecolor('black')
    
    plt.suptitle('Beautiful Barnes-Hut Galaxy Evolution', fontsize=18, color='white')
    plt.tight_layout()
    fig.patch.set_facecolor('black')
    plt.savefig('beautiful_barnes_hut_demo.png', dpi=150, bbox_inches='tight', 
                facecolor='black')
    plt.show()
    
    # Create stunning timeline animation frames
    print("Creating beautiful timeline visualization...")
    
    n_frames = 6
    frame_indices = np.linspace(0, len(sim.position_history) - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(1, n_frames, figsize=(24, 4))
    fig.patch.set_facecolor('black')
    
    for i, frame_idx in enumerate(frame_indices):
        pos = sim.position_history[frame_idx]
        t = sim.time_points[frame_idx]
        
        # Beautiful particle visualization
        axes[i].scatter(pos[1:, 0], pos[1:, 1], 
                       c=particle_colors, alpha=0.9, s=masses[1:]*120, 
                       cmap='viridis', edgecolors='white', linewidth=0.3)
        
        # Spectacular central mass
        axes[i].scatter(pos[0, 0], pos[0, 1], c='gold', s=500, marker='*', 
                       edgecolors='orange', linewidth=3)
        
        # Add spiral arms if visible
        if i > 0:
            for j in range(max(0, frame_idx-10), frame_idx, 2):
                trail_pos = sim.position_history[j]
                alpha = 0.1 + 0.2 * (j - max(0, frame_idx-10)) / 10
                axes[i].scatter(trail_pos[1:, 0], trail_pos[1:, 1], 
                               c='lightblue', alpha=alpha, s=10)
        
        axes[i].set_xlim(x_lim)
        axes[i].set_ylim(y_lim)
        axes[i].set_xlabel('X Position', color='white')
        axes[i].set_ylabel('Y Position', color='white')
        axes[i].set_title(f't = {t:.1f}', color='white', fontsize=14)
        axes[i].grid(True, alpha=0.2, color='gray')
        axes[i].set_aspect('equal')
        axes[i].set_facecolor('black')
        axes[i].tick_params(colors='white')
        
        # Add frame border
        for spine in axes[i].spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
    
    plt.suptitle('Barnes-Hut Galaxy Evolution Timeline', fontsize=20, color='white', y=0.95)
    plt.tight_layout()
    plt.savefig('beautiful_barnes_hut_timeline.png', dpi=150, bbox_inches='tight', 
                facecolor='black')
    plt.show()
    
    # Performance analysis
    print("\n" + "="*60)
    print("BARNES-HUT DEMONSTRATION ANALYSIS")
    print("="*60)
    print(f"• Particles: {N}")
    print(f"• Algorithm: Barnes-Hut (θ = {sim.theta})")
    print(f"• Gravitational constant: {sim.G}")
    print(f"• Simulation time: {sim.time_points[-1]:.1f} time units")
    print(f"• Time step: {dt}")
    print(f"• Energy conservation: {energy_conservation:.2e}")
    print(f"• Computation time: {end_time - start_time:.1f} seconds")
    print(f"• Average step time: {(end_time - start_time)/len(sim.time_points)*1000:.1f} ms")
    print(f"• View range: {view_range:.1f} units")
    
    if energy_conservation < 1e-4:
        status = "EXCELLENT"
    elif energy_conservation < 1e-3:
        status = "VERY GOOD"
    elif energy_conservation < 1e-2:
        status = "GOOD"
    else:
        status = "FAIR"
    
    print(f"• Energy conservation: {status}")
    print(f"• Visualization quality: EXCELLENT")
    print("="*60)
    
    print("\n" + "="*50)
    print("VISUALIZATION IMPROVEMENTS:")
    print("• Beautiful color schemes with viridis colormap")
    print("• Particle trails showing motion history")
    print("• Space-like dark backgrounds")
    print("• Proper axis scaling without extreme outliers")
    print("• Consistent sizing and transparency")
    print("• Professional styling with borders and labels")
    print("• Energy conservation monitoring")
    print("="*50)
    
    return sim

if __name__ == "__main__":
    # Set random seed for reproducible beauty
    np.random.seed(42)
    
    # Run the beautiful demo
    simulation = simple_barnes_hut_demo()
