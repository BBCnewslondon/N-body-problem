"""
Numerically Stable Barnes-Hut N-Body Simulation
===============================================

This version fixes numerical instability issues in energy calculations by:
1. Adding softening parameters to prevent singularities
2. Implementing adaptive time stepping
3. Using higher precision energy calculations
4. Adding energy conservation monitoring
5. Improved force calculation with proper softening
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
import time
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class QuadTreeStable:
    """
    Numerically stable quadtree implementation with softening parameters.
    """
    
    def __init__(self, center, size, max_depth=10, theta=0.5, softening=1e-3):
        """
        Initialize quadtree node with softening parameter.
        
        Parameters:
        center: (x, y) center of this quadrant
        size: size of this quadrant
        max_depth: maximum tree depth
        theta: Barnes-Hut accuracy parameter
        softening: softening parameter to prevent singularities
        """
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.max_depth = max_depth
        self.theta = theta
        self.softening = softening
        self.depth = 0
        
        # Node properties
        self.total_mass = 0.0
        self.center_of_mass = np.array([0.0, 0.0], dtype=np.float64)
        self.particles = []
        
        # Child nodes (NW, NE, SW, SE)
        self.children = [None, None, None, None]
        self.is_leaf = True
        
    def insert_particle(self, particle_idx, position, mass):
        """Insert a particle into the quadtree."""
        position = np.array(position, dtype=np.float64)
        
        # Update total mass and center of mass for this node
        if self.total_mass == 0:
            self.center_of_mass = position.copy()
            self.total_mass = mass
        else:
            # Weighted average for center of mass
            self.center_of_mass = (self.center_of_mass * self.total_mass + position * mass) / (self.total_mass + mass)
            self.total_mass += mass
        
        # If this is a leaf node and empty, just add the particle
        if self.is_leaf and len(self.particles) == 0:
            self.particles.append((particle_idx, position, mass))
            return
        
        # If this is a leaf with one particle, we need to subdivide
        if self.is_leaf and len(self.particles) == 1:
            if self.depth < self.max_depth:
                self._subdivide()
                # Re-insert existing particle
                old_particle = self.particles[0]
                self.particles = []
                self.is_leaf = False
                self._insert_into_child(old_particle[0], old_particle[1], old_particle[2])
                self._insert_into_child(particle_idx, position, mass)
            else:
                # Maximum depth reached, store multiple particles in leaf
                self.particles.append((particle_idx, position, mass))
        else:
            # Internal node, insert into appropriate child
            self._insert_into_child(particle_idx, position, mass)
    
    def _subdivide(self):
        """Create four child nodes."""
        half_size = self.size / 2
        quarter_size = half_size / 2
        
        child_centers = [
            [self.center[0] - quarter_size, self.center[1] + quarter_size],  # NW
            [self.center[0] + quarter_size, self.center[1] + quarter_size],  # NE
            [self.center[0] - quarter_size, self.center[1] - quarter_size],  # SW
            [self.center[0] + quarter_size, self.center[1] - quarter_size]   # SE
        ]
        
        for i in range(4):
            self.children[i] = QuadTreeStable(
                child_centers[i], half_size, self.max_depth, self.theta, self.softening
            )
            self.children[i].depth = self.depth + 1
    
    def _insert_into_child(self, particle_idx, position, mass):
        """Insert particle into appropriate child quadrant."""
        quadrant = 0
        if position[0] >= self.center[0]:
            quadrant += 1
        if position[1] < self.center[1]:
            quadrant += 2
        
        self.children[quadrant].insert_particle(particle_idx, position, mass)
    
    def calculate_force(self, particle_idx, position, mass, G=1.0):
        """
        Calculate gravitational force using softened potential.
        
        Parameters:
        particle_idx: index of the particle
        position: particle position
        mass: particle mass
        G: gravitational constant
        
        Returns:
        force: (fx, fy) gravitational force
        """
        force = np.array([0.0, 0.0], dtype=np.float64)
        
        # If this node has no mass, no force
        if self.total_mass == 0:
            return force
        
        # Distance to center of mass of this node
        r_vec = self.center_of_mass - position
        r_squared = np.dot(r_vec, r_vec)
        r = np.sqrt(r_squared)
        
        # Avoid self-interaction
        if r < self.softening:
            return force
        
        # Barnes-Hut criterion: if s/d < theta, treat as single particle
        if self.is_leaf or (self.size / r) < self.theta:
            # Check if this is a direct particle interaction (avoid self-force)
            if self.is_leaf and len(self.particles) == 1:
                if self.particles[0][0] == particle_idx:
                    return force  # Don't apply force to itself
            
            # Calculate force from center of mass with softening
            r_softened_squared = r_squared + self.softening**2
            r_softened = np.sqrt(r_softened_squared)
            
            force_magnitude = G * mass * self.total_mass / r_softened_squared
            force = force_magnitude * (r_vec / r_softened)
        else:
            # Recursively calculate forces from children
            for child in self.children:
                if child is not None:
                    force += child.calculate_force(particle_idx, position, mass, G)
        
        return force


class BarnesHutNBodyStable:
    """
    Numerically stable N-Body simulation using the Barnes-Hut algorithm.
    """
    
    def __init__(self, N, G=1.0, theta=0.5, softening=1e-3):
        """
        Initialize N-body system with stability parameters.
        
        Parameters:
        N: number of particles
        G: gravitational constant
        theta: Barnes-Hut accuracy parameter
        softening: softening parameter for force calculations
        """
        self.N = N
        self.G = G
        self.theta = theta
        self.softening = softening
        
        # Use double precision for better numerical stability
        self.positions = np.zeros((N, 2), dtype=np.float64)
        self.velocities = np.zeros((N, 2), dtype=np.float64)
        self.masses = np.ones(N, dtype=np.float64)
        
        # Simulation data
        self.time_points = []
        self.position_history = []
        self.energy_history = []
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        
        # Adaptive time stepping parameters
        self.adaptive_dt = True
        self.dt_min = 1e-6
        self.dt_max = 0.1
        self.energy_tolerance = 1e-3
        
    def set_initial_conditions(self, positions, velocities, masses):
        """Set initial conditions for all particles."""
        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)
        self.masses = np.array(masses, dtype=np.float64)
    
    def build_tree(self):
        """Build quadtree for current particle positions."""
        # Find bounding box
        min_pos = np.min(self.positions, axis=0)
        max_pos = np.max(self.positions, axis=0)
        
        # Create square bounding box with some padding
        center = (min_pos + max_pos) / 2
        size = np.max(max_pos - min_pos) * 1.2
        
        # Build tree
        root = QuadTreeStable(center, size, theta=self.theta, softening=self.softening)
        
        for i in range(self.N):
            root.insert_particle(i, self.positions[i], self.masses[i])
        
        return root
    
    def calculate_forces_barnes_hut(self):
        """Calculate forces using Barnes-Hut algorithm with softening."""
        tree = self.build_tree()
        forces = np.zeros((self.N, 2), dtype=np.float64)
        
        for i in range(self.N):
            forces[i] = tree.calculate_force(i, self.positions[i], self.masses[i], self.G)
        
        return forces
    
    def calculate_forces_direct(self):
        """Calculate forces using direct method with softening."""
        forces = np.zeros((self.N, 2), dtype=np.float64)
        
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_vec = self.positions[j] - self.positions[i]
                    r_squared = np.dot(r_vec, r_vec)
                    
                    # Apply softening
                    r_softened_squared = r_squared + self.softening**2
                    r_softened = np.sqrt(r_softened_squared)
                    
                    force_magnitude = self.G * self.masses[i] * self.masses[j] / r_softened_squared
                    forces[i] += force_magnitude * (r_vec / r_softened)
        
        return forces
    
    def calculate_total_energy(self):
        """Calculate total energy with improved precision."""
        # Kinetic energy
        kinetic = 0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1))
        
        # Potential energy with softening
        potential = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                r_vec = self.positions[j] - self.positions[i]
                r_squared = np.dot(r_vec, r_vec)
                r_softened = np.sqrt(r_squared + self.softening**2)
                
                potential -= self.G * self.masses[i] * self.masses[j] / r_softened
        
        return kinetic + potential, kinetic, potential
    
    def adaptive_time_step(self, forces, current_dt):
        """Calculate adaptive time step based on forces and energy conservation."""
        if not self.adaptive_dt:
            return current_dt
        
        # Calculate maximum acceleration
        accelerations = forces / self.masses[:, np.newaxis]
        max_acceleration = np.max(np.sqrt(np.sum(accelerations**2, axis=1)))
        
        # Time step based on acceleration (Courant condition)
        if max_acceleration > 0:
            dt_acc = np.sqrt(self.softening / max_acceleration) * 0.1
        else:
            dt_acc = current_dt
        
        # Limit time step
        dt_new = np.clip(dt_acc, self.dt_min, self.dt_max)
        
        return dt_new
    
    def step(self, dt, use_barnes_hut=True):
        """
        Advance simulation by one time step using velocity-Verlet integration.
        
        Parameters:
        dt: time step
        use_barnes_hut: if True, use Barnes-Hut; if False, use direct method
        """
        # Calculate initial forces
        if use_barnes_hut:
            forces = self.calculate_forces_barnes_hut()
        else:
            forces = self.calculate_forces_direct()
        
        # Adaptive time stepping
        if self.adaptive_dt:
            dt = self.adaptive_time_step(forces, dt)
        
        # Velocity-Verlet integration (more stable than leapfrog)
        accelerations = forces / self.masses[:, np.newaxis]
        
        # Update positions
        self.positions += self.velocities * dt + 0.5 * accelerations * dt**2
        
        # Calculate forces at new positions
        if use_barnes_hut:
            forces_new = self.calculate_forces_barnes_hut()
        else:
            forces_new = self.calculate_forces_direct()
        
        accelerations_new = forces_new / self.masses[:, np.newaxis]
        
        # Update velocities
        self.velocities += 0.5 * (accelerations + accelerations_new) * dt
        
        return dt
    
    def simulate(self, t_span, dt=0.01, use_barnes_hut=True, save_interval=1):
        """
        Run the N-body simulation with energy monitoring.
        
        Parameters:
        t_span: (t_start, t_end) simulation time range
        dt: initial time step
        use_barnes_hut: whether to use Barnes-Hut algorithm
        save_interval: save data every N steps
        """
        t_start, t_end = t_span
        t = t_start
        step_count = 0
        
        # Initialize history
        self.time_points = [t]
        self.position_history = [self.positions.copy()]
        
        total_energy, kinetic, potential = self.calculate_total_energy()
        self.energy_history = [total_energy]
        self.kinetic_energy_history = [kinetic]
        self.potential_energy_history = [potential]
        
        initial_energy = total_energy
        
        print(f"Starting stable simulation with {self.N} particles...")
        print(f"Method: {'Barnes-Hut' if use_barnes_hut else 'Direct O(N²)'}")
        print(f"Softening parameter: {self.softening}")
        print(f"Initial total energy: {initial_energy:.6f}")
        
        start_time = time.time()
        
        while t < t_end:
            # Advance simulation
            actual_dt = self.step(dt, use_barnes_hut)
            t += actual_dt
            step_count += 1
            
            # Save data periodically
            if step_count % save_interval == 0:
                self.time_points.append(t)
                self.position_history.append(self.positions.copy())
                
                total_energy, kinetic, potential = self.calculate_total_energy()
                self.energy_history.append(total_energy)
                self.kinetic_energy_history.append(kinetic)
                self.potential_energy_history.append(potential)
                
                # Energy conservation check
                energy_drift = abs(total_energy - initial_energy) / abs(initial_energy)
                if energy_drift > self.energy_tolerance:
                    print(f"Warning: Energy drift at t={t:.3f}: {energy_drift:.2e}")
            
            # Progress update
            if step_count % 100 == 0:
                progress = (t - t_start) / (t_end - t_start) * 100
                elapsed = time.time() - start_time
                energy_drift = abs(self.energy_history[-1] - initial_energy) / abs(initial_energy)
                print(f"Progress: {progress:.1f}% (t={t:.2f}, dt={actual_dt:.2e}, energy_drift={energy_drift:.2e})")
        
        elapsed_time = time.time() - start_time
        final_energy = self.energy_history[-1]
        energy_conservation = abs(final_energy - initial_energy) / abs(initial_energy)
        
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {elapsed_time/step_count*1000:.2f} ms")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Energy conservation: {energy_conservation:.2e}")
        
        # Convert to numpy arrays
        self.time_points = np.array(self.time_points)
        self.position_history = np.array(self.position_history)
        self.energy_history = np.array(self.energy_history)
        self.kinetic_energy_history = np.array(self.kinetic_energy_history)
        self.potential_energy_history = np.array(self.potential_energy_history)
        
        return energy_conservation
    
    def plot_energy_analysis(self, figsize=(15, 10)):
        """Create comprehensive energy analysis plots."""
        if len(self.energy_history) == 0:
            raise ValueError("Run simulation first!")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Total energy
        axes[0, 0].plot(self.time_points, self.energy_history, 'b-', linewidth=2)
        axes[0, 0].axhline(y=self.energy_history[0], color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Total Energy')
        axes[0, 0].set_title('Total Energy Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy components
        axes[0, 1].plot(self.time_points, self.kinetic_energy_history, 'r-', linewidth=2, label='Kinetic')
        axes[0, 1].plot(self.time_points, self.potential_energy_history, 'b-', linewidth=2, label='Potential')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Energy')
        axes[0, 1].set_title('Energy Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy drift
        energy_drift = (self.energy_history - self.energy_history[0]) / abs(self.energy_history[0])
        axes[0, 2].plot(self.time_points, energy_drift, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Relative Energy Drift')
        axes[0, 2].set_title('Energy Conservation')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Current configuration
        current_pos = self.position_history[-1]
        scatter = axes[1, 0].scatter(current_pos[:, 0], current_pos[:, 1], 
                                   c=self.masses, cmap='viridis', s=50, alpha=0.8)
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Y Position')
        axes[1, 0].set_title('Final Configuration')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[1, 0], label='Mass')
        
        # Energy drift histogram
        if len(energy_drift) > 1:
            axes[1, 1].hist(energy_drift, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Relative Energy Drift')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Energy Drift Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Energy drift vs time (log scale)
        axes[1, 2].semilogy(self.time_points, np.abs(energy_drift), 'r-', linewidth=2)
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('|Relative Energy Drift|')
        axes[1, 2].set_title('Energy Drift (Log Scale)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def create_stable_test_system():
    """Create a stable test system to demonstrate the fixes."""
    N = 50
    sim = BarnesHutNBodyStable(N=N, G=0.1, theta=0.5, softening=0.05)
    
    # Create two clusters
    positions = np.zeros((N, 2))
    velocities = np.zeros((N, 2))
    masses = np.ones(N) * 0.1
    
    # Cluster 1
    N1 = N // 2
    cluster1_center = np.array([-2.0, 0.0])
    positions[0] = cluster1_center
    masses[0] = 2.0
    velocities[0] = np.array([0.1, 0.0])
    
    for i in range(1, N1):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, 1.0)
        
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster1_center + pos_rel
        
        # Stable circular velocity
        v_circular = np.sqrt(sim.G * masses[0] / radius)
        vel_rel = v_circular * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = velocities[0] + vel_rel * 0.8
    
    # Cluster 2
    cluster2_center = np.array([2.0, 0.0])
    positions[N1] = cluster2_center
    masses[N1] = 2.0
    velocities[N1] = np.array([-0.1, 0.0])
    
    for i in range(N1 + 1, N):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, 1.0)
        
        pos_rel = radius * np.array([np.cos(angle), np.sin(angle)])
        positions[i] = cluster2_center + pos_rel
        
        v_circular = np.sqrt(sim.G * masses[N1] / radius)
        vel_rel = v_circular * np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = velocities[N1] + vel_rel * 0.8
    
    sim.set_initial_conditions(positions, velocities, masses)
    return sim


if __name__ == "__main__":
    print("Testing Numerically Stable Barnes-Hut Simulation")
    print("=" * 50)
    
    # Create and run stable simulation
    sim = create_stable_test_system()
    
    # Run simulation
    energy_conservation = sim.simulate((0, 5), dt=0.02, use_barnes_hut=True, save_interval=5)
    
    # Create energy analysis plots
    sim.plot_energy_analysis()
    
    print(f"\nFinal energy conservation: {energy_conservation:.2e}")
    if energy_conservation < 1e-4:
        print("✓ EXCELLENT energy conservation!")
    elif energy_conservation < 1e-3:
        print("✓ Very good energy conservation")
    elif energy_conservation < 1e-2:
        print("✓ Good energy conservation")
    else:
        print("⚠ Energy conservation could be improved")