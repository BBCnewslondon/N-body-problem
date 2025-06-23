"""
Barnes-Hut N-Body Simulation
============================

Implementation of the Barnes-Hut algorithm for efficient N-body gravitational simulations.
This algorithm reduces computational complexity from O(N²) to O(N log N) by using 
a quadtree spatial data structure to approximate distant forces.

Key concepts:
- Quadtree: Recursively subdivides space into quadrants
- Theta parameter: Controls accuracy vs speed tradeoff
- Center of mass approximation: Distant groups treated as single particles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
import time

# Import axis scaling utilities if available
try:
    from axis_scaling_utils import calculate_dynamic_limits, set_equal_aspect_with_limits
    AXIS_UTILS_AVAILABLE = True
except ImportError:
    AXIS_UTILS_AVAILABLE = False
    
    def calculate_dynamic_limits(position_history, padding=0.15, min_range=2.0):
        """Fallback axis calculation if utils not available."""
        all_positions = np.vstack(position_history)
        x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
        y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
        x_range = max(x_max - x_min, min_range)
        y_range = max(y_max - y_min, min_range)
        x_pad = padding * x_range
        y_pad = padding * y_range
        return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)
    
    def set_equal_aspect_with_limits(ax, x_lim, y_lim):
        """Fallback axis setting if utils not available."""
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal', adjustable='box')

class QuadTree:
    """
    Quadtree data structure for spatial subdivision in Barnes-Hut algorithm.
    """
    
    def __init__(self, center, size, max_depth=10, theta=0.5):
        """
        Initialize quadtree node.
        
        Parameters:
        center: (x, y) center of this quadrant
        size: size of this quadrant
        max_depth: maximum tree depth to prevent infinite recursion
        theta: Barnes-Hut accuracy parameter (smaller = more accurate)
        """
        self.center = np.array(center)
        self.size = size
        self.max_depth = max_depth
        self.theta = theta
        self.depth = 0
        
        # Node properties
        self.total_mass = 0.0
        self.center_of_mass = np.array([0.0, 0.0])
        self.particles = []
        
        # Child nodes (NW, NE, SW, SE)
        self.children = [None, None, None, None]
        self.is_leaf = True
        
    def insert_particle(self, particle_idx, position, mass):
        """
        Insert a particle into the quadtree.
        
        Parameters:
        particle_idx: index of the particle
        position: (x, y) position
        mass: particle mass
        """
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
                # Re-insert the existing particle
                old_particle = self.particles[0]
                self.particles = []
                self._insert_into_child(old_particle[0], old_particle[1], old_particle[2])
                self.is_leaf = False
            else:
                # Maximum depth reached, just add to particle list
                self.particles.append((particle_idx, position, mass))
                return
        
        # Insert new particle into appropriate child
        if not self.is_leaf:
            self._insert_into_child(particle_idx, position, mass)
        else:
            self.particles.append((particle_idx, position, mass))
    
    def _subdivide(self):
        """Create four child quadrants."""
        half_size = self.size / 2
        quarter_size = half_size / 2
        
        # Create child centers (NW, NE, SW, SE)
        child_centers = [
            self.center + np.array([-quarter_size, quarter_size]),   # NW
            self.center + np.array([quarter_size, quarter_size]),    # NE
            self.center + np.array([-quarter_size, -quarter_size]),  # SW
            self.center + np.array([quarter_size, -quarter_size])    # SE
        ]
        
        for i in range(4):
            self.children[i] = QuadTree(child_centers[i], half_size, self.max_depth, self.theta)
            self.children[i].depth = self.depth + 1
    
    def _insert_into_child(self, particle_idx, position, mass):
        """Insert particle into appropriate child quadrant."""
        # Determine which quadrant the particle belongs to
        quadrant = 0
        if position[0] > self.center[0]:
            quadrant += 1
        if position[1] < self.center[1]:
            quadrant += 2
        
        self.children[quadrant].insert_particle(particle_idx, position, mass)
    
    def calculate_force(self, particle_idx, position, mass, G=1.0):
        """
        Calculate gravitational force on a particle using Barnes-Hut approximation.
        
        Parameters:
        particle_idx: index of the particle (to avoid self-interaction)
        position: particle position
        mass: particle mass
        G: gravitational constant
        
        Returns:
        force: (fx, fy) gravitational force
        """
        force = np.array([0.0, 0.0])
        
        # If this node has no mass, no force
        if self.total_mass == 0:
            return force
        
        # Distance to center of mass of this node
        r_vec = self.center_of_mass - position
        r = np.linalg.norm(r_vec)
        
        # Avoid self-interaction and division by zero
        if r < 1e-10:
            return force
        
        # Barnes-Hut criterion: if s/d < theta, treat as single particle
        if self.is_leaf or (self.size / r) < self.theta:
            # Check if this is a direct particle interaction (avoid self-force)
            if self.is_leaf and len(self.particles) == 1:
                if self.particles[0][0] == particle_idx:
                    return force  # Don't apply force to itself
            
            # Calculate force from center of mass
            force_magnitude = G * mass * self.total_mass / (r**2)
            force = force_magnitude * (r_vec / r)
        else:
            # Recursively calculate forces from children
            for child in self.children:
                if child is not None:
                    force += child.calculate_force(particle_idx, position, mass, G)
        
        return force


class BarnesHutNBody:
    """
    N-Body simulation using the Barnes-Hut algorithm.
    """
    
    def __init__(self, N, G=1.0, theta=0.5):
        """
        Initialize N-body system.
        
        Parameters:
        N: number of particles
        G: gravitational constant
        theta: Barnes-Hut accuracy parameter
        """
        self.N = N
        self.G = G
        self.theta = theta
        
        # Particle properties
        self.positions = np.zeros((N, 2))
        self.velocities = np.zeros((N, 2))
        self.masses = np.ones(N)
        
        # Simulation data
        self.time_points = []
        self.position_history = []
        self.energy_history = []
        
    def set_initial_conditions(self, positions, velocities, masses):
        """
        Set initial conditions for all particles.
        
        Parameters:
        positions: (N, 2) array of initial positions
        velocities: (N, 2) array of initial velocities
        masses: (N,) array of masses
        """
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.masses = np.array(masses)
    
    def create_galaxy_disk(self, center=(0, 0), radius=5.0, central_mass=10.0):
        """
        Create a galaxy-like disk distribution of particles.
        
        Parameters:
        center: center of the galaxy
        radius: disk radius
        central_mass: mass of central supermassive object
        """
        # Central supermassive object
        self.positions[0] = center
        self.velocities[0] = [0, 0]
        self.masses[0] = central_mass
        
        # Spiral disk particles
        for i in range(1, self.N):
            # Random position in disk
            r = np.sqrt(np.random.uniform(0.5, radius**2))
            theta = np.random.uniform(0, 2 * np.pi)
            
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            self.positions[i] = [x, y]
            
            # Circular velocity for stable orbit
            v_circular = np.sqrt(self.G * central_mass / r)
            # Add some random velocity dispersion
            v_dispersion = 0.1 * v_circular
            
            vx = -v_circular * np.sin(theta) + np.random.normal(0, v_dispersion)
            vy = v_circular * np.cos(theta) + np.random.normal(0, v_dispersion)
            self.velocities[i] = [vx, vy]
            
            # Smaller masses for disk particles
            self.masses[i] = 0.1
    
    def create_cluster(self, center=(0, 0), radius=2.0, velocity_dispersion=0.5):
        """
        Create a spherical cluster of particles.
        
        Parameters:
        center: center of the cluster
        radius: cluster radius
        velocity_dispersion: random velocity spread
        """
        for i in range(self.N):
            # Random position in sphere (using rejection sampling for uniform distribution)
            while True:
                x = np.random.uniform(-radius, radius)
                y = np.random.uniform(-radius, radius)
                if x**2 + y**2 <= radius**2:
                    break
            
            self.positions[i] = [center[0] + x, center[1] + y]
            
            # Random velocities with given dispersion
            vx = np.random.normal(0, velocity_dispersion)
            vy = np.random.normal(0, velocity_dispersion)
            self.velocities[i] = [vx, vy]
            
            # Equal masses
            self.masses[i] = 1.0
    
    def build_tree(self):
        """
        Build quadtree for current particle positions.
        
        Returns:
        root: root node of the quadtree
        """
        # Determine bounding box
        min_pos = np.min(self.positions, axis=0)
        max_pos = np.max(self.positions, axis=0)
        
        # Create square bounding box with some padding
        center = (min_pos + max_pos) / 2
        size = max(max_pos - min_pos) * 1.2  # Add 20% padding
        
        # Create root node
        root = QuadTree(center, size, theta=self.theta)
        
        # Insert all particles
        for i in range(self.N):
            root.insert_particle(i, self.positions[i], self.masses[i])
        
        return root
    
    def calculate_forces_barnes_hut(self):
        """
        Calculate forces on all particles using Barnes-Hut algorithm.
        
        Returns:
        forces: (N, 2) array of forces
        """
        # Build tree for current configuration
        tree = self.build_tree()
        
        forces = np.zeros((self.N, 2))
        
        # Calculate force on each particle
        for i in range(self.N):
            forces[i] = tree.calculate_force(i, self.positions[i], self.masses[i], self.G)
        
        return forces
    
    def calculate_forces_direct(self):
        """
        Calculate forces using direct N² method for comparison.
        
        Returns:
        forces: (N, 2) array of forces
        """
        forces = np.zeros((self.N, 2))
        
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_vec = self.positions[j] - self.positions[i]
                    r = np.linalg.norm(r_vec)
                    
                    # Avoid division by zero
                    if r > 1e-10:
                        force_magnitude = self.G * self.masses[i] * self.masses[j] / (r**2)
                        forces[i] += force_magnitude * (r_vec / r)
        
        return forces
    
    def step(self, dt, use_barnes_hut=True):
        """
        Advance simulation by one time step using leapfrog integration.
        
        Parameters:
        dt: time step
        use_barnes_hut: if True, use Barnes-Hut; if False, use direct method
        """
        # Calculate forces
        if use_barnes_hut:
            forces = self.calculate_forces_barnes_hut()
        else:
            forces = self.calculate_forces_direct()
        
        # Leapfrog integration
        accelerations = forces / self.masses[:, np.newaxis]
        
        # Update velocities (half step)
        self.velocities += 0.5 * dt * accelerations
        
        # Update positions
        self.positions += dt * self.velocities
        
        # Recalculate forces for new positions
        if use_barnes_hut:
            forces = self.calculate_forces_barnes_hut()
        else:
            forces = self.calculate_forces_direct()
        
        accelerations = forces / self.masses[:, np.newaxis]
        
        # Update velocities (second half step)
        self.velocities += 0.5 * dt * accelerations
    
    def simulate(self, t_span, dt=0.01, use_barnes_hut=True, save_interval=1):
        """
        Run the N-body simulation.
        
        Parameters:
        t_span: (t_start, t_end) simulation time range
        dt: integration time step
        use_barnes_hut: whether to use Barnes-Hut algorithm
        save_interval: save data every N steps
        """
        t_start, t_end = t_span
        t = t_start
        step_count = 0
        
        # Initialize history
        self.time_points = [t]
        self.position_history = [self.positions.copy()]
        self.energy_history = [self.calculate_total_energy()]
        
        print(f"Starting simulation with {self.N} particles...")
        print(f"Method: {'Barnes-Hut' if use_barnes_hut else 'Direct O(N²)'}")
        
        start_time = time.time()
        
        while t < t_end:
            # Advance simulation
            self.step(dt, use_barnes_hut)
            t += dt
            step_count += 1
            
            # Save data periodically
            if step_count % save_interval == 0:
                self.time_points.append(t)
                self.position_history.append(self.positions.copy())
                self.energy_history.append(self.calculate_total_energy())
            
            # Progress update
            if step_count % 100 == 0:
                progress = (t - t_start) / (t_end - t_start) * 100
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}% (t={t:.2f}, elapsed={elapsed:.1f}s)")
        
        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {elapsed_time/step_count*1000:.2f} ms")
        
        # Convert to numpy arrays
        self.time_points = np.array(self.time_points)
        self.position_history = np.array(self.position_history)
        self.energy_history = np.array(self.energy_history)
    
    def calculate_total_energy(self):
        """Calculate total energy of the system."""
        # Kinetic energy
        kinetic = 0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1))
        
        # Potential energy
        potential = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                r = np.linalg.norm(self.positions[j] - self.positions[i])
                if r > 1e-10:
                    potential -= self.G * self.masses[i] * self.masses[j] / r
        
        return kinetic + potential
    
    def animate(self, figsize=(12, 10), interval=50, trail_length=20, xlim=None, ylim=None):
        """
        Create animation of the N-body system.
        
        Parameters:
        figsize: figure size
        interval: animation interval in ms
        trail_length: number of previous positions to show
        xlim, ylim: axis limits (auto-calculated if None)
        """
        if len(self.position_history) == 0:
            raise ValueError("Run simulation first!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
          # Set axis limits using dynamic calculation
        if xlim is None or ylim is None:
            xlim, ylim = calculate_dynamic_limits(self.position_history, padding=0.1)
        
        set_equal_aspect_with_limits(ax1, xlim, ylim)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'Barnes-Hut N-Body Simulation (N={self.N}, θ={self.theta})')
        ax1.grid(True, alpha=0.3)
        
        # Energy plot
        ax2.plot(self.time_points, self.energy_history, 'b-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Total Energy')
        ax2.set_title('Energy Conservation')
        ax2.grid(True, alpha=0.3)
        
        # Initialize particle plots
        particles, = ax1.plot([], [], 'bo', markersize=3)
        trails = [ax1.plot([], [], 'b-', alpha=0.3, linewidth=1)[0] for _ in range(self.N)]
        
        # Energy indicator
        energy_line = ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        def animate_frame(frame):
            if frame < len(self.position_history):
                # Update particle positions
                positions = self.position_history[frame]
                particles.set_data(positions[:, 0], positions[:, 1])
                
                # Update trails
                for i in range(self.N):
                    start_frame = max(0, frame - trail_length)
                    trail_positions = self.position_history[start_frame:frame+1, i]
                    if len(trail_positions) > 1:
                        trails[i].set_data(trail_positions[:, 0], trail_positions[:, 1])
                
                # Update energy indicator
                if frame < len(self.time_points):
                    energy_line.set_xdata([self.time_points[frame]])
            
            return [particles] + trails + [energy_line]
        
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=len(self.position_history),
            interval=interval, blit=True, repeat=True
        )
        
        plt.tight_layout()
        return fig, anim


def benchmark_comparison(N_values=[10, 25, 50, 100]):
    """
    Benchmark Barnes-Hut vs direct method for different N values.
    """
    print("Benchmarking Barnes-Hut vs Direct Method")
    print("=" * 50)
    
    results = {'N': [], 'Direct': [], 'Barnes-Hut': [], 'Speedup': []}
    
    for N in N_values:
        print(f"\nTesting with N = {N} particles...")
        
        # Create test system
        system = BarnesHutNBody(N, theta=0.5)
        system.create_cluster(radius=2.0, velocity_dispersion=0.3)
        
        # Time direct method
        start_time = time.time()
        for _ in range(10):  # 10 force calculations
            forces_direct = system.calculate_forces_direct()
        direct_time = (time.time() - start_time) / 10
        
        # Time Barnes-Hut method
        start_time = time.time()
        for _ in range(10):  # 10 force calculations
            forces_bh = system.calculate_forces_barnes_hut()
        bh_time = (time.time() - start_time) / 10
        
        speedup = direct_time / bh_time if bh_time > 0 else float('inf')
        
        results['N'].append(N)
        results['Direct'].append(direct_time * 1000)  # Convert to ms
        results['Barnes-Hut'].append(bh_time * 1000)
        results['Speedup'].append(speedup)
        
        print(f"Direct method: {direct_time*1000:.2f} ms")
        print(f"Barnes-Hut:   {bh_time*1000:.2f} ms")
        print(f"Speedup:      {speedup:.2f}x")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(results['N'], results['Direct'], 'r-o', label='Direct O(N²)', linewidth=2)
    ax1.plot(results['N'], results['Barnes-Hut'], 'b-o', label='Barnes-Hut O(N log N)', linewidth=2)
    ax1.set_xlabel('Number of Particles')
    ax1.set_ylabel('Time per Force Calculation (ms)')
    ax1.set_title('Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(results['N'], results['Speedup'], 'g-o', linewidth=2)
    ax2.set_xlabel('Number of Particles')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Barnes-Hut Speedup')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Example: Galaxy simulation
    print("Barnes-Hut N-Body Simulation Demo")
    print("=" * 40)
    
    # Create galaxy-like system
    N = 100
    system = BarnesHutNBody(N, G=1.0, theta=0.5)
    system.create_galaxy_disk(radius=4.0, central_mass=50.0)
    
    # Run simulation
    system.simulate((0, 2), dt=0.02, use_barnes_hut=True, save_interval=2)
    
    # Create animation
    print("Creating animation...")
    fig, anim = system.animate(interval=50, trail_length=15)
    plt.show()
    
    # Run benchmark
    benchmark_comparison([10, 25, 50, 100])
