import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

class TwoBodyProblem:
    """
    A class to simulate the 2-body problem in classical mechanics.
    
    This simulation models two bodies under mutual gravitational attraction,
    solving the differential equations numerically and providing visualization.
    """
    
    def __init__(self, m1=1.0, m2=1.0, G=1.0):
        """
        Initialize the two-body system.
        
        Parameters:
        m1, m2: masses of the two bodies
        G: gravitational constant
        """
        self.m1 = m1
        self.m2 = m2
        self.G = G
        self.total_mass = m1 + m2
        
        # Store simulation results
        self.time_points = None
        self.positions = None
        
    def set_initial_conditions(self, r1_init, r2_init, v1_init, v2_init):
        """
        Set initial positions and velocities for both bodies.
        
        Parameters:
        r1_init, r2_init: initial position vectors [x, y] for body 1 and 2
        v1_init, v2_init: initial velocity vectors [vx, vy] for body 1 and 2
        """
        self.r1_init = np.array(r1_init)
        self.r2_init = np.array(r2_init)
        self.v1_init = np.array(v1_init)
        self.v2_init = np.array(v2_init)
        
        # Create initial state vector [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.initial_state = np.concatenate([
            self.r1_init, self.r2_init, self.v1_init, self.v2_init
        ])
    
    def equations_of_motion(self, t, state):
        """
        Define the system of differential equations for the 2-body problem.
        
        Parameters:
        t: time
        state: current state vector [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        
        Returns:
        derivatives: [dx1/dt, dy1/dt, dx2/dt, dy2/dt, dvx1/dt, dvy1/dt, dvx2/dt, dvy2/dt]
        """
        # Extract positions and velocities
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
        
        # Calculate relative position vector
        dx = x2 - x1
        dy = y2 - y1
        r = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        if r < 1e-10:
            r = 1e-10
        
        # Calculate gravitational force components
        r3 = r**3
        fx = self.G * dx / r3
        fy = self.G * dy / r3
        
        # Accelerations (F = ma, so a = F/m)
        ax1 = self.m2 * fx
        ay1 = self.m2 * fy
        ax2 = -self.m1 * fx
        ay2 = -self.m1 * fy
        
        # Return derivatives
        return np.array([vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2])
    
    def simulate(self, t_span, num_points=1000):
        """
        Solve the differential equations numerically.
        
        Parameters:
        t_span: tuple (t_start, t_end) for simulation time range
        num_points: number of time points to evaluate
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)          # Solve the system of ODEs with improved settings
        solution = solve_ivp(
            self.equations_of_motion,
            t_span,
            self.initial_state,
            t_eval=t_eval,
            method='DOP853',  # Higher-order method for better accuracy
            rtol=1e-9,        # Tighter relative tolerance
            atol=1e-12,       # Absolute tolerance
            max_step=0.01     # Limit step size for stability
        )
        
        if not solution.success:
            print(f"Integration error message: {solution.message}")
            print(f"Integration status: {solution.status}")
            print(f"Last successful time: {solution.t[-1] if len(solution.t) > 0 else 'None'}")
            raise RuntimeError(f"Integration failed: {solution.message}")
        
        self.time_points = solution.t
        
        # Extract positions for both bodies
        self.positions = {
            'body1': solution.y[:2],  # x1, y1
            'body2': solution.y[2:4], # x2, y2
            'velocities': {
                'body1': solution.y[4:6],  # vx1, vy1
                'body2': solution.y[6:8]   # vx2, vy2
            }
        }
        
        return solution
    
    def calculate_center_of_mass(self):
        """Calculate the center of mass trajectory."""
        if self.positions is None:
            raise ValueError("Run simulation first!")
        
        x1, y1 = self.positions['body1']
        x2, y2 = self.positions['body2']
        
        x_cm = (self.m1 * x1 + self.m2 * x2) / self.total_mass
        y_cm = (self.m1 * y1 + self.m2 * y2) / self.total_mass
        
        return x_cm, y_cm
    
    def calculate_energy(self, state):
        """
        Calculate total energy (kinetic + potential) of the system.
        
        Parameters:
        state: current state vector [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        
        Returns:
        total_energy: kinetic + potential energy
        """
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
        
        # Kinetic energy
        v1_squared = vx1**2 + vy1**2
        v2_squared = vx2**2 + vy2**2
        kinetic_energy = 0.5 * (self.m1 * v1_squared + self.m2 * v2_squared)
        
        # Potential energy
        dx = x2 - x1
        dy = y2 - y1
        r = np.sqrt(dx**2 + dy**2)
        potential_energy = -self.G * self.m1 * self.m2 / r
        
        return kinetic_energy + potential_energy

    def check_conservation(self):
        """
        Check energy conservation throughout the simulation.
        Returns the relative energy drift.
        """
        if self.time_points is None:
            return None
        
        energies = []
        for i in range(len(self.time_points)):
            state = np.array([
                self.positions['body1'][0][i], self.positions['body1'][1][i],
                self.positions['body2'][0][i], self.positions['body2'][1][i],
                self.positions['velocities']['body1'][0][i], self.positions['velocities']['body1'][1][i],
                self.positions['velocities']['body2'][0][i], self.positions['velocities']['body2'][1][i]
            ])
            energies.append(self.calculate_energy(state))
        
        initial_energy = energies[0]
        final_energy = energies[-1]
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        
        return energy_drift, energies

    def plot_orbits(self, figsize=(12, 8)):
        """
        Create a static plot of the orbital trajectories.
        """
        if self.positions is None:
            raise ValueError("Run simulation first!")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        x1, y1 = self.positions['body1']
        x2, y2 = self.positions['body2']
        
        # Plot 1: Orbital trajectories
        ax1.plot(x1, y1, 'b-', label=f'Body 1 (m={self.m1})', linewidth=2)
        ax1.plot(x2, y2, 'r-', label=f'Body 2 (m={self.m2})', linewidth=2)
        ax1.plot(x1[0], y1[0], 'bo', markersize=8, label='Body 1 start')
        ax1.plot(x2[0], y2[0], 'ro', markersize=8, label='Body 2 start')
        
        # Mark center of mass
        x_cm, y_cm = self.calculate_center_of_mass()
        ax1.plot(x_cm, y_cm, 'k--', alpha=0.5, label='Center of Mass')
        ax1.plot(x_cm[0], y_cm[0], 'ko', markersize=6)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Orbital Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
          # Plot 2: Distance between bodies over time
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        ax2.plot(self.time_points, distance, 'g-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Distance')
        ax2.set_title('Distance Between Bodies')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Energy conservation
        energy_drift, energies = self.check_conservation()
        
        # Calculate kinetic and potential energies for plotting
        kinetic_energies = []
        potential_energies = []
        
        for i in range(len(self.time_points)):
            state = np.array([
                self.positions['body1'][0][i], self.positions['body1'][1][i],
                self.positions['body2'][0][i], self.positions['body2'][1][i],
                self.positions['velocities']['body1'][0][i], self.positions['velocities']['body1'][1][i],
                self.positions['velocities']['body2'][0][i], self.positions['velocities']['body2'][1][i]
            ])
            
            # Kinetic energy
            x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
            v1_squared = vx1**2 + vy1**2
            v2_squared = vx2**2 + vy2**2
            kinetic = 0.5 * (self.m1 * v1_squared + self.m2 * v2_squared)
            
            # Potential energy
            dx = x2 - x1
            dy = y2 - y1
            r = np.sqrt(dx**2 + dy**2)
            potential = -self.G * self.m1 * self.m2 / r
            
            kinetic_energies.append(kinetic)
            potential_energies.append(potential)
        
        ax3.plot(self.time_points, kinetic_energies, 'b-', label='Kinetic', linewidth=2)
        ax3.plot(self.time_points, potential_energies, 'r-', label='Potential', linewidth=2)
        ax3.plot(self.time_points, energies, 'k-', label='Total', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Energy')
        ax3.set_title(f'Energy Conservation (drift: {energy_drift:.2e})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
          # Plot 4: Phase space (position vs velocity for body 1)
        x1, y1 = self.positions['body1']
        vx1, vy1 = self.positions['velocities']['body1']
        speed1 = np.sqrt(vx1**2 + vy1**2)
        r1 = np.sqrt(x1**2 + y1**2)
        ax4.plot(r1, speed1, 'b-', linewidth=2)
        ax4.set_xlabel('Distance from origin')
        ax4.set_ylabel('Speed')
        ax4.set_title('Phase Space (Body 1)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def animate_orbits(self, figsize=(10, 8), interval=50, trail_length=100, time_range=None):
        """
        Create an animated visualization of the orbital motion.
        
        Parameters:
        figsize: figure size tuple
        interval: animation interval in milliseconds
        trail_length: number of previous positions to show as trails
        time_range: tuple (start_time, end_time) to focus animation on specific time period
        """
        if self.positions is None:
            raise ValueError("Run simulation first!")        
        fig, ax = plt.subplots(figsize=figsize)
        
        x1, y1 = self.positions['body1']
        x2, y2 = self.positions['body2']
        
        # Apply time range filtering if specified
        if time_range is not None:
            start_time, end_time = time_range
            # Find indices corresponding to the time range
            start_idx = np.searchsorted(self.time_points, start_time)
            end_idx = np.searchsorted(self.time_points, end_time)
            
            # Slice the data to focus on the specified time range
            time_slice = slice(start_idx, end_idx)
            x1, y1 = x1[time_slice], y1[time_slice]
            x2, y2 = x2[time_slice], y2[time_slice]
            time_points = self.time_points[time_slice]
        else:
            time_points = self.time_points

        # Set up the plot with expanded limits to ensure bodies stay in frame
        x_min = min(np.min(x1), np.min(x2))
        x_max = max(np.max(x1), np.max(x2))
        y_min = min(np.min(y1), np.min(y2))
        y_max = max(np.max(y1), np.max(y2))
        
        # Add extra padding (20% on each side) to ensure bodies don't go out of frame
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.2  # 20% padding
        
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Two-Body Problem Animation')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Initialize plot elements
        body1_point, = ax.plot([], [], 'bo', markersize=10 * np.sqrt(self.m1), label=f'Body 1 (m={self.m1})')
        body2_point, = ax.plot([], [], 'ro', markersize=10 * np.sqrt(self.m2), label=f'Body 2 (m={self.m2})')
        trail1, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
        trail2, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)
          # Center of mass (calculate for the filtered time range)
        if time_range is not None:
            # Calculate center of mass for the filtered data
            x_cm = (self.m1 * x1 + self.m2 * x2) / (self.m1 + self.m2)
            y_cm = (self.m1 * y1 + self.m2 * y2) / (self.m1 + self.m2)
        else:
            x_cm, y_cm = self.calculate_center_of_mass()
        cm_point, = ax.plot([], [], 'ko', markersize=5, label='Center of Mass')
        
        ax.legend()
        
        def animate(frame):
            # Current positions
            body1_point.set_data([x1[frame]], [y1[frame]])
            body2_point.set_data([x2[frame]], [y2[frame]])
            cm_point.set_data([x_cm[frame]], [y_cm[frame]])
            
            # Trails
            start_idx = max(0, frame - trail_length)
            trail1.set_data(x1[start_idx:frame+1], y1[start_idx:frame+1])
            trail2.set_data(x2[start_idx:frame+1], y2[start_idx:frame+1])
            
            return body1_point, body2_point, cm_point, trail1, trail2
          # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(x1),  # Use length of filtered data
            interval=interval, blit=True, repeat=True
        )
        
        return fig, anim


def create_preset_scenarios():
    """
    Create some interesting preset scenarios for the 2-body problem.
    """
    scenarios = {}
    
    # Scenario 1: Circular orbit (equal masses)
    scenarios['circular_equal'] = {
        'description': 'Circular orbit with equal masses',
        'm1': 1.0, 'm2': 1.0, 'G': 1.0,
        'r1_init': [-0.5, 0], 'r2_init': [0.5, 0],
        'v1_init': [0, -0.5], 'v2_init': [0, 0.5],
        't_span': (0, 20)
    }
    
    # Scenario 2: Elliptical orbit (unequal masses)
    scenarios['elliptical_unequal'] = {
        'description': 'Elliptical orbit with unequal masses',
        'm1': 2.0, 'm2': 0.5, 'G': 1.0,
        'r1_init': [-0.2, 0], 'r2_init': [0.8, 0],
        'v1_init': [0, -0.3], 'v2_init': [0, 1.2],
        't_span': (0, 30)
    }      # Scenario 3: Stable elliptical orbit (center of mass frame)
    scenarios['elliptical_stable'] = {
        'description': 'Stable elliptical orbit (center of mass frame)',
        'm1': 1.0, 'm2': 1.0, 'G': 1.0,
        'r1_init': [-0.5, 0], 'r2_init': [0.5, 0],
        'v1_init': [0, -0.5], 'v2_init': [0, 0.5],
        't_span': (0, 10)
    }
    
    # Scenario 4: High eccentricity orbit
    scenarios['high_eccentricity'] = {
        'description': 'High eccentricity orbit',
        'm1': 1.0, 'm2': 1.0, 'G': 1.0,
        'r1_init': [-0.5, 0], 'r2_init': [0.5, 0],
        'v1_init': [0, -0.8], 'v2_init': [0, 0.8],
        't_span': (0, 15)
    }
      # Scenario 5: Near-collision trajectory (highly dynamic)
    scenarios['near_collision'] = {
        'description': 'Near-collision trajectory (highly dynamic)',
        'm1': 1.0, 'm2': 1.0, 'G': 1.0,
        'r1_init': [-1.5, 0.2], 'r2_init': [1.5, -0.2],
        'v1_init': [0.6, 0.3], 'v2_init': [-0.6, -0.3],
        't_span': (0, 15)
    }
    
    return scenarios


if __name__ == "__main__":
    # Demo: Run a simulation with circular orbit
    print("Two-Body Problem Simulation")
    print("=" * 40)
    
    # Create scenarios
    scenarios = create_preset_scenarios()
    
    # Let user choose a scenario
    print("\nAvailable scenarios:")
    for i, (key, scenario) in enumerate(scenarios.items(), 1):
        print(f"{i}. {scenario['description']}")
    
    try:
        choice = input(f"\nChoose a scenario (1-{len(scenarios)}) or press Enter for circular_equal: ")
        if choice.strip() == "":
            scenario_key = 'circular_equal'
        else:
            scenario_key = list(scenarios.keys())[int(choice) - 1]
    except (ValueError, IndexError):
        scenario_key = 'circular_equal'
    
    scenario = scenarios[scenario_key]
    print(f"\nRunning scenario: {scenario['description']}")
    
    # Create and run simulation
    system = TwoBodyProblem(scenario['m1'], scenario['m2'], scenario['G'])
    system.set_initial_conditions(
        scenario['r1_init'], scenario['r2_init'],
        scenario['v1_init'], scenario['v2_init']
    )
    
    print("Solving differential equations...")
    system.simulate(scenario['t_span'], num_points=2000)
    
    print("Creating plots...")
    fig = system.plot_orbits()
    plt.show()

    # Check energy conservation
    energy_drift, energies = system.check_conservation()
    print(f"\nEnergy conservation check:")
    print(f"Initial energy: {energies[0]:.6f}, Final energy: {energies[-1]:.6f}")
    print(f"Relative energy drift: {energy_drift:.6f}%")
    
    # Create animation with scenario-specific settings
    print("Creating animation...")
    if scenario_key == 'near_collision':
        # Faster animation for the dynamic near-collision scenario
        # Focus on the interesting part: from start to 8 time units (captures approach and close encounter)
        fig_anim, anim = system.animate_orbits(interval=10, trail_length=30, time_range=(0, 8))
    else:
        # Standard animation speed for other scenarios
        fig_anim, anim = system.animate_orbits(interval=30, trail_length=50)
    plt.show()
    
    print("\nSimulation complete!")
