import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

class ThreeBodyProblem:
    """
    A class to simulate the 3-body problem in classical mechanics.
    
    This simulation models three bodies under mutual gravitational attraction,
    solving the differential equations numerically and providing visualization.
    The 3-body problem is chaotic and has no general analytical solution.
    """
    
    def __init__(self, m1=1.0, m2=1.0, m3=1.0, G=1.0):
        """
        Initialize the three-body system.
        
        Parameters:
        m1, m2, m3: masses of the three bodies
        G: gravitational constant
        """
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.G = G
        self.total_mass = m1 + m2 + m3
        
        # Store simulation results
        self.time_points = None
        self.positions = None
        
    def set_initial_conditions(self, r1_init, r2_init, r3_init, v1_init, v2_init, v3_init):
        """
        Set initial positions and velocities for all three bodies.
        
        Parameters:
        r1_init, r2_init, r3_init: initial position vectors [x, y] for bodies 1, 2, and 3
        v1_init, v2_init, v3_init: initial velocity vectors [vx, vy] for bodies 1, 2, and 3
        """
        self.r1_init = np.array(r1_init)
        self.r2_init = np.array(r2_init)
        self.r3_init = np.array(r3_init)
        self.v1_init = np.array(v1_init)
        self.v2_init = np.array(v2_init)
        self.v3_init = np.array(v3_init)
        
        # Create initial state vector [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        self.initial_state = np.concatenate([
            self.r1_init, self.r2_init, self.r3_init, 
            self.v1_init, self.v2_init, self.v3_init
        ])
    
    def equations_of_motion(self, t, state):
        """
        Define the system of differential equations for the 3-body problem.
        
        Parameters:
        t: time
        state: current state vector [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        
        Returns:
        derivatives: [dx1/dt, dy1/dt, dx2/dt, dy2/dt, dx3/dt, dy3/dt, 
                      dvx1/dt, dvy1/dt, dvx2/dt, dvy2/dt, dvx3/dt, dvy3/dt]
        """
        # Extract positions and velocities
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
        
        # Calculate distances between all pairs
        r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        
        # Check for potential collisions (stop integration if bodies get too close)
        min_distance = 0.01  # Minimum allowed distance
        if r12 < min_distance or r13 < min_distance or r23 < min_distance:
            # Set very small accelerations to effectively stop the integration
            return np.array([vx1, vy1, vx2, vy2, vx3, vy3, 0, 0, 0, 0, 0, 0])

        # Avoid division by zero (add small epsilon)
        epsilon = 1e-10
        r12 = max(r12, epsilon)
        r13 = max(r13, epsilon)
        r23 = max(r23, epsilon)
        
        # Calculate gravitational forces on each body
        # Force on body 1 from bodies 2 and 3
        f12_x = self.G * self.m2 * (x2 - x1) / r12**3
        f12_y = self.G * self.m2 * (y2 - y1) / r12**3
        f13_x = self.G * self.m3 * (x3 - x1) / r13**3
        f13_y = self.G * self.m3 * (y3 - y1) / r13**3
        
        # Force on body 2 from bodies 1 and 3
        f21_x = -f12_x  # Newton's third law
        f21_y = -f12_y
        f23_x = self.G * self.m3 * (x3 - x2) / r23**3
        f23_y = self.G * self.m3 * (y3 - y2) / r23**3
        
        # Force on body 3 from bodies 1 and 2
        f31_x = -f13_x  # Newton's third law
        f31_y = -f13_y
        f32_x = -f23_x  # Newton's third law
        f32_y = -f23_y
        
        # Calculate accelerations (F = ma, so a = F/m)
        ax1 = (f12_x + f13_x) / self.m1
        ay1 = (f12_y + f13_y) / self.m1
        ax2 = (f21_x + f23_x) / self.m2
        ay2 = (f21_y + f23_y) / self.m2
        ax3 = (f31_x + f32_x) / self.m3
        ay3 = (f31_y + f32_y) / self.m3
        
        # Return derivatives
        return np.array([vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3])
    
    def simulate(self, t_span, num_points=1000):
        """
        Solve the differential equations numerically.
        
        Parameters:
        t_span: tuple (t_start, t_end) for simulation time range
        num_points: number of time points to evaluate
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        # Solve the system of ODEs with improved settings
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
        
        # Extract positions for all three bodies
        self.positions = {
            'body1': solution.y[:2],   # x1, y1
            'body2': solution.y[2:4],  # x2, y2
            'body3': solution.y[4:6],  # x3, y3
            'velocities': {
                'body1': solution.y[6:8],   # vx1, vy1
                'body2': solution.y[8:10],  # vx2, vy2
                'body3': solution.y[10:12]  # vx3, vy3
            }
        }
        
    def calculate_center_of_mass(self):
        """
        Calculate the center of mass trajectory.
        """
        if self.positions is None:
            raise ValueError("Run simulation first!")
            
        x1, y1 = self.positions['body1']
        x2, y2 = self.positions['body2']
        x3, y3 = self.positions['body3']
        
        x_cm = (self.m1 * x1 + self.m2 * x2 + self.m3 * x3) / self.total_mass
        y_cm = (self.m1 * y1 + self.m2 * y2 + self.m3 * y3) / self.total_mass
        
        return x_cm, y_cm
    
    def calculate_energy(self, state):
        """
        Calculate total energy (kinetic + potential) of the system.
        
        Parameters:
        state: current state vector [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        
        Returns:
        total_energy: kinetic + potential energy
        """
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
        
        # Kinetic energy
        v1_squared = vx1**2 + vy1**2
        v2_squared = vx2**2 + vy2**2
        v3_squared = vx3**2 + vy3**2
        kinetic_energy = 0.5 * (self.m1 * v1_squared + self.m2 * v2_squared + self.m3 * v3_squared)
        
        # Potential energy
        r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        
        potential_energy = -self.G * (
            self.m1 * self.m2 / r12 + 
            self.m1 * self.m3 / r13 + 
            self.m2 * self.m3 / r23
        )
        
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
                self.positions['body3'][0][i], self.positions['body3'][1][i],
                self.positions['velocities']['body1'][0][i], self.positions['velocities']['body1'][1][i],
                self.positions['velocities']['body2'][0][i], self.positions['velocities']['body2'][1][i],
                self.positions['velocities']['body3'][0][i], self.positions['velocities']['body3'][1][i]
            ])
            energies.append(self.calculate_energy(state))
        
        initial_energy = energies[0]
        final_energy = energies[-1]
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        
        return energy_drift, energies

    def plot_orbits(self, figsize=(15, 10)):
        """
        Create comprehensive plots of the orbital motion.
        """
        if self.positions is None:
            raise ValueError("Run simulation first!")
            
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=figsize)
        
        # Extract position data
        x1, y1 = self.positions['body1']
        x2, y2 = self.positions['body2']
        x3, y3 = self.positions['body3']
        
        # Plot 1: Orbital trajectories
        ax1.plot(x1, y1, 'b-', label=f'Body 1 (m={self.m1})', linewidth=2)
        ax1.plot(x2, y2, 'r-', label=f'Body 2 (m={self.m2})', linewidth=2)
        ax1.plot(x3, y3, 'g-', label=f'Body 3 (m={self.m3})', linewidth=2)
        ax1.plot(x1[0], y1[0], 'bo', markersize=8, label='Body 1 start')
        ax1.plot(x2[0], y2[0], 'ro', markersize=8, label='Body 2 start')
        ax1.plot(x3[0], y3[0], 'go', markersize=8, label='Body 3 start')
        
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
        
        # Plot 2: Distances between bodies over time
        distance_12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        distance_23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        
        ax2.plot(self.time_points, distance_12, 'purple', linewidth=2, label='Body 1-2')
        ax2.plot(self.time_points, distance_13, 'orange', linewidth=2, label='Body 1-3')
        ax2.plot(self.time_points, distance_23, 'brown', linewidth=2, label='Body 2-3')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Distance')
        ax2.set_title('Distances Between Bodies')
        ax2.legend()
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
                self.positions['body3'][0][i], self.positions['body3'][1][i],
                self.positions['velocities']['body1'][0][i], self.positions['velocities']['body1'][1][i],
                self.positions['velocities']['body2'][0][i], self.positions['velocities']['body2'][1][i],
                self.positions['velocities']['body3'][0][i], self.positions['velocities']['body3'][1][i]
            ])
            
            # Kinetic energy
            x1_i, y1_i, x2_i, y2_i, x3_i, y3_i, vx1, vy1, vx2, vy2, vx3, vy3 = state
            v1_squared = vx1**2 + vy1**2
            v2_squared = vx2**2 + vy2**2
            v3_squared = vx3**2 + vy3**2
            kinetic = 0.5 * (self.m1 * v1_squared + self.m2 * v2_squared + self.m3 * v3_squared)
            
            # Potential energy
            r12 = np.sqrt((x2_i - x1_i)**2 + (y2_i - y1_i)**2)
            r13 = np.sqrt((x3_i - x1_i)**2 + (y3_i - y1_i)**2)
            r23 = np.sqrt((x3_i - x2_i)**2 + (y3_i - y2_i)**2)
            
            potential = -self.G * (
                self.m1 * self.m2 / r12 + 
                self.m1 * self.m3 / r13 + 
                self.m2 * self.m3 / r23
            )
            
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
        
        # Plot 5: Angular momentum components
        L1_z = self.m1 * (x1 * self.positions['velocities']['body1'][1] - y1 * self.positions['velocities']['body1'][0])
        L2_z = self.m2 * (x2 * self.positions['velocities']['body2'][1] - y2 * self.positions['velocities']['body2'][0])
        L3_z = self.m3 * (x3 * self.positions['velocities']['body3'][1] - y3 * self.positions['velocities']['body3'][0])
        L_total = L1_z + L2_z + L3_z
        
        ax5.plot(self.time_points, L1_z, 'b-', label='Body 1', linewidth=2)
        ax5.plot(self.time_points, L2_z, 'r-', label='Body 2', linewidth=2)
        ax5.plot(self.time_points, L3_z, 'g-', label='Body 3', linewidth=2)
        ax5.plot(self.time_points, L_total, 'k--', label='Total', linewidth=2)
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Angular Momentum (z-component)')
        ax5.set_title('Angular Momentum Conservation')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Center of mass motion
        ax6.plot(x_cm, y_cm, 'k-', linewidth=2, label='CM trajectory')
        ax6.plot(x_cm[0], y_cm[0], 'ko', markersize=8, label='CM start')
        ax6.plot(x_cm[-1], y_cm[-1], 'ks', markersize=8, label='CM end')
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        ax6.set_title('Center of Mass Motion')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axis('equal')
        
        plt.tight_layout()
        return fig
    
    def animate_orbits(self, figsize=(12, 10), interval=50, trail_length=100, time_range=None, custom_limits=None):
        """
        Create an animated visualization of the orbital motion.
        
        Parameters:
        figsize: figure size tuple
        interval: animation interval in milliseconds
        trail_length: number of previous positions to show as trails
        time_range: tuple (start_time, end_time) to focus animation on specific time period
        custom_limits: dict with 'xlim' and 'ylim' tuples to override automatic limits
        """
        if self.positions is None:
            raise ValueError("Run simulation first!")        
        fig, ax = plt.subplots(figsize=figsize)
        
        x1, y1 = self.positions['body1']
        x2, y2 = self.positions['body2']
        x3, y3 = self.positions['body3']
        
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
            x3, y3 = x3[time_slice], y3[time_slice]
            time_points = self.time_points[time_slice]
        else:
            time_points = self.time_points        # Set up the plot with expanded limits to ensure bodies stay in frame
        if custom_limits:
            # Use custom limits if provided
            ax.set_xlim(custom_limits['xlim'])
            ax.set_ylim(custom_limits['ylim'])
        else:
            # Automatic calculation with improved padding
            x_min = min(np.min(x1), np.min(x2), np.min(x3))
            x_max = max(np.max(x1), np.max(x2), np.max(x3))
            y_min = min(np.min(y1), np.min(y2), np.min(y3))
            y_max = max(np.max(y1), np.max(y2), np.max(y3))
            
            # Add extra padding (30% on each side) to ensure bodies don't go out of frame
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding = 0.3  # Increased padding from 20% to 30%
            
            # Ensure minimum range to avoid too tight zoom
            min_range = 0.5
            if x_range < min_range:
                x_center = (x_max + x_min) / 2
                x_min, x_max = x_center - min_range/2, x_center + min_range/2
                x_range = min_range
            if y_range < min_range:
                y_center = (y_max + y_min) / 2
                y_min, y_max = y_center - min_range/2, y_center + min_range/2
                y_range = min_range
            
            ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
            ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Three-Body Problem Animation')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Initialize plot elements
        body1_point, = ax.plot([], [], 'bo', markersize=10 * np.sqrt(self.m1), label=f'Body 1 (m={self.m1})')
        body2_point, = ax.plot([], [], 'ro', markersize=10 * np.sqrt(self.m2), label=f'Body 2 (m={self.m2})')
        body3_point, = ax.plot([], [], 'go', markersize=10 * np.sqrt(self.m3), label=f'Body 3 (m={self.m3})')
        trail1, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
        trail2, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)
        trail3, = ax.plot([], [], 'g-', alpha=0.5, linewidth=1)
        
        # Center of mass (calculate for the filtered time range)
        if time_range is not None:
            # Calculate center of mass for the filtered data
            x_cm = (self.m1 * x1 + self.m2 * x2 + self.m3 * x3) / self.total_mass
            y_cm = (self.m1 * y1 + self.m2 * y2 + self.m3 * y3) / self.total_mass
        else:
            x_cm, y_cm = self.calculate_center_of_mass()
        cm_point, = ax.plot([], [], 'ko', markersize=5, label='Center of Mass')
        
        ax.legend()
        
        def animate(frame):
            # Current positions
            body1_point.set_data([x1[frame]], [y1[frame]])
            body2_point.set_data([x2[frame]], [y2[frame]])
            body3_point.set_data([x3[frame]], [y3[frame]])
            cm_point.set_data([x_cm[frame]], [y_cm[frame]])
            
            # Trails
            start_idx = max(0, frame - trail_length)
            trail1.set_data(x1[start_idx:frame+1], y1[start_idx:frame+1])
            trail2.set_data(x2[start_idx:frame+1], y2[start_idx:frame+1])
            trail3.set_data(x3[start_idx:frame+1], y3[start_idx:frame+1])
            
            return body1_point, body2_point, body3_point, cm_point, trail1, trail2, trail3
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(x1),  # Use length of filtered data
            interval=interval, blit=True, repeat=True
        )
        
        return fig, anim


def create_three_body_scenarios():
    """
    Create some interesting preset scenarios for the 3-body problem.
    """
    scenarios = {}
    
    # Scenario 1: Classic Figure-8 orbit (Chenciner-Montgomery solution)
    scenarios['figure_eight'] = {
        'description': 'Figure-8 orbit (Chenciner-Montgomery solution)',
        'm1': 1.0, 'm2': 1.0, 'm3': 1.0, 'G': 1.0,
        'r1_init': [-0.97000436, 0.24308753], 
        'r2_init': [0.97000436, -0.24308753],
        'r3_init': [0.0, 0.0],
        'v1_init': [0.4662036850, 0.4323657300], 
        'v2_init': [0.4662036850, 0.4323657300],
        'v3_init': [-0.93240737, -0.86473146],
        't_span': (0, 6.32)  # One complete period
    }
    
    # Scenario 2: Linear configuration (unstable but interesting)
    scenarios['linear_unstable'] = {
        'description': 'Linear configuration (unstable)',
        'm1': 1.0, 'm2': 2.0, 'm3': 1.0, 'G': 1.0,
        'r1_init': [-2.0, 0], 'r2_init': [0.0, 0], 'r3_init': [2.0, 0],
        'v1_init': [0, 0.5], 'v2_init': [0, -0.7], 'v3_init': [0, 0.2],
        't_span': (0, 10)
    }
    
    # Scenario 3: Triangular configuration
    scenarios['triangular'] = {
        'description': 'Triangular configuration',
        'm1': 1.0, 'm2': 1.0, 'm3': 1.0, 'G': 1.0,
        'r1_init': [1.0, 0], 'r2_init': [-0.5, 0.866], 'r3_init': [-0.5, -0.866],
        'v1_init': [0, 0.5], 'v2_init': [0.433, -0.25], 'v3_init': [-0.433, -0.25],
        't_span': (0, 12)
    }    # Scenario 4: Mild chaotic orbit (more stable)
    scenarios['mild_chaotic'] = {
        'description': 'Mild chaotic orbit (more stable)',
        'm1': 1.0, 'm2': 1.0, 'm3': 1.0, 'G': 1.0,
        'r1_init': [-1.2, 0], 'r2_init': [1.2, 0], 'r3_init': [0, 1.5],
        'v1_init': [0.1, 0.3], 'v2_init': [-0.1, 0.3], 'v3_init': [0, -0.6],
        't_span': (0, 10)
    }
    
    # Scenario 5: Hierarchical system (binary + distant third body)
    scenarios['hierarchical'] = {
        'description': 'Hierarchical system (binary + distant third)',
        'm1': 1.0, 'm2': 1.0, 'm3': 0.5, 'G': 1.0,
        'r1_init': [-0.5, 0], 'r2_init': [0.5, 0], 'r3_init': [0, 3.0],
        'v1_init': [0, -0.7], 'v2_init': [0, 0.7], 'v3_init': [0.6, 0],
        't_span': (0, 20)
    }
    
    return scenarios


if __name__ == "__main__":
    # Demo: Run a simulation with the Figure-8 orbit
    print("Three-Body Problem Simulation")
    print("=" * 40)
    
    scenarios = create_three_body_scenarios()
    
    print("Available scenarios:")
    for i, (key, scenario) in enumerate(scenarios.items(), 1):
        print(f"{i}. {scenario['description']}")
    
    try:
        choice = input(f"\nChoose a scenario (1-{len(scenarios)}) or press Enter for figure-8: ")
        if choice.strip() == "":
            scenario_key = 'figure_eight'
        else:
            scenario_key = list(scenarios.keys())[int(choice) - 1]
    except (ValueError, IndexError):
        scenario_key = 'figure_eight'
    
    scenario = scenarios[scenario_key]
    print(f"\nRunning scenario: {scenario['description']}")
    
    # Create and run simulation
    system = ThreeBodyProblem(scenario['m1'], scenario['m2'], scenario['m3'], scenario['G'])
    system.set_initial_conditions(
        scenario['r1_init'], scenario['r2_init'], scenario['r3_init'],
        scenario['v1_init'], scenario['v2_init'], scenario['v3_init']
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
    if scenario_key == 'figure_eight':
        # Slower animation for the beautiful figure-8 with custom axis limits (2x speed)
        custom_limits = {'xlim': (-1.5, 1.5), 'ylim': (-1.0, 1.0)}
        fig_anim, anim = system.animate_orbits(interval=25, trail_length=100, custom_limits=custom_limits)
    elif scenario_key in ['linear_unstable', 'mild_chaotic']:
        # Faster animation for chaotic scenarios, focus on early time (2x speed)
        fig_anim, anim = system.animate_orbits(interval=10, trail_length=50, time_range=(0, 8))
    elif scenario_key == 'hierarchical':
        # Wider view for hierarchical system (2x speed)
        custom_limits = {'xlim': (-4, 4), 'ylim': (-3, 4)}
        fig_anim, anim = system.animate_orbits(interval=15, trail_length=75, custom_limits=custom_limits)
    else:
        # Standard animation speed for other scenarios (2x speed)
        fig_anim, anim = system.animate_orbits(interval=15, trail_length=75)
    plt.show()
    
    print("\nSimulation complete!")
