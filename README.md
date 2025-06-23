# N-Body Problem Physics Simulation

This project contains comprehensive simulations of classical gravitational N-body problems using Python. It includes both two-body and three-body problem simulations with interactive visualizations and animations.

## Project Overview

The N-body problem is a fundamental problem in classical mechanics and astrophysics that involves predicting the motion of multiple celestial bodies interacting through gravitational forces. While the two-body problem has analytical solutions, the three-body problem and higher-order systems exhibit chaotic behavior and require numerical methods.

## Features

### Two-Body Problem (`two_body_simulation.py`)
- **Analytical Foundation**: Based on Kepler's laws and conservation principles
- **Multiple Scenarios**: Circular orbits, elliptical orbits, near-collision trajectories
- **Energy Conservation**: Real-time monitoring of energy drift
- **Interactive Animations**: Customizable speed and focus windows
- **Comprehensive Plotting**: Orbital trajectories, energy plots, phase space diagrams

### Three-Body Problem (`three_body_simulation.py`)
- **Chaotic Dynamics**: Exhibits sensitive dependence on initial conditions
- **Famous Solutions**: Includes the Figure-8 orbit (Chenciner-Montgomery solution)
- **Multiple Configurations**: Triangular, linear, hierarchical systems
- **Advanced Visualizations**: 6-panel comprehensive analysis
- **Conservation Laws**: Energy and angular momentum monitoring

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib scipy
```

### Running Simulations

#### Two-Body Problem
```bash
python two_body_simulation.py
```

#### Three-Body Problem
```bash
python three_body_simulation.py
```

Both simulations offer interactive scenario selection with predefined interesting cases.

## Available Scenarios

### Two-Body Problem
1. **Circular Orbit**: Equal masses in circular motion
2. **Elliptical Orbit**: Unequal masses with eccentric orbits
3. **Stable Elliptical**: Center-of-mass frame demonstration
4. **High Eccentricity**: Dramatic orbital variations
5. **Near-Collision**: Fast-paced gravitational encounters

### Three-Body Problem
1. **Figure-8 Orbit**: The famous Chenciner-Montgomery periodic solution
2. **Linear Configuration**: Unstable but fascinating dynamics
3. **Triangular Configuration**: Symmetric three-body system
4. **Mild Chaotic**: Controlled chaotic behavior
5. **Hierarchical System**: Binary pair with distant third body

## Physics Implementation

### Numerical Methods
- **Integration**: High-order Dormand-Prince method (DOP853)
- **Adaptive Step Size**: Automatic adjustment for stability
- **Collision Detection**: Prevents numerical singularities
- **Conservation Monitoring**: Real-time energy and momentum tracking

### Key Equations
- **Gravitational Force**: F = G·m₁·m₂/r²
- **Newton's Second Law**: F = ma
- **Energy Conservation**: E = T + V (kinetic + potential)
- **Angular Momentum**: L = r × p

## Visualization Features

### Static Plots
- Orbital trajectories with starting positions
- Distance evolution between bodies
- Energy conservation monitoring
- Phase space diagrams
- Angular momentum tracking
- Center of mass motion

### Animations
- Real-time orbital motion
- Customizable trail lengths
- Adaptive animation speeds
- Time-window focusing for interesting events
- Mass-proportional body sizes

## Educational Value

This simulation suite is designed for:
- **Physics Students**: Understanding gravitational dynamics
- **Computational Physics**: Numerical integration techniques
- **Chaos Theory**: Demonstrating sensitive dependence on initial conditions
- **Celestial Mechanics**: Real-world applications in astrophysics
- **Scientific Computing**: Best practices in numerical simulation

## Advanced Features

### Energy Conservation Analysis
The simulations provide quantitative measures of numerical accuracy through energy drift calculations:
```
Energy drift = |E_final - E_initial| / |E_initial|
```

### Collision Handling
Automatic detection of close approaches prevents numerical instabilities while maintaining physical realism.

### Customizable Parameters
All scenarios support modification of:
- Initial positions and velocities
- Body masses
- Gravitational constant
- Integration time spans
- Animation parameters

## Scientific Accuracy

The simulations maintain:
- **Machine Precision**: Energy conservation to ~10⁻¹⁵ relative error
- **Physical Realism**: Proper center-of-mass behavior
- **Numerical Stability**: Adaptive integration with error control
- **Conservation Laws**: Momentum, energy, and angular momentum

## Future Extensions

Potential enhancements include:
- N-body simulations (N > 3)
- Relativistic corrections
- Non-gravitational forces
- 3D visualization
- Interactive parameter adjustment
- Data export for analysis

## File Structure

```
├── two_body_simulation.py     # Two-body problem simulation
├── three_body_simulation.py   # Three-body problem simulation
├── examples.py               # Usage examples and benchmarks
├── requirements.txt          # Python dependencies
└── README.md                # This documentation
```

## Contributing

This project follows scientific computing best practices:
- Clear variable naming with physics conventions
- Comprehensive documentation
- Energy conservation verification
- Modular, extensible design

For educational use, modifications, or enhancements, please maintain the scientific rigor and documentation standards established in the codebase.