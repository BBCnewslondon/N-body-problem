<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Two-Body Problem Physics Simulation

This project simulates the classic two-body problem in physics using numerical integration and provides interactive visualizations.

## Project Context
- This is a scientific computing project focusing on classical mechanics
- Uses Python with numpy for numerical computations, matplotlib for visualization
- Implements the gravitational two-body problem with various orbital scenarios
- Includes both static plots and animated visualizations

## Code Style Guidelines
- Use clear, descriptive variable names following physics conventions (e.g., `m1`, `m2` for masses, `r` for position, `v` for velocity)
- Include comprehensive docstrings for all classes and methods
- Add inline comments for complex physics calculations
- Follow PEP 8 style guidelines
- Use type hints where appropriate

## Physics Conventions
- Use standard physics notation: `r` for position, `v` for velocity, `a` for acceleration, `m` for mass, `G` for gravitational constant
- Implement energy conservation checks in simulations
- Include center of mass calculations
- Ensure proper handling of coordinate systems and reference frames

## Visualization Guidelines
- Use matplotlib for all plotting and animation
- Include clear axis labels with units
- Provide legends for multi-body systems
- Use appropriate colors (blue for body 1, red for body 2, black for center of mass)
- Include grid lines for better readability
- Ensure animations have reasonable frame rates and trail lengths
