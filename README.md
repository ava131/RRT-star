# 🤖 3-DOF Robot Arm Path Planning (RRT & RRT*)

A Python-based simulation and benchmarking tool for 3-Degree-of-Freedom (3-DOF) robot arm path planning in a 2D Configuration Space (C-Space). This project implements and compares the standard **Rapidly-exploring Random Tree (RRT)** and its asymptotically optimal variant, **RRT***.

## ✨ Features

- **Configuration Space Planning**: Solves the inverse kinematics ambiguity by mapping 2D workspace obstacles into a 3D joint-angle C-Space.
- **Interactive GUI**: Built with Pygame, offering a real-time visualization of the tree growth and the final smoothed robot arm movement.
- **Headless Benchmarking**: Includes a Monte Carlo testing suite to evaluate algorithm performance across randomized environments without GUI overhead.
- **MVC Architecture**: Fully decoupled codebase separating the physical environment, planning algorithms, and UI rendering.

## 📁 Project Structure

The project follows a modular design for maintainability:

- `config.py`: Centralized configuration for magic numbers, robot kinematics, and hyper-parameters.
- `environment.py`: Pure mathematical logic for Forward Kinematics and collision detection (distance from segments to circles).
- `planner.py`: The core algorithmic implementations of RRT and RRT*, including `Rewire` logic and path interpolation.
- `gui.py`: The presentation layer using Pygame for rendering trees, paths, and robot entities.
- `main.py`: The main entry point supporting both GUI Demonstration and Headless Benchmarking modes.

## 🚀 Installation & Usage

### Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install numpy pygame pandas
```
Running the ProjectOpen main.py and modify the execution mode at the bottom of the script:Mode A: Interactive GUI DemoUncomment the GUI lines to visualize the planning process:
```Python
if __name__ == "__main__":
    env = RobotArmEnv(LINK_LENGTHS, CENTER, DEFAULT_OBSTACLE)
    app = RobotArmGUI(env)
    app.run()
```
Mode B: Headless BenchmarkUncomment the benchmark line to run 50 randomized trials and generate statistical data:
```Python
if __name__ == "__main__":
    env = RobotArmEnv(LINK_LENGTHS, CENTER, DEFAULT_OBSTACLE)
    run_randomized_benchmark(env, trials=50)
```
