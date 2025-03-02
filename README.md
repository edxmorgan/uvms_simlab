# uvms_simlab

**uvms_simlab** extends [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) with advanced features including **Interactive Marker Control**, **Self Collision Avoidance**, PS4 joystick support for teleoperation, and the capability to execute coverage plans—usable both in simulation and on real hardware.

## Key Features

- **Interactive Marker Control:**  
  Easily control and plan system motions in RViz using interactive markers. Adjust vehicle and end-effector poses with intuitive 6-DOF markers that update in real time.

- **Self Collision Avoidance:**  
  Integrated collision avoidance ensures safe operation by automatically adjusting robot paths to avoid self collisions during interactive and automated maneuvers.

- **Coverage Planning:**  
  Execute advanced coverage plans that ensure complete workspace coverage while considering obstacles and robot dynamics.

- **Teleoperation with PS4 Joystick:**  
  Enjoy seamless teleoperation via PS4 joystick support, making manual control straightforward and intuitive.

- **Real Hardware & Simulation Support:**  
  Deploy and test your system on real hardware or in a simulated environment with a unified interface.

---

## Dependencies

- **pyPS4Controller**  
  Install with:
  ```bash
  sudo pip install pyPS4Controller
  ```
- **scipy**
- **casadi**

---

## Quick Start

1. **Clone the Repository:**  
   Navigate to the `src` folder of your ROS2 workspace using the absolute path:
   ```bash
   cd /absolute/path/to/your_ros2_workspace/src
   git clone https://github.com/edxmorgan/uvms_simlab.git
   ```
   Then, return to the workspace root:
   ```bash
   cd /absolute/path/to/your_ros2_workspace
   ```

2. **Build the Workspace:**  
   Compile the project with:
   ```bash
   colcon build
   source install/setup.bash
   ```

---

## Running the Simulation

### Interactive Markers Control (Recommended)

Control the system interactively in RViz with full 6-DOF marker support and built-in self collision avoidance:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=interactive
```

### Manual Mode (Joystick Control)

Ensure a PS4 joystick is connected via Bluetooth, then launch:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=manual
```

### Coverage Execution

Execute coverage plans with built-in self collision avoidance:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=coverage
```

---

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request.