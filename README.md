# uvms_simlab

**uvms_simlab** extends [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) by adding PS4 joystick support for teleoperation, as well as capabilities for executing coverage plans and interactive control via markers—both in simulation and on real hardware.

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

### Manual Mode (Joystick Control)

Ensure a PS4 joystick is connected via Bluetooth, then launch:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=manual
```

### Coverage Execution

For executing coverage plans, run:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=coverage
```

### Interactive Markers Control

To control the system interactively via markers in rviz:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=interactive
```

---

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request.
