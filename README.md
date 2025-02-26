# uvms_simlab

**uvms_simlab** extends [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) by adding PS4 joystick support for teleoperation as well as capabilities for executing coverage plans in simulation or on real hardware.

---

## Dependencies

- **pyPS4Controller**  
  Install using:
  ```bash
  sudo pip install pyPS4Controller
  ```
- **scipy**
- **casadi**

---

## Quick Start

1. **Clone the Repository:**

   Clone this repository into the `src` folder of your ROS2 workspace:
   ```bash
   cd <your_ros2_workspace>/src
   git clone https://github.com/edxmorgan/uvms_simlab.git
   cd ..
   ```

2. **Build the Workspace:**

   Build with `colcon`:
   ```bash
   colcon build
   source install/setup.bash
   ```

---

## Running the Simulation

### Manual Mode (Joystick Control)

Make sure a PS4 joystick is connected via Bluetooth. Then, run:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=manual
```

### Coverage Execution

To run the simulation with coverage execution:
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=coverage
```

---

## Contributions

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---