# uvms_simlab

uvms_simlab extends [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) by adding PS4 joystick support for teleoperation as well as capabilities for executing coverage plans in simulation or on real hardware.

## Dependencies

- **pyPS4Controller:**  
  ```bash
  sudo pip install pyPS4Controller
  ```
- **scipy**
- **casadi**

## Quick Start

Clone this repository into the `src` folder of your ROS2 workspace and build with colcon:

```bash
cd <your_ros2_workspace>/src
git clone https://github.com/edxmorgan/uvms_simlab.git
cd ..
colcon build
source install/setup.bash
```

Run simulation with manual mode (joystick)
Make sure a PS4 joystick is connected via bluetooth.
```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py use_manipulator_hardware:=false use_vehicle_hardware:=false sim_robot_count:=1 task:=manual
```

Run simulation with coverage excution

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py use_manipulator_hardware:=false use_vehicle_hardware:=false sim_robot_count:=1 task:=coverage
```
## Contributions

Contributions welcome! Fork and submit a pull request.