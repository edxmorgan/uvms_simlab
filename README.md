# uvms_simlab

uvms_simlab extends [uvms-simulator](https://github.com/edxmorgan/uvms_simulator) by adding PS4 joystick support for teleoperation as well as capabilities for executing coverage plans in simulation or on real hardware.

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
cd <your_ros2_workspace>
colcon build
```

## Contributions

Contributions welcome! Fork and submit a pull request.