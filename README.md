# uvms_simlab 🚀

A field-ready ROS 2 lab for **Underwater Vehicle–Manipulator Systems**. `uvms_simlab` layers interactive teleoperation, collision-aware planning, and hardware-in-the-loop tooling on top of [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) so you can go from concept to wet tests without rebuilding infrastructure.


## Highlights

- 🌀🖱️ **Direct RViz manipulation** – 6‑DoF interactive markers drive the vehicle body or end-effector without custom plugins.
- 🤖 **Continuous self-collision monitoring** – convex-hull broad phase backed by FCL keeps the arm safe during planning and teleop.
- 🗺️ **SE(3) planning with live visualization** – OMPL planners stream candidate paths, coverage sweeps, and executed waypoints to RViz.
- 🎮 **Controller flexibility** – PS4/joy teleop, PID controllers, or your own ROS 2 nodes can be swapped via launch arguments.
- 📡 **Visualization overlays** – vehicle hulls, workspace clouds, goal menus, and path trails are preconfigured for situational awareness.
- 📓 **Data logging hooks** – CSV logs per robot make it easy to build ML datasets or audit controllers.

## Requirements

- ROS 2 jazzy plus the [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) stack installed exactly as documented in its README (system packages, `vcs import`, `rosdep`, CasADi, etc.).
- Python deps: `pyPS4Controller`, `pynput`, `scipy`, `casadi`, `pandas`.
- OMPL with Python bindings (`install-ompl-ubuntu.sh --python` from Kavraki Lab works well).
- Optional hardware: BlueROV2 Heavy + Reach Alpha 5 + Blue Robotics A50 DVL (or any robot stack you map through the provided interfaces).

## Quick start ⚡

1. **Install uvms-simulator and dependencies**  
   Follow the [uvms-simulator installation guide](https://github.com/edxmorgan/uvms-simulator/blob/main/README.md). 

2. **Add uvms_simlab and planning extras**

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/edxmorgan/uvms_simlab.git

   sudo pip install pyPS4Controller pynput scipy casadi pandas
   wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh
   chmod u+x install-ompl-ubuntu.sh
   ./install-ompl-ubuntu.sh --python

   cd ..
   colcon build
   source install/setup.bash
   ```

## Launch recipes 🚢

**Interactive planner & RViz**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    sim_robot_count:=1 task:=interactive \
    use_manipulator_hardware:=false use_vehicle_hardware:=false
```

**PS4 joystick teleop**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    task:=manual
```

**Headless data collection**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    gui:=false task:=manual record_data:=true
```

> 💡 Hardware swap: set `use_vehicle_hardware:=true` and `use_manipulator_hardware:=true` to put your BlueROV2 Heavy, Reach Alpha 5, and A50 DVL directly into the loop.

## Project layout 🧭

```
simlab/
├── simlab/interactive_control.py   # RViz markers, path planner, teleop
├── simlab/robot.py                 # Robot wrapper, logging, command mux
├── simlab/fcl_checker.py           # Collision environment utilities
├── simlab/interactive_utils.py     # Marker helpers & viz utilities
└── resource/                       # models
```

## Contributing & community 🤝

Have a planner, sensor, or teleop workflow that should live here? Open an issue or PR
