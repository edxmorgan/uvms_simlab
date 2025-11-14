# uvms_simlab 🚀

Your plug-and-play lab for **Underwater Vehicle–Manipulator Systems**. `uvms_simlab` layers interactive control, collision-aware planning, and hardware-ready tooling on top of [uvms-simulator](https://github.com/edxmorgan/uvms-simulator). If you’re prototyping underwater manipulation or HIL experiments, star the repo and dive in!

![UVMS SimLab](https://raw.githubusercontent.com/edxmorgan/uvms-simulator/main/doc/uvms_env.png)

## Why teams love uvms_simlab ⭐

- 🌀🖱️ **Drag-and-drive RViz controls** – steer the vehicle or end-effector directly with 6‑DoF markers.
- 🤖 **Continuous self-collision checking** – convex hull + FCL narrow-phase keeps the manipulator safe.
- 🗺️ **SE(3) planning & visualization** – plan coverage paths with OMPL and stream waypoints live.
- 🎮 **PS4/Joy teleop & HIL switches** – toggle between simulation and hardware with launch args.
- 📡 **Rich visualization suite** – workspace clouds, vehicle hulls, path trails, and marker menus baked in.
- 📓 **Data logging hooks** – CSV logs per robot for ML datasets or controller tuning.

## Quick start ⚡

```bash
sudo pip install pyPS4Controller pynput scipy casadi pandas
wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh
chmod u+x install-ompl-ubuntu.sh
./install-ompl-ubuntu.sh --python   # optional but recommended

cd ~/ros2_ws/src
git clone https://github.com/edxmorgan/uvms-simulator.git
git clone https://github.com/edxmorgan/uvms_simlab.git
cd ..
colcon build --packages-select uvms-simulator simlab
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
    task:=manual controllers:=pid
```

**Headless data collection**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    gui:=false task:=planner record_data:=true
```

> 💡 Hardware swap: set `use_vehicle_hardware:=true` and `use_manipulator_hardware:=true` to drop your BlueROV2 Heavy + Reach Alpha 5 directly into the loop.

## Project layout 🧭

```
simlab/
├── simlab/interactive_control.py   # RViz markers, path planner, teleop
├── simlab/robot.py                 # Robot wrapper, logging, command mux
├── simlab/fcl_checker.py           # Collision environment utilities
├── simlab/interactive_utils.py     # Marker helpers & viz utilities
└── resource/                       # RViz configs, meshes, assets
```

## Contributing & community 🤝

Have a new planner, sensor, or teleop idea? We’d love to merge it.

1. Fork the repo & branch (`feat/coverage-planner`).
2. `colcon test --packages-select simlab`.
3. Attach screenshots/gifs to your PR so others can see the feature in action.

Star the project, share your demos on social media or ROS Discourse, and help grow the UVMS community! 
