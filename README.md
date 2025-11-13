# uvms_simlab 🚀

Extension of the [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) for interactive control, collision aware planning, and hardware ready UVMS experiments.

## Features ⭐

* 🎮 **Interactive 6 DOF RViz Markers**
  Direct vehicle and end effector control.

* 🤖 **Self Collision Avoidance**
  Broad phase and narrow phase checks using FCL.

* 🗺️ **Coverage Planning**
  Automated workspace coverage with collision handling.

* 🕹️ **PS4 Teleoperation**
  Bluetooth controller support for manual operation.

* 🔧 **Simulation and Hardware**
  Swap backends with simple launch arguments.

* ⚡ **Fast Collision Detection**
  Efficient bounding volume checks for planning.

## Dependencies 📦

```bash
sudo pip install pyPS4Controller pynput scipy casadi pandas
```

Optional OMPL with Python bindings:

```bash
wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh
chmod u+x install-ompl-ubuntu.sh
./install-ompl-ubuntu.sh --python
```

## Installation 🛠️

```bash
cd ros2_ws/src
git clone https://github.com/edxmorgan/uvms_simlab.git
cd ..
colcon build
source install/setup.bash
```

Requires `uvms-simulator` in the same workspace.

## Launch Examples 🚢

**Interactive Control**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=interactive
```

**PS4 Manual Mode**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=manual
```

**Coverage Planning**

```bash
ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
    use_manipulator_hardware:=false use_vehicle_hardware:=false \
    sim_robot_count:=1 task:=coverage
```

## Contributing 🤝

PRs are welcome. Fork the repo and submit improvements.

---