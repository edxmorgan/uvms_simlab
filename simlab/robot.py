import numpy as np
from typing import Dict
from control_msgs.msg import DynamicJointState
from scipy.spatial.transform import Rotation as R
import ament_index_python
import os
import casadi as ca
from nav_msgs.msg import Path
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSHistoryPolicy
import csv
from datetime import datetime
import random
import copy
from blue_rov import Params as blue
from alpha_reach import Params as alpha

# Import the PS4 controller library.
from pyPS4Controller.controller import Controller
import threading
import random
import time
import glob


class PS4Controller(Controller):
    def __init__(self, ros_node, **kwargs):
        super().__init__(**kwargs)
        self.ros_node = ros_node

        sim_gain = 0.5
        real_gain = 5

        self.gain = sim_gain

        # Gains for different DOFs
        self.max_torque = self.gain * 2.0             # for surge/sway
        self.heave_max_torque = self.gain * 3.0         # for heave (L2/R2)
        self.orient_max_torque = self.gain * 0.7        # for roll, pitch, yaw

        # # Create a lock specifically for updating gain values.
        # self.gain_lock = threading.Lock()

        # # Start a thread to update the gain every few seconds.
        # # gain randomization for good data collection
        # self.gain_thread = threading.Thread(target=self._update_gain, daemon=True)
        # self.gain_thread.start()

    # def _update_gain(self):
    #     """Randomize the gain value every few seconds and update the torque parameters."""
    #     while True:
    #         # For example, choose a new gain between 4 and 6.
    #         new_gain = random.uniform(3, 8)
    #         with self.gain_lock:
    #             self.gain = new_gain
    #             self.max_torque = self.gain * 2.0
    #             self.heave_max_torque = self.gain * 3.0
    #             self.orient_max_torque = self.gain * 0.7
    #         # Keep this gain for 5 seconds.
    #         time.sleep(5)

    # --- Analog stick and button callbacks below ---
    def on_L2_press(self, value):
        scaled_value = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = -scaled_value

    def on_L2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0

    def on_R2_press(self, value):
        scaled_value = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = scaled_value

    def on_R2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0

    def on_L3_up(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = -scaled

    def on_L3_down(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = -scaled

    def on_L3_right(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = scaled

    def on_L3_left(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = scaled

    def on_L3_x_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = 0.0

    def on_L3_y_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = 0.0

    def on_R3_up(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = scaled

    def on_R3_down(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = scaled

    def on_R3_left(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = scaled

    def on_R3_right(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = scaled

    def on_R3_x_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = 0.0

    def on_R3_y_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = 0.0

    def on_L1_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = -self.orient_max_torque

    def on_L1_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = 0.0

    def on_R1_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = self.orient_max_torque

    def on_R1_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = 0.0


    #
    # ---------------- Manipulator Controls ----------------
    # Mapping:
    #   - Manipulator index 0 (left/right):  
    #       on_left_arrow_press  → -1.0  
    #       on_right_arrow_press → +1.0  
    #       on_left_right_arrow_release → 0.0  
    #
    #   - Manipulator index 1 (up/down):  
    #       on_up_arrow_press   → +1.0  
    #       on_down_arrow_press → -1.0  
    #       on_up_down_arrow_release → 0.0  
    #
    #   - Indices 2, 3, and 4 remain unchanged.
    #
    # Manipulator index 0 (left/right):
    def on_left_arrow_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointe = -3.0

    def on_right_arrow_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointe = 3.0

    def on_left_right_arrow_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointe = 0.0

    # Manipulator index 1 (up/down):
    def on_up_arrow_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointd = 2.0

    def on_down_arrow_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointd = -2.0

    def on_up_down_arrow_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointd = 0.0

    # Manipulator index 2: Triangle (positive) / X (negative)
    def on_triangle_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 2.0

    def on_triangle_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 0.0

    def on_x_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = -2.0

    def on_x_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 0.0

    # Manipulator index 3: Square (positive) / Circle (negative)
    def on_square_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 1.0

    def on_square_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 0.0

    def on_circle_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = -1.0

    def on_circle_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 0.0

    # Manipulator index 4: Options (positive) / Share (negative)
    def on_R3_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 1.0

    def on_R3_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 0.0

    def on_L3_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = -1.0

    def on_L3_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 0.0
            

class Base:
    def get_interface_value(self, msg: DynamicJointState, dof_names: list, interface_names: list):
        names = msg.joint_names
        return [
            msg.interface_values[names.index(joint_name)].values[
                msg.interface_values[names.index(joint_name)].interface_names.index(interface_name)
            ]
            for joint_name, interface_name in zip(dof_names, interface_names)
        ]

class Axis_Interface_names:
    manipulator_position = 'position'
    manipulator_velocity = 'velocity'
    manipulator_effort = 'effort'
    
    floating_base_x = 'position.x'
    floating_base_y = 'position.y'
    floating_base_z = 'position.z'

    floating_base_roll = 'roll'
    floating_base_pitch = 'pitch'
    floating_base_yaw = 'yaw'

    floating_dx = 'velocity.x'
    floating_dy = 'velocity.y'
    floating_dz = 'velocity.z'

    floating_roll_vel = 'angular_velocity.x'
    floating_pitch_vel = 'angular_velocity.y'
    floating_yaw_vel = 'angular_velocity.z'

    floating_force_x = 'force.x'
    floating_force_y = 'force.y'
    floating_force_z = 'force.z'
    floating_torque_x = 'torque.x'
    floating_torque_y = 'torque.y'
    floating_torque_z = 'torque.z'

    sim_time = 'sim_time'
    sim_period = 'sim_period'

    imu_roll = "imu_roll"
    imu_pitch = "imu_pitch"
    imu_yaw = "imu_yaw"

    imu_q_w = "imu_orientation_w"
    imu_q_x = "imu_orientation_x"
    imu_q_y = "imu_orientation_y"
    imu_q_z = "imu_orientation_z"

    imu_wx = "imu_angular_vel_x"
    imu_wy = "imu_angular_vel_y"
    imu_wz = "imu_angular_vel_z"

    imu_ax = "imu_linear_acceleration_x"
    imu_ay = "imu_linear_acceleration_y"
    imu_az = "imu_linear_acceleration_z"

    depth_pressure2 = "depth_from_pressure2"

    dvl_roll = "dvl_gyro_roll"
    dvl_pitch = "dvl_gyro_pitch"
    dvl_yaw = "dvl_gyro_yaw"

    dvl_speed_x = "dvl_speed_x"
    dvl_speed_y = "dvl_speed_y"
    dvl_speed_z = "dvl_speed_z"

    
class Manipulator(Base):
    def __init__(self, node: Node, n_joint, prefix):
        self.node = node
        self.n_joint = n_joint
        self.q = [0]*n_joint
        self.dq = [0]*n_joint
        self.sim_period = [0.0]
        self.effort = [0]*n_joint
        self.alpha_axis_a = f'{prefix}_axis_a'
        self.alpha_axis_b = f'{prefix}_axis_b'
        self.alpha_axis_c = f'{prefix}_axis_c'
        self.alpha_axis_d = f'{prefix}_axis_d'
        self.alpha_axis_e = f'{prefix}_axis_e'


        self.a0 = 20e-3
        self.a1 = np.sqrt(40**2 + (154.3)**2)*(10**-3)
        self.a2 = 20e-3
        self.a3 = 0
        self.a4 = 0
        
        self.d0 = 46.2e-3
        self.d1 = 0
        self.d2 = 0
        self.d3 = -180e-3
        self.d4 = 0
        
        self.l1 = self.a1
        self.l2 = np.sqrt(self.a2**2 + self.d3**2)

        self.task_Space_Box ={'x_min': -0.05772180470827806, 'x_max': 0.03015406093747365,
                          'y_min': -0.06853860224338981, 'y_max': 0.057466988088834825,
                            'z_min': -0.2216507997959984, 'z_max': 0.3100041333006747}

    def generate_random_point(self):
        x = random.uniform(self.task_Space_Box['x_min'], self.task_Space_Box['x_max'])
        y = random.uniform(self.task_Space_Box['y_min'], self.task_Space_Box['y_max'])
        z = random.uniform(self.task_Space_Box['z_min'], self.task_Space_Box['z_max'])
        return (x, y, z)


    def update_state(self, msg: DynamicJointState):
        self.q = self.get_interface_value(
            msg,
            [self.alpha_axis_e,
             self.alpha_axis_d,
             self.alpha_axis_c,
             self.alpha_axis_b],
            [Axis_Interface_names.manipulator_position] * 4
        )

        self.dq = self.get_interface_value(
            msg,
            [self.alpha_axis_e,
             self.alpha_axis_d,
             self.alpha_axis_c,
             self.alpha_axis_b],
            [Axis_Interface_names.manipulator_velocity] * 4
        )

        self.effort = self.get_interface_value(
            msg,
            [self.alpha_axis_e,
             self.alpha_axis_d,
             self.alpha_axis_c,
             self.alpha_axis_b],
            [Axis_Interface_names.manipulator_effort] * 4
        )


        self.sim_period = self.get_interface_value(
            msg,
            [self.alpha_axis_e],
            [Axis_Interface_names.sim_period]
        )
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'arm_effort':self.effort,
            'q':self.q,
            'dq':self.dq,
            'dt':self.sim_period[0]
        }

    def ik_solver(self, target_position, pose="underarm"):
        x = target_position[0]
        y = target_position[1]
        z = target_position[2]

        thet0, thet1, thet2 = float("nan"), float("nan"), float("nan")
        try:
            R = np.sqrt(x**2 + y**2)
            l1 = self.a1
            l2 = np.sqrt(self.a2**2 + self.d3**2)

            if pose == 'underarm':
                thet0 = np.arctan2(y, x) + np.pi
                l3 = np.sqrt((R - self.a0)**2 + (z - self.d0)**2)
                
                # Compute argument for arccos and arcsin safely by clipping
                arg1 = (l1**2 + l2**2 - l3**2) / (2 * l1 * l2)
                term1 = np.arccos(np.clip(arg1, -1, 1))
                
                term2 = np.arcsin(np.clip((2 * self.a2) / l1, -1, 1))
                term3 = np.arcsin(np.clip(self.a2 / l2, -1, 1))
                
                thet2 = term1 - term2 - term3

                arg2 = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
                term4 = np.arccos(np.clip(arg2, -1, 1))
                
                thet1 = (np.pi / 2) + np.arctan2(z - self.d0, R - self.a0) - term4 - term2

            elif pose == 'overarm':
                thet0 = np.arctan2(y, x)
                l3 = np.sqrt((R + self.a0)**2 + (z - self.d0)**2)
                
                arg1 = (l1**2 + l2**2 - l3**2) / (2 * l1 * l2)
                term1 = np.arccos(np.clip(arg1, -1, 1))
                
                term2 = np.arcsin(np.clip((2 * self.a2) / l1, -1, 1))
                term3 = np.arcsin(np.clip(self.a2 / l2, -1, 1))
                
                thet2 = term1 - term2 - term3

                arg2 = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
                term4 = np.arccos(np.clip(arg2, -1, 1))
                
                thet1 = ((3 * np.pi) / 2) - np.arctan2(z - self.d0, R + self.a0) - term4 - term2

        except Exception as e:
            self.node.get_logger().error(f"An error occurred: {e}")
        return thet0, thet1, thet2


class Robot(Base):
    def __init__(self, node: Node, k_robot, n_joint, prefix, initial_pos, record=False,  controller='pid'):
        self.subscription = node.create_subscription(
                DynamicJointState,
                'dynamic_joint_states',
                self.listener_callback,
                10
            )
        self.k_robot = k_robot
        self.subscription  # prevent unused variable warning
    
        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        diff_iK_path = os.path.join(package_share_directory, 'diff_iK.casadi')
        fk_path = os.path.join(package_share_directory, 'fk_eval.casadi')

        vehicle_C_path = os.path.join(package_share_directory, 'vehicle/C.casadi')
        vehicle_M_path = os.path.join(package_share_directory, 'vehicle/M.casadi')
        vehicle_J_path = os.path.join(package_share_directory, 'vehicle/J_uv.casadi')

        manipulator_C_path = os.path.join(package_share_directory, 'manipulator/C.casadi')
        manipulator_M_path = os.path.join(package_share_directory, 'manipulator/M.casadi')

        self.diff_iK = ca.Function.load(diff_iK_path) # differential inverse kinematics
        self.fk_eval = ca.Function.load(fk_path) # differential inverse kinematics

        self.vehicle_C = ca.Function.load(vehicle_C_path)
        self.vehicle_M = ca.Function.load(vehicle_M_path)
        self.vehicle_J = ca.Function.load(vehicle_J_path)

        self.manipulator_C = ca.Function.load(manipulator_C_path)
        self.manipulator_M = ca.Function.load(manipulator_M_path)

        self.node = node

        self.sensors = [Axis_Interface_names.imu_roll,
                    Axis_Interface_names.imu_pitch,
                    Axis_Interface_names.imu_yaw,
                    Axis_Interface_names.imu_q_w,
                    Axis_Interface_names.imu_q_x,
                    Axis_Interface_names.imu_q_y,
                    Axis_Interface_names.imu_q_z,
                    Axis_Interface_names.imu_wx,
                    Axis_Interface_names.imu_wy,
                    Axis_Interface_names.imu_wz,
                    Axis_Interface_names.imu_ax,
                    Axis_Interface_names.imu_ay,
                    Axis_Interface_names.imu_az,
                    Axis_Interface_names.depth_pressure2,
                    Axis_Interface_names.dvl_roll,
                    Axis_Interface_names.dvl_pitch,
                    Axis_Interface_names.dvl_yaw,
                    Axis_Interface_names.dvl_speed_x,
                    Axis_Interface_names.dvl_speed_y,
                    Axis_Interface_names.dvl_speed_z]

        self.n_joint = n_joint
        self.floating_base = f'{prefix}IOs'
        self.arm = Manipulator(node, n_joint, prefix)
        self.ned_pose = [0] * 6
        self.body_vel = [0] * 6
        self.sensor_reading = [0] * len(self.sensors)
        self.body_forces = [0] * 6
        self.prefix = prefix
        self.status = 'inactive'
        self.sim_time = 0.0
        self.start_time = 0.0

        self.use_controller = controller

 
        self.uvms_ll = [-1000, -1000, 0.0, -np.pi/6, -np.pi/6, -1000, 1, 0.01, 0.01, 0.01]
        self.uvms_ul = [ 1000, 1000, 1000, np.pi/6, np.pi/6, 1000, 5.50, 3.40, 3.40, 5.70]
        self.k0 = [1,1,1 , 1,1,1, 1,1,1,1]
        self.base_To = [3.142, 0.0, 0.0, 0.19, 0.0, -0.12]


        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.path_publisher = self.node.create_publisher(Path, f'/{self.prefix}desiredPath', qos_profile)
        self.trajectory_path_publisher = self.node.create_publisher(Path, f'/{self.prefix}robotPath', qos_profile)

        self.path_ops_publisher = self.node.create_publisher(Path, f'/{self.prefix}desiredOpsPath', qos_profile)
        self.trajectory_path_ops_publisher = self.node.create_publisher(Path, f'/{self.prefix}robotOpsPath', qos_profile)

        self.ref_acc = np.zeros(10)
        self.ref_vel = np.zeros(10)
        self.ref_pos = initial_pos
        self.ops_pos = self.map_to_workspace(self.ref_pos)

        self.node.get_logger().info(f"Initial ref ops pose={self.ops_pos} ")

       # Initialize path poses
        self.path_poses = []
        self.traj_path_poses = []

        self.path_ops_poses = []
        self.traj_path__ops_poses = []


        self.MAX_POSES = 10000

        # robot trajectory
        self.trajectory_twist = []
        self.trajectory_poses = []
        self.trajectory_ops_twist = []
        self.trajectory_ops_poses = []

        self.record = record

        self.initiaize_data_writer()


        # Search for joystick device in /dev/input
        device_interface = f"/dev/input/js{self.k_robot}"
        self.has_joystick_interface = False
        joystick_device = glob.glob(device_interface)

        if device_interface in joystick_device:
            self.node.get_logger().info(f"Found joystick device: {device_interface}")
            self.start_joystick(device_interface)
            self.has_joystick_interface = True
        else:
            self.node.get_logger().info(f"No joystick device found for robot {self.k_robot}.")


    def start_joystick(self, device_interface):
        # Shared variables updated by the PS4 controller callbacks.
        self.controller_lock = threading.Lock()
        self.rov_surge = 0.0      # Left stick horizontal (sway)
        self.rov_sway = 0.0      # Left stick vertical (surge)
        self.rov_z = 0.0      # Heave from triggers
        self.rov_roll = 0.0   # roll
        self.rov_pitch = 0.0  # Right stick vertical (pitch)
        self.rov_yaw = 0.0    # Right stick horizontal (yaw)

        self.jointe = 0.0
        self.jointd = 0.0
        self.jointc = 0.0
        self.jointb = 0.0
        self.jointa = 0.0

        # Instantiate the PS4 controller.
        # If you are not receiving analog stick events, try adjusting the event_format.
        self.ps4_controller = PS4Controller(
            ros_node=self,
            interface=device_interface,
            connecting_using_ds4drv=False,
            event_format="3Bh2b"  # Try "LhBB" if you experience mapping issues.
        )
        # Enable debug mode to print raw event data.
        self.ps4_controller.debug = True

        # Start the PS4 controller listener in a separate (daemon) thread.
        self.controller_thread = threading.Thread(target=self.ps4_controller.listen, daemon=True)
        self.controller_thread.start()

        self.node.get_logger().info(f"PS4 Teleop node initialized for robot {self.k_robot} to be control with js{self.k_robot}.")


    def update_state(self, msg: DynamicJointState):
        self.arm.update_state(msg)
        self.ned_pose = self.get_interface_value(
            msg,
            [self.floating_base] * 6,
            [
                Axis_Interface_names.floating_base_x,
                Axis_Interface_names.floating_base_y,
                Axis_Interface_names.floating_base_z,
                Axis_Interface_names.floating_base_roll,
                Axis_Interface_names.floating_base_pitch,
                Axis_Interface_names.floating_base_yaw
            ]
        )


        self.body_vel = self.get_interface_value(
            msg,
            [self.floating_base] * 6,
            [
                Axis_Interface_names.floating_dx,
                Axis_Interface_names.floating_dy,
                Axis_Interface_names.floating_dz,
                Axis_Interface_names.floating_roll_vel,
                Axis_Interface_names.floating_pitch_vel,
                Axis_Interface_names.floating_yaw_vel
            ]
        )
        
        self.sensor_reading = self.get_interface_value(
            msg,
            [self.floating_base] * len(self.sensors),
            self.sensors
        )

        self.body_forces = self.get_interface_value(
            msg,
            [self.floating_base] * 6,
            [
            Axis_Interface_names.floating_force_x,
            Axis_Interface_names.floating_force_y, 
            Axis_Interface_names.floating_force_z,
            Axis_Interface_names.floating_torque_x,
            Axis_Interface_names.floating_torque_y,
            Axis_Interface_names.floating_torque_z
            ]
        )
        dynamics_sim_time = self.get_interface_value(msg,[self.floating_base],[Axis_Interface_names.sim_time])[0]
        if self.status == 'inactive':
            self.start_time = copy.copy(dynamics_sim_time)
            self.status = 'active'
        elif self.status == 'active':
            self.sim_time = dynamics_sim_time - self.start_time

    def get_state(self) -> Dict:
        xq = self.arm.get_state()
        xq['name'] = self.prefix
        xq['pose'] = self.ned_pose
        xq['body_vel'] =self.body_vel
        xq['body_forces'] = self.body_forces
        xq['status'] = self.status
        xq['sim_time'] = self.sim_time
        xq['prefix'] = self.prefix
        xq['raw_sensor_readings'] = self.sensor_reading
        # self.node.get_logger().info(f"body forces {xq['raw_sensor_readings']}")

        return xq

    def map_to_workspace(self, robot_configuration):
        self.ops_pos = self.fk_eval(robot_configuration,  self.base_To).full().flatten().tolist()
        return self.ops_pos

    def to_body_velocity(self, ned_vel):
        velocity_body = copy.copy(ned_vel)
        J_UV_REF = self.vehicle_J(self.ref_pos[3:6])
        velocity_body[:6] = np.linalg.inv(J_UV_REF.full())@ned_vel[:6]
        return velocity_body

    # def set_operation_space_goals(self, future_desired_body_vel, delay=True):
    #     robot_configuration = self.get_state()['pose'] + self.get_state()['q']

    #     data = np.zeros((11,))
    #     desired_generalized_vel = self.diff_iK(future_desired_body_vel,
    #                                 robot_configuration,
    #                                 self.uvms_ul,
    #                                 self.uvms_ll,
    #                                 self.k0,
    #                                 self.base_To
    #                                 ).full()
    #     data[0:10] = desired_generalized_vel.flatten()

    #     self.set_robot_goals(data, delay)
        

    def set_robot_goals(self, desired_ned_vel, desired_ned_pos):
        self.ned_vel = desired_ned_vel
        self.ref_vel = self.to_body_velocity(desired_ned_vel)
        self.ref_pos = desired_ned_pos
        # self.ops_pos = self.map_to_workspace(self.ref_pos)

        # Accumulate reference trajectory
        self.trajectory_twist.append(self.ref_vel.tolist().copy())  # Append a copy of the reference velocity
        self.trajectory_poses.append(self.ref_pos.copy())

        self.goal = dict()
        self.goal['ref_acc'] = self.ref_acc.tolist()
        self.goal['ref_vel'] = self.trajectory_twist[-1]
        self.goal['ref_pos'] = self.trajectory_poses[-1]

    def get_robot_goals(self, ref_type):
        return self.goal.get(ref_type)

    # def publish_ops_reference_path(self):
    #     # Publish the reference path to RViz
    #     path_msg = Path()
    #     path_msg.header.stamp = self.node.get_clock().now().to_msg()
    #     path_msg.header.frame_id = f"{self.prefix}map"  # Set to robot map frame

    #     # Create PoseStamped from ref_pos
    #     pose = PoseStamped()
    #     pose.header = path_msg.header
    #     pose.pose.position.x = float(self.ops_pos[0])
    #     pose.pose.position.y = float(self.ops_pos[1])
    #     pose.pose.position.z = float(self.ops_pos[2])
    #     pose.pose.orientation.w = 1.0  # No rotation
    #     pose.pose.orientation.x = 0.0  # No rotation
    #     pose.pose.orientation.y = 0.0  # No rotation
    #     pose.pose.orientation.z = 0.0  # No rotation
        
    #     # Accumulate poses
    #     self.path_ops_poses.append(pose)
    #     path_msg.poses = self.path_ops_poses

    #     # Limit the number of poses
    #     if len(self.path_ops_poses) > self.MAX_POSES:
    #         self.path_ops_poses.pop(0)
    #     self.path_ops_publisher.publish(path_msg)

    def publish_reference_path(self):
        # Publish the reference path to RViz
        path_msg = Path()
        path_msg.header.stamp = self.node.get_clock().now().to_msg()
        path_msg.header.frame_id = f"{self.prefix}map"  # Set to robot map frame

        # Create PoseStamped from ref_pos
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = float(self.ref_pos[0])
        pose.pose.position.y = -float(self.ref_pos[1])
        pose.pose.position.z = -float(self.ref_pos[2])
        pose.pose.orientation.w = 1.0  # No rotation
        pose.pose.orientation.x = 0.0  # No rotation
        pose.pose.orientation.y = 0.0  # No rotation
        pose.pose.orientation.z = 0.0  # No rotation

        # Accumulate poses
        self.path_poses.append(pose)
        path_msg.poses = self.path_poses

        # Limit the number of poses
        if len(self.path_poses) > self.MAX_POSES:
            self.path_poses.pop(0)
        self.path_publisher.publish(path_msg)

    def publish_robot_path(self):
        # Publish the robot trajectory path to RViz
        tra_path_msg = Path()
        tra_path_msg.header.stamp = self.node.get_clock().now().to_msg()
        tra_path_msg.header.frame_id = f"{self.prefix}map"  # Set to your appropriate frame

        # Create PoseStamped from ref_pos
        traj_pose = PoseStamped()
        traj_pose.header = tra_path_msg.header
        traj_pose.pose.position.x = float(self.ned_pose[0])
        traj_pose.pose.position.y = -float(self.ned_pose[1])
        traj_pose.pose.position.z = -float(self.ned_pose[2])
        traj_pose.pose.orientation.w = 1.0  # No rotation

        # Accumulate poses
        self.traj_path_poses.append(traj_pose)
        tra_path_msg.poses = self.traj_path_poses

        self.trajectory_path_publisher.publish(tra_path_msg)

    def orient_towards_velocity(self):
        """
        Orient the robot to face the direction of its current positive velocity.
        This updates the robot's reference orientation based on its body velocity.
        """
        vx = self.ned_vel[0]
        vy = self.ned_vel[1]

        # Compute the magnitude of the horizontal velocity
        horizontal_speed = np.hypot(vx, vy)

        # Threshold to avoid undefined behavior when velocity is near zero
        velocity_threshold = 1e-3

        if horizontal_speed > velocity_threshold:
            desired_yaw = np.arctan2(vy, vx)

            # -- Get the CURRENT yaw from the last pose in trajectory_poses
            if len(self.trajectory_poses)>1:
                current_yaw = self.trajectory_poses[-2][5]
            else:
                current_yaw = self.trajectory_poses[-1][5]
            # -- Compute the shortest-path yaw
            adjusted_yaw = self.adjust_desired_yaw(desired_yaw, current_yaw)

            self.trajectory_poses[-1][5] = adjusted_yaw

            # self.node.get_logger().info(f"Orienting towards velocity:current yaw={current_yaw} radians  desired yaw={desired_yaw} radians adjusted yaw={adjusted_yaw} radians")


    def adjust_desired_yaw(self, desired_yaw, current_yaw):
            # Compute the smallest angular difference
        angle_diff = desired_yaw - current_yaw
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to (-π, π)

        # Adjust desired_yaw to ensure the shortest rotation path
        adjusted_desired_yaw = current_yaw + angle_diff

        # # Normalize the adjusted desired yaw to [0, 2π)
        # adjusted_desired_yaw = adjusted_desired_yaw % (2 * np.pi)

        return adjusted_desired_yaw

    def initiaize_data_writer(self):
            if self.record:
                # Create a timestamp string
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create a folder with the timestamp as its name (in the current working directory)
                folder_path = os.path.join(os.getcwd(), timestamp_str)
                os.makedirs(folder_path, exist_ok=True)
                
                # Create a timestamped filename for the CSV
                filename = f"{timestamp_str}_{self.prefix}.csv"
                file_path = os.path.join(folder_path, filename)
                
                # Open the CSV file and prepare to write data
                self.csv_file = open(file_path, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)

                # Write a header row for clarity
                columns = [
                    'timestamp',
                    'base_x_force', 'base_y_force', 'base_z_force', 'base_x_torque', 'base_y_torque', 'base_z_torque',
                    'base_x', 'base_y', 'base_z', 'base_roll', 'base_pitch', 'base_yaw',
                    'base_dx', 'base_dy', 'base_dz', 'base_vel_roll', 'base_vel_pitch', 'base_vel_yaw',
                    
                    'effort_alpha_axis_e', 'effort_alpha_axis_d', 'effort_alpha_axis_c', 'effort_alpha_axis_b',
                    'q_alpha_axis_e', 'q_alpha_axis_d', 'q_alpha_axis_c', 'q_alpha_axis_b',
                    'dq_alpha_axis_e', 'dq_alpha_axis_d', 'dq_alpha_axis_c', 'dq_alpha_axis_b',

                    'imu_roll', 'imu_pitch', 'imu_yaw', 'imu_q_w', 'imu_q_x', 'imu_q_y', 'imu_q_z',
                    'imu_ang_vel_x', 'imu_ang_vel_y','imu_ang_vel_z',
                    'imu_linear_acc_x', 'imu_linear_acc_y','imu_linear_acc_z',
                    'depth_from_pressure2',
                    'dvl_roll', 'dvl_pitch', 'dvl_yaw',
                    'dvl_speed_x', 'dvl_speed_y', 'dvl_speed_z'
                ]
                self.csv_writer.writerow(columns)

    def write_data_to_file(self):
        if self.record:
            row_data = []
            info = self.get_state()
        
            row_data.extend([info['sim_time']])
            
            row_data.extend(info['body_forces'])
            row_data.extend(info['pose'])
            row_data.extend(info['body_vel'])

            row_data.extend(info['arm_effort'])
            row_data.extend(info['q'])
            row_data.extend(info['dq'])
            
            row_data.extend(info['raw_sensor_readings'])

            """Write a single row of data to the CSV file."""
            self.csv_writer.writerow(row_data)
            self.csv_file.flush()

    def close_csv(self):
        # Close the CSV file when the node is destroyed
        self.csv_file.close()

    def listener_callback(self, msg: DynamicJointState):
        self.update_state(msg)