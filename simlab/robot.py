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

class Base:
    def nano_and_sec_to_sec(self, nanoseconds, seconds):
        """Converts nanoseconds and seconds to total seconds."""
        return seconds + (nanoseconds / 1e9)
        
    def get_sim_sec(self,  msg: DynamicJointState):
        total_sec = self.nano_and_sec_to_sec(msg.header.stamp.nanosec, msg.header.stamp.sec)
        return total_sec
    
    def get_interface_value(self, msg: DynamicJointState, dof_names: list, interface_names: list):
        names = msg.joint_names
        return [
            msg.interface_values[names.index(joint_name)].values[
                msg.interface_values[names.index(joint_name)].interface_names.index(interface_name)
            ]
            for joint_name, interface_name in zip(dof_names, interface_names)
        ]

class Axis_Interface_names:
    position = 'position'
    velocity = 'velocity'
    sim_time = 'sim_time'
    effort = 'effort'
    
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

    sim_period = 'sim_period'

class Manipulator(Base):
    def __init__(self, n_joint, prefix):
        self.n_joint = n_joint
        self.q = [0]*n_joint
        self.dq = [0]*n_joint
        self.sim_period = [0.0]

        self.alpha_axis_a = f'{prefix}_axis_a'
        self.alpha_axis_b = f'{prefix}_axis_b'
        self.alpha_axis_c = f'{prefix}_axis_c'
        self.alpha_axis_d = f'{prefix}_axis_d'
        self.alpha_axis_e = f'{prefix}_axis_e'

    def update_state(self, msg: DynamicJointState):
        self.q = self.get_interface_value(
            msg,
            [self.alpha_axis_e,
             self.alpha_axis_d,
             self.alpha_axis_c,
             self.alpha_axis_b],
            [Axis_Interface_names.position] * 4
        )

        self.dq = self.get_interface_value(
            msg,
            [self.alpha_axis_e,
             self.alpha_axis_d,
             self.alpha_axis_c,
             self.alpha_axis_b],
            [Axis_Interface_names.velocity] * 4
        )

        self.sim_period = self.get_interface_value(
            msg,
            [self.alpha_axis_e],
            [Axis_Interface_names.sim_period]
        )
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'q':self.q,
            'dq':self.dq,
            'dt':self.sim_period[0]
        }


class Robot(Base):
    def __init__(self, node: Node, n_joint, prefix):

        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        ref_intg_path = os.path.join(package_share_directory, 'ref_intg.casadi')
        j_uvms_path = os.path.join(package_share_directory, 'J_uvms.casadi')

        self.ref_intg_eval = ca.Function.load(ref_intg_path)
        self.J_uvms = ca.Function.load(j_uvms_path) # ned tf
        self.node = node

        self.n_joint = n_joint
        self.floating_base = f'{prefix}IOs'
        self.arm = Manipulator(n_joint, prefix)
        self.ned_pose = [0] * 6
        self.body_vel = [0] * 6
        self.prefix = prefix
        self.status = 'inactive'
        self.sim_time = 0.0
        self.start_time = 0.0

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.path_publisher = self.node.create_publisher(Path, f'/{self.prefix}Path', qos_profile)
        self.trajectory_path_publisher = self.node.create_publisher(Path, f'/{self.prefix}TrajectoryPath', qos_profile)

        self.ref_acc = np.zeros(11)
        self.ref_vel = np.zeros(11)
        self.ref_pos = np.array([3.0, 0.0, 5.0, 0,0,0, 3.1, 0.7, 0.4, 2.1, 0.0])

       # Initialize path poses
        self.path_poses = []
        self.traj_path_poses = []

        self.MAX_POSES = 10000

        # robot trajectory
        self.trajectory_twist = []
        self.trajectory_poses = []

        self.record = False

        self.initiaize_data_writer()

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
        if self.status == 'inactive':
            self.start_time = self.get_sim_sec(msg)
            self.status = 'active'
        elif self.status == 'active':
            self.sim_time = self.get_sim_sec(msg) - self.start_time

    def get_state(self) -> Dict:
        xq = self.arm.get_state()
        xq['name'] = self.prefix
        xq['pose'] = self.ned_pose
        xq['body_vel'] =self.body_vel
        xq['status'] = self.status
        xq['sim_time'] = self.sim_time
        return xq

    def to_ned_velocity(self, desired_body_vel):
        J_UVMS_REF = self.J_uvms(self.ref_pos[3:6])
        J_UVMS_REF_np = J_UVMS_REF.full()
        v_ned_ref = J_UVMS_REF_np@desired_body_vel[:-1]
        return v_ned_ref
    
    def integrate_vel_trajectory(self, desired_body_vel):
        dt = self.get_state()['dt']
        self.ned_vel = self.to_ned_velocity(desired_body_vel)
        self.ref_pos = self.ref_intg_eval(self.ref_pos[:-1], self.ned_vel, dt).full().flatten().tolist() + [0.0]

    def set_robot_goals(self, future_desired_body_vel):
        self.ref_vel = future_desired_body_vel.copy()
        self.integrate_vel_trajectory(future_desired_body_vel.copy())

        # Accumulate reference trajectory
        self.trajectory_twist.append(self.ref_vel.tolist().copy())  # Append a copy of the reference velocity
        self.trajectory_poses.append(self.ref_pos.copy())

        self.orient_towards_velocity()
        
        self.goal = dict()
        self.goal['ref_acc'] = self.ref_acc.tolist()
        self.goal['ref_vel'] = self.trajectory_twist[0]
        self.goal['ref_pos'] = self.trajectory_poses[0]

    def get_robot_goals(self, ref_type):
        return self.goal.get(ref_type)

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
                # Create a timestamped filename for the CSV
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"joint_data_{timestamp_str}.csv"
                
                # Open the CSV file and prepare to write data
                self.csv_file = open(filename, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                
                # Write a header row for clarity
                columns = [
                    'timestamp',
                    'base_x', 'base_y', 'base_z', 'base_roll', 'base_pitch', 'base_yaw',
                    'base_dx', 'base_dy', 'base_dz', 'base_vel_roll', 'base_vel_pitch', 'base_vel_yaw',
                    'base_x_force', 'base_y_force', 'base_z_force', 'base_x_torque', 'base_y_torque', 'base_z_torque',
                    'q_alpha_axis_e', 'q_alpha_axis_d', 'q_alpha_axis_c', 'q_alpha_axis_b',
                    'dq_alpha_axis_e', 'dq_alpha_axis_d', 'dq_alpha_axis_c', 'dq_alpha_axis_b',
                    'effort_alpha_axis_e', 'effort_alpha_axis_d', 'effort_alpha_axis_c', 'effort_alpha_axis_b'
                ]
                self.csv_writer.writerow(columns)

    def write_data_to_file(self, row_data):
        if self.record:
            """Write a single row of data to the CSV file."""
            self.csv_writer.writerow(row_data)
            self.csv_file.flush()

    def close_csv(self):
        # Close the CSV file when the node is destroyed
        self.csv_file.close()