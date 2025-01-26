import numpy as np
from typing import Dict
from control_msgs.msg import DynamicJointState


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
    position = 'position'
    velocity = 'velocity'
    sim_time = 'sim_time'
    effort = 'effort'
    floating_base_x = 'position.x'
    floating_base_y = 'position.y'
    floating_base_z = 'position.z'
    floating_base_iw = 'orientation.w'
    floating_base_ix = 'orientation.x'
    floating_base_iy = 'orientation.y'
    floating_base_iz = 'orientation.z'

    floating_dx = 'velocity.x'
    floating_dy = 'velocity.y'
    floating_dz = 'velocity.z'
    floating_roll_vel = 'angular_velocity.x'
    floating_pitch_vel = 'angular_velocity.y'
    floating_yaw_vel = 'angular_velocity.z'

class Manipulator(Base):
    def __init__(self, n_joint, prefix):
        self.n_joint = n_joint
        self.q = [0]*n_joint
        self.dq = [0]*n_joint

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
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'q':self.q,
            'dq':self.dq
        }


class Robot(Base):
    def __init__(self, n_joint, prefix):
        self.n_joint = n_joint
        self.floating_base = f'{prefix}IOs'
        self.arm = Manipulator(n_joint, prefix)
        self.ned_pose = [0] * 7
        self.body_vel = [0] * 6
        self.prefix = prefix

    def update_state(self, msg: DynamicJointState):
        self.arm.update_state(msg)
        self.ned_pose = self.get_interface_value(
            msg,
            [self.floating_base] * 7,
            [
                Axis_Interface_names.floating_base_x,
                Axis_Interface_names.floating_base_y,
                Axis_Interface_names.floating_base_z,
                Axis_Interface_names.floating_base_iw,
                Axis_Interface_names.floating_base_ix,
                Axis_Interface_names.floating_base_iy,
                Axis_Interface_names.floating_base_iz
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

    def get_state(self) -> Dict:
        xq = self.arm.get_state()
        xq['name'] = self.prefix
        xq['pose'] = self.ned_pose
        xq['body_vel'] =self.body_vel
        return xq

