import numpy as np
from typing import Dict
from control_msgs.msg import DynamicJointState

class Manipulator:
    def __init__():
        pass
    def initialize_mappings(self, msg: DynamicJointState):
        pass
    def update_state(self, msg: DynamicJointState):
        pass
    def get_state(self) -> Dict[str, np.ndarray]:
        pass


class Robot:
    def __init__():
        pass
    def initialize_mappings(self, msg: DynamicJointState):
        pass

    def update_state(self, msg: DynamicJointState):
        pass

    def get_state(self) -> Dict[str, Dict[str, np.ndarray]]:
        pass

    def get_interface_value(self, msg: DynamicJointState, dof_names: list, interface_names: list):
        names = msg.joint_names
        return [
            msg.interface_values[names.index(joint_name)].values[
                msg.interface_values[names.index(joint_name)].interface_names.index(interface_name)
            ]
            for joint_name, interface_name in zip(dof_names, interface_names)
        ]