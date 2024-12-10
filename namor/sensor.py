import rclpy
from rclpy.node import Node
from control_msgs.msg import DynamicJointState

class Joint_Names:
    alpha_axis_a = 'robot_real__axis_a'
    alpha_axis_b = 'robot_real__axis_b'
    alpha_axis_c = 'robot_real__axis_c'
    alpha_axis_d = 'robot_real__axis_d'
    alpha_axis_e = 'robot_real__axis_e'
    floating_base = 'robot_real_IOs'


class Axis_Interface_names:
    position = 'position'
    floating_base_x = 'position.x'
    floating_base_y = 'position.y'
    floating_base_z = 'position.z'
    floating_base_iw = 'orientation.w'
    floating_base_ix = 'orientation.x'
    floating_base_iy = 'orientation.y'
    floating_base_iz = 'orientation.z'


class SystemSensor(Node):
    def __init__(self):
        super().__init__('sensor')

        self.subscription = self.create_subscription(
            DynamicJointState,
            'dynamic_joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.q = None
        self.ned_T = None


    def listener_callback(self, msg: DynamicJointState):
        # Retrieve current joint positions and velocities
        self.q = self.get_interface_value(
            msg,
            [Joint_Names.alpha_axis_e,
             Joint_Names.alpha_axis_d,
             Joint_Names.alpha_axis_c,
             Joint_Names.alpha_axis_b],
            [Axis_Interface_names.position] * 4
        )
        self.ned_T = self.get_interface_value(
            msg,
            [Joint_Names.floating_base]* 7,
            [Axis_Interface_names.floating_base_x,
             Axis_Interface_names.floating_base_y,
             Axis_Interface_names.floating_base_z,
             Axis_Interface_names.floating_base_iw,
             Axis_Interface_names.floating_base_ix,
             Axis_Interface_names.floating_base_iy,
             Axis_Interface_names.floating_base_iz]
        )
        # self.get_logger().info(f'{self.q}, {self.ned_T}')

    def get_interface_value(self, window_item, joint_names, interface_names):
        names = window_item.joint_names
        return [
            window_item.interface_values[names.index(joint_name)].values[
                window_item.interface_values[names.index(joint_name)].interface_names.index(interface_name)
            ]
            for joint_name, interface_name in zip(joint_names, interface_names)
        ]

def main(args=None):
    rclpy.init(args=args)

    SystemSensorNode = SystemSensor()
    try:
        rclpy.spin(SystemSensorNode)
    except KeyboardInterrupt:
        SystemSensorNode.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
