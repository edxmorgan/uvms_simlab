import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
import ament_index_python
import os
import casadi as ca

class ExcitationNode(Node):
    def __init__(self):
        super().__init__('excitation_node',
                          automatically_declare_parameters_from_overrides=True)
        package_share_directory = ament_index_python.get_package_share_directory(
                'namor')
        
        lookup0_eval_path = os.path.join(package_share_directory, 'lookup_function_0.casadi')
        lookup1_eval_path = os.path.join(package_share_directory, 'lookup_function_1.casadi')
        lookup2_eval_path = os.path.join(package_share_directory, 'lookup_function_2.casadi')
        lookup3_eval_path = os.path.join(package_share_directory, 'lookup_function_3.casadi')

        self.lkp0_eval = ca.Function.load(lookup0_eval_path)
        self.lkp1_eval = ca.Function.load(lookup1_eval_path)
        self.lkp2_eval = ca.Function.load(lookup2_eval_path)
        self.lkp3_eval = ca.Function.load(lookup3_eval_path)

        # Get parameter values
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value

        self.total_no_efforts = self.no_robot * self.no_efforts

        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', 10)
        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

        self.get_logger().info("Excitation node has been initialized.")


    def timer_callback(self):
        # Create and publish the command message
        command_msg = Command()
        command_msg.command_type = "force"

        real_data = [0.0] * 5

        real_data[3] = 0.0

        real_data[2] = 0.0

        real_data[1] = 0.0

        real_data[0] = 0.0

        # Combine the data for all robots
        data = []
        
        # Add controlled x, y, z, roll, pitch, yaw
        data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6 elements
        data.extend(real_data)  # Add real manipulator joint positions (5 elements)

        other_data = [0.0] * (self.total_no_efforts - len(data))
        data.extend(other_data)

        assert len(data) == self.total_no_efforts, f"Data length mismatch. Expected {self.total_no_efforts}, got {len(data)}"
        # Convert all data to float
        dataF = [float(value) for value in data]
        command_msg.input.data = dataF

        # Publish the command
        self.publisher_.publish(command_msg)

    def destroy_node(self):
        self.listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    excit_node = ExcitationNode()

    try:
        rclpy.spin(excit_node)
    except KeyboardInterrupt:
        excit_node.get_logger().info('ExcitationNode node stopped by KeyboardInterrupt.')
    finally:
        excit_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
