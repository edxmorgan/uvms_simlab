import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
import ament_index_python
import os
import casadi as ca
import time  # Import time module to get system time in seconds

class ExcitationNode(Node):
    def __init__(self):
        super().__init__('excitation_node',
                          automatically_declare_parameters_from_overrides=True)
        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        
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
        
        self.start_time = time.time()  # Store the start time for elapsed time calculation
        self.excitation_duration = self.lkp3_eval(self.start_time)[1].__int__()
        self.lookup_t0=0
        self.lookup_t1=0
        self.lookup_t2=0
        self.lookup_t3=0
        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', 10)

        self.timer = self.create_timer(1.0 / 1000, self.timer_callback)
        self.compute_effort_timer = self.create_timer(1.0 / 200, self.compute_effort_callback)

        self.get_logger().info("Excitation node has been initialized.")


    def compute_effort_callback(self):
        # Calculate time_seconds as elapsed time in seconds
        time_seconds = (time.time() - self.start_time) % self.excitation_duration
        # self.get_logger().info(f"{time_seconds}")
        self.lookup_t3 = self.lkp3_eval(time_seconds)[0].__float__()

        self.lookup_t2 = self.lkp2_eval(time_seconds)[0].__float__()

        self.lookup_t1 = self.lkp1_eval(time_seconds)[0].__float__()

        self.lookup_t0 = self.lkp0_eval(time_seconds)[0].__float__()



    def timer_callback(self):
        # Create and publish the command message
        command_msg = Command()
        command_msg.command_type = "force"

        real_data = [0.0] * 5

        real_data[3] = self.lookup_t3

        real_data[2] = self.lookup_t2

        real_data[1] = self.lookup_t1

        real_data[0] = self.lookup_t0

        # Combine the data for all robots
        data = []
        
        # Add controlled x, y, z, roll, pitch, yaw
        data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6 elements
        data.extend(real_data)  # Add real manipulator joint positions (5 elements)

        other_data = [0.0] * (self.total_no_efforts - len(data))
        data.extend(other_data)

        # assert len(data) == self.total_no_efforts, f"Data length mismatch. Expected {self.total_no_efforts}, got {len(data)}"
        # Convert all data to float
        dataF = [float(value) for value in data]
        command_msg.input.data = dataF
        # Publish the command
        self.publisher_.publish(command_msg)

    def destroy_node(self):
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
