#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

# Import ROS2 QoS settings and message type.
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from uvms_interfaces.msg import Command

# Import your robot class
from robot import Robot


class Experimental_Node(Node):
    def __init__(self):
        super().__init__('exp_node',
                         automatically_declare_parameters_from_overrides=True)

        # Retrieve parameters (e.g. number of robots, efforts, and robot prefixes).
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.record = self.get_parameter('record_data').value
        self.controllers = self.get_parameter('controllers').value

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        # Initialize robots (make sure your Robot class is defined properly).
        initial_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])
        self.robots = [Robot(self, k, 4, prefix, initial_pos, self.record) for k, prefix in enumerate(self.robots_prefix)]
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.uvms_publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)
        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)


    def timer_callback(self):
        command_msg = Command()
        command_msg.command_type = self.controllers
        command_msg.acceleration.data = [0.0]*11
        command_msg.twist.data = [0.0]*11
        command_msg.pose.data = []
        for robot in self.robots:
            # robot.publish_robot_path()
            ref0= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0]
            ref1= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 3.0, 4.0, 0.0]

            ref = ref1
            command_msg.pose.data.extend(ref)
            robot.write_data_to_file(ref)
        # Publish the command
        # self.get_logger().info(f'{command_msg.pose.data}')
        self.uvms_publisher_.publish(command_msg)

    def destroy_node(self):
        super().destroy_node()


###############################################################################
# Main entry point.
###############################################################################
def main(args=None):
    rclpy.init(args=args)
    exp_node = Experimental_Node()
    try:
        rclpy.spin(exp_node)
    except KeyboardInterrupt:
        exp_node.get_logger().info('Experimental node stopped by KeyboardInterrupt.')
    finally:
        exp_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
