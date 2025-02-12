#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import threading

# Import ROS2 QoS settings and message type.
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from uvms_interfaces.msg import Command

# Import the PS4 controller library.
from pyPS4Controller.controller import Controller

# Import your robot class (make sure you have this implemented elsewhere).
from robot import Robot


###############################################################################
# PS4 Controller subclass for ROS2 teleoperation.
#
# Mapping for ROV control:
#   - Left analog stick (L3): surge (forward/back) and sway (left/right)
#       (raw values normalized: (value/32767) * 20)
#   - Right analog stick (R3): pitch and yaw (normalized to ±20)
#   - L2 & R2 triggers: analog heave (vertical translation), normalized to ±20.
###############################################################################
class PS4Controller(Controller):
    def __init__(self, ros_node, **kwargs):
        super().__init__(**kwargs)
        # Save a reference to the ROS node to update shared variables.
        self.ros_node = ros_node

    # --- Analog stick callbacks ---
    # Note: The pyPS4Controller library by default does not provide a combined move 
    # event for the analog sticks. If your version does support on_L3_move and on_R3_move,
    # these callbacks will be used. Otherwise, you may need to override the directional events 
    # (on_L3_left, on_L3_right, etc.) and combine the axis data yourself.
    
    def on_L3_move(self, x, y):
        # Normalize raw x and y (expected range ±32767) to ±20.
        scaled_x = 20 * (x / 32767.0)
        scaled_y = 20 * (y / 32767.0)
        with self.ros_node.controller_lock:
            # For the ROV, we map: x -> sway and y -> surge.
            self.ros_node.rov_x = scaled_x
            self.ros_node.rov_y = scaled_y
        self.ros_node.get_logger().info(
            f"L3 move: scaled x = {scaled_x:.2f}, scaled y = {scaled_y:.2f}"
        )

    def on_R3_move(self, x, y):
        # Normalize raw x and y to ±20.
        scaled_yaw = 20 * (x / 32767.0)
        scaled_pitch = 20 * (y / 32767.0)
        with self.ros_node.controller_lock:
            # For the ROV, we map: x -> yaw and y -> pitch.
            self.ros_node.rov_yaw = scaled_yaw
            self.ros_node.rov_pitch = scaled_pitch
        self.ros_node.get_logger().info(
            f"R3 move: scaled yaw = {scaled_yaw:.2f}, scaled pitch = {scaled_pitch:.2f}"
        )

    # --- Trigger callbacks for heave ---
    def on_R2_press(self, value):
        # For R2, we assume the raw value is in [0, 32767] (downward heave).
        scaled_value = 20 * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = -scaled_value
        self.ros_node.get_logger().info(f"R2 pressed: Heave (down) = {-scaled_value:.2f}")

    def on_R2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0
        self.ros_node.get_logger().info("R2 released: Heave = 0")

    def on_L2_press(self, value):
        # For L2, we assume the raw value is in [0, 32767] (upward heave).
        scaled_value = 20 * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = scaled_value
        self.ros_node.get_logger().info(f"L2 pressed: Heave (up) = {scaled_value:.2f}")

    def on_L2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0
        self.ros_node.get_logger().info("L2 released: Heave = 0")


###############################################################################
# ROS2 Node that uses the PS4 controller for ROV teleoperation.
#
# The ROV command is built as follows:
#   - ROV Command (6 elements): [surge, sway, heave, roll, pitch, yaw]
#       surge  = - (left stick vertical)   (inverted so that pushing forward is positive)
#       sway   = left stick horizontal
#       heave  = analog value from triggers
#       roll   = 0.0 (unused)
#       pitch  = right stick vertical
#       yaw    = right stick horizontal
#
#   - Manipulator Command (5 elements): all zeros.
#
# Total command for each robot is 11 elements.
###############################################################################
class PS4TeleopNode(Node):
    def __init__(self):
        super().__init__('ps4_teleop_node',
                         automatically_declare_parameters_from_overrides=True)

        # Retrieve parameters (e.g. number of robots, efforts, and robot prefixes).
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        # Initialize robots (make sure your Robot class is defined properly).
        initial_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])
        self.robots = [Robot(self, 4, prefix, initial_pos) for prefix in self.robots_prefix]

        # Setup a publisher with a QoS profile.
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

        # Shared variables updated by the PS4 controller callbacks.
        self.controller_lock = threading.Lock()
        self.rov_x = 0.0      # Left stick horizontal (sway)
        self.rov_y = 0.0      # Left stick vertical (surge)
        self.rov_z = 0.0      # Heave from triggers
        self.rov_roll = 0.0   # Unused
        self.rov_pitch = 0.0  # Right stick vertical (pitch)
        self.rov_yaw = 0.0    # Right stick horizontal (yaw)

        # Instantiate the PS4 controller.
        # If you are not receiving analog stick events, try adjusting the event_format.
        self.ps4_controller = PS4Controller(
            ros_node=self,
            interface="/dev/input/js0",
            connecting_using_ds4drv=False,
            event_format="3Bh2b"  # Try "LhBB" if you experience mapping issues.
        )
        # Enable debug mode to print raw event data.
        self.ps4_controller.debug = True

        # Start the PS4 controller listener in a separate (daemon) thread.
        self.controller_thread = threading.Thread(target=self.ps4_controller.listen, daemon=True)
        self.controller_thread.start()

        self.get_logger().info("PS4 Teleop node initialized for ROV control with normalized scaling.")

    def timer_callback(self):
        # Create a new command message.
        command_msg = Command()
        command_msg.command_type = "force"

        # Safely acquire the latest controller values.
        with self.controller_lock:
            left_x = self.rov_x
            left_y = self.rov_y
            heave = self.rov_z
            pitch = self.rov_pitch
            yaw = self.rov_yaw

        # Map joystick values to ROV command.
        surge = -left_y   # Invert so that pushing forward is positive.
        sway = left_x
        roll = 0.0

        rov_command = [surge, sway, heave, roll, pitch, yaw]
        manipulator_command = [0.0] * 5  # Manipulator command (unused).

        # Build the full command list for all robots.
        data = []
        for robot in self.robots:
            robot.publish_robot_path()  # Assumes each Robot instance handles its own publishing.
            data.extend(rov_command + manipulator_command)

        # Adjust the data length if needed.
        current_length = len(data)
        if current_length < self.total_no_efforts:
            data.extend([0.0] * (self.total_no_efforts - current_length))
        elif current_length > self.total_no_efforts:
            self.get_logger().warning(
                f"Data length ({current_length}) exceeds total_no_efforts ({self.total_no_efforts}). Truncating data."
            )
            data = data[:self.total_no_efforts]

        # Ensure that the command has the expected number of elements.
        assert len(data) == self.total_no_efforts, (
            f"Data length mismatch. Expected {self.total_no_efforts}, got {len(data)}"
        )
        command_msg.force.data = [float(value) for value in data]

        # Publish the command.
        self.publisher_.publish(command_msg)

    def destroy_node(self):
        # Optionally, stop the PS4 controller listener here if needed.
        super().destroy_node()


###############################################################################
# Main entry point.
###############################################################################
def main(args=None):
    rclpy.init(args=args)
    teleop_node = PS4TeleopNode()
    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        teleop_node.get_logger().info('PS4 Teleop node stopped by KeyboardInterrupt.')
    finally:
        teleop_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
