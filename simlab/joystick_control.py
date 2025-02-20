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

        sim_gain = 0.5
        real_gain = 5

        gain = real_gain

        # Gains for different DOFs
        self.max_torque = gain*1.0             # for surge/sway
        self.heave_max_torque = gain*3.0      # for R2 "down" heave
        self.orient_max_torque = gain*0.2        # fOR ROLL, PITCH, YAW
    # --- Analog stick callbacks ---
    # Note: The pyPS4Controller library by default does not provide a combined move 
    # event for the analog sticks. If your version does support on_L3_move and on_R3_move,
    # these callbacks will be used. Otherwise, you may need to override the directional events 
    # (on_L3_left, on_L3_right, etc.) and combine the axis data yourself.
    # ---------------- L2 / R2 for Heave ----------------
    #
    def on_L2_press(self, value):
        # L2 -> positive "up" heave
        scaled_value = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = -scaled_value

    def on_L2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0

    def on_R2_press(self, value):
        # R2 -> negative "down" heave
        scaled_value = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = scaled_value

    def on_R2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0


    #
    # ---------------- L3 Stick for Surge & Sway ----------------
    #
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


    #
    # ---------------- R3 Stick for Pitch & Yaw ----------------
    #
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

    #
    # ---------------- Roll via L1 / R1 ----------------
    #
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
        self.record = self.get_parameter('record_data').value

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        # Initialize robots (make sure your Robot class is defined properly).
        initial_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])
        self.robots = [Robot(self, 4, prefix, initial_pos, self.record) for prefix in self.robots_prefix]

        # Setup a publisher with a QoS profile.
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

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
            surge = self.rov_surge
            sway = self.rov_sway
            heave = self.rov_z
            roll = self.rov_roll
            pitch = self.rov_pitch
            yaw = self.rov_yaw

            e_joint= self.jointe
            d_joint= self.jointd
            c_joint= self.jointc
            b_joint= self.jointb
            a_joint= self.jointa


        rov_command = [surge, sway, heave, roll, pitch, yaw]
        manipulator_command = [e_joint, d_joint, c_joint, b_joint, a_joint] #[0]*5 # Manipulator command (unused).

        # Build the full command list for all robots.
        data = []
        for robot in self.robots:
            robot.write_data_to_file()
            # robot.publish_robot_path()  # Assumes each Robot instance handles its own publishing.
            data.extend(rov_command + manipulator_command)

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
