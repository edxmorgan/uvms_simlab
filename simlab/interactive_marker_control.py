#!/usr/bin/env python3
import numpy as np
np.float = float  # Patch NumPy to satisfy tf_transformations' use of np.float

import copy
import math
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from uvms_interfaces.msg import Command
from robot import Robot
import tf2_ros
from tf_transformations import quaternion_multiply, quaternion_from_euler
from geometry_msgs.msg import TransformStamped


def quaternion_to_euler(orientation):
    # Convert geometry_msgs Quaternion to Euler angles (roll, pitch, yaw) using SciPy.
    # Note: geometry_msgs Quaternion ordering is [x, y, z, w].
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    r = R.from_quat(quat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return roll, pitch, yaw

def normalize_angle(angle):
    # Normalize an angle to [-pi, pi]
    return (angle + math.pi) % (2 * math.pi) - math.pi

class BasicControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls',
                         automatically_declare_parameters_from_overrides=True)
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Get parameter values
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.record = self.get_parameter('record_data').value
        self.controllers = self.get_parameter('controllers').value
        self.get_logger().info(f"robots controllers : {self.controllers}")
        self.get_logger().info(f"robot prefixes found in task node: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"robots total number of commands : {self.total_no_efforts}")

        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.uvms_publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        frequency = 500  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)
        self.get_logger().info("CoverageTask node has been initialized with optimal control.")

        initial_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])
        self.robots = [Robot(self, k, 4, prefix, initial_pos, self.record)
                       for k, prefix in enumerate(self.robots_prefix)]

        # Define your task-space boundaries as a class attribute or constant.
        self.TASK_SPACE_BOX = {
            'x_min': 0.0, 'x_max': 0.4,
            'y_min': -0.4, 'y_max': 0.4,
            'z_min': -0.4,  'z_max': 0.4
        }

        # Internal attributes for planning.
        # In your __init__ method:
        self.last_marker_pose = Pose()
        self.last_marker_pose.position.x = 0.0
        self.last_marker_pose.position.y = 0.0
        self.last_marker_pose.position.z = 0.0
        self.last_marker_pose.orientation.x = 0.0
        self.last_marker_pose.orientation.y = 0.0
        self.last_marker_pose.orientation.z = 0.0
        self.last_marker_pose.orientation.w = 1.0

        self.selected_robot_index = None  # Which robot is being targeted
        self.execute_plan = False         # Flag to trigger planning command

        # Create a new pose for the small box transform.
        self.arm_base_pose = Pose()
        self.arm_base_pose.position.x = 0.19
        self.arm_base_pose.position.y = 0.0
        self.arm_base_pose.position.z = -0.12

        # Convert the given Euler angles to a quaternion.
        r = R.from_euler('xyz', [3.142, 0.0, 0.0])
        q = r.as_quat()  # returns [x, y, z, w]
        self.arm_base_pose.orientation.x = q[0]
        self.arm_base_pose.orientation.y = q[1]
        self.arm_base_pose.orientation.z = q[2]
        self.arm_base_pose.orientation.w = q[3]

        # Create the interactive marker server and menu handler.
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")
        self.menu_handler = MenuHandler()

        # Set up the menu entries:
        # Menu entry ID 1: "execute" command.
        self.menu_handler.insert("execute", callback=self.processFeedback)
        sub_menu_handle = self.menu_handler.insert("Robots")
        # For each robot, add a plan option. (Assuming these get IDs starting at 3.)
        for prefix in self.robots_prefix:
            self.menu_handler.insert(f"{prefix} plan", parent=sub_menu_handle, callback=self.processFeedback)

        self.base_frame = "base_link"
        self.marker_frame = "marker_frame"
        self.arm_base_frame = "arm_frame"

        # Create a 10-DOF uvms marker with controls.
        self.make_UVMS_Dof_Marker(name = 'uvms marker', 
                            description = 'interactive marker for controlling vehicle', 
                            frame_id = self.base_frame, 
                            robot = 'uvms', 
                            fixed = False, 
                            interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D,
                            initial_position = self.last_marker_pose.position, 
                            callback = self.processFeedback, 
                            show_6dof=True)


        
        self.server.applyChanges()

 

    def timer_callback(self):
        self.broadcast_pose(self.last_marker_pose, self.base_frame, self.marker_frame)
        command_msg = Command()
        command_msg.command_type = self.controllers
        command_msg.acceleration.data = []
        command_msg.twist.data = []
        command_msg.pose.data = []

        for k, robot in enumerate(self.robots):
            state = robot.get_state()
            if state['status'] == 'active':
                command_msg.acceleration.data.extend([0.0]*self.no_efforts)
                command_msg.twist.data.extend([0.0]*self.no_efforts)
                robot.publish_robot_path()
                # If executing plan for the selected robot.
                if self.execute_plan and (k == self.selected_robot_index) and (self.last_marker_pose is not None):
                    planned = self.last_marker_pose
                    # Extract NWU pose values from the planned marker pose.
                    x_nwu = planned.position.x
                    y_nwu = planned.position.y
                    z_nwu = planned.position.z
                    roll_nwu, pitch_nwu, yaw_nwu = quaternion_to_euler(planned.orientation)

                    # Convert position from NWU to NED.
                    x_ned = x_nwu
                    y_ned = -y_nwu
                    z_ned = -z_nwu

                    # Convert orientation from NWU to NED.
                    # (Here we invert pitch and yaw; roll remains the same.)
                    # These are our absolute target angles in NED.
                    raw_roll_ned = roll_nwu
                    raw_pitch_ned = -pitch_nwu
                    raw_yaw_ned = -yaw_nwu

                    # Get current orientation (assuming state['pose'][3:6] are [roll, pitch, yaw] in NED).
                    curr_roll, curr_pitch, curr_yaw = state['pose'][3:6]

                    # Compute minimal differences.
                    delta_roll = normalize_angle(raw_roll_ned - curr_roll)
                    delta_pitch = normalize_angle(raw_pitch_ned - curr_pitch)
                    delta_yaw = normalize_angle(raw_yaw_ned - curr_yaw)

                    # The new target angles are the current ones plus the minimal difference.
                    target_roll = curr_roll + delta_roll
                    target_pitch = curr_pitch + delta_pitch
                    target_yaw = curr_yaw + delta_yaw

                    # self.get_logger().debug(
                    #     f"Executing plan for robot {self.robots_prefix[k]}: NWU pose: "
                    #     f"({x_nwu:.2f}, {y_nwu:.2f}, {z_nwu:.2f}, {roll_nwu:.2f}, {pitch_nwu:.2f}, {yaw_nwu:.2f}) | "
                    #     f"Raw NED: ({x_ned:.2f}, {y_ned:.2f}, {z_ned:.2f}, {raw_roll_ned:.2f}, {raw_pitch_ned:.2f}, {raw_yaw_ned:.2f}) | "
                    #     f"Target NED: ({x_ned:.2f}, {y_ned:.2f}, {z_ned:.2f}, {target_roll:.2f}, {target_pitch:.2f}, {target_yaw:.2f})"
                    # )

                    q0, q1, q2, q3 = state['q']
                    command_msg.pose.data.extend([x_ned, y_ned, z_ned, target_roll, target_pitch, target_yaw, q0, q1, q2, q3, 0.0])

                    # For error check, compare current NWU state with the planned NWU target.
                    current_pos = np.array(state['pose'])
                    target_pos = np.array([x_ned, y_ned, z_ned, target_roll, target_pitch, target_yaw])
                    error = np.linalg.norm(current_pos - target_pos)
                    self.get_logger().debug(f"Target error: {error}")
                    if error < 0.1:  # threshold in meters
                        self.get_logger().info("Target reached; resetting execution flag.")
                        self.execute_plan = False
                else:
                    [x, y, z, r, p, y_angle] = state['pose']
                    q0, q1, q2, q3 = state['q']
                    command_msg.pose.data.extend([x, y, z, r, p, y_angle, q0, q1, q2, q3, 0.0])
        self.uvms_publisher_.publish(command_msg)

    def processFeedback(self, feedback):
        s = f"Feedback from marker '{feedback.marker_name}' / control '{feedback.control_name}'"
        mp = ""
        if feedback.mouse_point_valid:
            mp = f" at {feedback.mouse_point.x}, {feedback.mouse_point.y}, {feedback.mouse_point.z} in frame {feedback.header.frame_id}"
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self.get_logger().debug(s + ": button click" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            self.get_logger().debug(s + f": menu item {feedback.menu_entry_id} clicked" + mp + ".")
            if feedback.menu_entry_id == 1:
                # For execute, update the planned pose if available.
                if feedback.pose:
                    self.last_marker_pose = feedback.pose
                if self.selected_robot_index is not None and self.last_marker_pose is not None:
                    self.execute_plan = True
                    self.get_logger().info(
                        f"Execute clicked: plan will be applied to robot {self.robots_prefix[self.selected_robot_index]}."
                    )
                else:
                    self.get_logger().warn("Execute clicked but no robot was selected or no planned pose available.")
            else:
                # Assume menu entry IDs for robot selection start at 3.
                robot_index = feedback.menu_entry_id - 3
                if 0 <= robot_index < len(self.robots_prefix):
                    self.selected_robot_index = robot_index
                    self.get_logger().info(f"Robot {self.robots_prefix[robot_index]} selected for planning.")
                else:
                    self.get_logger().warn("Invalid robot selection from menu.")
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            if feedback.pose.position.z > 0.0:
                self.get_logger().debug(
                    s + f": pose changed with z above 0 ({feedback.pose.position.z}); clamping to 0."
                )
                feedback.pose.position.z = 0.0
                self.server.setPose(feedback.marker_name, feedback.pose)
            else:
                self.get_logger().debug(s + ": pose changed")
            self.last_marker_pose = feedback.pose
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.get_logger().debug(s + ": mouse down" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.get_logger().debug(s + ": mouse up" + mp + ".")
        self.server.applyChanges()

    # def processConstrainedFeedback(self, feedback):
    #     """
    #     Feedback callback for the constrained marker.
    #     It clamps the position so that it remains within TASK_SPACE_BOX.
    #     """
    #     s = f"Constrained marker feedback from '{feedback.marker_name}'"
    #     if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
    #         new_pose = feedback.pose

    #         # Clamp the position within task space bounds.
    #         new_pose.position.x = max(self.TASK_SPACE_BOX['x_min'],
    #                                     min(new_pose.position.x, self.TASK_SPACE_BOX['x_max']))
    #         new_pose.position.y = max(self.TASK_SPACE_BOX['y_min'],
    #                                     min(new_pose.position.y, self.TASK_SPACE_BOX['y_max']))
    #         new_pose.position.z = max(self.TASK_SPACE_BOX['z_min'],
    #                                     min(new_pose.position.z, self.TASK_SPACE_BOX['z_max']))

    #         self.get_logger().debug(
    #             f"{s}: Clamped position to ({new_pose.position.x:.3f}, {new_pose.position.y:.3f}, {new_pose.position.z:.3f})"
    #         )
    #         # Update the pose on the server.
    #         self.server.setPose(feedback.marker_name, new_pose)
    #     # You can add additional processing for BUTTON_CLICK, MENU_SELECT, etc.
    #     self.server.applyChanges()

    def makeBox(self, fixed, scale, marker_type, initial_pose):
        marker = Marker()
        marker.type = marker_type
        marker.pose = initial_pose
        marker.scale.x = scale * 0.25
        marker.scale.y = scale * 0.25
        marker.scale.z = scale * 0.25

        if fixed:
            # red
            marker.color.r = 1.0 
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            # gray
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 1.0
        return marker

    def makeBoxControl(self, msg, fixed, interaction_mode, marker_type, scale = 1.0, show_6dof =False, initial_pose = Pose()):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeBox(fixed, scale, marker_type, initial_pose))
        control.interaction_mode = interaction_mode
        msg.controls.append(control)

        if show_6dof:

            # control = InteractiveMarkerControl()
            # control.orientation.w = 1.0
            # control.orientation.x = 1.0
            # control.orientation.y = 0.0
            # control.orientation.z = 0.0
            # control.name = "roll"
            # control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            # if fixed:
            #     control.orientation_mode = InteractiveMarkerControl.FIXED
            # msg.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 1.0
            control.orientation.y = 0.0
            control.orientation.z = 0.0
            control.name = "surge"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "yaw"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "heave"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

            # control = InteractiveMarkerControl()
            # control.orientation.w = 1.0
            # control.orientation.x = 0.0
            # control.orientation.y = 0.0
            # control.orientation.z = 1.0
            # control.name = "pitch"
            # control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            # if fixed:
            #     control.orientation_mode = InteractiveMarkerControl.FIXED
            # msg.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 0.0
            control.orientation.z = 1.0
            control.name = "sway"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)
        return control

    def make_UVMS_Dof_Marker(self, name, description, frame_id, robot, fixed, interaction_mode, initial_position, callback, 
                       show_6dof=False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.pose.position = initial_position
        int_marker.scale = 1.0
        int_marker.name = name
        int_marker.description = description
        self.makeBoxControl(int_marker, fixed, interaction_mode, Marker.CUBE, int_marker.scale, show_6dof, Pose())

        if robot == 'uvms':
            self.makeBoxControl(int_marker, True, InteractiveMarkerControl.NONE, Marker.CUBE, 0.2, False ,self.arm_base_pose)
            initial_task_pose = self.compute_end_effector_pose(self.arm_base_pose)
            self.makeBoxControl(int_marker, False, InteractiveMarkerControl.MOVE_ROTATE_3D, Marker.SPHERE, 0.2, True, initial_task_pose)

        # Add a menu control to the marker.
        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.MENU
        menu_control.name = "robots_control_menu"
        menu_control.description = "target"
        menu_control.always_visible = True
        int_marker.controls.append(copy.deepcopy(menu_control))

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, callback)
        self.menu_handler.apply(self.server, int_marker.name)

    def compute_end_effector_pose(self, arm_base_pose):
        # Extract arm base position and orientation.
        base_position = np.array([
            arm_base_pose.position.x,
            arm_base_pose.position.y,
            arm_base_pose.position.z
        ])
        base_orientation = [arm_base_pose.orientation.x,
                            arm_base_pose.orientation.y,
                            arm_base_pose.orientation.z,
                            arm_base_pose.orientation.w]
        
        relative_offset_position = np.array([0.1, 0.0, 0.0])
        relative_offset_orientation = quaternion_from_euler(0.0, 0.0, 0.0)  # [x, y, z, w]

        # For position: Rotate the relative offset by the arm base’s orientation.
        rotation_matrix = R.from_quat(base_orientation).as_matrix()
        offset_rotated = rotation_matrix.dot(relative_offset_position)
        end_effector_position = base_position + offset_rotated

        # For orientation: Combine (multiply) the arm base’s orientation with the relative offset orientation.
        end_effector_orientation = quaternion_multiply(base_orientation, relative_offset_orientation)

        # Construct a new Pose.
        end_effector_pose = Pose()
        end_effector_pose.position.x, end_effector_pose.position.y, end_effector_pose.position.z = end_effector_position
        end_effector_pose.orientation.x, end_effector_pose.orientation.y, end_effector_pose.orientation.z, end_effector_pose.orientation.w = end_effector_orientation

        return end_effector_pose

    def broadcast_pose(self, pose, parent_frame, child_frame):
        """
        Broadcasts the given pose as a transform from parent_frame to child_frame.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Set translation from the pose position.
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z

        # Set rotation from the pose orientation.
        t.transform.rotation = pose.orientation

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().debug(
            f"Broadcasting transform from {parent_frame} to {child_frame}: "
            f"translation=({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = BasicControlsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
