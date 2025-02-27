#!/usr/bin/env python3
import copy
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler

class BasicControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls', automatically_declare_parameters_from_overrides=True)

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

        # Create the interactive marker server and menu handler
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")
        self.menu_handler = MenuHandler()

        # Set up the menu entries
        self.menu_handler.insert("robots execute", callback=self.processFeedback)
        sub_menu_handle = self.menu_handler.insert("Robots")
        for prefix in self.robots_prefix:
            sub_sub_menu_handle = self.menu_handler.insert(f"{prefix}execute", parent=sub_menu_handle, callback=self.processFeedback)

        # Create a 6-DOF marker using MOVE_ROTATE_3D with additional axis controls, combined with a menu.
        self.make6DofMarker(False, InteractiveMarkerControl.MOVE_ROTATE_3D, Point(x=0.0, y=0.0, z=0.0), show_6dof=True)

        # Apply all changes
        self.server.applyChanges()


    def processFeedback(self, feedback):
        s = "Feedback from marker '" + feedback.marker_name + "' / control '" + feedback.control_name + "'"
        mp = ""
        if feedback.mouse_point_valid:
            mp = (" at " + str(feedback.mouse_point.x) + ", " +
                str(feedback.mouse_point.y) + ", " +
                str(feedback.mouse_point.z) +
                " in frame " + feedback.header.frame_id)
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self.get_logger().info(s + ": button click" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            self.get_logger().info(s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            # Check if the marker's center (pose) is moving above z=0
            if feedback.pose.position.z > 0.0:
                self.get_logger().info(
                    s + ": pose changed with z above 0 (" +
                    str(feedback.pose.position.z) + "); clamping to 0."
                )
                # Clamp the z coordinate to zero
                feedback.pose.position.z = 0.0
                # Update the marker's pose on the server
                self.server.setPose(feedback.marker_name, feedback.pose)
            else:
                self.get_logger().info(s + ": pose changed")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.get_logger().info(s + ": mouse down" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.get_logger().info(s + ": mouse up" + mp + ".")
        self.server.applyChanges()


    def makeBox(self, msg):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = msg.scale * 0.25
        marker.scale.y = msg.scale * 0.25
        marker.scale.z = msg.scale * 0.25
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
        return marker

    def makeBoxControl(self, msg):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeBox(msg))
        msg.controls.append(control)
        return control

    def make6DofMarker(self, fixed, interaction_mode, position, show_6dof=False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "interactive controller"
        int_marker.description = "Whole-body Interactive Controller"
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if show_6dof:
            # Add additional axis controls (rotation and translation along X, Y, and Z)
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 1.0
            control.orientation.y = 0.0
            control.orientation.z = 0.0
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 1.0
            control.orientation.y = 0.0
            control.orientation.z = 0.0
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 0.0
            control.orientation.z = 1.0
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 0.0
            control.orientation.z = 1.0
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

        # Add a menu control to the same marker
        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.MENU
        menu_control.name = "robots_control_menu"
        menu_control.description = "options"
        menu_control.always_visible = True
        int_marker.controls.append(copy.deepcopy(menu_control))

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)

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
