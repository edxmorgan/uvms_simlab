#!/usr/bin/env python3
"""
Copyright (c) 2011, Willow Garage, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Willow Garage, Inc. nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import copy
import math
from random import random

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler

from tf2_ros import TransformBroadcaster


class BasicControlsNode(Node):
    def __init__(self):
        super().__init__('basic_controls')
        # Create the interactive marker server and menu handler
        self.server = InteractiveMarkerServer(self, "basic_controls")
        self.menu_handler = MenuHandler()
        self.br = TransformBroadcaster(self)
        self.counter = 0

        # Create a timer to update the moving frame (10ms period)
        self.create_timer(0.01, self.frame_callback)

        # Set up the menu entries
        self.menu_handler.insert("First Entry", callback=self.processFeedback)
        self.menu_handler.insert("Second Entry", callback=self.processFeedback)
        sub_menu_handle = self.menu_handler.insert("Submenu")
        self.menu_handler.insert("First Entry", parent=sub_menu_handle, callback=self.processFeedback)
        self.menu_handler.insert("Second Entry", parent=sub_menu_handle, callback=self.processFeedback)

        # Create markers at different positions
        self.make6DofMarker(False, InteractiveMarkerControl.NONE, Point(x=-3.0, y=3.0, z=0.0), show_6dof=True)
        self.make6DofMarker(True, InteractiveMarkerControl.NONE, Point(x=0.0, y=3.0, z=0.0), show_6dof=True)
        self.makeRandomDofMarker(Point(x=3.0, y=3.0, z=0.0))
        self.make6DofMarker(False, InteractiveMarkerControl.ROTATE_3D, Point(x=-3.0, y=0.0, z=0.0), show_6dof=False)
        self.make6DofMarker(False, InteractiveMarkerControl.MOVE_ROTATE_3D, Point(x=0.0, y=0.0, z=0.0), show_6dof=True)
        self.make6DofMarker(False, InteractiveMarkerControl.MOVE_3D, Point(x=3.0, y=0.0, z=0.0), show_6dof=False)
        self.makeViewFacingMarker(Point(x=-3.0, y=-3.0, z=0.0))
        self.makeQuadrocopterMarker(Point(x=0.0, y=-3.0, z=0.0))
        self.makeChessPieceMarker(Point(x=3.0, y=-3.0, z=0.0))
        self.makePanTiltMarker(Point(x=-3.0, y=-6.0, z=0.0))
        self.makeMovingMarker(Point(x=0.0, y=-6.0, z=0.0))
        self.makeMenuMarker(Point(x=3.0, y=-6.0, z=0.0))

        # Apply all changes
        self.server.applyChanges()

    def frame_callback(self):
        # Create a TransformStamped message for the moving frame
        ts = TransformStamped()
        now = self.get_clock().now().to_msg()
        ts.header.stamp = now
        ts.header.frame_id = "base_link"
        ts.child_frame_id = "moving_frame"
        ts.transform.translation.x = 0.0
        ts.transform.translation.y = 0.0
        ts.transform.translation.z = math.sin(self.counter / 140.0) * 2.0
        ts.transform.rotation.x = 0.0
        ts.transform.rotation.y = 0.0
        ts.transform.rotation.z = 0.0
        ts.transform.rotation.w = 1.0
        self.br.sendTransform(ts)
        self.counter += 1

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
            self.get_logger().info(s + ": pose changed")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.get_logger().info(s + ": mouse down" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.get_logger().info(s + ": mouse up" + mp + ".")
        self.server.applyChanges()

    def alignMarker(self, feedback):
        # Adjust the marker's position to the nearest half unit along X and Y
        new_pose = feedback.pose
        new_pose.position.x = round(new_pose.position.x - 0.5) + 0.5
        new_pose.position.y = round(new_pose.position.y - 0.5) + 0.5
        self.get_logger().info(
            f"{feedback.marker_name}: aligning position = {feedback.pose.position.x},"
            f"{feedback.pose.position.y},{feedback.pose.position.z} to "
            f"{new_pose.position.x},{new_pose.position.y},{new_pose.position.z}"
        )
        self.server.setPose(feedback.marker_name, new_pose)
        self.server.applyChanges()

    @staticmethod
    def rand(min_, max_):
        return min_ + random() * (max_ - min_)

    def makeBox(self, msg):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = msg.scale * 0.45
        marker.scale.y = msg.scale * 0.45
        marker.scale.z = msg.scale * 0.45
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

    def saveMarker(self, int_marker):
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)

    def make6DofMarker(self, fixed, interaction_mode, position, show_6dof=False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if fixed:
            int_marker.name += "_fixed"
            int_marker.description += "\n(fixed orientation)"

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: "MOVE_3D",
                InteractiveMarkerControl.ROTATE_3D: "ROTATE_3D",
                InteractiveMarkerControl.MOVE_ROTATE_3D: "MOVE_ROTATE_3D"
            }
            int_marker.name += "_" + control_modes_dict[interaction_mode]
            int_marker.description = "3D Control"
            if show_6dof:
                int_marker.description += " + 6-DOF controls"
            int_marker.description += "\n" + control_modes_dict[interaction_mode]

        if show_6dof:
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

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)

    def makeRandomDofMarker(self, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "6dof_random_axes"
        int_marker.description = "6-DOF\n(Arbitrary Axes)"
        self.makeBoxControl(int_marker)

        control = InteractiveMarkerControl()
        for _ in range(3):
            control.orientation.w = self.rand(-1, 1)
            control.orientation.x = self.rand(-1, 1)
            control.orientation.y = self.rand(-1, 1)
            control.orientation.z = self.rand(-1, 1)
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(copy.deepcopy(control))
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(copy.deepcopy(control))

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)

    def makeViewFacingMarker(self, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "view_facing"
        int_marker.description = "View Facing 6-DOF"

        control = InteractiveMarkerControl()
        control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.orientation.w = 1.0
        control.name = "rotate"
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.independent_marker_orientation = True
        control.name = "move"
        control.markers.append(self.makeBox(int_marker))
        control.always_visible = True
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)

    def makeQuadrocopterMarker(self, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "quadrocopter"
        int_marker.description = "Quadrocopter"
        self.makeBoxControl(int_marker)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE
        int_marker.controls.append(copy.deepcopy(control))
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)

    def makeChessPieceMarker(self, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "chess_piece"
        int_marker.description = "Chess Piece\n(2D Move + Alignment)"

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        int_marker.controls.append(copy.deepcopy(control))

        control.markers.append(self.makeBox(int_marker))
        control.always_visible = True
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)
        # Set a special callback for POSE_UPDATE events to align the marker
        self.server.setCallback(int_marker.name, self.alignMarker, InteractiveMarkerFeedback.POSE_UPDATE)

    def makePanTiltMarker(self, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "pan_tilt"
        int_marker.description = "Pan / Tilt"
        self.makeBoxControl(int_marker)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 1.0
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.orientation_mode = InteractiveMarkerControl.INHERIT
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)

    def makeMenuMarker(self, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "context_menu"
        int_marker.description = "Context Menu\n(Right Click)"

        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.MENU
        control.description = "Options"
        control.name = "menu_only_control"
        int_marker.controls.append(copy.deepcopy(control))

        marker = self.makeBox(int_marker)
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)

    def makeMovingMarker(self, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "moving_frame"
        int_marker.pose.position = position
        int_marker.scale = 1.0
        int_marker.name = "moving"
        int_marker.description = "Marker Attached to a\nMoving Frame"

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 1.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(copy.deepcopy(control))

        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.always_visible = True
        control.markers.append(self.makeBox(int_marker))
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.processFeedback)


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
