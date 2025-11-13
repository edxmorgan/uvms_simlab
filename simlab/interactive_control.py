# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
import numpy as np
np.float = float  # Patch NumPy to satisfy tf_transformations' use of np.float

import copy
import rclpy
from rclpy.node import Node
import casadi as ca
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from robot import Robot
import tf2_ros
import ament_index_python
import os
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import ConvexHull
from alpha_reach import Params as alpha
from se3_ompl_planner import plan_se3_path
from fcl_checker import FCLWorld
from interactive_utils import *
from planner_markers import PathPlanner
from frame_utils import PoseX
from typing import List

class BasicControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls',
                         automatically_declare_parameters_from_overrides=True)

        # FCL for planning, env in world frame
        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        self.fcl_world = FCLWorld(urdf_string=urdf_string, world_frame='world', vehicle_radius=0.4)

        self.get_logger().info(f"Minimum coordinates (min_x, min_y, min_z): {self.fcl_world.min_coords}")
        self.get_logger().info(f"Maximum coordinates (max_x, max_y, max_z): {self.fcl_world.max_coords}")
        self.get_logger().info(f"Oriented Bounding Box corners: {self.fcl_world.obb_corners}")


        package_share_directory = ament_index_python.get_package_share_directory('simlab')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Example: get some parameters
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.record = self.get_parameter('record_data').value
        self.controllers = self.get_parameter('controllers').value
        self.total_no_efforts = self.no_robot * self.no_efforts

        viz_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
    
        )
        self.planner_marker_publisher = self.create_publisher(Marker, "planned_waypoints_marker", viz_qos)

        pointcloud_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.taskspace_pc_publisher_ = self.create_publisher(PointCloud2,'workspace_pointcloud',pointcloud_qos)
        self.rov_pc_publisher_ = self.create_publisher(PointCloud2, 'base_pointcloud', pointcloud_qos)

        workspace_pts_path = os.path.join(package_share_directory, 'manipulator/workspace.npy')
        self.workspace_pts = np.load(workspace_pts_path)
        self.workspace_hull = ConvexHull(self.workspace_pts)

        self.workspace_pts_list = self.workspace_pts.tolist()
        self.rov_ellipsoid_cl_pts = generate_rov_ellipsoid(a=0.3, b=0.3, c=0.2, num_points=10000)
        self.vehicle_body_hull = ConvexHull(self.rov_ellipsoid_cl_pts)

        # Combine clouds that represent the vehicle occupied volume
        # Use what you already have in this node
        all_pts = np.vstack([
            np.asarray(self.rov_ellipsoid_cl_pts, dtype=float),
            np.asarray(self.workspace_pts, dtype=float)
        ])

        planner_radius = compute_bounding_sphere_radius(all_pts, quantile=0.995, pad=0.03)
        self.get_logger().info(f"Planner robot approximation sphere radius set to {planner_radius:.3f} m")
        self.fcl_world.set_planner_radius(planner_radius)
        
        frequency = 500  # Hz
        self.create_timer(1.0 / frequency, self.timer_callback)
        self.create_timer(1.0 / frequency, lambda: self.fcl_world.update_from_tf(self.tf_buffer, rclpy.time.Time()))


        self.current_target_vehicle_marker_pose = Pose()
        self.current_target_vehicle_marker_pose.orientation.w = 1.0
        self.selected_robot_index = 0 # by default robot 0 is selected

        self.arm_base_pose = Pose()
        self.arm_base_pose.position.x = 0.19
        self.arm_base_pose.position.y = 0.0
        self.arm_base_pose.position.z = -0.12
        r = R.from_euler('xyz', [3.142, 0.0, 0.0])
        q = r.as_quat()
        (self.arm_base_pose.orientation.x,
         self.arm_base_pose.orientation.y,
         self.arm_base_pose.orientation.z,
         self.arm_base_pose.orientation.w) = q

        # Create marker server, menu handler
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")

        self.menu_handler = MenuHandler()
        self.menu_id_to_robot_index = {}

        self.execute_handle = self.menu_handler.insert("Plan & execute", callback=self.processFeedback)
        sub_menu_handle = self.menu_handler.insert("Robots")

        self.robots:List[Robot] = []
        for k, (prefix, controller) in enumerate(zip(self.robots_prefix, self.controllers)):
            robot_k = Robot(self, k, 4, prefix, self.record, controller)

            # unique planner per robot
            robot_k.planner = PathPlanner(self, ns=f"planner/{prefix}", base_id=k)

            # add a menu item for this robot and remember which handle maps to which index
            h = self.menu_handler.insert(f"Use {prefix}", parent=sub_menu_handle, callback=self.processFeedback)
            self.menu_id_to_robot_index[h] = k

            self.robots.append(robot_k)

        self.base_frame = "base_link"
        self.vehicle_marker_frame = "vehicle_marker_frame"
        self.endeffector_marker_frame = "endeffector_marker_frame"

        # Create markers
        self.uv_marker = make_UVMS_Dof_Marker(
            name='uv_marker',
            description='interactive marker for controlling vehicle',
            frame_id=self.base_frame,
            control_frame='uv',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.current_target_vehicle_marker_pose,
            scale=1.0,
            arm_base_pose=self.arm_base_pose,
            show_6dof=True,
            ignore_dof=['roll','pitch']
        )

        self.server.insert(self.uv_marker)
        self.server.setCallback(self.uv_marker.name, self.processFeedback)

        desired_q_orientation = [
                self.current_target_vehicle_marker_pose.orientation.x,
                self.current_target_vehicle_marker_pose.orientation.y,
                self.current_target_vehicle_marker_pose.orientation.z,
                self.current_target_vehicle_marker_pose.orientation.w
            ]

        # Unpack the Euler angles from the returned array.
        roll, pitch, yaw = R.from_quat(desired_q_orientation).as_euler('xyz', degrees=False)
        [self.q0_des, self.q1_des, self.q2_des, self.q3_des] = self.robots[0].arm.q_command
        self.n_int_est = ca.DM([self.current_target_vehicle_marker_pose.position.x,
                      self.current_target_vehicle_marker_pose.position.y,
                      self.current_target_vehicle_marker_pose.position.z,
                      roll,
                      pitch,
                      yaw,
                      self.q0_des,
                      self.q1_des,
                      self.q2_des, 
                      self.q3_des])
        temp_dm = Robot.uvms_Forward_kinematics(self.n_int_est, alpha.base_T0)
        self.last_valid_task_pose = dm_to_pose(temp_dm[4])

        self.task_marker = make_UVMS_Dof_Marker(
            name='task_marker',
            description='interactive marker for controlling endeffector',
            frame_id=self.vehicle_marker_frame,
            control_frame='task',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.last_valid_task_pose,
            scale=0.2,
            arm_base_pose=self.arm_base_pose,
            show_6dof=True,
            ignore_dof=['yaw']
        )

        self.server.insert(self.task_marker)
        self.server.setCallback(self.task_marker.name, self.processFeedback)

        # Add menu control
        menu_control = make_menu_control()
        self.uv_marker.controls.append(copy.deepcopy(menu_control))
        self.menu_handler.apply(self.server, self.uv_marker.name)
        self.server.applyChanges()

        self.header = Header()
        self.header.frame_id = self.vehicle_marker_frame

    def timer_callback(self):        
        stamp_now = self.get_clock().now().to_msg()

        t = get_broadcast_tf(stamp_now, self.current_target_vehicle_marker_pose, self.base_frame, self.vehicle_marker_frame)
        self.tf_broadcaster.sendTransform(t)
        
        self.header.stamp = stamp_now
        rov_cloud_msg = pc2.create_cloud_xyz32(self.header, self.workspace_pts_list)
        self.taskspace_pc_publisher_.publish(rov_cloud_msg)

        cloud_msg = pc2.create_cloud_xyz32(self.header, self.rov_ellipsoid_cl_pts)
        self.rov_pc_publisher_.publish(cloud_msg)

        for k, robot in enumerate(self.robots):
            k_planner = robot.planner
            if robot.prefix == self.robots_prefix[self.selected_robot_index]:
                if k_planner.planned_result and k_planner.planned_result['status']:
                    k_planner.update(
                        stamp=stamp_now,
                        frame_id=self.base_frame,
                        xyz_np=k_planner.planned_result["xyz"],
                        step=3,
                        wp_size=0.08,
                        goal_size=0.14,
                    )


            state = robot.get_state()
            if state['status'] == 'active':
                desired_body_acc = robot.body_acc_command + robot.arm.ddq_command
                desired_body_vel = robot.body_vel_command + robot.arm.dq_command
                robot.publish_robot_path()

                if robot.final_goal is not None and k_planner.planned_result and k_planner.planned_result['status']:
                    waypoints = k_planner.planned_result["xyz"]
                    quats = k_planner.planned_result["quat_wxyz"]

                    # Initialize index if not present
                    if not hasattr(robot, "wp_index"):
                        robot.wp_index = 0

                    # Use current waypoint
                    target_nwu = waypoints[robot.wp_index]
                    target_quat = quats[robot.wp_index]

                    # Convert to NED for control
                    target_pose = PoseX.from_pose(
                        xyz=target_nwu,
                        rot=target_quat,
                        rot_rep="quat_wxyz",
                        frame="NWU",
                    )
                    p_cmd_ned, rpy_cmd_ned = target_pose.get_pose(frame="NED", rot_rep="euler_xyz")

                    # Get the velocity-based yaw.
                    adjusted_yaw = robot.orient_towards_velocity()

                    # Compute current manifold errors
                    wp_err_trans, wp_err_rot, wp_err_joint, goal_err_trans, goal_err_rot = robot.compute_manifold_errors()
                    # self.get_logger().info(f"yaw now at pos err manifold {goal_err_rot}")
                    
                    goal_xyz_error = np.linalg.norm(goal_err_trans)


                    # Define a threshold error at which we start blending.
                    pos_blend_threshold = 1.1  # Adjust based on your system's scale

                    # Calculate the blend factor.
                    # When pos_error >= pos_blend_threshold, blend_factor will be 0 (full velocity_yaw).
                    # When pos_error == 0, blend_factor will be 1 (full target_yaw).
                    blend_factor = np.clip((pos_blend_threshold - goal_xyz_error) / pos_blend_threshold, 0.0, 1.0)

                    # If velocity_yaw is not available, simply use the target yaw.
                    if adjusted_yaw is None:
                        self.get_logger().info(f"adjusted_yaw is None")
                        final_yaw = rpy_cmd_ned[2]
                    else:
                        # Blend the yaw values: more weight to target_yaw as the position error decreases.
                        final_yaw = (1 - blend_factor) * adjusted_yaw + blend_factor * rpy_cmd_ned[2]

                    # self.get_logger().info(f"{final_yaw} yaw now at pos err {goal_xyz_error}")

                    robot.pose_command = [
                        float(p_cmd_ned[0]),
                        float(p_cmd_ned[1]),
                        float(p_cmd_ned[2]),
                        float(rpy_cmd_ned[0]),
                        float(rpy_cmd_ned[1]),
                        float(final_yaw),
                    ]

                    wp_pos_error = np.linalg.norm(wp_err_trans)
                    wp_rot_error = np.linalg.norm(wp_err_rot)

                    # Check if translation error small enough to move to next waypoint
                    if wp_pos_error < 5e-1 and wp_rot_error < 5e-1:
                        if robot.wp_index < len(waypoints) - 1:
                            robot.wp_index += 1
                            # self.get_logger().info(f"{robot.prefix} → waypoint {robot.wp_index}")
                        else:
                            # Reached final waypoint
                            self.get_logger().info(f"{robot.prefix} path complete", once=True)

                    robot.arm.q_command = [self.q0_des, self.q1_des, self.q2_des, self.q3_des]
                                            

            veh_state_vec = np.array(
                list(state['pose']) + list(state['body_vel']),
                dtype=float
            )
            # log to terminal
            # self.get_logger().info(f"robot command = {robot.pose_command}")

            cmd_body_wrench = robot.ll_controllers.vehicle_controller(
                state=veh_state_vec,
                target=np.array(robot.pose_command, dtype=float),
                dt=state["dt"]
            )

            # cmd_body_wrench = np.zeros(6)
            # cmd_body_wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
            # Arm PID
            cmd_arm_tau = robot.ll_controllers.arm_controller(
                q=state["q"],
                q_dot=state["dq"],
                q_ref=robot.arm.q_command,
                Kp=alpha.Kp,
                Ki=alpha.Ki,
                Kd=alpha.Kd,
                dt=state["dt"],
                u_max=alpha.u_max,
                u_min=alpha.u_min,
            )

            arm_tau_list = list(np.asarray(cmd_arm_tau, dtype=float).reshape(-1))
            # always produce 5 values, slice if longer, pad if shorter
            arm_tau_list = arm_tau_list[:5] + [0.0]

            robot.publish_commands(cmd_body_wrench, arm_tau_list)

            ref=robot.pose_command+robot.arm.q_command
            robot.write_data_to_file(ref)


    def processFeedback(self, feedback):
        # For uv_marker
        if feedback.marker_name == "uv_marker":
            if feedback.pose:
                if feedback.pose.position.z > 0.0:
                    feedback.pose.position.z = 0.0
                    self.server.setPose(feedback.marker_name, feedback.pose)
                    self.server.applyChanges()
                self.current_target_vehicle_marker_pose = feedback.pose
            if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
                if feedback.menu_entry_id == self.execute_handle:
                    if self.selected_robot_index is None or self.current_target_vehicle_marker_pose is None:
                        self.get_logger().warn("Execute clicked but robot selection or planned pose is missing.")
                        return
                    robot = self.robots[self.selected_robot_index]
                    state = robot.get_state()

                    # Build a pose from the current NED state, then query NWU for planning
                    pose_now = PoseX.from_pose(
                        xyz=np.array(state['pose'][0:3], float),
                        rot=np.array(state['pose'][3:6], float),   # roll, pitch, yaw
                        rot_rep="euler_xyz",
                        frame="NED",
                    )
                    start_xyz, start_quat_wxyz = pose_now.get_pose(frame="NWU", rot_rep="quat_wxyz")

                    gx = self.current_target_vehicle_marker_pose.position.x
                    gy = self.current_target_vehicle_marker_pose.position.y
                    gz = self.current_target_vehicle_marker_pose.position.z
                    goal_xyz = np.array([gx, gy, gz], float)

                    goal_quat_wxyz = np.array([
                        self.current_target_vehicle_marker_pose.orientation.w,
                        self.current_target_vehicle_marker_pose.orientation.x,
                        self.current_target_vehicle_marker_pose.orientation.y,
                        self.current_target_vehicle_marker_pose.orientation.z,
                    ], float)


                    goal_now = PoseX.from_pose(
                        xyz=np.array(goal_xyz, float),
                        rot=np.array(goal_quat_wxyz, float),   # roll, pitch, yaw
                        rot_rep="quat_wxyz",
                        frame="NWU",
                    )

                    # Goal from the UV marker is in NWU, convert that to NED for control and save it.
                    robot.final_goal = goal_now.get_pose(frame="NED", rot_rep="euler_xyz")
                    robot.wp_index = 0
   
                    k_planner = robot.planner
                    try:
                        # self.get_logger().info(
                        #     "Planning request\n"
                        #     f"  start_xyz         {start_xyz.tolist()}\n"
                        #     f"  start_quat_wxyz   {start_quat_wxyz.tolist()}\n"
                        #     f"  goal_xyz          {goal_xyz.tolist()}\n"
                        #     f"  goal_quat_wxyz    {goal_quat_wxyz.tolist()}\n"
                        # )


                        k_planner.planned_result = plan_se3_path(
                            self,
                            start_xyz=start_xyz,
                            start_quat_wxyz=start_quat_wxyz,
                            goal_xyz=goal_xyz,
                            goal_quat_wxyz=goal_quat_wxyz,
                            time_limit=1.0,
                            fcl_world=self.fcl_world,
                            safety_margin=0.1,
                        )
                        self.get_logger().info(
                            f"Planned path with {k_planner.planned_result['count']} states"
                        )
                    except Exception as e:
                        self.get_logger().error(f"Planner failed, {e}")
                        k_planner.planned_result = {
                                    "status":False,
                                    "message":"Planner did not find a solution"
                                }
                else:
                    # otherwise, a robot menu item was clicked
                    if feedback.menu_entry_id in self.menu_id_to_robot_index:
                        self.selected_robot_index = self.menu_id_to_robot_index[feedback.menu_entry_id]
                        self.get_logger().info(f"Robot {self.robots_prefix[self.selected_robot_index]} selected for planning.")

            elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                pass

        elif feedback.marker_name == "task_marker" and feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            task_point = np.array([feedback.pose.position.x,
                                feedback.pose.position.y,
                                feedback.pose.position.z])
            if is_point_valid(self.workspace_hull, self.vehicle_body_hull, task_point):
                self.last_valid_task_pose = feedback.pose
                relative_pose = get_relative_pose(self.arm_base_pose, self.last_valid_task_pose)
                self.q0_des, self.q1_des, self.q2_des = self.robots[self.selected_robot_index].arm.ik_solver([
                    relative_pose.position.x, relative_pose.position.y, relative_pose.position.z
                ], pose="underarm")
                self.get_logger().debug(
                    f"Task marker updated with IK: {self.q0_des, self.q1_des, self.q2_des, self.q3_des}"
                )
            else:
                # The task marker is at the boundary; compute the displacement since the last valid pose.
                dx = feedback.pose.position.x - self.last_valid_task_pose.position.x
                dy = feedback.pose.position.y - self.last_valid_task_pose.position.y
                dz = feedback.pose.position.z - self.last_valid_task_pose.position.z

                # Shift the uv_marker by this delta so that the task marker remains at the boundary.
                self.current_target_vehicle_marker_pose.position.x += dx
                self.current_target_vehicle_marker_pose.position.y += dy
                self.current_target_vehicle_marker_pose.position.z += dz

                # Update the uv_marker pose on the server.
                self.server.setPose("uv_marker", self.current_target_vehicle_marker_pose)
                self.server.applyChanges()

                # Reset the task marker back to the last valid pose (i.e. at the boundary).
                self.server.setPose("task_marker", self.last_valid_task_pose)
                self.server.applyChanges()

   

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
