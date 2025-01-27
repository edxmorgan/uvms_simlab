import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
from control_msgs.msg import DynamicJointState
from robot import Robot
from task import Task
import ament_index_python
import os
import casadi as ca
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class CoverageTask(Node):
    def __init__(self):
        super().__init__('coverage_task',
                          automatically_declare_parameters_from_overrides=True)
        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        ref_intg_path = os.path.join(package_share_directory, 'ref_intg.casadi')
        j_uvms_path = os.path.join(package_share_directory, 'J_uvms.casadi')

        self.ref_intg_eval = ca.Function.load(ref_intg_path)
        self.J_uvms = ca.Function.load(j_uvms_path) # ned tf

        self.subscription = self.create_subscription(
            DynamicJointState,
            'dynamic_joint_states',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Get parameter values
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        
        self.get_logger().info(f"robot prefixes found in task node: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"robots total number of commands : {self.total_no_efforts}")

        self.robots = [Robot(4, prefix) for prefix in self.robots_prefix]

        self.ref_acc = np.zeros(11)
        self.ref_vel = np.zeros(11)
        self.ref_pos = np.array([3.0, 0.0, 5.0, 0,0,0, 3.1, 0.7, 0.4, 2.1, 0.0])

        self.path_publisher = self.create_publisher(Path, '/Path', 10)
        self.uvms_publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', 10)

        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)
        self.get_logger().info("CoverageTask node has been initialized with optimal control.")

        # Initialize path poses
        self.path_poses = []
        self.trajectory_twist = []
        self.trajectory_poses = []
        self.MAX_POSES = 10000

    def timer_callback(self):
        states = [robot.get_state() for robot in self.robots]
        if states[0]['status']=='active':
            configuration = states[0]['pose'] + states[0]['q']
            t = states[0]['sim_time']
            dt = states[0]['dt']

            J_UVMS_REF = self.J_uvms(self.ref_pos[3:6])
            J_UVMS_REF_np = J_UVMS_REF.full()

            self.ref_vel = Task.square_velocity_uv_ref(self, t, T_side=50.0, speed=0.1, manput=False).flatten()

            v_ned_ref = J_UVMS_REF_np@self.ref_vel[:-1]

            self.ref_pos = self.ref_intg_eval(self.ref_pos[:-1], v_ned_ref, dt).full().flatten().tolist() + [0.0]

            # Publish the reference path to RViz
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = f"{self.robots[0].prefix}map"  # Set to your appropriate frame

            # Create PoseStamped from ref_pos
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(self.ref_pos[0])
            pose.pose.position.y = -float(self.ref_pos[1])
            pose.pose.position.z = -float(self.ref_pos[2])
            pose.pose.orientation.w = 1.0  # No rotation

            # Accumulate poses
            self.path_poses.append(pose)
            path_msg.poses = self.path_poses

            # Limit the number of poses and twists
            if len(self.path_poses) > self.MAX_POSES:
                self.path_poses.pop(0)
            self.path_publisher.publish(path_msg)

            # Accumulate reference trajectory
            self.trajectory_twist.append(self.ref_vel.tolist().copy())  # Append a copy of the reference velocity
            self.trajectory_poses.append(self.ref_pos.copy())

            # self.get_logger().info(f"{len(self.trajectory_twist)}")
            if len(self.trajectory_twist) > 500:

                command_msg = Command()
                command_msg.command_type = "optimal"
                command_msg.acceleration.data = self.ref_acc.tolist()
                command_msg.twist.data = self.trajectory_twist[0]
                command_msg.pose.data = self.trajectory_poses[0]

                self.trajectory_twist.pop(0)
                self.trajectory_poses.pop(0)
            
                # Publish the command
                self.uvms_publisher_.publish(command_msg)



    def listener_callback(self, msg: DynamicJointState):
        for robot in self.robots:
            robot.update_state(msg)


    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    coveragetask = CoverageTask()

    try:
        rclpy.spin(coveragetask)
    except KeyboardInterrupt:
        coveragetask.get_logger().info('CoverageTask node stopped by KeyboardInterrupt.')
    finally:
        coveragetask.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()