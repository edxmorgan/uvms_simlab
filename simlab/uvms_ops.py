import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
from control_msgs.msg import DynamicJointState
from robot import Robot
from task import Task
from rclpy.qos import QoSProfile, QoSHistoryPolicy


class OperationalSpace(Node):
    def __init__(self):
        super().__init__('operation_space',
                          automatically_declare_parameters_from_overrides=True)


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

        self.robots = [Robot(self, 4, prefix) for prefix in self.robots_prefix]


        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.uvms_publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)
        self.get_logger().info("CoverageTask node has been initialized with optimal control.")


    def timer_callback(self):
        command_msg = Command()
        command_msg.command_type = "pid"
        command_msg.acceleration.data = []
        command_msg.twist.data = []
        command_msg.pose.data = []

        for robot in self.robots:
            state = robot.get_state()
            if state['status']=='active':
                sim_t = state['sim_time']

                ref_body_vel = Task.square_velocity_ops_ref(self, t=sim_t, T_side=50.0, speed=0.5).tolist() #task

                robot.set_operation_space_goals(ref_body_vel, True)
                robot.publish_reference_path()
                robot.publish_ops_reference_path()
                robot.publish_robot_path()

                command_msg.acceleration.data.extend(robot.get_robot_goals('ref_acc'))
                command_msg.twist.data.extend(robot.get_robot_goals('ref_vel'))
                command_msg.pose.data.extend(robot.get_robot_goals('ref_pos'))

                if len(robot.trajectory_twist) > 500:
                    robot.trajectory_twist.pop(0)
                    robot.trajectory_poses.pop(0)
                
        # Publish the command
        self.uvms_publisher_.publish(command_msg)



    def listener_callback(self, msg: DynamicJointState):
        for robot in self.robots:
            robot.update_state(msg)


    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    opsSpaceTask = OperationalSpace()

    try:
        rclpy.spin(opsSpaceTask)
    except KeyboardInterrupt:
        opsSpaceTask.get_logger().info('opsSpaceTask node stopped by KeyboardInterrupt.')
    finally:
        opsSpaceTask.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()