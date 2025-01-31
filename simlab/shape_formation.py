import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
from control_msgs.msg import DynamicJointState
from robot import Robot
from shape_task import ShapeFormationController
from rclpy.qos import QoSProfile, QoSHistoryPolicy
import tf2_ros

class ShapeFormationTask(Node):
    def __init__(self):
        super().__init__('shape_task',
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

        self.shape_formation = ShapeFormationController(0.1*np.ones((3,len(self.robots))))
        self.waypoint = np.array([[0,1,0],[0,0,0],[0,0,1],[1,0,0], [0.5,0.5,0]]).T

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.uvms_publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)
        self.get_logger().info("CoverageTask node has been initialized with optimal control.")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def timer_callback(self):
        command_msg = Command()
        command_msg.command_type = "pid"
        command_msg.acceleration.data = []
        command_msg.twist.data = []
        command_msg.pose.data = []
        current_states = np.zeros((3,5))

        for i, robot in enumerate(self.robots):
            state = robot.get_state()
            current_states[:,i] = state['pose'][0:3]

        # self.get_logger().info(f'current_states {current_states}')
        cmd_vel = self.shape_formation.get_control_input(current_states, self.waypoint)
        # self.get_logger().info(f'cmd_vel {cmd_vel}')

        for i, robot in enumerate(self.robots):
            state = robot.get_state()
            if state['status']=='active':
                task = np.zeros((11,))
                task[0:3] = cmd_vel[:,i].reshape(3,)
                robot.set_robot_goals(task, False)
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
    shapeTask = ShapeFormationTask()

    try:
        rclpy.spin(shapeTask)
    except KeyboardInterrupt:
        shapeTask.get_logger().info('shapeTask node stopped by KeyboardInterrupt.')
    finally:
        shapeTask.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()