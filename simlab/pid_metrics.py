import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
from control_msgs.msg import DynamicJointState
from robot import Robot
from rclpy.qos import QoSProfile, QoSHistoryPolicy

class pidMetrics(Node):
    def __init__(self):
        super().__init__('pid_metrics',
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
        self.waypoint = np.array([[0,-1,0],[0,0,0],[0,0,1],[1,0,0], [0.5,0.5,0]]).T
        self.robots = [Robot(self, 4, prefix, True) for prefix in self.robots_prefix]

        self.loop_count = 0

        self.cached_point = None

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

        # Increase the loop counter
        self.loop_count += 1

        for i, robot in enumerate(self.robots):
            state = robot.get_state()
            if state['status']=='active':
                robot.publish_robot_path()

                # joint_min = {-1000, -1000, -1000,  -1000, -1000, -1000,  1, 0.01, 0.01, 0.01};
                # joint_max = {1000, 1000, 1000,  1000, 1000, 1000,   5.50, 3.40, 3.40, 5.70};

                command_msg.acceleration.data.extend([0,0,0 ,0,0,0, 0,0,0,0,0])
                command_msg.twist.data.extend([0,0,0 ,0,0,0, 0,0,0,0,0])
                ref = [2.0, 1.0, 1.0,  1.0]
                # ref = [ 4.0, 3.0, 3.0, 4.0]
                robot.write_data_to_file(ref)
                command_msg.pose.data.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ref[0], ref[1], ref[2], ref[3], 0.0])

        # Publish the command
        self.uvms_publisher_.publish(command_msg)

    def listener_callback(self, msg: DynamicJointState):
        for robot in self.robots:
            robot.update_state(msg)


    def destroy_node(self):
        for i, robot in enumerate(self.robots):
            robot.close_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    pidmetrics = pidMetrics()

    try:
        rclpy.spin(pidmetrics)
    except KeyboardInterrupt:
        pidmetrics.get_logger().info('pidmetrics node stopped by KeyboardInterrupt.')
    finally:
        pidmetrics.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()