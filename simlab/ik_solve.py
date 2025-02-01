import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
from control_msgs.msg import DynamicJointState
from robot import Robot
from task import Task
from rclpy.qos import QoSProfile, QoSHistoryPolicy


class iKtask(Node):
    def __init__(self):
        super().__init__('iK_task',
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

        for i, robot in enumerate(self.robots):
            state = robot.get_state()
            if state['status']=='active':
                robot.publish_robot_path()
                x= self.waypoint[0,i]
                y =self.waypoint[1,i]
                z =self.waypoint[2,i]
                thet0 , thet1, thet2 = robot.arm.ik_solver([x,y,z])
                command_msg.acceleration.data.extend([0,0,0 ,0,0,0, 0,0,0,0,0])
                command_msg.twist.data.extend([0,0,0 ,0,0,0, 0,0,0,0,0])
                command_msg.pose.data.extend([0,0,0 ,0,0,0, thet0, thet1, thet2, 2.1, 0])
                
        # Publish the command
        self.uvms_publisher_.publish(command_msg)



    def listener_callback(self, msg: DynamicJointState):
        for robot in self.robots:
            robot.update_state(msg)


    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    iktask = iKtask()

    try:
        rclpy.spin(iktask)
    except KeyboardInterrupt:
        iktask.get_logger().info('iktask node stopped by KeyboardInterrupt.')
    finally:
        iktask.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()