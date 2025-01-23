import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command
from control_msgs.msg import DynamicJointState
from robot import Robot
from task import Task

class CoverageTask(Node):
    def __init__(self):
        super().__init__('coverage_task',
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

        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', 10)
        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)
        self.get_logger().info("CoverageTask node has been initialized with optimal control.")

    def timer_callback(self):
        command_msg = Command()
        # command_msg.command_type = "optimal"
        # command_msg.pose.data = []
        # command_msg.twist.data = self.square_velocity_uv_ref(t[i], T_side=10.0, speed=0.1, manput=True).tolist()
        # command_msg.acceleration.data = np.zeros((10,)).tolist()

        # # Create and publish the command message

        # xt0 = res_ref[0][:,i-1].reshape(10,1) # x(t-1)

        # J_UVMS_REF_np = J_UVMS_REF.full()
        # v_ned_ref = J_UVMS_REF_np@res_ref[1][:,i].reshape(10,1) # dx(t)
        
        # res_ref[0][:,i] = ref_intg(xt0, v_ned_ref, alpha.delta_t).full().flatten()


        # # Publish the command
        # self.publisher_.publish(command_msg)

    def listener_callback(self, msg: DynamicJointState):
        pass


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