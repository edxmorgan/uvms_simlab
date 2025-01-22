import rclpy
from rclpy.node import Node
import numpy as np
from uvms_interfaces.msg import Command

class CoverageTask(Node):
    def __init__(self):
        super().__init__('coverage_task',
                          automatically_declare_parameters_from_overrides=True)

        # Get parameter values
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value

        self.total_no_efforts = self.no_robot * self.no_efforts

        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', 10)
        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)
        self.get_logger().info("CoverageTask node has been initialized with optimal control.")

    def timer_callback(self):
        command_msg = Command()
        command_msg.command_type = "optimal"
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


    def square_velocity_uv_ref(self, t: float,
                            T_side: float = 10.0,
                            speed: float = 0.1,
                            manput: bool = False,
                            ops: bool = False) -> np.ndarray:
        """
        Generate a velocity reference that traces a square in the XY-plane.

        This function returns a 6D velocity reference [u, v, w, p, q, r] 
        in the vehicle's body frame (or inertial frame, depending on convention). 
        The motion completes one square loop every 4*T_side seconds, 
        moving at the given `speed`.

        Args:
            t (float): Current time in seconds.
            T_side (float, optional): Time to traverse one side of the square [s]. 
                                    Defaults to 10.0.
            speed (float, optional): Linear speed along each side [m/s]. 
                                    Defaults to 0.1.
            manput (bool, optional): If True, returns a 10-element array 
                                    with the last 4 elements as zeros. 
                                    If False, returns a 6-element array.
                                    Defaults to False.

        Returns:
            np.ndarray:
                - shape (6,) if manput == False, i.e. [u, v, w, p, q, r]
                - shape (10,) if manput == True, i.e. [u, v, w, p, q, r, 0, 0, 0, 0]

        Example:
            >>> # Suppose we want the velocity at time t = 15s
            >>> v_ref = square_velocity_uv_ref(t=15.0, T_side=10.0, speed=0.1, manput=False)
            >>> print(v_ref)
            array([0. , 0.1, 0. , 0. , 0. , 0. ])
            # This corresponds to movement along the y-axis on the second leg.
        """
        # Total time for one full loop (4 sides)
        period = 4 * T_side
        
        # Repeat the motion every 'period' seconds
        t_mod = t % period

        # Determine which side (leg) of the square we are on
        if t_mod < T_side:
            # Leg 1: move along +x
            u, v = speed, 0.0
            dq0, dq1, dq2, dq3 = 0.1, -0.1, 0.1, -0.1
        elif t_mod < 2 * T_side:
            # Leg 2: move along +y
            u, v = 0.0, speed
            dq0, dq1, dq2, dq3 = -0.1, 0.1, -0.2, 0.1
        elif t_mod < 3 * T_side:
            # Leg 3: move along -x
            u, v = -speed, 0.0
            dq0, dq1, dq2, dq3 = 0.1, -0.1, 0.14, -0.1
        else:
            # Leg 4: move along -y
            u, v = 0.0, -speed
            dq0, dq1, dq2, dq3 = -0.1, 0.1, -0.1, 0.01

        # Zero out the other velocity components
        w, p, q, r = 0.0, 0.0, 0.0, 0.0
        
        if ops:
            # Return 6-element array
            return np.array([u, v, w, p, q, r])
        # Return the desired output format
        if manput:
            # Return 10-element array (extra zeros at the end)
            return np.array([u, v, w, p, q, r, dq0, dq1, dq2, dq3])
        else:
            # Return 6-element array
            return np.array([u, v, w, p, q, r])
    
    def destroy_node(self):
        self.listener.stop()
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
