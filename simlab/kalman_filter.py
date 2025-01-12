#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node

# Standard ROS 2 message imports
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
# MAVROS scaled pressure message for SCALED_PRESSURE2
# from mavros_msgs.msg import ScaledPressure
# QoS imports
from rclpy.qos import QoSProfile, ReliabilityPolicy

class BlueROVKF(Node):
    """
    A ROS 2 Node implementing the same Kalman Filter logic/parameters as your C++ code,
    but using standard messages (no custom .msg files).
    Reads:
      - IMU from /mavros/imu/data
      - DVL from /dvl/twist (TwistWithCovarianceStamped)
      - Scaled Pressure from /mavros/imu/scaled_pressure2
    Publishes:
      - nav_msgs/Odometry (/blue_rov/odom)
      - geometry_msgs/PoseStamped (/blue_rov/pose)
    """

    def __init__(self):
        super().__init__('blue_rov_kf')

        # ------------------------ STATE DEFINITION -------------------------
        # State vector (12x1):
        #  [ x, y, z, roll, pitch, yaw, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot ]^T
        self.state_ = np.zeros((12, 1), dtype=float)

        # State covariance (12x12)
        self.cov_ = np.zeros((12, 12), dtype=float)
        for i in range(12):
            if i < 6:
                # Position/orientation states get smaller initial covariance
                self.cov_[i, i] = 1.0
            else:
                # Velocity/orientation_rate states are more uncertain initially
                self.cov_[i, i] = 1000.0

        # ------------------------ F MATRIX -------------------------
        # (12x12) Identity + top-right 6x6 block for dt in positions
        self.F_ = np.eye(12, dtype=float)
        # We will update self.F_[i, i+6] = dt at each prediction step for i in [0..5]
        for i in range(6):
            self.F_[i, i+6] = 0.0

        # ------------------------ PROCESS NOISE (Q) -------------------------
        # Exactly matching your C++ code
        # We'll build Q_ as 12x12, mostly zeros, except the diagonal:
        self.Q_ = np.zeros((12, 12), dtype=float)
        for i in range(12):
            # x, y => Q=0
            if i < 2:
                self.Q_[i, i] = 0.0
            # z => Q=0.0001
            elif i == 2:
                self.Q_[i, i] = 0.0001
            # roll, pitch, yaw => Q=0.00001
            elif 3 <= i < 6:
                self.Q_[i, i] = 0.00001
            # x_dot, y_dot, z_dot => Q=0.00001
            elif 6 <= i < 9:
                self.Q_[i, i] = 0.00001
            # roll_dot, pitch_dot, yaw_dot => Q=0.00001
            elif 9 <= i < 12:
                self.Q_[i, i] = 0.00001

        # ------------------------ MEASUREMENT MATRICES ----------------------
        # 1) IMU => [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot], 6x12
        self.H_IMU_ = np.zeros((6, 12), dtype=float)
        # map roll, pitch, yaw => indices 3,4,5
        for i in range(3):
            self.H_IMU_[i, i+3] = 1.0
        # map roll_dot, pitch_dot, yaw_dot => indices 9,10,11
        for i in range(3, 6):
            self.H_IMU_[i, i+6] = 1.0  # (3->9, 4->10, 5->11)

        # IMU measurement noise (6x6)
        #   [0.15, 0.15, 0.15, 0.01, 0.01, 0.01] on diagonal
        self.R_IMU_ = np.zeros((6, 6), dtype=float)
        for i in range(6):
            if i < 3:
                self.R_IMU_[i, i] = 0.15
            else:
                self.R_IMU_[i, i] = 0.01

        # 2) DVL => [x_dot, y_dot, z_dot], 3x12
        self.H_DVL_ = np.zeros((3, 12), dtype=float)
        for i in range(3):
            self.H_DVL_[i, i+6] = 1.0

        # DVL measurement noise (3x3)
        #   0.4 on diagonal
        self.R_DVL_ = np.eye(3, dtype=float) * 0.4

        # 3) PRESSURE => z => 1x12
        #   same approach as your original "depth" measurement, R_depth_=1
        self.H_pressure_ = np.zeros((1, 12), dtype=float)
        self.H_pressure_[0, 2] = 1.0
        self.R_pressure_ = np.array([[1.0]], dtype=float)

        # ------------------------ MISCELLANEOUS ----------------------------
        self.is_initialized_ = False
        self.prev_time_ = None
        self.MAX_VEL = 5.0  # clamp for DVL if velocity is too large

        # If using a different density for water, set here (kg/m^3)
        # e.g. 997 for freshwater, ~1029 for saltwater
        self.water_type = 997.0  

        # Buffers for the last DVL / pressure messages
        self.dvl_msg_ = None
        self.scaled_pressure_msg_ = None

        # ------------------------ ROS 2 PUBLISHERS -------------------------
        # We'll publish the final filtered state as nav_msgs/Odometry
        self.odom_pub_ = self.create_publisher(Odometry, '/blue_rov/odom', 10)
        # And a PoseStamped for visualization
        self.pose_pub_ = self.create_publisher(PoseStamped, '/blue_rov/pose', 10)

        # ------------------------ ROS 2 SUBSCRIPTIONS ----------------------
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # Use that QoS profile when creating the subscription
        self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile
        )
        self.create_subscription(TwistWithCovarianceStamped, '/dvl/twist', self.dvl_callback, 10)
        # self.create_subscription(ScaledPressure, '/mavros/imu/scaled_pressure2', self.scaled_pressure_callback, 10)

        self.get_logger().info("BlueROVKF Node initialized with the same parameters as your C++ code.")

    # ----------------------------------------------------------------------
    #                          ROS 2 CALLBACKS
    # ----------------------------------------------------------------------
    def imu_callback(self, msg: Imu):
        """
        Main callback from IMU sensor_msgs/Imu on /mavros/imu/data.
        We'll:
        1) Initialize the filter if needed.
        2) Compute dt from last IMU time.
        3) Predict step.
        4) Check if DVL/pressure messages have arrived, do update steps for them.
        5) IMU update step.
        6) Publish final result.
        """
        if not self.is_initialized_:
            # ------------------- INITIALIZATION --------------------
            ax = -msg.linear_acceleration.x
            ay = -msg.linear_acceleration.y
            az = -msg.linear_acceleration.z

            # roll, pitch, yaw from accelerations
            roll = math.atan2(ay, az)
            pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
            yaw = math.atan2(az, math.sqrt(ax*ax + az*az))

            # set state
            self.state_[3, 0]  = roll
            self.state_[4, 0]  = pitch
            self.state_[5, 0]  = yaw
            # also take the IMU's angular velocity for roll_dot, pitch_dot, yaw_dot
            self.state_[9, 0]  = msg.angular_velocity.x
            self.state_[10, 0] = msg.angular_velocity.y
            self.state_[11, 0] = msg.angular_velocity.z

            self.prev_time_ = msg.header.stamp
            self.is_initialized_ = True
            return

        # ------------------- COMPUTE DT --------------------
        curr_time = msg.header.stamp
        curr_sec = float(curr_time.sec) + float(curr_time.nanosec) * 1e-9
        if self.prev_time_ is None:
            self.prev_time_ = msg.header.stamp
            return
        prev_sec = float(self.prev_time_.sec) + float(self.prev_time_.nanosec) * 1e-9
        dt = curr_sec - prev_sec
        self.prev_time_ = msg.header.stamp

        # clamp dt if invalid or large
        if dt <= 0.0 or dt > 1.0:
            dt = 0.005  # fallback, matches your C++ example (1/200 = 0.005)

        # ------------------- PREDICT STEP --------------------
        # Fill in the dt into F_
        for i in range(6):
            self.F_[i, i+6] = dt

        self.predict()

        # ------------------- OPTIONAL UPDATES ------------------
        # 1) If we have DVL data, update from DVL
        if self.dvl_msg_ is not None:
            self.update_dvl(self.dvl_msg_)
            self.dvl_msg_ = None

        # 2) If we have scaled_pressure data, treat it as depth
        # if self.scaled_pressure_msg_ is not None:
        #     self.update_pressure(self.scaled_pressure_msg_)
        #     self.scaled_pressure_msg_ = None

        # ------------------- IMU UPDATE --------------------
        self.update_imu(msg)

        # ------------------- PUBLISH RESULTS ----------------
        self.publish_filtered_result()

    def dvl_callback(self, msg: TwistWithCovarianceStamped):
        """
        Store DVL data for the next filter cycle.
        """
        self.dvl_msg_ = msg

    # def scaled_pressure_callback(self, msg: ScaledPressure):
    #     """
    #     Store scaled pressure message (SCALED_PRESSURE2) for the next cycle.
    #     """
    #     self.scaled_pressure_msg_ = msg

    # ----------------------------------------------------------------------
    #                         KALMAN FILTER STEPS
    # ----------------------------------------------------------------------
    def predict(self):
        """
        Predict step: X = F * X, P = F P F^T + Q
        """
        self.state_ = self.F_ @ self.state_
        self.cov_   = self.F_ @ self.cov_ @ self.F_.T + self.Q_

    def update_imu(self, imu_msg: Imu):
        """
        Kalman update for IMU:
         - roll, pitch, yaw from linear_acceleration
         - roll_dot, pitch_dot, yaw_dot from angular_velocity
        Matches the same logic / noise parameters in your C++ code.
        """
        # Extract orientation from IMU acceleration
        ax = -imu_msg.linear_acceleration.x
        ay = -imu_msg.linear_acceleration.y
        az = -imu_msg.linear_acceleration.z

        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        yaw = math.atan2(az, math.sqrt(ax**2 + az**2))

        # Build measurement vector z
        z_meas = np.array([
            [roll],
            [pitch],
            [yaw],
            [imu_msg.angular_velocity.x],
            [imu_msg.angular_velocity.y],
            [imu_msg.angular_velocity.z]
        ], dtype=float)

        # Predicted measurement
        z_pred = self.H_IMU_ @ self.state_

        # Residual
        y = z_meas - z_pred

        # S, K
        S = self.H_IMU_ @ self.cov_ @ self.H_IMU_.T + self.R_IMU_
        K = self.cov_ @ self.H_IMU_.T @ np.linalg.inv(S)

        # Update state
        self.state_ += K @ y
        I = np.eye(12)
        self.cov_ = (I - K @ self.H_IMU_) @ self.cov_

    def update_dvl(self, dvl_msg: TwistWithCovarianceStamped):
        """
        Kalman update for DVL velocities => x_dot, y_dot, z_dot.
        Using R_DVL_ = diag(0.4, 0.4, 0.4).
        """
        x_vel = dvl_msg.twist.twist.linear.x
        y_vel = dvl_msg.twist.twist.linear.y
        z_vel = dvl_msg.twist.twist.linear.z

        # Simple threshold filter (like in your C++ code)
        if abs(x_vel) > self.MAX_VEL:
            x_vel = 0.0
        if abs(y_vel) > self.MAX_VEL:
            y_vel = 0.0
        if abs(z_vel) > self.MAX_VEL:
            z_vel = 0.0

        z_meas = np.array([
            [x_vel],
            [y_vel],
            [z_vel]
        ], dtype=float)

        z_pred = self.H_DVL_ @ self.state_
        y = z_meas - z_pred
        S = self.H_DVL_ @ self.cov_ @ self.H_DVL_.T + self.R_DVL_
        K = self.cov_ @ self.H_DVL_.T @ np.linalg.inv(S)

        self.state_ += K @ y
        I = np.eye(12)
        self.cov_ = (I - K @ self.H_DVL_) @ self.cov_

    # def update_pressure(self, scaled_msg: ScaledPressure):
    #     """
    #     Kalman update for depth from scaled pressure => z.
    #     Using R_depth_ = 1, from your C++ code's R_depth_.
    #     """
    #     # Convert pressure (hPa) into depth (m).
    #     # scaled_msg.press_abs is in hPa => multiply by 100 => Pa.
    #     # Then remove ~101300 Pa atmospheric => hydrostatic portion.
    #     # Then divide by rho*g to get depth in m.
    #     pressure_abs = scaled_msg.press_abs
    #     depth = (100.0 * pressure_abs - 101300.0) / (self.water_type * 9.80665)

    #     z_meas = np.array([[depth]], dtype=float)
    #     z_pred = self.H_pressure_ @ self.state_

    #     y = z_meas - z_pred
    #     S = self.H_pressure_ @ self.cov_ @ self.H_pressure_.T + self.R_pressure_
    #     K = self.cov_ @ self.H_pressure_.T @ np.linalg.inv(S)

    #     self.state_ += K @ y
    #     I = np.eye(12)
    #     self.cov_ = (I - K @ self.H_pressure_) @ self.cov_

    # ----------------------------------------------------------------------
    #                      PUBLISH FILTERED RESULT
    # ----------------------------------------------------------------------
    def publish_filtered_result(self):
        """
        Publish the final state as nav_msgs/Odometry and geometry_msgs/PoseStamped.
        """
        now = self.get_clock().now().to_msg()

        # 1) Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        # Fill position
        odom_msg.pose.pose.position.x = float(self.state_[0])
        odom_msg.pose.pose.position.y = float(self.state_[1])
        # If 'z' is depth downward, store negative for typical NED->ENU conversion
        odom_msg.pose.pose.position.z = -float(self.state_[2])

        # Orientation (roll, pitch, yaw -> quaternion)
        roll  = float(self.state_[3])
        pitch = float(self.state_[4])
        yaw   = float(self.state_[5])
        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)

        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        # Covariance: we have a 12x12 in the filter, but Odometry uses 6x6 for pose & twist.
        # For demonstration, just fill the diagonals with some small values:
        # (In a real system, you'd map blocks of your covariance properly.)
        for i in range(6):
            odom_msg.pose.covariance[i*6 + i]  = 0.1
            odom_msg.twist.covariance[i*6 + i] = 0.2

        # Linear velocity
        odom_msg.twist.twist.linear.x = float(self.state_[6])
        odom_msg.twist.twist.linear.y = float(self.state_[7])
        odom_msg.twist.twist.linear.z = float(self.state_[8])

        # Angular velocity (roll_dot, pitch_dot, yaw_dot)
        odom_msg.twist.twist.angular.x = float(self.state_[9])
        odom_msg.twist.twist.angular.y = float(self.state_[10])
        odom_msg.twist.twist.angular.z = float(self.state_[11])

        self.odom_pub_.publish(odom_msg)

        # 2) PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = odom_msg.pose.pose.position.x
        pose_msg.pose.position.y = odom_msg.pose.pose.position.y
        pose_msg.pose.position.z = odom_msg.pose.pose.position.z

        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.pose_pub_.publish(pose_msg)

    @staticmethod
    def rpy_to_quat(roll, pitch, yaw):
        """
        Convert roll, pitch, yaw (radians) to quaternion (x, y, z, w).
        Matches the approach in your C++ code.
        """
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) \
             - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) \
             + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) \
             - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) \
             + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = BlueROVKF()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
