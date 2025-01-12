#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node

# Standard ROS 2 messages
from sensor_msgs.msg import Imu
from geometry_msgs.msg import (
    TwistWithCovarianceStamped,
    PoseStamped,
    AccelStamped,
)
from nav_msgs.msg import Odometry
# from mavros_msgs.msg import ScaledPressure

# If you have tf_transformations or a similar library installed:
#   from tf_transformations import euler_from_quaternion
# Otherwise, we'll manually convert quaternions to RPY below.

class BlueROVKF(Node):
    """
    A ROS 2 Node implementing the same Kalman Filter logic (Q/R, state layout),
    but using the IMU quaternion for roll, pitch, yaw. This prevents large
    wrap-around angles (e.g., -3.115) when robot is flat. Instead, it should
    match the IMU reading (-0.027).
    """

    def __init__(self):
        super().__init__('blue_rov_kf')

        # ------------------------ STATE DEFINITION -------------------------
        #  [x, y, z, roll, pitch, yaw, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot]
        self.state_ = np.zeros((12, 1), dtype=float)

        self.cov_ = np.zeros((12, 12), dtype=float)
        for i in range(12):
            if i < 6:
                self.cov_[i, i] = 1.0
            else:
                self.cov_[i, i] = 1000.0

        # State transition
        self.F_ = np.eye(12, dtype=float)
        for i in range(6):
            self.F_[i, i+6] = 0.0  # updated at runtime with dt

        # ------------------------ PROCESS NOISE (Q) ------------------------
        # Matches your original code exactly
        self.Q_ = np.zeros((12, 12), dtype=float)
        for i in range(12):
            if i < 2:                  # x, y
                self.Q_[i, i] = 0.0
            elif i == 2:               # z
                self.Q_[i, i] = 0.0001
            elif 3 <= i < 6:           # roll, pitch, yaw
                self.Q_[i, i] = 0.00001
            elif 6 <= i < 12:          # x_dot..yaw_dot
                self.Q_[i, i] = 0.00001

        # ----------------------- MEASUREMENT MATRICES ----------------------
        # IMU => [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot]
        self.H_IMU_ = np.zeros((6, 12), dtype=float)
        for i in range(3):
            self.H_IMU_[i, i+3] = 1.0
        for i in range(3, 6):
            self.H_IMU_[i, i+6] = 1.0

        self.R_IMU_ = np.zeros((6, 6), dtype=float)
        for i in range(6):
            if i < 3:
                self.R_IMU_[i, i] = 0.15
            else:
                self.R_IMU_[i, i] = 0.01

        # DVL => x_dot, y_dot, z_dot
        self.H_DVL_ = np.zeros((3, 12), dtype=float)
        for i in range(3):
            self.H_DVL_[i, i+6] = 1.0

        self.R_DVL_ = np.eye(3, dtype=float) * 0.4

        # Pressure => z
        self.H_pressure_ = np.zeros((1, 12), dtype=float)
        self.H_pressure_[0, 2] = 1.0
        self.R_pressure_ = np.array([[1.0]], dtype=float)

        # Water density for depth conversion
        self.water_type = 997.0  # fresh water

        self.is_initialized_ = False
        self.prev_time_ = None

        self.MAX_VEL = 5.0  # clamp for DVL
        # For approximate acceleration output
        self.prev_vx_ = 0.0
        self.prev_vy_ = 0.0
        self.prev_vz_ = 0.0

        # ------------------------ ROS 2 PUBLISHERS -------------------------
        self.odom_pub_ = self.create_publisher(Odometry, '/blue_rov/odom', 10)
        self.pose_pub_ = self.create_publisher(PoseStamped, '/blue_rov/pose', 10)
        self.accel_pub_ = self.create_publisher(AccelStamped, '/blue_rov/accel', 10)

        # ------------------------ ROS 2 SUBSCRIPTIONS ----------------------
        # Use Best Effort QoS to match a typical MAVROS setting if necessary:
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos)
        self.create_subscription(TwistWithCovarianceStamped, '/dvl/twist', self.dvl_callback, qos)
        # self.create_subscription(ScaledPressure, '/mavros/imu/scaled_pressure2', self.scaled_pressure_callback, qos)

        self.dvl_msg_ = None
        self.scaled_pressure_msg_ = None

        self.get_logger().info("BlueROVKF Node using IMU quaternion orientation to fix roll mismatch.")

    # ----------------------------------------------------------------------
    #                          ROS 2 CALLBACKS
    # ----------------------------------------------------------------------
    def imu_callback(self, msg: Imu):
        """
        Main callback for IMU data. We'll read orientation from the quaternion
        instead of inferring from linear_acceleration. This prevents the large angle wrap.
        """
        if not self.is_initialized_:
            self.initialize_filter_from_imu_quaternion(msg)
            return

        # compute dt
        curr_time = msg.header.stamp
        curr_sec = float(curr_time.sec) + float(curr_time.nanosec)*1e-9
        if self.prev_time_ is None:
            self.prev_time_ = msg.header.stamp
            return
        prev_sec = float(self.prev_time_.sec) + float(self.prev_time_.nanosec)*1e-9
        dt = curr_sec - prev_sec
        self.prev_time_ = msg.header.stamp

        if dt <= 0.0 or dt > 1.0:
            dt = 0.005  # fallback

        for i in range(6):
            self.F_[i, i+6] = dt

        # Predict
        self.predict()

        # If we have DVL data
        if self.dvl_msg_ is not None:
            self.update_dvl(self.dvl_msg_)
            self.dvl_msg_ = None

        # # If we have pressure data
        # if self.scaled_pressure_msg_ is not None:
        #     self.update_pressure(self.scaled_pressure_msg_)
        #     self.scaled_pressure_msg_ = None

        # IMU update using orientation from quaternion
        self.update_imu(msg)

        # Publish final
        self.publish_filtered_result(dt)

    def dvl_callback(self, msg: TwistWithCovarianceStamped):
        self.dvl_msg_ = msg

    # def scaled_pressure_callback(self, msg: ScaledPressure):
    #     self.scaled_pressure_msg_ = msg

    # ----------------------------------------------------------------------
    #                           KALMAN FILTER
    # ----------------------------------------------------------------------
    def initialize_filter_from_imu_quaternion(self, imu_msg: Imu):
        """
        Use the IMU's quaternion orientation to find initial roll, pitch, yaw.
        We'll also set roll_dot, pitch_dot, yaw_dot from angular_velocity.
        """
        roll, pitch, yaw = self.quat_to_rpy(
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        )
        self.state_[3, 0]  = roll
        self.state_[4, 0]  = pitch
        self.state_[5, 0]  = yaw

        # Also set the angular velocity
        self.state_[9, 0]  = imu_msg.angular_velocity.x
        self.state_[10, 0] = imu_msg.angular_velocity.y
        self.state_[11, 0] = imu_msg.angular_velocity.z

        self.prev_time_ = imu_msg.header.stamp
        self.is_initialized_ = True

    def predict(self):
        self.state_ = self.F_ @ self.state_
        self.cov_   = self.F_ @ self.cov_ @ self.F_.T + self.Q_

    def update_imu(self, imu_msg: Imu):
        """
        Convert quaternion -> roll, pitch, yaw, then gather angular velocity.
        The measurement vector => [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot].
        """
        roll, pitch, yaw = self.quat_to_rpy(
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        )

        z_meas = np.array([
            [roll],
            [pitch],
            [yaw],
            [imu_msg.angular_velocity.x],
            [imu_msg.angular_velocity.y],
            [imu_msg.angular_velocity.z]
        ], dtype=float)

        z_pred = self.H_IMU_ @ self.state_
        y = z_meas - z_pred
        S = self.H_IMU_ @ self.cov_ @ self.H_IMU_.T + self.R_IMU_
        K = self.cov_ @ self.H_IMU_.T @ np.linalg.inv(S)

        self.state_ += K @ y
        I = np.eye(12)
        self.cov_ = (I - K @ self.H_IMU_) @ self.cov_

    def update_dvl(self, dvl_msg: TwistWithCovarianceStamped):
        x_vel = dvl_msg.twist.twist.linear.x
        y_vel = dvl_msg.twist.twist.linear.y
        z_vel = dvl_msg.twist.twist.linear.z

        if abs(x_vel) > self.MAX_VEL:
            x_vel = 0.0
        if abs(y_vel) > self.MAX_VEL:
            y_vel = 0.0
        if abs(z_vel) > self.MAX_VEL:
            z_vel = 0.0

        z_meas = np.array([[x_vel], [y_vel], [z_vel]], dtype=float)
        z_pred = self.H_DVL_ @ self.state_
        y = z_meas - z_pred
        S = self.H_DVL_ @ self.cov_ @ self.H_DVL_.T + self.R_DVL_
        K = self.cov_ @ self.H_DVL_.T @ np.linalg.inv(S)

        self.state_ += K @ y
        I = np.eye(12)
        self.cov_ = (I - K @ self.H_DVL_) @ self.cov_

    # def update_pressure(self, scaled_msg: ScaledPressure):
    #     """
    #     Convert scaled pressure to depth => z
    #     using the same formula as your C++ code.
    #     """
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
    #                     PUBLISH FILTERED RESULTS
    # ----------------------------------------------------------------------
    def publish_filtered_result(self, dt):
        now = self.get_clock().now().to_msg()

        # 1) Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = float(self.state_[0])
        odom_msg.pose.pose.position.y = float(self.state_[1])
        odom_msg.pose.pose.position.z = -float(self.state_[2])  # invert if z is depth

        roll  = float(self.state_[3])
        pitch = float(self.state_[4])
        yaw   = float(self.state_[5])
        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        for i in range(6):
            odom_msg.pose.covariance[i*6 + i]  = 0.1
            odom_msg.twist.covariance[i*6 + i] = 0.2

        vx = float(self.state_[6])
        vy = float(self.state_[7])
        vz = float(self.state_[8])
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.linear.z = vz

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
        pose_msg.pose.orientation = odom_msg.pose.pose.orientation
        self.pose_pub_.publish(pose_msg)

        # 3) AccelStamped (Optional: approximate from velocity difference)
        accel_msg = AccelStamped()
        accel_msg.header.stamp = now
        accel_msg.header.frame_id = "map"

        if dt > 1e-6:
            ax = (vx - self.prev_vx_) / dt
            ay = (vy - self.prev_vy_) / dt
            az = (vz - self.prev_vz_) / dt
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        accel_msg.accel.linear.x = ax
        accel_msg.accel.linear.y = ay
        accel_msg.accel.linear.z = az
        accel_msg.accel.angular.x = 0.0
        accel_msg.accel.angular.y = 0.0
        accel_msg.accel.angular.z = 0.0

        self.accel_pub_.publish(accel_msg)

        self.prev_vx_ = vx
        self.prev_vy_ = vy
        self.prev_vz_ = vz

    # ----------------------------------------------------------------------
    #                 QUATERNION <--> ROLL/PITCH/YAW
    # ----------------------------------------------------------------------
    @staticmethod
    def quat_to_rpy(qx, qy, qz, qw):
        """
        Convert quaternion to roll, pitch, yaw in radians.
        This prevents angle wrap issues that can occur with atan2 on linear accel.
        """
        # Method 1: If tf_transformations is installed:
        #   euler = euler_from_quaternion([qx, qy, qz, qw])
        #   return euler[0], euler[1], euler[2]

        # Method 2: Manual math:
        # (Ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles )

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)  # clamp
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def rpy_to_quat(roll, pitch, yaw):
        """
        Convert roll, pitch, yaw to quaternion (x, y, z, w).
        Matches the approach in your original code.
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
