#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from geometry_msgs.msg import (
    TwistWithCovarianceStamped,
    PoseStamped,
    AccelStamped,
    PointStamped,  # If you implement position measurements
)
from nav_msgs.msg import Odometry

# For time conversions
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy


class BlueROVKF(Node):
    """
    A ROS 2 Node implementing a Kalman Filter for BlueROV with:
      - 150 Hz timer-based predict/update/publish
      - Adjusted covariance matrices to reduce drift
      - Publishing in both ENU ('map') and NED frames
    """

    def __init__(self):
        super().__init__('blue_rov_kf')

        # ------------------------ STATE DEFINITION -------------------------
        # state_ = [x, y, z, roll, pitch, yaw, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot]
        self.state_ = np.zeros((12, 1), dtype=float)

        # ------------------------ INITIAL COVARIANCE -----------------------
        self.cov_ = np.zeros((12, 12), dtype=float)
        for i in range(12):
            if i < 6:
                self.cov_[i, i] = 1.0  # Positions and orientations
            else:
                self.cov_[i, i] = 10.0  # Velocities and angular velocities (reduced from 1000.0)

        # ------------------------ STATE TRANSITION MATRIX ------------------
        self.F_ = np.eye(12, dtype=float)
        for i in range(6):
            self.F_[i, i + 6] = 0.0  # To be set with dt at runtime

        # ------------------------ PROCESS NOISE (Q) ------------------------
        self.Q_ = np.zeros((12, 12), dtype=float)
        for i in range(12):
            if i < 2:                   # x, y positions
                self.Q_[i, i] = 0.01    # Small process noise
            elif i == 2:                # z position
                self.Q_[i, i] = 0.0001
            elif 3 <= i < 6:            # roll, pitch, yaw
                self.Q_[i, i] = 0.0001
            elif 6 <= i < 12:           # velocities and angular velocities
                self.Q_[i, i] = 0.001    # Adjusted from 0.00001

        # ----------------------- MEASUREMENT MATRICES ----------------------
        # IMU => [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot]
        self.H_IMU_ = np.zeros((6, 12), dtype=float)
        for i in range(3):
            self.H_IMU_[i, i + 3] = 1.0
        for i in range(3, 6):
            self.H_IMU_[i, i + 6] = 1.0

        self.R_IMU_ = np.zeros((6, 6), dtype=float)
        for i in range(6):
            if i < 3:
                self.R_IMU_[i, i] = 0.15
            else:
                self.R_IMU_[i, i] = 0.01

        # DVL => x_dot, y_dot, z_dot
        self.H_DVL_ = np.zeros((3, 12), dtype=float)
        for i in range(3):
            self.H_DVL_[i, i + 6] = 1.0
        self.R_DVL_ = np.eye(3, dtype=float) * 0.1  # Reduced from 0.4

        # Pressure => z
        self.H_pressure_ = np.zeros((1, 12), dtype=float)
        self.H_pressure_[0, 2] = 1.0
        self.R_pressure_ = np.array([[1.0]], dtype=float)

        # Optional: Position Measurements
        # self.H_position_ = np.zeros((3, 12), dtype=float)
        # self.H_position_[0, 0] = 1.0  # x
        # self.H_position_[1, 1] = 1.0  # y
        # self.H_position_[2, 2] = 1.0  # z
        # self.R_position_ = np.eye(3) * 0.5  # Adjust based on sensor

        self.water_type = 997.0  # fresh water

        self.is_initialized_ = False
        self.prev_time_ = None  # Will store an rclpy.time.Time object

        self.MAX_VEL = 5.0  # Clamp for DVL

        # For approximate acceleration output
        self.prev_vx_ = 0.0
        self.prev_vy_ = 0.0
        self.prev_vz_ = 0.0

        # ------------------------ ROS 2 PUBLISHERS -------------------------
        self.odom_pub_ = self.create_publisher(Odometry, '/blue_rov/odom', 10)
        self.pose_pub_ = self.create_publisher(PoseStamped, '/blue_rov/pose', 10)
        self.accel_pub_ = self.create_publisher(AccelStamped, '/blue_rov/accel', 10)


        # ------------------------ ROS 2 SUBSCRIPTIONS ----------------------
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, qos)
        self.create_subscription(TwistWithCovarianceStamped, '/dvl/twist', self.dvl_callback, qos)
        # self.create_subscription(ScaledPressure, '/mavros/imu/scaled_pressure2', self.scaled_pressure_callback, qos)

        # Optional: Position Subscription
        # self.create_subscription(PointStamped, '/position/data', self.position_callback, qos)

        # Store the latest incoming sensor messages
        self.imu_msg_ = None
        self.dvl_msg_ = None
        self.scaled_pressure_msg_ = None
        # self.position_msg_ = None

        # ------------------------ TIMER (150 Hz) -------------------------
        self.timer_period_ = 1.0 / 150.0  # ~0.00667 seconds
        self.timer_ = self.create_timer(self.timer_period_, self.timer_callback)

        self.get_logger().info("BlueROVKF Node started: 150 Hz timer, adjusted covariances to reduce drift.")

    # ----------------------------------------------------------------------
    #                          ROS 2 CALLBACKS
    # ----------------------------------------------------------------------
    def imu_callback(self, msg: Imu):
        """
        Store the IMU message for use in the timer callback.
        """
        self.imu_msg_ = msg

    def dvl_callback(self, msg: TwistWithCovarianceStamped):
        """
        Store the DVL message for use in the timer callback.
        """
        self.dvl_msg_ = msg

    # def scaled_pressure_callback(self, msg: ScaledPressure):
    #     self.scaled_pressure_msg_ = msg

    # Optional: Position Callback
    # def position_callback(self, msg: PointStamped):
    #     self.position_msg_ = msg

    # ----------------------------------------------------------------------
    #                   FIXED-RATE TIMER CALLBACK (150 Hz)
    # ----------------------------------------------------------------------
    def timer_callback(self):
        """
        Runs at 150 Hz:
          1) Initialize if needed,
          2) Predict,
          3) Update from available sensors,
          4) Publish filter results.
        """

        # Current node time as an rclpy.time.Time
        now = self.get_clock().now()

        # 1) If not initialized, attempt to initialize with IMU
        if not self.is_initialized_:
            if self.imu_msg_ is not None:
                self.initialize_filter_from_imu_quaternion(self.imu_msg_)
            else:
                return  # Still waiting for IMU to initialize

        # 2) Compute dt from the previous timer iteration
        if self.prev_time_ is None:
            # First iteration after initialization
            self.prev_time_ = now
            return

        dt = (now - self.prev_time_).nanoseconds * 1.0e-9
        self.prev_time_ = now

        # Clamp or handle unusual dt
        if dt <= 0.0 or dt > 1.0:
            dt = 0.005  # Fallback to 5 ms

        # Update the F_ matrix with current dt
        for i in range(6):
            self.F_[i, i + 6] = dt

        # --------------------- PREDICT STEP ---------------------
        self.predict()

        # --------------------- UPDATE STEPS ---------------------
        # Update from DVL
        if self.dvl_msg_ is not None:
            self.update_dvl(self.dvl_msg_)
            self.dvl_msg_ = None

        # Update from Pressure (if implemented)
        # if self.scaled_pressure_msg_ is not None:
        #     self.update_pressure(self.scaled_pressure_msg_)
        #     self.scaled_pressure_msg_ = None

        # Update from IMU
        if self.imu_msg_ is not None:
            self.update_imu(self.imu_msg_)
            # Optionally, clear if IMU messages are only relevant once per update
            # self.imu_msg_ = None

        # Optional: Update from Position
        # if self.position_msg_ is not None:
        #     self.update_position(self.position_msg_)
        #     self.position_msg_ = None

        # --------------------- PUBLISH RESULTS ---------------------
        self.publish_filtered_result(dt)

    # ----------------------------------------------------------------------
    #                           KALMAN FILTER
    # ----------------------------------------------------------------------
    def initialize_filter_from_imu_quaternion(self, imu_msg: Imu):
        """
        Convert IMU quaternion to roll, pitch, yaw.
        Set angular velocities.
        Mark filter as initialized.
        """
        roll, pitch, yaw = self.quat_to_rpy(
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        )
        self.state_[3, 0] = roll
        self.state_[4, 0] = pitch
        self.state_[5, 0] = yaw

        # Initialize velocities if available
        self.state_[6, 0] = 0.0  # x_dot (assuming stationary initially)
        self.state_[7, 0] = 0.0  # y_dot
        self.state_[8, 0] = 0.0  # z_dot

        # Angular velocities from IMU
        self.state_[9, 0] = imu_msg.angular_velocity.x
        self.state_[10, 0] = imu_msg.angular_velocity.y
        self.state_[11, 0] = imu_msg.angular_velocity.z

        # Store current node time as prev_time_ (for dt computation)
        self.prev_time_ = self.get_clock().now()
        self.is_initialized_ = True

        self.get_logger().info("Kalman Filter initialized from IMU quaternion.")

    def predict(self):
        """
        Predict the next state and covariance.
        """
        self.state_ = self.F_ @ self.state_
        self.cov_ = self.F_ @ self.cov_ @ self.F_.T + self.Q_

    def update_imu(self, imu_msg: Imu):
        """
        Update the filter with IMU data.
        Measurement: [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot].
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
        y = z_meas - z_pred  # Innovation

        S = self.H_IMU_ @ self.cov_ @ self.H_IMU_.T + self.R_IMU_  # Innovation covariance
        K = self.cov_ @ self.H_IMU_.T @ np.linalg.inv(S)  # Kalman Gain

        self.state_ += K @ y
        I = np.eye(12)
        self.cov_ = (I - K @ self.H_IMU_) @ self.cov_

        self.get_logger().debug("IMU update performed.")

    def update_dvl(self, dvl_msg: TwistWithCovarianceStamped):
        """
        Update the filter with DVL data.
        Measurement: [x_dot, y_dot, z_dot].
        """
        x_vel = dvl_msg.twist.twist.linear.x
        y_vel = dvl_msg.twist.twist.linear.y
        z_vel = dvl_msg.twist.twist.linear.z

        # Clamp velocities to prevent extreme values
        x_vel = np.clip(x_vel, -self.MAX_VEL, self.MAX_VEL)
        y_vel = np.clip(y_vel, -self.MAX_VEL, self.MAX_VEL)
        z_vel = np.clip(z_vel, -self.MAX_VEL, self.MAX_VEL)

        z_meas = np.array([
            [x_vel],
            [y_vel],
            [z_vel]
        ], dtype=float)

        z_pred = self.H_DVL_ @ self.state_
        y = z_meas - z_pred  # Innovation

        S = self.H_DVL_ @ self.cov_ @ self.H_DVL_.T + self.R_DVL_
        K = self.cov_ @ self.H_DVL_.T @ np.linalg.inv(S)

        self.state_ += K @ y
        I = np.eye(12)
        self.cov_ = (I - K @ self.H_DVL_) @ self.cov_

        self.get_logger().debug("DVL update performed.")

    # def update_pressure(self, scaled_msg: ScaledPressure):
    #     """
    #     Update the filter with pressure data.
    #     Measurement: [z] (depth).
    #     """
    #     pressure_abs = scaled_msg.press_abs
    #     depth = (100.0 * pressure_abs - 101300.0) / (self.water_type * 9.80665)

    #     z_meas = np.array([[depth]], dtype=float)
    #     z_pred = self.H_pressure_ @ self.state_
    #     y = z_meas - z_pred  # Innovation

    #     S = self.H_pressure_ @ self.cov_ @ self.H_pressure_.T + self.R_pressure_
    #     K = self.cov_ @ self.H_pressure_.T @ np.linalg.inv(S)

    #     self.state_ += K @ y
    #     I = np.eye(12)
    #     self.cov_ = (I - K @ self.H_pressure_) @ self.cov_

    #     self.get_logger().debug("Pressure update performed.")

    # Optional: Update from Position Measurements
    # def update_position(self, pos_msg: PointStamped):
    #     """
    #     Update the filter with position data.
    #     Measurement: [x, y, z].
    #     """
    #     z_meas = np.array([
    #         [pos_msg.point.x],
    #         [pos_msg.point.y],
    #         [pos_msg.point.z]
    #     ], dtype=float)

    #     z_pred = self.H_position_ @ self.state_
    #     y = z_meas - z_pred  # Innovation

    #     S = self.H_position_ @ self.cov_ @ self.H_position_.T + self.R_position_
    #     K = self.cov_ @ self.H_position_.T @ np.linalg.inv(S)

    #     self.state_ += K @ y
    #     I = np.eye(12)
    #     self.cov_ = (I - K @ self.H_position_) @ self.cov_

    #     self.get_logger().debug("Position update performed.")

    # ----------------------------------------------------------------------
    #                     PUBLISH FILTERED RESULTS
    # ----------------------------------------------------------------------
    def publish_filtered_result(self, dt):
        """
        Publish the filtered state as Odometry, PoseStamped, AccelStamped, and NED PoseStamped/Odometry.
        """
        now_msg = self.get_clock().now().to_msg()

        # 1) Odometry (ENU - 'map' frame)
        odom_msg = Odometry()
        odom_msg.header.stamp = now_msg
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        # Position (z is depth, invert sign if necessary)
        odom_msg.pose.pose.position.x = float(self.state_[0])
        odom_msg.pose.pose.position.y = float(self.state_[1])
        odom_msg.pose.pose.position.z = -float(self.state_[2])  # Invert if using NED

        # Orientation
        roll = float(self.state_[3])
        pitch = float(self.state_[4])
        yaw = float(self.state_[5])
        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        # Covariances (placeholder values)
        for i in range(6):
            odom_msg.pose.covariance[i * 6 + i] = 0.1
            odom_msg.twist.covariance[i * 6 + i] = 0.2

        # Linear and angular velocity
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

        # 2) PoseStamped (ENU - 'map' frame)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now_msg
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = odom_msg.pose.pose.position.x
        pose_msg.pose.position.y = odom_msg.pose.pose.position.y
        pose_msg.pose.position.z = odom_msg.pose.pose.position.z
        pose_msg.pose.orientation = odom_msg.pose.pose.orientation
        self.pose_pub_.publish(pose_msg)

        # 3) AccelStamped (approximate from velocity difference)
        accel_msg = AccelStamped()
        accel_msg.header.stamp = now_msg
        accel_msg.header.frame_id = "map"

        if dt > 1.0e-6:
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

        # Update previous velocities for acceleration calculation
        self.prev_vx_ = vx
        self.prev_vy_ = vy
        self.prev_vz_ = vz

        self.get_logger().debug("Published filtered results.")

    # ----------------------------------------------------------------------
    #                 QUATERNION <--> ROLL/PITCH/YAW
    # ----------------------------------------------------------------------
    @staticmethod
    def quat_to_rpy(qx, qy, qz, qw):
        """
        Convert quaternion to roll, pitch, yaw in radians
        (avoiding angle wrap issues).
        """
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def rpy_to_quat(roll, pitch, yaw):
        """
        Convert roll, pitch, yaw to quaternion (x, y, z, w).
        """
        qx = (math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2)
              - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2))
        qy = (math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
              + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2))
        qz = (math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
              - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2))
        qw = (math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2)
              + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2))
        return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = BlueROVKF()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
