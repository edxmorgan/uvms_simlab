import numpy as np
import ament_index_python
import os
import casadi as ca
from blue_rov import Params as blue

class LowLevelController:
    def __init__(self, arm_dof: int = 4):
        package_share_directory = ament_index_python.get_package_share_directory('simlab')

        # Vehicle PID
        uv_pid_controller_path = os.path.join(package_share_directory, 'vehicle/uv_reg_pid_controller.casadi')
        self.uv_pid_controller = ca.Function.load(uv_pid_controller_path)
        self.vehicle_pid_i_buffer = np.zeros(6, dtype=float)  # 6 dof vehicle integral buffer

        # Integral buffer hardening, clamp and leak
        self.vehicle_i_limit = np.array([3, 3, 3, 3, 3, 3], dtype=float)  # per axis clamp
        self.vehicle_i_leak_per_s = 0.0  # leak coefficient in 1 per second, set 0.0 to disable

        # Arm PID
        arm_pid_controller_path = os.path.join(package_share_directory, 'manipulator/arm_pid.casadi')
        self.arm_pid_controller = ca.Function.load(arm_pid_controller_path)
        self.arm_dof = int(arm_dof)
        self.arm_pid_i_buffer = np.zeros(self.arm_dof, dtype=float)  # arm integral buffer
        self.g_ff = [0,0,0,0]  # feedforward gravity compensation

        self.vehicle_model_params = [3.72028553e+01, 2.21828075e+01, 6.61734807e+01, 3.38909801e+00,
                                  6.41362046e-01, 6.41362034e-01, 3.38909800e+00, 1.39646394e+00,
                                  4.98032205e-01, 2.53118738e+00, 1.05000000e+02, 9.78296453e+01,
                                  8.27479545e-01, 1.36822559e-01, 4.25841171e+00, -7.36416666e+01,
                                  -3.36082112e+01, -8.94055107e+01, -2.98736214e+00, -1.57921531e+00,
                                  -3.39766499e+00, -1.47912104e-04, -5.16373030e-04, -9.85522538e+01,
                                  -3.05907788e-02, -1.27877517e-01, -1.63514832e+00]


        self.kp = np.array([15.0, 15.0, 20.0, 1.5, 1.5, 3.5])
        self.ki = np.array([1.0, 1.0, 1.0, 1e-1, 1e-1, 1e-1])
        self.kd = np.array([5.0, 5.0, 5.0, 5e-1, 5e-1, 1])

    def vehicle_controller(self, state: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute vehicle body wrench with a CasADi PID function.

        state  shape (12,) = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
        target shape (6,)  = [x, y, z, roll, pitch, yaw]
        """
        state  = np.asarray(state, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)

        if state.size != 12:
            raise ValueError(f"state must have 12 elements, got {state.size}")
        if target.size != 6:
            raise ValueError(f"target must have 6 elements, got {target.size}")

        buf = np.zeros(6, dtype=float)  # Disable integral action for now

        pid_control, i_buf_next = self.uv_pid_controller(
            self.vehicle_model_params,
            self.kp,
            self.ki,
            self.kd,
            ca.DM(buf),
            ca.DM(state),
            ca.DM(target),
            float(dt),
        )

        # Update and clamp the returned integral buffer
        self.vehicle_pid_i_buffer = np.clip(
            np.asarray(i_buf_next).reshape(-1)[:6],
            -self.vehicle_i_limit,
            self.vehicle_i_limit,
        )

        return np.asarray(pid_control).reshape(-1)

    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        dt: float,
        u_max: np.ndarray,
        u_min: np.ndarray,
    ) -> np.ndarray:
        """
        Compute arm joint torques with the simple PID CasADi function you built.

        All vectors are length arm_dof.
        Returns saturated torque command, and updates the internal integral buffer.
        """
        # Coerce shapes
        def v(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != self.arm_dof:
                raise ValueError(f"expected length {self.arm_dof}, got {x.size}")
            return x

        q     = v(q)
        q_dot = v(q_dot)
        q_ref = v(q_ref)
        Kp    = v(Kp)
        Ki    = v(Ki)
        Kd    = v(Kd)
        u_max = v(u_max)
        u_min = v(u_min)

        buf = np.asarray(self.arm_pid_i_buffer, dtype=float).reshape(-1)
        if buf.size != self.arm_dof:
            buf = np.zeros(self.arm_dof, dtype=float)

        u_sat, err, buf_next = self.arm_pid_controller(
            ca.DM(q),
            ca.DM(q_dot),
            ca.DM(q_ref),
            ca.DM(Kp),
            ca.DM(Ki),
            ca.DM(Kd),
            ca.DM(buf),
            float(dt),
            ca.DM(self.g_ff),
            ca.DM(u_max),
            ca.DM(u_min),
        )

        self.arm_pid_i_buffer = np.asarray(buf_next).reshape(-1)[: self.arm_dof]
        return np.asarray(u_sat).reshape(-1)
