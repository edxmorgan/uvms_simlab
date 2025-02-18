# task.py
import numpy as np

class Task:
    def __init__(self, initial_pos):
        self.initial_pos = initial_pos

    def square_velocity_uv_ref(self, t_now, T_side=10.0, speed=0.05, depth_rate=0.005):
        """
        Produces:
          1) A velocity vector (u, v, w, p, q, r, q1, q2, q3, q4) for a periodic square path
             in the XY-plane, with an additional constant descent rate along Z,
          2) The integrated state [x, y, z, roll, pitch, yaw, s1, s2, s3, s4] where
             s1..s4 correspond to q1..q4 (or any other states you want to integrate).

        :param t_now: Current time (in seconds)
        :param T_side: Time (in seconds) for one side of the square.
        :param speed: Desired linear speed along a side.
        :param depth_rate: The constant rate (in m/s) at which depth (z) decreases.
        :return:
            velocity: np.array of shape (10,) with 
                      [u, v, w, p, q, r, q1, q2, q3, q4]
            state:   np.array of shape (10,) with 
                      [x, y, z, roll, pitch, yaw, s1, s2, s3, s4]
        """
        # ------------------------------------------
        # 1) If first time calling, init last time & state.
        #    We'll store a 10-element state: 
        #    [x, y, z, roll, pitch, yaw, s1, s2, s3, s4]
        # ------------------------------------------
        if not hasattr(self, '_last_t'):
            self._last_t = t_now
        
        if not hasattr(self, '_state'):
            self._state = self.initial_pos.copy()

        # ------------------------------------------
        # 2) Calculate dt.
        # ------------------------------------------
        dt = t_now - self._last_t
        self._last_t = t_now

        # ------------------------------------------
        # 3) Determine which side of the square.
        # ------------------------------------------
        cycle_time = 4.0 * T_side
        t_mod = t_now % cycle_time

        # ------------------------------------------
        # 4) Initialize velocity vector: 
        #    [u, v, w, p, q, r, q1, q2, q3, q4]
        # ------------------------------------------
        velocity = np.zeros(10, dtype=float)

        # ------------------------------------------
        # 5) Piecewise constant velocity for X and Y directions:
        #    - Side 1: +X
        #    - Side 2: +Y
        #    - Side 3: -X
        #    - Side 4: -Y
        # ------------------------------------------
        if 0 <= t_mod < T_side:
            velocity[0] = speed   # move in +X direction
        elif T_side <= t_mod < 2 * T_side:
            velocity[1] = speed   # move in +Y direction
        elif 2 * T_side <= t_mod < 3 * T_side:
            velocity[0] = -speed  # move in -X direction
        else:
            velocity[1] = -speed  # move in -Y direction

        # ------------------------------------------
        # 6) Add a constant downward (decreasing depth) velocity
        # ------------------------------------------
        velocity[2] = -depth_rate

        # ------------------------------------------
        # 7) Integrate velocity to update state.
        #    state[i] = state[i] + velocity[i] * dt
        # ------------------------------------------
        for i in range(10):
            self._state[i] += velocity[i] * dt

        # ------------------------------------------
        # 8) Return velocity & the new integrated state.
        # ------------------------------------------
        return velocity.flatten(), self._state.copy().flatten()
