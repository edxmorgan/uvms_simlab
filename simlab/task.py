import numpy as np

class Task:

    @staticmethod
    def square_velocity_uv_ref(node, t: float,
                            T_side: float = 10.0,
                            speed: float = 0.1,
                            manput: bool = False) -> np.ndarray:
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
        # node.get_logger().info(f"ref path period {t_mod}")

        # Determine which side (leg) of the square we are on
        if t_mod < T_side:
            # Leg 1: move along +x
            u, v = speed, 0.0
            dq0, dq1, dq2, dq3, dq4 = 0.1, -0.1, 0.1, -0.1, 0.0
        elif t_mod < 2 * T_side:
            # Leg 2: move along +y
            u, v = 0.0, speed
            dq0, dq1, dq2, dq3, dq4  = -0.1, 0.1, -0.2, 0.1, 0.0
        elif t_mod < 3 * T_side:
            # Leg 3: move along -x
            u, v = -speed, 0.0
            dq0, dq1, dq2, dq3, dq4  = 0.1, -0.1, 0.14, -0.1, 0.0
        else:
            # Leg 4: move along -y
            u, v = 0.0, -speed
            dq0, dq1, dq2, dq3, dq4  = -0.1, 0.1, -0.1, 0.01, 0.0

        # Zero out the other velocity components
        w, p, q, r = 0.0, 0.0, 0.0, 0.0

        # Return the desired output format
        if manput:
            # Return 10-element array (extra zeros at the end)
            return np.array([u, v, w, p, q, r, dq0, dq1, dq2, dq3, dq4])
        else:
            # Return 6-element array
            return np.array([u, v, w, p, q, r, 0.0, 0.0, 0.0, 0.0, 0.0])



    @staticmethod
    def square_velocity_ops_ref(node, t: float,
                            T_side: float = 10.0,
                            speed: float = 0.1) -> np.ndarray:
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
        # node.get_logger().info(f"ref path period {t_mod}")

        # Determine which side (leg) of the square we are on
        if t_mod < T_side:
            # Leg 1: move along +x
            u, v = speed, 0.0
        elif t_mod < 2 * T_side:
            # Leg 2: move along +y
            u, v = 0.0, speed
        elif t_mod < 3 * T_side:
            # Leg 3: move along -x
            u, v = -speed, 0.0
        else:
            # Leg 4: move along -y
            u, v = 0.0, -speed

        # Zero out the other velocity components
        w, p, q, r = 0.0, 0.0, 0.0, 0.0


        # Return 6-element array
        return np.array([u, v, w, p, q, r])
