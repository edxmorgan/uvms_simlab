# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pandas as pd

class Params:

    # Ocean current velocities. 
    v_flow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # (m/s). Assume irrotational, constant.

    # Thrust configuration matrix by BlueROV2 Heavy. Converts thrust force to body and vice versa. 
    # TODO Check Calcs to determine how to find. 
    # thrust_config = np.array([
    #     [-0.7070, -0.7070,  0.7070,  0.7070,  0.0000,  0.0000,  0.0000, 0.0000],
    #     [ 0.7070, -0.7070,  0.7070, -0.7070,  0.0000,  0.0000,  0.0000, 0.0000],
    #     [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -1.0000, -1.0000, 1.0000],
    #     [ 0.0000,  0.0000,  0.0000,  0.0000, -0.2180, -0.2180,  0.2180, 0.2180],
    #     [ 0.0000,  0.0000,  0.0000,  0.0000, -0.1200,  0.1200, -0.1200, 0.1200],
    #     [ 0.1888, -0.1888, -0.1888,  0.1888,  0.0000,  0.0000,  0.0000, 0.0000]])
    
    # Alternative thrust config matrix. 
    thrust_config = np.array([[0.707, 0.707, -0.707, -0.707, 0.0, 0.0, 0.0, 0.0],
                            [-0.707, 0.707, -0.707, 0.707, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0,-1.0],
                            [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
                            [0.06, 0.06, -0.06, -0.06, 0.12, -0.12, 0.12, -0.12],
                            [-0.1888, 0.1888, 0.1888, -0.1888, 0.0, 0.0, 0.0, 0.0]])
    
    # Thrust coefficient matrix
    K = np.diag([40, 40, 40, 40, 40, 40, 40, 40])

    ### Parameters in rigid body dynamics and restoring forces
    # Based on BlueRobotics 2018b technical specs. 
    # Based on Table 5.1
    m = 11.5 #(kg)
    W = m*9.81 #(N). 112.8 N. Weight. 
    B = 114.8 #(N). bouyancy 
    rb  = np.array([0, 0, 0]) #(m). center of buoyancy (CoB) coincides with the center of origin
    rg = np.array([0, 0, 0.02]) #(m). 

    # Axis inertias. 
    # BAsed on Table 5.1.
    I_x = 0.16 #(kg m2)
    I_y = 0.16 #(kg m2)
    I_z = 0.16 #(kg m2)
    I_xz = 0
    Io = np.array([I_x, I_y, I_z, I_xz])

    # Added mass parameters.
    # Based on Table 5.2. 
    X_du = -5.5 #(kg). Surge. 
    Y_dv = -12.7 #(kg). Sway. 
    Z_dw = -14.57 #(kg). Heave. 
    K_dp = -0.12 #(kg m2/rad). Roll.
    M_dq = -0.12 #(kg m2/rad). Pitch. 
    N_dr = -0.12 #(kg m2/rad). Yaw. 
    added_m = np.array([X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr])

    coupl_added_m = np.array([0, 0, 0, 0]) # ASSUMING decoupling motion

    # Linear damping coeffs. 
    Xu = -4.03 #(Ns/m). Surge. 
    Yv = -6.22 #(Ns/m). Sway.
    Zw = -5.18 #(Ns/m). Heave.  
    Kp = -0.07 #(Ns/rad). Roll.
    Mq = -0.07 #(Ns/rad). Pitch.
    Nr = -0.07 #(Ns/rad). Yaw. 
    linear_dc = np.array([Xu, Yv, Zw, Kp,  Mq, Nr])

    # Quadratic damping coeffs. 
    Xuu = -18.18 #(Ns2/m2). Surge. 
    Yvv = -21.66 #(Ns2/m2). Sway. 
    Zww = -36.99 #(Ns2/m2). Heave. 
    Kpp = -1.55 #(Ns2/rad2). Roll. 
    Mqq = -1.55 #(Ns2/rad2). Pitch. 
    Nrr = -1.55 #(Ns2/rad2). Yaw. 
    quadratic_dc = np.array([Xuu, Yvv, Zww, Kpp, Mqq, Nrr])


    rg = np.array([0, 0, 0.02]) #(m). 
    rb = np.array([0, 0, 0]) #(m). center of buoyancy (CoB) coincides with the center of origin

    sim_n = 100 # time horizon in seconds
    at_surface = 0.0
    below_surface = 1.0 #random
    delta_t = 0.05
    N = int(sim_n/delta_t) # number of control intervals


    kp = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ki = np.array([0, 0, 0, 0, 0, 0])
    kd = np.array([2, 2, 2, 2, 2, 2])

    sim_p = np.concatenate(( np.array([m]) , np.array([W]), np.array([B]), 
                                           rg, rb, Io, added_m, coupl_added_m, linear_dc, quadratic_dc, v_flow))
    
    u_min = np.array([-3, -3, -3, -3, -3, -3])
    u_max = np.array([3, 3, 3, 3, 3, 3])
    # # https://gist.github.com/edxmorgan/4d38ca349537a36214927f16359848a1


    # # ----------------------------
    # # User-specified parameters
    # # ----------------------------
    # forward_poly_order = 9    # Order for PWM -> Thrust mapping
    # reverse_poly_order = 25   # Order for Thrust -> PWM mapping
    # no_thrusters = 8          # Number of thrusters for CasADi mapping

    # # Define bounds for the forward mapping (PWM -> Thrust)
    # # (Adjust these values as appropriate for your system.)
    # pwm_input_lb = 1100.0  # Lower bound for PWM input
    # pwm_input_ub = 1900.0  # Upper bound for PWM input

    # # ----------------------------
    # # Read CSV file and convert thrust from Ibf (lbf) to Newtons (N)
    # # ----------------------------
    # csv = pd.read_csv('t200.csv')
    # conversion_factor = 4.44822
    # csv['thrust 16V'] = csv['thrust 16V'] * conversion_factor

    # # ----------------------------
    # # Forward mapping: PWM -> Thrust
    # # ----------------------------

    # # Extract data from CSV
    # x_pwm = csv['pwm 16V'].tolist()
    # y_thrust = csv['thrust 16V'].tolist()

    # # Fit a polynomial of user-specified order: PWM -> Thrust
    # coefs = np.polyfit(x_pwm, y_thrust, forward_poly_order)
    # ffit = np.polyval(coefs, x_pwm)

    # # # Plot the forward mapping (raw data vs. fitted curve)
    # # plt.figure(figsize=(10, 6))
    # # plt.plot(x_pwm, y_thrust, 'bo', label='Original Data')
    # # plt.plot(x_pwm, ffit, 'r-', label=f'Fitted Polynomial (order {forward_poly_order})')
    # # plt.xlabel('PWM 16V')
    # # plt.ylabel('Thrust (N)')
    # # plt.legend()
    # # plt.title('PWM to Thrust Conversion for bluerov T200 thruster')
    # # plt.show()

    # # # Calculate and print the RMS error for the forward mapping
    # # RMS = np.sqrt(np.mean((np.array(ffit) - np.array(y_thrust))**2))
    # # print("Forward mapping RMS error: {:.3f} N".format(RMS))

    # # Define output bounds for thrust based on the polynomial evaluated at the PWM bounds
    # thrust_output_lb = np.polyval(coefs, pwm_input_lb)
    # thrust_output_ub = np.polyval(coefs, pwm_input_ub)

    # # Create a CasADi function for the forward mapping with bounds.
    # # The input is first saturated (clipped) to [pwm_input_lb, pwm_input_ub],
    # # then the polynomial is evaluated, and the output is clipped to [thrust_output_lb, thrust_output_ub].
    # pwm = ca.SX.sym('pwm')

    # # Saturate the PWM input:
    # pwm_sat = ca.fmax(pwm_input_lb, ca.fmin(pwm, pwm_input_ub))

    # # Build the polynomial expression using the saturated input.
    # thrust_expr = 0
    # for i, coef in enumerate(coefs):
    #     # Exponent decreases from forward_poly_order down to 0.
    #     exponent = forward_poly_order - i
    #     thrust_expr += coef * pwm_sat**exponent

    # # Saturate the output thrust:
    # thrust_bounded = ca.fmax(thrust_output_lb, ca.fmin(thrust_expr, thrust_output_ub))

    # # Create the forward mapping CasADi function and map it to no_thrusters.
    # pwm_to_thrust = ca.Function('Pwm_to_thrust', [pwm], [thrust_bounded]).map(no_thrusters)

    # # ----------------------------
    # # Reverse mapping: Thrust -> PWM
    # # ----------------------------

    # # For the inverse mapping, we fit a polynomial with thrust as the independent variable.
    # inv_coefs = np.polyfit(y_thrust, x_pwm, reverse_poly_order)
    # ffit_inv = np.polyval(inv_coefs, y_thrust)

    # # # Plot the reverse mapping (raw data vs. fitted curve)
    # # plt.figure(figsize=(10, 6))
    # # plt.plot(y_thrust, x_pwm, 'bo', label='Original Inverse Data')
    # # plt.plot(y_thrust, ffit_inv, 'r-', label=f'Fitted Inverse Polynomial (order {reverse_poly_order})')
    # # plt.xlabel('Thrust (N)')
    # # plt.ylabel('PWM 16V')
    # # plt.legend()
    # # plt.title('Thrust to PWM Conversion for bluerov T200 thruster')
    # # plt.show()

    # # Calculate and print the RMS error for the reverse mapping
    # # RMS_inv = np.sqrt(np.mean((np.array(ffit_inv) - np.array(x_pwm))**2))
    # # print("Reverse mapping RMS error: {:.3f}".format(RMS_inv))

    # # For the reverse mapping, set input bounds based on the thrust data.
    # # Here, we use the min and max values from the original thrust data.
    # thrust_input_lb = min(y_thrust)
    # thrust_input_ub = max(y_thrust)

    # # And output bounds for PWM based on the original PWM data.
    # pwm_output_lb = min(x_pwm)
    # pwm_output_ub = max(x_pwm)

    # # Create a CasADi function for the reverse mapping with bounds.
    # # The input thrust is clipped to [thrust_input_lb, thrust_input_ub],
    # # then the inverse polynomial is evaluated, and the output PWM is clipped to [pwm_output_lb, pwm_output_ub].
    # thrust = ca.SX.sym('thrust')

    # # Saturate the thrust input:
    # thrust_sat = ca.fmax(thrust_input_lb, ca.fmin(thrust, thrust_input_ub))

    # # Build the inverse polynomial expression using the saturated thrust input.
    # pwm_expr = 0
    # for i, coef in enumerate(inv_coefs):
    #     exponent = reverse_poly_order - i
    #     pwm_expr += coef * thrust_sat**exponent

    # # Saturate the PWM output:
    # pwm_bounded = ca.fmax(pwm_output_lb, ca.fmin(pwm_expr, pwm_output_ub))

    # # Create the reverse mapping CasADi function and map it to no_thrusters.
    # thrust_to_pwm = ca.Function('Thrust_to_pwm', [thrust], [pwm_bounded]).map(no_thrusters)
    # n_thrust = ca.SX.sym('thrust',8)
    # getNpwm_func = ca.Function('getNpwm', [n_thrust], [thrust_to_pwm(n_thrust)])