import casadi as cs
from aurora import AuroraFerryActuatorsParameters, AuroraFerryParameters
from python_vehicle_simulator.utils.math_fn import R_casadi
from typing import Tuple

SHIP_PARAMS = AuroraFerryParameters()
ACTUATORS_PARAMS = AuroraFerryActuatorsParameters()

def get_continuous_3dof_dynamics_as_fn() -> cs.Function:
    """
    Docstring for get_full_3dof_dynamics_as_fn
    
    :param x: state containing [eta, nu, alpha, n]
    :type x: cs.SX
    :param u: control input [alpha_d, n_d]
    :type u: cs.SX
    :return: state derivative
    :rtype: Function
    """
    x = cs.SX.sym('x', 14)
    u = cs.SX.sym('u', 8)

    f = cs.SX.zeros(14)

    eta, nu, alpha, n = x[0:3], x[3:6], x[6:10], x[10:14]
    alpha_d, n_d = u[0:4], u[4:8]

    # Actuator's generalized force
    thrust = ACTUATORS_PARAMS.k_pos * n * n
    tau_actuators = ACTUATORS_PARAMS.B(alpha[0], alpha[1], alpha[2], alpha[3]) @ thrust

    # Hull dynamics (body frame)
    Minv, C, D = SHIP_PARAMS.Minv, SHIP_PARAMS.CA(nu) + SHIP_PARAMS.CRB(nu), SHIP_PARAMS.D
    f[3:6] = Minv @ (tau_actuators - C @ nu - D @ nu)

    # Ship's kinematics (NED frame)
    f[0:3] = cs.mtimes(R_casadi(eta[2]), nu)

    # Actuator's dynamics
    f[6:10] = (alpha_d - alpha) / ACTUATORS_PARAMS.T_a
    f[10:14] = (n_d - n) / ACTUATORS_PARAMS.T_n

    return cs.Function('continuous_dynamics', [x, u], [f])

def get_discrete_3dof_dynamics_as_fn(dt: float) -> cs.Function:
    """
    Create discrete-time dynamics using RK4 integration
    
    :param dt: sampling time
    :type dt: float
    :return: discrete-time dynamics as cs.Function
    :rtype: cs.Function
    """
    # Get continuous dynamics
    f_continuous = get_continuous_3dof_dynamics_as_fn()
    
    # Create symbolic variables
    x = cs.SX.sym('x', 14)
    u = cs.SX.sym('u', 8)
    
    # RK4 integration
    k1 = f_continuous(x, u)
    k2 = f_continuous(x + 0.5 * dt * k1, u)
    k3 = f_continuous(x + 0.5 * dt * k2, u)
    k4 = f_continuous(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    return cs.Function('discrete_dynamics', [x, u], [x_next])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    def test_discrete_dynamics():
        # Simulation parameters
        dt = 0.1  # 100ms sampling time
        T_sim = 60.0  # 60 seconds simulation
        N = int(T_sim / dt)
        
        # Get discrete dynamics function
        discrete_dynamics = get_discrete_3dof_dynamics_as_fn(dt)
        
        # Initialize state and control arrays
        x = np.zeros((14, N+1))  # State history
        u = np.zeros((8, N))     # Control input history
        
        # Initial state at origin
        x[:, 0] = 0
        
        # Define step inputs at different times
        for k in range(N):
            t = k * dt
            
            # Step inputs for azimuth angles (radians)
            if 10 <= t < 50:  # From 10s to 50s
                u[0, k] = 1  # Azimuth 1
            if 15 <= t < 45:
                u[1, k] = -1.8  # Azimuth 2
            if 20 <= t < 40:
                u[2, k] = 1.4  # Azimuth 3
            if 25 <= t < 35:
                u[3, k] = -0.2  # Azimuth 4
                
            # Step inputs for propeller speeds
            if 5 <= t < 55:
                u[4, k] = 30.0  # Propeller 1
            if 8 <= t < 52:
                u[5, k] = 80.0   # Propeller 2
            if 12 <= t < 48:
                u[6, k] = 60.0  # Propeller 3
            if 18 <= t < 42:
                u[7, k] = 40.0   # Propeller 4
        
        # Simulate dynamics
        for k in range(N):
            x[:, k+1] = np.array(discrete_dynamics(x[:, k], u[:, k])).flatten()
        
        # Time vector
        time = np.linspace(0, T_sim, N+1)
        time_u = np.linspace(0, T_sim, N)
        
        # Plot results
        plt.figure(figsize=(15, 12))
        
        # 1. N-E trajectory
        plt.subplot(3, 3, 1)
        plt.plot(x[1, :], x[0, :], 'b-', linewidth=2)
        plt.plot(x[1, 0], x[0, 0], 'go', markersize=8, label='Start')
        plt.plot(x[1, -1], x[0, -1], 'ro', markersize=8, label='End')
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.title('Ship Trajectory')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        
        # 2. Azimuth angles (2x2)
        for i in range(4):
            plt.subplot(3, 4, 5 + i)
            plt.plot(time, x[6+i, :], 'b-', label=f'Actual α{i+1}')
            plt.plot(time_u, u[i, :], 'r--', label=f'Setpoint α{i+1}')
            # Add actuator limits
            plt.axhline(y=ACTUATORS_PARAMS.a_min[i], color='k', linestyle=':', alpha=0.7, label='Min limit')
            plt.axhline(y=ACTUATORS_PARAMS.a_max[i], color='k', linestyle=':', alpha=0.7, label='Max limit')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (rad)')
            plt.title(f'Azimuth {i+1}')
            plt.grid(True)
            plt.legend()
        
        # 3. Propeller speeds (2x2)
        for i in range(4):
            plt.subplot(3, 4, 9 + i)
            plt.plot(time, x[10+i, :], 'b-', label=f'Actual n{i+1}')
            plt.plot(time_u, u[4+i, :], 'r--', label=f'Setpoint n{i+1}')
            # Add actuator limits
            plt.axhline(y=ACTUATORS_PARAMS.n_min[i], color='k', linestyle=':', alpha=0.7, label='Min limit')
            plt.axhline(y=ACTUATORS_PARAMS.n_max[i], color='k', linestyle=':', alpha=0.7, label='Max limit')
            plt.xlabel('Time (s)')
            plt.ylabel('Speed')
            plt.title(f'Propeller {i+1}')
            plt.grid(True)
            plt.legend()
        
        # 4. Body velocities and heading
        plt.subplot(3, 3, 2)
        plt.plot(time, x[3, :], 'r-', label='u (surge)')
        plt.plot(time, x[4, :], 'g-', label='v (sway)')
        plt.plot(time, x[5, :], 'b-', label='r (yaw rate)')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.title('Body Velocities')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 3, 3)
        plt.plot(time, np.rad2deg(x[2, :]), 'k-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Heading (deg)')
        plt.title('Ship Heading')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Run the test
    test_discrete_dynamics()