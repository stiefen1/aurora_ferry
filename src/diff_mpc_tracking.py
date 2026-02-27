from python_vehicle_simulator.lib.control import Control
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.obstacle import Obstacle
import numpy as np
from csnlp.wrappers import Mpc, NlpSensitivity
from csnlp import Nlp
import casadi as cs
from typing import Literal, Any, Tuple, List, Optional
from src.aurora import AuroraFerryActuatorsParameters, AuroraFerryParameters

SHIP_PARAMS = AuroraFerryParameters()
ACTUATORS_PARAMS = AuroraFerryActuatorsParameters()


class DiffMPCTrajectoryTracking(Control, Mpc):
    """
    Model-Predictive Controller for path tracking of the ReVolt ASV, based on 

    Reinforcement learning-based NMPC for tracking control of ASVs:Theory and experiments

    and 

    MPC-based Reinforcement Learning for a Simplified Freight Mission of Autonomous Surface Vehicles


    STATES:
    eta = [n, e, psi]
    nu = [u, v, r]

    INPUTS:
    u   = [alphas, forces]
        = [
            a1, a2, a3,
            f1, f2, f3
        ]
    
    Docstring for DiffMPCTrajectoryTracking

    States: x = [
                    eta,            # N, E, psi
                    nu,             # u, v, r
                    alpha_actual    # actual azimuth angles
                    n_actual        # actual propellers speed
                ]

    Control inputs: u = [
                            alpha   # azimuth angle setpoint
                            n       # propeller speed setpoint
                        ]

    The dynamics function accounts for both hull and thruster's dynamics.

    """

    def __init__(
            self,
            dynamics: cs.Function,
            horizon: int = 20,
            shooting: Literal["single", "multi"] = "multi",
            ship_params: Any = SHIP_PARAMS,             # Number of states (N, E, psi, u, v, r)
            actuators_params: Any = ACTUATORS_PARAMS,   # Number of control inputs ((n1, a1), (n2, a2), (n3, a3), (n4, a4))
            Q: Optional[np.ndarray] = None,             # state cost (stage)
            QN: Optional[np.ndarray] = None,            # state cost (terminal)
            Qpsi: Optional[float] = None,               # heading cost (stage & terminal)
            Ra: Optional[np.ndarray] = None,            # alpha cost (stage)
            Rn: Optional[np.ndarray] = None,            # propeller cost (stage)
            gamma: float = 1.0,
            u_des: float = 3.0,
    ):
        Control.__init__(self)
        Mpc.__init__(self, Nlp[cs.SX](), prediction_horizon=horizon, shooting=shooting)
        self.dynamics = dynamics
        self.u_des = u_des
        self.ship_params = ship_params
        self.actuators_params = actuators_params
        self.Q = Q if Q is not None else np.diag([1e3, 10, 10])
        self.QN = QN if QN is not None else self.Q.copy()
        self.Qpsi = Qpsi if Qpsi is not None else 1e1  # Reduced from 1e6
        self.Ra = Ra if Ra is not None else np.eye(self.nu // 2) * 1e-2
        self.Rn = Rn if Rn is not None else np.eye(self.nu // 2) * 1e-7  # Slightly increased from 1e-7
        self.Theta_v = 1e3*np.diag([1, 1, 1e4, 10, 10, 10])

        # Hyperparameters
        self.huber_penalty_slope = 200 # 10 # delta
        self.huber_penalty_weight = 100 # q_x,y
        self.heading_penalty_weight = 100 # 50 # q_psi
        self.singular_value_penalty = 1e-3 # epsilon -> for nonsigular thruster configuration
        self.singular_value_weight = 1e-8 # 1e-6 # 1e-5 # 8e-4 # 1e-9 #  5e-4 # 1e-5 # rho -> for nonsigular thruster configuration
        self.gamma = gamma # Discount factor -> self.gamma^k

        self.init_ocp()

    def init_ocp(self) -> None:
        # Declare states
        x, x0 = self.state("x", self.nx)                            # x.shape = (nx, N+1), x0.shape = (nx,)
        assert x is not None, f"x was not defined correctly. x={x}"

        # Actions
        u_lb = np.concatenate([self.actuators_params.a_min, self.actuators_params.n_min])
        u_ub = np.concatenate([self.actuators_params.a_max, self.actuators_params.n_max])
        
        u, _ = self.action(                                         # u.shape = (8, N)
            "u",
            self.nu,  # 4 alphas + 4 propeller speeds
            lb=u_lb.reshape(-1, 1),
            ub=u_ub.reshape(-1, 1)
        )

        # Declare non-learnable parameters
        ### Desired speed
        self.nu_des = self.nlp.parameter("nu_des", (3, 1))
        # print("nu_des shape: ", self.nu_des.shape)

        ### Desired waypoints (2D)
        self.desired_timestamped_wpts = self.nlp.parameter("wpts", shape=(3, self.horizon + 1)) 
        

        # Declare learnable parameters
        # e.g.
        # self.p = self.nlp.parameter("p")

        # Set dynamics
        self.set_nonlinear_dynamics(self.dynamics)

        # Set additional constraints
        # self.constraint("x_lb", x, ">=", self.ship_params.x_min)

        # Set cost function
        self.minimize(self.cost_function(x, u))

        # Initialize solver
        opts = {
            "error_on_fail": False,  # Don't crash on solver failure
            "expand": True,
            "print_time": False,
            "record_time": True,
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,  # Increased iterations
            "ipopt.tol": 1e-6,      # Relaxed tolerance
            "ipopt.acceptable_tol": 1e-4,  # Acceptable solution tolerance
            "ipopt.mu_init": 1e-3,  # Initial barrier parameter
            "ipopt.warm_start_init_point": "no",  # Use warm start
        }
        # Use IPOPT for nonlinar optimization
        self.init_solver(opts, "ipopt")

    def lagrange(self, xk: cs.SX, ak: cs.SX, nk: cs.SX, wk: cs.SX, nu_des: cs.SX) -> cs.SX:
        # Huber cost to penalize deviation from path
        pos_cost = self.huber_penalty_slope**2 * (cs.sqrt(1 + ((xk[0]-wk[0])**2 + (xk[1]-wk[1])**2) / self.huber_penalty_slope **2) - 1 ) 
        
        # Heading tracking cost
        heading_cost = 0.5 * (1 - cs.cos(xk[2] - wk[2]))

        # Singular configurations
        B = self.actuators_params.B(ak[0], ak[1], ak[2], ak[3])
        singular_cost = 1 / (self.singular_value_penalty + cs.det(B.T @ B))

        # Speed cost
        speed_cost = (xk[3:6]-nu_des).T @ self.Q @ (xk[3:6]-nu_des)

        # Control costs
        control_cost = nk.T @ self.Rn @ nk + ak.T @ self.Ra @ ak
        
        return self.huber_penalty_weight * pos_cost + \
             self.heading_penalty_weight * heading_cost + \
             self.singular_value_penalty * singular_cost + \
             speed_cost + control_cost
    
    def mayer(self, xN: cs.SX, wN: cs.SX, nu_des: cs.SX) -> cs.SX:
        xd = cs.vertcat(wN, nu_des)
        return (xN[0:6]-xd).T @ self.Theta_v @ (xN[0:6]-xd)
    
    def compute_cost_components(self, x: np.ndarray, u: np.ndarray, waypoints: np.ndarray) -> dict:
        """
        Compute individual cost components for analysis and tuning.
        
        Args:
            x: State trajectory (nx, N+1)
            u: Control trajectory (nu, N) 
            waypoints: Reference waypoints (2, N+1)
            
        Returns:
            Dictionary with individual cost components
        """
        costs = {
            'position_tracking': 0.0,
            'heading_tracking': 0.0, 
            'azimuth_control': 0.0,
            'propeller_control': 0.0,
            'terminal_position': 0.0,
            'terminal_heading': 0.0,
            'total': 0.0
        }
        
        # Stage costs
        for k in range(self.horizon):
            xk = x[:, k]
            uk = u[:, k] 
            wk = waypoints[:, k]
            
            ak = uk[0:4]  # azimuth angles
            nk = uk[4:8]  # propeller speeds
            
            # Position tracking cost
            pos_error = xk[0:2] - wk
            pos_cost = pos_error.T @ self.Q @ pos_error
            costs['position_tracking'] += float(pos_cost)
            
            # Heading tracking cost
            # dE = wk_next[1] - wk[1]
            # dN = wk_next[0] - wk[0] 
            # psi_des = np.arctan2(dE, dN)
            # psi_error = xk[2] - psi_des
            # # Wrap angle
            # psi_error = np.arctan2(np.sin(psi_error), np.cos(psi_error))
            # heading_cost = self.Qpsi * psi_error**2
            heading_cost = self.Qpsi * xk[4]**2
            costs['heading_tracking'] += float(heading_cost)
            
            # Control costs
            azimuth_cost = ak.T @ self.Ra @ ak
            prop_cost = nk.T @ self.Rn @ nk
            costs['azimuth_control'] += float(azimuth_cost)
            costs['propeller_control'] += float(prop_cost)
        
        # Terminal costs
        xN = x[:, -1]
        wN = waypoints[:, -1]
        wN_prev = waypoints[:, -2]
        
        # Terminal position cost
        pos_error_N = xN[0:2] - wN
        terminal_pos_cost = pos_error_N.T @ self.QN @ pos_error_N
        costs['terminal_position'] = float(terminal_pos_cost)
        
        # Terminal heading cost
        dE_N = wN[1] - wN_prev[1]
        dN_N = wN[0] - wN_prev[0]
        psi_des_N = np.arctan2(dE_N, dN_N)
        psi_error_N = xN[2] - psi_des_N
        psi_error_N = np.arctan2(np.sin(psi_error_N), np.cos(psi_error_N))
        terminal_heading_cost = self.Qpsi * psi_error_N**2
        costs['terminal_heading'] = float(terminal_heading_cost)
        
        # Total cost
        costs['total'] = sum(costs.values()) - costs['total']  # Subtract the 0 we initialized
        
        return costs
    
    def cost_function(self, x: cs.SX, u: cs.SX) -> cs.SX:
        cost: cs.SX = cs.SX(0)
        for k in range(self.horizon):
            xk, ak, nk, wk = x[:, k], u[0:4, k], u[4:8, k], self.desired_timestamped_wpts[:, k]
            cost += self.lagrange(xk, ak, nk, wk, self.nu_des)
        
        # Terminal cost - use difference between last two waypoints to compute desired heading
        wN = self.desired_timestamped_wpts[:, self.horizon]
        cost += self.mayer(x[:, self.horizon], wN, self.nu_des)
        return cost
    
    
    def __get__(
            self,
            timestamped_wpts: List[Tuple[float, float, float]],
            eta: np.ndarray,
            nu: np.ndarray,
            alpha: np.ndarray,
            n: np.ndarray,
            current: Current,
            wind: Wind,
            obstacles: List = [],
            target_vessels: List = [],
            return_cost_analysis: bool = False,
            u_des: Optional[float] = None
        ):
        
        waypoints_array = np.array(timestamped_wpts).T # reshape(3, self.horizon + 1)
        
        sol = self.solve(
            pars={
                "x_0": np.concatenate([eta, nu, alpha, n]).reshape(14, 1),
                "wpts": waypoints_array,
                "nu_des": np.array([u_des or self.u_des, 0, 0])
            })
        
        # Extract control commands for the first time step
        u_cmd = sol.vals["u"][:, 0].full().flatten()
        alpha_cmd = u_cmd[0:4]  
        n_cmd = u_cmd[4:8]     

        # Extract trajectory
        self.x_prev = sol.vals["x"].full()
        self.sol_prev = sol

        # mpc_ = NlpSensitivity(self)
        # print(mpc_.parametric_sensitivity(self.f, solution=sol, second_order=False)) 
        
        # Optional: Compute and return cost analysis
        if return_cost_analysis:
            # Get full state and control trajectories
            x_traj = sol.vals["x"].full()
            u_traj = sol.vals["u"].full()
            
            # Compute individual cost components
            cost_components = self.compute_cost_components(x_traj, u_traj, waypoints_array)
            
            return [np.array([alpha_cmd[i], n_cmd[i]]) for i in range(4)], cost_components
        
        return [np.array([alpha_cmd[i], n_cmd[i]]) for i in range(4)]

    def reset(self):
        pass

    @property
    def nx(self) -> int:
        return 2 * self.ship_params.Minv.shape[0] + self.nu
    
    @property
    def nu(self) -> int:
        return 2 * self.actuators_params.nu

    @property
    def horizon(self) -> int:
        return self.prediction_horizon

if __name__ == "__main__":
    from discrete_dynamics import get_discrete_3dof_dynamics_as_fn
    import matplotlib.pyplot as plt

    def test_mpc_trajectory_tracking():
        # Simulation parameters
        dt = 1  # 500ms sampling time (increased for slower trajectory)
        T_sim = 80.0  # 80 seconds simulation (increased to see more of the trajectory)
        N = int(T_sim / dt)
        H = 30  # MPC horizon
        
        # Get discrete dynamics function
        discrete_dynamics = get_discrete_3dof_dynamics_as_fn(2)
        
        # Create MPC controller
        mpc = DiffMPCTrajectoryTracking(discrete_dynamics, H)
        
        # Define reference trajectory (a simple path at 0.2 m/s)
        def get_reference_trajectory(t):
            # Constant speed trajectory at 0.2 m/s
            speed = 0.5  # m/s
            
            # Trajectory segments with smooth transitions
            if t < 50:  # First segment: move northeast (45 degrees)
                distance = speed * t
                return [distance * np.cos(np.pi/4), distance * np.sin(np.pi/4)]  # N, E
            elif t < 100:  # Second segment: move north 
                # Continue from end of first segment
                t_segment = t - 50
                distance_segment = speed * t_segment
                base_n = 50 * speed * np.cos(np.pi/4)
                base_e = 50 * speed * np.sin(np.pi/4)
                return [base_n + distance_segment, base_e]  # Move north
            else:  # Third segment: move east
                t_segment = t - 100
                distance_segment = speed * t_segment
                base_n = 50 * speed * np.cos(np.pi/4) + 50 * speed
                base_e = 50 * speed * np.sin(np.pi/4)
                return [base_n, base_e + distance_segment]  # Move east
        
        # Initialize state and control arrays
        x = np.zeros((14, N+1))  # State history
        u_mpc = np.zeros((8, N))  # MPC control commands
        ref_traj = np.zeros((2, N+1))  # Reference trajectory
        
        # Initial state: start at origin
        x[:, 0] = np.array([0, 0, np.deg2rad(45), 0.5, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50])
        
        # Simulate with MPC control
        for k in range(N):
            t = k * dt
            
            # Get current reference point and future waypoints for MPC
            current_ref = get_reference_trajectory(t)
            ref_traj[:, k] = current_ref
            
            # Create waypoint sequence for MPC horizon
            waypoints = []
            for h in range(H + 1):
                future_time = t + h * dt
                waypoints.append(get_reference_trajectory(future_time))
            
            # Get current state components
            eta = x[0:3, k]      # [N, E, psi]
            nu = x[3:6, k]       # [u, v, r]  
            alpha = x[6:10, k]   # azimuth angles
            n = x[10:14, k]      # propeller speeds
            
            # Get MPC control commands with cost analysis every 10 steps
            try:
                if k % 10 == 0:  # Analyze costs every 10 steps
                    mpc_output, cost_analysis = mpc.__get__(
                        waypoints,
                        eta, nu, alpha, n,
                        None, None, [], [],  # current, wind, obstacles, target_vessels
                        True  # return_cost_analysis
                    )
                    
                    print(f"\nTime {t:.1f}s - Cost Analysis:")
                    print(f"  Position tracking: {cost_analysis['position_tracking']:.2f}")
                    print(f"  Heading tracking:  {cost_analysis['heading_tracking']:.2f}")
                    print(f"  Azimuth control:   {cost_analysis['azimuth_control']:.4f}")
                    print(f"  Propeller control: {cost_analysis['propeller_control']:.4f}")
                    print(f"  Terminal costs:    {cost_analysis['terminal_position']:.2f} + {cost_analysis['terminal_heading']:.2f}")
                    print(f"  Total cost:        {cost_analysis['total']:.2f}")
                else:
                    mpc_output = mpc.__get__(
                        waypoints,
                        eta, nu, alpha, n,
                        None, None, [], []  # current, wind, obstacles, target_vessels
                    )
                
                # Extract control commands (alpha, n for each thruster)
                for i in range(4):
                    u_mpc[i, k] = mpc_output[i][0]      # azimuth angle commands
                    u_mpc[4+i, k] = mpc_output[i][1]    # propeller speed commands
                    
            except Exception as e:
                print(f"MPC failed at time {t:.1f}s: {str(e)[:100]}...")  # Truncate long error messages
                # Use previous commands if MPC fails
                if k > 0:
                    u_mpc[:, k] = u_mpc[:, k-1]
                else:
                    # If first step fails, use small commands
                    u_mpc[:4, k] = 0.1  # Small azimuth angles
                    u_mpc[4:, k] = 5.0  # Small propeller speeds
            
            # Simulate dynamics forward
            x[:, k+1] = np.array(discrete_dynamics(x[:, k], u_mpc[:, k])).flatten()
        
        # Get final reference point
        ref_traj[:, -1] = get_reference_trajectory(T_sim)
        
        # Time vectors
        time = np.linspace(0, T_sim, N+1)
        time_u = np.linspace(0, T_sim, N)
        
        # Plot results
        plt.figure(figsize=(15, 12))
        
        # 1. Trajectory tracking in N-E coordinates
        plt.subplot(2, 3, 1)
        plt.plot(ref_traj[1, :], ref_traj[0, :], 'r--', linewidth=2, label='Reference')
        plt.plot(x[1, :], x[0, :], 'b-', linewidth=2, label='Actual')
        plt.plot(x[1, 0], x[0, 0], 'go', markersize=8, label='Start')
        plt.plot(x[1, -1], x[0, -1], 'ro', markersize=8, label='End')
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.title('Trajectory Tracking')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        
        # 2. Position tracking errors
        plt.subplot(2, 3, 2)
        pos_error = np.sqrt((x[0, :] - ref_traj[0, :])**2 + (x[1, :] - ref_traj[1, :])**2)
        plt.plot(time, pos_error, 'r-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Position Tracking Error')
        plt.grid(True)
        
        # 3. Actual azimuth angles
        plt.subplot(2, 3, 3)
        for i in range(4):
            plt.plot(time, np.rad2deg(x[6+i, :]), label=f'α{i+1} actual', linewidth=2)
            plt.plot(time_u, np.rad2deg(u_mpc[i, :]), '--', alpha=0.7, label=f'α{i+1} cmd')
        plt.xlabel('Time (s)')
        plt.ylabel('Azimuth Angle (deg)')
        plt.title('Azimuth Angles (Actual vs Commands)')
        plt.grid(True)
        plt.legend()
        
        # 4. Actual propeller speeds
        # 4. Actual propeller speeds
        plt.subplot(2, 3, 4)
        for i in range(4):
            plt.plot(time, x[10+i, :], label=f'n{i+1} actual', linewidth=2)
            plt.plot(time_u, u_mpc[4+i, :], '--', alpha=0.7, label=f'n{i+1} cmd')
        plt.xlabel('Time (s)')
        plt.ylabel('Propeller Speed')
        plt.title('Propeller Speeds (Actual vs Commands)')
        plt.grid(True)
        plt.legend()
        
        # 5. Ship velocities
        plt.subplot(2, 3, 5)
        plt.plot(time, x[3, :], 'r-', label='u (surge)')
        plt.plot(time, x[4, :], 'g-', label='v (sway)')
        plt.plot(time, x[5, :], 'b-', label='r (yaw rate)')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.title('Ship Velocities')
        plt.grid(True)
        plt.legend()
        
        # 6. Ship heading
        plt.subplot(2, 3, 6)
        plt.plot(time, np.rad2deg(x[2, :]), 'k-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Heading (deg)')
        plt.title('Ship Heading')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        final_error = pos_error[-1]
        max_error = np.max(pos_error)
        mean_error = np.mean(pos_error)
        
        print("=== MPC Trajectory Tracking Results ===")
        print(f"Final position error: {final_error:.2f} m")
        print(f"Maximum tracking error: {max_error:.2f} m") 
        print(f"Mean tracking error: {mean_error:.2f} m")
        print(f"Simulation completed successfully!")
    
    # Run the test
    test_mpc_trajectory_tracking()