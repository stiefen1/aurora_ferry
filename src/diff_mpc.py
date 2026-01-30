from python_vehicle_simulator.lib.control import Control
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.obstacle import Obstacle
import numpy as np
from csnlp.wrappers import Mpc
from csnlp import Nlp
import casadi as cs
from typing import Literal, Any, Tuple, List, Optional
from aurora import AuroraFerryActuatorsParameters, AuroraFerryParameters

SHIP_PARAMS = AuroraFerryParameters()
ACTUATORS_PARAMS = AuroraFerryActuatorsParameters()





class DiffMPCTrajectoryTracking(Control, Mpc):
    """
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
            Ra: Optional[np.ndarray] = None,            # alpha cost (stage)
            Rn: Optional[np.ndarray] = None             # propeller cost (stage)
    ):
        Control.__init__(self)
        Mpc.__init__(self, Nlp[cs.SX](), prediction_horizon=horizon, shooting=shooting)
        self.dynamics = dynamics
        self.ship_params = ship_params
        self.actuators_params = actuators_params
        self.Q = Q or np.eye(2)
        self.QN = QN or self.Q.copy()
        self.Ra = Ra or np.eye(self.nu // 2) * 1e-3
        self.Rn = Rn or np.eye(self.nu // 2) * 1e-6
        self.init_ocp()

    def init_ocp(self) -> None:
        # Declare states
        x, x0 = self.state("x", self.nx)                            # x.shape = (nx, N+1), x0.shape = (nx,)
        assert x is not None, f"x was not defined correctly. x={x}"

        # Declare actions
        alphas, _ = self.action(                                    # alphas.shape = (4, N)
            "alphas",
            self.actuators_params.a_min.shape[0],
            lb=self.actuators_params.a_min.reshape(-1, 1),
            ub=self.actuators_params.a_max.reshape(-1, 1)
        )
        ns, _ = self.action(                                        # ns.shape = (4, N)
            "ns",
            self.actuators_params.n_min.shape[0],
            lb=self.actuators_params.n_min.reshape(-1, 1),
            ub=self.actuators_params.n_max.reshape(-1, 1)
        )
        
        # print("alphas: ", alphas)
        # print("exp: ", alphas_exp)
        # Declare non-learnable parameters
        ### Desired waypoints (2D)
        self.desired_timestamped_wpts = self.nlp.parameter("wpts", shape=(2, self.horizon + 1)) 
        
        # Declare learnable parameters
        # e.g.
        # self.p = self.nlp.parameter("p")

        # Set dynamics
        self.set_nonlinear_dynamics(self.dynamics)

        # Set additional constraints
        # self.constraint("x_lb", x, ">=", self.ship_params.x_min)

        # Set cost function
        self.minimize( self.cost_function(x, alphas, ns) )

        # Initialize solver
        opts = {
            "error_on_fail": True,
            "expand": True,
            "print_time": False,
            "record_time": True,
            "verbose": True,
            "printLevel": "none",
        }
        self.init_solver(opts, "qpoases", type="conic")

    def lagrange(self, xk: cs.SX, ak: cs.SX, nk: cs.SX, wk: cs.SX) -> cs.SX:
        return (xk[0:2]-wk).T @ self.Q @ (xk[0:2]-wk) + nk.T @ self.Rn @ nk + ak.T @ self.Ra @ ak
    
    def mayer(self, xN: cs.SX, wN: cs.SX) -> cs.SX:
        return (xN[0:2]-wN).T @ self.QN @ (xN[0:2]-wN) 
    
    def cost_function(self, x: cs.SX, alphas: cs.SX, ns: cs.SX, ) -> cs.SX:
        cost: cs.SX = cs.SX(0)
        for k in range(self.horizon):
            xk, ak, nk, wk = x[:, k], alphas[:, k], ns[:, k], self.desired_timestamped_wpts[:, k]
            cost += self.lagrange(xk, ak, nk, wk)
        cost += self.mayer(x[:, -1], self.desired_timestamped_wpts[:, -1])
        return cost
    
    
    def __get__(
            self,
            timestamped_wpts: List[Tuple[float, float]],
            eta: np.ndarray,
            nu: np.ndarray,
            alpha: np.ndarray,
            n: np.ndarray,
            current: Current,
            wind: Wind,
            *args,
            **kwargs
        ) -> List[np.ndarray]:
        
        sol = self.solve(
            pars={
                "x_0": np.concatenate([eta, nu, alpha, n]).reshape(14, 1),
                "wpts": np.array(timestamped_wpts).reshape(2, self.horizon + 1)
            })
        
        print(sol.f)
        return [np.array([a, n]) for a, n in zip(sol.vals["alphas"][:, 0].full().reshape(self.nu // 2), sol.vals["ns"][:, 0].full().reshape(self.nu // 2))]

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

    # Something is not working as expected, control commands are always zero. 
    # Maybe because of the name of control commands in discrete_dynamics ? Idk how cs.Function are working under the hood

    dt = 0.1
    H = 10
    mpc = DiffMPCTrajectoryTracking(get_discrete_3dof_dynamics_as_fn(dt), H)
    print(mpc.__get__(
        [(0, 0) for _ in range(H+1)],
        np.array([-1, 0, 0]),
        np.array([0, 0, 0]),
        np.array(4*[0]),
        np.array(4*[0]),
        None, 
        None
    ))

    # print(mpc.nlp.f)