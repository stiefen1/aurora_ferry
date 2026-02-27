from stable_baselines3.nmpc import BaseNMPC
from python_vehicle_simulator.lib.control import Control
from src.discrete_dynamics import get_discrete_3dof_dynamics_as_fn
from src.aurora import AuroraFerryActuatorsParameters, AuroraFerryParameters
import casadi as cs, numpy as np
from typing import Optional, Dict, Tuple

NX = 14
NU = 8

class ParametricTrajectoryTracker(BaseNMPC):
    def __init__(
            self,
            horizon: int,
            dt: float,
            # Q: Optional[np.ndarray] = None,             # state cost (stage) -> used as learnable param for testing
            QN: Optional[np.ndarray] = None,            # state cost (terminal)
            Qpsi: Optional[float] = None,               # heading cost (stage & terminal)
            Ra: Optional[np.ndarray] = None,            # alpha cost (stage)
            Rn: Optional[np.ndarray] = None,            # propeller cost (stage)
            gamma: float = 1.0,
            learnable_params: Optional[Dict[str, Tuple[int, int]]] = {"Q": (3, 1)},
            **kwargs
    ):
        self.actuators_params = AuroraFerryActuatorsParameters()
        self.aurora_params = AuroraFerryParameters()
        # self.Q = Q if Q is not None else np.diag([1e3, 10, 10])
        self.QN = QN if QN is not None else np.diag([1e3, 10, 10]) # self.Q.copy()
        self.Qpsi = Qpsi if Qpsi is not None else 1e1  # Reduced from 1e6
        self.Ra = Ra if Ra is not None else np.eye(NU // 2) * 1e-2
        self.Rn = Rn if Rn is not None else np.eye(NU // 2) * 1e-7  # Slightly increased from 1e-7
        self.Theta_v = 1e3*np.diag([1, 1, 1e4, 10, 10, 10])

        # Hyperparameters
        self.huber_penalty_slope = 200 # 10 # delta
        self.huber_penalty_weight = 100 # q_x,y
        self.heading_penalty_weight = 100 # 50 # q_psi
        self.singular_value_penalty = 1e-3 # epsilon -> for nonsigular thruster configuration
        self.singular_value_weight = 1e-8 # 1e-6 # 1e-5 # 8e-4 # 1e-9 #  5e-4 # 1e-5 # rho -> for nonsigular thruster configuration
        self.gamma = gamma # Discount factor -> self.gamma^k

        # Control.__init__(self)    
        BaseNMPC.__init__(
            self,
            get_discrete_3dof_dynamics_as_fn(dt),
            horizon,
            NX,
            NU,
            np.concatenate([self.actuators_params.a_min, self.actuators_params.n_min]),
            np.concatenate([self.actuators_params.a_max, self.actuators_params.n_max]),
            learnable_params=learnable_params,
            params={
                "wpts": (3, horizon + 1),
                "nu_des": (3, 1)
            },
            **kwargs
        )

    def lagrange(self, xk: cs.SX, uk: cs.SX, k: int, *args, **kwargs) -> cs.SX:
        """
        learnable/non-learnable parameters can be used here by accessing its name:

        self.learnable_params["Q"]
        """
        # Extract parameters
        wk = self.params["wpts"][:, k]
        nu_des = self.params["nu_des"]
        ak, nk = uk[0:4], uk[4:8]

        # Extract learnable parameters
        Q = cs.diag(self.learnable_params["Q"])

        # Huber cost to penalize deviation from path
        pos_cost = self.huber_penalty_slope**2 * (cs.sqrt(1 + ((xk[0]-wk[0])**2 + (xk[1]-wk[1])**2) / self.huber_penalty_slope **2) - 1 ) 
        
        # Heading tracking cost
        heading_cost = 0.5 * (1 - cs.cos(xk[2] - wk[2]))

        # Singular configurations
        B = self.actuators_params.B(ak[0], ak[1], ak[2], ak[3])
        singular_cost = 1 / (self.singular_value_penalty + cs.det(B.T @ B))

        # Speed cost
        speed_cost = (xk[3:6]-nu_des).T @ Q @ (xk[3:6]-nu_des)

        # Control costs
        control_cost = nk.T @ self.Rn @ nk + ak.T @ self.Ra @ ak
        
        return self.huber_penalty_weight * pos_cost + \
             self.heading_penalty_weight * heading_cost + \
             self.singular_value_penalty * singular_cost + \
             speed_cost + control_cost

    def mayer(self, xN: cs.SX, *args, **kwargs) -> cs.SX:
        """
        learnable/non-learnable parameters can be used here by accessing its name:

        self.params["wpts"]
        """
        xd = cs.vertcat(self.params["wpts"][:, -1], self.params["nu_des"])
        return (xN[0:6]-xd).T @ self.Theta_v @ (xN[0:6]-xd)

    # def __get__(self):
    #     pass
