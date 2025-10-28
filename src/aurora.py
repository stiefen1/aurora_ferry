from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.physics import RHO, GRAVITY
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec, DEG2RAD
from math import pi, sqrt
from dataclasses import dataclass, field
import numpy as np, casadi as ca
from typing import Tuple, List


INF = float('inf')

@dataclass
class SingleAzimuthThrusterParameters:
    ## Propellers       
    T_n: float = 0.3                                        # Propeller time constant (s)
    T_a: float = 3.0                                        # Azimuth angle time constant (s) -> Chosen by me
    k_pos: float = 200                                      # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: float = 200                                      # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: float = 38_798                                   # Max positive force, one propeller
    f_min: float = 0 # -25                                  # Max negative force, one propeller
    n_max: float = sqrt(f_max/k_pos)                        # Max (positive) propeller speed
    n_min: float = -sqrt(-f_min/k_neg)
    a_max: float = pi                                       # Max (positive) propeller speed
    a_min: float = -pi                                      # Min (negative) propeller speed
    max_radians_per_step: float = pi/6
    max_newton_per_step: float = 10

@dataclass
class AuroraFerryActuatorsParameters:
    ## Propellers       
    # T_n: float = 0.3                                            # Propeller time constant (s)
    # T_a: float = 3.0 
    thrusters: List = field(init=False)                         # Azimuth angle time constant (s) -> Chosen by me
    k_pos: np.ndarray = field(init=False)                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: np.ndarray = field(init=False)                       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: np.ndarray = field(init=False)                       # Max positive force, one propeller
    f_min: np.ndarray = field(init=False)                       # Max negative force, one propeller
    n_max: np.ndarray = field(init=False)                       # Max (positive) propeller speed
    n_min: np.ndarray = field(init=False)
    lba: np.ndarray = field(init=False)                       # Max (positive) propeller speed
    uba: np.ndarray = field(init=False)                         # Min (negative) propeller speed
    xy: np.ndarray = field(init=False) # azimuth, azimuth, thruster from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.56)
    max_radians_per_step: np.ndarray = field(init=False)
    max_newton_per_step: np.ndarray = field(init=False)
    time_constant: np.ndarray = field(init=False)

    def __post_init__(self):
        self.thrusters = [SingleAzimuthThrusterParameters(), SingleAzimuthThrusterParameters(), SingleAzimuthThrusterParameters(), SingleAzimuthThrusterParameters]  
        self.k_pos: np.ndarray = np.array([thruster.k_pos for thruster in self.thrusters])    # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
        self.k_neg: np.ndarray = np.array([thruster.k_neg for thruster in self.thrusters])    # Negative Bollard, one propeller
        self.f_max: np.ndarray = np.array([thruster.f_max for thruster in self.thrusters])    # Max positive force, one propeller
        self.f_min: np.ndarray = np.array([thruster.f_min for thruster in self.thrusters]) 
        self.n_max = np.array([thruster.n_max for thruster in self.thrusters])
        self.n_min = np.array([thruster.n_min for thruster in self.thrusters])
        self.lba = np.array([thruster.a_min for thruster in self.thrusters])                       # Azimuth angles constraints
        self.uba = np.array([thruster.a_max for thruster in self.thrusters])
        self.xy = np.array([[-35, -9.4], [-35, 9.4], [35, 9.4], [35, -9.4]])    
        self.max_radians_per_step = np.array([np.pi/6, np.pi/6, np.pi/36])
        self.max_newton_per_step = np.array([10.0, 10.0, 4.0])
        self.time_constant = np.array([thruster.T_n for thruster in self.thrusters] + [thruster.T_a for thruster in self.thrusters])

        self.Ti = lambda alpha, lx, ly : np.array([
            ca.cos(alpha),
            ca.sin(alpha),
            lx*ca.sin(alpha) - ly * ca.cos(alpha)
        ])

        ####### WARNING : IF YOU CHANGE T YOU HAVE TO DO IT AS WELL IN RL ENVIRONMENTS, IT IS NOT LINKED ######
        self.T = lambda a1, a2, a3, a4 : ca.vertcat(
            ca.horzcat(ca.cos(a1), ca.sin(a1), self.xy[0, 0]*ca.sin(a1) - self.xy[0, 1] * ca.cos(a1)),
            ca.horzcat(ca.cos(a2), ca.sin(a2), self.xy[1, 0]*ca.sin(a2) - self.xy[1, 1] * ca.cos(a2)),
            ca.horzcat(ca.cos(a3), ca.sin(a3), self.xy[2, 0]*ca.sin(a3) - self.xy[2, 1] * ca.cos(a3)),
            ca.horzcat(ca.cos(a4), ca.sin(a4), self.xy[3, 0]*ca.sin(a4) - self.xy[3, 1] * ca.cos(a4))
        )

@dataclass
class AuroraFerryParameters:
    # Many of the following parameters are based on:
    ## (1) https://www.faergelejet.dk/faerge.php?id=164&n=1         -> seems to be a better source for tonnage info
    ## (2) https://www.ferry-site.dk/ferry.php?id=9007128&lang=en

    ## Mass & Payload
    GT: float = 10918                                   # Gross tonnage (1)
    NT: float = 3275                                    # Net tonnage (1)
    DWT: float = 2250                                   # Dead-Weight Tonnage (1)
    m: float = DWT * 1e3                                # Mass (kg)
    # n_passengers: int = 500                             # Max number of passenger is 1'250
    # n_cars: int = 150                                   # Max number of cars is 250
    # mp: float = n_passengers * 78 + n_cars * 1200       # Payload mass (kg)
    mp: float = 0
    

    Nx: int = 3
    loa: float = 111.2                                  # Length Over All (m) assumed equal to LPP
    beam: float = 28.2                                  # Beam (m)      
    initial_draft:float = 5.5                           # Initial draft
    volume:float = (m+mp) / RHO                         # m^3 volume 
    volume_iz:float = 3 * 1e6                           # m^5 volum moment of inertia -> Computed assuming Viz ~ V * (b^2 + loa^2) using similarity laws with Revolt vessel
                                                        

    R44: float = 0.36 * beam                            # radii of gyration (m)
    R55: float = 0.26 * loa
    R66: float = 0.26 * loa

    ## Time constants
    T_yaw: float = 3.0                                  # Time constant in yaw (s)

    rg: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))  
    rp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))          # Location of payload (m)

    ## State constraints
    lbx: np.ndarray = field(default_factory=lambda: np.array([-INF, -INF, -pi, -knot_to_m_per_sec(14.9), -3, -pi/6]))
    ubx: np.ndarray = field(default_factory=lambda: np.array([INF, INF, pi, knot_to_m_per_sec(14.9), 3, pi/6]))

    def __post_init__(self):
        self.m_tot = self.m + self.mp
        self.rg = (self.m * self.rg + self.mp * self.rp) / (self.m + self.mp) # corrected center of gravity with payload

        # Basic calculations
        self.Xudot, self.Yvdot, self.Yrdot, self.Nvdot = (-self.volume*RHO*np.array([0.0253, 0.1802, 0.0085 * self.loa, 0.0099 * self.loa**2])).tolist()
        self.Xu, self.Yv, self.Yr, self.Nv, self.Nr = (-np.array([
            0.102 * RHO * self.volume * GRAVITY / self.loa,
            1.212 * GRAVITY / self.loa,
            0.056 * RHO * self.volume * np.sqrt(GRAVITY * self.loa),
            0.056 * RHO * self.volume * np.sqrt(GRAVITY * self.loa),
            0.0601 * RHO * self.volume * np.sqrt(GRAVITY * self.loa)
            ])).tolist()
        self.Iz = self.m_tot / self.volume * self.volume_iz # = approx 3 * 1e9 
                                                            # this value looks huge, but as a comparison, inertia of RV Gunnerus is 41'237'080 with LOA=31.25 and m=574 tons
 
        ## Inertia Matrix
        MRB = np.array([
            [self.m_tot, 0, 0],
            [0, self.m_tot, self.m_tot*self.rg[0]],
            [0, self.m_tot*self.rg[0], self.Iz]
        ])
        MA = np.array([
            [-self.Xudot, 0, 0],
            [0, -self.Yvdot, -self.Yrdot],
            [0, -self.Yrdot, -self.Nvdot]
        ])
        self.Minv = np.linalg.inv(MA + MRB)

        ## Coriolis-Centripetal Matrix
        self.CRB = lambda nu : np.array([
            [0, 0, -self.m_tot * (self.rg[0]*nu[2] + nu[1])],
            [0, 0, self.m_tot * nu[0]],
            [self.m_tot * (self.rg[0]*nu[2] + nu[1]), -self.m_tot * nu[0], 0]
        ])
        self.CA = lambda nu_r : np.array([
            [0, 0, self.Yvdot * nu_r[1] + self.Yrdot * nu_r[2]],
            [0, 0, -self.Xudot * nu_r[0]],
            [-self.Yvdot * nu_r[1]-self.Yrdot * nu_r[2], self.Xudot * nu_r[0], 0]
        ])
        ## Damping Matrix
        self.D = np.array([
            [-self.Xu, 0, 0],
            [0, -self.Yv, -self.Yr],
            [0, -self.Nv, -self.Nr]
        ])

class AuroraFerry(IVessel):
    def __init__(
            self,
            dt: float,
            eta0: Tuple,
            nu0: Tuple,
            *args, 
            **kwargs
    ):
        super().__init__(
            params=AuroraFerryParameters(),
            dt=dt,
            eta=eta0,
            nu=nu0,
            *args,
            **kwargs
        )

    def __dynamics__(self, tau_actuators:np.ndarray, current:Current, wind:Wind, *args, **kwargs) -> np.ndarray:
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the Otter USV equations of motion using Euler's method.
        """
        nu = np.array(self.nu.uvr)
        tau_actuators_3 = np.array([tau_actuators[0], tau_actuators[1], tau_actuators[5]])

        CRB = self.params.CRB(nu)
        CA = self.params.CA(nu)
        C = CRB + CA

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = np.matmul(self.params.D, nu)

        # State derivatives (with dimension)
        sum_tau = (
            tau_actuators_3
            - tau_damp
            - np.matmul(C, nu)
        )

        # USV dynamics
        nu_dot = np.matmul(self.params.Minv, sum_tau)

        return np.array([nu_dot[0], nu_dot[1], 0, 0, 0, nu_dot[2]])


def print_f_max_from_u_max() -> None:
    params = AuroraFerryParameters()
    nu = np.array([knot_to_m_per_sec(14.9), 0, 0])

    CRB = params.CRB(nu)
    CA = params.CA(nu)
    C = CRB + CA
    D = params.D
    nu = nu.reshape(3, 1)

    print(f"Total force:\n{D @ nu - C @ nu}")
    print(f"single actuator force:\n{0.25*(D @ nu - C @ nu)}")

if __name__ == "__main__":
    print_f_max_from_u_max()