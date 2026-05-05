from python_vehicle_simulator.lib.diagnosis import IDiagnosis
from python_vehicle_simulator.lib.dynamics import IDynamics
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.navigation import INavigation
from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.physics import RHO, GRAVITY
from python_vehicle_simulator.utils.math_fn import R_casadi, Rzyx
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from python_vehicle_simulator.lib.thruster import THRUSTER_LENGTH_DEFAULT, THRUSTER_WIDTH_DEFAULT, THRUSTER_GEOMETRY, ROTATION_MATRIX
from math import pi, sqrt
from dataclasses import dataclass, field
import numpy as np, numpy.typing as npt, casadi as cs
from typing import Tuple, List, Optional


INF = float('inf')

@dataclass
class SingleAzimuthThrusterParameters:
    ## Propellers       
    T_n: float = 3 # 3                                      # Propeller time constant (s)
    T_a: float = 20 # 20.0                                  # Azimuth angle time constant (s) -> Chosen by me
    k_pos: float = 20 # 200                                  # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: float = 20                                        # Negative Bollard, one propeller (Division by two beuse there are two propellers, values are obtained with a Bollard pull)
    f_max: float = 570_000 # 600_000 # 40_000 # 38_798                          # Max positive force, one propeller
    f_min: float = 0                                        # Max negative force, one propeller
    speed_max: float = sqrt(f_max/k_pos)                        # Max (positive) propeller speed
    speed_min: float = 0                                        # We don't allow negative thruster speed
    alpha_max: float = pi                                       # Max (positive) propeller speed
    alpha_min: float = -pi                                      # Min (negative) propeller speed
    max_radians_per_step: float = pi/6
    max_newton_per_step: float = 10
    geometry = THRUSTER_GEOMETRY(5, 2)


@dataclass
class AuroraFerryActuatorsParameters:
    ## Propellers       
    # T_n: float = 0.3                                            # Propeller time constant (s)
    # T_a: float = 3.0 
    thrusters: List = field(init=False)                         # Azimuth angle time constant (s) -> Chosen by me
    k_pos: np.ndarray = field(init=False)                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: np.ndarray = field(init=False)                       # Negative Bollard, one propeller (Division by two beuse there are two propellers, values are obtained with a Bollard pull)
    f_max: np.ndarray = field(init=False)                       # Max positive force, one propeller
    f_min: np.ndarray = field(init=False)                       # Max negative force, one propeller
    speed_max: np.ndarray = field(init=False)                       # Max (positive) propeller speed
    speed_min: np.ndarray = field(init=False)
    alpha_min: np.ndarray = field(init=False)
    alpha_max: np.ndarray = field(init=False)
    xy: np.ndarray = field(init=False) # azimuth, azimuth, thruster from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.56)
    max_radians_per_step: np.ndarray = field(init=False)
    max_newton_per_step: np.ndarray = field(init=False)
    time_constant: np.ndarray = field(init=False)
    geometries: List = field(init=False)

    def __post_init__(self):
        self.thrusters = [SingleAzimuthThrusterParameters(), SingleAzimuthThrusterParameters(), SingleAzimuthThrusterParameters(), SingleAzimuthThrusterParameters()]  
        self.k_pos: np.ndarray = np.array([thruster.k_pos for thruster in self.thrusters])      # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
        self.k_neg: np.ndarray = np.array([thruster.k_neg for thruster in self.thrusters])      # Negative Bollard, one propeller
        self.f_max: np.ndarray = np.array([thruster.f_max for thruster in self.thrusters])      # Max positive force, one propeller
        self.f_min: np.ndarray = np.array([thruster.f_min for thruster in self.thrusters]) 
        self.speed_min = np.array([thruster.speed_min for thruster in self.thrusters])
        self.speed_max = np.array([thruster.speed_max for thruster in self.thrusters])          # Thruster speed constraints
        self.alpha_min = np.array([thruster.alpha_min for thruster in self.thrusters])          # Azimuth angles constraints
        self.alpha_max = np.array([thruster.alpha_max for thruster in self.thrusters])

        self.xy = np.array([[-45, -9.4], [-45, 9.4], [25, 9.4], [25, -9.4]]) # np.array([[-35, -9.4], [-35, 9.4], [35, 9.4], [35, -9.4]])    
        self.time_constant = np.array([thruster.T_n for thruster in self.thrusters] + [thruster.T_a for thruster in self.thrusters])
        self.T_n = np.array([thruster.T_n for thruster in self.thrusters])
        self.T_a = np.array([thruster.T_a for thruster in self.thrusters])

        self.Ai = lambda alpha, lx, ly : np.array([
            cs.cos(alpha),
            cs.sin(alpha),
            lx*cs.sin(alpha) - ly * cs.cos(alpha)
        ])

        ####### WARNING : IF YOU CHANGE T YOU HAVE TO DO IT AS WELL IN RL ENVIRONMENTS, IT IS NOT LINKED ######
        self.Alpha = lambda a1, a2, a3, a4 : cs.vertcat(
            cs.horzcat(cs.cos(a1), cs.sin(a1), self.xy[0, 0]*cs.sin(a1) - self.xy[0, 1] * cs.cos(a1)),
            cs.horzcat(cs.cos(a2), cs.sin(a2), self.xy[1, 0]*cs.sin(a2) - self.xy[1, 1] * cs.cos(a2)),
            cs.horzcat(cs.cos(a3), cs.sin(a3), self.xy[2, 0]*cs.sin(a3) - self.xy[2, 1] * cs.cos(a3)),
            cs.horzcat(cs.cos(a4), cs.sin(a4), self.xy[3, 0]*cs.sin(a4) - self.xy[3, 1] * cs.cos(a4))
        ).T

        self.geometries = [thruster.geometry for thruster in self.thrusters]

@dataclass
class AuroraFerryParameters:
    # Many of the following parameters are based on:
    ## (1) https://www.faergelejet.dk/faerge.php?id=164&n=1         -> seems to be a better source for tonnage info
    ## (2) https://www.ferry-site.dk/ferry.php?id=9007128&lang=en

    ## Mass & Payload
    # GT: float = 10918                                   # Gross tonnage (1)
    # NT: float = 3275                                    # Net tonnage (1)
    # DWT: float = 2250                                   # Dead-Weight Tonnage (1)
    m: float = 2250 * 1e3                               # Mass (kg) -> m = 2'250'000
    n_passengers: int = 0                               # Max number of passenger is 1'250 -> m = 100'000
    n_cars: int = 0                                     # Max number of cars is 240 -> m = 280'000
    passenger_mean_mass: float = 78.0
    passenger_dmass: float = 0.0                        # Additional average mass due to uncertainties
    car_mean_mass: float = 1200
    car_dmass: float = 0.0                              # Additional average mass due to uncertainties
    

    Nx: int = 3
    loa: float = 111.2                                  # Length Over All (m) assumed equal to LPP
    beam: float = 28.2                                  # Beam (m)      
    initial_draft:float = 5.5                           # Initial draft
    mean_height_above_water: float = 15.0                    
    volume_iz:float = 3 * 1e6                           # m^5 volum moment of inertia -> Computed assuming Viz ~ V * (b^2 + loa^2) using similarity laws with Revolt vessel

    # Wind coefficients                                     
    cx: float = 0.5
    cy: float = 0.7
    cn: float = 0.08

    R44: float = 0.36 * beam                            # radii of gyration (m)
    R55: float = 0.26 * loa
    R66: float = 0.26 * loa

    surge_speed_max: float = knot_to_m_per_sec(14.9)

    rg: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))  
    rp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))          # Location of payload (m)

    ## State constraints
    lbx: np.ndarray = field(default_factory=lambda: np.array([-INF, -INF, -pi, -knot_to_m_per_sec(14.9), -3, -pi/6]))
    ubx: np.ndarray = field(default_factory=lambda: np.array([INF, INF, pi, knot_to_m_per_sec(14.9), 3, pi/6]))

    def __post_init__(self):
        self.mp: float = self.n_passengers * (self.passenger_mean_mass + self.passenger_dmass) + self.n_cars * (self.car_mean_mass + self.car_dmass)       # Payload mass (kg) -> passengers and cars lead to ~+-10% inertia variation
        self.mp_estimated: float = self.n_passengers * self.passenger_mean_mass + self.n_cars * self.car_mean_mass
        self.volume:float = (self.m+self.mp) / RHO                         # m^3 volume 
        self.m_tot = self.m + self.mp
        self.m_tot_estimated = self.m + self.mp_estimated
        self.rg = (self.m * self.rg + self.mp * self.rp) / (self.m + self.mp) # corrected center of gravity with payload

        self.proj_area_f: float = self.mean_height_above_water * self.beam
        self.proj_area_l: float = self.mean_height_above_water * self.loa

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
        # self.D = np.array([
        #     [-self.Xu, 0, 0],
        #     [0, -self.Yv, -self.Yr],
        #     [0, -self.Nv, -self.Nr]
        # ])
        self.D = np.array([
            [2e4 * 15, 0, 0],
            [0, 5e6, 1e6],
            [0, 1e6, 1e9]
        ])

class Aurora3Dynamics(IDynamics):
    def __init__(
            self,
            dt: float,
            *args,
            vessel_params: Optional[AuroraFerryParameters] = None,
            actuator_params: Optional[AuroraFerryActuatorsParameters] = None,
            **kwargs
    ):
        self.vessel_params = vessel_params or AuroraFerryParameters()
        self.actuator_params = actuator_params or AuroraFerryActuatorsParameters()
        nx, nu, np, nd = 20, 8, 8, 3
        super().__init__(nx, nu, np, nd, dt, *args, **kwargs)

    def continuous_time_dynamics(self, x: cs.SX, u: cs.SX, theta: cs.SX, disturbance: cs.SX | None, *args, **kwargs) -> cs.SX:
        x_dot = cs.SX.zeros(self.nx, 1) # type: ignore

        eta_3dofs, nu_3dofs = cs.vcat([x[0], x[1], x[5]]), cs.vcat([x[6], x[7], x[11]])
        azimuth, thruster_speed = x[12:16], x[16:20]
        azimuth_setpoint = cs.fmax(cs.fmin(u[0:4], self.actuator_params.alpha_max), self.actuator_params.alpha_min)
        speed_setpoint = cs.fmax(cs.fmin(u[4:8], self.actuator_params.speed_max), self.actuator_params.speed_min)
        azimuth_stucked, propeller_effectiveness = theta[0:4], theta[4:8]

        # Generalized force generated by actuators
        thrust = propeller_effectiveness * self.actuator_params.k_pos * thruster_speed * thruster_speed
        tau_actuators = self.actuator_params.Alpha(azimuth[0], azimuth[1], azimuth[2], azimuth[3]) @ thrust

        # Hull dynamics (body frame)
        Minv, C, D = self.vessel_params.Minv, self.vessel_params.CA(nu_3dofs) + self.vessel_params.CRB(nu_3dofs), self.vessel_params.D
        nu_dot_3dofs = Minv @ (tau_actuators + disturbance - C @ nu_3dofs - D @ nu_3dofs)
        x_dot[6] = nu_dot_3dofs[0]
        x_dot[7] = nu_dot_3dofs[1]
        x_dot[11] = nu_dot_3dofs[2]

        # Ship's kinematics (body -> NED)
        eta_dot_3dofs = cs.mtimes(R_casadi(eta_3dofs[2]), nu_3dofs)
        x_dot[0] = eta_dot_3dofs[0]
        x_dot[1] = eta_dot_3dofs[1]
        x_dot[5] = eta_dot_3dofs[2]

        # Actuator's dynamics: Low-pass
        x_dot[12:16] = azimuth_stucked * (azimuth_setpoint - azimuth) / self.actuator_params.T_a
        x_dot[16:20] = (speed_setpoint - thruster_speed) / self.actuator_params.T_n

        return x_dot

class AuroraFerry(IVessel):
    def __init__(
            self,
            dt: float,
            eta: Tuple = (0, 0, 0),
            nu: Tuple = (0, 0, 0),
            thruster_speeds: Tuple = (0, 0, 0, 0),
            azimuth_angles: Tuple = (0, 0, 0, 0),
            guidance: Optional[IGuidance] = None,
            navigation: Optional[INavigation] = None,
            control: Optional[IControl] = None,
            diagnosis: Optional[IDiagnosis] = None,
            mmsi: Optional[str] = None,
            verbose_level: int = 0,
            mass: float = 2_250_000,
            n_passengers: int = 0,
            n_cars: int = 0,
            passenger_dmass: float = 0.0,
            car_dmass: float = 0.0,
    ):
        """
        Aurora autonomous ferry with 3DOF dynamics.
        
        dt:             Sampling time
        eta:            Initial position [x, y, psi]        (3,)
        nu:             Initial velocity [u, v, r]          (3,)
        guidance:       Guidance system (optional)
        navigation:     Navigation system (optional)
        control:        Control system (optional)
        diagnosis:      Diagnosis system (optional)
        mmsi:           Maritime Mobile Service Identity
        verbose_level:  Verbosity level for logging
        """

        self.vessel_params = AuroraFerryParameters(m=mass, n_passengers=n_passengers, n_cars=n_cars, passenger_dmass=passenger_dmass, car_dmass=car_dmass)
        self.actuator_params = AuroraFerryActuatorsParameters()

        super().__init__(
            self.vessel_params.loa,
            self.vessel_params.beam,
            Aurora3Dynamics(dt, vessel_params=self.vessel_params, actuator_params=self.actuator_params),
            states=(eta[0], eta[1], 0, 0, 0, eta[2], nu[0], nu[1], 0, 0, 0, nu[2], *azimuth_angles, *thruster_speeds),
            guidance=guidance,
            navigation=navigation,
            control=control,
            diagnosis=diagnosis,
            name='Aurora3',
            mmsi=mmsi,
            verbose_level=verbose_level,
        )

    def __dynamics__(self, control_commands:npt.NDArray, current:Current, wind:Wind, *args, theta: Optional[npt.NDArray] = None, **kwargs) -> np.ndarray:
        """
        Vessel dynamics step with environmental disturbances.
        
        control_commands:   Thruster commands [azimuth, speeds]  (6,)
        current:            Current disturbance model
        wind:               Wind disturbance model
        theta:              Fault parameters (optional)          (6,)
        
        Returns:
            List: [next eta, next nu, next alpha, next thruster speed]

        Azimuth thrusters are organized in the following order:
        - Stern port        (xy=[-1.65, -0.15])
        - Stern starboard   (xy=[-1.65, 0.15])
        - Bow               (xy=[1.15, 0.0])
        """

        # Wind perturbations
        uw = wind.u(self.eta.yaw)
        vw = wind.v(self.eta.yaw)

        u_rw = uw - self.nu.u
        v_rw = vw - self.nu.v

        gamma_w = wind.gamma_w(self.eta.yaw)
        wind_rw2 = u_rw**2 + v_rw**2
        c_x = -self.vessel_params.cx * np.cos(gamma_w)
        c_y = self.vessel_params.cy * np.sin(gamma_w)
        c_n = self.vessel_params.cn * np.sin(2 * gamma_w)

        tau_coeff = 0.5 * wind.get_air_density() * wind_rw2
        tau_w = np.array([
            tau_coeff * c_x * self.vessel_params.proj_area_f,
            tau_coeff * c_y * self.vessel_params.proj_area_l,
            tau_coeff * c_n * self.vessel_params.proj_area_l * self.vessel_params.loa
        ]) 

        # Current perturbations
        v_c = np.array([current.u(self.eta.yaw), current.v(self.eta.yaw), 0]) # current speed in ship frame
        tau_c_coriolis = self.vessel_params.CA(self.nu.uvr) @ self.nu.uvr - self.vessel_params.CA(self.nu.uvr - v_c) @ (self.nu.uvr - v_c) # cancel CA(nu) @ nu and add CA(nu_r) @ nu_r
        tau_c_damping = self.vessel_params.D @ v_c
        tau_c = tau_c_coriolis + tau_c_damping

        disturbance = tau_w + tau_c # Define it as a function of current, wind
        theta = theta if theta is not None else np.array(self.dynamics.nt * [1.0])
        x = self.dynamics.fd(self.states, control_commands, theta, disturbance).flatten()
        return x
    
    def __plot__(self, ax, *args, verbose:int=0, **kwargs):
        """
        x = East
        y = North
        z = -depth
        """
        ax.scatter(self.eta[1], self.eta[0], *args, **kwargs)
        ax.plot(*self.geometry_for_2D_plot, *args, **kwargs)
        for i in range(4):
            envelope = (ROTATION_MATRIX(self.states[12 + i]) @ self.actuator_params.geometries[i].T) + self.actuator_params.xy[i].reshape(-1, 1)
            envelope_in_ned_frame = Rzyx(*self.eta.to_numpy()[3:6].tolist())[0:2, 0:2] @ envelope + self.eta.to_numpy()[0:2, None]
            ax.plot(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :], *args, **kwargs)
        return ax

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
    aurora = AuroraFerry(0.1)
    print("M: ", np.linalg.inv(aurora.vessel_params.Minv))
    print("C: ", aurora.vessel_params.CA(np.array([0, 0, 0])) + aurora.vessel_params.CRB(np.array([0, 0, 0])))
    print("D: ", aurora.vessel_params.D)

    print_f_max_from_u_max()