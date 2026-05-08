from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter
from typing import Optional
import numpy as np
from python_vehicle_simulator.utils.math_fn import ssa


class TargetTrackerSequentialCTRV(IExtendedKalmanFilter):
    """
    Extended Kalman filter to estimate target ships position and speed using AIS & camera data.
    Sensor fusion is done in a sequential way to handle different data acquisition frequencies.

    States:

        north   [m]
        east    [m]
        sog     [m/s]
        cog     [rad]
        omega   [rad/s]

    CTRV-type Model (no control inputs):

        north += f_ctrv(sog, cog, omega, dt)
        east += f_ctrv(sog, cog, omega, dt)
        sog += 0
        cog += omega * dt
        omega += 0

    If omega is near zero, the model falls back to a CV-like straight-line form.
    """

    def __init__(
            self,
            Q,  # Process covariance
            R_ais,  # AIS Measurement Covariance
            R_camera,  # Camera Measurement Covariance
            x0,  # Initial states
            P0,  # Initial error covariance
            dt,  # Sampling time, needed when building the system's model.
            sog_softplus_beta: float = 10.0,
            omega_min: float = -0.1,
            omega_max: float = 0.1,
            *args,
            **kwargs
        ):
        super().__init__(Q, R_ais, x0, P0, dt, *args, **kwargs)
        self.R_ais = R_ais
        self.R_camera = R_camera
        self.sog_softplus_beta = max(float(sog_softplus_beta), 1e-6)
        if omega_max <= omega_min:
            raise ValueError("omega_max must be strictly greater than omega_min")
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

    def _omega_effective(self, omega: float) -> float:
        """Smoothly bound omega to [omega_min, omega_max] with tanh."""
        omega_center = 0.5 * (self.omega_max + self.omega_min)
        omega_half_range = 0.5 * (self.omega_max - self.omega_min)
        return float(omega_center + omega_half_range * np.tanh((omega - omega_center) / omega_half_range))

    def _omega_effective_derivative(self, omega: float) -> float:
        """Derivative d(omega_eff)/d(omega)."""
        omega_center = 0.5 * (self.omega_max + self.omega_min)
        omega_half_range = 0.5 * (self.omega_max - self.omega_min)
        xi = (omega - omega_center) / omega_half_range
        return float(1.0 - np.tanh(xi) ** 2)

    def _softplus(self, sog: float) -> float:
        """Smooth nonlinearity that behaves like ReLU but stays differentiable."""
        z = self.sog_softplus_beta * sog
        return float((np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))) / self.sog_softplus_beta)

    def _softplus_derivative(self, sog: float) -> float:
        """Derivative of the smooth speed nonlinearity wrt sog."""
        z = self.sog_softplus_beta * sog
        if z >= 0.0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = np.exp(z)
        return float(ez / (1.0 + ez))

    def f(self, x: np.ndarray, u: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v

        x = [north, east, sog, cog, omega]
        """
        sog_eff = self._softplus(float(x[2]))
        cog = float(x[3])
        omega = float(x[4])
        omega_eff = self._omega_effective(omega)
        omega_eff_safe = omega_eff + np.copysign(1e-6, omega_eff)
        cog_next = cog + omega_eff * self.dt
        north_next = x[0] + (sog_eff / omega_eff_safe) * (np.sin(cog_next) - np.sin(cog))
        east_next = x[1] + (sog_eff / omega_eff_safe) * (-np.cos(cog_next) + np.cos(cog))

        return np.array([
            north_next,
            east_next,
            sog_eff,
            ssa(cog_next),
            omega_eff
        ])

    def dfdx(self, x: np.ndarray, u: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        sog_eff = self._softplus(float(x[2]))
        dsog_dsog = self._softplus_derivative(float(x[2]))
        cog = float(x[3])
        omega = float(x[4])
        omega_eff = self._omega_effective(omega)
        domegaeff_domega = self._omega_effective_derivative(omega)
        omega_eff_safe = omega_eff + np.copysign(1e-6, omega_eff)
        cog_next = cog + omega_eff * self.dt
        A = (np.sin(cog_next) - np.sin(cog)) / omega_eff_safe
        B = (-np.cos(cog_next) + np.cos(cog)) / omega_eff_safe
        dA_domegaeff = (omega_eff_safe * self.dt * np.cos(cog_next) - (np.sin(cog_next) - np.sin(cog))) / (omega_eff_safe**2)
        dB_domegaeff = (omega_eff_safe * self.dt * np.sin(cog_next) - (-np.cos(cog_next) + np.cos(cog))) / (omega_eff_safe**2)

        d_north_d_sog = A * dsog_dsog
        d_north_d_cog = (sog_eff / omega_eff_safe) * (np.cos(cog_next) - np.cos(cog))
        d_north_d_omega = sog_eff * dA_domegaeff * domegaeff_domega
        d_east_d_sog = B * dsog_dsog
        d_east_d_cog = (sog_eff / omega_eff_safe) * (np.sin(cog_next) - np.sin(cog))
        d_east_d_omega = sog_eff * dB_domegaeff * domegaeff_domega

        return np.array([
            [1, 0, d_north_d_sog, d_north_d_cog, d_north_d_omega],
            [0, 1, d_east_d_sog, d_east_d_cog, d_east_d_omega],
            [0, 0, dsog_dsog, 0, 0],
            [0, 0, 0, 1, self.dt * domegaeff_domega],
            [0, 0, 0, 0, domegaeff_domega]
        ])

    def h(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise RuntimeError(
            "dhdx must be avoided for TargetTrackerSequentialCTRV since it uses a sequential approach. "
            "Instead, call either dhdx_ais and dhdx_camera sequentially."
        )

    def h_ais(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Measurement model for AIS

        y = h(x) = [north, east, sog, cog]
        """
        z_ais = self.dhdx_ais(x) @ x
        return z_ais

    def h_camera(self, x: np.ndarray, *args, os_neyaw: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Measurement model for camera

        z = h(x) = [alpha, d] (relative angle, distance)
        """
        assert os_neyaw is not None, ""
        alpha = ssa(np.atan2(x[1] - os_neyaw[1], x[0] - os_neyaw[0]) - os_neyaw[2])
        d = np.linalg.norm(x[0:2] - os_neyaw[0:2])
        z_camera = np.array([alpha, d])
        return z_camera

    def dhdx(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        raise RuntimeError(
            "dhdx must be avoided for TargetTrackerSequentialCTRV since it uses a sequential approach. "
            "Instead, call either dhdx_ais and dhdx_camera sequentially."
        )

    def dhdx_ais(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model (ais): dh/dx for z = h(x)
        """
        return np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ])

    def dhdx_camera(self, x: np.ndarray, *args, os_neyaw: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model (camera): dh/dx for z = h(x)

        z = [alpha, d] (relative angle, distance)
        """
        assert os_neyaw is not None, ""
        delta_n = x[0] - os_neyaw[0]
        delta_e = x[1] - os_neyaw[1]
        r_squared = delta_n**2 + delta_e**2
        r = np.sqrt(r_squared)

        if r < 1e-6:
            r = 1e-6
            r_squared = r**2

        da_dn = -delta_e / r_squared
        da_de = delta_n / r_squared
        dd_dn = delta_n / r
        dd_de = delta_e / r

        return np.array([
            [da_dn, da_de, 0, 0, 0],
            [dd_dn, dd_de, 0, 0, 0]
        ])

    def update_ais(self, z: np.ndarray) -> np.ndarray:
        """
        Returns updated state estimate based on measurement z from ais
        """
        dHdx_ais = self.dhdx_ais(self.x)
        S = dHdx_ais @ self.P @ dHdx_ais.T + self.R_ais
        K = self.P @ dHdx_ais.T @ np.linalg.inv(S)
        y = z - self.h_ais(self.x)
        y[3] = ssa(y[3])
        self.x = self.x + K @ y
        self.x[3] = ssa(self.x[3])
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ dHdx_ais) @ self.P
        return self.x

    def update_camera(self, z: np.ndarray, os_neyaw: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns updated state estimate based on measurement z from camera

        z = [alpha, d] (relative angle, distance)
        """
        dHdx_camera = self.dhdx_camera(self.x, os_neyaw=os_neyaw)
        S = dHdx_camera @ self.P @ dHdx_camera.T + self.R_camera
        K = self.P @ dHdx_camera.T @ np.linalg.inv(S)
        y = z - self.h_camera(self.x, os_neyaw=os_neyaw)
        y[0] = ssa(y[0])
        self.x = self.x + K @ y
        self.x[3] = ssa(self.x[3])
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ dHdx_camera) @ self.P @ (I - K @ dHdx_camera).T + K @ self.R_camera @ K.T
        return self.x


if __name__ == "__main__":
    Q = np.eye(5, 5)
    R_ais = np.eye(4, 4)
    R_camera = np.eye(2, 2)
    x0 = np.ones((5,)) * 0.1
    P0 = np.ones((5, 5)) * 0.1
    dt = 0.2

    target_tracker = TargetTrackerSequentialCTRV(
        Q=Q,
        R_ais=R_ais,
        R_camera=R_camera,
        x0=x0,
        P0=P0,
        dt=dt
    )

    print(target_tracker.predict(np.array([0.0, 0.0])))
    print(target_tracker.update_ais(np.array([0.1, 0.1, 1.0, 0.0])))
    print(target_tracker.update_camera(np.array([0.1, 20.0]), os_neyaw=np.array([0.0, 0.0, 0.0])))
