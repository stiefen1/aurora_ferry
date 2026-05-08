from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter
from typing import Optional
import numpy as np
from python_vehicle_simulator.utils.math_fn import ssa

class TargetTrackerSequentialEKF(IExtendedKalmanFilter):
    """
    Extended Kalman filter to estimate target ships position and speed using AIS & camera data.
    Sensor fusion is done in a sequential way to handle different data acquisition frequencies.
    An EKF is needed since AIS returns velocity as (sog, cog), which must be rotated in world frame.

    States:

        north   [m]
        east    [m]
        sog     [m/s]
        cog     [rad]
    
    Model:

        north += sog * cos(cog) * dt
        east += sog * sin(cog) * dt
        sog += 0
        cog += 0
    """
    def __init__(
            self,
            Q, # Process covariance
            R_ais, # AIS Measurement Covariance
            R_camera, # Camera Measurement Covariance
            x0, # Initial states
            P0, # Initial error covariance
            dt, # Sampling time, needed when building the system's model.
            sog_softplus_beta: float = 10.0,
            *args,
            **kwargs
        ):
        super().__init__(Q, R_ais, x0, P0, dt, *args, **kwargs)
        self.R_ais = R_ais
        self.R_camera = R_camera
        self.sog_softplus_beta = max(float(sog_softplus_beta), 1e-6)

    def _softplus(self, sog: float) -> float:
        """
        Smooth nonlinearity that behaves like ReLU but stays differentiable.
        """
        z = self.sog_softplus_beta * sog
        return float((np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))) / self.sog_softplus_beta)

    def _softplus_derivative(self, sog: float) -> float:
        """
        Derivative of the smooth speed nonlinearity wrt sog.
        """
        z = self.sog_softplus_beta * sog
        if z >= 0.0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = np.exp(z)
        return float(ez / (1.0 + ez))

    def f(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v

        x = [north, east, sog, cog]
        """
        sog_eff = self._softplus(float(x[2]))
        return np.array([
            x[0] + self.dt * sog_eff * np.cos(x[3]),
            x[1] + self.dt * sog_eff * np.sin(x[3]),
            sog_eff,
            x[3]
        ])
    
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        sog_eff = self._softplus(float(x[2]))
        dsog_dsog = self._softplus_derivative(float(x[2]))
        return np.array([
            [1, 0, self.dt * np.cos(x[3]) * dsog_dsog, -self.dt * sog_eff * np.sin(x[3])],
            [0, 1, self.dt * np.sin(x[3]) * dsog_dsog, self.dt * sog_eff * np.cos(x[3])],
            [0, 0, dsog_dsog, 0],
            [0, 0, 0, 1]
        ])
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        raise RuntimeError(f"dhdx must be avoided for TargetTrackerSequentialEKF since it uses a sequential approach. Instead, call either dhdx_ais and dhdx_camera sequentially.")

    
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
        assert os_neyaw is not None, f""
        alpha = ssa(np.atan2(x[1]-os_neyaw[1], x[0]-os_neyaw[0]) - os_neyaw[2])
        d = np.linalg.norm(x[0:2] - os_neyaw[0:2])
        z_camera = np.array([alpha, d])
        return z_camera

    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        raise RuntimeError(f"dhdx must be avoided for TargetTrackerSequentialEKF since it uses a sequential approach. Instead, call either dhdx_ais and dhdx_camera sequentially.")
    
    def dhdx_ais(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model (ais): dh/dx for z = h(x)
        """
        return np.eye(4) # We measure north, east, cog, sog directly
    
    def dhdx_camera(self, x: np.ndarray, *args, os_neyaw: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model (camera): dh/dx for z = h(x)

        z = [alpha, d] (relative angle, distance)
        alpha = atan2(ye - y_os, xn - x_os) - psi_os
        d = sqrt((xn - x_os)^2 + (ye - y_os)^2)

        J = [
                da/dn, da/de, da/dsog, da/dcog
                dd/dn, dd/de, dd/dsog, dd/dcog
            ]
        """
        assert os_neyaw is not None, f""
        # Position differences
        delta_n = x[0] - os_neyaw[0]  # north difference
        delta_e = x[1] - os_neyaw[1]  # east difference
        r_squared = delta_n**2 + delta_e**2
        r = np.sqrt(r_squared)
        
        # Avoid singularities when target is at same location as own ship
        if r < 1e-6:
            r = 1e-6
            r_squared = r**2
        
        # Partial derivatives for relative angle alpha = atan2(delta_e, delta_n) - psi_os
        da_dn = -delta_e / r_squared  # ∂α/∂north = -Δe/r²
        da_de = delta_n / r_squared   # ∂α/∂east = Δn/r²
        
        # Partial derivatives for distance d = sqrt(delta_n² + delta_e²)
        dd_dn = delta_n / r  # ∂d/∂north = Δn/r
        dd_de = delta_e / r  # ∂d/∂east = Δe/r
        
        return np.array([
            [da_dn, da_de, 0, 0],
            [dd_dn, dd_de, 0, 0]
        ])
    
    def update_ais(self, z:np.ndarray) -> np.ndarray:
        """
        Returns updated state estimate based on measurement z from ais
        """
        dHdx_ais = self.dhdx_ais(self.x)
        S = dHdx_ais @ self.P @ dHdx_ais.T + self.R_ais # Residual covariance -> Expected combined uncertainty of prediction & measurement
        K = self.P @ dHdx_ais.T @ np.linalg.inv(S) # Kalman Gain -> balance factor for blending prediction and measurements
        y = z - self.h_ais(self.x) # Residuals between measurement and measurement model
        y[3] = ssa(y[3]) # Wrap cog residual to [-pi, pi] to handle angle discontinuity
        self.x = self.x + K @ y # Update state estimate through innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ dHdx_ais) @ self.P
        return self.x
    
    def update_camera(self, z:np.ndarray, os_neyaw: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns updated state estimate based on measurement z from camera

        z = [alpha, d] (relative angle, distance)
        """
        
        dHdx_camera = self.dhdx_camera(self.x, os_neyaw=os_neyaw)
        S = dHdx_camera @ self.P @ dHdx_camera.T + self.R_camera # Residual covariance -> Expected combined uncertainty of prediction & measurement
        K = self.P @ dHdx_camera.T @ np.linalg.inv(S) # Kalman Gain -> balance factor for blending prediction and measurements
        y = z - self.h_camera(self.x, os_neyaw=os_neyaw) # Residuals between measurement and measurement model
        y[0] = ssa(y[0]) # Wrap relative angle residual to [-pi, pi] to handle angle discontinuity
        self.x = self.x + K @ y # Update state estimate through innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ dHdx_camera) @ self.P
        return self.x


    
if __name__ == "__main__":
    Q = np.eye(4, 4)
    R = np.eye(4, 4)
    x0 = np.ones((4,)) * 0.1
    P0 = np.ones((4, 4)) * 0.1
    dt = 0.2

    target_tracker = TargetTrackerSequentialEKF(
        Q,
        R,
        x0,
        P0,
        dt
    )


    print(target_tracker.predict(np.array([0, 0])))
    print(target_tracker.update_ais(np.array([0.1, 0.1, 1, 0])))
    print(target_tracker.update_camera(np.array([0.1, 0.1, 1, 0])))