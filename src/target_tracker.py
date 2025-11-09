from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter
import numpy as np

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
            R, # Measurement Covariance
            x0, # Initial states
            P0, # Initial error covariance
            dt, # Sampling time, needed when building the system's model.
            *args,
            **kwargs
        ):
        super().__init__(Q, R, x0, P0, dt, *args, **kwargs)

    def f(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v
        """
        return np.array([
            x[0] + self.dt * x[2] * np.cos(x[3]),
            x[1] + self.dt * x[2] * np.sin(x[3]),
            x[2],
            x[3]
        ])
    
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        return np.array([
            [1, 0, self.dt * np.cos(x[3]), -self.dt * x[2] * np.sin(x[3])],
            [0, 1, self.dt * np.sin(x[3]), self.dt * x[2] * np.cos(x[3])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        raise RuntimeError(f"dhdx must be avoided for TargetTrackerSequentialEKF since it uses a sequential approach. Instead, call either dhdx_ais and dhdx_camera sequentially.")

    
    def h_ais(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        z_ais = self.dhdx_ais(x) @ x
        return z_ais
    
    def h_camera(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        z_camera = self.dhdx_camera(x) @ x
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
        return np.eye(4)
    
    def dhdx_camera(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model (camera): dh/dx for z = h(x)
        """
        return np.block([
            [np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 4))],
            ])
    
    def update_ais(self, z:np.ndarray) -> np.ndarray:
        """
        Returns updated state estimate based on measurement z from ais
        """
        dHdx_ais = self.dhdx_ais(self.x)
        S = dHdx_ais @ self.P @ dHdx_ais.T + self.R # Residual covariance -> Expected combined uncertainty of prediction & measurement
        K = self.P @ dHdx_ais.T @ np.linalg.inv(S) # Kalman Gain -> balance factor for blending prediction and measurements
        y = z - self.h_ais(self.x) # Residuals between measurement and measurement model
        self.x = self.x + K @ y # Update state estimate through innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ dHdx_ais) @ self.P
        return self.x
    
    def update_camera(self, z:np.ndarray) -> np.ndarray:
        """
        Returns updated state estimate based on measurement z from camera
        """
        dHdx_camera = self.dhdx_camera(self.x)
        S = dHdx_camera @ self.P @ dHdx_camera.T + self.R # Residual covariance -> Expected combined uncertainty of prediction & measurement
        K = self.P @ dHdx_camera.T @ np.linalg.inv(S) # Kalman Gain -> balance factor for blending prediction and measurements
        y = z - self.h_ais(self.x) # Residuals between measurement and measurement model
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