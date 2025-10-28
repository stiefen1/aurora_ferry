from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter
import numpy as np

class TargetTracker(IExtendedKalmanFilter):
    def __init__(self, Q, R, x0, P0, dt, *args, **kwargs):
        super().__init__(Q, R, x0, P0, dt, *args, **kwargs)

    def f(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v
        """
        return x
    
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        return np.eye(*x.shape)
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        z = x # All states are measured
        return z

    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        return np.eye(x.shape[0])