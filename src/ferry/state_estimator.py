from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter
from src.ferry.aurora import Aurora3Dynamics
import numpy as np

class StateEstimatorEKF(IExtendedKalmanFilter):
    """
    Extended Kalman filter to estimate own ship states.

    States:
        Eta: 
            north   [m]
            east    [m]
            down    [m]
            roll    [rad]
            pitch   [rad]
            yaw     [rad]

        Nu: 
            surge speed [m/s]
            sway speed  [m/s]
            heave speed [m/s]
            roll rate   [rad/s]
            pitch rate  [rad/s]
            yaw rate    [rad/s]
        
        Azimuth angles:
            angle 1 [rad]
            angle 2 [rad]
            angle 3 [rad]
            angle 4 [rad]

        Thruster speeds:
            speed 1 [rad]
            speed 2 [rad]
            speed 3 [rad]
            speed 4 [rad]
    
    Model: Nonlinear 3DOFs

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
        self.aurora_dynamics = Aurora3Dynamics(dt)

    def f(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v
        """
        return self.aurora_dynamics.fd(x, u, np.ones((8,)), np.zeros((3,))).squeeze()
    
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        return self.aurora_dynamics.Ad(x, u, np.ones((8,)), np.zeros((3,)))
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.array([x[0], x[1], x[5], x[6], x[7], x[11], *x[12:16]])

    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        return np.array([
            [1] + 19 * [0],                 # North
            [0, 1] + 18 * [0],              # East
            5 * [0] + [1] + 14 * [0],       # yaw
            6 * [0] + [1] + 13 * [0],       # surge speed
            7 * [0] + [1] + 12 * [0],       # sway speed
            11 * [0] + [1] + 8 * [0],       # yaw rate
            12 * [0] + [1] + 7 * [0],       # azimuth 1
            13 * [0] + [1] + 6 * [0],       # azimuth 2
            14 * [0] + [1] + 5 * [0],       # azimuth 3
            15 * [0] + [1] + 4 * [0],       # azimuth 4
        ])



    
if __name__ == "__main__":
    Q = np.eye(20, 20)
    R = np.eye(10, 10)
    x0 = np.ones((20,)) * 0.1
    P0 = np.ones((20, 20)) * 0.1
    dt = 0.2

    state_estimator = StateEstimatorEKF(
        Q,
        R,
        x0,
        P0,
        dt
    )


    print(state_estimator.predict(np.array(8*[0])))