import gymnasium as gym, numpy as np, numpy.typing as npt, sys, os
from typing import Dict, Optional, Tuple, List, Literal
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.utils.math_fn import ssa
from python_vehicle_simulator.utils.math_fn import Rzyx
from python_vehicle_simulator.lib.thruster import ROTATION_MATRIX
from python_vehicle_simulator.lib.path import PWLPath

from src.aurora import AuroraFerry
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


"""
IMPORTANT NOTES:

- Current and Wind are not handled in the dynamics, i.e. using non-zero values won't have any effect
- This environment is made to train the Revolt to reach a target point
- Action repeat  = number of time a single action is applied to the system (see "step" method)
- To adapt this environment to other tasks, you must modify the following methods:
    - reward()
    - init_action_space()
    - init_observation_space()
    - get_obs()                 ->              extract observation from the simulation
    - map_action_to_command()

"""

DEFAULT_PATH_PARAMS = {
    "d_tot": 5000, "max_turn_deg": 45, "seg_len_range":(100, 500), "start":(0.0, 0.0), "N":1
}

class TrajTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            own_vessel:AuroraFerry,
            target_vessels:List[IVessel] = [],
            obstacles:List[Obstacle] = [],
            wind:Optional[Wind] = None,
            current:Optional[Current] = None,    
            render_mode:Optional[Literal['human']] = None,
            map_bounds: float = 1e3,
            path_params: Dict = DEFAULT_PATH_PARAMS,
            V_range: Optional[Tuple[float, float]] = None,
            n_wpts: int = 5,
            wpts_space_multiplicator: int = 10,
            initial_angle_range: Tuple[float, float] = (-30, 30)
    ):
        """
        Gymnasium navigation environment for vessel control.
        
        own_vessel:     Vessel to be controlled
        target_vessels: List of other vessels in environment
        obstacles:      List of obstacles to avoid
        wind:           Wind disturbance model
        current:        Current disturbance model
        render_mode:    Visualization mode ('human' or None)
        map_bounds:     Map boundary limits (±map_bounds for both x and y)
        """
        self.own_vessel = own_vessel
        self.target_vessels = target_vessels
        self.obstacles = obstacles
        self.wind = wind or Wind(0, 0)
        self.current = current or Current(0, 0)
        self.map_bounds = map_bounds
        self.path_params = path_params
        self.V_range = V_range if V_range is not None else (0, self.own_vessel.vessel_params.surge_speed_max)
        self.n_wpts = n_wpts
        self.wpts_space_multiplicator = wpts_space_multiplicator
        self.initial_angle_range = initial_angle_range

        self.init_action_space()
        self.init_observation_space()

        self.safety_radius = 2.5
        self.action_repeat = 10 # if dt is 0.02, this leads the RL frequency to 1/(10*0.02) = 1/0.2 = 5Hz

        # Rendering 
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        # Current step (for plot purpose)
        self._step = 0
        self.max_steps = 500 # i.e. 100 seconds for dt=0.02 and action_repeat=10

        # Reward function
        self.huber_penalty_slope = 10 # delta
        self.huber_penalty_weight = 30 # q_x,y
        self.heading_penalty_weight = 50 # 50 # q_psi
        self.singular_value_penalty = 1e-3 # epsilon -> for nonsigular thruster configuration
        self.singular_value_weight = 1e-5 # 1e-5 # rho -> for nonsigular thruster configuration

        self.Q = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ]) # Velocity weight matrix

        self.Ra = np.eye(4) * 1e-2 # Azimuth weight matrix
        self.Rf = np.eye(4) * 1e-8 # 1e-1 # Force weight matrix

    def reset(self, seed: int | None = None, options: Dict | None = None) -> Tuple[Dict, Dict]:
        """
        Start a new episode.

        seed:       Random seed for reproducible episodes
        options:    Additional configuration (unused)

        Returns:
            Tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed) # type: ignore

        # Reset own vessel position to 0
        self.own_vessel.reset()
        for target_vessel in self.target_vessels:
            target_vessel.reset()

        # Sample a new target position within map bounds
        # self.target = self.np_random.uniform(-self.map_bounds/2, self.map_bounds/2, size=2)
        self.path = PWLPath.sample(**self.path_params, initial_angle=float(self.np_random.uniform(*self.initial_angle_range)), seed=seed)
        self.V_des = float(self.np_random.uniform(*self.V_range))

        observation = self._get_obs()
        info = self._get_info()

        # Reset figure
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        # Reset step (for plot purpose)
        self._step = 0

        return observation, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one timestep within the environment.

        action:     Action to take (normalized to [-1,1])    (6,)

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        self._step += 1

        # Step vessels
        for _ in range(self.action_repeat):
            for vessel in self.target_vessels:
                vessel.step(self.current, self.wind, self.obstacles, [])
            self.own_vessel.step(self.current, self.wind, self.obstacles, self.target_vessels, control_commands=self.map_action_to_command(action), theta=np.array(8*[1.0]))

        # Reward result of action
        reward = self.reward()

        # Get observation
        observation = self._get_obs()

        # Check for collisions
        terminated = self.collision()

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = self._step >= self.max_steps
        
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def cost_tracking(self, p:np.ndarray, p_d:np.ndarray) -> float:
        return float(self.huber_penalty_slope**2 * (np.sqrt(1 + ((p[0:2]-p_d[0:2]).T @ (p[0:2]-p_d[0:2])) / self.huber_penalty_slope**2) - 1))
    
    def cost_heading(self, psi:float, psi_d:float) -> float:
        return float((1 - np.cos(psi - psi_d)) / 2)
    
    def cost_speed(self, nu:np.ndarray, nu_d:np.ndarray) -> float:
        return float((nu-nu_d).T @ self.Q @ (nu-nu_d))
    
    def cost_alpha(self, alpha:np.ndarray) -> float:
        return (alpha.T @ self.Ra @ alpha).astype(float)
    
    def cost_thruster_speed(self, thruster_speed:np.ndarray) -> float:
        return (thruster_speed.T @ self.Rf @ thruster_speed).astype(float)
    
    # def cost_singular_alpha(self, alpha:np.ndarray) -> float:
    #     return 0
    
    def cost(self, eta:np.ndarray, nu:np.ndarray, eta_d:np.ndarray, nu_d:np.ndarray, alpha:np.ndarray, thruster_speed:np.ndarray) -> float:
        cost = self.huber_penalty_weight * self.cost_tracking(eta[0:2], eta_d[0:2])
        cost += self.heading_penalty_weight * self.cost_heading(eta[2], eta_d[2])
        cost += self.singular_value_weight * 0
        cost += self.cost_speed(nu, nu_d)
        cost += self.cost_alpha(alpha)
        cost += self.cost_thruster_speed(thruster_speed)
        return cost

    # def reward(self) -> float:
    #     """
    #     Compute reward based on distance to target, power consumption and fault diagnosis performance (to be implemented)
        
    #     Returns:
    #         float: Reward value (higher is better)
    #     """
    #     eta = np.array(self.own_vessel.eta.neyaw)
    #     nu = np.array(self.own_vessel.nu.uvr)
    #     eta_d = np.array(self.path.get_target_wpts_from(eta[0], eta[1], 0, 1)[0]) # dp is wrong but we don't care because we only take the first point
    #     nu_d = np.array([self.V_des, 0, 0])
    #     return np.exp(-self.cost(eta, nu, eta_d, nu_d, self.own_vessel.states[12:16], self.own_vessel.states[16:20])/10)

    def reward(self) -> float:
        """
        Compute reward based on distance to target, power consumption and fault diagnosis performance (to be implemented)
        
        Returns:
            float: Reward value (higher is better)
        """
        return (
            1 -
            (self.dist_to_target()/500) -
            self.speed_error() / self.own_vessel.vessel_params.surge_speed_max -
            self.weighted_power_consumption()
        )
    
    def speed_error(self) -> float:
        return np.abs(np.linalg.norm(self.own_vessel.states[6:8]).astype(float) - self.V_des)

    def dist_to_target(self) -> float:
        """
        Calculate Euclidean distance to target path.
        
        Returns:
            float: Distance to path in meters
        """
        ne = np.array(self.own_vessel.eta.neyaw[0:2])
        closest_ne = self.path.closest_point(ne[0], ne[1]) # type: ignore
        target = np.array([closest_ne[1], closest_ne[0]])
        return np.linalg.norm(ne-target) # type: ignore
    
    def weighted_power_consumption(self, w: Optional[npt.NDArray] = None) -> float:
        if w is None:
            w = np.diag(np.concatenate([
                10 / (self.own_vessel.actuator_params.alpha_max - self.own_vessel.actuator_params.alpha_min)**2,
                1 / (self.own_vessel.actuator_params.speed_max - self.own_vessel.actuator_params.speed_min)**2,
            ])) / 50.0
        return float((self.own_vessel.states[12:20] @ w @ self.own_vessel.states[12:20]))
    
    def collision(self) -> bool:
        """
        Check for collisions with boundaries or obstacles.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        for obs in self.obstacles:
            if obs.distance(*self.own_vessel.eta.neyaw[0:2]) < self.safety_radius:
                return True
        return False

    def _normalize(self, x, min_val, max_val):
        """
        Normalize x from [min_val, max_val] to [-1, 1].
        
        x:          Value to normalize
        min_val:    Minimum value of range
        max_val:    Maximum value of range
        
        Returns:
            Normalized value in [-1, 1]
        """
        return 2 * (x - min_val) / (max_val - min_val) - 1

    def _get_obs(self) -> Dict:
        """
        Convert internal state to normalized observation format.
        
        Returns:
            Dict: Normalized observations with keys 'ne', 'uvr', 'rel_target', 'rel_yaw'
        """
        # Get raw values
        ne = np.array([self.own_vessel.eta.n, self.own_vessel.eta.e])
        uvr = np.array(self.own_vessel.nu.uvr)
        distances, rel_yaws = [], []
        for target in self.path.get_target_wpts_from(ne[0], ne[1], self.action_repeat*self.own_vessel.dynamics.dt*self.V_des*self.wpts_space_multiplicator, self.n_wpts): # type: ignore
            delta = target[0:2] - ne
            distance = float(np.linalg.norm(delta))
            rel_yaw = ssa(self.own_vessel.eta[5] + np.atan2(-delta[1], delta[0]))
            distances.append(distance)
            rel_yaws.append(rel_yaw)

        azimuth_angles = self.own_vessel.states[12:16]  # The outcome of a thruster depends on the azimuth angle -> it's probably needed here
        thruster_speeds = self.own_vessel.states[16:20]

        # Normalize each and cast to float32
        ne_norm = self._normalize(ne, self.ne_range["min"], self.ne_range["max"]).astype(np.float32)
        uvr_norm = self._normalize(uvr, self.uvr_range["min"], self.uvr_range["max"]).astype(np.float32)
        rel_target_norm = self._normalize(np.array(distances), self.rel_target_range["min"], self.rel_target_range["max"]).astype(np.float32)
        rel_yaw_norm = self._normalize(np.array(rel_yaws), self.rel_yaw_range["min"], self.rel_yaw_range["max"]).astype(np.float32)
        speed_error_norm = self._normalize(np.array(self.own_vessel.nu.u - self.V_des), self.speed_error_range["min"], self.speed_error_range["max"]).astype(np.float32)
        azimuth_angles_norm = self._normalize(azimuth_angles, self.azimuth_angles_range["min"], self.azimuth_angles_range["max"]).astype(np.float32)
        thruster_speeds_norm = self._normalize(thruster_speeds, self.thruster_speeds_range["min"], self.thruster_speeds_range["max"]).astype(np.float32)

        return {
            "ne": ne_norm,
            "uvr": uvr_norm,
            "rel_target": rel_target_norm,
            "rel_yaw": rel_yaw_norm,
            "speed_error": speed_error_norm,
            "azimuth_angles": azimuth_angles_norm,
            "thruster_speeds": thruster_speeds_norm,
        }

    def init_observation_space(self) -> None:
        """
        Initialize observation space with normalized ranges.
        
        Sets up Dict observation space with bounds [-1,1] for all components
        and defines mapping ranges for normalization.
        """
        # Observation space is normalized to enhance learning stability
        self.observation_space = gym.spaces.Dict(
            {
                "ne": gym.spaces.Box(-1.0, 1.0, shape=(2,)),            # Because we want the vessel to remain within bounds
                "uvr": gym.spaces.Box(-1.0, 1.0, shape=(3,)),           # Surge-Sway-YawRate
                "rel_target": gym.spaces.Box(-1.0, 1.0, shape=(self.n_wpts,)),    # Easier to figure out using relative pose
                "rel_yaw": gym.spaces.Box(-1.0, 1.0, shape=(self.n_wpts,)),
                "speed_error": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "azimuth_angles": gym.spaces.Box(-1.0, 1.0, shape=(4,)),
                "thruster_speeds": gym.spaces.Box(-1.0, 1.0, shape=(4,))
            }
        )

        # Used to map normalized observations to actual values (see method get_obs)
        self.ne_range = {"min": np.array(2*[-self.path_params['d_tot']]), "max": np.array(2*[self.path_params['d_tot']])}
        self.uvr_range = {"min": np.array([-10, -10, -10]), "max": np.array([10, 10, 10])}
        self.rel_target_range = {"min":np.array(self.n_wpts*[0]), "max": np.array(self.n_wpts*[self.path_params['d_tot']])} # relative distance to a point of the horizon
        self.rel_yaw_range = {"min": np.array(self.n_wpts*[-np.pi]), "max": np.array(self.n_wpts*[np.pi])} # relative bearing angle to a point of the horizon
        self.speed_error_range = {"min": np.array([-3*self.V_range[1]]), "max": np.array([3*self.V_range[1]])}
        self.azimuth_angles_range = {"min": self.own_vessel.actuator_params.alpha_min, "max": self.own_vessel.actuator_params.alpha_max}
        self.thruster_speeds_range = {"min": self.own_vessel.actuator_params.speed_min, "max": self.own_vessel.actuator_params.speed_max}
    
    def _get_info(self) -> Dict:
        """
        Compute auxiliary information for debugging.

        Returns:
            Dict: Empty info dictionary (can be extended)
        """
        return {

        }
    
    def map_action_to_command(self, action) -> None:
        """
        Map normalized action [-1,1] to actuator command range.
        
        action:     Normalized action                   (6,)
        
        Returns:
            Command for vessel actuators [azimuth, speeds]
        """
        command = np.zeros_like(action)
        alpha_min, alpha_max = self.own_vessel.actuator_params.alpha_min, self.own_vessel.actuator_params.alpha_max
        thruster_speed_min, thruster_speed_max = self.own_vessel.actuator_params.speed_min, self.own_vessel.actuator_params.speed_max
        command[0:4] = action[0:4] * (alpha_max - alpha_min) / 2 + (alpha_min + alpha_max) / 2
        command[4:8] = action[4:8] * (thruster_speed_max - thruster_speed_min) / 2 + (thruster_speed_min + thruster_speed_max) / 2
        return command

    def init_action_space(self) -> None:
        """
        Initialize action space with normalized range [-1,1].
        
        Sets up Box action space matching vessel's control input dimensions.
        """
        # Observation space is normalized to enhance learning stability
        self.action_space = gym.spaces.Box(
            -np.ones(shape=(self.own_vessel.dynamics.nu,)), 
            +np.ones(shape=(self.own_vessel.dynamics.nu,))
        ) # action space is -1, +1

    def render(self, mode=None):
        """
        Render the environment for visualization.
        
        mode:       Render mode ('human' for matplotlib visualization)
        
        Creates and updates a 2D plot showing vessel position and target.
        """
        mode = mode or self.render_mode
        if mode not in ("human",):
            return

        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.vessel_plot, = self.ax.plot([], [], label='Vessel')
            self.actuators_plot: List[Line2D] = [self.ax.plot([], [], label=f'Th{i}')[0] for i in range(4)]
            self.waypoints_plot = self.ax.scatter([], [], label='waypoints')
            self.ax.set_xlim(-self.map_bounds, self.map_bounds)
            self.ax.set_ylim(-self.map_bounds, self.map_bounds)
            self.ax.plot(self.path.waypoints[:, 1], self.path.waypoints[:, 0], c='red') # type: ignore
            self.ax.set_xlabel('East')
            self.ax.set_ylabel('North')
                
            self.ax.legend()
            plt.ion()
            plt.show()

        waypoints = np.array(self.path.get_target_wpts_from(
            self.own_vessel.states[0],
            self.own_vessel.states[1],
            self.action_repeat*self.own_vessel.dynamics.dt*self.V_des*self.wpts_space_multiplicator,
            self.n_wpts
        ))
        # waypoints should be in format [[x1,y1], [x2,y2], ...] but path returns [[n1,e1], [n2,e2], ...]
        # Convert from North,East to East,North for plotting
        self.waypoints_plot.set_offsets(np.flip(waypoints[:, 0:2]))
        self.vessel_plot.set_data(*self.own_vessel.geometry_for_2D_plot) # type: ignore
        for i, actuator_plot in enumerate(self.actuators_plot):
            envelope = (ROTATION_MATRIX(self.own_vessel.states[12 + i]) @ self.own_vessel.actuator_params.geometries[i].T) + self.own_vessel.actuator_params.xy[i].reshape(-1, 1)
            envelope_in_ned_frame = Rzyx(*self.own_vessel.eta.to_numpy()[3:6].tolist())[0:2, 0:2] @ envelope + self.own_vessel.eta.to_numpy()[0:2, None]
            actuator_plot.set_data(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :])

        self.ax.set_title(f"Step: {self._step} | Time: {self._step * self.action_repeat * self.own_vessel.dynamics.dt} | V_des: {self.V_des} | V: {np.linalg.norm(self.own_vessel.states[6:8])}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

# # Register the environment so we can create it with gym.make()
# gym.register(
#     id="gymnasium_env/GymNavEnv-v0",
#     entry_point=GymNavEnv,
#     max_episode_steps=300,  # Prevent infinite episodes
# )

def check_environment() -> None:
    """
    Validate environment implementation using gymnasium checker.
    
    Creates test environment and runs standard validation checks
    to ensure compatibility with RL training frameworks.
    """
    from gymnasium.utils.env_checker import check_env
    from python_vehicle_simulator.lib.weather import Wind, Current
    from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
    from src.aurora import AuroraFerry

    dt = 0.1

    env = TrajTrackingEnv(
        own_vessel=AuroraFerry(dt),
        render_mode='human',
    )
    # This will catch many common issues
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    # Run an episode
    obs, info = env.reset()
    
    for step in range(env.max_steps):
        action = env.action_space.sample()  # Random action
        # action = np.array([0, 0, 0.5, 0.5, 1, 1, 1, 1])
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        print(f"Step {step}: reward={reward:.3f}, distance={env.dist_to_target():.1f}m")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    plt.show(block=True)  # Keep plot open

if __name__=="__main__":
    check_environment()