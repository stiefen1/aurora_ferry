import gymnasium as gym, numpy as np, numpy.typing as npt, sys, os
from typing import Dict, Optional, Tuple, List, Literal
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.utils.math_fn import ssa
from python_vehicle_simulator.utils.math_fn import Rzyx
from python_vehicle_simulator.lib.thruster import ROTATION_MATRIX

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


class GymNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            own_vessel:AuroraFerry,
            target_vessels:List[IVessel] = [],
            obstacles:List[Obstacle] = [],
            wind:Optional[Wind] = None,
            current:Optional[Current] = None,    
            render_mode:Optional[Literal['human']] = None,
            map_bounds: float = 1000.0
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
        self.delta = 0.1

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
        self.target = self.np_random.uniform(-self.map_bounds/2, self.map_bounds/2, size=2)

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

    def reward(self) -> float:
        """
        Compute reward based on distance to target, power consumption and fault diagnosis performance (to be implemented)
        
        Returns:
            float: Reward value (higher is better)
        """
        return (
            1 -
            (self.dist_to_target()/self.map_bounds) -
            self.weighted_power_consumption() +
            self.fault_diagnosis_performance()
        )

    # def no_sideways_speed

    def dist_to_target(self) -> float:
        """
        Calculate Euclidean distance to target position.
        
        Returns:
            float: Distance to target in meters
        """
        ne = np.array(self.own_vessel.eta.neyaw[0:2])
        return np.linalg.norm(ne-self.target).astype(float)
    
    def weighted_power_consumption(self, w: Optional[npt.NDArray] = None) -> float:
        if w is None:
            w = np.diag(np.concatenate([
                19 / (self.own_vessel.actuator_params.alpha_max - self.own_vessel.actuator_params.alpha_min)**2,
                1 / (self.own_vessel.actuator_params.speed_max - self.own_vessel.actuator_params.speed_min)**2,
            ])) / 20.0 * (4/3)
        return float((self.own_vessel.states[12:20] @ w @ self.own_vessel.states[12:20]))

    def fault_diagnosis_performance(self) -> float:
        # previous_diagnosis = self.own_vessel.diagnosis.prev
        # actual_values = ... # Extract theta whatever it is
        return 0
    
    def collision(self) -> bool:
        """
        Check for collisions with boundaries or obstacles.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        if np.any(self.own_vessel.eta.to_numpy()[0:2] > self.map_bounds) or np.any(self.own_vessel.eta.to_numpy()[0:2] < -self.map_bounds):
            return True
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
        delta = self.target - ne
        rel_yaw = np.array([ssa(self.own_vessel.eta[5] + np.atan2(-delta[1], delta[0]))])
        azimuth_angles = self.own_vessel.states[12:16]  # The outcome of a thruster depends on the azimuth angle -> it's probably needed here
        thruster_speeds = self.own_vessel.states[16:20]

        # Normalize each and cast to float32
        ne_norm = self._normalize(ne, self.ne_range["min"], self.ne_range["max"]).astype(np.float32)
        uvr_norm = self._normalize(uvr, self.uvr_range["min"], self.uvr_range["max"]).astype(np.float32)
        rel_target_norm = self._normalize(delta, self.rel_target_range["min"], self.rel_target_range["max"]).astype(np.float32)
        rel_yaw_norm = self._normalize(rel_yaw, self.rel_yaw_range["min"], self.rel_yaw_range["max"]).astype(np.float32)
        azimuth_angles_norm = self._normalize(azimuth_angles, self.azimuth_angles_range["min"], self.azimuth_angles_range["max"]).astype(np.float32)
        thruster_speeds_norm = self._normalize(thruster_speeds, self.thruster_speeds_range["min"], self.thruster_speeds_range["max"]).astype(np.float32)

        return {
            "ne": ne_norm,
            "uvr": uvr_norm,
            "rel_target": rel_target_norm,
            "rel_yaw": rel_yaw_norm,
            "azimuth_angles": azimuth_angles_norm,
            "thruster_speeds": thruster_speeds_norm
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
                "rel_target": gym.spaces.Box(-1.0, 1.0, shape=(2,)),    # Easier to figure out using relative pose
                "rel_yaw": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "azimuth_angles": gym.spaces.Box(-1.0, 1.0, shape=(4,)),
                "thruster_speeds": gym.spaces.Box(-1.0, 1.0, shape=(4,))
            }
        )

        # Used to map normalized observations to actual values (see method get_obs)
        self.ne_range = {"min": np.array([-self.map_bounds, -self.map_bounds]), "max": np.array([self.map_bounds, self.map_bounds])}
        self.uvr_range = {"min": np.array([-10, -10, -2]), "max": np.array([10, 10, 2])}
        self.rel_target_range = {"min":np.array([-self.map_bounds*1.5, -self.map_bounds*1.5]), "max": np.array([self.map_bounds*1.5, self.map_bounds*1.5])}
        self.rel_yaw_range = {"min": np.array([-np.pi]), "max": np.array([np.pi])}
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
            self.ax.set_xlim(-self.map_bounds, self.map_bounds)
            self.ax.set_ylim(-self.map_bounds, self.map_bounds)
            self.ax.scatter(*np.flip(self.target), c='red')
            self.ax.set_xlabel('East')
            self.ax.set_ylabel('North')
                
            self.ax.legend()
            plt.ion()
            plt.show()

        self.vessel_plot.set_data(*self.own_vessel.geometry_for_2D_plot) # type: ignore
        for i, actuator_plot in enumerate(self.actuators_plot):
            envelope = (ROTATION_MATRIX(self.own_vessel.states[12 + i]) @ self.own_vessel.actuator_params.geometries[i].T) + self.own_vessel.actuator_params.xy[i].reshape(-1, 1)
            envelope_in_ned_frame = Rzyx(*self.own_vessel.eta.to_numpy()[3:6].tolist())[0:2, 0:2] @ envelope + self.own_vessel.eta.to_numpy()[0:2, None]
            actuator_plot.set_data(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :])

        self.ax.set_title(f"Step: {self._step} | Time: {self._step * self.action_repeat * self.own_vessel.dynamics.dt}")
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

    env = GymNavEnv(
        own_vessel=AuroraFerry(dt),
        render_mode='human',
        map_bounds=1000.0
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