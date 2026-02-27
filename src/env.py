import gymnasium as gym, numpy as np
from typing import Optional, Tuple, Dict, Literal, List
from src.map import HelsingborgMap
from src.ais import AIS
from python_vehicle_simulator.lib.weather import Wind, Current
from datetime import datetime, timedelta
from src.aurora import SingleAzimuthThrusterParameters, AuroraFerry
from python_vehicle_simulator.lib.actuator import AzimuthThruster
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from python_vehicle_simulator.lib.guidance import PathFollowingGuidance
from colav import MovingShip
from src.navigation import NavigationAurora
import pathlib, os, glob
import matplotlib.pyplot as plt



class AuroraNavEnv(gym.Env): 
    def __init__(
            self,
            u_des: float = knot_to_m_per_sec(8),
            path_to_ais_folder: str = os.path.join('data'),     # Folder containing all AIS database that will be used for training
            action_repeat: int = 1,
            max_steps: int = 200,
            render_mode: Literal['human', None] = None,
            dt: float = 1,
            buffer_moving: float = 200,
            join_style: Literal['round', 'mitre', 'bevel'] = 'mitre',
            mpc_horizon: int = 10,
    ):
        self.u_des = u_des
        self.map = HelsingborgMap()
        self.action_repeat = action_repeat

        # The Aurora Ferry
        self.n_azimuth_thrusters = 4

        # AIS & time 
        self.path_to_ais_folder = path_to_ais_folder
        self.paths_to_ais = glob.glob(os.path.join(self.path_to_ais_folder, '*.csv'))
        self.n_ais = len(self.paths_to_ais)
        self.ais_max_age_seconds: int = 60

        # Target vessel buffering
        self.buffer_moving = buffer_moving
        self.join_style = join_style

        # MPC horizon
        self.mpc_horizon = mpc_horizon

        # Weather generators -> Provide bounds and then use objects as self.wind_generator.sample() at the beginning of each episode
        self.wind_generator = None
        self.current_generator = None

        # Rendering
        self.render_mode = render_mode

        # Episode
        self.current_step: int = 0
        self.max_steps = max_steps

        # Simulation
        self.dt = dt

        # Reward
        self.gamma = 10

        # Termination conditions
        self.distance_threshold_to_docking_area = 500
        self.safety_distance_threshold = 60 # Aurora has a LOA of 111.2. If distance < 60, we consider it as a collision

        self.reset()

        # Initialize action and observation spaces
        self.init_action_space()
        self.init_observation_space()

    def init_action_space(self) -> None:
        """
        Action space is made of azimuth angles & propeller's speed
        """
        self.action_space = gym.spaces.Box(
            low=np.array([actuator.u_min[0] for actuator in self.ferry.actuators] + [actuator.u_min[1] for actuator in self.ferry.actuators]),
            high=np.array([actuator.u_max[0] for actuator in self.ferry.actuators] + [actuator.u_max[1] for actuator in self.ferry.actuators]),
            dtype=np.float32
        )

    def init_observation_space(self) -> None:
        """
        Observation space is ferry state + nonlearnable parameters
        """
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14, 1), dtype=np.float32),
                "wpts": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.mpc_horizon+1)),
                "nu_des": gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array(3*[10]))
            }
        )
        # self.observation_space = gym.spaces.Box(
        #     low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        # )

    def reward(self) -> float:
        """
        Ferry must aim at surviving --> reward > 0
        better tracking -> higher reward (controlled by a constant value gamma)
        """
        
        # nd, ed = self.ferry.guidance.trajectory.get(self.t)
        nd, ed = 0, 0
        dist = ( (nd - self.ferry.eta.n)**2 + (ed - self.ferry.eta.e)**2 )**0.5
        return np.exp(-dist / self.gamma) 
    
    def collision_with_target_vessels(self) -> bool:
        """
        Docstring for collision
        
        :param self: Description
        :return: Description
        :rtype: bool
        """

        vessels = self.ais.get_vessels_at_time(self.t, max_age_seconds=self.ais_max_age_seconds)
        for vessel in vessels:
            dist = ( ( vessel.north - self.ferry.eta.n )**2 + ( vessel.east - self.ferry.eta.e )**2 )**0.5
            if dist <= self.safety_distance_threshold:
                return True
        return False
    
    def inside_docking_area(self) -> bool:
        nd, ed = self.route.interpolate(1, normalized=True) # Docking position (north, east)
        dist = ( (nd - self.ferry.eta.n)**2 + (ed - self.ferry.eta.e)**2 )**0.5
        return dist <= self.distance_threshold_to_docking_area
    
    def overtime(self) -> bool:
        """
        time is outside the AIS range
        """
        return self.t >= self.last_ais_timestamp
    
    def terminated(self) -> bool:
        """
        When the agent reaches a terminal state of the environment:
        - Collision with a target ship
        - Reaches the docking area
        """

        if self.collision_with_target_vessels():
            print("terminated: collision with target vessel")
            return True
        
        if self.inside_docking_area():
            print("terminated: inside docking area")
            return True

        if self.overtime():
            print("terminated: overtime")
            return True

        return False
       
    def truncated(self) -> bool:
        """
        When the agent reaches the time limit for the environment
        """
        if self.current_step >= self.max_steps:
            print("truncated")
            return True
        return False

    def reset(self, *args, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # Reset seeds
        super().reset(seed=seed)

        # Reset steps
        self.current_step = 0

        # Randomly sample a crossing
        self.ais = AIS(self.paths_to_ais[self.np_random.integers(0, self.n_ais)])
        self.t: datetime = self.ais.get_first_timestamp()
        self.last_ais_timestamp: datetime = self.ais.get_last_timestamp()

        # Select initial pose and states
        self.route_name = 'Helsingør (DK) - Helsingborg (S)' if self.np_random.integers(0, 1, endpoint=True) else 'Helsingør (DK) - Helsingborg (SE)'
        self.route = self.map.get_ferry_routes()[self.route_name]

        # Select weather from desired range
        self.wind = Wind(np.deg2rad(45), 3)
        self.current = Current(np.deg2rad(40), 1)

        # Reset ferry
        eta0, nu0 = self.route.get_initial_pose(radians=True), (0, 0, 0)
        self.ferry = AuroraFerry(
            eta0=eta0,
            nu0=nu0,
            dt=self.dt,
            actuators=[
                AzimuthThruster(xy=(-35, -9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
                AzimuthThruster(xy=(-35, 9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
                AzimuthThruster(xy=(35, -9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
                AzimuthThruster(xy=(35, 9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters()))
            ],
            navigation=NavigationAurora(
                eta=np.array(eta0),
                nu=np.array(nu0),
                dt=self.dt,
                max_age_seconds=self.ais_max_age_seconds,
                sensors = {'ais': self.ais}
            ),
            guidance=PathFollowingGuidance(self.route, self.mpc_horizon, self.dt, desired_speed=self.u_des)
        )
        # self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs
        observation = self.get_obs({"eta": eta0, "nu": nu0, "current": self.current, "wind": self.wind, "obstacles": [], "target_vessels": []}) # type: ignore
        info = {}

        return observation, info
    
    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        target_vessels = []
        for vessel in self.ais.get_vessels_at_time(self.t):
            if vessel.heading is not None and vessel.cog is not None and vessel.sog is not None:
                target_vessels.append(
                    MovingShip.from_csog(
                        (vessel.east, vessel.north),
                        vessel.heading,
                        vessel.cog,
                        knot_to_m_per_sec(vessel.sog),
                        vessel.length,
                        vessel.width,
                        degrees=True,
                        mmsi=vessel.mmsi
                    ).buffer(self.buffer_moving, join_style=self.join_style)
                )

        # Simulation
        for _ in range(self.action_repeat):
            self.ferry.step(
                self.current,
                self.wind,
                self.map.get_shore_as_obstacles(),
                target_vessels=target_vessels,
                timestamp=self.t,
                control_commands=[np.array([action[i], action[i+4]]) for i in range(4)] # 4 azimuth thrusters (alpha, n)
            )
            self.t += timedelta(seconds=self.dt)

        self.current_step += 1
        obs = self.get_obs(self.ferry.navigation.last_observation) # type: ignore
        r = self.reward()
        term = self.terminated()
        trunc = self.truncated()
        info = {}

        return obs, r, term, trunc, info
    
    def get_obs(self, navigation_obs: Dict) -> Dict:
        return {
            "state": np.concatenate([
                    self.ferry.eta.to_numpy(dofs=3),
                    self.ferry.nu.to_numpy(dofs=3),
                    [actuator.u_actual_prev[0] for actuator in self.ferry.actuators],
                    [actuator.u_actual_prev[1] for actuator in self.ferry.actuators],
                ]).reshape(14, 1),
            "wpts": np.array(self.ferry.guidance(**navigation_obs)[2]["path"]).T,
            "nu_des": np.array([self.u_des, 0, 0])
        }
    
    def render(self) -> None:
        """Render the environment if render_mode is 'human'."""
        if self.render_mode != 'human':
            return
            
        # Initialize figure if not exists
        if not hasattr(self, '_fig'):
            self._fig, self._ax = plt.subplots(figsize=(10, 8))
            self._fig.suptitle('Aurora Ferry Navigation Environment')
            plt.ion()  # Enable interactive mode
            
        # Clear and redraw
        self._ax.clear()
        
        # Plot map
        self.map.plot(ax=self._ax)
        self._ax.set_xlim(self.map.xlim)
        self._ax.set_ylim(self.map.ylim)
        
        # Plot ferry (own ship)
        self.ferry.plot(ax=self._ax)
        self.route.plot(ax=self._ax)
        
        # Plot vessels from AIS (target ships)
        vessels = self.ais.get_vessels_at_time(self.t, max_age_seconds=self.ais_max_age_seconds)
        for vessel in vessels:
            # Plot vessel position as red dot
            self._ax.scatter(vessel.east, vessel.north, c='red', s=30, alpha=0.8)
            
            # Plot vessel geometry if available
            if vessel.geometry is not None:
                self._ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], 
                            'r-', linewidth=1.5, alpha=0.7)
        
        # Update display
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.01)




def check_environment() -> None:
    from gymnasium.utils.env_checker import check_env
    from python_vehicle_simulator.vehicles.otter import Otter, OtterParameters, OtterThrusterParameters
    from python_vehicle_simulator.lib.actuator import Thruster
    from python_vehicle_simulator.lib.weather import Wind, Current
    from python_vehicle_simulator.utils.unit_conversion import DEG2RAD

    env = AuroraNavEnv()
    # This will catch many common issues
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

def check_rendering(n_steps: int = 50) -> None:
    """Test the rendering functionality by running N steps with random actions."""
    import time
    
    # Create environment with rendering enabled
    env = AuroraNavEnv(render_mode='human')
    
    print(f"Starting rendering test for {n_steps} steps...")
    
    try:
        # Reset environment
        obs, info = env.reset()
        env.render()
        
        for step in range(n_steps):
            print(f"Step {step + 1}/{n_steps}")
            
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render
            env.render()
            
            # Small delay for visualization
            time.sleep(0.1)
            
            # Check if episode ended
            if terminated or truncated:
                print(f"Episode ended at step {step + 1}")
                obs, info = env.reset()
                env.render()
        
        print("Rendering test completed successfully!")
        
    except Exception as e:
        print(f"Rendering test failed: {e}")
    
    finally:
        # Close matplotlib figure if it exists
        if hasattr(env, '_fig') and env._fig is not None:
            plt.close(env._fig)

if __name__=="__main__":
    check_environment()
    check_rendering()