from src.aurora import AuroraFerry
from src.rl.traj_tracking_controller import PPOTrajTrackingController
from src.navigation import NavigationAurora
from src.guidance import TimespaceGuidance
from src.map import HelsingborgMap
from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.obstacle import Obstacle
import numpy as np

helsingborg = HelsingborgMap()

u_des = 5
dt = 0.2
path_to_ppo_params = 'models\\ppo'
start_ne = helsingborg.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)'].waypoints[0] 
start_ne[1] += 500
states = np.array([*start_ne] + 3 * [0] + [70] + 14*[0])
aurora = AuroraFerry(
    dt,
    eta = (states[0], states[1], states[5]),
    nu = (states[6], states[7], states[11]),
    control=PPOTrajTrackingController( # TODO: Check if action_repeat creates an issue
        path_to_params=path_to_ppo_params
    ),
    navigation=NavigationAurora(
        states,
        dt
    ),
    guidance=TimespaceGuidance(
        global_path=helsingborg.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)'],
        u_des=u_des,
        shore=[poly.simplify(3e2) for poly in helsingborg.polygons]
    )
)

# print(list(zip(*helsingborg.polygons[0].exterior.coords.xy)))
env = NavEnv(aurora, [], [Obstacle(geometry=list(zip(*poly.exterior.coords.xy))) for poly in helsingborg.polygons], dt)
sim = Simulator(env, dt=dt, skip_frames=10, render_mode='human', window_size=(1000, 1000))
sim.run(1000, render=True, store_data=False)