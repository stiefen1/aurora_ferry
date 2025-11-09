from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.weather import Current
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
from python_vehicle_simulator.lib.map import RandomMapGenerator
from python_vehicle_simulator.lib.actuator import AzimuthThruster
from python_vehicle_simulator.lib.path import PWLPath
import numpy as np, matplotlib.pyplot as plt
from aurora import SingleAzimuthThrusterParameters, AuroraFerry

dt = 0.2

ferry = AuroraFerry(
        eta0=np.array([0, 0, 0, 0, 0, 0]),
        nu0=np.array([0, 0, 0, 0, 0, 0]),
        dt=dt,
        actuators=[
            AzimuthThruster(xy=(-35, -9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
            AzimuthThruster(xy=(-35, 9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
            AzimuthThruster(xy=(35, -9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
            AzimuthThruster(xy=(35, 9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters()))
        ],
    )

env = NavEnv(
    own_vessel=ferry,
    target_vessels=[],
    obstacles=[], # map_generator.get([(vessel.eta[0], vessel.eta[1])], min_density=0.3)[0],
    dt=dt,
    current=Current(beta=-30.0*DEG2RAD, v=0.)
)

sim = Simulator(
        env,
        dt=dt,
        render_mode="human",
        verbose=2,
        skip_frames=1,
        window_size=(200, 200)
    )


sim.run(tf=100, render=True, store_data=True, control_commands=[
        np.array([-np.pi/4, 1e6]),
        np.array([-3*np.pi/4, 1e6]),
        np.array([np.pi/4, 1e6]),
        np.array([3*np.pi/4, 1e6])
    ])
# sim.save_animation(os.path.join('videos', 'vessel_simulation'), format="gif", fps=15)

fig2 = sim.plot_gnc_data_multi([
    'vessel.nu[0]', 'vessel.nu[1]', 'vessel.nu[5]'
])

plt.show(block=True)
