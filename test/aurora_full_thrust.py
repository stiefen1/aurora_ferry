from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.weather import Current
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
import numpy as np, matplotlib.pyplot as plt
from src.ferry.aurora import AuroraFerry

dt = 0.2

ferry = AuroraFerry(
        dt
    )

env = NavEnv(
    own_vessel=ferry,
    target_vessels=[],
    obstacles=[],
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

sim.run(tf=100, render=True, store_data=True, control_commands=np.array([0, 0, 0, 0, 1e5, 1e5, 1e5, 1e5]))

fig2 = sim.plot_gnc_data_multi([
    'vessel.nu[0]', 'vessel.nu[1]', 'vessel.nu[5]'
])

plt.show(block=True)
