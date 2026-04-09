from enum import Enum
import numpy as np, numpy.typing as npt

class Weather(Enum):
    SUNNY = 0
    CLOUDY = 1
    FOGGY = 2

def get_detection_probability(size: float, d: float | npt.NDArray, weather: Weather = Weather.SUNNY) -> float | npt.NDArray:
    """
    Compute the probability of detection using a camera.

    size: size of the target ship. A representative value such as loa or (loa**2 + beam**2)**0.5 can be used
    d: distance of target ship with respect to the camera
    weather: Weather object

    ratio_at_prob_50 is the ratio between distance and size of the target at which the detection probability is 50%
    scale affects the transition speed between probabilities 0 and 1
    """
    x = d / size # ratio between distance and TS size -> inverse prop. to FOV
    match weather:
        case Weather.SUNNY:
            ratio_at_prob_50 = 15 # when distance is 10*size, detection probability is 0.5
            scale = 1
        case Weather.CLOUDY:
            ratio_at_prob_50 = 12
            scale = 1.5
        case Weather.FOGGY:
            ratio_at_prob_50 = 10
            scale = 2
        case _:
            ratio_at_prob_50 = 15
            scale = 1
    return 1 / (1 + np.exp( (x - ratio_at_prob_50) / scale ).astype(float))


def is_target_detected(distance: float, loa: float, beam: float, weather: Weather = Weather.SUNNY):
    size = (loa**2 + beam**2)**0.5
    p = get_detection_probability(size, distance, weather=weather)
    val = np.random.uniform(low=0, high=1)
    return bool(val <= p)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    d = np.linspace(0, 3000, 300)
    size = [80, 90, 100]
    weathers = [Weather.SUNNY, Weather.CLOUDY, Weather.FOGGY]
    color_map = {Weather.SUNNY: 'blue', Weather.CLOUDY: 'red', Weather.FOGGY: 'green'}
    linestyle_map = {size[0]: '-', size[1]: '--', size[2]: '-.'}
    xline = 1000

    for w in weathers:
        for s in size:
            p = get_detection_probability(s, d, w)
            ax.plot(d, p, c=color_map[w], linestyle=linestyle_map[s], label=f"{s} - {w}")
            print(f"Detected (w={w} - s={s}): ", is_target_detected(1000, s/np.sqrt(2), s/np.sqrt(2), weather=w))
    ax.vlines(xline, -2, 2, 'black', linestyles=':')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("detection probability [-]")
    plt.legend()
    plt.show()