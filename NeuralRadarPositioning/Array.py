import numpy as np
from scipy.spatial.transform import Rotation as R

class Array:
    def __init__(self, config):
        self.id = config["id"]
        self.position = np.array(config["position"])
        self.azimut = config["azimut"]
        self.elevation = config["elevation"]
        self.roll = config["roll"]
        self.antennas = np.array(config["antennas"])
        self.n_antennas = self.antennas.shape[0]

        rotation_matrix = R.from_euler('xyz', [self.roll, -self.elevation, self.azimut], degrees=True).as_matrix() @ R.from_euler('y', [90], degrees=True).as_matrix()[0]
        self.antennas = self.antennas @ rotation_matrix.T + self.position

