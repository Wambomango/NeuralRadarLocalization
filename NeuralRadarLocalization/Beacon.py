from NeuralRadarLocalization.MotionModels.CTRA import CTRA as CTRA
from NeuralRadarLocalization.MotionModels.GaussianRandomWalk import (
    GaussianRandomWalk as GaussianRandomWalk,
)
import numpy as np


class Beacon:
    def __init__(self, config):
        self.tracking_space = np.array(config["beacon"]["movement"]["tracking_space"]).T
        self.motion_models = {}
        self.CTRA = CTRA(config)
        self.GaussianRandomWalk = GaussianRandomWalk(config)
        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]
        self.training_trajectories = config["training"]["training_trajectories"]
        self.test_trajectories = config["training"]["test_trajectories"]

    def generate_trajectories(self, n_trajectories=1):
        n_ctra = int(np.ceil(self.CTRA.proportion * n_trajectories))
        n_grw = int(np.ceil(self.CTRA.proportion * n_trajectories))

        trajectories = np.zeros((n_ctra + n_grw, self.samples_per_trajectory, 3))
        trajectories[:n_ctra] = self.CTRA.generate_trajectories(n_ctra)
        trajectories[n_ctra:] = self.GaussianRandomWalk.generate_trajectories(n_grw)
        return trajectories[:n_trajectories]
