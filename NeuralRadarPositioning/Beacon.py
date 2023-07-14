from NeuralRadarPositioning.MotionModels.CTRA import CTRA as CTRA
from NeuralRadarPositioning.MotionModels.GaussianRandomWalk import GaussianRandomWalk as GaussianRandomWalk
import numpy as np

class Beacon:
    def __init__(self, config): 

        self.tracking_space = np.array(config["beacon"]["movement"]["tracking_space"]).T
        self.samples_per_trajectory = config["beacon"]["movement"]["samples_per_trajectory"]
        self.motion_models = {}
        self.motion_models["CTRA"] = CTRA(config)
        self.motion_models["GaussianRandomWalk"] = GaussianRandomWalk(config)

    def generate_trajectories(self):

        offsets = [0]
        for key in self.motion_models:
            offsets.append(offsets[-1] + self.motion_models[key].trajectories)

        trajectories = np.zeros((offsets[-1],self.samples_per_trajectory,3))

        i = 0
        for key in self.motion_models:
            trajectories[offsets[i]:offsets[i+1]] = self.motion_models[key].generate_trajectories()
            i += 1

        return trajectories