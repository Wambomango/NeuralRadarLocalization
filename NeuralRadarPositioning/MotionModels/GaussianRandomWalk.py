import numpy as np
from tqdm import tqdm
from NeuralRadarPositioning.MotionModels.MotionModel import MotionModel

class GaussianRandomWalk(MotionModel):
    def __init__(self, config): 
        super().__init__()
        self.tracking_space = np.array(config["beacon"]["movement"]["tracking_space"])
        self.sampling_rate = config["beacon"]["movement"]["sampling_rate"]
        self.proportion = config["beacon"]["movement"]["motion_models"]["GaussianRandomWalk"]["proportion"]
        self.minimum_variance = config["beacon"]["movement"]["motion_models"]["GaussianRandomWalk"]["minimum_variance"] / self.sampling_rate
        self.maximum_variance = config["beacon"]["movement"]["motion_models"]["GaussianRandomWalk"]["maximum_variance"] / self.sampling_rate
        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]
         

    def generate_trajectories(self, n_trajectories = 1):
        print("Generating Gaussian Random Walk trajectories")

        trajectories = np.zeros((n_trajectories, self.samples_per_trajectory, 3))
        for i in tqdm(range(n_trajectories)):
            point = np.random.uniform(self.tracking_space[0], self.tracking_space[1])
            standard_deviation = np.sqrt(np.random.uniform(self.minimum_variance, self.maximum_variance))

            for j in range(self.samples_per_trajectory):
                trajectories[i,j] = point
                point += np.random.normal(0, standard_deviation, 3)

        return trajectories

