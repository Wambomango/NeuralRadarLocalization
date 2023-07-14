import json
import numpy as np
import matplotlib.pyplot as plt
from NeuralRadarPositioning.MotionModels.CTRA import CTRA
from NeuralRadarPositioning.MotionModels.GaussianRandomWalk import GaussianRandomWalk




config_file = './config.json'
trajectories_per_model = 10



with open(config_file, 'r') as f:
    config = json.load(f)

ctra = CTRA(config)
grw = GaussianRandomWalk(config)

ctra_trajectories = ctra.generate_trajectories(trajectories_per_model)
grw_trajectories = grw.generate_trajectories(trajectories_per_model)

ax = plt.figure().add_subplot(projection='3d')
tracking_space = np.array(config["beacon"]["movement"]["tracking_space"])
minimum_coordinates = np.min(tracking_space, axis = 0)
maximum_coordinates = np.max(tracking_space, axis = 0)

for i in range(trajectories_per_model):
    ax.scatter(ctra_trajectories[i,:,0], ctra_trajectories[i,:,1], ctra_trajectories[i,:,2])
    ax.scatter(grw_trajectories[i,:,0], grw_trajectories[i,:,1], grw_trajectories[i,:,2])

ax.legend()
ax.set_xlim3d(minimum_coordinates[0], maximum_coordinates[0])
ax.set_ylim3d(minimum_coordinates[1], maximum_coordinates[1])
ax.set_zlim3d(minimum_coordinates[2], maximum_coordinates[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()