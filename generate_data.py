import json
import numpy as np

from NeuralRadarPositioning.Environment import Environment
from NeuralRadarPositioning.Beacon import Beacon

config_file = "./config.json"

with open(config_file, "r") as f:
    config = json.load(f)

environment = Environment(config)
beacon = Beacon(config)

print("Generating training data")
training_trajectories = beacon.generate_trajectories(
    config["training"]["training_trajectories"]
)
training_measurements = environment.generate_measurements(training_trajectories)
training_data_file = config["training"]["training_data"]
np.savez(
    training_data_file,
    trajectories=training_trajectories,
    measurements=training_measurements,
)

print("Generating test data")
test_trajectories = beacon.generate_trajectories(
    config["training"]["test_trajectories"]
)
test_measurements = environment.generate_measurements(test_trajectories)
test_data_file = config["training"]["test_data"]
np.savez(test_data_file, trajectories=test_trajectories, measurements=test_measurements)
