import json
import numpy as np

from NeuralRadarPositioning.Environment import Environment
from NeuralRadarPositioning.Beacon import Beacon

config_file = './config.json'

with open(config_file, 'r') as f:
    config = json.load(f)

environment = Environment(config)
beacon = Beacon(config)

trajectories = beacon.generate_trajectories()
measurements = environment.generate_measurements(trajectories)

