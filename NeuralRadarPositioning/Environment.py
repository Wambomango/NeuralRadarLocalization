import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from NeuralRadarPositioning.Array import Array

class Environment:
    def __init__(self, config):
        self.tracking_space = np.array(config["beacon"]["movement"]["tracking_space"])
        self.frequency = config["beacon"]["frequency"]
        self.c = config["beacon"]["c"]
        self.wavelength = self.c / self.frequency
        self.arrays = []
        for array_config in config["arrays"]:
             self.arrays.append(Array(array_config))

        self.n_arrays = len(self.arrays)
        self.n_antennas = 0
        for array in self.arrays:
            self.n_antennas += array.n_antennas        

    def plot_arrays(self):

        ax = plt.figure().add_subplot(projection='3d')

        minimum_coordinates = np.min(self.tracking_space, axis = 0)
        maximum_coordinates = np.max(self.tracking_space, axis = 0)

        for array in self.arrays:
            minimum_coordinates_array = np.min(array.antennas, axis = 0)
            maximum_coordinates_array = np.max(array.antennas, axis = 0)

            minimum_coordinates = np.minimum(minimum_coordinates, minimum_coordinates_array)
            maximum_coordinates = np.maximum(maximum_coordinates, maximum_coordinates_array)

            ax.scatter(array.antennas[:,0], array.antennas[:,1], array.antennas[:,2], label='Array ' + str(array.id))

        ax.legend()
        ax.set_xlim3d(minimum_coordinates[0], maximum_coordinates[0])
        ax.set_ylim3d(minimum_coordinates[1], maximum_coordinates[1])
        ax.set_zlim3d(minimum_coordinates[2], maximum_coordinates[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def generate_measurements(self, positions):
        print("Generating measurements")
        measurements = np.zeros((positions.shape[0], positions.shape[1], self.n_antennas - self.n_arrays))
        offset = 0
        for i in tqdm(range(self.n_arrays)):
            array = self.arrays[i]
            phases = np.zeros((positions.shape[0], positions.shape[1], array.n_antennas))
            for j in range(array.n_antennas):
                phases[:,:,j] = np.sqrt(np.sum((array.antennas[j] - positions)**2, axis = 2)) / self.wavelength * 2 * np.pi
            measurements[:,:,offset:offset + array.n_antennas - 1] = np.mod((phases - phases[:,:,0,None])[:,:,1:] + np.pi, 2 * np.pi) - np.pi
            offset += array.n_antennas - 1

        return measurements

