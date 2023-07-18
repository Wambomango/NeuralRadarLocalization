import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from NeuralRadarLocalization.MotionModels.MotionModel import MotionModel


class CTRA(MotionModel):
    def __init__(self, config):
        super().__init__()
        self.tracking_space = np.array(config["beacon"]["movement"]["tracking_space"])
        self.sampling_rate = config["beacon"]["movement"]["sampling_rate"]
        self.proportion = config["beacon"]["movement"]["motion_models"]["CTRA"][
            "proportion"
        ]
        self.minimum_initial_velocity = config["beacon"]["movement"]["motion_models"][
            "CTRA"
        ]["minimum_initial_velocity"]
        self.maximum_initial_velocity = config["beacon"]["movement"]["motion_models"][
            "CTRA"
        ]["maximum_initial_velocity"]
        self.minimum_acceleration = config["beacon"]["movement"]["motion_models"][
            "CTRA"
        ]["minimum_acceleration"]
        self.maximum_acceleration = config["beacon"]["movement"]["motion_models"][
            "CTRA"
        ]["maximum_acceleration"]
        self.minimum_turn_rate = config["beacon"]["movement"]["motion_models"]["CTRA"][
            "minimum_turn_rate"
        ]
        self.maximum_turn_rate = config["beacon"]["movement"]["motion_models"]["CTRA"][
            "maximum_turn_rate"
        ]
        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]

        self.SIMULATION_OVERSAMPLING = 1
        self.time_step = 1 / (self.sampling_rate * self.SIMULATION_OVERSAMPLING)
        self.trajectory_duration = self.samples_per_trajectory / self.sampling_rate
        self.trajectory_steps = int(np.ceil(self.trajectory_duration / self.time_step))

    def generate_trajectories(self, n_trajectories=1):
        print("Generating CTRA trajectories")

        trajectories = np.zeros((n_trajectories, self.samples_per_trajectory, 3))
        for i in tqdm(range(n_trajectories)):
            state_history = np.zeros((self.trajectory_steps, 6))
            state_history[0, 3] = np.random.uniform(
                self.minimum_initial_velocity, self.maximum_initial_velocity
            )
            state_history[0, 4] = np.random.uniform(
                self.minimum_acceleration, self.maximum_acceleration
            )
            state_history[0, 5] = np.random.uniform(
                self.minimum_turn_rate, self.maximum_turn_rate
            )

            for j in range(1, self.trajectory_steps):
                x_j_1 = state_history[j - 1, 0]
                y_j_1 = state_history[j - 1, 1]
                phi_j_1 = state_history[j - 1, 2]
                v_j_1 = state_history[j - 1, 3]
                a_j_1 = state_history[j - 1, 4]
                omega_j_1 = state_history[j - 1, 5]

                phi_j = phi_j_1 + omega_j_1 * self.time_step
                v_j = v_j_1 + a_j_1 * self.time_step
                a_j = a_j_1
                omega_j = omega_j_1
                x_j = (
                    x_j_1
                    + (v_j * np.sin(phi_j) - v_j_1 * np.sin(phi_j_1)) / omega_j
                    + a_j * (np.cos(phi_j) - np.cos(phi_j_1)) / omega_j_1**2
                )
                y_j = (
                    y_j_1
                    - (v_j * np.cos(phi_j) - v_j_1 * np.cos(phi_j_1)) / omega_j
                    + a_j * (np.sin(phi_j) - np.sin(phi_j_1)) / omega_j_1**2
                )

                state_history[j, 0] = x_j
                state_history[j, 1] = y_j
                state_history[j, 2] = phi_j
                state_history[j, 3] = v_j
                state_history[j, 4] = a_j
                state_history[j, 5] = omega_j

            flat_trajectory = np.zeros((self.samples_per_trajectory, 3))

            for j in range(self.samples_per_trajectory):
                flat_trajectory[j, 0] = state_history[
                    j * self.SIMULATION_OVERSAMPLING, 0
                ]
                flat_trajectory[j, 1] = state_history[
                    j * self.SIMULATION_OVERSAMPLING, 1
                ]

            offset = np.random.uniform(self.tracking_space[0], self.tracking_space[1])
            axis = np.random.uniform(0, 1, 3)
            axis *= np.random.uniform(0, 2 * np.pi) / np.linalg.norm(axis)
            rotation_matrix = R.from_rotvec(axis).as_matrix()

            trajectories[i] = flat_trajectory @ rotation_matrix.T + offset

        return trajectories
