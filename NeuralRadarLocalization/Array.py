import numpy as np
from scipy.spatial.transform import Rotation as R


class Array:
    def __init__(self, config, idx):
        self.frequency = config["beacon"]["frequency"]
        self.c = config["beacon"]["c"]
        self.wavelength = self.c / self.frequency

        self.id = config["arrays"][idx]["id"]
        self.position = np.array(config["arrays"][idx]["position"], dtype=np.float64)
        self.azimut = config["arrays"][idx]["azimut"]
        self.elevation = config["arrays"][idx]["elevation"]
        self.roll = config["arrays"][idx]["roll"]
        self.antennas = np.array(config["arrays"][idx]["antennas"], dtype=np.float64)
        self.n_antennas = self.antennas.shape[0]
        self.n_phase_differences = int(self.n_antennas * (self.n_antennas - 1) / 2)

        self.enable_phase_noise = config["arrays"][idx]["augmentations"][
            "enable_phase_noise"
        ]
        self.minimum_phase_variance = config["arrays"][idx]["augmentations"][
            "minimum_phase_variance"
        ]
        self.maximum_phase_variance = config["arrays"][idx]["augmentations"][
            "maximum_phase_variance"
        ]
        self.enable_pose_noise = config["arrays"][idx]["augmentations"][
            "enable_pose_noise"
        ]
        self.angle_variance = config["arrays"][idx]["augmentations"]["angle_variance"]
        self.position_variance = config["arrays"][idx]["augmentations"][
            "position_variance"
        ]

        self.rotation_matrix = (
            R.from_euler(
                "xyz", [self.roll, -self.elevation, self.azimut], degrees=True
            ).as_matrix()
            @ R.from_euler("y", [90], degrees=True).as_matrix()[0]
        )

        self.difference_matrix = np.zeros((self.n_phase_differences, self.n_antennas))
        row = 0
        for i in range(self.n_antennas):
            for j in range(i + 1, self.n_antennas):
                self.difference_matrix[row, i] = 1
                self.difference_matrix[row, j] = -1
                row += 1
        self.difference_matrix = self.difference_matrix.T

    def generate_measurements(self, positions):
        phases = np.zeros((positions.shape[0], positions.shape[1], self.n_antennas))

        offsets = np.repeat(self.position[None, :], positions.shape[0], 0)
        rotation_matrices = np.repeat(
            self.rotation_matrix[None, :, :], positions.shape[0], 0
        )
        if self.enable_pose_noise:
            offsets += np.random.normal(0, self.position_variance, offsets.shape)
            for i in range(positions.shape[0]):
                axis = np.random.uniform(0, 1, 3)
                axis *= np.random.normal(
                    0, np.sqrt(self.angle_variance)
                ) / np.linalg.norm(axis)
                rotation_matrices[i] = (
                    R.from_rotvec(axis).as_matrix() @ rotation_matrices[i]
                )
        for i in range(self.n_antennas):
            antenna = np.repeat(
                (rotation_matrices @ self.antennas[i] + offsets)[:, None, :],
                positions.shape[1],
                1,
            )
            phases[:, :, i] = (
                np.sqrt(np.sum((antenna - positions) ** 2, axis=2))
                / self.wavelength
                * 2
                * np.pi
            )

        if self.enable_phase_noise:
            phase_variances = np.repeat(
                np.repeat(
                    np.random.uniform(
                        self.minimum_phase_variance,
                        self.maximum_phase_variance,
                        positions.shape[0],
                    )[:, None, None],
                    positions.shape[1],
                    1,
                ),
                self.n_antennas,
                2,
            )

            phases += np.random.normal(0, np.sqrt(phase_variances))

        return np.mod((phases @ self.difference_matrix) + np.pi, 2 * np.pi) - np.pi
