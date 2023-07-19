import torch


class NeuRaL(torch.nn.Module):
    def __init__(self, config):
        super(NeuRaL, self).__init__()

        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]
        self.feature_size = 0
        for array in config["arrays"]:
            n_antennas = len(array["antennas"])
            self.feature_size += int(n_antennas * (n_antennas - 1) / 2)

        self.lstm_steps = 5
        self.lstm_state_size = 64
        self.LSTM = torch.nn.LSTM(
            self.feature_size, self.lstm_state_size, batch_first=True
        )

        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_state_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3),
        )

    def forward(self, x):
        x = x.reshape((x.shape[0] * self.samples_per_trajectory, 1, self.feature_size))
        x = x.repeat((1, self.lstm_steps, 1), (0, 1, 2))

        x = self.LSTM(x)[0]

        x = self.block1(x)
        x = x.reshape((-1, self.samples_per_trajectory, self.lstm_steps, 3))

        return x
