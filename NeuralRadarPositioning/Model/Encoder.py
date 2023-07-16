import torch


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]
        self.feature_size = 0
        for array_config in config["arrays"]:
            self.feature_size += len(array_config["antennas"]) - 1

        self.conv1 = torch.nn.Conv1d(
            1, 128, self.feature_size, padding="same", padding_mode="circular"
        )
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv1d(
            128,
            256,
            self.feature_size,
            padding="same",
            padding_mode="circular",
        )
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv1d(
            256,
            128,
            self.feature_size,
            padding="same",
            padding_mode="circular",
        )
        self.relu3 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(self.feature_size * 128, 192)
        self.relu4 = torch.nn.ReLU()

    def forward(self, x):
        x = x.reshape((-1, 1, self.feature_size))

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.reshape((-1, self.feature_size * 128))
        x = self.fc1(x)
        x = self.relu4(x)
        x = x.reshape(-1, self.samples_per_trajectory, 192)

        return x
