import torch


class EncoderHead(torch.nn.Module):
    def __init__(self, config, feature_size):
        super(EncoderHead, self).__init__()

        self.feature_size = feature_size

        self.conv1 = torch.nn.Conv1d(
            1, 128, self.feature_size, padding="same", padding_mode="zeros"
        )
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv1d(
            128,
            256,
            self.feature_size,
            padding="same",
            padding_mode="zeros",
        )
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv1d(
            256,
            128,
            self.feature_size,
            padding="same",
            padding_mode="zeros",
        )
        self.relu3 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(self.feature_size * 128, 32)

    def forward(self, x):
        x = self.conv1(x[:, None, :])
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.reshape((-1, self.feature_size * 128))
        x = self.fc1(x)
        x = x.reshape(-1, 32)

        return x


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]

        self.feature_sizes = []
        self.heads = torch.nn.ModuleList()
        for array in config["arrays"]:
            n_antennas = len(array["antennas"])
            feature_size = int(n_antennas * (n_antennas - 1) / 2)
            self.heads.append(EncoderHead(config, feature_size))
            self.feature_sizes.append(feature_size)

        self.conv1 = torch.nn.Conv1d(len(self.heads), 32, 21, padding="same")
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv1d(32, 128, 21, padding="same")
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv1d(128, 32, 21, padding="same")
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        y = torch.zeros(
            (x.shape[0] * x.shape[1], len(self.heads), 32),
            device=x.device,
        )

        offset = 0
        for i in range(len(self.heads)):
            feature_size = self.feature_sizes[i]
            x_array = x[:, :, offset : offset + feature_size].reshape(
                (-1, feature_size)
            )
            y[:, i, :] = self.heads[i](x_array)
            offset += feature_size

        y = self.conv1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.relu2(y)

        y = self.conv3(y)
        y = self.relu3(y)

        return y
