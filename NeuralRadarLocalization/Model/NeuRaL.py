import torch
from NeuralRadarLocalization.Model.Encoder import Encoder as Encoder


class NeuRaL(torch.nn.Module):
    def __init__(self, config):
        super(NeuRaL, self).__init__()

        self.encoder = Encoder(config)

        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256)
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64)
        )

        self.block3 = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        encoding = self.encoder(x).reshape((x.shape[0], x.shape[1], 1024))
        encoding = self.block1(encoding)
        encoding = self.block2(encoding)
        y = self.block3(encoding)

        return y
