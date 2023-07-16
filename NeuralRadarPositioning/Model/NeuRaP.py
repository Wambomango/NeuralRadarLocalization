import torch
from NeuralRadarPositioning.Model.Encoder import Encoder as Encoder
from NeuralRadarPositioning.Model.Decoder import Decoder as Decoder


class NeuRaP(torch.nn.Module):
    def __init__(self, config):
        super(NeuRaP, self).__init__()
        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]

        self.encoder = Encoder(config)
        self.lstm = torch.nn.LSTM(192, 24, batch_first=True)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.lstm(x)[0]
        x = self.decoder(x)

        return x
