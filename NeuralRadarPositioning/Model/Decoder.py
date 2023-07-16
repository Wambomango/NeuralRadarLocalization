import torch


class Decoder(torch.nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.samples_per_trajectory = config["training"]["samples_per_trajectory"]

        self.fc1 = torch.nn.Linear(24, 12)
        self.relu1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(12, 6)
        self.relu2 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(6, 4)
        self.relu3 = torch.nn.ReLU()

        self.fc4 = torch.nn.Linear(4, 3)

    def forward(self, x):
        x = x.reshape((-1, 24))
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)

        x = x.reshape((-1, self.samples_per_trajectory, 3))
        return x
