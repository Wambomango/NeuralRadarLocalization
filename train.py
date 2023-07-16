import os
import datetime
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from NeuralRadarPositioning.Model.NeuRaP import NeuRaP as NeuRaP
from NeuralRadarPositioning.Generator.NumpyDataset import NumpyDataset

config_file = "./config.json"


with open(config_file, "r") as f:
    config = json.load(f)


training_data = NumpyDataset(config["training"]["training_data"])
test_data = NumpyDataset(config["training"]["test_data"])

train_dataloader = DataLoader(
    training_data, batch_size=config["training"]["batch_size"], shuffle=True
)
test_dataloader = DataLoader(
    test_data, batch_size=config["training"]["batch_size"], shuffle=True
)

model = NeuRaP(config).cuda()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), config["training"]["learning_rate"])


checkpoint_path = os.path.join(
    config["training"]["checkpoints"],
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
)
os.makedirs(checkpoint_path, exist_ok=True)


best_val_loss = float("inf")
for epoch in range(config["training"]["epochs"]):
    print("EPOCH :", epoch)

    model.train(True)
    total_loss = 0
    for batch in train_dataloader:
        measurements = batch["measurements"].cuda()
        trajectory = batch["trajectories"].cuda()

        estimated_trajectory = model(measurements)

        loss = loss_function(estimated_trajectory, trajectory)
        loss.backward()

        optimizer.step()

        total_loss += loss

    print("Training loss", torch.sqrt(total_loss / len(train_dataloader)))

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            measurements = batch["measurements"].cuda()
            trajectory = batch["trajectories"].cuda()

            estimated_trajectory = model(measurements)

            loss = loss_function(estimated_trajectory, trajectory)

            total_loss += loss

        print("Test loss", torch.sqrt(total_loss / len(test_dataloader)))

        if total_loss < best_val_loss:
            print("New best val loss")
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, "epoch_" + str(epoch) + ".pt"),
            )
            best_val_loss = total_loss
