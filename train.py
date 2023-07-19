import os
import datetime
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from NeuralRadarLocalization.Model.NeuRaL import NeuRaL as NeuRaL
from NeuralRadarLocalization.Generator.NumpyDataset import NumpyDataset


config_file = "./config.json"
experiment_name = "LSTM 5 Iterations 64 State Loss over all Steps"


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

model = NeuRaL(config).cuda()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)
writer = SummaryWriter(os.path.join("runs", experiment_name))


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

        loss = 0
        for i in range(model.lstm_steps):
            loss += torch.sum((estimated_trajectory[:, :, i, :] - trajectory) ** 2) * (
                0.9 ** (model.lstm_steps - i - 1)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for name, W in model.named_parameters():
                if "bias" in name:
                    W[torch.abs(W) < 0.2] = 0

        total_loss += loss

    total_loss = torch.sqrt(total_loss / len(train_dataloader)).item()
    writer.add_scalar("Loss/Train", total_loss, epoch)
    print("Training loss", total_loss)

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            measurements = batch["measurements"].cuda()
            trajectory = batch["trajectories"].cuda()

            estimated_trajectory = model(measurements)

            loss = loss_function(estimated_trajectory[:, :, -1, :], trajectory)

        total_loss += loss

        scheduler.step(total_loss)
        total_loss = torch.sqrt(total_loss / len(test_dataloader)).item()
        writer.add_scalar("Loss/Test", total_loss, epoch)
        print("Test loss", total_loss)

        if total_loss < best_val_loss:
            print("New best val loss")
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, "epoch_" + str(epoch) + ".pt"),
            )
            best_val_loss = total_loss


writer.flush()
writer.close()
