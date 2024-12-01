from src.custom_data_utils import getDataset
from src.cnn import CNN

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


def train_evaluate(parameters):
    torch.manual_seed(42)

    sequence_length = parameters["sequence_length"]
    n_epochs = parameters["num_epochs"]
    learning_rate = parameters["lr"]
    droprate = parameters["droprate"]
    filters_layer1 = parameters["filters_layer1"]
    filters_layer2 = parameters["filters_layer2"]
    filters_layer3 = parameters["filters_layer3"]

    # Load datasets
    train_dataset = getDataset("train", sequence_length)
    test_dataset = getDataset("test", sequence_length)

    # Split train dataset into train and validation (90% train, 10% validation)
    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_steps_per_epoch = len(train_subset)
    val_steps_per_epoch = len(val_subset)
    test_steps_per_epoch = len(test_dataset)

    model = CNN(
        sequence_length=sequence_length,
        droprate=droprate,
        filters_layer1=filters_layer1,
        filters_layer2=filters_layer2,
        filters_layer3=filters_layer3,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    writer = SummaryWriter()

    epoch_loss = list()
    val_loss_history = list()

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for idx in range(train_steps_per_epoch):
            optimizer.zero_grad()

            input_sequence = train_subset[idx][0].unsqueeze(0).unsqueeze(0)
            target = train_subset[idx][1]

            outputs = model(input_sequence)

            loss = criterion_mae(outputs.squeeze(), target)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                writer.add_scalar(
                    "Loss/train", loss.item(), epoch * train_steps_per_epoch + idx
                )

        avg_train_loss = running_loss / train_steps_per_epoch
        epoch_loss.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for idx in range(val_steps_per_epoch):
                input_sequence = val_subset[idx][0].unsqueeze(0).unsqueeze(0)
                target = val_subset[idx][1]

                output = model(input_sequence)
                output = output.squeeze(0).squeeze(0)

                loss = criterion_mae(output, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / val_steps_per_epoch
        val_loss_history.append(avg_val_loss)

        # Log validation loss to TensorBoard
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)

        print(
            f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

    # torch.save(model.state_dict(), f"model_runs/model_epoch_{epoch+1}.pth")
    writer.close()

    # Testing phase
    model.eval()
    total_test_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for idx in range(test_steps_per_epoch):
            input_sequence = test_dataset[idx][0].unsqueeze(0).unsqueeze(0)
            target = test_dataset[idx][1]

            output = model(input_sequence)
            output = output.squeeze(0).squeeze(0)

            loss = criterion_mse(output, target)
            total_test_loss += loss.item()
            num_samples += input_sequence.size(0)

    avg_test_loss = total_test_loss / num_samples
    print(f"Average Loss on Test Set: {avg_test_loss:.4f}")

    return avg_test_loss


# Example usage
avg_test_loss, epoch_loss, val_loss_history = train_evaluate(
    {
        "lr": 1e-4,
        "num_epochs": 50,
        "sequence_length": 161,
        "droprate": 0.2,
        "filters_layer1": 32,
        "filters_layer2": 1,
        "filters_layer3": 1,
    }
)

# Plot training and validation loss curves
plt.plot(epoch_loss, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.savefig("loss_curve.png")
