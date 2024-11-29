from src.custom_data_utils import getDataset
from src.cnn import CNN

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


def train_evaluate(parameters):
    sequence_length = parameters["sequence_length"]
    n_epochs = parameters["num_epochs"]
    learning_rate = parameters["lr"]

    train_dataset = getDataset("train", sequence_length)
    train_set = train_dataset[0]

    test_df = getDataset("test", sequence_length)
    test_set = test_df[0]

    train_steps_per_epoch = len(train_set[1])
    test_steps_per_epoch = len(test_set[1])

    model = CNN(sequence_length=sequence_length)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mse = nn.L1Loss()
    writer = SummaryWriter()

    epoch_loss = list()
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        indices = np.arange(train_steps_per_epoch)
        np.random.shuffle(indices)
        for idx in indices:
            optimizer.zero_grad()

            input_sequence = train_set[0][idx].unsqueeze(0).unsqueeze(0)
            target = train_set[1][idx]

            outputs = model(input_sequence)

            loss = criterion_mse(outputs.squeeze(), target)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                writer.add_scalar(
                    "Loss/train", loss.item(), epoch * train_steps_per_epoch + idx
                )

        avg_train_loss = running_loss / train_steps_per_epoch
        epoch_loss.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), f"model_runs/model_epoch_{epoch+1}.pth")
    writer.close()

    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        num_samples = 0
        indices = np.arange(test_steps_per_epoch)
        np.random.shuffle(indices)
        for idx in indices:
            input_sequence = test_set[0][idx].unsqueeze(0).unsqueeze(0)
            target = test_set[1][idx]

            output = model(input_sequence)
            output = output.squeeze(0).squeeze(0)

            loss = criterion_mse(output, target)
            total_loss += loss.item()
            num_samples += input_sequence.size(0)

        avg_loss = total_loss / num_samples
        print(f"Average Loss on Test Set: {avg_loss:.4f}")

    return avg_loss


train_evaluate({"lr": 0.001, "num_epochs": 10, "sequence_length": 18})
