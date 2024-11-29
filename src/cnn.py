import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(
        self,
        sequence_length=18,
        n_features=23,
        filters_layer1=4,
        filters_layer2=8,
        filters_layer3=16,
        droprate=0.1,
    ):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, filters_layer1, kernel_size=(1, n_features))
        self.conv2 = nn.Conv2d(filters_layer1, filters_layer2, kernel_size=(3, 1))
        self.conv3 = nn.Conv2d(filters_layer2, filters_layer3, kernel_size=(3, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1))
        self.dropout = nn.Dropout(droprate)
        self.flatten = nn.Flatten()

        dummy_input = torch.zeros(1, 1, sequence_length, n_features)
        output_size = self.getShapeOfLinearLayer(dummy_input)

        self.fc = nn.Linear(output_size, 1)

    def getShapeOfLinearLayer(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)

        return x.size(1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x
