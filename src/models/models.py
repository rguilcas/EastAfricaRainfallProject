"""
File documentation:
This file defines the neural network architectures for the regression model.
It includes a simple one-hidden-layer MLP, a two-hidden-layer MLP, a two-layer CNN, and a three-layer CNN.
The MLPs are fully connected feedforward networks, while the CNN is designed to capture spatial patterns in the input data, which is useful for the atmospheric predictor variables that have a spatial structure.
"""

import torch.nn as nn

class OneHiddenLayerMLP(nn.Module):
    """
    This neural network architecture consists of a single hidden layer with ReLU activation.
    The input is first flattened, then passed through a fully connected layer, followed by a ReLU activation, and finally through another fully connected layer to produce the output.
    """
    def __init__(self, input_size, hidden_size_mlp, target_size):
        super(OneHiddenLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size_mlp)
        self.fc2 = nn.Linear(hidden_size_mlp, target_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TwoHiddenLayerMLP(nn.Module):
    """
    This neural network architecture consists of two hidden layers with ReLU activation.
    The input is first flattened, then passed through the first fully connected layer, followed by a ReLU activation, then through the second fully connected layer, followed by another ReLU activation, and finally through the third fully connected layer to produce the output.
    """
    def __init__(self, input_size, hidden_size_mlp, target_size):
        super(TwoHiddenLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size_mlp)
        self.fc2 = nn.Linear(hidden_size_mlp, hidden_size_mlp)
        self.fc3 = nn.Linear(hidden_size_mlp, target_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
class TwoLayerCNN(nn.Module):
    """
    This neural network architecture consists of two convolutional layers followed by a fully connected prediction head.
    The convolutional layers are designed to capture spatial patterns in the input data, which is useful for the atmospheric predictor variables that have a spatial structure.
    The input is passed through two convolutional blocks, each consisting of a convolutional layer, a ReLU activation, and a max pooling layer. 
    The output of the convolutional layers is then flattened and passed through a fully connected prediction head to produce the final output.
    """
    def __init__(self, n_channels_input_cnn, target_size, image_size,
                 n_kernels_cnn=8, hidden_size_mlp=64):
        super(TwoLayerCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(n_channels_input_cnn, n_kernels_cnn, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n_kernels_cnn, n_kernels_cnn*2, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Infer the flattened size at first forward to support non-divisible image sizes.
        self.fc1 = nn.LazyLinear(hidden_size_mlp)
        self.fc2 = nn.Linear(hidden_size_mlp, target_size)

    def forward(self, x):
        # Convolutional encoder part
        ## Block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        ## Block 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        # Flatten
        out = self.flatten(out)
        # Prediction head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ThreeLayerCNN(nn.Module):
    """
    This neural network architecture consists of three convolutional layers followed by a fully connected prediction head.
    The convolutional blocks capture increasingly abstract spatial features from the atmospheric predictor maps.
    The flattened convolution output is passed to a small MLP head to produce the final regression output.
    """
    def __init__(self, n_channels_input_cnn, target_size, image_size,
                 n_kernels_cnn=8, hidden_size_mlp=64):
        super(ThreeLayerCNN, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(n_channels_input_cnn, n_kernels_cnn, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(n_kernels_cnn, n_kernels_cnn * 2, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(n_kernels_cnn * 2, n_kernels_cnn * 4, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # Infer flattened size at first forward for robust spatial-size handling.
        self.fc1 = nn.LazyLinear(hidden_size_mlp)
        self.fc2 = nn.Linear(hidden_size_mlp, target_size)

    def forward(self, x):
        # Convolutional encoder part
        ## Block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        ## Block 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        ## Block 3
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        # Flatten
        out = self.flatten(out)
        # Prediction head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out