"""
File documentation:
This file defines the neural network architectures for the regression model.
It includes a simple one-hidden-layer MLP, a two-hidden-layer MLP, a two-layer CNN, and a three-layer CNN.
The MLPs are fully connected feedforward networks, while the CNN is designed to capture spatial patterns in the input data, which is useful for the atmospheric predictor variables that have a spatial structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class VisionTransformer(nn.Module):
    """
    Lightweight Vision Transformer for the same tabular-config interface as the CNN models.
    Input:  [batch, channels, height, width]
    Output: [batch, target_size]
    """
    def __init__(
        self,
        n_channels_input_cnn,
        target_size,
        image_size,
        patch_size=4,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.image_size = image_size  # Kept for config compatibility.

        patch_dim = n_channels_input_cnn * patch_size * patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, target_size)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _positional_encoding(self, seq_len, device, dtype):
        """
        Returns sinusoidal positional encoding [1, seq_len, embed_dim].
        """
        pe = torch.zeros(seq_len, self.embed_dim, device=device, dtype=dtype)
        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / self.embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        b, _, h, w = x.shape
        # Pad to the next patch boundary so arbitrary spatial sizes are supported.
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # [B, C*P*P, N] -> [B, N, C*P*P]
        patches = self.unfold(x).transpose(1, 2)
        tokens = self.patch_embed(patches)

        cls_token = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)

        pos = self._positional_encoding(tokens.shape[1], tokens.device, tokens.dtype)
        tokens = self.dropout(tokens + pos)

        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        out = self.head(cls_out)
        return out