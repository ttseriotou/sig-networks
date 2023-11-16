from __future__ import annotations

import torch
import torch.nn as nn


class FeedforwardNeuralNetModel(nn.Module):
    """
    Feed-forward Neural Network model with ReLU activation layers for
    classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list[int] | tuple[int] | int,
        output_dim: int,
        dropout_rate: float,
    ):
        """
        Feed-forward Neural Network model with ReLU activation layers for
        classification.

        Parameters
        ----------
        input_dim : int
            Dimension of input layer.
        hidden_dim : list[int] | tuple[int] | int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of output layer.
        dropout_rate : float
            Probability of dropout.
        """
        super().__init__()

        if type(hidden_dim) == int:
            hidden_dim = [hidden_dim]
        self.hidden_dim = hidden_dim

        # FFN: input layer
        self.input_layer = nn.Linear(input_dim, self.hidden_dim[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        input_dim = self.hidden_dim[0]

        # FFN: hidden layers
        self.linear_layers = []
        self.non_linear_layers = []
        self.dropout_layers = []
        for layer in range(1, len(self.hidden_dim)):
            self.linear_layers.append(nn.Linear(input_dim, self.hidden_dim[layer]))
            self.non_linear_layers.append(nn.ReLU())
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            input_dim = self.hidden_dim[layer]

        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.non_linear_layers = nn.ModuleList(self.non_linear_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)

        # FFN: readout
        self.final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # FFN: input layer
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.dropout(out)

        # FFN: hidden layers
        for layer in range(len(self.linear_layers)):
            out = self.linear_layers[layer](out)
            out = self.non_linear_layers[layer](out)
            out = self.dropout_layers[layer](out)

        # FFN: readout
        out = self.final_layer(out)

        return out
