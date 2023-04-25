from __future__ import annotations
import torch
import torch.nn as nn


class FeedforwardNeuralNetModel(nn.Module):
    """
    Feed-forward Neural Network model with ReLU activation layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_ffn: list[int] | tuple[int] | int,
        output_dim: int,
        dropout_rate: float
    ):
        """
        Feed-forward Neural Network model with ReLU activation layers.

        Parameters
        ----------
        input_dim : int
            Dimension of input layer.
        hidden_dim_ffn : list[int] | tuple[int] | int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of output layer.
        dropout_rate : float
            Probability of dropout.
        """
        super(FeedforwardNeuralNetModel, self).__init__()
        if type(hidden_dim_ffn) == int:
            hidden_dim_ffn = [hidden_dim_ffn]
        self.hidden_dim_ffn = hidden_dim_ffn
        
        # FNN: input layer
        self.ffn_input_layer = nn.Linear(input_dim, self.hidden_dim_ffn[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        input_dim = self.hidden_dim_ffn[0]
        
        # FNN: hidden layers
        self.ffn_linear_layers = []
        self.ffn_non_linear_layers = []
        self.dropout_layers = []
        for l in range(len(self.hidden_dim_ffn)):
            self.ffn_linear_layers.append(nn.Linear(input_dim, self.hidden_dim_ffn[l]))
            self.ffn_non_linear_layers.append(nn.ReLU())
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            input_dim = self.hidden_dim_ffn[l]
        
        self.ffn_linear_layers = nn.ModuleList(self.ffn_linear_layers)
        self.ffn_non_linear_layers = nn.ModuleList(self.ffn_non_linear_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        
        # FNN: readout
        self.ffn_final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # FFN: input layer
        out = self.ffn_input_layer(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # FFN: hidden layers    
        for l in range(len(self.hidden_dim_ffn)):
            out = self.ffn_linear_layers[l](out)
            out = self.ffn_non_linear_layers[l](out)
            out = self.dropout_layers[l](out)

        # FFN: readout
        out = self.ffn_final_layer(out)

        return out
