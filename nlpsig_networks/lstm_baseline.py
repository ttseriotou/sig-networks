from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class LSTMModel(nn.Module):
    """
    LSTM network model for  classification.
    """
    
    def __init__(
        self,
        input_dim: int, 
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        output_dim: int,
        dropout_rate: float
    ):
        """
        LSTM network model for  classification.

        Parameters
        ----------
        input_dim : int
            The number of expected features in the input x
        hidden_dim : int
            Dimensions of the hidden layers in the LSTM blocks.
        num_layers : int
            Number of recurrent layers.
        bidirectional : bool
            Whether or not a birectional LSTM is used,
            by default False (unidirectional LSTM is used in this case).
        output_dim : int
            Dimension of output layer.
        dropout_rate : float
            Probability of dropout.
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional).double()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor):
        # x has dimensions [batch, length of signal, channels]
        # assume empty units are padded using zeros and padded from below
        # find length of paths by finding how many non-zero rows there are
        seq_lengths = torch.sum(torch.sum(x, 2) != 0, 1)
        # sort sequences by length in a decreasing order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]

        # pack a tensor containing padded sequences of variable length
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=seq_lengths,
            batch_first=True
        )
        
        # pass through LSTM
        out, (out_h, _) = self.lstm(x_pack)
        
        # obtain last hidden states
        if self.bidirectional:
            # element-wise add if have BiLSTM
            out = out_h[-1, :, :] + out_h[-2, :, :]
        else:
            out = out_h[-1, :, :]
        
        # need to reverse the original indexing afterwards
        inverse_perm = np.argsort(perm_idx)
        out = out[inverse_perm]

        # readout
        out = self.dropout(out)
        out = self.fc(out.float())
        
        return out