from __future__ import annotations
from signatory import Signature, LogSignature, signature_channels, logsignature_channels
import torch
import torch.nn as nn


class SWNU(nn.Module):
    """
    Signature Window Network Unit.
    """
    
    def __init__(
        self,
        input_size: int,
        log_signature: bool,
        sig_depth: int,
        hidden_dim: list[int] | int,
        BiLSTM: bool = False,
    ):
        """
        Applies a multi-layer Signature Window Network Unit (SWNU) to
        an input sequence.

        Parameters
        ----------
        input_size : int
            The number of expected features in the input x.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        hidden_dim : list[int] | int
            Dimensions of the hidden layers in the LSTM blocks in the SWNU.
        BiLSTM : bool, optional
            Whether or not a birectional LSTM is used for the final SWNU block,
            by default False (unidirectional LSTM is used in this case).
        """
        # logging inputs to the class
        self.input_size = input_size
        self.log_signature = log_signature
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        self.hidden_dim = hidden_dim
        self.BiLSTM = BiLSTM
        
        # creating expanding window signature layers and corresponding LSTM layers
        self.signature_layers = []
        self.lstm_layers = []
        for l in range(len(self.hidden_dim)):
            # create expanding window signature layer and compute the input dimension to LSTM
            if self.log_signature:    
                self.signature_layers.append(LogSignature(depth=sig_depth, stream=True))
                if l == 0:
                    input_dim_lstm = logsignature_channels(in_channels=input_size,
                                                           depth=sig_depth)
                else:
                    input_dim_lstm = logsignature_channels(in_channels=self.hidden_dim[l-1],
                                                           depth=sig_depth)
            else:
                self.signature_layers.append(Signature(depth=sig_depth, stream=True))
                if l == 0:
                    input_dim_lstm = signature_channels(channels=input_size,
                                                        depth=sig_depth)
                else:
                    input_dim_lstm = signature_channels(channels=self.hidden_dim[l-1],
                                                        depth=sig_depth)
            
            # create LSTM layer (if last layer, this can be a BiLSTM)
            self.lstm_layers.append(nn.LSTM(
                input_size=input_dim_lstm,
                hidden_size=self.hidden_dim[l],
                num_layers=1,
                batch_first=True,
                bidirectional=False if l!=(len(self.hidden_dim)-1) else self.BiLSTM,
            ))
        
        # make a ModuleList from the signatures and LSTM layers
        self.signature_layers = nn.ModuleList(self.signature_layers)
        self.lstm_layers = nn.ModuleList(self.lstm_layers)

        # final signature without lift (i.e. no expanding windows)
        if self.log_signature:
            self.signature2 = LogSignature(depth=sig_depth, stream=False)
        else:
            self.signature2 = Signature(depth=sig_depth, stream=False)
            
    def forward(self, x: torch.Tensor):
        # x has dimensions [batch, length of signal, channels]
        
        # take signature lifts and lstm
        for l in range(len(self.hidden_dim)):
            out = self.signature_layers[l](out)
            out, _ = self.lstm_layers[l](out)
        
        # take final signature
        out = self.signature2(out)
        
        return out