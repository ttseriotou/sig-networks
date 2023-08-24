from __future__ import annotations
import signatory
import torch
import torch.nn as nn
import numpy as np
from nlpsig_networks.swnu import SWNU
from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel
from nlpsig_networks.feature_concatenation import FeatureConcatenation


class SeqSigNet(nn.Module):
    """
    BiLSTM of Deep Signature Neural Network Units for classification.
    """
    
    def __init__(
        self, 
        input_channels: int, 
        num_features: int, 
        embedding_dim: int, 
        log_signature: bool,
        sig_depth: int, 
        hidden_dim_swnu: list[int] | int,
        hidden_dim_lstm: int,
        hidden_dim_ffn: list[int] | int,
        output_dim: int, 
        dropout_rate: float, 
        output_channels: int | None = None,
        augmentation_type: str = 'Conv1d', 
        hidden_dim_aug: list[int] | int | None = None,
        BiLSTM: bool = False,
        comb_method: str ='concatenation'):
        """
        SeqSigNet network for classification.
        
        Input data will have the size: [batch size, window size (w), all embedding dimensions (history + time + post), unit size (n)]
        Note: unit sizes will be in reverse chronological order, starting from the more recent and ending with the one further back in time.
        
        Parameters
        ----------
        input_channels : int
            Dimension of the (dimensonally reduced) history embeddings that will be passed in. 
        num_features : int
            Number of time features to add to FFN input. If none, set to zero.
        embedding_dim: int
            Dimensions of current BERT post embedding. Usually 384 or 768.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        hidden_dim_swnu : list[int] | int
            Dimensions of the hidden layers in the SNWU blocks.
        hidden_dim_lstm : int
            Dimensions of the hidden layers in the final BiLSTM applied to the output
            of the SWNU units.
        hidden_dim_ffn : list[int] | int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.
        output_channels : int | None, optional
            Requested dimension of the embeddings after convolution layer.
            If None, will be set to the last item in `hidden_dim`, by default None.
        augmentation_type : str, optional
            Method of augmenting the path, by default "Conv1d".
            Options are:
            - "Conv1d": passes path through 1D convolution layer.
            - "signatory": passes path through `Augment` layer from `signatory` package.
        hidden_dim_aug : list[int] | int | None
            Dimensions of the hidden layers in the augmentation layer.
            Passed into `Augment` class from `signatory` package if
            `augmentation_type='signatory'`, by default None.
        BiLSTM : bool, optional
            Whether or not a birectional LSTM is used in the SWNUs,
            by default False (unidirectional LSTM is used in this case).
        comb_method : str, optional
            Determines how to combine the path signature and embeddings,
            by default "gated_addition".
            Options are:
            - concatenation: concatenation of path signature and embedding vector
            - gated_addition: element-wise addition of path signature and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature and embedding vector
            - scaled_concatenation: concatenation of single value scaled path signature and embedding vector
        """

        super(SeqSigNet, self).__init__()
        
        self.input_channels = input_channels

        self.swnu = SWNU(input_channels=input_channels,
                         output_channels=output_channels,
                         log_signature=log_signature,
                         sig_depth=sig_depth,
                         hidden_dim=hidden_dim_swnu,
                         augmentation_type=augmentation_type,
                         hidden_dim_aug=hidden_dim_aug,
                         BiLSTM=BiLSTM)
            
        # BiLSTM
        # compute the input dimension
        if log_signature:
            input_dim_lstmsig = signatory.logsignature_channels(
                in_channels=self.swnu.hidden_dim[-1], 
                depth=sig_depth
            )
        else:
            input_dim_lstmsig = signatory.signature_channels(
                in_channels=self.swnu.hidden_dim[-1],
                depth=sig_depth
            )
        
        # BiLSTM that processes the outputs from SWNUs for each window
        self.lstm_sig = nn.LSTM(input_size=input_dim_lstmsig,
                                hidden_size=hidden_dim_lstm,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        # determining how to concatenate features to the SWNU features
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.comb_method = comb_method
        self.feature_concat = FeatureConcatenation(
            input_dim=hidden_dim_lstm,
            num_features=self.num_features,
            embedding_dim=self.embedding_dim,
            comb_method=self.comb_method,
        )
        
        # FFN for classification
        # make sure hidden_dim_ffn a list of integers
        if isinstance(hidden_dim_ffn, int):
            hidden_dim_ffn = [hidden_dim_ffn]
        self.hidden_dim_ffn = hidden_dim_ffn
        
        self.ffn = FeedforwardNeuralNetModel(input_dim=self.feature_concat.output_dim,
                                             hidden_dim=self.hidden_dim_ffn,
                                             output_dim=output_dim,
                                             dropout_rate=dropout_rate)

    def forward(
        self,
        path: torch.Tensor,
        features: torch.Tensor | None = None
    ):
        # path has dimensions [batch, units, history, channels]
        # features has dimensions [batch, num_features+embedding_dim]
        # SWNU for each history window
        out = self.swnu(path[:,0,:,:])
        out = out.unsqueeze(1)
        for window in range(1, path.shape[1]):
            out_unit = self.swnu(path[:,window,:,:])
            out_unit = out_unit.unsqueeze(1)
            out = torch.cat((out, out_unit), dim=1)
        
        # order sequences based on sequence length of input
        seq_lengths = torch.sum(torch.sum(torch.sum(path, 1) != 0, 1) != 0, 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        out = out[perm_idx]
        out = torch.nn.utils.rnn.pack_padded_sequence(out, seq_lengths.cpu(), batch_first=True)
        
        # BiLSTM that combines all deepsignet windows together
        _, (out, _) = self.lstm_sig(out)
        out = out[-1, :, :] + out[-2, :, :]
        
        # reverse sequence padding
        inverse_perm = torch.argsort(perm_idx)
        out = out[inverse_perm]
        
        # combine with features provided
        out = self.feature_concat(out, features)
        
        # FFN
        out = self.ffn(out.float())

        return out