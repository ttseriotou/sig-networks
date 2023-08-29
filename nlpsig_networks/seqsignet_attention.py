from __future__ import annotations
import signatory
import torch
import torch.nn as nn
from nlpsig_networks.swmhau import SWMHA, SWMHAU
from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel
from nlpsig_networks.feature_concatenation import FeatureConcatenation


class SeqSigNetAttention(nn.Module):
    """
    MHA applied to Deep Signature Neural Network Units for classification.
    """
    
    def __init__(
        self, 
        input_channels: int, 
        output_channels: int,
        num_features: int, 
        embedding_dim: int, 
        log_signature: bool,
        sig_depth: int, 
        num_heads: int,
        num_layers: int,
        hidden_dim_ffn: list[int] | int,
        output_dim: int, 
        dropout_rate: float,
        augmentation_type: str = 'Conv1d', 
        hidden_dim_aug: list[int] | int | None = None,
        comb_method: str ='concatenation'):
        """
        SeqSigNetAttention network for classification.
        
        Input data will have the size: [batch size, window size (w), all embedding dimensions (history + time + post), unit size (n)]
        Note: unit sizes will be in reverse chronological order, starting from the more recent and ending with the one further back in time.
        
        Parameters
        ----------
        input_channels : int
            Dimension of the (dimensonally reduced) history embeddings that will be passed in.
        output_channels : int
            Requested dimension of the embeddings after convolution layer.
        num_features : int
            Number of time features to add to FFN input. If none, set to zero.
        embedding_dim: int
            Dimensions of current BERT post embedding. Usually 384 or 768.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        num_heads : int
            The number of heads in the Multihead Attention blocks.
        num_layers : int
            The number of layers in the SWMHAU.
        hidden_dim_ffn : list[int] | int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.
        augmentation_type : str, optional
            Method of augmenting the path, by default "Conv1d".
            Options are:
            - "Conv1d": passes path through 1D convolution layer.
            - "signatory": passes path through `Augment` layer from `signatory` package.
        hidden_dim_aug : list[int] | int | None
            Dimensions of the hidden layers in the augmentation layer.
            Passed into `Augment` class from `signatory` package if
            `augmentation_type='signatory'`, by default None.
        comb_method : str, optional
            Determines how to combine the path signature and embeddings,
            by default "gated_addition".
            Options are:
            - concatenation: concatenation of path signature and embedding vector
            - gated_addition: element-wise addition of path signature and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature and embedding vector
            - scaled_concatenation: concatenation of single value scaled path signature and embedding vector
        """

        super(SeqSigNetAttention, self).__init__()

        # SWMHAU applied to the input (the unit includes the convolution layer)
        self.swmhau = SWMHAU(
            input_channels=input_channels,
            output_channels=output_channels,
            log_signature=log_signature,
            sig_depth=sig_depth,
            num_heads=num_heads,
            num_layers=num_layers,
            augmentation_type=augmentation_type,
            hidden_dim_aug=hidden_dim_aug
        )
        
        # linear layer to project the output of the SWMHAU to the output dimension of the convolution
        self.linear_layer = nn.Linear(self.swmhau.swmha.signature_terms, output_channels)
        
        # SWMHA applied to the output of the linear layer
        self.swmha = SWMHA(
            input_size=output_channels,
            log_signature=log_signature,
            sig_depth=sig_depth,
            num_heads=num_heads,
            num_layers=1,
        )
        
        # determining how to concatenate features to the SWMHAU features
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.comb_method = comb_method
        self.feature_concat = FeatureConcatenation(
            input_dim=self.swmhau.swmha.signature_terms,
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
        # SWMHAU for each history window by flattening and unflattening the path
        # first flatten the path to a three-dimensional tensor of dimensions [batch*units, history, channels]
        out_flat = path.flatten(0, 1)
        # apply SWMHAU to out_flat
        out = self.swmhau(out_flat)
        # unflatten out to have dimensions [batch, units, hidden_dim]
        out = out.unflatten(0, (path.shape[0], path.shape[1]))
        
        # apply the linear layer to the output of the SWMHAU
        out = self.linear_layer(out)
        
        # apply SWMHA to linear projections of the SWMHAU outputs
        out = self.swmha(out)
        
        # combine with features provided
        out = self.feature_concat(out, features)
        
        # FFN
        out = self.ffn(out.float())

        return out