from __future__ import annotations

import torch
import torch.nn as nn

from sig_networks.feature_concatenation import FeatureConcatenation
from sig_networks.ffn_baseline import FeedforwardNeuralNetModel
from sig_networks.swmhau import SWMHAU


class SWMHAUNetwork(nn.Module):
    """
    Signature Window using Multihead Attention Unit (SWMHAU) network for classification.
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
        pooling: str,
        reverse_path: bool = False,
        augmentation_type: str = "Conv1d",
        hidden_dim_aug: list[int] | int | None = None,
        comb_method: str = "concatenation",
    ):
        """
        Signature Window using Multihead Attention Unit (SWMHAU)
        network for classification.

        Parameters
        ----------
        input_channels : int
            Dimension of the embeddings in the path that will be passed in.
        output_channels : int | None, optional
            Requested dimension of the embeddings after convolution layer.
        num_features : int
            Number of time features to add to FFN input. If none, set to zero.
        embedding_dim : int
            Dimension of embedding to add to FFN input. If none, set to zero.
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
            Dropout rate in the FFN and SWMHAU.
        pooling: str | None
            Pooling operation to apply in SWMHAU to obtain history representation.
            Options are:
                - "signature": apply signature on a FFN of the MHA units at the end
                  to obtain the final history representation
                - "cls": introduce a CLS token and return the MHA output for this token
        reverse_path : bool, optional
            Whether or not to reverse the path before passing it through the
            signature layers, by default False.
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
            - gated_addition: element-wise addition of path signature
              and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature
              and embedding vector
            - scaled_concatenation: concatenation of single value scaled path
              signature and embedding vector
        """
        super().__init__()

        self.swmhau = SWMHAU(
            input_channels=input_channels,
            output_channels=output_channels,
            log_signature=log_signature,
            sig_depth=sig_depth,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            pooling=pooling,
            reverse_path=reverse_path,
            augmentation_type=augmentation_type,
            hidden_dim_aug=hidden_dim_aug,
        )

        # determining how to concatenate features to the SWNU features
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

        self.ffn = FeedforwardNeuralNetModel(
            input_dim=self.feature_concat.output_dim,
            hidden_dim=self.hidden_dim_ffn,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, path: torch.Tensor, features: torch.Tensor | None = None):
        # path has dimensions [batch, length of signal, channels]
        # features has dimensions [batch, num_features+embedding_dim]
        # use SWMHAU to obtain feature set
        out = self.swmhau(path)

        # combine with features provided
        out = self.feature_concat(out, features)

        # FFN
        out = self.ffn(out.float())

        return out
