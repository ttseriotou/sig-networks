from __future__ import annotations

import signatory
import torch
import torch.nn as nn

from nlpsig_networks.feature_concatenation import FeatureConcatenation
from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel
from nlpsig_networks.swnu import SWNU


class SWNUNetwork(nn.Module):
    """
    Stacked Deep Signature Neural Network for classification.
    """

    def __init__(
        self,
        input_channels: int,
        num_features: int,
        embedding_dim: int,
        log_signature: bool,
        sig_depth: int,
        hidden_dim_swnu: list[int] | int,
        hidden_dim_ffn: list[int] | int,
        output_dim: int,
        dropout_rate: float,
        reverse_path: bool = False,
        output_channels: int | None = None,
        augmentation_type: str = "Conv1d",
        hidden_dim_aug: list[int] | int | None = None,
        BiLSTM: bool = False,
        comb_method: str = "concatenation",
    ):
        """
        SWNU network for classification.

        Parameters
        ----------
        input_channels : int
            Dimension of the embeddings that will be passed in.
        num_features : int
            Number of time features to add to FFN input. If none, set to zero.
        embedding_dim : int
            Dimension of embedding to add to FFN input. If none, set to zero.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        hidden_dim_swnu : list[int] | int
            Dimensions of the hidden layers in the SNWU blocks.
        hidden_dim_ffn : list[int] | int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.
        reverse_path : bool, optional
            Whether or not to reverse the path before passing it through the
            signature layers, by default False.
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
            Whether or not a birectional LSTM is used,
            by default False (unidirectional LSTM is used in this case).
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
        super(SWNUNetwork, self).__init__()

        self.swnu = SWNU(
            input_channels=input_channels,
            output_channels=output_channels,
            log_signature=log_signature,
            sig_depth=sig_depth,
            hidden_dim=hidden_dim_swnu,
            reverse_path=reverse_path,
            BiLSTM=BiLSTM,
            augmentation_type=augmentation_type,
            hidden_dim_aug=hidden_dim_aug,
        )

        # signature without lift (for passing into FFN)
        if log_signature:
            signature_output_channels = signatory.logsignature_channels(
                in_channels=self.swnu.hidden_dim[-1], depth=sig_depth
            )
        else:
            signature_output_channels = signatory.signature_channels(
                channels=self.swnu.hidden_dim[-1], depth=sig_depth
            )

        # determining how to concatenate features to the SWNU features
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.comb_method = comb_method
        self.feature_concat = FeatureConcatenation(
            input_dim=signature_output_channels,
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
        # use SWNU to obtain feature set
        out = self.swnu(path)

        # combine with features provided
        out = self.feature_concat(out, features)

        # FFN
        out = self.ffn(out.float())

        return out
