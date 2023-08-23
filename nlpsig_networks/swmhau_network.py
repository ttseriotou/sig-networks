from __future__ import annotations
import torch
import torch.nn as nn
from nlpsig_networks.swmhau import SWMHAU
from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel


class SWMHAUNetwork(nn.Module):
    """
    Signature Window using Multihead Attention Unit (SWMHAU) network for classification.
    """

    def __init__(
        self,
        input_channels: int,
        num_features: int,
        embedding_dim: int,
        log_signature: bool,
        sig_depth: int,
        num_heads: int,
        num_layers: int,
        hidden_dim_ffn: list[int] | int,
        output_dim: int,
        dropout_rate: float,
        output_channels: int | None = None,
        augmentation_type: str = "Conv1d",
        hidden_dim_aug: list[int] | int | None = None,
        comb_method: str = "concatenation",
    ):
        """
        Signature Window using Multihead Attention Unit (SWMHAU) network for classification.

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
        num_heads : int
            The number of heads in the Multihead Attention blocks.
        num_layers : int
            The number of layers in the SWMHA.
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
        comb_method : str, optional
            Determines how to combine the path signature and embeddings,
            by default "gated_addition".
            Options are:
            - concatenation: concatenation of path signature and embedding vector
            - gated_addition: element-wise addition of path signature and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature and embedding vector
            - scaled_concatenation: concatenation of single value scaled path signature and embedding vector
        """
        super(SWMHAUNetwork, self).__init__()

        self.input_channels = input_channels
        
        self.swmhau = SWMHAU(input_channels=input_channels,
                             output_channels=output_channels,
                             log_signature=log_signature,
                             sig_depth=sig_depth,
                             num_heads=num_heads,
                             num_layers=num_layers,
                             augmentation_type=augmentation_type,
                             hidden_dim_aug=hidden_dim_aug)
        
        signature_output_channels = self.swmhau.swmha.signature_terms
        
        # determining how to concatenate features to the SWMHAU features
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        if comb_method not in ["concatenation", "gated_addition", "gated_concatenation", "scaled_concatenation"]:
            raise ValueError(
                "`comb_method` must be either 'concatenation', 'gated_addition', 'gated_concatenation' or 'scaled_concatenation."
            )
        self.comb_method = comb_method
        if augmentation_type not in ["Conv1d", "signatory"]:
            raise ValueError("`augmentation_type` must be 'Conv1d' or 'signatory'.")
        
        # find dimension of features to pass through FFN
        if self.comb_method == "concatenation":
            input_dim = (
                signature_output_channels
                + self.num_features
                + self.embedding_dim
            )
        elif self.comb_method == "gated_addition":
            input_gated_linear = (
                signature_output_channels
                + self.num_features
            )
            if self.embedding_dim > 0:
                self.fc_scale = nn.Linear(input_gated_linear, self.embedding_dim)
                self.scaler = torch.nn.Parameter(torch.zeros(1, self.embedding_dim))
                input_dim = self.embedding_dim
            else:
                self.fc_scale = nn.Linear(input_gated_linear, input_gated_linear)
                self.scaler = torch.nn.Parameter(torch.zeros(1, input_gated_linear))
                input_dim = input_gated_linear
            # non-linearity
            self.tanh = nn.Tanh()
        elif comb_method=='gated_concatenation':
            # input dimensions for FFN
            input_dim = (
                signature_output_channels
                + self.num_features
                + self.embedding_dim
            )
            # define the scaler parameter
            input_gated_linear = signature_output_channels + self.num_features
            self.scaler1 = torch.nn.Parameter(torch.zeros(1,input_gated_linear))
        elif comb_method=='scaled_concatenation':
            input_dim = (
                signature_output_channels
                + self.num_features
                + self.embedding_dim
            )
            # define the scaler parameter
            self.scaler2 = torch.nn.Parameter(torch.tensor([0.0]))

        # FFN for classification
        # make sure hidden_dim_ffn a list of integers
        if isinstance(hidden_dim_ffn, int):
            hidden_dim_ffn = [hidden_dim_ffn]
        self.hidden_dim_ffn = hidden_dim_ffn
        
        self.ffn = FeedforwardNeuralNetModel(input_dim=input_dim,
                                             hidden_dim=self.hidden_dim_ffn,
                                             output_dim=output_dim,
                                             dropout_rate=dropout_rate)

    def forward(
        self,
        path: torch.Tensor,
        features: torch.Tensor | None = None
    ):
        # x has dimensions [batch, length of signal, channels]
        # features has dimensions [batch, num_features+embedding_dim]
        # use SWMHAU to obtain feature set
        out = self.swmhau(path)

        if features is not None:
            # combine with features provided
            if self.comb_method == "concatenation":
                out = torch.cat((out, features), dim=1)
            else:
                # concatenate any additional features
                if self.num_features > 0:
                    out_gated = torch.cat((out, features[:, : self.num_features]), dim=1)
                else:
                    out_gated = out
                
                if self.comb_method == "gated_addition":
                    # apply gated addition
                    out_gated = self.fc_scale(out_gated.float())
                    out_gated = self.tanh(out_gated)
                    out_gated = torch.mul(self.scaler, out_gated)
                    # element-wise addition of embedding vector if provided
                    if self.embedding_dim > 0:
                        out = out_gated + features[:, self.num_features :]
                    else:
                        out = out_gated
                elif self.comb_method =="gated_concatenation":
                    # apply gated concatenation
                    out_gated = torch.mul(self.scaler1, out_gated) 
                    # concatenate current post embedding if provided
                    if self.embedding_dim > 0:    
                        out = torch.cat((out_gated, features[:, self.num_features :]), dim=1)
                    else:
                        out = out_gated
                elif self.comb_method=="scaled_concatenation":
                    # apply scaled concatenation
                    out_gated = self.scaler2 * out_gated  
                    # concatenate current post embedding if provided
                    if self.embedding_dim > 0:    
                        out = torch.cat((out_gated, features[:, self.num_features :]), dim=1)
                    else:
                        out = out_gated

        # FFN
        out = self.ffn(out.float())

        return out