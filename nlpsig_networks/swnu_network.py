from __future__ import annotations
import signatory
import torch
import torch.nn as nn
from nlpsig_networks.swnu import SWNU
from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel


class SWNUNetwork(nn.Module):
    """
    Stacked Deep Signature Neural Network for classification.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_time_features: int,
        embedding_dim: int,
        log_signature: bool,
        sig_depth: int,
        hidden_dim_swnu: list[int] | int,
        hidden_dim_ffn: list[int] | int,
        output_dim: int,
        dropout_rate: float,
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
        output_channels : int
            Requested dimension of the embeddings after convolution layer.
        num_time_features : int
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
            - gated_addition: element-wise addition of path signature and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature and embedding vector
            - scaled_concatenation: concatenation of single value scaled path signature and embedding vector
        """
        super(SWNUNetwork, self).__init__()

        self.input_channels = input_channels
        
        self.swnu = SWNU(input_channels=input_channels,
                         output_channels=output_channels,
                         log_signature=log_signature,
                         sig_depth=sig_depth,
                         hidden_dim=hidden_dim_swnu,
                         augmentation_type=augmentation_type,
                         hidden_dim_aug=hidden_dim_aug,
                         BiLSTM=BiLSTM)
        
        # signature without lift (for passing into FFN)
        if log_signature:
            signature_output_channels = signatory.logsignature_channels(
                in_channels= self.swnu.hidden_dim[-1], depth=sig_depth
            )
        else:
            signature_output_channels = signatory.signature_channels(
                channels= self.swnu.hidden_dim[-1], depth=sig_depth
            )
        
        # determining how to concatenate features to the SWNU features
        self.embedding_dim = embedding_dim
        self.num_time_features = num_time_features
        if comb_method not in ["concatenation", "gated_addition","gated_concatenation","scaled_concatenation"]:
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
                + self.num_time_features
                + self.embedding_dim
            )
        elif self.comb_method == "gated_addition":
            input_gated_linear = (
                signature_output_channels
                + self.num_time_features
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
                + self.num_time_features
                + self.embedding_dim
            )
            # define the scaler parameter
            input_gated_linear = signature_output_channels + self.num_time_features
            self.scaler1 = torch.nn.Parameter(torch.zeros(1,input_gated_linear))
        elif comb_method=='scaled_concatenation':
            input_dim = (
                signature_output_channels
                + self.num_time_features
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

    def forward(self, x: torch.Tensor):
        # x has dimensions [batch, length of signal, channels]
        # use SWNU to obtain feature set
        out = self.swnu(x)

        # combine last post embedding
        if x.shape[2] > self.input_channels:
            # we have things to concatenate to the path
            if self.comb_method == "concatenation":
                if self.num_time_features > 0:
                    # concatenate any time features
                    # take the maximum for the latest time
                    out = torch.cat(
                        (
                            out,
                            x[
                                :,
                                :,
                                self.input_channels : (
                                    self.input_channels + self.num_time_features
                                ),
                            ].max(1)[0],
                        ),
                        dim=1,
                    )
                if self.embedding_dim > 0:
                    # concatenate current post embedding if provided
                    out = torch.cat(
                        (
                            out,
                            x[:, 0, (self.input_channels + self.num_time_features) :],
                        ),
                        dim=1,
                    )
            elif self.comb_method == "gated_addition":
                if self.num_time_features > 0:
                    # concatenate any time features
                    out_gated = torch.cat(
                        (
                            out,
                            x[
                                :,
                                :,
                                self.input_channels : (
                                    self.input_channels + self.num_time_features
                                ),
                            ].max(1)[0],
                        ),
                        dim=1,
                    )
                else:
                    out_gated = out
                out_gated = self.fc_scale(out_gated.float())
                out_gated = self.tanh(out_gated)
                out_gated = torch.mul(self.scaler, out_gated)
                if self.embedding_dim > 0:
                    # concatenate current post embedding if provided
                    out = out_gated + x[:, 0, (self.input_channels + self.num_time_features) :]
                else:
                    out = out_gated
            elif self.comb_method =="gated_concatenation":
                if self.num_time_features > 0:
                    # concatenate any time features
                    out_gated = torch.cat(
                        (
                            out,
                            x[
                                :,
                                :,
                                self.input_channels : (
                                    self.input_channels + self.num_time_features
                                ),
                            ].max(1)[0],
                        ),
                        dim=1,
                    )
                else:
                    out_gated = out
                out_gated = torch.mul(self.scaler1, out_gated) 
                if self.embedding_dim > 0:
                    # add current post embedding if provided 
                    out = torch.cat((out_gated, x[:, 0, (self.input_channels+self.num_time_features):]), dim=1 )
                else:
                    out = out_gated        
            elif self.comb_method=="scaled_concatenation":
                if self.num_time_features > 0:
                    # concatenate any time features
                    out_gated = torch.cat(
                        (
                            out,
                            x[
                                :,
                                :,
                                self.input_channels : (
                                    self.input_channels + self.num_time_features
                                ),
                            ].max(1)[0],
                        ),
                        dim=1,
                    )                
                else:
                    out_gated = out
                out_gated = self.scaler2 * out_gated  
                if self.embedding_dim > 0:
                    # add current post embedding if provided 
                    out = torch.cat((out_gated , x[:,0, (self.input_channels+self.num_time_features):]) , dim=1 ) 
                else:
                    out = out_gated


        # FFN
        out = self.ffn(out.float())

        return out