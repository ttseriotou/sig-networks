from __future__ import annotations
import signatory
import torch
import torch.nn as nn
from nlpsig_networks.swnu import SWNU


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
        augmentation_args: dict | None = None,
        hidden_dim_aug: list[int] | int | None = None,
        BiLSTM: bool = False,
        comb_method: str = "gated_addition",
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
        augmentation_args : dict | None, optional
            Arguments to pass into `torch.Conv1d` or `signatory.Augment`, by default None.
            If None, by default will set `kernel_size=3`, `stride=1`, `padding=0`.
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
        """
        self.input_channels = input_channels
        
        if isinstance(hidden_dim_swnu, int):
            hidden_dim_swnu = [hidden_dim_swnu]
        if isinstance(hidden_dim_ffn, int):
            hidden_dim_ffn = [hidden_dim_ffn]
        self.hidden_dim_swnu = hidden_dim_swnu
        self.hidden_dim_ffn = hidden_dim_ffn
        
        self.embedding_dim = embedding_dim
        self.num_time_features = num_time_features
        if comb_method not in ["concatenation", "gated_addition"]:
            raise ValueError(
                "`comb_method` must be either 'concatenation' or 'gated_addition'."
            )
        self.comb_method = comb_method
        if augmentation_type not in ["Conv1d", "signatory"]:
            raise ValueError("`augmentation_type` must be 'Conv1d' or 'signatory'.")
        
        self.augmentation_type = augmentation_type
        if isinstance(hidden_dim_aug, int):
            hidden_dim_aug = [hidden_dim_aug]
        elif hidden_dim_aug is None:
            hidden_dim_aug = []
        self.hidden_dim_aug = hidden_dim_aug
        if augmentation_args is None:
            augmentation_args = {"kernel_size": 3,
                                 "stride": 1,
                                 "padding": 1}
        # convolution
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            **augmentation_args,
        )
        self.augment = signatory.Augment(
            in_channels=input_channels,
            layer_sizes=self.hidden_dim_aug + [output_channels],
            include_original=False,
            include_time=False,
            **augmentation_args,
        )
        # non-linearity
        self.tanh1 = nn.Tanh()
        
        self.swnu = SWNU(input_size=output_channels,
                         hidden_dim=self.hidden_dim_swnu,
                         log_signature=log_signature,
                         sig_depth=sig_depth,
                         BiLSTM=BiLSTM)
        
        # signature without lift (for passing into FFN)
        mult = 2 if BiLSTM else 1
        if self.log_signature:
            signature_output_channels = signatory.logsignature_channels(
                in_channels=mult * self.hidden_dim_swnu[-1], depth=sig_depth
            )
        else:
            signature_output_channels = signatory.signature_channels(
                channels=mult * self.hidden_dim_swnu[-1], depth=sig_depth
            )
        
        # find dimension of features to pass through FFN
        if self.comb_method == "concatenation":
            input_dim = (
                signature_output_channels
                + self.num_time_features
                + self.embedding_dim
            )
        elif self.comb_method == "gated_addition":
            input_dim = self.embedding_dim
            input_gated_linear = (
                signature_output_channels
                + self.num_time_features
            )
            if self.embedding_dim > 0:
                self.fc_scale = nn.Linear(input_gated_linear, self.embedding_dim)
                self.scaler = torch.nn.Parameter(torch.zeros(1, self.embedding_dim))
            else:
                self.fc_scale = nn.Linear(input_gated_linear, input_gated_linear)
                self.scaler = torch.nn.Parameter(torch.zeros(1, input_gated_linear))
            # non-linearity
            self.tanh2 = nn.Tanh()

        # FFN: input layer
        self.ffn_input_layer = nn.Linear(input_dim, self.hidden_dim_ffn[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        input_dim = self.hidden_dim_ffn[0]
        
        # FFN: hidden layers
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
        
        # FFN: readout
        self.ffn_final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x has dimensions [batch, length of signal, channels]

        # convolution
        if self.augmentation_type == "Conv1d":
            # input has dimensions [batch, length of signal, channels]
            # swap dimensions to get [batch, channels, length of signal]
            # (nn.Conv1d expects this)
            out = torch.transpose(x, 1, 2)
            # get only the path information
            out = self.conv(out[:, : self.input_channels, :])
            out = self.tanh1(out)
            # make output have dimensions [batch, length of signal, channels]
            out = torch.transpose(out, 1, 2)
        elif self.augmentation_type == "signatory":
            # input has dimensions [batch, length of signal, channels]
            # (signatory.Augment expects this)
            # and get only the path information
            # output has dimensions [batch, length of signal, channels]
            out = self.augment(x[:, :, : self.input_channels])

        # use SWNU to obtain feature set
        out = self.swnu(out)

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
                if x.shape[2] > self.input_channels + self.num_time_features:
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
                out_gated = self.tanh2(out_gated)
                out_gated = torch.mul(self.scaler, out_gated)
                if x.shape[2] > self.input_channels + self.num_time_features:
                    # concatenate current post embedding if provided
                    out = (
                        out_gated
                        + x[:, 0, (self.input_channels + self.num_time_features) :],
                    )
                else:
                    out = out_gated

        # FFN: input layer
        out = self.ffn_input_layer(out)
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
