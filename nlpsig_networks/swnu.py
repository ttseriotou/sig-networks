from __future__ import annotations

import torch
import torch.nn as nn
from signatory import (
    Augment,
    LogSignature,
    Signature,
    logsignature_channels,
    signature_channels,
)

from nlpsig_networks.utils import obtain_signatures_mask


class SWLSTM(nn.Module):
    """
    Signature Window using LSTM (SWLSTM).
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
        Applies a multi-layer Signature & LSTM block (SWLSTM) to
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
            Dimensions of the hidden layers in the LSTM blocks in the SWLSTM.
        BiLSTM : bool, optional
            Whether or not a birectional LSTM is used for the final SWLSTM block,
            by default False (unidirectional LSTM is used in this case).
        """
        super(SWLSTM, self).__init__()

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
        for layer in range(len(self.hidden_dim)):
            # create expanding window signature layer and
            # compute the input dimension to LSTM
            if self.log_signature:
                self.signature_layers.append(LogSignature(depth=sig_depth, stream=True))
                if layer == 0:
                    input_dim_lstm = logsignature_channels(
                        in_channels=input_size, depth=sig_depth
                    )
                else:
                    input_dim_lstm = logsignature_channels(
                        in_channels=self.hidden_dim[layer - 1], depth=sig_depth
                    )
            else:
                self.signature_layers.append(Signature(depth=sig_depth, stream=True))
                if layer == 0:
                    input_dim_lstm = signature_channels(
                        channels=input_size, depth=sig_depth
                    )
                else:
                    input_dim_lstm = signature_channels(
                        channels=self.hidden_dim[layer - 1], depth=sig_depth
                    )

            # create LSTM layer (if last layer, this can be a BiLSTM)
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_dim_lstm,
                    hidden_size=self.hidden_dim[layer],
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False
                    if layer != (len(self.hidden_dim) - 1)
                    else self.BiLSTM,
                )
            )

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
        for layer in range(len(self.hidden_dim)):
            x = self.signature_layers[layer](x)
            # compute the length of the stream incase we need to handle empty units
            stream_dim = x.shape[1]

            # padding for LSTM (i.e. find the padding mask on the
            # streamed signatures to get length of the stream)
            # obtain padding mask on the streamed signatures
            mask = obtain_signatures_mask(x)
            # obtain the length of the stream for each item in the batch dimension
            seq_lengths = torch.sum(mask is False, 1)
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            x = x[perm_idx]
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True
            )

            # apply LSTM layer
            x, _ = self.lstm_layers[layer](x)

            # reverse soring of sequences
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            inverse_perm = torch.argsort(perm_idx)
            x = x[inverse_perm]

            # if last layer and using BiLSTM, need to add element-wise
            if (self.BiLSTM) & (layer == (len(self.hidden_dim) - 1)):
                # using BiLSTM on the last layer - need to add element-wise
                # the forward and backward LSTM states
                x = (
                    x[:, :, : self.hidden_dim[layer]]
                    + x[:, :, self.hidden_dim[layer] :]
                )

            # handle error in cases of empty units
            if x.shape[1] == 1:
                x = x.repeat(1, stream_dim, 1)

        # take final signature
        out = self.signature2(x)

        return out


class SWNU(nn.Module):
    """
    Signature Window Network Unit (SWNU) class (using LSTM blocks).
    """

    def __init__(
        self,
        input_channels: int,
        log_signature: bool,
        sig_depth: int,
        hidden_dim: list[int] | int,
        output_channels: int | None = None,
        augmentation_type: str = "Conv1d",
        hidden_dim_aug: list[int] | int | None = None,
        BiLSTM: bool = False,
    ):
        """
        Signature Window Network Unit (SWNU) class (using LSTM blocks).

        Parameters
        ----------
        input_channels : int
            Dimension of the embeddings that will be passed in.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        hidden_dim : list[int] | int
            Dimensions of the hidden layers in the SNWU blocks.
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
        """
        super(SWNU, self).__init__()

        self.input_channels = input_channels
        self.log_signature = log_signature
        self.sig_depth = sig_depth

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        self.hidden_dim = hidden_dim

        self.output_channels = (
            output_channels if output_channels is not None else hidden_dim[-1]
        )

        if augmentation_type not in ["Conv1d", "signatory"]:
            raise ValueError("`augmentation_type` must be 'Conv1d' or 'signatory'.")
        self.augmentation_type = augmentation_type

        if isinstance(hidden_dim_aug, int):
            hidden_dim_aug = [hidden_dim_aug]
        elif hidden_dim_aug is None:
            hidden_dim_aug = []
        self.hidden_dim_aug = hidden_dim_aug
        self.BiLSTM = BiLSTM

        # convolution
        self.conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # alternative to convolution: using Augment from signatory
        self.augment = Augment(
            in_channels=self.input_channels,
            layer_sizes=self.hidden_dim_aug + [self.output_channels],
            include_original=False,
            include_time=False,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # non-linearity
        self.tanh = nn.Tanh()

        # signature window & LSTM blocks
        self.swlstm = SWLSTM(
            input_size=self.output_channels,
            hidden_dim=self.hidden_dim,
            log_signature=self.log_signature,
            sig_depth=self.sig_depth,
            BiLSTM=self.BiLSTM,
        )

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
            out = self.tanh(out)
            # make output have dimensions [batch, length of signal, channels]
            out = torch.transpose(out, 1, 2)
        elif self.augmentation_type == "signatory":
            # input has dimensions [batch, length of signal, channels]
            # (signatory.Augment expects this)
            # and get only the path information
            # output has dimensions [batch, length of signal, channels]
            out = self.augment(x[:, :, : self.input_channels])

        out = self.swlstm(out)

        return out
