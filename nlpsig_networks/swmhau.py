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

from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel
from nlpsig_networks.utils import obtain_signatures_mask


class SWMHA(nn.Module):
    """
    Signature Window using Multihead Attention (SWMHA).
    """

    def __init__(
        self,
        input_size: int,
        log_signature: bool,
        sig_depth: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pooling: str | None,
        reverse_path: bool = False,
    ):
        """
        Applies a multi-layer Signature & Multihead Attention block (SWMHA) to
        an input sequence.

        Parameters
        ----------
        input_size : int
            The number of expected features in the input x.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        num_heads : int
            The number of heads in the Multihead Attention blocks.
        num_layers : int
            The number of layers in the SWMHAU.
        dropout_rate : float
            Probability of dropout. Applied to Multihead Attention,
            and to the FFN layers (before layer norm and residual connection).
        pooling: str | None
            Pooling operation to apply. If None, no pooling is applied.
            Options are:
                - "signature": apply signature on a FFN of the MHA units at the end
                  to obtain the final history representation
                - "cls": introduce a CLS token and return the MHA output for this token
                - None: no pooling is applied (return the FFN of the MHA units)
        reverse_path : bool, optional
            Whether or not to reverse the path before passing it through the
            signature layers, by default False.
        """
        super(SWMHA, self).__init__()

        self.signature_terms = None
        # check if the parameters are compatible with each other
        # set the number of signature terms from the input size and signature depth
        self._check_signature_terms_divisible_num_heads(
            input_size=input_size,
            log_signature=log_signature,
            sig_depth=sig_depth,
            num_heads=num_heads,
        )

        # logging inputs to the class
        self.input_size = input_size
        self.log_signature = log_signature
        self.sig_depth = sig_depth
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.reverse_path = reverse_path

        # create signature layers
        if self.log_signature:
            self.signature_layers = nn.ModuleList(
                [
                    LogSignature(depth=sig_depth, stream=True)
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.signature_layers = nn.ModuleList(
                [
                    Signature(depth=sig_depth, stream=True)
                    for _ in range(self.num_layers)
                ]
            )

        # create Multihead Attention layers
        self.mha_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.signature_terms,
                    num_heads=self.num_heads,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # create dropout layers for MHA
        self.dropout_mha = nn.ModuleList(
            [nn.Dropout(self.dropout_rate) for _ in range(self.num_layers)]
        )

        # create layer norm layers for MHA
        self.layer_norm_mha = nn.ModuleList(
            [nn.LayerNorm(self.signature_terms) for _ in range(self.num_layers)]
        )

        # if num_layers > 1, create FFN layer(s) to project the output of
        # the MHA layers down to original input size so that we can continually
        # compute streams of signatures
        self.ffn_layers = nn.ModuleList(
            [
                FeedforwardNeuralNetModel(
                    input_dim=self.signature_terms,
                    hidden_dim=2 * self.signature_terms,
                    output_dim=self.input_size,
                    dropout_rate=dropout_rate,
                )
                for _ in range(self.num_layers - 1)
            ]
        )

        # create dropout layers for FFN
        self.dropout_ffn = nn.ModuleList(
            [nn.Dropout(self.dropout_rate) for _ in range(self.num_layers)]
        )

        # determine final FFN layer
        if self.pooling == "signature":
            # final FFN layer to project the output of the MHA layers down to
            # original input size to compute final signature
            self.ffn_layers.append(
                FeedforwardNeuralNetModel(
                    input_dim=self.signature_terms,
                    hidden_dim=2 * self.signature_terms,
                    output_dim=self.input_size,
                    dropout_rate=dropout_rate,
                )
            )

            # final signature without lift (i.e. no expanding windows)
            if self.log_signature:
                self.final_signature = LogSignature(depth=self.sig_depth, stream=False)
            else:
                self.final_signature = Signature(depth=self.sig_depth, stream=False)

            # no layer norm for final FFN
            self.final_layer_norm = None
        else:
            # final FFN layer to project the output of the MHA layers without any
            # dimension reduction
            self.ffn_layers.append(
                FeedforwardNeuralNetModel(
                    input_dim=self.signature_terms,
                    hidden_dim=2 * self.signature_terms,
                    output_dim=self.signature_terms,
                    dropout_rate=dropout_rate,
                )
            )

            # no final signature
            self.final_signature = None

            # layer norm for final FFN
            self.final_layer_norm = nn.LayerNorm(self.signature_terms)

        if self.pooling == "cls":
            # define the classification token
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.signature_terms))
        else:
            self.cls_token = None

    def _check_signature_terms_divisible_num_heads(
        self, input_size: int, log_signature: bool, sig_depth: int, num_heads: int
    ):
        # check that the signature terms are divisible by the number of heads
        # compute the output size of the signature
        if log_signature:
            self.signature_terms = logsignature_channels(
                in_channels=input_size, depth=sig_depth
            )
        else:
            self.signature_terms = signature_channels(
                channels=input_size, depth=sig_depth
            )

        # check that the output size is divisible by the number of heads
        if self.signature_terms % num_heads != 0:
            raise ValueError(
                f"Output size of the signature ({self.signature_terms}) not "
                f"divisible by number of heads ({num_heads})."
            )

    def forward(self, x: torch.Tensor):
        # x has dimensions [batch, length of signal, channels]

        # take signature lifts and lstm
        for layer in range(self.num_layers):
            if self.reverse_path:
                # reverse the posts in dim 1 (i.e. the time dimension)
                # as the first post is the most recent
                # (or padding if the path is shorter than the window size)
                x = torch.flip(x, dims=[1])

            # apply signature with lift layer
            x = self.signature_layers[layer](x)

            if self.reverse_path:
                # reverse the posts back to the original order
                x = torch.flip(x, dims=[1])

            if layer == self.num_layers - 1 and self.pooling == "cls":
                # prepend classification token to the streamed signatures
                x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)

            # obtain padding mask on the streamed signatures
            mask = obtain_signatures_mask(x)
            # apply MHA layer to the signatures
            attention_out = self.mha_layers[layer](x, x, x, key_padding_mask=mask)[0]

            # apply layer norm and residual connection (with dropout)
            x = self.layer_norm_mha[layer](x + self.dropout_mha[layer](attention_out))

            # apply FFN
            ffn_out = self.ffn_layers[layer](x)
            # apply dropout to FFN output
            ffn_out = self.dropout_ffn[layer](ffn_out)

            if layer != self.num_layers - 1:
                x = ffn_out

        if self.pooling == "signature":
            # take final signature
            out = self.final_signature(ffn_out)
        else:
            # apply layer norm and residual connection to final FFN output
            out = self.final_layer_norm(x + ffn_out)

            if self.pooling == "cls":
                # extract the classification token output
                out = out[:, 0, :]

        return out


class SWMHAU(nn.Module):
    """
    Signature Window using Multihead Attention Unit (SWMHAU).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int | None,
        log_signature: bool,
        sig_depth: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pooling: str | None,
        reverse_path: bool = False,
        augmentation_type: str = "Conv1d",
        hidden_dim_aug: list[int] | int | None = None,
    ):
        """
        Signature Window using Multihead Attention Unit (SWMHAU).

        Parameters
        ----------
        input_channels : int
            Dimension of the embeddings that will be passed in.
        output_channels : int
            Requested dimension of the embeddings after convolution layer.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        num_heads : int
            The number of heads in the Multihead Attention blocks.
        num_layers : int
            The number of layers in the SWMHAU.
        dropout_rate : float
            Probability of dropout. Applied to Multihead Attention,
            and to the FFN layers (before layer norm and residual connection).
        pooling: str | None
            Pooling operation to apply. If None, no pooling is applied.
            Options are:
                - "signature": apply signature on a FFN of the MHA units at the end
                  to obtain the final history representation
                - "cls": introduce a CLS token and return the MHA output for this token
                - None: no pooling is applied (return the FFN of the MHA units)
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
        """
        super(SWMHAU, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.log_signature = log_signature
        self.sig_depth = sig_depth
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.reverse_path = reverse_path

        if augmentation_type not in ["Conv1d", "signatory"]:
            raise ValueError("`augmentation_type` must be 'Conv1d' or 'signatory'.")
        self.augmentation_type = augmentation_type

        if isinstance(hidden_dim_aug, int):
            hidden_dim_aug = [hidden_dim_aug]
        elif hidden_dim_aug is None:
            hidden_dim_aug = []
        self.hidden_dim_aug = hidden_dim_aug

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

        # signature window & Multihead Attention blocks
        self.swmha = SWMHA(
            input_size=self.output_channels,
            log_signature=self.log_signature,
            sig_depth=self.sig_depth,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            pooling=self.pooling,
            reverse_path=self.reverse_path,
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

        out = self.swmha(out)

        return out
