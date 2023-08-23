from __future__ import annotations
from signatory import Signature, LogSignature, signature_channels, logsignature_channels, Augment
import torch
import torch.nn as nn

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
            The number of layers in the SWMHA.
        """
        super(SWMHA, self).__init__()
        
        self.signature_terms = None
        # check if the parameters are compatible with each other
        # set the number of signature terms from the input size and signature depth
        self._check_signature_terms_divisible_num_heads(input_size=input_size,
                                                        log_signature=log_signature,
                                                        sig_depth=sig_depth,
                                                        num_heads=num_heads)
        
        # logging inputs to the class
        self.input_size = input_size
        self.log_signature = log_signature
        self.sig_depth = sig_depth
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # create signature layers
        if self.log_signature:
            self.signature_layers = nn.ModuleList(
                [LogSignature(depth=sig_depth, stream=True) for _ in range(self.num_layers)]
            )
        else:
            self.signature_layers = nn.ModuleList(
                [Signature(depth=sig_depth, stream=True) for _ in range(self.num_layers)]
            )
            
        # create Multihead Attention layers
        self.mha_layers = nn.ModuleList(
            [nn.MultiheadAttention(
                embed_dim=self.signature_terms,
                num_heads=self.num_heads,
                batch_first=True
            ) for _ in range(self.num_layers)]
        )
        
        # create linear layers to project the output of the MHA layers down to original input size
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.signature_terms, self.input_size) for _ in range(self.num_layers)]
        )

        # layer norm
        self.norm = nn.LayerNorm(self.signature_terms)
        
        # final signature without lift (i.e. no expanding windows)
        if self.log_signature:
            self.signature2 = LogSignature(depth=sig_depth, stream=False)
        else:
            self.signature2 = Signature(depth=sig_depth, stream=False)
        
    def _check_signature_terms_divisible_num_heads(self,
                                                   input_size: int,
                                                   log_signature: bool,
                                                   sig_depth: int,
                                                   num_heads: int):
        # check that the signature terms are divisible by the number of heads
        # compute the output size of the signature
        if log_signature:
            self.signature_terms = logsignature_channels(in_channels=input_size,
                                                         depth=sig_depth)
        else:
            self.signature_terms = signature_channels(channels=input_size,
                                                      depth=sig_depth)
            
        # check that the output size is divisible by the number of heads
        if self.signature_terms % num_heads != 0:
            raise ValueError(f"Output size of the signature ({self.signature_terms}) not "
                             f"divisible by number of heads ({num_heads}).")
        
    def forward(self, x: torch.Tensor):
        # x has dimensions [batch, length of signal, channels]
        
        # take signature lifts and lstm
        for l in range(self.num_layers):
            # apply signature with lift layer
            x = self.signature_layers[l](x)
            # obtain padding mask for the MHA layer (ignore zero vectors)
            # create a binary mask (2d tensor with dimensions [batch, length_of_signal])
            # where if a value is True, the corresponding value on the attention layer will be ignored
            mask = torch.sum(x, 2) == 0
            # apply MHA layer to the signatures
            attention_out = self.mha_layers[l](x, x, x, key_padding_mask=mask)[0]
            # apply layer norm and residual connection
            x = self.norm(x + attention_out)
            # apply linear layer
            x = self.linear_layers[l](x)

        # take final signature
        out = self.signature2(x)
        
        return out
    
    
class SWMHAU(nn.Module):
    """
    Signature Window using Multihead Attention Unit (SWMHAU).
    """
    
    def __init__(
        self,
        input_channels: int,
        log_signature: bool,
        sig_depth: int,
        num_heads: int,
        num_layers: int,
        output_channels: int,
        augmentation_type: str = "Conv1d",
        hidden_dim_aug: list[int] | int | None = None,
    ):
        """
        Signature Window using Multihead Attention Unit (SWMHAU).

        Parameters
        ----------
        input_channels : int
            Dimension of the embeddings that will be passed in.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        num_heads : int
            The number of heads in the Multihead Attention blocks.
        num_layers : int
            The number of layers in the SWMHA.
        output_channels : int
            Requested dimension of the embeddings after convolution layer.
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
        self.log_signature = log_signature
        self.sig_depth = sig_depth
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_channels = output_channels
            
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
        self.swmha = SWMHA(input_size=self.output_channels,
                           log_signature=self.log_signature,
                           sig_depth=self.sig_depth,
                           num_heads=self.num_heads,
                           num_layers=self.num_layers)
        
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