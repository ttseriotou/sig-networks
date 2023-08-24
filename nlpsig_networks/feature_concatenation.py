from __future__ import annotations
import torch
import torch.nn as nn
from nlpsig_networks.swmhau import SWMHAU
from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel


class FeatureConcatenation(nn.Module):
    """
    Feature concatentation module.
    """
    def __init__(
        self,
        input_dim: int,
        num_features: int,
        embedding_dim: int,
        comb_method: str = "concatenation",
    ):
        """
        Feature concatentation module that concatenates the output of the
        signatures with additional features if provided. The additional features
        we concatenate is a two dimensional tensor with dimensions 
        [batch, num_features+embedding_dim].

        Parameters
        ----------
        input_dim : int
            Dimension of the embeddings that we want to concatenate to.
        num_features : int
            Number of additional features in the tensor we want to concatenate with.
        embedding_dim : int
            Embedding dimensions in the tensor we want to concatenate with.
        comb_method : str, optional
            Determines how to combine the path signature and embeddings,
            by default "gated_addition".
            Options are:
            - concatenation: concatenation of path signature and embedding vector
            - gated_addition: element-wise addition of path signature and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature and embedding vector
            - scaled_concatenation: concatenation of single value scaled path signature and embedding vector
        """
        if comb_method not in ["concatenation", "gated_addition", "gated_concatenation", "scaled_concatenation"]:
            raise ValueError(
                "`comb_method` must be either 'concatenation', 'gated_addition', 'gated_concatenation' or 'scaled_concatenation."
            )
        self.comb_method = comb_method
        self.input_dim = input_dim
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # find dimension of features to pass through FFN
        if self.comb_method == "concatenation":
            input_dim = (
                self.input_dim
                + self.num_features
                + self.embedding_dim
            )
        elif self.comb_method == "gated_addition":
            input_gated_linear = (
                self.input_dim
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
                self.input_dim
                + self.num_features
                + self.embedding_dim
            )
            # define the scaler parameter
            input_gated_linear = self.input_dim + self.num_features
            self.scaler1 = torch.nn.Parameter(torch.zeros(1,input_gated_linear))
        elif comb_method=='scaled_concatenation':
            input_dim = (
                self.input_dim
                + self.num_features
                + self.embedding_dim
            )
            # define the scaler parameter
            self.scaler2 = torch.nn.Parameter(torch.tensor([0.0]))
        
        # determine how many features will be outputted after concatenation
        self.output_dim = input_dim
        
    def forward(self, out: torch.tensor, features: torch.Tensor | None):
        """
        Concatenates `out` with `features` if `features` is not None.
        
        If `features` is None, this simply returns `out` to deal with the case
        that no additional features are provided and are to be concatenated.

        Parameters
        ----------
        out : torch.tensor
            Two dimensional tensor with dimensions [batch, embedding].
        features : torch.Tensor | None
            Two dimensional tensor with dimensions [batch, embedding]
            or None.
        """
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
        
        return out
        