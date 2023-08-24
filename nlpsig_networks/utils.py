from __future__ import annotations
import torch
import torch.nn as nn

def obtain_paths_mask(path: torch.tensor) -> torch.tensor:
    """
    Given a path, given as a tensor of dimensions [batch, length, channels],
    the function returns a mask of dimensions [batch, length] where
    the mask is True if the path includes a zero vector.
    
    Function assumes that the padding in the path is given by a zero vector.
    
    This can be passed into `torch.nn.MultiheadAttention` as the
    `key_padding_mask` argument when performing multihead attention
    on a path. In the returned binary mask when a value is True,
    the corresponding value on the attention layer will be ignored.
    
    Parameters
    ----------
    signatures : torch.tensor
        Tensor of dimensions [batch, length, channels] where
        the channels are the signature terms.

    Returns
    -------
    torch.tensor
        Tensor of dimensions [batch, length] where the mask is
        True if the signature is equal to the previous signature.
    """
    # sum up path in dimension 2 and see if it is equal to zero
    return torch.sum(path, 2) == 0
    

def obtain_signatures_mask(signatures: torch.tensor) -> torch.tensor:
    """
    Given a tensor of streamed signatures (i.e. signatures with lift)
    applied to a path, given as a tensor of dimensions [batch, length, channels],
    the function returns a mask of dimensions [batch, length] where
    the mask is True if the signature is equal to the previous signature.
    
    Function assumes that the padding in the path was applied from below.
    In such case, the streamed signatures will have repeated rows
    if the signature of one expanding window is equal to the signature
    of the previous expanding window.

    This can be passed into `torch.nn.MultiheadAttention` as the
    `key_padding_mask` argument when performing multihead attention
    on streamed signatures. In the returned binary mask when a value is True,
    the corresponding value on the attention layer will be ignored.
    
    Parameters
    ----------
    signatures : torch.tensor
        Tensor of dimensions [batch, length, channels] where
        the channels are the signature terms.

    Returns
    -------
    torch.tensor
        Tensor of dimensions [batch, length] where the mask is
        True if the signature is equal to the previous signature.
    """    
    # compare each row with the row above it (for each batch)
    equal_to_previous = torch.eq(signatures[:,1:], signatures[:,:-1])
    
    # look for cases when the entire row is equal to the previous row
    equal_to_previous_row = torch.all(equal_to_previous, dim=2)
    
    # equal_to_previous_row has dimensions [batch, length-1]
    # to make it the same length as the signatures tensor, add a column of False,
    # since we assume padding of the path was applied from below
    # and so we always need to attend to the first row of the signatures tensor
    false_tensor = torch.full((signatures.shape[0], 1), False, dtype=torch.bool)
    
    # return bool tensor of dimension [batch, length]
    return torch.cat((false_tensor, equal_to_previous_row), dim=1)