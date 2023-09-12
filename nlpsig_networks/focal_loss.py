import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Focal Loss

    source: https://github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(
        self,
        gamma: float = 0.0,
        alpha: Optional[Union[float, list]] = None,
        size_average: bool = True,
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def set_alpha_from_y(self, y: torch.Tensor):
        y_list = y.tolist()
        self.alpha = torch.Tensor(
            [math.sqrt(y.shape[0] / y_list.count(i)) for i in set(y_list)]
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ClassBalanced_FocalLoss(nn.Module):
    """
    Class Balanced Focal Loss

    source: https://github.com/vandit15/Class-balanced-loss-pytorch
    """

    def __init__(
        self,
        gamma: float,
        beta: float,
        no_of_classes: int,
        samples_per_cls: Optional[List] = None,
    ):
        super(ClassBalanced_FocalLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.no_of_classes = no_of_classes
        self.samples_per_cls = samples_per_cls

    def set_samples_per_cls_from_y(self, y: torch.Tensor):
        y_list = y.tolist()
        self.samples_per_cls = torch.Tensor([y_list.count(i) for i in set(y_list)])

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute the Class Balanced Loss between `logits` and
        the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        cb_loss = self.focal_loss(labels_one_hot, logits, weights)
        return cb_loss

    def focal_loss(
        self, labels: torch.Tensor, logits: torch.Tensor, alpha: torch.Tensor
    ):
        """
        Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels, reduction="none"
        )

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(
                -self.gamma * labels * logits
                - self.gamma * torch.log(1 + torch.exp(-1.0 * logits))
            )

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss
