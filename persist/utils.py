import torch
import torch.nn as nn
from copy import deepcopy


class HurdleLoss(nn.BCEWithLogitsLoss):
    '''
    Hurdle loss that incorporates ZCELoss for each output, as well as MSE for
    each output that surpasses the threshold value. This can be understood as
    the negative log-likelihood of a hurdle distribution.

    Args:
      lam: weight for the ZCELoss term (the hurdle).
      thresh: threshold that an output must surpass to be considered turned on.
    '''
    def __init__(self, lam=10.0, thresh=0):
        super().__init__()
        self.lam = lam
        self.thresh = thresh

    def forward(self, pred, target):
        # Verify prediction shape.
        if pred.shape[1] != 2 * target.shape[1]:
            raise ValueError(
                'Predictions have incorrect shape! For HurdleLoss, the'
                ' predictions must have twice the dimensionality of targets'
                ' ({})'.format(target.shape[1] * 2))

        # Reshape predictions, get distributional.
        pred = pred.reshape(*pred.shape[:-1], -1, 2)
        pred = pred.permute(-1, *torch.arange(len(pred.shape))[:-1])
        mu = pred[0]
        p_logit = pred[1]

        # Calculate loss.
        zero_target = (target <= self.thresh).float().detach()
        hurdle_loss = super().forward(p_logit, zero_target)
        mse = (1 - zero_target) * (target - mu) ** 2

        loss = self.lam * hurdle_loss + mse
        return torch.mean(torch.sum(loss, dim=-1))


class ZCELoss(nn.BCEWithLogitsLoss):
    '''
    Binary classification loss on whether outputs surpass a threshold. Expects
    logits.

    Args:
      thresh: threshold that an output must surpass to be considered on.
    '''
    def __init__(self, thresh=0):
        super().__init__(reduction='none')

    def forward(self, pred, target):
        zero_target = (target == 0).float()
        loss = super().forward(pred, zero_target)
        return torch.mean(torch.sum(loss, dim=-1))


class ZeroAccuracy(nn.Module):
    '''
    Classification accuracy on whether outputs surpass a threshold. Expects
    logits.

    Args:
      thresh: threshold that an output must surpass to be considered turned on.
    '''
    def __init__(self, thresh=0):
        super().__init__()
        self.thresh = thresh

    def forward(self, pred, target):
        zero_pred = (pred > 0).float()
        zero_target = (target <= self.thresh).float()
        acc = (zero_pred == zero_target).float()
        return torch.mean(acc)


class MSELoss(nn.Module):
    '''MSE loss that sums over output dimensions.'''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = torch.sum((pred - target) ** 2, dim=-1)
        return torch.mean(loss)


class Accuracy(nn.Module):
    '''0-1 classification loss.'''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return (torch.argmax(pred, dim=1) == target).float().mean()
