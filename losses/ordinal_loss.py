import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from typing import Optional


class OrdinalLoss(nn.Module):
    '''
      - weight: tensor(N)
    '''
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super(OrdinalLoss, self).__init__()
        self.weight = weight.squeeze() if (weight != None) else None

    '''
      - pred:  tensor(B, N)
      - labels:  tensor(B)
    '''
    def forward(self, pred: torch.Tensor, labels: torch.Tensor):
        pred_ordinal = self.pred_to_ordinal(pred)
        encoded_labels = self.ordinal_encode_label(labels, pred_ordinal.size(-1))
        mse_loss = nn.MSELoss(reduction='none')(sigmoid(pred), encoded_labels).sum(axis=1)
        if self.weight != None:
            mse_loss = self.weight[labels] * mse_loss
        return mse_loss.sum()

    '''
      - labels:  tensor(B)
      - num_classes:  (int)

      returns: tensor(B, N)
    '''
    def ordinal_encode_label(self, labels: torch.Tensor, num_classes: int):
        batch_size = labels.size(0)
        labels = labels + 1
        encoded_labels = torch.zeros((batch_size, num_classes))
        class_indices = torch.arange(num_classes).unsqueeze(0)
        encoded_labels[torch.arange(batch_size).unsqueeze(1), class_indices] = (labels.unsqueeze(1) > class_indices).float()
        return encoded_labels

    '''
      - pred:  tensor(B, N)

      returns: tensor(B, N)
    '''
    def pred_to_ordinal(self, pred: torch.Tensor):
        return sigmoid(pred) > 0.5

    '''
      - pred:  tensor(B, N)

      returns: tensor(N)
    '''
    def pred_to_label(self, pred: torch.Tensor):
        return torch.max(self.pred_to_ordinal(pred).cumprod(axis=-1).sum(axis=-1) - 1, other=torch.tensor(0))