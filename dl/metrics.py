import torch
import torch.nn as nn


def dice_calc_multiclass(pred, gt):
    dice_scores = []
    for i in range(gt.shape[0]):
        dice_scores.append(dice_score(pred[i], gt[i]).item())

    return dice_scores


def dice_score(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()
    union = prediction * target
    return 2 * torch.sum(union) / (torch.sum(prediction) + torch.sum(target) + 1e-9)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        output = torch.sigmoid(input)
        # output = (output > 0.5).float()
        return 1 - dice_score(output[:, 1, :, :], target[:, 1, :, :])


if __name__ == '__main__':
    print()
