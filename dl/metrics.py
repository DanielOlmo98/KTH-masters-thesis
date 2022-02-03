import torch
import torch.nn as nn


def dice_calc_multiclass(input, target):
    dice_scores = []
    for i in range(input.shape[0]):
        dice_scores.append(dice_score(input[i], target[i]).item())

    return dice_scores


def dice_score(input, target):
    input = input.flatten()
    target = target.flatten()
    union = input * target
    return 2 * torch.sum(union) / (torch.sum(input) + torch.sum(target) + 1e-7)


class DiceLoss(nn.Module):
    def __init__(self, num_classes=1, weights=None):
        super(DiceLoss, self).__init__()
        self.n_classes = num_classes
        if weights is None:
            self.weights = torch.ones(self.n_classes, device='cuda:0')
        else:
            self.weights = weights

    def forward(self, input, target):
        softmax = nn.Softmax(dim=1)
        input = softmax(input)
        # output = (output > 0.5).float()
        output = torch.zeros(1, device='cuda:0')
        for n in range(self.n_classes):
            output += (1 - dice_score(input[:, n, :, :], target[:, n, :, :])).mul(self.weights[n])

        return output


def precision(input, target, flatten=True):
    if flatten:
        input = input.flatten()
        target = target.flatten()
    union = input * target
    return torch.sum(union) / (torch.sum(input) + 1e-7)


def recall(input, target, flatten=True):
    if flatten:
        input = input.flatten()
        target = target.flatten()
    union = input * target
    return torch.sum(union) / (torch.sum(target) + 1e-7)


def f1_score(input, target):
    input = input.flatten()
    target = target.flatten()
    r = recall(input, target, flatten=False)
    p = precision(input, target, flatten=False)
    return 2 / ((1 / r) + (1 / p))


def print_metrics(input, target):
    scores = dice_calc_multiclass(input, target)
    p = precision(input, target)
    r = recall(input, target)

    f1 = f1_score(input, target)
    print(f"Dice:   ")  # Target: {scores[1]:.3f}")
    for score in scores:
        print(f"   {score:.3f}")
    print(f"Recall: {r:.3f}, Precision: {p:.3f}, F1: {f1:.3f}")


if __name__ == '__main__':
    print()
