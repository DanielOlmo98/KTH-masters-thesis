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
    def __init__(self, num_classes=1, weights=None, f1_weight=True):
        super(DiceLoss, self).__init__()
        self.n_classes = num_classes
        self.f1_weight = f1_weight
        if weights is None:
            self.weights = torch.ones(self.n_classes, device='cuda:0')
        else:
            self.weights = weights

    def forward(self, input, target):
        softmax = nn.Softmax(dim=1)
        input = softmax(input)
        output = torch.zeros(1, device='cuda:0')
        for n in range(self.n_classes):
            # calculate dice score for class and mult with weight
            score = 1 - dice_score(input[:, n, :, :], target[:, n, :, :]).mul(self.weights[n])

            if self.f1_weight is not None:
                # multiply dice score with f1
                score += 1 - f1_score((input[:, n, :, :] > 0.5).float(), target[:, n, :, :]).mul(
                    self.f1_weight).mul(self.weights[n])

            output += score

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
    dice_scores = dice_calc_multiclass(input, target)
    print(f"Dice:   ")  # Target: {scores[1]:.3f}")
    for score in dice_scores:
        print(f"   {score:.3f}")

    for n in range(input.size()[0]):
        p = precision(input[n], target[n])
        r = recall(input[n], target[n])
        f1 = f1_score(input[n], target[n])

        print(f"Recall: {r:.3f}, Precision: {p:.3f}, F1: {f1:.3f}")


if __name__ == '__main__':
    print()
