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
        output = torch.zeros(1, device='cuda:0')
        for n in range(self.n_classes):
            # calculate dice score for class and mult with weight
            score = 1 - dice_score(input[:, n, :, :], target[:, n, :, :]).mul(self.weights[n])

            output += score

        return output


def precision(input, target):
    union = input * target
    return torch.sum(union) / (torch.sum(input) + 1e-7)


def recall(input, target):
    union = input * target
    return torch.sum(union) / (torch.sum(target) + 1e-7)


def f1_score(input, target, beta):
    input = input.flatten()
    target = target.flatten()
    r = recall(input, target)
    p = precision(input, target)
    return ((1 + beta ** 2) * r * p) / (r + p * (beta ** 2))


class FscoreLoss(nn.Module):
    def __init__(self, class_weights, f1_weight):
        super(FscoreLoss, self).__init__()
        self.class_weights = class_weights
        self.num_classes = len(class_weights)
        self.f1_weight = f1_weight

    def __str__(self):
        return f'{type(self)}:\n    Class weights: {self.class_weights}\n    F weight: {self.f1_weight}'

    def to_json(self):
        return {'class_weights': self.class_weights.tolist(),
                'num_classes': self.num_classes,
                'f1_weight': self.f1_weight}

    def forward(self, input, target):
        score = torch.zeros(1, device="cuda:0")
        softmax = nn.Softmax(dim=1)
        input = softmax(input)
        for n in range(self.num_classes):
            score += (1 - f1_score(input.unsqueeze(dim=0), target.unsqueeze(dim=0), self.f1_weight)) * (
                self.class_weights[n])

        return score


def print_metrics(input, target):
    for n in range(input.size()[0]):
        p, r, f1 = get_f1_metrics(input[n], target[n])
        print(f"Class {n + 1}:\n  Recall: {r:.3f}, Precision: {p:.3f}, F1: {f1:.3f}")


def get_f1_metrics(input, target):
    p = precision(input, target)
    r = recall(input, target)
    f1 = f1_score(input, target, 1)
    return p, r, f1


if __name__ == '__main__':
    print()
