import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import ExpConfig


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.sigmoid()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        if torch.isnan(dice):
            print("input", torch.isnan(inputs))
            print("target", torch.isnan(targets))
            print("intersection", torch.isnan(intersection))
            raise RuntimeError

        return 1 - dice


# https://github.com/zhaoyuzhi/PyTorch-Sobel/blob/main/pytorch-sobel.py
class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self, x):
        """
        Convert image to its gray one.
        """
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)


class DiceGradLoss(nn.Module):
    def __init__(self):
        super(DiceGradLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.grad_loss = GradLoss()

    def forward(self, output, gt_img):
        dice_loss = self.dice_loss(output, gt_img)
        grad_loss = self.grad_loss(output, gt_img)
        return dice_loss + 0.5 * grad_loss


def set_loss(config: ExpConfig) -> nn.Module:
    if config.loss_type == "dice":
        return DiceLoss()
    elif config.loss_type == "grad":
        return GradLoss()
    elif config.loss_type == "dice_grad":
        return DiceGradLoss()
    else:
        print(f"loss {config.loss} is not supported")
        raise NotImplementedError
