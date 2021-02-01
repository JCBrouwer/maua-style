import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np


def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(x.device)
    z = x + alpha * (y - x)

    # gradient penalty
    z = autograd.Variable(z, requires_grad=True).to(x.device)
    o = f(z)
    g = autograd.grad(o, z, grad_outputs=torch.ones(o.size()).to(z.device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
    return gp


def R1Penalty(real_img, f):
    # gradient penalty
    reals = autograd.Variable(real_img, requires_grad=True).to(real_img.device)
    real_logit = f(reals)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

    real_logit = apply_loss_scaling(torch.sum(real_logit))
    real_grads = autograd.grad(
        real_logit, reals, grad_outputs=torch.ones(real_logit.size()).to(reals.device), create_graph=True
    )[0].view(reals.size(0), -1)
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty


# Scale gradients in the backward pass
class ScaleGradients(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, strength):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input * self.strength * self.strength, None


# Divide weights by channel size
def normalize_weights(content_losses, style_losses):
    for n, i in enumerate(content_losses):
        i.strength = i.strength / max(i.target.size())
    for n, i in enumerate(style_losses):
        i.strength = i.strength / max(i.target.size())


# Define an nn Module to compute content loss
class ContentLoss(nn.Module):
    def __init__(self, strength, normalize=False):
        super(ContentLoss, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = "None"
        self.weights = None
        self.normalize = normalize

    def forward(self, input):
        if self.mode == "loss":
            if input.shape != self.target.shape:
                return input
            if self.weights is not None:
                loss = self.crit(input * self.weights, self.target)
            else:
                loss = self.crit(input, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = loss * self.strength
        elif self.mode == "capture":
            self.target = input.detach()
        return input


class GramMatrix(nn.Module):
    def forward(self, input, use_covariance=False):
        B, C, H, W = input.size()
        x_flat = input.view(C, H * W)

        if use_covariance:
            x_flat = x_flat - x_flat.mean(1).unsqueeze(1)

        return torch.mm(x_flat, x_flat.t())


# Define an nn Module to compute style loss
class StyleLoss(nn.Module):
    def __init__(self, strength, use_covariance=False, normalize=False):
        super(StyleLoss, self).__init__()
        self.target = torch.Tensor()
        self.strength = strength
        self.gram = GramMatrix()
        self.crit = nn.MSELoss()
        self.mode = "None"
        self.blend_weight = None
        self.use_covariance = use_covariance
        self.normalize = normalize

    def forward(self, input):
        self.G = self.gram(input, self.use_covariance)
        self.G = self.G.div(input.nelement())
        if self.mode == "capture":
            if self.blend_weight == None:
                self.target = self.G.detach()
            elif self.target.nelement() == 0:
                self.target = self.G.detach().mul(self.blend_weight)
            else:
                self.target = self.target.add(self.blend_weight, self.G.detach())
        elif self.mode == "loss":
            loss = self.crit(self.G, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = loss * self.strength
        return input


class TVLoss(nn.Module):
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input
