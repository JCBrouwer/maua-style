import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from utils import info


# def gradient_penalty(x, y, f):
#     # interpolation
#     shape = [x.size(0)] + [1] * (x.dim() - 1)
#     alpha = torch.rand(shape).to(x.device)
#     z = x + alpha * (y - x)

#     # gradient penalty
#     z = autograd.Variable(z, requires_grad=True).to(x.device)
#     o = f(z)
#     g = autograd.grad(o, z, grad_outputs=torch.ones(o.size()).to(z.device), create_graph=True)[0].view(z.size(0), -1)
#     gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
#     return gp


# def R1Penalty(real_img, f):
#     # gradient penalty
#     reals = autograd.Variable(real_img, requires_grad=True).to(real_img.device)
#     real_logit = f(reals)
#     apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
#     undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

#     real_logit = apply_loss_scaling(torch.sum(real_logit))
#     real_grads = autograd.grad(
#         real_logit, reals, grad_outputs=torch.ones(real_logit.size()).to(reals.device), create_graph=True
#     )[0].view(reals.size(0), -1)
#     real_grads = undo_loss_scaling(real_grads)
#     r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
#     return r1_penalty


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
        self.mode = "none"
        self.weights = None
        self.normalize = normalize
        self.loss = 0
        self.target = torch.Tensor()

    def forward(self, input):
        if self.mode == "none" or (input.shape[1:] != self.target.shape[1:] and self.target.nelement() != 0):
            return input

        self.loss = 0
        for idx in range(input.shape[0]):

            if self.mode == "loss":
                if self.weights is not None:
                    loss = self.crit(input[[idx]] * self.weights, self.target)
                else:
                    loss = self.crit(input[[idx]], self.target)
                if self.normalize:
                    loss = ScaleGradients.apply(loss, self.strength)
                self.loss += loss * self.strength / input.shape[0]

            if self.mode == "capture":
                self.target = input.detach()

        return input


class GramMatrix(nn.Module):
    def forward(self, x, shift_x=0, shift_y=0, shift_t=0, flip_h=False, flip_v=False, use_covariance=False):
        B, C, H, W = x.size()

        # maybe apply transforms before calculating gram matrix
        if not (shift_x == 0 and shift_y == 0):
            x = x[:, :, shift_y:, shift_x:]
            y = x[:, :, : H - shift_y, : W - shift_x]
            B, C, H, W = x.size()
        if flip_h:
            y = x[:, :, :, ::-1]
        if flip_v:
            y = x[:, :, ::-1, :]
        else:
            # TODO does this double the required memory?
            y = x

        x_flat = x.reshape(B * C, H * W)
        y_flat = y.reshape(B * C, H * W)

        if use_covariance:
            x_flat = x_flat - x_flat.mean(1).unsqueeze(1)
            y_flat = y_flat - y_flat.mean(1).unsqueeze(1)

        return torch.mm(x_flat, y_flat.t())


# Define an nn Module to compute style loss
class StyleLoss(nn.Module):
    def __init__(
        self,
        strength,
        use_covariance=False,
        normalize=False,
        video_style_factor=0,
        shift_factor=0,
        flip_factor=0,
        rotation_factor=0,
    ):
        super(StyleLoss, self).__init__()

        self.reset_targets()

        self.strength = strength
        self.blend_weight = None
        self.video_style_factor = video_style_factor
        self.shift_factor = shift_factor
        self.flip_factor = flip_factor
        self.rotation_factor = rotation_factor

        self.gram = GramMatrix()
        self.crit = nn.MSELoss()
        self.loss = 0

        self.mode = "none"
        self.use_covariance = use_covariance
        self.normalize = normalize

    def reset_targets(self):
        self.target = torch.Tensor()
        self.video_target = torch.Tensor()
        self.shift_targets_x = []
        self.shift_targets_y = []

    def forward(self, input):
        if self.mode == "none":
            return input

        self.static_loss(input)

        if self.video_style_factor > 0:
            self.dynamic_loss(input)

        return input

    def static_loss(self, input):
        # static style losses average styles over all individual frames
        for idx in range(input.shape[0]):
            gram = self.gram(input[idx].unsqueeze(0), use_covariance=self.use_covariance) / input[idx].nelement()

            if self.mode == "capture":
                if self.target.nelement() == 0:
                    self.target = self.blend_weight * gram.detach() / input.shape[0]
                else:
                    self.target += self.blend_weight * gram.detach() / input.shape[0]

            if self.mode == "loss":
                loss = self.crit(gram, self.target)
                if self.normalize:
                    loss = ScaleGradients.apply(loss, self.strength)
                self.loss += loss * self.strength / input.shape[0]

            # if self.shift_factor > 0:
            #     self.shift_loss(input)
            # if self.flip_factor > 0:
            #     self.flip_loss(input)

    def dynamic_loss(self, input):
        if self.video_target.nelement() != 0 and self.gram(input, False).shape[0] != self.video_target.shape[0]:
            return  # ignore image styles in dynamic style losses

        gram = self.gram(input, use_covariance=self.use_covariance) / input.nelement()

        if self.mode == "capture":
            if self.video_target.nelement() == 0:
                self.video_target = self.blend_weight * gram.detach()
            else:
                self.video_target += self.blend_weight * gram.detach()

        if self.mode == "loss":
            loss = self.crit(gram, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss += self.video_style_factor * loss * self.strength / input.shape[0]

        # if self.shift_factor > 0:
        #     self.shift_loss(input)
        # if self.flip_factor > 0:
        #     self.flip_loss(input)

    def shift_loss(self, input):
        # deltas are powers of 2 of up to 1/4 the input size
        deltas = 4 ** np.array(range(1, int(np.log2(input.shape[-1]) / 2 - 0.5)))
        for idx, delta in enumerate(deltas):
            gram_x = self.gram(input, use_covariance=self.use_covariance, shift_x=delta) / input.nelement()
            gram_y = self.gram(input, use_covariance=self.use_covariance, shift_y=delta) / input.nelement()

            if self.mode == "capture":
                if len(self.shift_targets_x) < len(deltas):
                    self.shift_targets_x.append(gram_x.detach() * self.blend_weight)
                    self.shift_targets_y.append(gram_y.detach() * self.blend_weight)
                else:
                    self.shift_targets_x[idx] += gram_x.detach() * self.blend_weight
                    self.shift_targets_y[idx] += gram_y.detach() * self.blend_weight

            if self.mode == "loss":
                self.loss += (
                    self.shift_factor
                    * self.strength
                    * (((self.shift_targets_x[idx] - gram_x) ** 2) + (self.shift_targets_y[idx] - gram_y) ** 2).sum()
                    / (8 * input.shape[0] ** 2 * input.shape[2] * input.shape[3] ** 2)
                )

    def flip_loss(self, input):
        N = input.shape[1]
        M = input.shape[2] * input.shape[3]

        A_lr = LR_flipped_gram_matrix(a)
        G_lr = LR_flipped_gram_matrix(x)
        A_ud = UD_flipped_gram_matrix(a)
        G_ud = UD_flipped_gram_matrix(x)

        loss = 1.0 / (8 * N ** 2 * M ** 2) * (((G_lr - A_lr) ** 2) + (G_ud - A_ud) ** 2).sum()
        return loss


class TVLoss(nn.Module):
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff)))
        return input
