import load
import os.path
import torch as th
import torch.nn as nn
from torch import optim
import loss
import math
import torch.nn.functional as F


def set_style_targets(net, style_images_big, content_size, opt=None):
    style_images = []
    content_area = content_size[0] * content_size[1]
    for img in style_images_big:
        style_scale = math.sqrt(content_area / (img.size(3) * img.size(2))) * opt.param.style_scale
        style_images.append(F.interpolate(th.clone(img), scale_factor=style_scale, mode="bilinear"))

    style_images = [img.type(opt.model.dtype) for img in style_images]

    # Capture style targets
    for i, image in enumerate(style_images):
        print("Capturing style target " + str(i + 1))
        for j in net.style_losses:
            j.mode = "capture"
            j.blend_weight = opt.param.style_blend_weights[i]
        net(style_images[i])
        for j in net.style_losses:
            j.mode = "loss"


def set_content_targets(net, content_image, content_scale=1, opt=None):
    if not content_scale == 1:
        content_image = F.interpolate(content_image, scale_factor=content_scale, mode="bilinear")
    content_image = content_image.type(opt.model.dtype)

    # Capture content targets
    for i in net.content_losses:
        i.mode = "capture"
    net(content_image)
    for i in net.content_losses:
        i.mode = "loss"


def set_temporal_targets(net, warp_image, warp_weights=None, opt=None):
    warp_image = warp_image.type(opt.model.dtype)

    # Capture temporal targets
    for i in net.temporal_losses:
        i.mode = "capture"
        if warp_weights is not None:
            i.weights = warp_weights.type(opt.model.dtype)
    net(warp_image)
    for i in net.temporal_losses:
        i.mode = "loss"


def optimize(net, init_image, opt):
    init_image = init_image.type(opt.model.dtype)

    # Maybe normalize content and style weights
    if opt.param.normalize_weights:
        loss.normalize_weights(net.content_losses, net.style_losses)

    img = nn.Parameter(init_image)

    i = [0]

    def feval():
        i[0] += 1

        optimizer.zero_grad()

        net(img)

        loss = 0
        for mod in net.content_losses:
            loss += mod.loss.to(opt.model.backward_device)
        for mod in net.style_losses:
            loss += mod.loss.to(opt.model.backward_device)
        if opt.param.tv_weight > 0:
            for mod in net.tv_losses:
                loss += mod.loss.to(opt.model.backward_device)
        loss.backward()

        if opt.optim.print_iter > 0 and i[0] % opt.optim.print_iter == 0:
            print("Iteration " + str(i[0]) + " / " + str(opt.param.num_iterations))
        if (opt.optim.save_iter > 0 and i[0] % opt.optim.save_iter == 0) or i[0] == opt.param.num_iterations:
            output_filename, file_extension = os.path.splitext(opt.output)
            if i[0] == opt.param.num_iterations:
                filename = output_filename + str(file_extension)
            else:
                filename = str(output_filename) + "_" + str(i[0]) + str(file_extension)
            disp = load.deprocess(img.clone())
            if opt.param.original_colors == 1:
                disp = load.original_colors(deprocess(content_image.clone()), disp)
            disp.save(str(filename))

        return loss

    optimizer, loopVal = setup_optimizer(img, opt)
    while i[0] <= loopVal:
        optimizer.step(feval)

    # del net, content_image, init_image, style_images
    #   th.cuda.empty_cache()

    return img


# Configure the optimizer
def setup_optimizer(img, opt):
    if opt.optim.optimizer == "lbfgs":
        # print("Running optimization with L-BFGS")
        optim_state = {"max_iter": opt.param.num_iterations, "tolerance_change": -1, "tolerance_grad": -1}
        if opt.optim.lbfgs_num_correction != 100:
            optim_state["history_size"] = opt.optim.lbfgs_num_correction
        optimizer = optim.LBFGS([img], **optim_state)
        loopVal = 1
    elif opt.optim.optimizer == "adam":
        # print("Running optimization with ADAM")
        optimizer = optim.Adam([img], lr=opt.optim.learning_rate)
        loopVal = opt.param.num_iterations - 1
    return optimizer, loopVal
