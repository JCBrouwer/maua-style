# Copyright (c) 2021 Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import math
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

import requests
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.append("./taming")

from taming.models import cond_transformer, vqgan

from CLIP import clip


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


def spherical_dist(x, y):
    x_normed = F.normalize(x, dim=-1)
    y_normed = F.normalize(y, dim=-1)
    return x_normed.sub(y_normed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f"unknown vqgan type: {config.model.target}")
    del model.loss
    return model


def size_to_fit(size, max_dim, scale_up=False):
    w, h = size
    if not scale_up and max(h, w) <= max_dim:
        return w, h
    new_w, new_h = max_dim, max_dim
    if h > w:
        new_w = round(max_dim * w / h)
    else:
        new_h = round(max_dim * h / w)
    return new_w, new_h


def fetch(path_or_url):
    if not (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        return open(path_or_url, "rb")
    return requests.get(path_or_url, stream=True).raw


module_order = [
    "conv_in",
    "mid.block_1",
    "mid.block_1.norm1",
    "mid.block_1.conv1",
    "mid.block_1.norm2",
    "mid.block_1.dropout",
    "mid.block_1.conv2",
    "mid.attn_1",
    "mid.attn_1.norm",
    "mid.attn_1.q",
    "mid.attn_1.k",
    "mid.attn_1.v",
    "mid.attn_1.proj_out",
    "mid.block_2",
    "mid.block_2.norm1",
    "mid.block_2.conv1",
    "mid.block_2.norm2",
    "mid.block_2.dropout",
    "mid.block_2.conv2",
    "up.4.block.0",
    "up.4.block.0.norm1",
    "up.4.block.0.conv1",
    "up.4.block.0.norm2",
    "up.4.block.0.dropout",
    "up.4.block.0.conv2",
    "up.4.attn.0",
    "up.4.attn.0.norm",
    "up.4.attn.0.q",
    "up.4.attn.0.k",
    "up.4.attn.0.v",
    "up.4.attn.0.proj_out",
    "up.4.block.1",
    "up.4.block.1.norm1",
    "up.4.block.1.conv1",
    "up.4.block.1.norm2",
    "up.4.block.1.dropout",
    "up.4.block.1.conv2",
    "up.4.attn.1",
    "up.4.attn.1.norm",
    "up.4.attn.1.q",
    "up.4.attn.1.k",
    "up.4.attn.1.v",
    "up.4.attn.1.proj_out",
    "up.4.block.2",
    "up.4.block.2.norm1",
    "up.4.block.2.conv1",
    "up.4.block.2.norm2",
    "up.4.block.2.dropout",
    "up.4.block.2.conv2",
    "up.4.attn.2",
    "up.4.attn.2.norm",
    "up.4.attn.2.q",
    "up.4.attn.2.k",
    "up.4.attn.2.v",
    "up.4.attn.2.proj_out",
    "up.4.upsample",
    "up.4.upsample.conv",
    "up.3.block.0",
    "up.3.block.0.norm1",
    "up.3.block.0.conv1",
    "up.3.block.0.norm2",
    "up.3.block.0.dropout",
    "up.3.block.0.conv2",
    "up.3.block.0.nin_shortcut",
    "up.3.block.1",
    "up.3.block.1.norm1",
    "up.3.block.1.conv1",
    "up.3.block.1.norm2",
    "up.3.block.1.dropout",
    "up.3.block.1.conv2",
    "up.3.block.2",
    "up.3.block.2.norm1",
    "up.3.block.2.conv1",
    "up.3.block.2.norm2",
    "up.3.block.2.dropout",
    "up.3.block.2.conv2",
    "up.3.upsample",
    "up.3.upsample.conv",
    "up.2.block.0",
    "up.2.block.0.norm1",
    "up.2.block.0.conv1",
    "up.2.block.0.norm2",
    "up.2.block.0.dropout",
    "up.2.block.0.conv2",
    "up.2.block.1",
    "up.2.block.1.norm1",
    "up.2.block.1.conv1",
    "up.2.block.1.norm2",
    "up.2.block.1.dropout",
    "up.2.block.1.conv2",
    "up.2.block.2",
    "up.2.block.2.norm1",
    "up.2.block.2.conv1",
    "up.2.block.2.norm2",
    "up.2.block.2.dropout",
    "up.2.block.2.conv2",
    "up.2.upsample",
    "up.2.upsample.conv",
    "up.1.block.0",
    "up.1.block.0.norm1",
    "up.1.block.0.conv1",
    "up.1.block.0.norm2",
    "up.1.block.0.dropout",
    "up.1.block.0.conv2",
    "up.1.block.0.nin_shortcut",
    "up.1.block.1",
    "up.1.block.1.norm1",
    "up.1.block.1.conv1",
    "up.1.block.1.norm2",
    "up.1.block.1.dropout",
    "up.1.block.1.conv2",
    "up.1.block.2",
    "up.1.block.2.norm1",
    "up.1.block.2.conv1",
    "up.1.block.2.norm2",
    "up.1.block.2.dropout",
    "up.1.block.2.conv2",
    "up.1.upsample",
    "up.1.upsample.conv",
    "up.0.block.0",
    "up.0.block.0.norm1",
    "up.0.block.0.conv1",
    "up.0.block.0.norm2",
    "up.0.block.0.dropout",
    "up.0.block.0.conv2",
    "up.0.block.1",
    "up.0.block.1.norm1",
    "up.0.block.1.conv1",
    "up.0.block.1.norm2",
    "up.0.block.1.dropout",
    "up.0.block.1.conv2",
    "up.0.block.2",
    "up.0.block.2.norm1",
    "up.0.block.2.conv1",
    "up.0.block.2.norm2",
    "up.0.block.2.dropout",
    "up.0.block.2.conv2",
    "norm_out",
    "conv_out",
]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

if torch.cuda.device_count() >= 2:
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    print("WARNING: sometimes multi-gpu completely fails to work for as of yet unknown reasons...")
elif torch.cuda.device_count() == 1:
    dev0 = torch.device("cuda")
    dev1 = torch.device("cuda")
else:
    dev0 = torch.device("cpu")
    dev1 = torch.device("cpu")


@torch.no_grad()
def load_models(model_dir, clip_backbone):
    model = (
        load_vqgan_model(glob(model_dir + "/*.yaml")[0], glob(model_dir + "/*.ckpt")[0]).eval().requires_grad_(False)
    )
    model.encoder = model.encoder.cpu().to(dev1)
    model.quantize = model.quantize.cpu().to(dev1)
    model.quant_conv = model.quant_conv.cpu().to(dev1)
    model.post_quant_conv = model.post_quant_conv.cpu().to(dev0)
    model.decoder = model.decoder.cpu().to(dev0)

    for name, module in model.decoder.named_modules():
        if name not in module_order:
            continue

        dev = dev0 if module_order.index(name) < module_order.index("up.0.block.1") else dev1

        def hook(m, x, n=name, d=dev):
            if isinstance(x, tuple):
                return tuple((y.to(d) if y is not None else None) for y in x)
            else:
                return x.to(d)

        module.register_forward_pre_hook(deepcopy(hook))
        mod = model.decoder
        names = name.split(".")
        for n in names[:-1]:
            mod = getattr(mod, n)
        getattr(mod, names[-1])
        setattr(mod, names[-1], module.to(dev))

    perceptor = clip.load(clip_backbone, jit=False, device=dev1)[0].eval().requires_grad_(False)

    make_cutouts = MakeCutouts(perceptor.visual.input_resolution, 64, cut_pow=1.0)

    def preprocess(inputs):
        return TF.normalize(make_cutouts(inputs), mean=CLIP_MEAN, std=CLIP_STD)

    res = 2 ** (model.decoder.num_resolutions - 1)
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None].to(dev0)
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None].to(dev0)

    return model, perceptor, preprocess, res, z_min, z_max


@torch.no_grad()
def initialize_targets(init, style, mask, content_text, style_text, model, perceptor, preprocess, res):
    _, _, h, w = init.shape
    toksX, toksY = w // res, h // res
    sideX, sideY = toksX * res, toksY * res
    init = resample(init, (sideX, sideY))
    if mask.dim() > 1:
        mask = resample(mask, (sideX, sideY))

    z = model.encode(init.to(dev1) * 2 - 1)[0].to(dev0)

    image_embed = perceptor.encode_image(preprocess(init).to(dev1))
    if style is not None:
        style_embed = perceptor.encode_image(preprocess(style_image).to(dev1))
    else:
        style_embed = None
    from_embed = perceptor.encode_text(clip.tokenize(content_text).to(dev1))
    to_embed = perceptor.encode_text(clip.tokenize(style_text).to(dev1))

    mask = mask.to(dev0)

    return [image_embed, from_embed, to_embed, style_embed], z, mask


def synth(model, z):
    # WTF, why does this require going through host memory? for some reason codebook is all 0s without .cpu()
    codebook = model.quantize.embedding.weight.data.cpu().to(dev0)
    z_q = vector_quantize(z.movedim(1, 3), codebook).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


def ascend_txt(model, perceptor, z, mask, preprocess, embeds, content_strength, style_strength, text_strength):
    image_embeds, from_embed, to_embed, style_embed = embeds
    out = synth(model, replace_grad(z, z * mask))
    out_embeds = perceptor.encode_image(preprocess(out).to(dev1))
    return [
        spherical_dist(out_embeds, image_embeds).mean() * content_strength,
        spherical_dist(out_embeds, style_embed).mean() * style_strength if style_embed is not None else 0,
        spherical_dist(out_embeds, from_embed).mean() * -text_strength,
        spherical_dist(out_embeds, to_embed).mean() * text_strength,
    ]


def optimize(
    init,
    style,
    mask,
    content_text,
    style_text,
    content_strength,
    style_strength,
    text_strength,
    model_dir,
    clip_backbone,
    iterations,
    out_dir,
    out_name,
):
    model, perceptor, preprocess, res, z_min, z_max = load_models(model_dir, clip_backbone)
    embeds, z, mask = initialize_targets(init, style, mask, content_text, style_text, model, perceptor, preprocess, res)

    z.requires_grad_()
    opt = optim.Adam([z], lr=0.05)

    for i in tqdm(range(iterations)):
        opt.zero_grad()
        losses = ascend_txt(
            model, perceptor, z, mask, preprocess, embeds, content_strength, style_strength, text_strength
        )
        sum(losses).backward()
        opt.step()

        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

            if i % 50 == 0:
                tqdm.write(f"i: {i}, loss: {sum(losses).item():g} [{', '.join(f'{l.item():g}' for l in losses)}")
                TF.to_pil_image(synth(model, z)[0].cpu()).save(out_dir + "/" + out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str)
    parser.add_argument("--content_text", type=str)
    parser.add_argument("--style_text", type=str)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--text_strength", default=1.0, type=float)
    parser.add_argument("--style_strength", default=1.0, type=float)
    parser.add_argument("--content_strength", default=1.0, type=float)
    parser.add_argument("--model_dir", default="/home/hans/modelzoo/vqgan/imagenet/")
    parser.add_argument("--clip_backbone", default="ViT-B/32", choices=["ViT-B/32", "RN50", "RN101", "RN50x4"])
    parser.add_argument("--out_dir", default="/home/hans/neurout/")
    parser.add_argument("--mask_path", type=str)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--force_square", action="store_true")
    parser.add_argument("--iterations", default=500, type=int)
    args = parser.parse_args()

    out_name = (
        "-".join(
            [Path(args.content).stem] + args.content_text.split() + [Path(args.style).stem]
            if args.style is not None
            else [] + args.style_text.split() + [Path(args.model_dir).stem]
        ).lower()
        + ".jpg"
    )

    style_image = Image.open(fetch(args.style)).convert("RGB")
    sideX, sideY = size_to_fit(style_image.size, args.image_size, True)
    style_image = style_image.resize((sideX, sideY), Image.LANCZOS)
    style_image = TF.to_tensor(style_image).unsqueeze(0)

    if args.content == "random":
        init_image = torch.rand((1, 3, args.image_size, args.image_size))
    else:
        content_image = Image.open(fetch(args.content)).convert("RGB")
        if args.force_square:
            content_image = content_image.resize((args.image_size, args.image_size), Image.LANCZOS)
        sideX, sideY = size_to_fit(content_image.size, args.image_size, True)
        init_image = TF.to_tensor(content_image).unsqueeze(0)

    if args.mask_path:
        pil_image = Image.open(fetch(args.mask_path))
        if "A" in pil_image.getbands():
            pil_image = pil_image.getchannel("A")
        elif "L" in pil_image.getbands():
            pil_image = pil_image.getchannel("L")
        else:
            raise RuntimeError("Mask must have an alpha channel or be one channel")
        mask = TF.to_tensor(pil_image).unsqueeze(0)
        if args.invert_mask:
            mask = 1 - mask
    else:
        mask = torch.ones([])

    optimize(
        init=init_image,
        style=style_image,
        mask=mask,
        content_text=args.content_text,
        style_text=args.style_text,
        content_strength=args.content_strength,
        style_strength=args.style_strength,
        text_strength=args.text_strength,
        model_dir=args.model_dir,
        clip_backbone=args.clip_backbone,
        iterations=args.iterations,
        out_dir=args.out_dir,
        out_name=out_name,
    )
