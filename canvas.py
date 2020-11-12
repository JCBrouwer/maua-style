import re
import math
import os.path
import load
from config import load_config
import torch as th
import torch.nn.functional as F
import models
import numpy as np
import ffmpeg
import traceback


def info(x):
    print(x.min(), x.mean(), x.max(), x.size())


def get_histogram(tensor, eps):
    mu_h = tensor.mean(list(range(len(tensor.shape) - 1)))
    h = tensor - mu_h
    h = h.permute(0, 3, 1, 2).reshape(tensor.size(3), -1)
    Ch = th.mm(h, h.T) / h.shape[1] + eps * th.eye(h.shape[0])
    return mu_h, h, Ch


def match_histogram(target_tensor, source_tensor, eps=1e-2, mode="avg"):
    if mode is "avg":
        elementwise = True
        random_frame = False
    else:
        elementwise = False
        random_frame = True

    if not isinstance(source_tensor, list):
        source_tensor = [source_tensor]

    output_tensor = th.zeros_like(target_tensor)
    for source in source_tensor:
        target = target_tensor.permute(0, 3, 2, 1)  # Function expects b,w,h,c
        source = source.permute(0, 3, 2, 1)  # Function expects b,w,h,c
        if elementwise:
            source = source.mean(0).unsqueeze(0)
        if random_frame:
            source = source[np.random.randint(0, source.shape[0])].unsqueeze(0)

        matched_tensor = th.zeros_like(target)
        for idx in range(target.shape[0] if elementwise else 1):
            frame = target[idx].unsqueeze(0) if elementwise else target
            _, t, Ct = get_histogram(frame, eps)
            mu_s, _, Cs = get_histogram(source, eps)

            # PCA
            eva_t, eve_t = th.symeig(Ct, eigenvectors=True, upper=True)
            Et = th.sqrt(th.diagflat(eva_t))
            Et[Et != Et] = 0  # Convert nan to 0
            Qt = th.mm(th.mm(eve_t, Et), eve_t.T)

            eva_s, eve_s = th.symeig(Cs, eigenvectors=True, upper=True)
            Es = th.sqrt(th.diagflat(eva_s))
            Es[Es != Es] = 0  # Convert nan to 0
            Qs = th.mm(th.mm(eve_s, Es), eve_s.T)

            ts = th.mm(th.mm(Qs, th.inverse(Qt)), t)

            match = ts.reshape(*frame.permute(0, 3, 1, 2).shape).permute(0, 2, 3, 1)
            match += mu_s

            if elementwise:
                matched_tensor[idx] = match
            else:
                matched_tensor = match
        output_tensor += matched_tensor.permute(0, 3, 2, 1) / len(source_tensor)
    return output_tensor


def determine_scaling(opt):
    def opt_to_ints(scales, scaling, num):
        scales = ("" + scales).split(",")
        if len(scales) == 2:
            up = max(float(scales[1]), float(scales[0]))
            low = min(float(scales[1]), float(scales[0]))
            if scaling == "linear":
                scales = [int(low + i * (up - low) / (num - 1)) for i in range(num)]
            elif scaling == "power":
                factor = (up / low) ** (1 / (num - 1))
                scales = [int(round(low * factor ** i)) for i in range(num)]
            else:
                print("Scaling type: " + str(scaling) + " not recognized. Options are: [linear, power]")
        else:
            scales = [int(s) for s in scales]
        return scales

    image_sizes = opt_to_ints(opt.image_sizes, opt.size_scaling, opt.num_scales)
    num_iters = opt_to_ints(opt.num_iterations, opt.iter_scaling, opt.num_scales)
    return image_sizes, num_iters


def get_flow_model(opt):
    if not opt.model.gpu == "cpu":
        import unflow

        model = unflow.UnFlow(opt.flow).cuda().eval()
        th.backends.cudnn.enabled = opt.model.backend == "cudnn"

        def process(im1, im2):
            tens1 = th.FloatTensor(im1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)[:3]
            tens2 = th.FloatTensor(im2[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)[:3]

            assert tens1.size(1) == tens2.size(1)
            assert tens1.size(2) == tens2.size(2)

            h, w = tens1.size()[1], tens1.size(2)

            prep1 = tens1.cuda().view(1, 3, h, w)
            prep2 = tens2.cuda().view(1, 3, h, w)

            evenW = int(math.floor(math.ceil(w / 64.0) * 64.0))
            evenH = int(math.floor(math.ceil(h / 64.0) * 64.0))

            prep1 = F.interpolate(input=prep1, size=(evenH, evenW), mode="bilinear", align_corners=False)
            prep2 = F.interpolate(input=prep2, size=(evenH, evenW), mode="bilinear", align_corners=False)

            output = F.interpolate(input=model(prep1, prep2), size=(h, w), mode="bilinear", align_corners=False)
            output[:, 0, :, :] *= float(evenW) / float(w)
            output[:, 1, :, :] *= float(evenH) / float(h)
            output = output[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
            return output

        return process
    else:
        from thoth.deepmatching import deepmatching
        from thoth.deepflow2 import deepflow2

        return lambda im1, im2: deepflow2(im1, im2, deepmatching(im1, im2), "-%s" % (opt.flow.model_type_cpu))


def img_img(opt):
    name = lambda s: s.split("/")[-1].split(".")[0]
    output = (
        opt.output_dir
        + name(opt.input.content)
        + "_"
        + "_".join([name(s) for s in opt.input.style.split(",")])
        + ".png"
    )

    style_images_big = load.process_style_images(opt)
    content_image_big = match_histogram(load.preprocess(opt.input.content), style_images_big)
    content_size = np.array(content_image_big.size()[-2:])
    if opt.input.init != "content" and opt.input.init != "random":
        init_image = load.preprocess(opt.input.init)

    for i, (current_size, num_iters) in enumerate(zip(*determine_scaling(opt.param))):
        # print("\nCurrent size {}px".format(current_size))

        net = models.load_model(opt.model, opt.param)

        content_scale = current_size / max(*content_size)
        net.set_content_targets(content_image_big, content_scale, opt)
        net.set_style_targets(style_images_big, content_scale * content_size, opt)

        # Initialize the image
        if opt.input.init == "random":
            B, C, H, W = 1, 3, content_scale * content_size
            init_image = th.randn(C, H, W).mul(0.001).unsqueeze(0)
        elif opt.input.init == "content":
            init_image = F.interpolate(
                content_image_big.clone(), tuple(np.int64(content_scale * content_size)), mode="bilinear"
            )
        else:
            init_image = F.interpolate(
                init_image.clone(), tuple(np.int64(content_scale * content_size)), mode="bilinear"
            )

        if current_size <= 1664:
            opt.model.gpu = 1
            opt.model.multidevice = False
        else:
            opt.model.gpu = "0,1"
            opt.model.multidevice = True
            opt.param.tv_weight = 0
        opt.param.num_iterations = num_iters
        opt.optim.print_iter = num_iters // 4
        opt.output = re.sub(r"(\..*)", "_{}".format(current_size) + r"\1", output)

        output_image = net.optimize(init_image, opt)

        init_image = match_histogram(output_image.detach().cpu(), style_images_big)

    disp = load.deprocess(init_image.clone())
    if opt.param.original_colors == 1:
        disp = load.original_colors(load.deprocess(content_image.clone()), disp)
    disp.save(str(output))


def vid_img(opt):
    name = lambda s: s.split("/")[-1].split(".")[0]
    output_dir = (
        opt.output_dir + "/" + name(opt.input.content) + "_" + "_".join([name(s) for s in opt.input.style.split(",")])
    )

    flow_model = get_flow_model(opt)
    frames = load.process_content_video(flow_model, opt)
    content_size = np.array(load.preprocess(frames[0]).size()[-2:])

    style_images_big = load.process_style_images(opt)

    for size_n, (current_size, num_iters) in enumerate(zip(*determine_scaling(opt.param))):
        print("\nCurrent size {}px".format(current_size))
        os.makedirs(output_dir + "/" + str(current_size), exist_ok=True)
        content_scale = current_size / max(*content_size)

        if current_size <= 1024:
            opt.model.gpu = 0
            opt.model.multidevice = False
        else:
            opt.model.gpu = "0,1"
            opt.model.multidevice = True
            opt.param.tv_weight = 0
        net = models.load_model(opt.model, opt.param)
        net.set_style_targets(style_images_big, content_scale * content_size, opt)

        for pass_n in range(opt.param.passes_per_scale):
            init_image = None
            for frame_n, (prev_frame, this_frame) in enumerate(zip(frames, frames[1:] + frames[:1])):
                # TODO add update_style() function to support changing styles per frame
                opt.output = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n + 1, name(this_frame))
                if os.path.isfile(opt.output):
                    print("Skipping pass: %s, frame: %s. File already exists." % (pass_n + 1, name(this_frame)))
                    continue
                print("Optimizing... size: %s, pass: %s, frame: %s" % (current_size, pass_n + 1, name(this_frame)))

                content_frames = [
                    F.interpolate(load.preprocess(prev_frame), scale_factor=content_scale, mode="bilinear"),
                    F.interpolate(load.preprocess(this_frame), scale_factor=content_scale, mode="bilinear"),
                ]
                noise = th.randn(content_frames[0].size()).mul(0.001)  # prevents singular matrix errors in hist match?
                content_frames = [match_histogram(f + noise, style_images_big[0]) for f in content_frames]
                net.set_content_targets(content_frames[1], opt=opt)

                # Initialize the image
                # TODO make sure initialization correct even when continuing half way through video stylization
                if size_n == 0 and pass_n == 0:
                    if opt.input.init == "random":
                        init_image = th.randn(content_frames[1].size()).mul(0.001)
                    elif opt.input.init == "prev_warp":
                        flo_file = "%s/flow/forward_%s_%s.flo" % (output_dir, name(prev_frame), name(this_frame))
                        flow_map = load.flow_warp_map(flo_file, content_scale)
                        if init_image is None:
                            init_image = content_frames[0]
                        init_image = F.grid_sample(init_image, flow_map, padding_mode="border")
                    else:
                        init_image = content_frames[1].clone()
                else:
                    if pass_n == 0:
                        # load images from last pass of previous size
                        if init_image is None:
                            ifile = "%s/%s/%s_%s.png" % (
                                output_dir,
                                prev_size,
                                opt.param.passes_per_scale,
                                name(prev_frame),
                            )
                            init_image = load.preprocess(ifile)
                            init_image = F.interpolate(init_image, size=content_frames[0].size()[2:], mode="bilinear")
                        bfile = "%s/%s/%s_%s.png" % (
                            output_dir,
                            prev_size,
                            opt.param.passes_per_scale,
                            name(this_frame),
                        )
                        blend_image = load.preprocess(bfile)
                        blend_image = F.interpolate(blend_image, size=content_frames[0].size()[2:], mode="bilinear")
                    else:
                        # load images from previous pass of current size
                        if init_image is None:
                            ifile = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n, name(prev_frame))
                            init_image = load.preprocess(ifile)
                        bfile = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n, name(this_frame))
                        blend_image = load.preprocess(bfile)

                    direction = "forward" if pass_n % 2 == 0 else "backward"
                    flo_file = "%s/flow/%s_%s_%s.flo" % (output_dir, direction, name(prev_frame), name(this_frame))
                    flow_map = load.flow_warp_map(flo_file)
                    flow_map = F.interpolate(
                        flow_map.permute(0, 3, 1, 2), size=init_image.size()[2:], mode="bilinear"
                    ).permute(0, 2, 3, 1)

                    warp_image = F.grid_sample(init_image, flow_map, padding_mode="border")

                    flow_weight_file = "%s/flow/%s_%s_%s.png" % (
                        output_dir,
                        direction,
                        name(prev_frame),
                        name(this_frame),
                    )
                    reliable_flow = load.reliable_flow_weighting(flow_weight_file)
                    reliable_flow = F.interpolate(reliable_flow, size=init_image.size()[2:], mode="bilinear")

                    net.set_temporal_targets(warp_image, warp_weights=reliable_flow, opt=opt)

                    blend_init_image = (1 - opt.param.blend_weight) * blend_image + opt.param.blend_weight * init_image
                    warp_blend_init_image = F.grid_sample(blend_init_image, flow_map, padding_mode="border")
                    init_image = warp_blend_init_image

                # TODO fix this (atm required) magical num_iter incantations
                opt.param.num_iterations = num_iters // opt.param.passes_per_scale
                output_image = net.optimize(init_image, opt)

                init_image = match_histogram(output_image.detach().cpu(), style_images_big[0])

                disp = load.deprocess(init_image.clone())
                if opt.param.original_colors == 1:
                    disp = load.original_colors(load.deprocess(content_frames[1].clone()), disp)
                disp.save(str(opt.output))

            # clean up / prepare for next pass
            frames = frames[7:] + frames[:7]  # rotate frames
            frames = list(reversed(frames))

        ffmpeg.input(output_dir + "/" + str(current_size) + "/" + str(pass_n) + "_%05d.png").output(
            "%s/%s_%s.mp4" % (output_dir, name(output_dir), current_size), **opt.ffmpeg
        ).overwrite_output().run()
        prev_size = current_size
        del net
        th.cuda.empty_cache()

    ffmpeg.input(output_dir + "/" + str(current_size) + "/" + str(pass_n) + "_%05d.png").output(
        "%s.mp4" % (output_dir), **opt.ffmpeg
    ).overwrite_output().run()


opt = load_config("config/ub94.yaml")
if opt.transfer_type == "img_img":
    img_img(opt)
elif opt.transfer_type == "vid_img":
    vid_img(opt)
