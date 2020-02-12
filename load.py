import os.path, math, itertools
from PIL import Image
import torch as th
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import scipy.ndimage, scipy.misc


Image.MAX_IMAGE_PIXELS = 1000000000  # Support gigapixel images


# Preprocess an image before passing it to a model.
# We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(image_name):
    image = Image.open(image_name).convert("RGB")
    rgb2bgr = T.Lambda(lambda x: x[th.LongTensor([2, 1, 0])])
    normalize = T.Normalize(mean=[103.939, 116.779, 123.68], std=[1, 1, 1])
    return normalize(rgb2bgr(T.ToTensor()(image) * 256)).unsqueeze(0)


#  Undo the above preprocessing.
def deprocess(output_tensor):
    normalize = T.Normalize(mean=[-103.939, -116.779, -123.68], std=[1, 1, 1])
    bgr2rgb = T.Lambda(lambda x: x[th.LongTensor([2, 1, 0])])
    output_tensor = bgr2rgb(normalize(output_tensor.squeeze(0).cpu())) / 256
    output_tensor.clamp_(0, 1)
    return T.ToPILImage()(output_tensor.cpu())


def process_style_images(opt):
    style_image_input = opt.input.style.split(",")
    style_image_list, ext = [], [".jpg", ".jpeg", ".png", ".tiff"]
    for image in style_image_input:
        if os.path.isdir(image):
            images = (image + "/" + file for file in os.listdir(image) if os.path.splitext(file)[1].lower() in ext)
            style_image_list.extend(images)
        else:
            style_image_list.append(image)
    style_images = []
    for image in style_image_list:
        style_images.append(preprocess(image))

    # Handle style blending weights for multiple style inputs
    style_blend_weights = []
    if opt.param.style_blend_weights is False:
        # Style blending not specified, so use equal weighting
        for i in style_image_list:
            style_blend_weights.append(1.0)
    else:
        style_blend_weights = [float(x) for x in opt.param.style_blend_weights.split(",")]
        assert len(style_blend_weights) == len(
            style_image_list
        ), "-style_blend_weights and -style_images must have the same number of elements!"

    # Normalize the style blending weights so they sum to 1
    style_blend_sum = sum(style_blend_weights)
    for i, blend_weight in enumerate(style_blend_weights):
        style_blend_weights[i] = blend_weight / style_blend_sum

    opt.param.style_blend_weights = style_blend_weights

    return style_images


# extract frames from video, calculate optical flow in forward and backward direction, save as flo and png files
def process_content_video(model, opt):
    import flow
    import ffmpeg

    name = lambda s: s.split("/")[-1].split(".")[0]
    work_dir = (
        opt.output_dir + "/" + name(opt.input.content) + "_" + "_".join([name(s) for s in opt.input.style.split(",")])
    )
    frames_dir = work_dir + "/frames/"
    flow_dir = work_dir + "/flow/"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)

    if len(os.listdir(frames_dir)) == 0:
        ffmpeg.input(opt.input.content).output(frames_dir + "/%05d.png").run()

    with th.no_grad():
        images = [frames_dir + file for file in sorted(os.listdir(frames_dir)) if ".png" in file and "_" not in file]
        images.append(images[0])
        for img_file1, img_file2 in zip(*(itertools.islice(images, i, None) for i in range(2))):
            if os.path.isfile("%s/backward_%s_%s.png" % (flow_dir, name(img_file2), name(img_file1))):
                continue
            img1 = np.array(Image.open(img_file1))
            img2 = np.array(Image.open(img_file2))

            forward_flow = model(img1, img2)

            write_flow(forward_flow, "%s/forward_%s_%s.flo" % (flow_dir, name(img_file1), name(img_file2)))
            # forward_flow_img = flow.flow_to_image(forward_flow)
            # Image.fromarray(forward_flow_img).save("%s/forward_%s_%s.png"%(flow_dir, name(img_file1), name(img_file2)))

            backward_flow = model(img2, img1)

            write_flow(backward_flow, "%s/backward_%s_%s.flo" % (flow_dir, name(img_file2), name(img_file1)))
            # backward_flow_img = flow.flow_to_image(backward_flow)
            # Image.fromarray(backward_flow_img).save("%s/backward_%s_%s.png"%(flow_dir, name(img_file2), name(img_file1)))

            reliable_flow_arr = flow.check_consistency(forward_flow, backward_flow)
            reliable_flow_img = Image.fromarray(((1 - reliable_flow_arr) * 255).astype(np.uint8)).convert("L")
            reliable_flow_img.save("%s/forward_%s_%s.png" % (flow_dir, name(img_file1), name(img_file2)))

            reliable_flow_arr = flow.check_consistency(backward_flow, forward_flow)
            reliable_flow_img = Image.fromarray(((1 - reliable_flow_arr) * 255).astype(np.uint8)).convert("L")
            reliable_flow_img.save("%s/backward_%s_%s.png" % (flow_dir, name(img_file2), name(img_file1)))

            print("processed optical flow: %s <---> %s" % (name(img_file1), name(img_file2)))

    images.pop(-1)
    return images


def flow_warp_map(filename):
    f = open(filename, "rb")
    magic = np.fromfile(f, np.float32, count=1)
    flow = None
    if 202021.25 != magic:
        print("Magic number incorrect. Invalid .flo file")
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print("Reading %d x %d flow file in .flo format" % (w, h))
        flow = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
        # reshape data into 3D array (columns, rows, channels)
        flow = np.resize(flow, (int(h[0]), int(w[0]), 2))
        flow[:, :, 0] /= int(w[0])
        flow[:, :, 1] /= int(h[0])
        flow = scipy.ndimage.gaussian_filter(flow, [5, 5, 0])
    f.close()
    neutral = np.array(np.meshgrid(np.linspace(-1, 1, int(w[0])), np.linspace(-1, 1, int(h[0]))))
    neutral = np.rollaxis(neutral, 0, 3)
    warp_map = th.FloatTensor(neutral + flow).unsqueeze(0)
    return warp_map


def reliable_flow_weighting(filename):
    return T.ToTensor()(Image.open(filename)).unsqueeze(0)


def write_flow(flow, filename):
    f = open(filename, "wb")
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


# Combine the Y channel of the generated image and the UV/CbCr channels of the
# content image to perform color-independent style transfer.
def original_colors(content, generated):
    content_channels = list(content.convert("YCbCr").split())
    generated_channels = list(generated.convert("YCbCr").split())
    content_channels[0] = generated_channels[0]
    return Image.merge("YCbCr", content_channels).convert("RGB")