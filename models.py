import torch
import torch.nn as nn
from loss import *
import copy
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url
import style
from types import MethodType


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


class VGG_SOD(nn.Module):
    def __init__(self, features, num_classes=100):
        super(VGG_SOD, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 100),
        )


class VGG_FCN32S(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_FCN32S, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, (7, 7)),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, (1, 1)),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


class VGG_PRUNED(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_PRUNED, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


class NIN(nn.Module):
    def __init__(self, pooling):
        super(NIN, self).__init__()
        if pooling == "max":
            pool2d = nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        elif pooling == "avg":
            pool2d = nn.AvgPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), (4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, (1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, (1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Dropout(0.5),
            nn.Conv2d(384, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1000, (1, 1)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((6, 6), (1, 1), (0, 0), ceil_mode=True),
            nn.Softmax(),
        )


def build_sequential(channel_list, pooling):
    layers = []
    in_channels = 3
    if pooling == "max":
        pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    elif pooling == "avg":
        pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    else:
        raise ValueError("Unrecognized pooling parameter")
    for c in channel_list:
        if c == "P":
            layers += [pool2d]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


channel_list = {
    "VGG-16p": [24, 22, "P", 41, 51, "P", 108, 89, 111, "P", 184, 276, 228, "P", 512, 512, 512, "P"],
    "VGG-16": [64, 64, "P", 128, 128, "P", 256, 256, 256, "P", 512, 512, 512, "P", 512, 512, 512, "P"],
    "VGG-19": [64, 64, "P", 128, 128, "P", 256, 256, 256, 256, "P", 512, 512, 512, 512, "P", 512, 512, 512, 512, "P"],
}
nin_dict = {
    "C": [
        "conv1",
        "cccp1",
        "cccp2",
        "conv2",
        "cccp3",
        "cccp4",
        "conv3",
        "cccp5",
        "cccp6",
        "conv4-1024",
        "cccp7-1024",
        "cccp8-1024",
    ],
    "R": [
        "relu0",
        "relu1",
        "relu2",
        "relu3",
        "relu5",
        "relu6",
        "relu7",
        "relu8",
        "relu9",
        "relu10",
        "relu11",
        "relu12",
    ],
    "P": ["pool1", "pool2", "pool3", "pool4"],
    "D": ["drop"],
}
vgg16_dict = {
    "C": [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "conv5_1",
        "conv5_2",
        "conv5_3",
    ],
    "R": [
        "relu1_1",
        "relu1_2",
        "relu2_1",
        "relu2_2",
        "relu3_1",
        "relu3_2",
        "relu3_3",
        "relu4_1",
        "relu4_2",
        "relu4_3",
        "relu5_1",
        "relu5_2",
        "relu5_3",
    ],
    "P": ["pool1", "pool2", "pool3", "pool4", "pool5"],
}
vgg19_dict = {
    "C": [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv3_4",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "conv4_4",
        "conv5_1",
        "conv5_2",
        "conv5_3",
        "conv5_4",
    ],
    "R": [
        "relu1_1",
        "relu1_2",
        "relu2_1",
        "relu2_2",
        "relu3_1",
        "relu3_2",
        "relu3_3",
        "relu3_4",
        "relu4_1",
        "relu4_2",
        "relu4_3",
        "relu4_4",
        "relu5_1",
        "relu5_2",
        "relu5_3",
        "relu5_4",
    ],
    "P": ["pool1", "pool2", "pool3", "pool4", "pool5"],
}


def select_model(model_file, pooling):
    vgg_list = ["fcn32s", "pruning", "sod", "vgg"]
    if any(name in model_file for name in vgg_list):
        if "pruning" in model_file:
            print("VGG-16 Architecture Detected")
            print("Using The Channel Pruning Model")
            cnn, layerList = VGG_PRUNED(build_sequential(channel_list["VGG-16p"], pooling)), vgg16_dict
        elif "fcn32s" in model_file:
            print("VGG-16 Architecture Detected")
            print("Using the fcn32s-heavy-pascal Model")
            cnn, layerList = VGG_FCN32S(build_sequential(channel_list["VGG-16"], pooling)), vgg16_dict
        elif "sod" in model_file:
            print("VGG-16 Architecture Detected")
            print("Using The SOD Fintune Model")
            cnn, layerList = VGG_SOD(build_sequential(channel_list["VGG-16"], pooling)), vgg16_dict
        elif "19" in model_file:
            print("VGG-19 Architecture Detected")
            if not path.exists(model_file):
                # Download the VGG-19 model and fix the layer names
                print("Model file not found: " + model_file)
                sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth")
                map = {
                    "classifier.1.weight": u"classifier.0.weight",
                    "classifier.1.bias": u"classifier.0.bias",
                    "classifier.4.weight": u"classifier.3.weight",
                    "classifier.4.bias": u"classifier.3.bias",
                }
                sd = OrderedDict([(map[k] if k in map else k, v) for k, v in sd.items()])
                torch.save(sd, path.join("models", "vgg19-d01eb7cb.pth"))
            cnn, layerList = VGG(build_sequential(channel_list["VGG-19"], pooling)), vgg19_dict
        elif "16" in model_file:
            print("VGG-16 Architecture Detected")
            if not path.exists(model_file):
                # Download the VGG-16 model and fix the layer names
                print("Model file not found: " + model_file)
                sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
                map = {
                    "classifier.1.weight": u"classifier.0.weight",
                    "classifier.1.bias": u"classifier.0.bias",
                    "classifier.4.weight": u"classifier.3.weight",
                    "classifier.4.bias": u"classifier.3.bias",
                }
                sd = OrderedDict([(map[k] if k in map else k, v) for k, v in sd.items()])
                torch.save(sd, path.join("models", "vgg16-00b39a1b.pth"))
            cnn, layerList = VGG(build_sequential(channel_list["VGG-16"], pooling)), vgg16_dict
        else:
            raise ValueError("VGG architecture not recognized.")
    elif "nin" in model_file:
        print("NIN Architecture Detected")
        if not path.exists(model_file):
            # Download the NIN model
            print("Model file not found: " + model_file)
            if version_info[0] < 3:
                import urllib

                urllib.URLopener().retrieve(
                    "https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth",
                    path.join("models", "nin_imagenet.pth"),
                )
            else:
                import urllib.request

                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth",
                    path.join("models", "nin_imagenet.pth"),
                )
        cnn, layerList = NIN(pooling), nin_dict
    else:
        raise ValueError("Model architecture not recognized.")
    return cnn, layerList


# Load the model, and configure pooling layer type
def load_model(opt, param):
    cnn, layer_list = select_model(str(opt.model_file).lower(), opt.pooling)

    cnn.load_state_dict(torch.load(opt.model_file), strict=(not opt.disable_check))
    print("Successfully loaded " + str(opt.model_file))

    # Maybe convert the model to cuda now, to avoid later issues
    if "c" not in str(opt.gpu).lower() or "c" not in str(opt.gpu[0]).lower():
        cnn = cnn.cuda()
    cnn = cnn.features

    content_layers = opt.content_layers.split(",")
    style_layers = opt.style_layers.split(",")

    # Set up the network, inserting style and content loss modules
    cnn = copy.deepcopy(cnn)
    content_losses, style_losses, tv_losses, temporal_losses = [], [], [], []
    next_content_idx, next_style_idx = 1, 1
    net = nn.Sequential()
    c, r = 0, 0

    if param.tv_weight > 0:
        tv_mod = TVLoss(param.tv_weight)
        net.add_module(str(len(net)), tv_mod)
        tv_losses.append(tv_mod)

    # HACK abuse of Options class (which is a dict) to avoid error here when temporal_weight not in img/img config
    if param.get("temporal_weight", 0) > 0:
        temporal_mod = ContentLoss(param.temporal_weight)
        net.add_module(str(len(net)), temporal_mod)
        temporal_losses.append(temporal_mod)

    for i, layer in enumerate(list(cnn), 1):
        if next_content_idx <= len(content_layers) or next_style_idx <= len(style_layers):
            if isinstance(layer, nn.Conv2d):
                net.add_module(str(len(net)), layer)

                if layer_list["C"][c] in content_layers:
                    print("Setting up content layer " + str(i) + ": " + str(layer_list["C"][c]))
                    loss_module = ContentLoss(param.content_weight)
                    net.add_module(str(len(net)), loss_module)
                    content_losses.append(loss_module)

                if layer_list["C"][c] in style_layers:
                    print("Setting up style layer " + str(i) + ": " + str(layer_list["C"][c]))
                    loss_module = StyleLoss(param.style_weight, param.use_covariance)
                    net.add_module(str(len(net)), loss_module)
                    style_losses.append(loss_module)
                c += 1

            if isinstance(layer, nn.ReLU):
                net.add_module(str(len(net)), layer)

                if layer_list["R"][r] in content_layers:
                    print("Setting up content layer " + str(i) + ": " + str(layer_list["R"][r]))
                    loss_module = ContentLoss(param.content_weight)
                    net.add_module(str(len(net)), loss_module)
                    content_losses.append(loss_module)
                    next_content_idx += 1

                if layer_list["R"][r] in style_layers:
                    print("Setting up style layer " + str(i) + ": " + str(layer_list["R"][r]))
                    loss_module = StyleLoss(param.style_weight, param.use_covariance)
                    net.add_module(str(len(net)), loss_module)
                    style_losses.append(loss_module)
                    next_style_idx += 1
                r += 1

            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                net.add_module(str(len(net)), layer)

    if opt.multidevice:
        net = setup_multi_device(net, opt.gpu, opt.multidevice_strategy)

    # Freeze the network in order to prevent unnecessary gradient calculations
    for param in net.parameters():
        param.requires_grad = False

    net.content_losses = content_losses
    net.style_losses = style_losses
    net.tv_losses = tv_losses
    net.temporal_losses = temporal_losses
    net.set_style_targets = MethodType(style.set_style_targets, net)
    net.set_content_targets = MethodType(style.set_content_targets, net)
    net.set_temporal_targets = MethodType(style.set_temporal_targets, net)
    net.optimize = MethodType(style.optimize, net)

    return net


class NewModelParallel(nn.Module):
    def __init__(self, net, device_ids, device_splits):
        super(ModelParallel, self).__init__()
        self.device_list = self.name_devices(device_ids.split(","))
        self.chunks = self.chunks_to_devices(self.split_net(net, device_splits.split(",")))

    def name_devices(self, input_list):
        device_list = []
        for i, device in enumerate(input_list):
            if str(device).lower() != "c":
                device_list.append("cuda:" + str(device))
            else:
                device_list.append("cpu")
        return device_list

    def split_net(self, net, device_splits):
        chunks, cur_chunk = [], nn.Sequential()
        for i, l in enumerate(net):
            cur_chunk.add_module(str(i), net[i])
            if str(i) in device_splits and device_splits != "":
                del device_splits[0]
                chunks.append(cur_chunk)
                cur_chunk = nn.Sequential()
        chunks.append(cur_chunk)
        return chunks

    def chunks_to_devices(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk.to(self.device_list[i])
        return chunks

    def c(self, input, i):
        if input.type() == "torch.FloatTensor" and "cuda" in self.device_list[i]:
            input = input.type("torch.cuda.FloatTensor")
        elif input.type() == "torch.cuda.FloatTensor" and "cpu" in self.device_list[i]:
            input = input.type("torch.FloatTensor")
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) - 1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i + 1).to(self.device_list[i + 1])
            else:
                input = chunk(input)
        return input


class ModelParallel(nn.Module):
    def __init__(self, chunks, device_list):
        super(ModelParallel, self).__init__()
        self.chunks = chunks
        self.device_list = device_list

    def c(self, input, i):
        if input.type() == "torch.FloatTensor" and "cuda" in self.device_list[i]:
            input = input.type("torch.cuda.FloatTensor")
        elif input.type() == "torch.cuda.FloatTensor" and "cpu" in self.device_list[i]:
            input = input.type("torch.FloatTensor")
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) - 1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i + 1).to(self.device_list[i + 1])
            else:
                input = chunk(input)
        return input


def setup_new_multi_device(net, gpu, multidevice_strategy):
    assert len(str(gpu).split(",")) - 1 == len(
        str(multidevice_strategy).split(",")
    ), "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."

    new_net = NewModelParallel(net, str(gpu), str(multidevice_strategy))
    return new_net


def setup_multi_device(net, gpu, multidevice_strategy):
    device_splits = str(multidevice_strategy).split(",")
    gpu = str(gpu).split(",")

    assert len(gpu) - 1 == len(
        device_splits
    ), "The number of -multidevice_strategy layer indices must be equal to the number of -gpu devices minus 1."

    device_list = []
    for i, device in enumerate(gpu):
        if str(device).lower() != "c":
            device_list.append("cuda:" + str(device))
        else:
            device_list.append("cpu")

    cur_chunk = nn.Sequential()
    chunks = []
    for i, l in enumerate(net):
        cur_chunk.add_module(str(i), net[i])
        if str(i) in device_splits and device_splits != "":
            del device_splits[0]
            chunks.append(cur_chunk)
            cur_chunk = nn.Sequential()
    chunks.append(cur_chunk)

    for i, chunk in enumerate(chunks):
        chunk.to(device_list[i])

    new_net = ModelParallel(chunks, device_list)
    return new_net
