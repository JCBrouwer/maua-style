import argparse
import re

import torch
import yaml


class Options(dict):
    """ Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessibly using attribute notation (Options.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse down nested dicts (like: Options.attr.attr)
    """

    def __init__(self, mapping=None):
        super(Options, self).__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Options(value)
        super(Options, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __str__(self):
        def print_recursive(d, col=0):
            ret = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    ret += "\t" * col + str(key) + ":\n" + print_recursive(value, col + 1)
                else:
                    col1 = str(key) + ": "
                    col2 = str(value)
                    ret += "\t" * col + "{:<30s}{:<30s}".format(col1, col2) + "\n"
            return ret

        return print_recursive(self.__dict__)

    __setattr__ = __setitem__


def FilePathString(v):
    try:
        return re.match(".*|random|content", v).group(0)
    except:
        raise argparse.ArgumentTypeError("String '%s' does not match required format" % (v,))


def parse_args():
    parser = argparse.ArgumentParser()

    # input options
    parser.add_argument(
        "-content", help="Content target image", default=argparse.SUPPRESS
    )  # ) #, default='examples/inputs/tubingen.jpg')
    parser.add_argument(
        "-style", help="Style target image", default=argparse.SUPPRESS
    )  # , default='examples/inputs/seated-nude.jpg')
    parser.add_argument("-output", default=argparse.SUPPRESS)  # , default='out.png')
    parser.add_argument("-init", type=FilePathString, default=argparse.SUPPRESS)  # , default='random')
    parser.add_argument("-style_blend_weights", default=argparse.SUPPRESS)  # , default=None)
    parser.add_argument("-style_scale", type=float, default=argparse.SUPPRESS)  # , default=1.0)
    parser.add_argument(
        "-image_size", help="Maximum height / width of generated image", default=argparse.SUPPRESS
    )  # ) #, default=512)
    parser.add_argument("-original_colors", type=int, choices=[0, 1], default=argparse.SUPPRESS)  # , default=0)
    parser.add_argument("-seed", type=int, default=argparse.SUPPRESS)  # , default=-1)

    # Optimization options
    parser.add_argument("-model_file", type=str, default=argparse.SUPPRESS)  # , default='models/vgg19-d01eb7cb.pth')
    parser.add_argument("-content_weight", type=float, default=argparse.SUPPRESS)  # , default=5e0)
    parser.add_argument("-temporal_weight", type=float, default=argparse.SUPPRESS)  # , default=5e0)
    parser.add_argument("-style_weight", type=float, default=argparse.SUPPRESS)  # , default=1e2)
    parser.add_argument("-normalize_weights", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("-normalize_gradients", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("-tv_weight", type=float, default=argparse.SUPPRESS)  # , default=1e-3)
    parser.add_argument("-num_frames", type=int, default=argparse.SUPPRESS)  # , default=1e-3)
    parser.add_argument("-avg_frame_window", type=int, default=argparse.SUPPRESS)  # , default=1e-3)
    parser.add_argument("-gram_frame_window", type=int, default=argparse.SUPPRESS)  # , default=1e-3)
    parser.add_argument("-use_covariance", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("-pooling", choices=["avg", "max"], default=argparse.SUPPRESS)  # , default='max')
    parser.add_argument("-content_layers", help="layers for content", default=argparse.SUPPRESS)  # , default='relu4_2')
    parser.add_argument(
        "-style_layers", help="layers for style", default=argparse.SUPPRESS
    )  # , default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
    parser.add_argument(
        "-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = c", default=argparse.SUPPRESS
    )  # ) #, default=0)
    parser.add_argument(
        "-backend",
        choices=["nn", "cudnn", "mkl", "mkldnn", "openmp", "mkl,cudnn", "cudnn,mkl"],
        default=argparse.SUPPRESS,
    )  # , default='nn')
    parser.add_argument("-multidevice_strategy", default=argparse.SUPPRESS)  # , default='4,7,29')
    parser.add_argument("-cudnn_autotune", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("-disable_check", action="store_true", default=argparse.SUPPRESS)

    # Output options
    parser.add_argument("-optimizer", choices=["lbfgs", "adam"], default=argparse.SUPPRESS)  # , default='lbfgs')
    parser.add_argument("-num_iterations", default=argparse.SUPPRESS)  # , default=1000)
    parser.add_argument("-learning_rate", type=float, default=argparse.SUPPRESS)  # , default=1e0)
    parser.add_argument("-lbfgs_num_correction", type=int, default=argparse.SUPPRESS)  # , default=100)
    parser.add_argument("-print_iter", type=int, default=argparse.SUPPRESS)  # , default=50)
    parser.add_argument("-save_iter", type=int, default=argparse.SUPPRESS)  # , default=100)

    return parser.parse_args()


# General config
def load_config(path, default_path=None):
    """ Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    args = parse_args()
    update_iterative(cfg, args.__dict__)

    cfg["model"]["dtype"], cfg["model"]["multidevice"], cfg["model"]["backward_device"] = setup_gpu(cfg)

    opt = Options(cfg)

    if opt.input.seed >= 0:
        torch.manual_seed(opt.input.seed)
        torch.cuda.manual_seed_all(opt.input.seed)
        if opt.model.backend == "cudnn":
            torch.backends.cudnn.deterministic = True

    print(opt)
    print()
    return opt


def update_recursive(dict1, dict2):
    """ Merge two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = None
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def update_iterative(dict1, dict2):
    for replace_key, replace_val in dict2.items():
        found = False
        for k in dict1.keys():
            if k == replace_key:
                dict1[k] = replace_val
                fount = True
            if isinstance(dict1[k], dict):
                for k2 in dict1[k].keys():
                    if replace_key == k2:
                        dict1[k][k2] = replace_val
                        found = True
        if not found:
            dict1[replace_key] = replace_val


def setup_gpu(cfg):
    def setup_cuda():
        if "cudnn" in cfg["model"]["backend"]:
            torch.backends.cudnn.enabled = True
            if cfg["model"]["cudnn_autotune"]:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False

    def setup_cpu():
        if "mkldnn" in cfg["model"]["backend"]:
            raise ValueError("MKL-DNN is not supported yet.")
        elif "mkl" in cfg["model"]["backend"]:
            torch.backends.mkl.enabled = True
        elif "openmp" in cfg["model"]["backend"]:
            torch.backends.openmp.enabled = True

    multidevice = False
    if "," in str(cfg["model"]["gpu"]):
        devices = cfg["model"]["gpu"].split(",")
        multidevice = True
        if "c" in str(devices[0]).lower():
            backward_device = "cpu"
            setup_cuda(), setup_cpu()
            dtype = torch.FloatTensor
        else:
            backward_device = "cuda:" + devices[0]
            setup_cuda()
            dtype = torch.cuda.FloatTensor
    elif "c" not in str(cfg["model"]["gpu"]).lower():
        setup_cuda()
        dtype, backward_device = torch.cuda.FloatTensor, "cuda:" + str(cfg["model"]["gpu"])
    else:
        setup_cpu()
        dtype, backward_device = torch.FloatTensor, "cpu"

    return dtype, multidevice, backward_device
