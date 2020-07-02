import argparse

import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        parser.add_argument(
            "--dataroot",
            required=True,
            help="path to images (should have subfolders trainA, trainB, valA, valB, etc)",
        )
        parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument(
            "--dataset_mode",
            type=str,
            default="unaligned",
            help="chooses how datasets are loaded. [unaligned | aligned | single]",
        )
        parser.add_argument(
            "--nThreads", default=4, type=int, help="# threads for loading data"
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )
        parser.add_argument(
            "--norm",
            type=str,
            default="instance",
            help="instance normalization or batch normalization",
        )
        parser.add_argument(
            "--serial_batches",
            action="store_true",
            help="if true, takes images in order to make batches, otherwise takes them randomly",
        )
        parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="network initialization [normal|xavier|kaiming|orthogonal]",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="if specified, print more debugging information",
        )
        parser.add_argument(
            "--suffix",
            default="",
            type=str,
            help="customized suffix: opt.name = opt.name + suffix: "
            "e.g., {model}_{which_model_netG}_size{loadSize}",
        )
        parser.add_argument("--num_volumes_train", type=int, default=None)
        parser.add_argument("--num_volumes_val", type=int, default=None)

        # adding for my project fmri
        parser.add_argument(
            "--kspace_keep_ratio",
            type=float,
            default=0.25,
            help="mask raio of kspace lines",
        )
        parser.add_argument(
            "--normalize_type",
            default="gan",
            type=str,
            choices=["gan", "zero_one", "imagenet", "cae"],
            help="normalizing type",
        )
        parser.add_argument(
            "--eval_full_valid",
            action="store_true",
            help="if specified, evaluate the full validation set, otherwised 10%",
        )
        parser.add_argument(
            "--nz", type=int, default=8, help="dimension of prior/posterior"
        )
        parser.add_argument(
            "--non_strict_state_dict",
            action="store_true",
            help="if specified, load dict non-strictly. "
            "Sometimes needed to avoid the naming issues (not sure why)",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=16,
            help="for vae models, the number of samples in sampling.py",
        )

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = None
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                allow_abbrev=False,
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

    def parse(self):

        opt = self.gather_options()

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        # for str_id in str_ids:
        for str_id in range(len(str_ids)):
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            opt.batchSize *= len(opt.gpu_ids)
            print(
                f"Use multiple GPUs, batchSize are increased by {len(opt.gpu_ids)} "
                f"times to {opt.batchSize}"
            )

        self.print_options(opt)

        return opt
