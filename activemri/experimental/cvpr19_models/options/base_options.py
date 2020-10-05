# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        parser.add_argument(
            "--dataset_dir", required=True, help="Path to fastmri dataset."
        )
        parser.add_argument(
            "--dataroot",
            required=True,
            help="Path to images (should have subfolders trainA, trainB, valA, valB, etc)",
        )
        parser.add_argument(
            "--batchSize", type=int, default=1, help="Input batch size."
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="GPU IDs: e.g. 0  0,1,2, 0,2. use -1 for CPU.",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="Name of the experiment. It determines the sub folder where results are stored.",
        )
        parser.add_argument(
            "--nThreads", default=4, type=int, help="Number of threads for data loader."
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="Root directory to save results and model checkpoints.",
        )
        parser.add_argument(
            "--init_type",
            type=str,
            choices=["normal", "xavier", "kaiming", "orthogonal"],
            default="normal",
            help="Network weights initialization type.",
        )

        parser.add_argument(
            "--num_volumes_train",
            type=int,
            default=None,
            help="Number of MRI volumes to use for training.",
        )
        parser.add_argument(
            "--num_volumes_val",
            type=int,
            default=None,
            help="Number of MRI volumes to use for validation.",
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

    def parse(self, silent=True):

        opt = self.gather_options()

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

        if not silent:
            self.print_options(opt)

        return opt
