import os
import os
import argparse
import sys
import pickle

class Args(argparse.ArgumentParser):
    """
        Contains global default arguments for experimentation.
        Override by passing in a dict on initialization.
    """

    def __init__(self, **overrides):
        super().__init__()
        self.add_argument('--seed', default=42, type=int)
        self.add_argument('--batch_size', default=128, type=int)
        self.add_argument('--eval_batch_size', default=-1, type=int,
            help="Larger batches can be used during eval as less memory is needed")
        self.add_argument('--subsample_reuse_mask', default=True, type=int)
        self.add_argument('--subsampling_seed', default=42, type=int)
        self.add_argument('--subsampling_ratio', default=4, type=int,
            help="Integer ratio of non-sampled to sampled rows. I.e. 4 indicates 25% of rows kept")
        self.add_argument('--subsampling_type', default="lf_focused", type=str,
            choices=['random', 'alternating', 'alternating_plus_lf', 'fromfile', 'lf_focused'])
        self.add_argument('--subsample_mask_file', default="data/mask128.png", type=str,
            help="If fromfile is specified for subsampling_type, this is the file used")
        self.add_argument('--center_fraction', default=0.078125, type=float)
        self.add_argument('--workers', default=4, type=int)
        self.add_argument('--distributed', default=False, type=bool, help="Distributed training")
        self.add_argument('--resolution', default=128, type=int)
        self.add_argument('--apex', default=False, type=bool, help="Turn on NVIDIA Apex half precision")
        self.add_argument('--log_every', default=20, type=int)
        self.add_argument('--log_during_eval', default=False, type=bool)
        self.add_argument('--block_type', default="bottleneck", type=str)
        self.add_argument('--adam_eps', default=1e-08, type=float)

        if "H2" in os.environ:
            dicom_root = '/checkpoint/jzb/data/mmap'
        else:
            dicom_root = '/checkpoint/jzb/data/mmap'    #Sumana : changed '/checkpoint/adefazio/fast-mri/data/mmap'

        self.add_argument('--dicom_root', default=dicom_root, type=str)

        # Override defaults with passed overrides
        self.set_defaults(**overrides)

        ## Some run specific context that we want globally accessible and saved out
        ## at the end of the run.
        self.set_defaults(
            main_pid = os.getpid(),
            cwd = os.getcwd()
        )

    def parse_args(self, args=[]):
        if len(sys.argv) == 2 and "--" not in sys.argv[1] and ".pkl" in sys.argv[1]:
            run_config_file = sys.argv[1]
            run_config = pickle.load(open(run_config_file, 'rb'))
            if not isinstance(run_config, list):
                print(f"Found single argument config file: {run_config_file}. Overriding defaults")
                #run_config = run_config[0]
                self.set_defaults(**run_config)

        return super().parse_args(args)
