# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import base_options


class TrainOptions(base_options.BaseOptions):
    def initialize(self, parser):
        parser = base_options.BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--beta1", type=float, default=0.5, help="momentum term of adam"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--mask_type",
            type=str,
            choices=[
                "basic",
                "symmetric_basic",
                "low_to_high",
                "grid",
                "symmetric_grid",
                "basic_rnl",
                "symmetric_basic_rnl",
                "low_to_high_rnl",
            ],
            help="The type of mask to use.",
        )
        parser.add_argument(
            "--rnl_params",
            type=str,
            default=None,
            help="Characterizes the distribution of initial masks (when these are sampled, see "
            "--train_with_fixed_initial_mask). "
            "Format is min_lowf_lines,max_lowf_lines,highf_beta_alpha,highf_beta_beta. "
            "Mask have a random number of low frequency lines active, uniform between "
            "min_lowf_lines and max_lowf_lines. The remaining number of lines is determined by "
            "a Beta(highf_beta_alpha, highf_beta_beta) distribution, which indicates the "
            "proportion of the remaining lines to sample.",
        )
        parser.add_argument(
            "--debug", action="store_true", help="Activates debug level messages."
        )

        parser.add_argument(
            "--add_mask_eval",
            action="store_true",
            help="Sum mask values to observation in evaluator model.",
        )
        parser.add_argument("--weights_checkpoint", type=str, default=None)
        # parser.add_argument("--validation_train_split_ratio", type=float, default=0.9)
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=100,
            help="number of epochs to train (default: 5)",
        )
        # parser.add_argument("--save_freq", type=int, default=200)

        # Options for Reconstruction Model
        parser.add_argument("--number_of_reconstructor_filters", type=int, default=128)
        parser.add_argument("--dropout_probability", type=float, default=0)
        parser.add_argument("--number_of_cascade_blocks", type=int, default=3)
        parser.add_argument(
            "--number_of_layers_residual_bottleneck", type=int, default=6
        )
        parser.add_argument("--n_downsampling", type=int, default=3)
        parser.add_argument("--use_deconv", type=bool, default=True)

        # Options for Evaluator Model
        parser.add_argument(
            "--no_evaluator", dest="use_evaluator", action="store_false"
        )
        parser.add_argument("--number_of_evaluator_filters", type=int, default=128)
        parser.add_argument(
            "--number_of_evaluator_convolution_layers", type=int, default=4
        )

        # Options for both Reconstructor and Evaluator Model
        parser.add_argument("--mask_embed_dim", type=int, default=6)
        parser.add_argument("--image_width", type=int, default=128)

        # Options moved from old model file
        parser.add_argument(
            "--use_mse_as_disc_energy",
            action="store_true",
            help="use MSE as evaluator energy",
        )
        parser.add_argument(
            "--grad_ctx",
            action="store_true",
            help="GAN criterion computes adversarial loss signal at provided k-space lines.",
        )
        parser.add_argument(
            "--lambda_gan",
            type=float,
            default=0.01,
            help="Weight for reconstruction loss.",
        )
        parser.add_argument("--gamma", type=int, default=100)

        parser.add_argument(
            "--only_evaluator", dest="only_evaluator", action="store_true"
        )

        return parser
