from options.train_options import TrainOptions
from trainer import Trainer
from util.hyperband import HyperbandTuner
from util.submitit_function_evaluator import SubmititEvaluator

import argparse
import os
import torch


# TODO write some unit tests for the tuning code part
def main(options):
    # Create a function evaluator to be passed to the tuner. Here you can pass the SLURM arguments as keywords.
    function_evaluator = SubmititEvaluator(Trainer, options, options.submitit_logs_dir, 3,
                                           job_name='hyperband_test',
                                           time=4320, partition='learnfair', num_gpus=8, cpus_per_task=16)

    # Specify the hyperparameter names and their possible values (for now only categorical distributions are supported).
    categorical_hp_classes = {
        'lr': [0.0001, 0.001, 0.01, 0.1],
        # 'momentum': [0.1, 0.3, 0.5, 0.7],
        'batchSize': [16, 32, 48]
    }

    # Create the tuner with evaluator and the specified classes
    tuner = HyperbandTuner(categorical_hp_classes,
                           function_evaluator,
                           results_file=os.path.join(options.checkpoints_dir, 'tuning.csv'))

    tuner.tune(options.R, eta=options.eta, n_max=10, use_interactive_prompt=options.interactive_init)


if __name__ == '__main__':
    options = TrainOptions().parse()  # TODO: need to clean up options list
    options.device = torch.device('cuda:{}'.format(options.gpu_ids[0])) if options.gpu_ids else torch.device('cpu')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoints_dir', type=str, default=None)
    # parser.add_argument('--submitit_logs_dir', type=str, default=None)
    # parser.add_argument('--print_freq', type=int, default=200)
    # parser.add_argument('--save_freq', type=int, default=200)
    # parser.add_argument('--train_batch_size', type=int, default=64,
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--val_batch_size', type=int, default=1000,
    #                     help='input batch size for validation (default: 1000)')
    # parser.add_argument('--max_epochs', type=int, default=5,
    #                     help='number of epochs to train (default: 5)')
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5,
    #                     help='SGD momentum (default: 0.5)')
    #
    # parser.add_argument('--R', type=int, default=10, help='Hyperband resource usage limit (default: 10)')
    # parser.add_argument('--eta', type=float, default=3.0, help='Hyperband elimination rate (default: 3.0)')
    # parser.add_argument('--interactive_init', dest='interactive_init', action='store_true',
    #                     help='Allows choosing R and eta interactively via a command line prompt (default: False)')
    #
    # args = parser.parse_args()

    main(options)
