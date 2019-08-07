import os

import hyperband
import hyperband.submitit_function_evaluator
import torch

import options.train_options
import trainer


def main(options):
    # Create a function evaluator to be passed to the tuner. Here you can pass the
    # SLURM arguments as keywords.
    function_evaluator = hyperband.submitit_function_evaluator.SubmititEvaluator(
        trainer.Trainer,
        options,
        options.submitit_logs_dir,
        3,
        job_name='raw_10',
        time=4320,
        # constraint='volta32gb',
        # gres='gpu:8',
        partition='learnfair',
        num_gpus=8,
        cpus_per_task=16)

    # Specify the hyperparameter names and their possible values (for now only categorical
    # distributions are supported).
    categorical_hp_classes = {
        # 'lr': [0.0002],
        # 'batchSize': [2],
        'lambda_gan':  [0, 0.01],

        # Reconstructor hyper-parameters
        # 'number_of_cascade_blocks': [3, 4],
        'number_of_reconstructor_filters': [128, 256],
        'n_downsampling': [4],   # 5 is distorting the 368 dimension
        'number_of_layers_residual_bottleneck': [5, 6],
        'dropout_probability': [0, 0.1, 0.2],

        # Evaluator hyper-parameters
        'number_of_evaluator_filters': [256],
        'number_of_evaluator_convolution_layers': [3, 4, 5],
        # 6 is changing the shape of output to (2, 368, 2) instead of (2, 368)

        'mask_embed_dim': [0, 6, 9]
    }

    # Create the tuner with evaluator and the specified classes
    tuner = hyperband.hyperband.HyperbandTuner(
        categorical_hp_classes,
        function_evaluator,
        results_file=os.path.join(options.checkpoints_dir, 'tuning.csv'))

    tuner.tune(
        options.R,
        eta=options.eta,
        n_max=options.max_jobs_tuner,
        use_interactive_prompt=options.interactive_init)


if __name__ == '__main__':
    options = options.train_options.TrainOptions().parse()
    options.device = torch.device('cuda:{}'.format(
        options.gpu_ids[0])) if options.gpu_ids else torch.device('cpu')
    main(options)
