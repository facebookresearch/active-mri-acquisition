import os

import hyperband
import hyperband.submitit_function_evaluator

import util.rl.dqn
import options.rl_options


def main(options_):
    # Create a function evaluator to be passed to the tuner. Here you can pass the
    # SLURM arguments as keywords.
    function_evaluator = hyperband.submitit_function_evaluator.SubmititEvaluator(
        util.rl.dqn.DQNTrainer,
        options_,
        os.path.join(options_.checkpoints_dir, 'submitit_logs'),
        3,
        resource_name='num_train_steps',
        resource_factor=50000,
        job_name='active_acq_tune_dqn',
        time=4320,
        partition='learnfair',
        num_gpus=1,
        cpus_per_task=2,
        signal_delay_s=600,
        mem=16000)

    # Specify the hyperparameter names and their possible values (for now only categorical
    # distributions are supported).
    categorical_hp_classes = {
        'gamma': [0, 0.25, 0.5, 0.75],
        'target_net_update_freq': [1000, 2000, 5000, 10000],
        'epsilon_decay': [100000, 500000, 1000000],
        'epsilon_start': [0.99, 0.95, 0.9],
        'rl_batch_size': [16, 32, 64],
        'dqn_learning_rate': [1.25e-4, 6.25e-5],
        'replay_buffer_size': [400000],
    }

    # Create the tuner with evaluator and the specified classes
    tuner = hyperband.hyperband.HyperbandTuner(
        categorical_hp_classes,
        function_evaluator,
        results_file=os.path.join(options_.checkpoints_dir, 'tuning.csv'))

    tuner.tune(
        options_.R,
        eta=options_.eta,
        n_max=options_.max_jobs_tuner,
        use_interactive_prompt=options_.interactive_init)


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()
    opts.batchSize = 1
    opts.mask_type = 'grid'  # This is ignored, only here for compatibility with loader

    opts.R = 10
    opts.eta = 3
    opts.max_jobs_tuner = 30
    opts.interactive_init = True

    main(opts)
