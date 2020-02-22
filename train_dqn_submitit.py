import os
import warnings

import submitit
import torch

import acquire_rl
import options.rl_options
import util.rl.dqn

if __name__ == '__main__':
    options_ = options.rl_options.RLOptions().parse()
    options_.batchSize = 1
    options_.masks_dir = None
    options_.dqn_resume = True

    options_.device = torch.device('cuda:{}'.format(
        options_.gpu_ids[0])) if options_.gpu_ids else torch.device('cpu')

    experiment_str = acquire_rl.get_experiment_str(options_)
    options_.checkpoints_dir = os.path.join(options_.checkpoints_dir, experiment_str)
    if not os.path.isdir(options_.checkpoints_dir):
        os.makedirs(options_.checkpoints_dir)

    submitit_logs_dir = os.path.join(options_.checkpoints_dir, 'submitit')
    print(f'Submitit logs dir will be stored at: {submitit_logs_dir}')
    constraint = 'volta32gb'
    executor = submitit.SlurmExecutor(submitit_logs_dir, max_num_timeout=3)
    executor.update_parameters(
        num_gpus=len(options_.gpu_ids),
        partition='learnfair',
        cpus_per_task=8,
        mem=256000,
        time=4320,
        constraint=constraint,
        job_name=options_.job_name,
        signal_delay_s=3600,
        comment='')

    # Launch trainer
    print('Launching trainer.')
    trainer_ = util.rl.dqn.DQNTrainer(options_)
    executor.submit(trainer_)

    # Launch tester if needed
    if options_.dqn_test_episode_freq is None:
        print('Launching tester.')
        tester_ = util.rl.dqn.DQNTester(options_.checkpoints_dir)
        executor.submit(tester_)
    else:
        warnings.warn(f'No dedicated tester will be launched, so testing will be done in the '
                      f'same process as training. If you want a dedicated tester, unset '
                      f'--dqn_test_episode_freq. Current value is {options_.dqn_test_episode_freq},'
                      f' and it should be None (the default).', UserWarning)
