import os

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

    trainer_ = util.rl.dqn.DQNTrainer(options_)

    submitit_logs_dir = os.path.join(options_.checkpoints_dir, 'submitit')

    print(f'Submitit logs dir will be stored at: {submitit_logs_dir}')

    constraint = 'volta32gb' if options_.dqn_alternate_opt_per_epoch else ''
    executor = submitit.SlurmExecutor(submitit_logs_dir, max_num_timeout=3)
    executor.update_parameters(
        num_gpus=len(options_.gpu_ids),
        partition='priority',
        cpus_per_task=8,
        mem=256000,
        time=4320,
        constraint=constraint,
        job_name=options_.job_name,
        signal_delay_s=3600,
        comment='MIDL 01/30')
    executor.submit(trainer_)
