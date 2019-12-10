import os

import submitit
import torch

import options.train_options
import trainer

if __name__ == '__main__':
    options_ = options.train_options.TrainOptions().parse()  # TODO: need to clean up options list
    options_.device = torch.device('cuda:{}'.format(
        options_.gpu_ids[0])) if options_.gpu_ids else torch.device('cpu')
    options_.checkpoints_dir = os.path.join(options_.checkpoints_dir, options_.name)

    if not os.path.exists(options_.checkpoints_dir):
        os.makedirs(options_.checkpoints_dir)

    trainer_ = trainer.Trainer(options_)

    executor = submitit.SlurmExecutor(folder=options_.submitit_logs_dir, max_num_timeout=3)
    executor.update_parameters(
        num_gpus=8,
        partition='learnfair',
        cpus_per_task=16,
        mem=128000,
        time=4320,
        comment='nothing')
    executor.submit(trainer_)
