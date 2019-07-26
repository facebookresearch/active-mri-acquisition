import hyperband.submitit_function_evaluator
import options.train_options
import trainer

from hyperband import HyperparametersConfig

import torch

def main(options):
    # Create a function evaluator to be passed to the tuner. Here you can pass the
    # SLURM arguments as keywords.
    function_evaluator = hyperband.submitit_function_evaluator.SubmititEvaluator(
        trainer.Trainer,
        options,
        options.submitit_logs_dir,
        3,
        job_name='submitit_test',
        time=4320,
        partition='learnfair',
        num_gpus=8,
        cpus_per_task=16)

    hp = ['batchSize', 'mask_embed_dim']

    hp_config = hyperband.HyperparametersConfig(hp)
    hp_config.assign('batchSize', 4)
    hp_config.assign('mask_embed_dim', 0)

    function_evaluator.submit_for_evaluation(hp_config, resource_budget=20)


if __name__ == '__main__':
    options = options.train_options.TrainOptions().parse()
    options.device = torch.device('cuda:{}'.format(
        options.gpu_ids[0])) if options.gpu_ids else torch.device('cpu')
    main(options)

