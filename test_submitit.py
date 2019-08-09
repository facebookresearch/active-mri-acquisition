import hyperband.submitit_function_evaluator
import options.train_options
import trainer

import torch


def main(options):
    # Create a function evaluator to be passed to the tuner. Here you can pass the
    # SLURM arguments as keywords.
    function_evaluator = hyperband.submitit_function_evaluator.SubmititEvaluator(
        trainer.Trainer,
        options,
        options.submitit_logs_dir,
        3,
        job_name='run50',
        # constraint='volta32gb',
        # gres='gpu:8',
        time=4320,
        partition='dev',
        num_gpus=8,
        cpus_per_task=16,
        comment='short job debugging stuff')

    hp = [
        'lr',
        'lambda_gan',
        'number_of_reconstructor_filters',
        'n_downsampling',
        'number_of_layers_residual_bottleneck',
        'dropout_probability',
        'number_of_evaluator_filters',
        'number_of_evaluator_convolution_layers',
        'mask_embed_dim',
    ]

    hp_config = hyperband.HyperparametersConfig(hp)
    hp_config.assign('batchSize', 2)
    hp_config.assign('lr', 0.0002)
    hp_config.assign('lambda_gan', 0)
    hp_config.assign('number_of_reconstructor_filters', 128)
    hp_config.assign('n_downsampling', 3)
    hp_config.assign('number_of_layers_residual_bottleneck', 5)
    hp_config.assign('dropout_probability', 0.2)
    hp_config.assign('number_of_evaluator_filters', 256)
    hp_config.assign('number_of_evaluator_convolution_layers', 5)
    hp_config.assign('mask_embed_dim', 0)

    jobid = function_evaluator.submit_for_evaluation(hp_config, resource_budget=50)
    print(jobid)

    while True:
        results = function_evaluator.get_any_results_for_ids([jobid])
        if len(results) > 0:
            break
        print('No results', flush=True)


if __name__ == '__main__':
    options = options.train_options.TrainOptions().parse()
    options.device = torch.device('cuda:{}'.format(
        options.gpu_ids[0])) if options.gpu_ids else torch.device('cpu')
    main(options)
