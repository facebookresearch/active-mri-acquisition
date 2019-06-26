from .hyperband import FunctionEvaluator, HyperparametersConfig

import argparse
import os
import submitit
import time

from typing import Callable, Dict, Iterable


# TODO add the CSV loading option
class SubmititEvaluator(FunctionEvaluator):
    """ A function evaluator that uses `submitit` to submit SLURM jobs with given hyperparameters and max epochs.

    Args:
        trainer_class:
            A callable class that runs a training job given a hyperparameter configuration. The
            class constructor must receive an `args.Namespace` object specifying the hyperparameter configuration to
            use (see `options`). The `call` method is in charge of running training and handle
            checkpointing appropriately. For this, a method `checkpoint` must be defined, as explained in the
            `submitit` documentation.
        options:
            Specifies the default options for the trainer. When a new hyperparameter configuration wants to be run,
            `SubmititEvaluator` will create a copy of these options and populate the hyperparameters appropriately
            (leaving all other options intact).

            This class assumes that trainer_class will store checkpoints under `options.checkpoints_dir` and
            that the number of epochs is specified as `options.max_epochs`.

        submitit_logs_dir:
            A directory where submitit will store logs.
        max_num_timeout:
            The maximum number of times a job will be requeued (see `submitit` documentation).
        kwargs:
            The keyword arguments will be passed to `submitit.SlurmExecutor.update_parameters`.

    The method `get_results_for_ids` will query the jobs for results and then sleep for 60 seconds before returning.

    """

    def __init__(self, trainer_class: Callable, options: argparse.Namespace,
                 submitit_logs_dir: str, max_num_timeout: int, **kwargs):
        super(FunctionEvaluator).__init__()
        self.executor = submitit.SlurmExecutor(folder=submitit_logs_dir, max_num_timeout=max_num_timeout)
        self.executor.update_parameters(**kwargs)
        self.default_options = options
        self.trainer_class = trainer_class
        self.jobs = {}

    # TODO Consider replacing the use of Namespace for some config object (protobuf or something similar)
    # However, this would require user to also be compatible with whatever config format is used
    def _create_options_from_hyperparameters(self, hyperparameters: HyperparametersConfig,
                                             max_epochs: int) -> argparse.Namespace:
        new_options = argparse.Namespace(**vars(self.default_options))
        for name, value in hyperparameters.name_value_map.items():
            if not hasattr(new_options, name):
                raise ValueError('Options has no attribute for hyperparameter {}'.format(name))
            setattr(new_options, name, value)
        if not hasattr(new_options, 'checkpoints_dir') or not hasattr(new_options, 'max_epochs'):
            raise ValueError('Options must have <checkpoints_dir> and <max_epochs> attributes')
        new_options.checkpoints_dir = os.path.join(self.default_options.checkpoints_dir,
                                                   str(hyperparameters))
        new_options.max_epochs = max_epochs

        return new_options

    def submit_for_evaluation(self, hyperparameters: HyperparametersConfig, resource_budget: int) -> str:
        new_options = self._create_options_from_hyperparameters(hyperparameters, resource_budget)
        trainer = self.trainer_class(new_options)
        job = self.executor.submit(trainer)
        self.jobs[str(job.job_id)] = job
        return str(job.job_id)

    def get_any_results_for_ids(self, ids: Iterable[str]) -> Dict[str, float]:
        all_results = {}
        for job_id in ids:
            assert job_id in self.jobs
            job = self.jobs[job_id]
            if job.done(force_check=True):
                all_results[job_id] = job.result()
        time.sleep(60)
        return all_results
