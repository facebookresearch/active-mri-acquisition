import abc
import math
import os
import re
import select
import sys
import torch

from functools import reduce
from torch.distributions import categorical
from typing import Any, Dict, Iterable, List, Tuple, Union


# TODO make a class for Schedule

class HyperparametersConfig:
    """ Represents a configuration of hyperparameters.

    Internally, a `HyperparametersConfig` object is a dictionary of hyperparameter name to values. The class
    provides some utility methods to generate a string representation of the parameters and populate the
    hyperparameters from such a string.
    """
    def __init__(self, hyperparameter_names: Iterable):
        self.name_value_map = {name: None for name in hyperparameter_names}

    def __str__(self) -> str:
        return '_'.join(['{}.{}'.format(x, y) for x,y in self.name_value_map.items()])

    def assign(self, hyperparameter_name: str, value: Union[int, float]):
        self.name_value_map[hyperparameter_name] = value

    def populate_from_string(self, hyperparameter_str: Any):
        items = hyperparameter_str.rstrip('\n').split('_')
        for name_value_str in items:
            idx = name_value_str.find('.')
            name = name_value_str[ : idx]
            value = name_value_str[idx + 1:]
            self.name_value_map[name] = int(value) if value.find('.') == -1 else float(value)


class FunctionEvaluator:
    """ Interface to represent objective functions.

    A `FunctionEvaluator` has two methods:
        -`submit_for_evaluation`: Calls the objective function with the given hyperparameter values. For objective
            functions that take a long time to compute, this function should preferably be non-blocking.
            For example, it can be a call to a bash script that runs in the background.
            The function must return a unique identifier for matching the results to the given hyperparameters.
        -`get_any_results_for_ids`: Gets the results of submitted jobs with the given ids if present.
            The return value is a dictionary of IDs (as provided by `submit_for_evaluation`) to results.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def submit_for_evaluation(self, hyperparameters: HyperparametersConfig, resource_budget: int) -> str:
        """ Submits the hyperparameters for function evaluation.

        :param hyperparameters: The hyperparameter values to evaluate.
        :param resource_budget: The budget for computation (e.g., number of training epochs).

        :return: An unique identifier string for the submitted evaluation job 
            (must be a 1-to-1 map to hyperparameter configuration).
        """
        pass

    @abc.abstractmethod
    def get_any_results_for_ids(self, ids: Iterable[str]) -> Dict[str, float]:
        """ Gets any available results for the jobs with the given ids.

            This method can be blocking but it is meant to run for a short period of time.
            
        :param ids: An iterable with the ids of the jobs to check.
        :return: A dictionary of identifiers to results.
        """
        pass


class HyperbandTuner:
    """ A hyperparameter tuner based on the Hyperband algorithm.

        See https://arxiv.org/pdf/1603.06560.pdf
    """
    def __init__(self, categorical_hp_classes: Dict[str, List], function_evaluator: FunctionEvaluator,
                 results_file: str = None, schedule_file: str = None, use_random_search: bool = False):
        """Creates as tuner for the given set of hyperparameters.

        :param categorical_hp_classes: Specifies the name of the classes for categorical hyperparameters, and a list of
            their possible values.
        :param function_evaluator: A function evaluator that computes the objective function value for a
        given set of hyperparameters
        :param: results_file: The file path where to store the final results in CSV format.
        :param: schedule_file: If given, the initial schedule is loaded from the given file.
        :param: use_random_search: If true, a simple random search is performed.
        """
        self.categorical_hyp_classes = categorical_hp_classes

        self.hyperparameters = {}
        for name, classes in categorical_hp_classes.items():
            probs = torch.ones(len(classes)) / len(classes)
            self.hyperparameters[name] = categorical.Categorical(probs)

        self.function_evaluator = function_evaluator

        self.results_file = results_file
        self.schedule_file = schedule_file

        self.use_random_search = use_random_search

    def sample_hyperparameters(self, num_samples: int = 1) -> List[HyperparametersConfig]:
        """ Samples hyperparameter configurations.

        :param num_samples: The number of hyperparameter configurations to sample.
        :return The hyperparameter configurations as a list of dictionaries names to values.
        """
        all_samples = []
        all_samples_str = set()     # Used to avoid duplicate hyperparameter configurations
        for i in range(num_samples):
            hp_config = HyperparametersConfig(self.hyperparameters.keys())
            for name, distribution in self.hyperparameters.items():
                if isinstance(distribution, categorical.Categorical):
                    hp_config.assign(name, self.categorical_hyp_classes[name][distribution.sample().item()])
                else:
                    hp_config.assign(name, distribution.sample().item())
            hp_str = str(hp_config)
            if hp_str not in all_samples_str:
                all_samples.append(hp_config)
                all_samples_str.add(hp_str)
        return all_samples

    def _get_schedule_from_file(self) -> Dict[str, List[Any]]:
        print('Getting schedule from file {}'.format(self.schedule_file), flush=True)
        bracket_settings = []
        current_round_bracket = []
        hyperparameters_in_bracket = []
        current_results_bracket = []
        jobs_running_in_bracket = []

        loading_hyperparams = False
        f = open(self.schedule_file, 'r')
        for line in f:
            match_hp_bracket_begin = re.compile('Hyperparameters for bracket [0-9]+').match(line)
            if match_hp_bracket_begin is not None:
                loading_hyperparams = True
                current_round_bracket.append(0)
                hyperparameters_in_bracket.append([])
                current_results_bracket.append({})
                jobs_running_in_bracket.append(set())
                print(line.rstrip('\n'), flush=True)
            else:
                match_hps_end = re.compile('The brackets are set as follows').match(line)
                if match_hps_end is not None:
                    loading_hyperparams = False

                if loading_hyperparams:
                    hp_config = HyperparametersConfig(self.hyperparameters.keys())
                    hyperparameters_in_bracket[-1].append(hp_config.populate_from_string(line))
                    print(line.rstrip('\n'), flush=True)
                else:
                    if line.startswith('[('):
                        items = re.compile('([0-9]+, [0-9]+)').findall(line)
                        bracket_settings.append([tuple(map(int, item.split(', '))) for item in items])

        f.close()

        return {
            'bracket_settings': bracket_settings,
            'current_round_bracket': current_round_bracket,
            'hyperparameters_in_bracket': hyperparameters_in_bracket,
            'current_results_bracket': current_results_bracket,
            'jobs_running_in_bracket': jobs_running_in_bracket
        }

    def compute_schedule_for_random_search(self, num_resources: int, num_hp_configs: int) -> Dict[str, List[Any]]:
        """ Computes a schedule for a simple random search.

        :param num_resources: The number of resources to allow per configuration.
        :param num_hp_configs: The number of hyperparameter configurations to evaluate.

        """
        # Pre-compute number of jobs/resources that need to be launched
        bracket_settings = []
        current_round_bracket = []
        hyperparameters_in_bracket = []
        current_results_bracket = []
        jobs_running_in_bracket = []
        bracket_settings.append([])
        current_round_bracket.append(0)
        hyperparameters_in_bracket.append(self.sample_hyperparameters(num_hp_configs))
        print('Hyperparameters for bracket 0:', flush=True)
        for hp in hyperparameters_in_bracket[-1]:
            print(hp, flush=True)
        current_results_bracket.append({})
        jobs_running_in_bracket.append(set())
        bracket_settings[-1].append((num_hp_configs, num_resources))

        return {
            'bracket_settings': bracket_settings,
            'current_round_bracket': current_round_bracket,
            'hyperparameters_in_bracket': hyperparameters_in_bracket,
            'current_results_bracket': current_results_bracket,
            'jobs_running_in_bracket': jobs_running_in_bracket
        }

    def compute_schedule(self, R, eta) -> Dict[str, List[Any]]:
        """ Computes the bracket schedule and additional information needed to run Hyperband.
        
        :param R: Represents the maximum amount of resources that can be allocated to a given configuration.
            If self.use_random_search = True, then it represents the number of resources to run per model.
        :param eta: Parameter controlling the elimination rate.
            If self.use_random_search = True, then it represents the number of hyperparameters to try.

        :return: A dictionary with the following fields:
            -bracket_settings: The resources and number of configurations per bracket and round.
            -current_round_bracket: An array with the current round that is running on each bracket.
            -hyperparameters_in_bracket: An array where each entry is an array of hyperparameter
                configurations that are left to be tried for the current round of a bracket.
            -current_results_bracket: An array of dictionaries where each entry stores the results 
                obtained in the current round for a given bracket.
            -jobs_running_in_bracket: The IDs of the jobs that have been submitted for evaluation in the
                current round for a given bracket.
        """
        if self.use_random_search:
            return self.compute_schedule_for_random_search(R, int(eta))

        s_max = math.floor(math.log(R, eta))
        B = (s_max + 1) * R

        print("Budget per bracket will be (approx.) {}".format(B), flush=True)
        
        # Pre-compute number of jobs/resources that need to be launched
        bracket_settings = []
        current_round_bracket = []
        hyperparameters_in_bracket = []
        current_results_bracket = []
        jobs_running_in_bracket = []
        for s in range(s_max, -1, -1):
            eta_s = eta ** s
            n = math.ceil(B * eta_s / (R * (s + 1)))
            r = R / eta_s
            bracket_settings.append([])            
            current_round_bracket.append(0)
            hyperparameters_in_bracket.append(self.sample_hyperparameters(n))
            print('Hyperparameters for bracket {}:'.format(s_max - s), flush=True)
            for hp in hyperparameters_in_bracket[-1]:
                print(hp, flush=True)
            current_results_bracket.append({})
            jobs_running_in_bracket.append(set())
            for i in range(s + 1):
                eta_i = eta ** i
                n_i = min(math.floor(n / eta_i), len(hyperparameters_in_bracket[-1]))   # unlikely corner case
                r_i = int(r * eta_i)
                bracket_settings[-1].append((n_i, r_i))
                
        return {
            'bracket_settings': bracket_settings,
            'current_round_bracket': current_round_bracket,
            'hyperparameters_in_bracket': hyperparameters_in_bracket,
            'current_results_bracket': current_results_bracket,
            'jobs_running_in_bracket': jobs_running_in_bracket
        }

    def _interactive_schedule_prompt(self) -> Tuple[int, float]:
        while True:
            print('What values for R (max. resource usage) and eta '
                  '(proportion of configs. discarded) would you like to use? Separate by spaces.')
            answer, _, _ = select.select([sys.stdin], [], [], 30)
            if answer:
                R, eta = sys.stdin.readline().strip().split(' ')
                R = int(R)
                eta = float(eta)
                brackets = self.compute_schedule(R, eta)['bracket_settings']
                total_epochs = 0
                for b in brackets:
                    b_with_epochs = [(n_i, int(math.ceil(r_i))) for n_i, r_i in b]
                    print(b_with_epochs)
                    for x in b_with_epochs:
                        total_epochs += x[0] * x[1]
                print('Total number of resources used will be: {}'.format(total_epochs))
                print('Do you like this schedule? ([y]/n, 30 seconds to answer)')
                answer, _, _ = select.select([sys.stdin], [], [], 30)
                if answer and sys.stdin.readline().strip() in ['N', 'n']:
                    continue
                else:
                    return R, eta

    def tune(self, R, eta=3, n_max=200, use_interactive_prompt=False):
        """ Tunes the hyperparameters using Hyperband.

        :param R: Represents the maximum amount of resources that can be allocated to a given configuration.
        :param eta: Parameter controlling the elimination rate.
        :param n_max: A limit on the maximum number of configurations that can be run simultaneously.
        :param use_interactive_prompt: If True, a prompt will be presented to choose R and eta interactively.

        :return: A dictionary with all the results observed.
        """
        if not self.use_random_search and use_interactive_prompt:
            R, eta = self._interactive_schedule_prompt()
        if self.schedule_file is not None and os.path.isfile(self.schedule_file):
            schedule = self._get_schedule_from_file()
        else:
            schedule = self.compute_schedule(R, eta)
        bracket_settings = schedule['bracket_settings']
        current_round_bracket = schedule['current_round_bracket']
        hyperparameters_in_bracket = schedule['hyperparameters_in_bracket']
        current_results_bracket = schedule['current_results_bracket']
        jobs_running_in_bracket = schedule['jobs_running_in_bracket']
        
        print('The brackets are set as follows:', flush=True)
        for i in range(len(bracket_settings)):
            print(bracket_settings[i], flush=True)
        
        # Start launching jobs across all brackets
        s_max = len(bracket_settings) - 1
        hyperparameters_running = set()
        best_result = -math.inf
        best_hyperparameters = None
        all_results = {}
        total_epochs = 0
        brackets_remaining = set(range(s_max + 1))
        job_id_to_hp = {}
        while len(brackets_remaining) > 0:
            # Checking if more jobs can be launched
            slack = n_max - len(hyperparameters_running)
            for i in range(s_max + 1):
                if i not in brackets_remaining:
                    continue
                r_i = bracket_settings[i][current_round_bracket[i]][1]
                
                attempts_left = len(hyperparameters_in_bracket[i])
                while len(hyperparameters_in_bracket[i]) > 0 and slack > 0 and attempts_left > 0:
                    attempts_left -= 1
                    slack -= 1
                    hp = hyperparameters_in_bracket[i].pop(0)
                    # Don't allow same hyperparameter config to run more than once simultaneously
                    if str(hp) in hyperparameters_running:
                        hyperparameters_in_bracket[i].append(hp)
                        continue
                    hyperparameters_running.add(str(hp))
                    print('Submit {} for {} epochs in bracket {}'.format(hp, r_i, i), flush=True)
                    job_id = self.function_evaluator.submit_for_evaluation(hp, r_i)                            
                    job_id_to_hp[job_id] = hp
                    jobs_running_in_bracket[i].add(job_id)
                    total_epochs += r_i
                
            # Checking which jobs have finished and update bracket results
            results = self.function_evaluator.get_any_results_for_ids(
                reduce((lambda x, y: x | y), jobs_running_in_bracket))
            for job_id, hp_results in results.items():
                hp_str = str(job_id_to_hp[job_id])
                hyperparameters_running.remove(hp_str)
                for i in brackets_remaining:
                    if job_id in jobs_running_in_bracket[i]:
                        jobs_running_in_bracket[i].remove(job_id)
                        current_results_bracket[i][job_id] = hp_results
                        if self.results_file is not None:
                            n_i, r_i = bracket_settings[i][current_round_bracket[i]]
                            f = open(self.results_file + '.tmp', 'a')
                            f.write('{},{},{}\n'.format(hp_str, hp_results, r_i))
                            f.close()
                
            # Check if any bracket completed a round (equivalent to a full run of SuccessiveHalving)
            for i in range(s_max + 1):
                if i not in brackets_remaining:
                    continue
                n_i, r_i = bracket_settings[i][current_round_bracket[i]]                        
                if len(current_results_bracket[i]) == n_i:
                    assert len(hyperparameters_in_bracket[i]) == 0
                    print('All results for round {} of bracket {} ready'.format(current_round_bracket[i], i))
                    
                    # Sort the results accumulated for this bracket during this round
                    round_results = sorted(current_results_bracket[i].items(), key=lambda kv: kv[1], reverse=True)
                    round_results = list(zip(*round_results))   # round_results = [(sorted_ids), (sorted_values)]
                    current_results_bracket[i] = {}
                    
                    # Saving best result seen so far
                    if round_results[1][0] > best_result:
                        best_result = round_results[1][0]
                        best_hyperparameters = job_id_to_hp[round_results[0][0]]

                    # Storing all results found in this round for future reference
                    for idx in range(len(round_results[0])):
                        hp, obj_func_value = job_id_to_hp[round_results[0][idx]], round_results[1][idx]
                        hp_str = str(hp)
                        if hp_str not in all_results or (hp_str in all_results and all_results[hp_str][1] < r_i):
                            all_results[hp_str] = (obj_func_value, r_i)
                    
                    # Check if bracket completed, if not go to the next round
                    current_round_bracket[i] += 1
                    if current_round_bracket[i] == len(bracket_settings[i]):
                        brackets_remaining.remove(i)
                        print('Completed bracket {}'.format(i), flush=True)
                    else:
                        new_ni = bracket_settings[i][current_round_bracket[i]][0]
                        print('Start new round for bracket {} with {} jobs'.format(i, new_ni), flush=True)
                        # Keep the best floor(n_i / eta)
                        hyperparameters_in_bracket[i] = [job_id_to_hp[job_id] for job_id in round_results[0][:new_ni]]

        if self.results_file is not None:
            f = open(self.results_file, 'w')
            for hp_str, obj_func_value_and_epochs in all_results.items():
                f.write('{},{},{}\n'.format(hp_str, obj_func_value_and_epochs[0], obj_func_value_and_epochs[1]))
            f.close()

        print('At most {} epochs were evaluated'.format(total_epochs), flush=True)
        
        return best_hyperparameters, best_result
