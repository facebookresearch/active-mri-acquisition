
import os
import logging
import sys
import torch

def setup(args):
    """
        Logging setup for runs with an argument struct for configuration.
        Expects args.run_name, and sets args.log_path with a folder name for log output.

    """
    rank = args.__dict__.get('rank', 0) # Distributed training rank

    log_level = logging.INFO

    # SETUP LOGGING
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    ch.setFormatter(formatter)
    #if len(root_logger.handlers) == 0:
    root_logger.addHandler(ch)

    # Log to a file also
    if rank == 0:
        if "base_path" in args.__dict__:
            args.__dict__["log_path"] = args.base_path
        else:
            if not os.path.exists('logs/'):
                os.mkdir('logs')

            args.log_path = f'logs/{args.run_name}'
            if not os.path.exists(args.log_path):
                os.mkdir(args.log_path)

        fh = logging.FileHandler(args.log_path + '/full_stdout.log', 'w')
        root_logger.addHandler(fh)

    ############
    logging.info("Run " + args.run_name)
    logging.info("#########")
    logging.info(args.__dict__)

    if 'cuda' in args.__dict__:
        args.cuda = args.cuda and torch.cuda.is_available()
    else:
        args.cuda = torch.cuda.is_available()
    logging.info("Using CUDA: {} CUDA AVAIL: {} #DEVICES: {}".format(
        args.cuda, torch.cuda.is_available(), torch.cuda.device_count()))

def log_model_statistics(model):
    nparams = 0
    group_idx = 0
    nlayers = 0
    for param in model.parameters():
        group_size = 1
        for g in param.size():
            group_size *= g
        nparams += group_size
        group_idx += 1
        if len(param.shape) >= 2:
            nlayers += 1
    logging.info(f"Model parameters: {nparams:,} layers: {nlayers}")
