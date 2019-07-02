import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from util.rl.evaluator_plus_plus import EvaluatorDataset, EvaluatorPlusPlus, EvaluatorPlusPlusTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    options = parser.parse_args()

    options.checkpoints_dir = os.path.join(options.checkpoints_dir,
                                           'bs_{}_lr_{}_beta1_{}_beta2_{}'.format(options.batch_size,
                                                                                  options.lr,
                                                                                  options.beta1,
                                                                                  options.beta2))
    options.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(options.checkpoints_dir):
        os.makedirs(options.checkpoints_dir)

    # Logger set up
    root_logger = logging.getLogger()
    root_logger.setLevel('INFO')
    if options.debug:
        root_logger.setLevel('DEBUG')
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(options.checkpoints_dir, 'train.log'))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Training set up
    train_dataloader = DataLoader(EvaluatorDataset(split='train'),
                                  batch_size=options.batch_size,
                                  shuffle=True,
                                  num_workers=options.num_workers)
    val_dataloader = DataLoader(EvaluatorDataset(split='val'),
                                batch_size=options.batch_size,
                                shuffle=True,
                                num_workers=options.num_workers)

    model = EvaluatorPlusPlus()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, betas=(options.beta1, options.beta2))

    trainer = EvaluatorPlusPlusTrainer(model, optimizer, {'train': train_dataloader, 'val': val_dataloader}, options)
    trainer()
