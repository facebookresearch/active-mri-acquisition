from util.rl.evaluator_plus_plus import EvaluatorDataset, EvaluatorPlusPlus, train

import argparse
import sys
import os
import torch.utils.data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default=None)
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
    if not os.path.exists(options.checkpoints_dir):
        os.makedirs(options.checkpoints_dir)

    # TODO replace this with a decent logger
    sys.stdout = open(os.path.join(options.checkpoints_dir, 'train.log'), 'w')

    options.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = EvaluatorDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=options.batch_size, shuffle=True, num_workers=8)

    train(EvaluatorPlusPlus(), data_loader, options)
