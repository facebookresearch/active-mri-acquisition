import os

import acquire_rl
import options.rl_options
import util.rl.dqn


def main(options_):
    tester = util.rl.dqn.DQNTester(options_.checkpoints_dir)
    tester()


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()

    experiment_str = acquire_rl.get_experiment_str(opts)
    opts.checkpoints_dir = os.path.join(opts.checkpoints_dir, experiment_str)

    main(opts)
