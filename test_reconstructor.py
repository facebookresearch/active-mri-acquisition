import argparse
import logging
import os

import ignite.contrib.handlers
import ignite.engine
import ignite.metrics
import torch
import torch.nn.functional as F

from data import create_data_loaders
from models.fft_utils import RFFT, IFFT, preprocess_inputs
from models.reconstruction import ReconstructorNetwork
from options.train_options import TrainOptions
from util import util


def inference(batch, reconstructor, fft_functions, options):
    reconstructor.eval()
    with torch.no_grad():
        zero_filled_reconstruction, target, mask = preprocess_inputs(batch, fft_functions, options)

        # Get reconstructor output
        reconstructed_image, uncertainty_map, mask_embedding = reconstructor(
            zero_filled_reconstruction, mask)

        mse = F.mse_loss(reconstructed_image[:, :1, ...], target[:, :1, ...], reduction='none')
        ssim = util.ssim_metric(
            reconstructed_image[:, :1, ...], target[:, :1, ...], size_average=False)

        mse = mse.mean([1, 2, 3])
        return {
            'MSE': mse,
            'SSIM': ssim,
            'ground_truth': target,
            'zero_filled_image': zero_filled_reconstruction,
            'reconstructed_image': reconstructed_image,
            'uncertainty_map': uncertainty_map
        }


def load_from_checkpoint_if_present(options: argparse.Namespace,
                                    reconstructor: ReconstructorNetwork):
    if not os.path.exists(options.checkpoints_dir):
        return
    print('Loading checkpoint found at {}'.format(options.checkpoints_dir))
    files = os.listdir(options.checkpoints_dir)
    for filename in files:
        if 'regular_checkpoint' in filename:
            logging.info('Loading checkpoint at {}'.format(filename))
            checkpoint = torch.load(os.path.join(options.checkpoints_dir, filename))
            reconstructor.load_state_dict(checkpoint['reconstructor'])


def main(options: argparse.Namespace):
    print('Creating test runner with the following options:')
    for key, value in vars(options).items():
        if key == 'device':  # TODO: clean this up!
            value = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif key == 'gpu_ids':
            value = 'cuda : ' + str(value) if torch.cuda.is_available() else 'cpu'
        print('    {:>25}: {:<30}'.format(key, 'None' if value is None else value), flush=True)

    # Create Reconstructor Model
    reconstructor = ReconstructorNetwork(
        number_of_cascade_blocks=options.number_of_cascade_blocks,
        n_downsampling=options.n_downsampling,
        number_of_filters=options.number_of_reconstructor_filters,
        number_of_layers_residual_bottleneck=options.number_of_layers_residual_bottleneck,
        mask_embed_dim=options.mask_embed_dim,
        dropout_probability=options.dropout_probability,
        img_width=options.image_width,
        use_deconv=options.use_deconv)

    reconstructor = torch.nn.DataParallel(reconstructor).cuda()  # TODO: make better with to_device

    test_loader = create_data_loaders(options, is_test=True)

    load_from_checkpoint_if_present(options, reconstructor)

    fft_functions = {'rfft': RFFT().to(options.device), 'ifft': IFFT().to(options.device)}
    test_engine = ignite.engine.Engine(lambda engine, batch: inference(
        batch, reconstructor, fft_functions, options))

    progress_bar = ignite.contrib.handlers.ProgressBar()
    progress_bar.attach(test_engine)

    all_mse = []
    all_ssim = []

    @test_engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def aggregate_metrics(engine):
        all_mse.append(engine.state.output['MSE'])
        all_ssim.append(engine.state.output['SSIM'])

    test_engine.run(test_loader)
    all_mse = torch.cat(all_mse)
    all_ssim = torch.cat(all_ssim)
    print('Results (median) - MSE: {}, SSIM: {}'.format(all_mse.median(), all_ssim.median()))


if __name__ == '__main__':
    options_ = TrainOptions().parse()  # TODO: need to clean up options list
    options_.isTrain = False
    options_.device = torch.device('cuda:{}'.format(
        options_.gpu_ids[0])) if options_.gpu_ids else torch.device('cpu')
    main(options_)
