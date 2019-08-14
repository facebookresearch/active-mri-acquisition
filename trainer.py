import argparse
import ignite.engine
import logging
import numpy as np
import os
import submitit
import tempfile
import torch
import torch.nn.functional as F
import torch.optim as optim

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Loss
from tensorboardX import SummaryWriter
from typing import Any, Dict, Tuple

from data import create_data_loaders
from models.evaluator import EvaluatorNetwork
from models.fft_utils import RFFT, IFFT, clamp, preprocess_inputs, gaussian_nll_loss
from models.networks import GANLossKspace  # TODO: maybe move GANLossKspace to a loss file?
from models.reconstruction import ReconstructorNetwork
from options.train_options import TrainOptions
from util import util


def run_validation_and_update_best_checkpoint(
        engine: ignite.engine.Engine,
        val_engine: ignite.engine.Engine = None,
        progress_bar: ignite.contrib.handlers.ProgressBar = None,
        val_loader: torch.utils.data.DataLoader = None,
        trainer: 'Trainer' = None):
    # TODO: take argument for which metric to use as score for checkpointing. Using MSE for now
    val_engine.run(val_loader)
    np.save(os.path.join(trainer.options.checkpoints_dir, 'snapshot_' + str(val_engine.state.epoch)),
            val_engine.state.output['snapshot'])
    metrics = val_engine.state.metrics
    progress_bar.log_message('Validation Results - Epoch: {}  MSE: {:.3f} SSIM: {:.3f}'.format(
        engine.state.epoch, metrics['mse'], metrics['ssim']))
    trainer.completed_epochs += 1
    score = -metrics['mse']
    if score > trainer.best_validation_score:
        trainer.best_validation_score = score
        full_path = save_checkpoint_function(trainer, 'best_checkpoint')
        progress_bar.log_message('Saved best checkpoint to {}. Score: {}. Iteration: {}'.format(
            full_path, score, engine.state.iteration))


def save_checkpoint_function(trainer: 'Trainer', filename: str) -> str:
    # Ensures atomic checkpoint save to avoid corrupted files if preempted during a save operation
    tmp_filename = tempfile.NamedTemporaryFile(delete=False, dir=trainer.options.checkpoints_dir)
    try:
        torch.save(trainer.create_checkpoint(), tmp_filename)
    except BaseException:
        tmp_filename.close()
        os.remove(tmp_filename.name)
        raise
    else:
        tmp_filename.close()
        full_path = os.path.join(trainer.options.checkpoints_dir, trainer.options.name,
                                 filename + '.pth')
        os.rename(tmp_filename.name, full_path)
        return full_path


def save_regular_checkpoint(engine: ignite.engine.Engine,
                            trainer: 'Trainer' = None,
                            progress_bar: ignite.contrib.handlers.ProgressBar = None):
    if (engine.state.iteration - 1) % trainer.options.save_freq == 0:
        full_path = save_checkpoint_function(trainer, 'regular_checkpoint')
        progress_bar.log_message('Saved regular checkpoint to {}. Epoch: {}, Iteration: {}'.format(
            full_path, trainer.completed_epochs, engine.state.iteration))


class Trainer:

    def __init__(self, options: argparse.Namespace):
        self.reconstructor = None
        self.evaluator = None
        self.options = options
        self.best_validation_score = -float('inf')
        self.completed_epochs = 0

        criterion_gan = GANLossKspace(
            use_mse_as_energy=options.use_mse_as_disc_energy, grad_ctx=options.grad_ctx).to(
                options.device)

        self.losses = {'GAN': criterion_gan, 'NLL': gaussian_nll_loss}

        self.fft_functions = {'rfft': RFFT().to(options.device), 'ifft': IFFT().to(options.device)}

        if not os.path.exists(options.checkpoints_dir):
            os.makedirs(options.checkpoints_dir)

    def create_checkpoint(self) -> Dict[str, Any]:
        return {
            'reconstructor': self.reconstructor.state_dict(),
            'evaluator': self.evaluator.state_dict() if self.options.use_evaluator else None,
            'options': self.options,
            'optimizer_G': self.optimizers['G'].state_dict(),
            'optimizer_D': self.optimizers['D'].state_dict()
            if self.options.use_evaluator else None,
            'completed_epochs': self.completed_epochs,
            'best_validation_score': self.best_validation_score,
        }

    def get_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_data_loader, val_data_loader = create_data_loaders(self.options)
        return train_data_loader, val_data_loader

    def inference(self, batch, reconstructor, fft_functions, options):
        reconstructor.eval()
        with torch.no_grad():
            zero_filled_reconstruction, target, mask = preprocess_inputs(
                batch[1], batch[0], fft_functions, options)

            # Get reconstructor output
            reconstructed_image, uncertainty_map, mask_embedding = reconstructor(
                zero_filled_reconstruction, mask)
            if options.dataroot == 'KNEE_RAW':
                mse = F.mse_loss(reconstructed_image[:, :1, 160:-160, 24:-24], target[:, :1, 160:-160, 24:-24], size_average=True)
                ssim = util.ssim_metric(reconstructed_image[:, :1, 160:-160, 24:-24], target[:, :1, 160:-160, 24:-24])

                target = target[:, :, 160:-160, 24:-24]
                zero_filled_reconstruction = zero_filled_reconstruction[:, :, 160:-160, 24:-24]
                reconstructed_image = reconstructed_image[:, :, 160:-160, 24:-24]
                uncertainty_map = uncertainty_map[:, :, 160:-160, 24:-24]
            else:
                mse = F.mse_loss(reconstructed_image, target, size_average=True)
                ssim = util.ssim_metric(reconstructed_image, target)

            # save to disk numpy array containing ground truth image, zero filled image and reconstructed image
            snapshot = np.array([target, zero_filled_reconstruction, reconstructed_image])

        return {
            'MSE': mse,
            'SSIM': ssim,
            'ground_truth': target,
            'zero_filled_image': zero_filled_reconstruction,
            'reconstructed_image': reconstructed_image,
            'uncertainty_map': uncertainty_map,
            'snapshot': snapshot
        }

    def load_from_checkpoint_if_present(self):
        if not os.path.exists(self.options.checkpoints_dir):
            return
        print('Loading checkpoint found at {}'.format(self.options.checkpoints_dir))
        files = os.listdir(self.options.checkpoints_dir)
        for filename in files:
            if 'regular_checkpoint' in filename:
                logging.info('Loading checkpoint at {}'.format(filename))
                checkpoint = torch.load(os.path.join(self.options.checkpoints_dir, filename))
                self.reconstructor.load_state_dict(checkpoint['reconstructor'])
                if self.options.use_evaluator:
                    self.evaluator.load_state_dict(checkpoint['evaluator'])
                    self.optimizers['D'].load_state_dict(checkpoint['optimizer_D'])
                self.optimizers['G'].load_state_dict(checkpoint['optimizer_G'])
                self.completed_epochs = checkpoint['completed_epochs']
                self.best_validation_score = checkpoint['best_validation_score']

    def load_weights_from_given_checkpoint(self):
        if self.options.weights_checkpoint is None or not os.path.exists(
                self.options.weights_checkpoint):
            return
        print('Loading weights from checkpoint found at {}'.format(self.options.weights_checkpoint))
        checkpoint = torch.load(
            os.path.join(self.options.checkpoints_dir, self.options.weights_checkpoint))
        self.reconstructor.load_state_dict(checkpoint['reconstructor'])
        if self.options.use_evaluator and 'evaluator' in checkpoint.keys():
            self.evaluator.load_state_dict(checkpoint['evaluator'])

    # TODO: fix sign leakage
    # TODO: consider adding learning rate scheduler
    # noinspection PyUnboundLocalVariable
    def update(self, batch, reconstructor, evaluator, optimizers, losses, fft_functions, options):
        zero_filled_reconstruction, target, mask = preprocess_inputs(batch[1], batch[0],
                                                                     fft_functions, options)

        # Get reconstructor output
        reconstructed_image, uncertainty_map, mask_embedding = reconstructor(
            zero_filled_reconstruction, mask)

        # ------------------------------------------------------------------------
        # Update evaluator and compute generator GAN Loss
        # ------------------------------------------------------------------------
        loss_G_GAN = 0
        if self.evaluator is not None:
            optimizers['D'].zero_grad()
            fake = clamp(reconstructed_image)
            detached_fake = fake.detach()
            if options.mask_embed_dim != 0:
                mask_embedding = mask_embedding.detach()
            output = evaluator(detached_fake, mask_embedding)
            loss_D_fake = losses['GAN'](
                output, False, mask, degree=0, pred_and_gt=(detached_fake[:, :1, ...], target))

            real = clamp(target)
            output = evaluator(real, mask_embedding)
            loss_D_real = losses['GAN'](
                output, True, mask, degree=1, pred_and_gt=(detached_fake[:, :1, ...], target))

            loss_D = loss_D_fake + loss_D_real
            loss_D.backward(retain_graph=True)
            optimizers['D'].step()

            output = evaluator(fake, mask_embedding)
            loss_G_GAN = losses['GAN'](
                output, True, mask, degree=1, updateG=True, pred_and_gt=(fake[:, :1, ...], target))
            loss_G_GAN *= options.lambda_gan

        # ------------------------------------------------------------------------
        # Update reconstructor
        # ------------------------------------------------------------------------
        optimizers['G'].zero_grad()
        loss_G = losses['NLL'](reconstructed_image, target, uncertainty_map).mean()
        loss_G += loss_G_GAN

        loss_G.backward()
        optimizers['G'].step()

        return {'loss_D': 0 if self.evaluator is None else loss_D.item(), 'loss_G': loss_G.item()}

    def __call__(self) -> float:
        print('Creating trainer with the following options:')
        for key, value in vars(self.options).items():
            if key == 'device':
                value = value.type
            elif key == 'gpu_ids':
                value = 'cuda : ' + str(value) if torch.cuda.is_available() else 'cpu'
            print('    {:>25}: {:<30}'.format(key, 'None' if value is None else value), flush=True)

        # Create Reconstructor Model
        self.reconstructor = ReconstructorNetwork(
            number_of_cascade_blocks=self.options.number_of_cascade_blocks,
            n_downsampling=self.options.n_downsampling,
            number_of_filters=self.options.number_of_reconstructor_filters,
            number_of_layers_residual_bottleneck=self.options.number_of_layers_residual_bottleneck,
            mask_embed_dim=self.options.mask_embed_dim,
            dropout_probability=self.options.dropout_probability,
            img_width=self.options.image_width,
            use_deconv=self.options.use_deconv)

        if self.options.device.type == 'cuda':
            self.reconstructor = torch.nn.DataParallel(self.reconstructor).to(self.options.device)
        self.optimizers = {
            'G':
            optim.Adam(
                self.reconstructor.parameters(),
                lr=self.options.lr,
                betas=(self.options.beta1, 0.999))
        }

        # Create Evaluator Model
        if self.options.use_evaluator:
            self.evaluator = EvaluatorNetwork(
                number_of_filters=self.options.number_of_evaluator_filters,
                number_of_conv_layers=self.options.number_of_evaluator_convolution_layers,
                use_sigmoid=False,  # TODO : do we keep this? Will add option based on the decision
                width=self.options.image_width,
                mask_embed_dim=self.options.mask_embed_dim)
            self.evaluator = torch.nn.DataParallel(self.evaluator).to(self.options.device)

            self.optimizers['D'] = optim.Adam(
                self.evaluator.parameters(), lr=self.options.lr, betas=(self.options.beta1, 0.999))

        train_loader, val_loader = self.get_loaders()

        self.load_from_checkpoint_if_present()
        self.load_weights_from_given_checkpoint()

        writer = SummaryWriter(os.path.join(self.options.checkpoints_dir, self.options.name))

        # Training engine and handlers
        train_engine = Engine(lambda engine, batch: self.
                              update(batch, self.reconstructor, self.evaluator, self.optimizers,
                                     self.losses, self.fft_functions, self.options))

        val_engine = Engine(lambda engine, batch: self.inference(batch, self.reconstructor, self.
                                                                 fft_functions, self.options))
        validation_mse = Loss(
            loss_fn=lambda x, y: x, output_transform=lambda x: (x['MSE'], x['ground_truth']))
        validation_mse.attach(val_engine, name='mse')
        validation_ssim = Loss(
            loss_fn=lambda x, y: x, output_transform=lambda x: (x['SSIM'], x['ground_truth']))
        validation_ssim.attach(val_engine, name='ssim')

        monitoring_metrics = ['loss_D', 'loss_G']

        progress_bar = ProgressBar()
        progress_bar.attach(train_engine, metric_names=monitoring_metrics)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation_and_update_best_checkpoint,
            val_engine=val_engine,
            progress_bar=progress_bar,
            val_loader=val_loader,
            trainer=self)

        # Tensorboard Plots
        @train_engine.on(Events.ITERATION_COMPLETED)
        def plot_training_loss(engine):
            writer.add_scalar("training/generator_loss", train_engine.state.output['loss_G'],
                              engine.state.iteration)
            writer.add_scalar("training/discriminator_loss", train_engine.state.output['loss_D'],
                              engine.state.iteration)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def plot_validation_loss(engine):
            writer.add_scalar("validation/MSE", val_engine.state.output['MSE'], engine.state.epoch)
            writer.add_scalar("validation/SSIM", val_engine.state.output['SSIM'],
                              engine.state.epoch)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def plot_validation_images(engine):
            ground_truth = util.create_grid_from_tensor(val_engine.state.output['ground_truth'])
            writer.add_image("validation_images/ground_truth", ground_truth, engine.state.epoch)

            zero_filled_image = util.create_grid_from_tensor(
                val_engine.state.output['zero_filled_image'])
            writer.add_image("validation_images/zero_filled_image", zero_filled_image,
                             engine.state.epoch)

            reconstructed_image = util.create_grid_from_tensor(
                val_engine.state.output['zero_filled_image'])
            writer.add_image("validation_images/reconstructed_image", reconstructed_image,
                             engine.state.epoch)

            uncertainty_map = util.gray2heatmap(
                util.create_grid_from_tensor(val_engine.state.output['uncertainty_map'].exp()),
                cmap='jet')
            writer.add_image("validation_images/uncertainty_map", uncertainty_map,
                             engine.state.epoch)

            difference = util.create_grid_from_tensor(
                torch.abs(val_engine.state.output['ground_truth'] -
                          val_engine.state.output['reconstructed_image']))
            difference = util.gray2heatmap(difference, cmap='gray')
            writer.add_image("validation_images/difference", difference, engine.state.epoch)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            save_regular_checkpoint,
            trainer=self,
            progress_bar=progress_bar)

        train_engine.run(train_loader, self.options.max_epochs - self.completed_epochs)

        writer.close()

        return self.best_validation_score

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:  # submitit expects this function
        save_checkpoint_function(self, 'regular_checkpoint')
        trainer = Trainer(self.options)
        return submitit.helpers.DelayedSubmission(trainer)


if __name__ == '__main__':
    options_ = TrainOptions().parse()  # TODO: need to clean up options list
    options_.device = torch.device('cuda:{}'.format(
        options_.gpu_ids[0])) if options_.gpu_ids else torch.device('cpu')

    trainer_ = Trainer(options_)
    trainer_()
