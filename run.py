from data import create_data_loaders
from models.fft_utils import RFFT, IFFT, clamp, preprocess_inputs, gaussian_nll_loss
from models import create_model
from models.networks import GANLossKspace
from models.reconstruction import ReconstructorNetwork
from options.train_options import TrainOptions
from util import util

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer


def inference(batch, reconstructor, fft_functions, options):
    reconstructor.eval()
    with torch.no_grad():
        zero_filled_reconstruction, target, mask = preprocess_inputs(batch[1], batch[0], fft_functions, options)

        # Get reconstructor output
        reconstructions_all_stages, logvars, mask_cond = reconstructor(zero_filled_reconstruction, mask)
        reconstruction_last_stage = reconstructions_all_stages

        mse = F.mse_loss(reconstruction_last_stage[:, :1, ...], target[:, :1, ...], size_average=True)
        ssim = util.ssim_metric(reconstruction_last_stage[:, :1, ...], target[:, :1, ...])

        return {'MSE': mse, 'SSIM': ssim}


def update(batch, reconstructor, evaluator, optimizers, losses, fft_functions, options):
    zero_filled_reconstruction, target, mask = preprocess_inputs(batch[1], batch[0], fft_functions, options)

    # Get reconstructor output
    reconstruction_last_stage, uncertainty_last_stage, mask_cond = reconstructor(zero_filled_reconstruction, mask)
    # reconstruction_last_stage = reconstructions_all_stages

    # ------------------------------------------------------------------------
    # Update evaluator
    # ------------------------------------------------------------------------
    optimizers['D'].zero_grad()
    # considering only the real component of reconstruction
    fake = torch.cat([clamp(reconstruction_last_stage[:, :1, ...]), mask_cond.detach()], dim=1)
    detached_fake = fake.detach()
    output = evaluator(detached_fake, mask)
    loss_D_fake = losses['GAN'](output, False, mask, degree=0, pred_and_gt=(detached_fake[:, :1, ...], target))

    real = torch.cat([clamp(target[:, :1, ...]), mask_cond.detach()], dim=1)
    output = evaluator(real, mask)
    loss_D_real = losses['GAN'](output, True, mask, degree=1, pred_and_gt=(detached_fake[:, :1, ...], target))

    loss_D = loss_D_fake + loss_D_real
    loss_D.backward()
    optimizers['D'].step()

    # ------------------------------------------------------------------------
    # Update reconstructor
    # ------------------------------------------------------------------------
    optimizers['G'].zero_grad()
    loss_G = 0
    # for stage, (reconstruction, logvar) in enumerate(zip(reconstructions_all_stages, logvars)):
    loss_G += losses['NLL'](reconstruction_last_stage, target, uncertainty_last_stage)
    loss_G = loss_G.mean()
    output = evaluator(fake, mask)
    loss_G_GAN = losses['GAN'](output, True, mask, degree=1, updateG=True, pred_and_gt=(fake[:, :1, ...], target))
    loss_G_GAN *= options.lambda_gan

    loss_G += loss_G_GAN
    loss_G.backward()
    optimizers['G'].step()

    return {
        'loss_D': loss_D.item(),
        'loss_G': loss_G.item()
    }


# TODO Add tensorboard visualization
def main(options):
    max_epochs = options.niter + options.niter_decay + 1
    # TODO remove this
    max_epochs = 1
    train_data_loader, val_data_loader = create_data_loaders(options)

    model = create_model(options)
    model.setup(options)

    # Create Reconstructor Model
    reconstructor = ReconstructorNetwork(
        number_of_cascade_blocks=options.number_of_cascade_blocks,
        n_downsampling=options.n_downsampling,
        number_of_filters=options.number_of_filters,
        number_of_layers_residual_bottleneck=options.number_of_layers_residual_bottleneck,
        mask_embed_dim=options.mask_embed_dim,
        dropout_probability=options.dropout_probability,
        img_width=128,   # TODO : CHANGE!
        use_deconv=options.use_deconv)
    reconstructor = torch.nn.DataParallel(reconstructor).cuda()

    # Create Evaluator Model
    evaluator = model.netD

    # Optimizers and losses
    optimizers = {
        'G': optim.Adam(reconstructor.parameters(), lr=options.lr, betas=(options.beta1, 0.999)),
        'D': optim.Adam(evaluator.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
    }
    criterion_gan = GANLossKspace(use_lsgan=not options.no_lsgan,
                                  use_mse_as_energy=options.use_mse_as_disc_energy,
                                  grad_ctx=model.opt.grad_ctx).to(model.device)
    losses = {'GAN': criterion_gan, 'NLL': gaussian_nll_loss}

    fft_functions = {'rfft': RFFT().to(options.device), 'ifft': IFFT().to(options.device)}

    trainer = Engine(
        lambda engine, batch: update(batch, reconstructor, evaluator, optimizers, losses, fft_functions, options))
    validation_engine = Engine(lambda engine, batch: inference(batch, reconstructor, fft_functions, options))

    # Checkpoint event handlers
    regular_checkpoint_handler = ModelCheckpoint(os.path.join(options.checkpoints_dir, options.name), 'checkpoint',
                                                 save_interval=1, n_saved=1, require_empty=False)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=regular_checkpoint_handler,
                              to_save={'reconstructor': reconstructor,
                                       'evaluator': evaluator})

    best_checkpoint_handler = ModelCheckpoint(os.path.join(options.checkpoints_dir, options.name), 'checkpoint',
                                              score_function=lambda ev: -ev.state.output['MSE'],
                                              score_name='mse',
                                              n_saved=1,
                                              require_empty=False)
    validation_engine.add_event_handler(event_name=Events.COMPLETED, handler=best_checkpoint_handler,
                                        to_save={'reconstructor': reconstructor,
                                                 'evaluator': evaluator})

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    monitoring_metrics = ['loss_D', 'loss_G']
    progress_bar = ProgressBar()
    progress_bar.attach(trainer, metric_names=monitoring_metrics)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % options.print_freq == 0:
            fname = os.path.join(options.checkpoints_dir, options.name, 'train_log.txt')
            message = '(epoch: {:d}/{:d}, iters: {:d}/{:d}) '.format(engine.state.epoch,
                                                                     max_epochs,
                                                                     engine.state.iteration % len(train_data_loader),
                                                                     len(train_data_loader))
            for k, v in engine.state.metrics.items():
                message += '{}: {:.3f}'.format(k, v)

            print(message, flush=True)
            with open(fname, 'a') as f:
                f.write('{}\n'.format(message))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times_iter(engine):
        progress_bar.log_message(
            'Iteration {} done. Time per batch: {:.3f}[s]'.format(engine.state.iteration, timer.value()))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times_epoch(engine):
        progress_bar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            regular_checkpoint_handler(engine, {
                'reconstructor_exception': reconstructor,
                'evaluator_exception': evaluator
            })

        else:
            raise e

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_engine.run(val_data_loader)
        output = validation_engine.state.output
        progress_bar.log_message(
            'Validation Results - Epoch: {}  MSE: {:.3f} SSIM: {:.3f}'
            .format(engine.state.epoch, output['MSE'], output['SSIM']))

    trainer.run(train_data_loader, max_epochs)


if __name__ == '__main__':
    options = TrainOptions().parse()
    options.device = torch.device('cuda:{}'.format(options.gpu_ids[0])) if options.gpu_ids else torch.device('cpu')
    main(options)
