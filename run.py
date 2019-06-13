from data import CreateFtTLoader
from models.fft_utils import RFFT, IFFT, create_mask
from models import create_model
from models.networks import GANLossKspace
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


def clamp(tensor):
    # TODO: supposed to be clamping to zscore 3, make option for this
    return tensor.clamp(-3, 3)


def certainty_loss(reconstruction, target, logvar):
    # gaussian nll loss
    l2 = F.mse_loss(reconstruction[:, :1, :, :], target[:, :1, :, :], reduce=False)

    # Clip logvar to make variance in [0.01, 5], for numerical stability
    logvar = logvar.clamp(-4.605, 1.609)
    one_over_var = torch.exp(-logvar)

    # uncertainty loss
    assert len(l2) == len(logvar)
    return 0.5 * (one_over_var * l2 + logvar)


def get_mask(mask, batch_size, options):
    if options.dynamic_mask_type == 'loader':
        return mask.to(options.device)
    return create_mask(batch_size, mask_type=options.dynamic_mask_type).to(options.device)


def preprocess_inputs(target, mask, fft_functions, options):
    # TODO move all the clamp calls to data pre-processing
    target = clamp(target.to(options.device)).detach()
    mask = get_mask(mask, target.shape[0], options)

    kspace_ground_truth = fft_functions['rfft'](target)
    zero_filled_reconstruction = fft_functions['ifft'](kspace_ground_truth * mask)

    target = torch.cat([target, torch.zeros_like(target)], dim=1)

    return zero_filled_reconstruction, target, mask


def inference(batch, reconstructor, fft_functions, options):
    reconstructor.eval()
    with torch.no_grad():
        zero_filled_reconstruction, target, mask = preprocess_inputs(batch[1], batch[0], fft_functions, options)

        # Get reconstructor output
        reconstructions_all_stages, logvars, mask_cond = reconstructor(zero_filled_reconstruction, mask)
        reconstruction_last_stage = reconstructions_all_stages[-1]

        mse = F.mse_loss(reconstruction_last_stage[:, :1, ...], target[:, :1, ...], size_average=True)
        ssim = util.ssim_metric(reconstruction_last_stage[:, :1, ...], target[:, :1, ...])

        return {'MSE': mse, 'SSIM': ssim}


def update(batch, reconstructor, evaluator, optimizers, losses, fft_functions, options):
    zero_filled_reconstruction, target, mask = preprocess_inputs(batch[1], batch[0], fft_functions, options)

    # Get reconstructor output
    reconstructions_all_stages, logvars, mask_cond = reconstructor(zero_filled_reconstruction, mask)
    reconstruction_last_stage = reconstructions_all_stages[-1]

    # ------------------------------------------------------------------------
    # Update evaluator
    # ------------------------------------------------------------------------
    optimizers['D'].zero_grad()
    fake = torch.cat([clamp(reconstruction_last_stage[:, :1, :, :]), mask_cond.detach()], dim=1)
    detached_fake = fake.detach()
    output = evaluator(detached_fake, mask)
    loss_D_fake = losses['GAN'](output, False, mask, degree=0, pred_and_gt=(detached_fake[:, :1, ...], target))

    real = torch.cat([clamp(target[:, :1, :, :]), mask_cond.detach()], dim=1)
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
    for stage, (reconstruction, logvar) in enumerate(zip(reconstructions_all_stages, logvars)):
        loss_G += losses['NLL'](reconstruction, target, logvar)
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
    train_data_loader, val_data_loader = CreateFtTLoader(options)

    model = create_model(options)
    model.setup(options)
    reconstructor = model.netG
    evaluator = model.netD

    optimizers = {
        'G': optim.Adam(reconstructor.parameters(), lr=options.lr, betas=(options.beta1, 0.999)),
        'D': optim.Adam(evaluator.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
    }

    criterion_gan = GANLossKspace(use_lsgan=not options.no_lsgan,
                                  use_mse_as_energy=options.use_mse_as_disc_energy,
                                  grad_ctx=model.opt.grad_ctx).to(model.device)

    # TODO replace certainty loss by the library Gaussian NLL
    losses = {'GAN': criterion_gan, 'NLL': certainty_loss}

    fft_functions = {'rfft': RFFT().to(options.device), 'ifft': IFFT().to(options.device)}

    trainer = Engine(
        lambda engine, batch: update(batch, reconstructor, evaluator, optimizers, losses, fft_functions, options))
    validation_engine = Engine(lambda engine, batch: inference(batch, reconstructor, fft_functions, options))

    # TODO implement code to save the best model
    checkpoint_handler = ModelCheckpoint(os.path.join(options.checkpoints_dir, options.name), 'networks',
                                         save_interval=1, n_saved=1, require_empty=True)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'reconstructor': reconstructor, 'evaluator': evaluator})

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

            checkpoint_handler(engine, {
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
