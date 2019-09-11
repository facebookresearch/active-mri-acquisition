import argparse
import os
import tempfile

import ignite.contrib.handlers
import ignite.engine
import ignite.metrics
import logging
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

import models.evaluator
import rl_env

from typing import Any, Callable, Dict, Tuple


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------
class EvaluatorDataset(torch.utils.data.Dataset):

    def __init__(self, horizon: int = 50, images_per_file: int = 100, split: str = 'train'):
        super(EvaluatorDataset).__init__()
        self.dataset_dir = \
            f'/checkpoint/lep/active_acq/train_no_evaluator_symmetric/il_dataset/{split}'
        self.horizon = horizon
        self.images_per_file = images_per_file
        self.num_images = 5000 if split == 'train' else 1000
        self.horizon = 32

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_index = index // self.horizon
        time_index = index % self.horizon

        file_index = image_index - (image_index % self.images_per_file)
        file_sub_index = image_index % self.images_per_file
        suffix = '{}-{}.npy'.format(file_index, file_index + self.images_per_file - 1)

        loaded_images = np.load(os.path.join(self.dataset_dir, 'images_{}'.format(suffix)))
        loaded_masks = np.load(os.path.join(self.dataset_dir, 'masks_{}'.format(suffix)))
        loaded_scores = np.load(os.path.join(self.dataset_dir, 'scores_{}'.format(suffix)))

        image = torch.from_numpy(loaded_images[file_sub_index]).float()
        mask = torch.from_numpy(loaded_masks[file_sub_index][time_index]).float()
        scores = torch.from_numpy(loaded_scores[file_sub_index][time_index]).float()

        return image, mask, scores

    def __len__(self):
        return self.num_images * self.horizon


# ---------------------------------------------------------------------------------------------
# Model and policy wrapper
# ---------------------------------------------------------------------------------------------
class EvaluatorPlusPlus(nn.Module):

    def __init__(self, img_width: int = 128, mask_embed_dim: int = 6, num_actions: int = 54):
        super(EvaluatorPlusPlus, self).__init__()
        self.evaluator = models.evaluator.EvaluatorNetwork(
            width=img_width, mask_embed_dim=mask_embed_dim)
        self.fc = nn.Linear(img_width, num_actions)
        self.embedding = nn.Linear(img_width, mask_embed_dim)

    def forward(self, reconstruction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.embedding(mask).view(mask.shape[0], -1, 1, 1)
        mask_embedding = mask_embedding.repeat(1, 1, reconstruction.shape[2],
                                               reconstruction.shape[3])
        x = self.evaluator(reconstruction, mask_embedding)
        return self.fc(x)


class EvaluatorPlusPlusPolicy:

    def __init__(self, model_path: str, initial_num_lines: int, device: torch.device):
        raise NotImplementedError('This code becomes stale with the new observation types')
        # self.evaluator = EvaluatorPlusPlus()
        # checkpoint = torch.load(model_path)
        # model_state_dict = {
        #     key.replace('module.', ''): value
        #     for (key, value) in checkpoint['model'].items()
        # }
        # self.evaluator.load_state_dict(model_state_dict)
        # self.evaluator.to(device)
        # self.initial_num_lines = initial_num_lines
        # self.device = device

    def get_action(self, obs: np.ndarray, _, __) -> int:
        reconstruction = torch.Tensor(obs[:1, :-1]).unsqueeze(0).to(self.device)
        mask = torch.Tensor(obs[:1, -1]).view(1, 1, 1, -1).to(self.device)
        scores = self.evaluator(reconstruction, mask)
        max_action = reconstruction.shape[
            3] // 2 if rl_env.CONJUGATE_SYMMETRIC else reconstruction.shape[3]
        scores.masked_fill_(mask.byte().squeeze()[self.initial_num_lines:max_action], -100000)
        return scores.argmax(dim=1).item()

    def init_episode(self):
        pass


# ---------------------------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------------------------
def evaluator_loss_fn(prediction: torch.Tensor,
                      ground_truth: torch.Tensor,
                      loss_type: str = 'soft_policy') -> torch.Tensor:
    if loss_type == 'soft_policy':
        gt = F.relu(ground_truth)
        gt /= gt.sum(dim=1).unsqueeze(1)
        return F.kl_div(F.log_softmax(prediction, dim=1), gt, reduction='batchmean')
    if loss_type == 'argmax_policy':
        best_actions = ground_truth.argmax(dim=1)
        return F.cross_entropy(prediction, best_actions)

    raise NotImplementedError


def preprocess_inputs(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = batch[0].unsqueeze(1).to(device)
    mask = batch[1].to(device)
    ground_truth = batch[2].to(device)
    return image, mask, ground_truth


def update(_: ignite.engine.Engine, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           model: EvaluatorPlusPlus, optimizer: optim.Optimizer, loss_fn: Callable,
           device: torch.device) -> Dict[str, torch.Tensor]:
    image, mask, ground_truth = preprocess_inputs(batch, device)
    optimizer.zero_grad()
    prediction = model(image, mask)
    loss = loss_fn(prediction, ground_truth)
    loss.backward()
    optimizer.step(None)
    return {'prediction': prediction, 'ground_truth': ground_truth, 'loss': loss.item()}


def inference(_: ignite.engine.Engine, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              model: EvaluatorPlusPlus, device: torch.device) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        image, mask, ground_truth = preprocess_inputs(batch, device)
        prediction = model(image, mask)

        return {
            'prediction': prediction,
            'ground_truth': ground_truth,
        }


def save_regular_checkpoint(engine: ignite.engine.Engine, trainer: 'EvaluatorPlusPlusTrainer',
                            progress_bar: ignite.contrib.handlers.ProgressBar):
    if (engine.state.iteration - 1) % trainer.options.save_freq == 0:
        full_path = save_checkpoint_function(trainer, 'regular_checkpoint')
        progress_bar.log_message('Saved regular checkpoint to {}. Epoch: {}, Iteration: {}'.format(
            full_path, trainer.completed_epochs, engine.state.iteration))


def save_checkpoint_function(trainer: 'EvaluatorPlusPlusTrainer', filename: str) -> str:
    # Ensures atomic checkpoint save to avoid corrupted files if it gets
    # preempted during a save operation
    tmp_filename = tempfile.NamedTemporaryFile(delete=False, dir=trainer.options.checkpoints_dir)
    try:
        torch.save(trainer.create_checkpoint(), tmp_filename)
    except BaseException:
        tmp_filename.close()
        os.remove(tmp_filename.name)
        raise
    else:
        tmp_filename.close()
        full_path = os.path.join(trainer.options.checkpoints_dir, filename + '.pth')
        os.rename(tmp_filename.name, full_path)
        return full_path


def run_validation_and_update_best_checkpoint(
        train_engine: ignite.engine.Engine, val_engine: ignite.engine.Engine,
        val_loader: torch.utils.data.DataLoader, progress_bar: ignite.contrib.handlers.ProgressBar,
        writer: tensorboardX.SummaryWriter, trainer: 'EvaluatorPlusPlusTrainer'):
    val_engine.run(val_loader)
    for key, value in val_engine.state.metrics.items():
        progress_bar.log_message('{} over validation set: {}'.format(key.capitalize(), value))
        writer.add_scalar('val/{}'.format(key), value, train_engine.state.epoch)
    trainer.completed_epochs += 1
    score = val_engine.state.metrics['mean_batch_regret']
    if score > trainer.best_validation_score:
        trainer.best_validation_score = score
        full_path = save_checkpoint_function(trainer, 'best_checkpoint')
        progress_bar.log_message('Saved best checkpoint to {}. Score: {}. Epoch: {}'.format(
            full_path, score, train_engine.state.iteration))


# ---------------------------------------------------------------------------------------------
# Utility functions for print/tensorboard logs
# ---------------------------------------------------------------------------------------------
def log_iteration(engine: ignite.engine.Engine, progress_bar: ignite.contrib.handlers.ProgressBar,
                  writer: tensorboardX.SummaryWriter):
    if (engine.state.iteration + 1) % 1 == 0:
        progress_bar.log_message('Loss: {}. Iter: {}/{}. Epoch: {}/{}'.format(
            engine.state.output['loss'], engine.state.iteration % len(engine.state.dataloader),
            len(engine.state.dataloader), engine.state.epoch, engine.state.max_epochs))
    writer.add_scalar('training/loss', engine.state.output['loss'], engine.state.iteration)


def regret_output_transform(output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return output['prediction'], output['ground_truth']


# TODO check why some scans are actually increasing MSE
def regret_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    chosen = F.softmax(prediction, dim=1).argmax(dim=1).unsqueeze(1)
    gt = (target - target.min(dim=1)[0].unsqueeze(1)) / (
        target.max(dim=1)[0] - target.min(dim=1)[0]).unsqueeze(1)
    regret = gt.max(1)[0] - gt.gather(1, chosen).squeeze()
    return regret.mean()


def log_train_metrics(engine: ignite.engine.Engine,
                      progress_bar: ignite.contrib.handlers.ProgressBar,
                      writer: tensorboardX.SummaryWriter):
    for key, value in engine.state.metrics.items():
        progress_bar.log_message('{} over training set: {}'.format(key.capitalize(), value))
        writer.add_scalar('train/{}'.format(key), value, engine.state.epoch)


# ---------------------------------------------------------------------------------------------
# Main training code
# ---------------------------------------------------------------------------------------------
class EvaluatorPlusPlusTrainer:

    def __init__(self, model: EvaluatorPlusPlus, optimizer: optim.Optimizer,
                 data_loaders: Dict[str, torch.utils.data.DataLoader], options: argparse.Namespace):
        self.model = model
        self.optimizer = optimizer
        self.data_loaders = data_loaders
        self.options = options
        self.best_validation_score = 0
        self.completed_epochs = 0
        self.writer = tensorboardX.SummaryWriter(os.path.join(options.checkpoints_dir, 'tb_logs'))

    def create_checkpoint(self) -> Dict[str, Any]:
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'completed_epochs': self.completed_epochs,
            'best_validation_score': self.best_validation_score,
        }

    def load_from_checkpoint_if_present(self):
        if not os.path.exists(self.options.checkpoints_dir):
            return
        logging.info('Loading checkpoint found at {}'.format(self.options.checkpoints_dir))
        files = os.listdir(self.options.checkpoints_dir)
        for filename in files:
            if 'regular_checkpoint' in filename:
                logging.info('Loading checkpoint at {}'.format(filename))
                checkpoint = torch.load(os.path.join(self.options.checkpoints_dir, filename))
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.completed_epochs = checkpoint['completed_epochs']
                self.best_validation_score = checkpoint['best_validation_score']

    def __call__(self) -> float:
        logging.info('Running EvaluatorPlusPlus trainer with the following options:')
        for key, value in vars(self.options).items():
            if key == 'device':
                value = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info('    {:>25}: {:<30}'.format(key, value))

        if self.options.device.type == 'cuda' and torch.torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.options.device)

        train_engine = ignite.engine.Engine(lambda engine, batch: update(
            engine, batch, self.model, self.optimizer, evaluator_loss_fn, self.options.device))
        val_engine = ignite.engine.Engine(lambda engine, batch: inference(
            engine, batch, self.model, self.options.device))

        self.load_from_checkpoint_if_present()

        progress_bar = ignite.contrib.handlers.ProgressBar()
        progress_bar.attach(train_engine)

        # Train engine events
        train_engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, log_iteration,
                                       progress_bar, self.writer)
        train_engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED,
                                       save_regular_checkpoint, self, progress_bar)
        train_engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED,
                                       run_validation_and_update_best_checkpoint, val_engine,
                                       self.data_loaders['val'], progress_bar, self.writer, self)

        # Metrics
        val_regret_metric = ignite.metrics.Loss(
            regret_loss, output_transform=regret_output_transform)
        val_regret_metric.attach(val_engine, 'mean_batch_regret')
        train_regret_metric = ignite.metrics.Loss(
            regret_loss, output_transform=regret_output_transform)
        train_regret_metric.attach(train_engine, 'mean_batch_regret')
        train_engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, log_train_metrics,
                                       progress_bar, self.writer)

        train_engine.run(self.data_loaders['train'],
                         self.options.max_epochs - self.completed_epochs)

        self.writer.close()

        return self.best_validation_score
