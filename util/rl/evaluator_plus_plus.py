from models.evaluator import EvaluatorNetwork

import argparse
import ignite.engine
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, RunningAverage
from tensorboardX import SummaryWriter
from typing import Callable, Dict, Tuple


# TODO add training validation split
class EvaluatorDataset(torch.utils.data.Dataset):
    def __init__(self, horizon: int = 50, images_per_file: int = 100, num_images: int = 400):
        super(EvaluatorDataset).__init__()
        self.dataset_dir = '/checkpoint/lep/active_acq/full_test_run_py/il_dataset/'
        self.horizon = horizon
        self.images_per_file = images_per_file
        self.num_images = num_images

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


class EvaluatorPlusPlus(nn.Module):
    def __init__(self, img_width: int = 128, mask_embed_dim: int = 6, num_actions: int = 54):
        super(EvaluatorPlusPlus, self).__init__()
        self.evaluator = EvaluatorNetwork(width=img_width, mask_embed_dim=mask_embed_dim)
        self.fc = nn.Linear(img_width, num_actions)
        self.embedding = nn.Linear(img_width, mask_embed_dim)

    def forward(self, reconstruction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.embedding(mask).view(mask.shape[0], -1, 1, 1)
        mask_embedding = mask_embedding.repeat(1, 1, reconstruction.shape[2], reconstruction.shape[3])
        x = self.evaluator(reconstruction, mask_embedding)
        return self.fc(x)


# ---------------------------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------------------------
def evaluator_loss_fn(prediction: torch.Tensor,
                      ground_truth: torch.Tensor,
                      loss_type: str = 'soft_policy') -> torch.Tensor:
    if loss_type == 'soft_policy':
        gt = ground_truth / ground_truth.sum(dim=1).unsqueeze(1)
        return F.kl_div(F.log_softmax(prediction, dim=1), gt, reduction='batchmean')
    if loss_type == 'argmax_policy':
        best_actions = ground_truth.argmax(dim=1)
        return F.cross_entropy(prediction, best_actions)

    raise NotImplementedError


def preprocess_inputs(batch: Tuple[torch.Tensor], device: torch.device):
    image = batch[0].unsqueeze(1).to(device)
    mask = batch[1].to(device)
    ground_truth = batch[2].to(device)
    return image, mask, ground_truth


def update(_: ignite.engine.Engine,
           batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           model: EvaluatorPlusPlus,
           optimizer: optim.Optimizer,
           loss_fn: Callable,
           device: torch.device) -> Dict[str, torch.Tensor]:
    image, mask, ground_truth = preprocess_inputs(batch, device)
    optimizer.zero_grad()
    prediction = model(image, mask)
    loss = loss_fn(prediction, ground_truth)
    loss.backward()
    optimizer.step(None)
    return {
        'prediction': prediction,
        'ground_truth': ground_truth,
        'loss': loss.item()
    }


# ---------------------------------------------------------------------------------------------
# Utility functions for print/tensorboard logs
# ---------------------------------------------------------------------------------------------
def log_iteration(engine: ignite.engine.Engine, progress_bar: ProgressBar = None, writer: SummaryWriter = None):
    if (engine.state.iteration + 1) % 50 == 0:
        progress_bar.log_message('Loss: {}. Iter: {}/{}. Epoch: {}/{}'.format(
            engine.state.output['loss'],
            engine.state.iteration % len(engine.state.dataloader),
            len(engine.state.dataloader),
            engine.state.epoch, engine.state.max_epochs
        ))
    writer.add_scalar('training/loss', engine.state.output['loss'], engine.state.iteration)


def log_metrics(engine: ignite.engine.Engine, progress_bar: ProgressBar = None, writer: SummaryWriter = None):
    for metric in engine.state.metrics.keys():
        progress_bar.log_message('{} over training set: {}'.format(metric.capitalize(), engine.state.metrics[metric]))
        writer.add_scalar('training/{}'.format(metric), engine.state.metrics[metric], engine.state.epoch)


def regret_output_transform(output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen = F.softmax(output['prediction'], dim=1).argmax(dim=1).unsqueeze(1)
    gt = output['ground_truth'] / output['ground_truth'].sum(dim=1).unsqueeze(1)
    regrets = gt.max(1)[0] - gt.gather(1, chosen).squeeze()
    return regrets.mean()


def accuracy_output_transform(output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return F.softmax(output['prediction'], dim=1), output['ground_truth'].argmax(dim=1)


# ---------------------------------------------------------------------------------------------
# Main training code
# ---------------------------------------------------------------------------------------------
def train(model: EvaluatorPlusPlus, data_loader: torch.utils.data.DataLoader, options: 'argparse.Namespace'):
    writer = SummaryWriter(os.path.join(options.checkpoints_dir, 'tb_logs'))

    if options.device.type == 'cuda' and torch.torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(options.device)

    optimizer = optim.Adam(model.parameters(), lr=options.lr, betas=(options.beta1, options.beta2))

    trainer = ignite.engine.Engine(
        lambda engine, batch: update(engine, batch, model, optimizer, evaluator_loss_fn, options.device))

    progress_bar = ProgressBar()
    # progress_bar.attach(trainer)

    trainer.add_event_handler(
        ignite.engine.Events.ITERATION_COMPLETED, log_iteration, progress_bar=progress_bar, writer=writer)
    regret_metric = RunningAverage(None, output_transform=regret_output_transform)
    regret_metric.attach(trainer, 'mean_batch_regret')
    # metric = Accuracy(output_transform=accuracy_output_transform)
    # metric.attach(trainer, 'accuracy')
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED, log_metrics, progress_bar=progress_bar, writer=writer)

    trainer.run(data_loader, options.max_epochs)
