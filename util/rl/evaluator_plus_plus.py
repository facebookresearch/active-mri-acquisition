from models.evaluator import EvaluatorNetwork

import h5py
import ignite.engine
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
    def __init__(self, horizon: int = 50):
        super(EvaluatorDataset).__init__()
        self.dataset = h5py.File('/checkpoint/lep/active_acq/full_test_run_py/il_dataset/dataset_0_4499.hdf5',
                                 'r', libver='latest', swmr=True)
        self.horizon = horizon

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_index = index // self.horizon
        time_index = index % self.horizon

        image = torch.from_numpy(self.dataset['images'][image_index]).float()
        mask = torch.from_numpy(self.dataset['masks'][image_index][time_index]).float()
        scores = torch.from_numpy(self.dataset['scores'][image_index][time_index]).float()

        return image, mask, scores

    def __len__(self):
        # return len(self.dataset['images']) * self.horizon
        return self.horizon


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
    gt = ground_truth.clone()
    if loss_type == 'soft_policy':
        max_gt = gt.max(dim=1)[0].unsqueeze(1)
        diff = max_gt - gt
        diff[gt == 0] = 0       # Ignore actions with MSE = 0
        probs = diff / diff.sum(dim=1).unsqueeze(1)
        return F.mse_loss(prediction, probs)        # TODO probably not the most appropriate (maybe use KLDiv?)
    if loss_type == 'one_hot_policy':
        gt[gt == 0] = float('inf')      # Discard actions with MSE = 0 (already scanned)
        best_actions = gt.argmin(dim=1)
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
        'ground_truth_scores': ground_truth,
        'loss': loss.item()
    }


# ---------------------------------------------------------------------------------------------
# Utility functions for print/tensorboard logs
# ---------------------------------------------------------------------------------------------
def log_iteration(engine: ignite.engine.Engine, progress_bar: ProgressBar = None):
    if (engine.state.iteration + 1) % 5 == 0:
        progress_bar.log_message('Loss: {}. Iter: {}/{}. Epoch: {}/{}'.format(
            engine.state.output['loss'], engine.state.iteration, len(engine.state.dataloader),
            engine.state.epoch, engine.state.max_epochs
        ))


def log_metrics(engine: ignite.engine.Engine, progress_bar: ProgressBar = None):
    for metric in engine.state.metrics.keys():
        progress_bar.log_message('{} over training set: {}'.format(metric.capitalize(), engine.state.metrics[metric]))


def regret_output_transform(output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = output['prediction']
    scores = output['ground_truth_scores'].clone()
    chosen = F.softmax(logits, dim=1).argmax(dim=1)
    return 0




def accuracy_output_transform(output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = output['prediction']
    scores = output['ground_truth_scores'].clone()
    scores[scores == 0] = float('inf')      # Discard actions with MSE = 0 (already scanned)

    return F.softmax(logits, dim=1), scores.argmin(dim=1)       # predicted probs, ground truth


# ---------------------------------------------------------------------------------------------
# Main training code
# ---------------------------------------------------------------------------------------------
def train(model: EvaluatorPlusPlus, data_loader: torch.utils.data.DataLoader, device: torch.device):
    if device.type == 'cuda' and torch.torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3.0e-4, betas=(0.5, 0.999))

    trainer = ignite.engine.Engine(
        lambda engine, batch: update(engine, batch, model, optimizer, evaluator_loss_fn, device))

    progress_bar = ProgressBar()
    progress_bar.attach(trainer)

    trainer.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, log_iteration, progress_bar=progress_bar)
    regret_metric = RunningAverage(None, output_transform=regret_output_transform)
    regret_metric.attach(trainer, 'regret')
    # metric = Accuracy(output_transform=accuracy_output_transform)
    # metric.attach(trainer, 'accuracy')
    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, log_metrics, progress_bar=progress_bar)

    trainer.run(data_loader, 10)
