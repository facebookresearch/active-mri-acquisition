import argparse
import ignite.engine
import logging
import os
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
from models.fft_utils import (
    preprocess_inputs,
    gaussian_nll_loss,
    GANLossKspace,
    to_magnitude,
    center_crop,
)
from models.reconstruction import ReconstructorNetwork
from options.train_options import TrainOptions
from util import util


def run_validation_and_update_best_checkpoint(
    engine: ignite.engine.Engine,
    val_engine: ignite.engine.Engine = None,
    progress_bar: ignite.contrib.handlers.ProgressBar = None,
    val_loader: torch.utils.data.DataLoader = None,
    trainer: "Trainer" = None,
):
    val_engine.run(val_loader)
    metrics = val_engine.state.metrics
    if trainer.options.use_evaluator:
        progress_bar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch}  "
            f"MSE: {metrics['mse']:.3f} SSIM: {metrics['ssim']:.3f} loss_D: "
            f"{metrics['loss_D']:.3f}"
        )
    else:
        progress_bar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch}  "
            f"MSE: {metrics['mse']:.3f} SSIM: {metrics['ssim']:.3f}"
        )
    trainer.completed_epochs += 1
    score = -metrics["loss_D"] if trainer.options.only_evaluator else -metrics["mse"]
    if score > trainer.best_validation_score:
        trainer.best_validation_score = score
        full_path = save_checkpoint_function(trainer, "best_checkpoint")
        progress_bar.log_message(
            f"Saved best checkpoint to {full_path}. Score: {score}. "
            f"Iteration: {engine.state.iteration}"
        )


def save_checkpoint_function(trainer: "Trainer", filename: str) -> str:
    # Ensures atomic checkpoint save to avoid corrupted files if preempted during a save operation
    tmp_filename = tempfile.NamedTemporaryFile(
        delete=False, dir=trainer.options.checkpoints_dir
    )
    try:
        torch.save(trainer.create_checkpoint(), tmp_filename)
    except BaseException:
        tmp_filename.close()
        os.remove(tmp_filename.name)
        raise
    else:
        tmp_filename.close()
        full_path = os.path.join(trainer.options.checkpoints_dir, filename + ".pth")
        os.rename(tmp_filename.name, full_path)
        return full_path


def save_regular_checkpoint(
    engine: ignite.engine.Engine,
    trainer: "Trainer" = None,
    progress_bar: ignite.contrib.handlers.ProgressBar = None,
):
    full_path = save_checkpoint_function(trainer, "regular_checkpoint")
    progress_bar.log_message(
        f"Saved regular checkpoint to {full_path}. Epoch: {trainer.completed_epochs}, "
        f"Iteration: {engine.state.iteration}"
    )


class Trainer:
    def __init__(self, options: argparse.Namespace):
        self.reconstructor = None
        self.evaluator = None
        self.options = options
        self.best_validation_score = -float("inf")
        self.completed_epochs = 0
        self.updates_performed = 0

        criterion_gan = GANLossKspace(
            use_mse_as_energy=options.use_mse_as_disc_energy,
            grad_ctx=options.grad_ctx,
            gamma=options.gamma,
            options=self.options,
        ).to(options.device)

        self.losses = {"GAN": criterion_gan, "NLL": gaussian_nll_loss}

        if self.options.only_evaluator:
            self.options.checkpoints_dir = os.path.join(
                self.options.checkpoints_dir, f"evaluator",
            )
        if not os.path.exists(self.options.checkpoints_dir):
            os.makedirs(self.options.checkpoints_dir)

    def create_checkpoint(self) -> Dict[str, Any]:
        return {
            "reconstructor": self.reconstructor.state_dict(),
            "evaluator": self.evaluator.state_dict()
            if self.options.use_evaluator
            else None,
            "options": self.options,
            "optimizer_G": self.optimizers["G"].state_dict(),
            "optimizer_D": self.optimizers["D"].state_dict()
            if self.options.use_evaluator
            else None,
            "completed_epochs": self.completed_epochs,
            "best_validation_score": self.best_validation_score,
            "updates_performed": self.updates_performed,
        }

    def get_loaders(
        self,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_data_loader, val_data_loader = create_data_loaders(self.options)
        return train_data_loader, val_data_loader

    def inference(self, batch):
        self.reconstructor.eval()

        with torch.no_grad():
            zero_filled_image, ground_truth, mask = preprocess_inputs(
                batch, self.options.dataroot, self.options.device
            )

            # Get reconstructor output
            reconstructed_image, uncertainty_map, mask_embedding = self.reconstructor(
                zero_filled_image, mask
            )

            reconstructor_eval = None
            ground_truth_eval = None
            if self.evaluator is not None:
                self.evaluator.eval()
                reconstructor_eval = self.evaluator(
                    reconstructed_image, mask_embedding, mask
                )
                ground_truth_eval = self.evaluator(ground_truth, mask_embedding, mask)

            # Compute magnitude (for val losses and plots)
            zero_filled_image_magnitude = to_magnitude(zero_filled_image)
            reconstructed_image_magnitude = to_magnitude(reconstructed_image)
            ground_truth_magnitude = to_magnitude(ground_truth)

            if self.options.dataroot == "KNEE_RAW":  # crop data
                reconstructed_image_magnitude = center_crop(
                    reconstructed_image_magnitude, [320, 320]
                )
                ground_truth_magnitude = center_crop(ground_truth_magnitude, [320, 320])
                zero_filled_image_magnitude = center_crop(
                    zero_filled_image_magnitude, [320, 320]
                )
                uncertainty_map = center_crop(uncertainty_map, [320, 320])

            return {
                "ground_truth": ground_truth,
                "zero_filled_image": zero_filled_image,
                "reconstructed_image": reconstructed_image,
                "ground_truth_magnitude": ground_truth_magnitude,
                "zero_filled_image_magnitude": zero_filled_image_magnitude,
                "reconstructed_image_magnitude": reconstructed_image_magnitude,
                "uncertainty_map": uncertainty_map,
                "mask": mask,
                "reconstructor_eval": reconstructor_eval,
                "ground_truth_eval": ground_truth_eval,
            }

    def load_from_checkpoint_if_present(self):
        if not os.path.exists(self.options.checkpoints_dir):
            return
        self.logger.info(f"Checkpoint folder found at {self.options.checkpoints_dir}")
        files = os.listdir(self.options.checkpoints_dir)
        for filename in files:
            if "regular_checkpoint" in filename:
                self.logger.info(f"Loading checkpoint {filename}.pth")
                checkpoint = torch.load(
                    os.path.join(self.options.checkpoints_dir, filename)
                )
                self.reconstructor.load_state_dict(checkpoint["reconstructor"])
                if self.options.use_evaluator:
                    self.evaluator.load_state_dict(checkpoint["evaluator"])
                    self.optimizers["D"].load_state_dict(checkpoint["optimizer_D"])
                self.optimizers["G"].load_state_dict(checkpoint["optimizer_G"])
                self.completed_epochs = checkpoint["completed_epochs"]
                self.best_validation_score = checkpoint["best_validation_score"]
                self.updates_performed = checkpoint["updates_performed"]

    def load_weights_from_given_checkpoint(self):
        if self.options.weights_checkpoint is None:
            return
        elif not os.path.exists(self.options.weights_checkpoint):
            raise FileNotFoundError("Specified weights checkpoint do not exist!")
        self.logger.info(
            f"Loading weights from checkpoint found at {self.options.weights_checkpoint}."
        )
        checkpoint = torch.load(self.options.weights_checkpoint)
        self.reconstructor.load_state_dict(checkpoint["reconstructor"])
        if (
            self.options.use_evaluator
            and "evaluator" in checkpoint.keys()
            and checkpoint["evaluator"] is not None
        ):
            self.evaluator.load_state_dict(checkpoint["evaluator"])
        else:
            self.logger.info("Evaluator was not loaded.")

    # TODO: consider adding learning rate scheduler
    def update(self, batch):
        if not self.options.only_evaluator:
            self.reconstructor.train()

        zero_filled_image, target, mask = preprocess_inputs(
            batch, self.options.dataroot, self.options.device
        )

        # Get reconstructor output
        reconstructed_image, uncertainty_map, mask_embedding = self.reconstructor(
            zero_filled_image, mask
        )

        # ------------------------------------------------------------------------
        # Update evaluator and compute generator GAN Loss
        # ------------------------------------------------------------------------
        loss_G_GAN = 0
        loss_D = torch.tensor(0.0)
        if self.evaluator is not None:
            self.evaluator.train()
            self.optimizers["D"].zero_grad()
            fake = reconstructed_image
            detached_fake = fake.detach()
            if self.options.mask_embed_dim != 0:
                mask_embedding = mask_embedding.detach()
            output = self.evaluator(
                detached_fake,
                mask_embedding,
                mask if self.options.add_mask_eval else None,
            )
            loss_D_fake = self.losses["GAN"](
                output, False, mask, degree=0, pred_and_gt=(detached_fake, target)
            )

            real = target
            output = self.evaluator(
                real, mask_embedding, mask if self.options.add_mask_eval else None
            )
            loss_D_real = self.losses["GAN"](
                output, True, mask, degree=1, pred_and_gt=(detached_fake, target)
            )

            loss_D = loss_D_fake + loss_D_real
            loss_D.backward(retain_graph=True)
            self.optimizers["D"].step()

            if not self.options.only_evaluator:
                output = self.evaluator(
                    fake, mask_embedding, mask if self.options.add_mask_eval else None
                )
                loss_G_GAN = self.losses["GAN"](
                    output,
                    True,
                    mask,
                    degree=1,
                    updateG=True,
                    pred_and_gt=(fake, target),
                )
                loss_G_GAN *= self.options.lambda_gan

        # ------------------------------------------------------------------------
        # Update reconstructor
        # ------------------------------------------------------------------------
        loss_G = torch.tensor(0.0)
        if not self.options.only_evaluator:
            self.optimizers["G"].zero_grad()
            loss_G = self.losses["NLL"](
                reconstructed_image, target, uncertainty_map, self.options
            ).mean()
            loss_G += loss_G_GAN
            loss_G.backward()
            self.optimizers["G"].step()

        self.updates_performed += 1

        return {"loss_D": loss_D.item(), "loss_G": loss_G.item()}

    def discriminator_loss(
        self,
        reconstructor_eval,
        target_eval,
        reconstructed_image=None,
        target=None,
        mask=None,
    ):
        if self.evaluator is None:
            return 0
        with torch.no_grad():
            loss_D_fake = self.losses["GAN"](
                reconstructor_eval,
                False,
                mask,
                degree=0,
                pred_and_gt=(reconstructed_image, target),
            )
            loss_D_real = self.losses["GAN"](
                target_eval,
                True,
                mask,
                degree=1,
                pred_and_gt=(reconstructed_image, target),
            )
            return loss_D_fake + loss_D_real

    def __call__(self) -> float:
        self.logger = logging.getLogger()
        if self.options.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            os.path.join(self.options.checkpoints_dir, "trainer.log")
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(levelname)s: %(message)s"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info("Creating trainer with the following options:")
        for key, value in vars(self.options).items():
            if key == "device":
                value = value.type
            elif key == "gpu_ids":
                value = "cuda : " + str(value) if torch.cuda.is_available() else "cpu"
            self.logger.info(f"    {key:>25}: {'None' if value is None else value:<30}")

        # Create Reconstructor Model
        self.reconstructor = ReconstructorNetwork(
            number_of_cascade_blocks=self.options.number_of_cascade_blocks,
            n_downsampling=self.options.n_downsampling,
            number_of_filters=self.options.number_of_reconstructor_filters,
            number_of_layers_residual_bottleneck=self.options.number_of_layers_residual_bottleneck,
            mask_embed_dim=self.options.mask_embed_dim,
            dropout_probability=self.options.dropout_probability,
            img_width=self.options.image_width,
            use_deconv=self.options.use_deconv,
        )

        if self.options.device.type == "cuda":
            self.reconstructor = torch.nn.DataParallel(self.reconstructor).to(
                self.options.device
            )
        self.optimizers = {
            "G": optim.Adam(
                self.reconstructor.parameters(),
                lr=self.options.lr,
                betas=(self.options.beta1, 0.999),
            )
        }

        # Create Evaluator Model
        if self.options.use_evaluator:
            self.evaluator = EvaluatorNetwork(
                number_of_filters=self.options.number_of_evaluator_filters,
                number_of_conv_layers=self.options.number_of_evaluator_convolution_layers,
                use_sigmoid=False,
                width=self.options.image_width,
                height=640 if self.options.dataroot == "KNEE_RAW" else None,
                mask_embed_dim=self.options.mask_embed_dim,
            )
            self.evaluator = torch.nn.DataParallel(self.evaluator).to(
                self.options.device
            )

            self.optimizers["D"] = optim.Adam(
                self.evaluator.parameters(),
                lr=self.options.lr,
                betas=(self.options.beta1, 0.999),
            )

        train_loader, val_loader = self.get_loaders()

        self.load_from_checkpoint_if_present()
        self.load_weights_from_given_checkpoint()

        writer = SummaryWriter(self.options.checkpoints_dir)

        # Training engine and handlers
        train_engine = Engine(lambda engine, batch: self.update(batch))
        val_engine = Engine(lambda engine, batch: self.inference(batch))

        validation_mse = Loss(
            loss_fn=F.mse_loss,
            output_transform=lambda x: (
                x["reconstructed_image_magnitude"],
                x["ground_truth_magnitude"],
            ),
        )
        validation_mse.attach(val_engine, name="mse")

        validation_ssim = Loss(
            loss_fn=util.compute_ssims,
            output_transform=lambda x: (
                x["reconstructed_image_magnitude"],
                x["ground_truth_magnitude"],
            ),
        )
        validation_ssim.attach(val_engine, name="ssim")

        if self.options.use_evaluator:
            validation_loss_d = Loss(
                loss_fn=self.discriminator_loss,
                output_transform=lambda x: (
                    x["reconstructor_eval"],
                    x["ground_truth_eval"],
                    {
                        "reconstructed_image": x["reconstructed_image"],
                        "target": x["ground_truth"],
                        "mask": x["mask"],
                    },
                ),
            )
            validation_loss_d.attach(val_engine, name="loss_D")

        progress_bar = ProgressBar()
        progress_bar.attach(train_engine)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation_and_update_best_checkpoint,
            val_engine=val_engine,
            progress_bar=progress_bar,
            val_loader=val_loader,
            trainer=self,
        )

        # Tensorboard Plots
        @train_engine.on(Events.ITERATION_COMPLETED)
        def plot_training_loss(engine):
            writer.add_scalar(
                "training/generator_loss",
                engine.state.output["loss_G"],
                self.updates_performed,
            )
            if "loss_D" in engine.state.output:
                writer.add_scalar(
                    "training/discriminator_loss",
                    engine.state.output["loss_D"],
                    self.updates_performed,
                )

        @train_engine.on(Events.EPOCH_COMPLETED)
        def plot_validation_loss(_):
            writer.add_scalar(
                "validation/MSE", val_engine.state.metrics["mse"], self.completed_epochs
            )
            writer.add_scalar(
                "validation/SSIM",
                val_engine.state.metrics["ssim"],
                self.completed_epochs,
            )
            if "loss_D" in val_engine.state.metrics:
                writer.add_scalar(
                    "validation/loss_D",
                    val_engine.state.metrics["loss_D"],
                    self.completed_epochs,
                )

        @train_engine.on(Events.EPOCH_COMPLETED)
        def plot_validation_images(_):
            ground_truth = val_engine.state.output["ground_truth_magnitude"]
            zero_filled_image = val_engine.state.output["zero_filled_image_magnitude"]
            reconstructed_image = val_engine.state.output[
                "reconstructed_image_magnitude"
            ]
            uncertainty_map = val_engine.state.output["uncertainty_map"]
            difference = torch.abs(ground_truth - reconstructed_image)

            # Create plots
            ground_truth = util.create_grid_from_tensor(ground_truth)
            writer.add_image(
                "validation_images/ground_truth", ground_truth, self.completed_epochs
            )

            zero_filled_image = util.create_grid_from_tensor(zero_filled_image)
            writer.add_image(
                "validation_images/zero_filled_image",
                zero_filled_image,
                self.completed_epochs,
            )

            reconstructed_image = util.create_grid_from_tensor(reconstructed_image)
            writer.add_image(
                "validation_images/reconstructed_image",
                reconstructed_image,
                self.completed_epochs,
            )

            uncertainty_map = util.gray2heatmap(
                util.create_grid_from_tensor(uncertainty_map.exp()), cmap="jet"
            )
            writer.add_image(
                "validation_images/uncertainty_map",
                uncertainty_map,
                self.completed_epochs,
            )

            difference = util.create_grid_from_tensor(difference)
            difference = util.gray2heatmap(difference, cmap="gray")
            writer.add_image(
                "validation_images/difference", difference, self.completed_epochs
            )

            mask = util.create_grid_from_tensor(
                val_engine.state.output["mask"].repeat(
                    1, 1, val_engine.state.output["mask"].shape[3], 1
                )
            )
            writer.add_image(
                "validation_images/mask_image", mask, self.completed_epochs
            )

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            save_regular_checkpoint,
            trainer=self,
            progress_bar=progress_bar,
        )

        train_engine.run(train_loader, self.options.max_epochs - self.completed_epochs)

        writer.close()

        return self.best_validation_score


if __name__ == "__main__":
    options_ = TrainOptions().parse()  # TODO: need to clean up options list
    options_.device = (
        torch.device("cuda:{}".format(options_.gpu_ids[0]))
        if options_.gpu_ids
        else torch.device("cpu")
    )
    options_.checkpoints_dir = os.path.join(options_.checkpoints_dir, options_.name)

    if not os.path.exists(options_.checkpoints_dir):
        os.makedirs(options_.checkpoints_dir)

    trainer_ = Trainer(options_)
    trainer_()
