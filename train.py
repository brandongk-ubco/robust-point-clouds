from robustpointclouds.lightningmodule import mmdetection3dLightningModule
from robustpointclouds.datamodule import mmdetection3dDataModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.cli import SaveConfigCallback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
import os
import sys
import robustpointclouds.models.detectors.adversarial_voxel_net  # noqa: F401
import robustpointclouds.models.detectors.adversarial_ssd3d_net  # noqa: F401
import robustpointclouds.models.adversaries.voxel_perturber  # noqa: F401


class MySaveConfigCallback(SaveConfigCallback):

    def on_train_start(self, trainer: Trainer, _: LightningModule) -> None:
        log_dir = trainer.log_dir or trainer.default_root_dir
        config_path = os.path.join(log_dir, self.config_filename)
        self.parser.save(self.config,
                         config_path,
                         skip_none=False,
                         overwrite=True)


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.config_file", "data.config_file")


if __name__ == "__main__":
    cli = MyLightningCLI(mmdetection3dLightningModule,
                         mmdetection3dDataModule,
                         seed_everything_default=42,
                         trainer_defaults={
                             "accelerator": "gpu",
                             "devices": 1,
                             "max_epochs": sys.maxsize
                         })
