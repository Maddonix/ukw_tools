import numpy as np
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torch import sigmoid


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class ImagePredictionLogger(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # print("_________________________-")
            # print(vars(trainer))
            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.val_dataloaders[0]))
            val_imgs, _val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs).to("cpu")
            # preds = torch.argmax(logits, axis=-1)
            preds = sigmoid(logits).numpy()  # .squeeze()

            _label_list = pl_module.labels

            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            preds = np.argwhere(preds)

            pred_labels = {i: [] for i in range(len(val_imgs))}
            for _pred in preds:
                pred_labels[_pred[0]].append(_label_list[_pred[1]])
            pred_labels = [pred_labels[i] for i in range(len(val_imgs))]

            _val_labels = _val_labels.to("cpu").numpy()
            _val_labels = np.argwhere(_val_labels)
            val_labels = {i: [] for i in range(len(val_imgs))}
            for _label in _val_labels:
                val_labels[_label[0]].append(_label_list[_label[1]])

            val_labels = [val_labels[i] for i in range(len(val_imgs))]

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            pred_labels[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )
