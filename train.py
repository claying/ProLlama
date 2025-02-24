import logging
import hydra

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from prollama.model import ProLlama


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver('eval', eval)

log = logging.getLogger(__name__)

@hydra.main(
    version_base="1.3", config_path="./config", config_name="train"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Resolved configs:\n{OmegaConf.to_container(cfg, resolve=True)}")
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate your model
    if cfg.model.pretrained_path is None:
        model = ProLlama(cfg)
    else:
        log.info(f"Loading model from {cfg.model.pretrained_path}...")
        model = ProLlama.load_from_checkpoint(cfg.model.pretrained_path, new_cfg=cfg)
    if torch.cuda.is_available():
        # torch.compile could accelerate training by ~10%
        model = torch.compile(model)

    # The datamodule has already been instantiate in the ProLlama module
    datamodule = model._datamodule

    # Create loggers
    logger = []
    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(project="ProLlama")
        logger.append(wandb_logger)
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    # We log learning rates during training, and the best model ckpt
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            monitor=f'val/loss',
            dirpath=cfg.logs.path,
            filename=cfg.model.model_name,
            mode='min',
        )
    ]

    # Instantiate the Trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(model, datamodule)

    trainer.save_checkpoint(f"{cfg.logs.path}/{cfg.model.model_name}-last.ckpt")

    trainer.validate(model, datamodule)

    samples = model.generate(10)
    print("Generated samples: ")
    print(samples)


if __name__ == "__main__":
    main()
