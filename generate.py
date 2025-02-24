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
    version_base="1.3", config_path="./config", config_name="generate"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Loading model from {cfg.model.pretrained_path}...")
    model = ProLlama.load_from_checkpoint(cfg.model.pretrained_path, new_cfg=cfg)
    if torch.cuda.is_available():
        model = torch.compile(model)

    datamodule = model._datamodule

    logger = []
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    if hasattr(datamodule, "test_dataset"):
        trainer.test(model, datamodule)
    else:
        trainer.validate(model, datamodule)

    samples = model.generate()
    print("Generated samples: ")
    for i, sample in enumerate(samples):
        print(f"Generated protein {i}: {sample}")


if __name__ == "__main__":
    main()