from torch.utils.data import DataLoader
import pytorch_lightning as pl
from esm import Alphabet
from .fasta_dataset import SwissProt


class ProteinDataset(pl.LightningDataModule):
    datasets_map = {
        "swissprot": SwissProt,
    }

    def __init__(
        self,
        root,
        dataset_name='swissprot',
        truncation_length=None,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.dataset_cls = self.datasets_map[dataset_name]
        self.kwargs = kwargs
        # we use ESM-2 tokenizer
        # source: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        self.collate_fn = lambda batch: self.alphabet.get_batch_converter(truncation_length)(batch)[-1]

    def setup(self, stage='fit'):
        self.train_dataset = self.dataset_cls(self.root, split='train')
        self.val_dataset = self.dataset_cls(self.root, split='val')
        print(
            f"# Training samples: {len(self.train_dataset)} \n"
            f"# Val samples: {len(self.val_dataset)} \n"
        )
        try:
            self.test_dataset = self.dataset_cls(self.root, split='test')
            print(f"# Test samples: {len(self.test_dataset)}")
        except:
            pass

    def dataloader(self, dataset, **kwargs):
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.train_dataset,
            shuffle=True,
            **self.kwargs
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self.dataloader(
            self.val_dataset,
            shuffle=False,
            **self.kwargs
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self.dataloader(
            self.test_dataset,
            shuffle=False,
            **self.kwargs
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.pred_dataset is not None
        return self.dataloader(
            self.pred_dataset,
            shuffle=False,
            **self.kwargs
        )
