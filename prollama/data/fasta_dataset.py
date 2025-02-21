import gzip
from pathlib import Path
import numpy as np
from esm.data import FastaBatchedDataset
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


def load_gzip_file(fasta_file):
    sequence = []
    cur_seq_label = None
    buf = []

    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence.append((cur_seq_label, "".join(buf)))
        cur_seq_label = None
        buf = []

    with gzip.open(fasta_file, "rt") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line
                cur_seq_label = line
            else:  # sequence line
                buf.append(line.strip())

    _flush_current_seq()

    assert len(set(sequence)) == len(sequence), "Found duplicate sequence labels"

    return sequence


def save_fasta(fasta, filename):
    with open(filename, "w") as outfile:
        outfile.write("\n".join(["".join(row) for row in fasta]))


class SwissProt(Dataset):
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    filename = "uniprot_sprot.fasta.gz"

    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split
        self.download()
        self.process()
        self.dataset = FastaBatchedDataset.from_file(self.processed_path[self.split])

    @property
    def processed_path(self):
        return {"train": self.root / "train.fasta", "val": self.root / "val.fasta"}

    @property
    def raw_file_path(self):
        return self.root / self.filename

    def download(self):
        download_url(self.url, self.root, self.filename)

    def process(self):
        if self.processed_path[self.split].exists():
            return
        # split the dataset
        seq_list = load_gzip_file(self.raw_file_path)
        n_samples = len(seq_list)

        n_val = int(n_samples * 0.1)
        n_train = n_samples - n_val

        np.random.default_rng(seed=1234).shuffle(seq_list)
        seq_train = seq_list[:n_train]
        seq_val = seq_list[n_train:]
        save_fasta(seq_train, self.processed_path['train'])
        save_fasta(seq_val, self.processed_path['val'])

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        return self.dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq)
