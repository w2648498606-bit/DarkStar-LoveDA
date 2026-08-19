import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler):
    """Deterministic epoch-wise permutation independent from model RNG.

    The permutation for a given epoch depends only on (seed, epoch, dataset length).
    That makes different model variants see samples in exactly the same order when
    they use the same data seed and epoch number.
    """

    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000003)
        order = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(order)

    def __len__(self):
        return len(self.data_source)


def seed_worker(worker_id: int):
    # PyTorch sets each worker's initial seed from the DataLoader generator.
    # Fold it into Python/NumPy so any legacy random code remains reproducible.
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_strict_determinism(enabled: bool):
    enabled = bool(enabled)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not enabled
        torch.backends.cudnn.deterministic = enabled
    if enabled:
        # warn_only avoids hard failures on a rare nondeterministic op while still
        # surfacing it in logs. The data pipeline remains fully deterministic.
        torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class LRGroupReport:
    name: str
    lr: float
    n_params: int
