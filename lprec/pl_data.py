from typing import List
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule

from .datasets.license_plate_utils import LicensePlateUtils

class PLDataModule(LightningDataModule):
    def __init__(
        self,
        train_datasets: List[Dataset] = [],
        val_datasets: List[Dataset] = [],
        test_datasets: List[Dataset] = [],
        train_batch_size=32,
        test_batch_size=1,
        train_num_workers=8,
        test_num_workers=0,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets

    def train_dataloader(self):
        if len(self.train_datasets) == 0:
            return
        return DataLoader(
            ConcatDataset(self.train_datasets),
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.train_num_workers,
            prefetch_factor=2,
            collate_fn=LicensePlateUtils.collate_fn,
        )

    def val_dataloader(self):
        if len(self.val_datasets) == 0:
            return
        return DataLoader(
            ConcatDataset(self.val_datasets),
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.train_num_workers,
            collate_fn=LicensePlateUtils.collate_fn,
        )

    def test_dataloader(self):
        if len(self.test_datasets) == 0:
            return
        return DataLoader(
            ConcatDataset(self.test_datasets),
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            collate_fn=LicensePlateUtils.collate_fn,
        )
