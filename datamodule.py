import pytorch_lightning as pl
from torch.utils.data import DataLoader

from fake_dataset import FakeDataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        """
        Data preparation
        Download, tokenize or operations that use disk and it's done on a single gpu in a distributed scenario
        DO NOT ASSIGN self.variable in this methods!!!!!!!!!!
        """

        """
        download_from_source("http://banananana.db")
        """

    def setup(self, stage: str = None) -> None:
        """
        Hook for pytorch lightning for train test and validation preparation
        Args:
            stage:

        Returns:

        """
        if stage == "fit" or stage is None:
            self.train_dataset = FakeDataset(mode="train")
            self.val_dataset = FakeDataset(mode="validation")
            self.test_dataset = FakeDataset(mode="test")

    def train_dataloader(self) -> DataLoader:
        """
        Returns: train dataloader
        """
        return DataLoader(self.train_dataset, batch_size=64)

    def val_dataloader(self) -> DataLoader:
        """
        Returns: val dataloader
        """
        return DataLoader(self.val_dataset, batch_size=64)

    def test_dataloader(self) -> DataLoader:
        """
        Returns: test dataloader
        """
        return DataLoader(self.test_dataset, batch_size=64)
