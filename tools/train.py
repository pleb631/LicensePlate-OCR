from pytorch_lightning.cli import LightningCLI

from lprec.pl_model import PLModel
from lprec.pl_data import PLDataModule


if __name__ == '__main__':
    LightningCLI(PLModel, PLDataModule)
