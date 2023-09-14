import logging
import logging.config
import sys
from pathlib import Path

from rich.logging import RichHandler
from dataclasses import dataclass
from simple_parsing.helpers import field

@dataclass
class Config:
    """
    Training configuration.
    """

    # * Path config
    BASE_DIR: str = Path(__file__).parent.parent.absolute()
    DATA_DIR: str = Path("/home/jovyan/haseeb-dataset-2080ti-1/datasets")

    # Train path
    train_root: str = DATA_DIR / "webface/CASIA-maxpy-clean"
    train_list: str = DATA_DIR / "webface/train_list.txt"

    # Test path
    lfw_root: str = DATA_DIR / "lfw-pytorch"

    # * Model config
    backbone: str = "model-timm"
    use_se: bool = False
    classify: str = "softmax"
    metric: str = "arc_margin"
    loss: str = "focal_loss"
    easy_margin: bool = False
    finetune: bool = False
    weights_path: str = ""

    # * Train config
    input_shape: tuple = (3, 128, 128)
    num_classes: int = 10575  # number of identities
    learning_rate: float = 1e-2  # inital learning rate for the optimizer
    train_batch_size: int = 64  # training batch size
    test_batch_size: int = 60  # test batch size
    max_epochs: int = 300 # maximum number of training epochs
    val_interval: int = 5  # validation interval
    optimizer: str = "sgd"
    sanity_check: bool = False # overfit sanity check
    checkpoints_path: str = "./checkpoints"
    patience: int = 20  # number of epochs with no improvement after which training will be stopped
    min_delta: float = 0.001  # minimum change in validation loss to be considered as an improvement
    use_gpu: bool = True  # whether to use gpu or not
    use_multi = True # whether to use distributed data training
    num_workers: int = 8

    # Learning rate scheduling
    scheduler: str = "multistep"  # either "multistep" or "step"
    milestones: tuple = (75, 150, 225)  # decay steps (for "multistep" LR only)
    step_size: int = 30  # decay every N epochs (for "step" LR only)
    gamma: float = 0.1  # multiplication factor for learning rate
    weight_decay: float = 5e-4

if __name__ == "__main__":
    args = Config()