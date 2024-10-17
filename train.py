from .data.dataset import CustomDataset
from omegaconf import OmegaConf
from accelerate import Accelerator
from speechbrain.dataio.batch import PaddedBatch
from torch.utils.data import DataLoader


def build_dataloader(dataset_cfg):
    train_dataset = CustomDataset(
        dataset_cfg.train_file_list,
        dataset_cfg.base_folder,
        dataset_cfg.attr_names
    )
    valid_dataset = CustomDataset(
        dataset_cfg.valid_file_list,
        dataset_cfg.base_folder,
        dataset_cfg.attr_names
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=PaddedBatch,
        **dataset_cfg.dataloader_args
    )
    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=PaddedBatch,
        **dataset_cfg.dataloader_args
    )

    return train_loader, valid_loader


def train(config="config/default.yaml"):
    cfg = OmegaConf.load(config)

    train_loader, valid_loader = build_dataloader(cfg.dataset)
    

    accelerator = Accelerator()
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    )
