from data.dataset import CustomDataset
from omegaconf import OmegaConf
from accelerate import Accelerator
from speechbrain.dataio.batch import PaddedBatch
from torch.utils.data import DataLoader
import hydra
import warnings

warnings.filterwarnings("ignore")
accelerator = Accelerator()


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
        batch_size=dataset_cfg.train_batch_size,
        **dataset_cfg.dataloader_args
    )
    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=PaddedBatch,
        batch_size=dataset_cfg.valid_batch_size,
        ** dataset_cfg.dataloader_args
    )

    return train_loader, valid_loader


@accelerator.on_main_process
def main_print(*args):
    print(*args)


@hydra.main("config", "default.yaml", "1.2")
def train(cfg):
    train_loader, valid_loader = build_dataloader(cfg.dataset)

    # model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    #     model, optimizer, training_dataloader, scheduler
    # )

    valid_loader = accelerator.prepare(valid_loader)
    for i, x in enumerate(valid_loader):
        main_print(i, x)


if __name__ == "__main__":
    train()
