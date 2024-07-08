import os
import logging
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf
import warnings

import torch
from tqdm import tqdm
import torch.nn.functional as F

from src.datasets.utils import pad_packed_collate
from src.datasets import dataset_factory
from src.models import model_factory

from src.utils.utils import get_cwd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = logging.getLogger("Validation")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="config", config_name="config_validation")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")
    logger.info("Loading parameters from config file")

    # training parameters
    checkpoint = cfg.checkpoint
    assert os.path.isfile(checkpoint), f"Checkpoint file {checkpoint} not found"

    model = model_factory(cfg.model).to(device)
    weights = torch.load(checkpoint)
    model.load_state_dict(weights["state_dict"])

    batch_size = cfg.hyperparams.batch_size

    # creates dataset and dataloader
    val_dataset = dataset_factory(cfg.dataset, data_partition="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_packed_collate)

    if cfg.model.backend.params.num_classes != val_dataset.number_classes:
        raise ValueError("Number of classes in the dataset and model do not match")

    logger.info("Start validation iterations ...")
    running_corrects = 0.
    with torch.no_grad():
        for batch_idx, (input, lengths, labels, landmarks) in enumerate(tqdm(val_loader)):
            logits = model(input.unsqueeze(1).to(device), landmarks.to(device), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

    accuracy_val = running_corrects / len(val_loader.dataset)

    logger.info(f"Accuracy_val= {accuracy_val:0.4f}")

if __name__ == "__main__":
    main()