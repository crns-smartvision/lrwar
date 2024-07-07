import os
from os import path as osp
import logging
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf
import warnings

import statistics
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from src.datasets.utils import pad_packed_collate
from src.datasets.utils import mixup_criterion, mixup_data
from src.datasets import dataset_factory
from src.utils.utils import CosineScheduler
from src.models import model_factory

from src.utils.utils import get_cwd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="config", config_name="config_train")
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
    nb_epoch = cfg.hyperparams.nb_epoch
    batch_size = cfg.hyperparams.batch_size
    initial_lr =cfg.hyperparams.lr
    weight_decay = cfg.hyperparams.weight_decay
    alpha = cfg.hyperparams.alpha

    resume_training = False
    starting_epoch = 0
    if "resume" in cfg.hyperparams:
        if cfg.hyperparams.resume:
            resume_training = cfg.hyperparams.resume
            checkpoint_file = cfg.hyperparams.checkpoint
            assert osp.exists(checkpoint_file), f"Checkpoint file {checkpoint_file} not found"
            checkpoint = torch.load(checkpoint_file)
            starting_epoch = checkpoint["epoch"]
            logger.info(f"Resuming training from epoch {starting_epoch}")

    # creates dataset and datalaoder
    train_dataset = dataset_factory(cfg.dataset, data_partition="train")
    val_dataset = dataset_factory(cfg.dataset, data_partition="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_packed_collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_packed_collate)
    model = model_factory(cfg.model).to(device)

    scheduler = CosineScheduler(initial_lr, nb_epoch)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

    if resume_training:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if cfg.model.backend.params.num_classes != train_dataset.number_classes:
        raise ValueError("Number of classes in the dataset and model do not match")


    logger.info("Start Epochs ...")

    for epoch in range(starting_epoch,nb_epoch):
        loss_epoch = []

        for batch_idx, (videos, lengths, labels, landmarks) in enumerate(tqdm(train_loader)):

            input_videos, labels_a, labels_b, lam ,input_landmarks = mixup_data(videos, labels,landmarks, alpha)
            labels_a, labels_b = labels_a.to(device), labels_b.to(device)

            optimizer.zero_grad()

            logits = model(input_videos.unsqueeze(1).to(device),input_landmarks.to(device), lengths=lengths)

            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, logits)
            loss.backward()
            loss_epoch.append(loss.item())

            optimizer.step()

        logger.info(f"Epoch= {epoch:04d} Loss= {statistics.mean(loss_epoch):0.4f}")
        running_loss = 0.
        running_corrects = 0.

        with torch.no_grad():
            for batch_idx, (input, lengths, labels, landmarks) in enumerate(tqdm(val_loader)):
                logits = model(input.unsqueeze(1).to(device), landmarks.to(device), lengths=lengths)
                _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
                running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

                loss = criterion(logits, labels.to(device))
                running_loss += loss.item() * input.size(0)

        loss_val = running_loss / len(val_loader.dataset)
        accuracy_val = running_corrects / len(val_loader.dataset)

        logger.info(f"Epoch= {epoch:04d} Loss_val= {loss_val:0.4f} ---  Accuracy_val= {accuracy_val:0.4f}")

        checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "loss": loss_epoch}
        checkpoint_export_path = os.path.join(output_dir, f"{epoch}.pth")
        torch.save(checkpoint, checkpoint_export_path)
        logger.info(f"Checkpoint savec to: {checkpoint_export_path}")
        scheduler.adjust_lr(optimizer, epoch)


if __name__ == "__main__":
    main()