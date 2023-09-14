"""
Created on 2023.08.09

@author: pytholic 
"""

from __future__ import print_function

import math
import os
import random
import time
from datetime import datetime
from test import *

import numpy as np
import torch
from clearml import Logger, Task
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, Subset, random_split

from config import Config, logger
from data import Dataset
from models import *
from utils import compute_model_complexity

import torch.multiprocessing as mp
from torch.nn import DataParallel

random.seed(17)
torch.manual_seed(17)

# Save model utility
def save_model(model, save_path, name, epoch):
    save_name = os.path.join(save_path, name + "_" + str(epoch) + ".pth")
    torch.save(model.state_dict(), save_name)

# Sanity check subset function
def create_subset(trainset, valset):
    train_subset = Subset(trainset, range(50))
    val_subset = Subset(valset, range(5))

    return train_subset, val_subset


def main():

    # * Read args
    logger.info("Reading arguments...")
    args = Config()
    logger.info(args)

    # * Load data
    logger.info("Preparing dataset...")
    dataset = Dataset(
        args.train_root,
        args.train_list,
        phase="train",
        input_shape=args.input_shape,
    )

    # Split into train and val set
    train_set, val_set = random_split(dataset, [0.95, 0.05])

    # Create subset (sanity check)
    if args.sanity_check:
        train_set, val_set = create_subset(train_set, val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True
    )

    logger.info(f"Total training images: {len(train_set)}")
    logger.info(f"Total validation images: {len(val_set)}")

    # identity_list = get_lfw_list(args.lfw_test_list)
    # img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]

    # * Training setup
    # Define model
    if args.backbone == "resnet18":
        model = resnet_face18(use_se=args.use_se)
    elif args.backbone == "resnet34":
        model = resnet34()
    elif args.backbone == "resnet50":
        model = resnet50()
    elif args.backbone == "mobilenetv3":
        model = MobileNetV3()
    elif args.backbone == "slimnet":
        model = SlimNet()
    elif args.backbone == "model-timm":
        model = create_model("mobilenetv3_small_050", pretrained=True)

    logger.info(f"Using {args.backbone} model...")
    params, flops = compute_model_complexity(model, (1, 3, 128, 128))
    print("Total params: %.2fM" % (params / 1000000.0))
    print("Total flops: %.2fM" % (flops / 1000000.0))

    # Loss, metric and optimizer
    if args.loss == "focal_loss":
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.metric == "add_margin":
        metric_fc = AddMarginProduct(512, args.num_classes, s=30, m=0.35)
    elif args.metric == "arc_margin":
        metric_fc = ArcMarginProduct(
            512,
            args.num_classes,
            s=30,
            m=0.5,
            easy_margin=args.easy_margin,
            use_gpu=args.use_gpu,
        )
    elif args.metric == "sphere":
        metric_fc = SphereProduct(512, args.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, args.num_classes)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [
                {"params": model.parameters()},
                {"params": metric_fc.parameters()},
            ],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": metric_fc.parameters()},
            ],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    
    # Set device
    if args.use_multi:
        model = DataParallel(model)
        metric_fc = DataParallel(metric_fc)
    if args.use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        metric_fc.to(device)

    if args.scheduler == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    # * Create clearml task
    logger.info("Initializing ClearML Task...")
    task = Task.init(
        project_name="3i-Facereid",
        task_name=f"{args.backbone}-webface-128x128-{datetime.now()}",
        # output_uri="/home/jovyan/haseeb-rnd/haseeb-data/artifacts/facereid/",
    )
    task.connect(args, name="args")  # add args to clearml logging

    # * Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    logger.info(f"Starting training...")
    start = time.perf_counter()

    train_loss = float("inf")
    val_loss = float("inf")

    for i in range(1, args.max_epochs + 1):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        logger.info(f"date and time = {dt_string}")
        epoch_start = time.perf_counter()
        logger.info(f"learning rate: {optimizer.param_groups[0]['lr']}")
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_iters = len(train_loader)
        for ii, data in enumerate(train_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch [{i}/{args.max_epochs}] - Iteration [{ii+1}/{total_iters}] - Batch loss: {loss:.4f}"
            )

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()

            batch_acc = np.mean((output == label).astype(int))
            # correct_predictions = np.sum(output == label)
            # batch_acc = correct_predictions / len(label)
            epoch_acc += batch_acc

        epoch_end = time.perf_counter()

        # Calculate and log train metrics
        train_acc = (epoch_acc / len(train_loader)) * 100
        train_loss = epoch_loss / len(train_loader)

        print(
            f"Epoch [{i}/{args.max_epochs}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.4f} - Epoch Time: {epoch_end-epoch_start:.4f}"
        )

        Logger.current_logger().report_scalar("train", "loss", iteration=i, value=train_loss)
        Logger.current_logger().report_scalar("train", "acc", iteration=i, value=train_acc)

        # * Validation step
        if i % args.val_interval == 0:
            logger.info("Running validation step...")
            val_start = time.perf_counter()
            model.eval()
            epoch_loss = 0.0
            epoch_acc = 0.0
            with torch.no_grad():
                for ii, data in enumerate(val_loader):
                    data_input, label = data
                    data_input = data_input.to(device)
                    label = label.to(device).long()
                    feature = model(data_input)
                    output = metric_fc(feature, label)
                    loss = criterion(output, label)
                    epoch_loss += loss.item()

                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1)
                    label = label.data.cpu().numpy()

                    batch_acc = np.mean((output == label).astype(int))
                    # correct_predictions = np.sum(output == label)
                    # batch_acc = correct_predictions / len(label)
                    epoch_acc += batch_acc

                val_end = time.perf_counter()

                # Calculate and log validation metrics
                val_acc = (epoch_acc / len(val_loader)) * 100
                val_loss = epoch_loss / len(val_loader)

            print(
                f"Epoch [{i}/{args.max_epochs}] - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f} - Validation Time: {val_end-val_start:.4f}"
            )

            Logger.current_logger().report_scalar("val", "loss", iteration=i, value=val_loss)
            Logger.current_logger().report_scalar("val", "acc", iteration=i, value=val_acc)

            # Save model
            if val_loss + args.min_delta < best_val_loss:
                if not os.path.exists(args.checkpoints_path):
                    os.makedirs(args.checkpoints_path)
                save_model(model, args.checkpoints_path, args.backbone, i)
                logger.info(f"Epoch [{i}] - Saved model successfully...")
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check early stopping criteria
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping at epoch {i}")
                break

        scheduler.step()
        torch.cuda.empty_cache()

    # End of training
    end = time.perf_counter()
    logger.info("Training completed!")
    print(f"Total training time: {end - start}")

if __name__ == "__main__":
    main()
