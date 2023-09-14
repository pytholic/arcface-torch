# -*- coding: utf-8 -*-
"""
Created on 2023.08.10

@author: pytholic 
"""

import os
import time

import cv2
import numpy as np
import torch

from config import Config, logger
from models import *
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from torch.nn import DataParallel
import torch.nn.functional as F

torch.manual_seed(17)

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def lfw_test(model, device, lfw_path, batch_size, num_workers):
    """
    Test function for LFW dataset.
    """

    model.eval()

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = T.Compose([T.Resize((128, 128)), T.ToTensor(), normalize])
    test_set = torchvision.datasets.LFWPeople(root=lfw_path, download=True, split="test", image_set="deepfunneled", transform=transform)
    
    logger.info("Loaded test data successfully...")
    print(f"Total test images: {len(test_set)}")
    
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            features = model(images)
            features = F.normalize(features, p=2, dim=1)  # Normalize features
            features = features.cpu().numpy()

            # Calculate cosine similarity
            similarities = np.dot(features, features.T)

            # Predicted labels are the indices of highest similarity for each image
            predicted_labels = np.argmax(similarities, axis=1)
            labels = labels.cpu().numpy()

            print(predicted_labels, labels)
            break

            total_correct += (predicted_labels == labels).sum()
            total_samples += len(labels)

    test_accuracy = (total_correct / total_samples) * 100
    logger.info(f"LFW Accuracy: {test_accuracy:.4f}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logger.info("Reading arguments...")
    args = Config()
    logger.info(args)

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
        model = create_model("mobilenetv3_small_050")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model.to(device)


    # Load weights
    state = torch.load("/home/jovyan/haseeb-rnd/haseeb-data/rnd.ml.pivo-facereid/checkpoints/resnet18_83.pth")
    model.load_state_dict(state)
    logger.info("Model loaded successfully...")
    # print(model)


    # # * Test
    logger.info("Testing model...")
    lfw_test(
        model,
        device,
        args.lfw_root,
        args.test_batch_size,
        args.num_workers,
    )

