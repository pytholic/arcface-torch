import timm
import torch
import torch.nn as nn

def create_model(model_name, pretrained=False):
    model = timm.create_model(model_name, num_classes=1, pretrained=pretrained)
    model.conv_head = nn.Conv2d(288, 512, kernel_size=(1, 1), stride=(1, 1))
    model.classifier = nn.Identity()
    return model