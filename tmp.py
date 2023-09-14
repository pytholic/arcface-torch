import cv2
import numpy as np
import torch
import torchvision.transforms as T


def load_image(img_path, normalize=False):
    image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    # image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    if normalize:
        mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225, 0.485, 0.456, 0.406]
        # mean = [0.5, 0.5]
        # std = [0.5, 0.5]
        image /= 255
        for i in range(image.shape[0]):
            image[i, :, :, :] -= mean[i]
            image[i, :, :, :] /= std[i]
    return image


def load_image2(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


img = load_image(
    "/Users/3i-a1-2021-15/Developer/projects/face-tracking/reid/arcface-pytorch/data/Datasets/webface/CASIA-maxpy-clean/0000045/001.jpg",
    normalize=True,
)
print(img.shape)
print(img[:10])
