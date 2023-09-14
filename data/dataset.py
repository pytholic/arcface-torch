import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms as T


class Dataset(Dataset):
    def __init__(
        self, root, data_list_file, phase="train", input_shape=(3, 128, 128)
    ):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), "r") as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if self.phase == "train":
            self.transforms = T.Compose(
                [
                    T.Resize((128, 128)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transforms = T.Compose(
                [T.Resize((128, 128)), T.ToTensor(), normalize]
            )

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path).convert("RGB")
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = Dataset(
        root="./Datasets/webface/CASIA-maxpy-clean",
        data_list_file="./Datasets/webface/train_list.txt",
        phase="test",
        input_shape=(3, 128, 128),
    )

    # Split into train and val set
    train_set, val_set = random_split(dataset, [0.95, 0.05])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    print(f"Total training images: {len(train_set)}")
    print(f"Total validation images: {len(val_set)}")

    trainloader = DataLoader(train_set, batch_size=16)
    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        img += np.array([1, 1, 1])
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img * std + mean
        img = np.clip(img, 0, 1)  # Clip values to [0, 1] range
        img *= 255
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q") or key == 27:  # "q" or "Esc" key
            break
    cv2.destroyAllWindows()
