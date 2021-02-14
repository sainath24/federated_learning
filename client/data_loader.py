import pandas as pd
from torch.utils import data
import os
import cv2
import torch


class ClassificationDataset(data.Dataset):
    def __init__(self, csv_file, path, labels transform=None,debug=False):
        """
        :param: csv_file - path to csv file with format [Image, Label_Columns]
        :param: path - image dir path
        :param: labels - list of label columns
        :param: transorm - albumentations transforms
        """

        self.path = path
        if debug:
            self.data = pd.read_csv(csv_file).head(50)
        else:
            self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_name = os.path.join(self.path, self.data.loc[idx, "Image"] + ".jpg")
        img = cv2.imread(img_name)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.labels:
            labels = torch.tensor(self.data.loc[idx, self.labels])
            return {"image": img, "labels": labels}

        else:
            return {"image": img}


def get_classification_dataset(
    train_csv_file,
    train_path,
    train_transform,
    train_labels,
    train_bs,
    test_csv_file,
    test_path,
    test_transform,
    test_labels,
    test_bs,
):

    train_dataset = ClassificationDataset(
        csv_file=train_csv_file,
        path=train_path,
        transform=train_transform,
        labels=train_labels,
    )

    test_dataset = ClassificationDataset(
        csv_file=test_csv_file,
        path=test_path,
        transform=test_transform,
        labels=test_labels,
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, num_workers=4
    )

    return train_loader, test_loader
