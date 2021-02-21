import pandas as pd
import numpy as np

import os
import cv2
import torch

from torch.utils import data


class ClassificationDataset(data.Dataset):
    def __init__(self, csv_file, path, labels, transform=None, debug=False):
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
    debug=False,
):

    train_dataset = ClassificationDataset(
        csv_file=train_csv_file,
        path=train_path,
        transform=train_transform,
        labels=train_labels,
        debug=debug,
    )

    test_dataset = ClassificationDataset(
        csv_file=test_csv_file,
        path=test_path,
        transform=test_transform,
        labels=test_labels,
        debug=debug,
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


class ObjectDetectionDataset(data.Dataset):
    def __init__(self, csv_file, path, transform=None, debug=False):
        super().__init__()
        if debug:
            self.data = pd.read_csv(csv_file).head(50)
        else:
            self.data = pd.read_csv(csv_file)
        self.image_ids = self.data['patientId'].unique()
        self.image_dir = path
        self.transforms = transform

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.data[self.data['patientId'] == image_id]
        target_present = records["Target"].values[0]

        image = cv2.imread(
            f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if target_present == 1:

            boxes = records[['x', 'y', 'width', 'height']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            area = (boxes[:, 3] - boxes[:, 1]) * \
                (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)

            # there is only one class
            labels = torch.ones((records.shape[0],), dtype=torch.int64)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['patientId'] = torch.tensor([index])
            target['area'] = area
            target['iscrowd'] = iscrowd
        else:

            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["patientId"] = torch.tensor([index])
            target["area"] = torch.as_tensor(
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            if target_present:
                target['boxes'] = torch.stack(
                    tuple(map(torch.FloatTensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

def collate_func(batch):
    return tuple(zip(*batch))

def get_detection_dataset(
    train_csv_file,
    train_path,
    train_transform,
    train_bs,
    test_csv_file,
    test_path,
    test_transform,
    test_bs,
    debug=False,
):

    train_dataset = ObjectDetectionDataset(
        csv_file=train_csv_file,
        path=train_path, 
        transform=train_transform, 
        debug=False
    )

    test_dataset = ObjectDetectionDataset(
        csv_file=test_csv_file,
        path=test_path,
        transform=test_transform,
        debug=False
    )


    train_loader = data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, collate_fn=collate_func
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, num_workers=4, collate_fn=collate_func
    )

    return train_loader, test_loader
