import os
import random
from bidict import bidict
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split
from PIL import Image


class MarineCreatureDataset(Dataset):
    label_map = bidict({
        'Corals': 0,
        'Crabs': 1,
        'Dolphin': 2,
        'Eel': 3,
        'Jelly Fish': 4,
        'Lobster': 5,
        'Nudibranchs': 6,
        'Octopus': 7,
        'Penguin': 8,
        'Puffers': 9,
        'Sea Rays': 10,
        'Sea Urchins': 11,
        'Seahorse': 12,
        'Seal': 13,
        'Sharks': 14,
        'Squid': 15,
        'Starfish': 16,
        'Turtle_Tortoise': 17,
        'Whale': 18
    })

    def __init__(self, image_dir, image_size, mode, random_seed=42):
        assert mode in ['train', 'test', 'valid']
        self.name = 'MarineCreatureDataset'
        self.image_dir = image_dir
        self.mode = mode
        self.image_size = image_size

        self.images = []
        self.labels = []
        label_dir = os.listdir(image_dir)
        for label in label_dir:
            for img_path in os.listdir(os.path.join(image_dir, label)):
                self.images.append(os.path.join(self.image_dir, label, img_path))
                self.labels.append(label)

        self.full_size = len(self.images)

        random.seed(random_seed)
        idx = [i for i in range(self.full_size)]
        random.shuffle(idx)
        if self.mode == 'train':
            idx = idx[: int(0.8 * self.full_size)]
        elif self.mode == 'valid':
            idx = idx[int(0.8 * self.full_size): int(0.9 * self.full_size)]
        elif self.mode == 'test':
            idx = idx[int(0.9 * self.full_size):]
        self.images = [self.images[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]

    def augment(self, img):
        tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomRotation(degrees=(0, 45)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        return tf(img)
    
    def transform(self, img):
        tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        return tf(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_str = self.labels[idx]

        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.mode == 'train':
            img = self.augment(img)
        else:
            img = self.transform(img)

        label = MarineCreatureDataset.label_map[label_str]
        return {'img': img, 'label': label, 'img_path': img_path, 'label_str': label_str}


class MarineCreatureFolder(object):
    def __init__(self, image_dir, image_size):
        self.name = 'MarineCreatureDataset'
        self.image_dir = image_dir
        self.image_size = image_size

        self.full_dataset = ImageFolder(root=image_dir, transform=self.transform())
        self.train_size = int(0.8 * len(self.full_dataset))
        self.valid_size = int(0.1 * len(self.full_dataset))
        self.test_size = len(self.full_dataset) - self.train_size - self.valid_size

    def transform(self):
        tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        return tf

    def createDataset(self):
        return random_split(self.full_dataset, [self.train_size, self.valid_size, self.test_size])
