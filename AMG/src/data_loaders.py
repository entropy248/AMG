import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import transforms


__all__ = ['CIFAR10DataLoader', 'ImageNetDataLoader',
           'CIFAR100DataLoader',
           'TinyImageNetDataLoader', 'FlowersDataLoader']


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CIFAR10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=True)

        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class CIFAR100DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR100(root=data_dir, train=train, transform=transform, download=True)

        super(CIFAR100DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):

        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


class TinyImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):

        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        super(TinyImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


class FlowersDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            train = True
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            train = False
        self.dataset = FlowersDataset(root_dir=data_dir, transform=transform, train=train)
        super(FlowersDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if train else False,
            num_workers=num_workers,
        )


class FlowersDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        if train:
            train_paths = os.path.join(root_dir, 'train.txt')
            val_paths = os.path.join(root_dir, 'valid.txt')
            with open(train_paths, 'r') as tp:
                paths = tp.readlines()
            with open(val_paths, 'r') as vp:
                val_labels = vp.readlines()
                for val_path in val_labels:
                    paths.append(val_path)
        else:
            test_paths = os.path.join(root_dir, 'test.txt')
            with open(test_paths, 'r') as tsp:
                paths = tsp.readlines()

        self.paths = []
        self.labels = []
        self.transform = transform
        for path in paths:
            # print(path)
            label = path.rstrip('\n').split(' ')
            # print(label)
            self.paths.append(os.path.join(root_dir, label[0]))
            self.labels.append(int(label[1]))

    def __getitem__(self, item):
        image_path = self.paths[item]
        # print(self.labels[item])
        image = pil_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[item]

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':

    imagenet = ImageFolder(root=os.path.join('../data/ImageNet', 'train'), transform=None)
    print(imagenet[1])
    # flowers = FlowersDataset('../data/flowers-102')
    # print(flowers[2])

