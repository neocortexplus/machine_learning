from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataLoaderFactory:
    def __init__(self, dataset_name, batch_size, num_workers=2):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = self._get_transform()

    def _get_transform(self):
        # This method can be expanded to customize transforms for different datasets
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Placeholder values, adjust per dataset
        ])
    
    def _load_dataset(self):
        if self.dataset_name == 'CIFAR10':
            train_ds = datasets.CIFAR10("data", train=True, download=True, transform=self.transform)
            test_ds = datasets.CIFAR10("data", train=False, download=True, transform=self.transform)
        elif self.dataset_name == 'MNIST':
            train_ds = datasets.MNIST("data", train=True, download=True, transform=self.transform)
            test_ds = datasets.MNIST("data", train=False, download=True, transform=self.transform)
        elif self.dataset_name == 'FashionMNIST':
            train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=self.transform)
            test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return train_ds, test_ds

    def get_dataloaders(self):
        train_ds, test_ds = self._load_dataset()
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dl, test_dl