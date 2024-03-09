from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size, num_workers=2):
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dl, test_dl
