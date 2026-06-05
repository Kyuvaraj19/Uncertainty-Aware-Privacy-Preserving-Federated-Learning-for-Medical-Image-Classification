# client/dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(batch_size=32):

    print("\nLoading Dataset...\n")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # TRAIN DATASET
    trainset = datasets.ImageFolder(
        root="data/train",
        transform=transform
    )

    # TEST DATASET
    testset = datasets.ImageFolder(
        root="data/test",
        transform=transform
    )

    # DATALOADERS
    trainloader = DataLoader( # type: ignore
        trainset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )

    # PRINT DATASET INFO
    print("Train Images :", len(trainset))
    print("Test Images  :", len(testset))
    print("Classes      :", trainset.classes)

    print("\nDataset Loaded Successfully!\n")

    return trainloader, testloader