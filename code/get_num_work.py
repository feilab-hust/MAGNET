from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms

transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,
    download=True,
    transform=transform
)

print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(2, mp.cpu_count(), 2):
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=num_workers, batch_size=64,
                                               pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))