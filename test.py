import cv2
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.aijack.attack import GradientInversion_Attack
from src.aijack.utils import NumpyDataset


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.BatchNorm2d(12),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def prepare_dataloader(path="MNIST/.", batch_size = 64, shuffle=True):
    at_t_dataset_train = torchvision.datasets.MNIST(
        root=path, train=True, download=True
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = NumpyDataset(
        at_t_dataset_train.train_data.numpy(),
        at_t_dataset_train.train_labels.numpy(),
        transform=transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return dataloader


torch.manual_seed(1)

shape_img = (28, 28)
num_classes = 10
channel = 1
hidden = 588

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
dataloader = prepare_dataloader()
for data in dataloader:
    x, y = data[0], data[1]
    break


plt.figure(figsize=(2, 4))
plt.imshow(x[:1].detach().numpy()[0][0], cmap="gray")
plt.axis("off")
plt.title("target")
plt.savefig('draw_result/fig_test1.png', dpi=600, format='png')
plt.savefig('draw_result/fig_test1.pdf', dpi=600, format='pdf')

criterion = nn.CrossEntropyLoss()
net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
pred = net(x[:1])
loss = criterion(pred, y[:1])
received_gradients = torch.autograd.grad(loss, net.parameters())
received_gradients = [cg.detach() for cg in received_gradients]

gs_attacker = GradientInversion_Attack(net, (1, 28, 28), lr=1.0, log_interval=0,
                                    num_iteration=3000,
                                    tv_reg_coef=0.01,
                                    distancename="cossim")

num_seeds=1
fig = plt.figure()
for s in tqdm(range(num_seeds)):
    gs_attacker.reset_seed(s)
    result = gs_attacker.attack(received_gradients)
    ax1 = fig.add_subplot(2, num_seeds, s+1)
    ax1.axis("off")
    ax1.imshow(result[0].detach().numpy()[0][0], cmap="gray")
    ax1.set_title(torch.argmax(result[1]).item())
    ax2 = fig.add_subplot(2, num_seeds, num_seeds+s+1)
    ax2.imshow(cv2.medianBlur(result[0].detach().numpy()[0][0], 5), cmap="gray")
    ax2.axis("off")
plt.suptitle("Result of GS")
plt.tight_layout()
plt.savefig('draw_result/fig_test2.png', dpi=600, format='png')
plt.savefig('draw_result/fig_test2.pdf', dpi=600, format='pdf')
