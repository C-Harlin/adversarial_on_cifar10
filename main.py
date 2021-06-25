import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mobilenetv2 import mobilenet_v2

norm = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                 std=[0.247, 0.243, 0.262])
transform = transforms.Compose([transforms.ToTensor()])

cifar10_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(cifar10_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for x,y in test_loader:
    x,y = x.to(device), y.to(device)
    break

def fgsm(model, x, y, epsilon):
    delta = torch.zeros_like(x, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(x + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd(model, x, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.data).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return delta.detach()


# 针对无穷范数的最速下降攻击
def sd_linf(model, x, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


# 计算l2范数
def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


# 针对l2范数的最速下降攻击
def sd_l2(model, x, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x + delta), y)
        loss.backward()
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()

    return delta.detach()

# DeepFool攻击
from tqdm import tqdm
from deepfool import deepfool
def DeepFool(model, x, y, num_classes=None, num_iter=None):
    delta = torch.zeros_like(x)
    for i in range(x.shape[0]):
        r = deepfool(x[i], y[i], model, device, num_classes=num_classes, max_iter=num_iter)
        delta[i] = torch.from_numpy(r[0])
    return delta

# C&W l2 attack
from cw import carlini_wagner_l2


def plot_images(x,y,yp,M,N):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N,M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow((255*x[i*N+j]).cpu().numpy().astype(np.uint8).transpose(1,2,0))
            title = ax[i][j].set_title(classes[yp[i*N+j].max(dim=0)[1]])
            plt.setp(title, color=('g' if classes[yp[i*N+j].max(dim=0)[1]] == classes[y[i*N+j]] else 'r'))
            # print(classes[yp[i*N+j].max(dim=0)[1]])
            ax[i][j].set_axis_off()
    plt.tight_layout()

## Quantitative Results
def epoch_standard(model, loader):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(norm(X))
        loss = nn.CrossEntropyLoss()(yp, y)

        total_err += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err, avg = 0., 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(norm(X + delta))
        loss = nn.CrossEntropyLoss()(yp, y)

        total_err += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        t = yp.max(dim=1)[1]!=y
        for i in range(len(t)):
            if t[i].item():
                avg += torch.mean(delta[i],dim=[0,1,2]).item()
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), avg / t.sum().item()
model = mobilenet_v2(pretrained=True,device=device).eval().to(device)

## Illustrate original predictions
yp = model(norm(x))
plot_images(x, y, yp, 3, 6)
## Illustrate FGSM attacked images
delta = fgsm(model, x, y, 0.01)
yp = model(norm(x + delta))
plot_images(x+delta, y, yp, 3, 6)
## Illustrate PGD attacked images
delta = pgd(model, x, y, 0.01, 1e4, 1000)
yp = model(norm(x + delta))
plot_images(x+delta, y, yp, 3, 6)
## Illustrate SD l_inf attacked images
delta = sd_linf(model, x, y, 0.01, 1e4, 1000)
yp = model(norm(x + delta))
plot_images(x+delta, y, yp, 3, 6)
## Illustrate SD l_2 attacked images
delta = sd_l2(model, x, y, 2, 1e4, 1000)
yp = model(norm(x + delta))
plot_images(x+delta, y, yp, 3, 6)
## Illustrate deepfool attacked images
delta = DeepFool(model, x, y, 10, 500)
yp = model(norm(x + delta))
plot_images(x+delta, y, yp, 3, 6)
## Illustrate c&w attacked images
# delta = carlini_wagner_l2(model,x,y) # cost too much time


print("Type", "Accurary","Mean of Distortion", sep="\t")
print("Unattacked", "{:.2%}".format(epoch_standard(model, test_loader)[0]), sep="\t")

res = epoch_adversarial(model, test_loader, fgsm, 0.01)
print("FGSM","{:.2%}".format(res[0]), "{:.2e}".format(res[2]), sep="\t")

res = epoch_adversarial(model, test_loader, pgd, 0.01, 1e4, 1000)
print("PGD","{:.2%}".format(res[0]), "{:.2e}".format(res[2]), sep="\t")

res = epoch_adversarial(model, test_loader, sd_linf, 0.01, 1e4, 1000)
print("SD_linf","{:.2%}".format(res[0]), "{:.2e}".format(res[2]), sep="\t")

res = epoch_adversarial(model, test_loader, sd_l2, 1, 1e4, 1000)
print("SD_l2","{:.2%}".format(res[0]), "{:.2e}".format(res[2]), sep="\t")

res = epoch_adversarial(model, test_loader, DeepFool, 10, 100)
print("DeepFool","{:.2%}".format(res[0]), "{:.2e}".format(res[2]), sep="\t")

res = epoch_adversarial(model, test_loader, carlini_wagner_l2)
print("C&W","{:.2%}".format(res[0]), "{:.2e}".format(res[2]), sep="\t")

# eval()
# Type	Acc	Mean	
# None	93.91%	0
# FGSM	71.27%	-3.29e-03
# PGD	74.12%	-5.76e-03
# SD_linf	73.88%	-5.19e-03
# SD_l2	57.83%	5.25e-02
# DeepF	7.07%	-2.47e-02
# C$W   86.28% 4.77e-03
