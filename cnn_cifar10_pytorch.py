import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as torchprune
from pytorch_model_summary import summary
import numpy as np
import time

torch.manual_seed(0)
np.random.seed(0)

def shuffle_tensor(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='~/Documents/datasets/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='~/Documents/datasets/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.fc = nn.Linear(64, 10)

        self.initial_state_dict = {}
        for k, v in self.state_dict().items():
            cloned = v.clone()
            self.initial_state_dict[k] = cloned
            self.initial_state_dict[k + "_orig"] = cloned

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x

    def reset_weights_to_init(self):
        self.load_state_dict(self.initial_state_dict, strict=False)

    def reset_weights_to_random(self):
        new_state_dict = {}
        for k, v in self.initial_state_dict.items():
            new_state_dict[k] = shuffle_tensor(v.clone())
        self.load_state_dict(new_state_dict, strict=False)

model = Model()
print(summary(model, torch.zeros(1, 3, 32, 32)))

criterion = nn.CrossEntropyLoss()
val_criterion = nn.CrossEntropyLoss(reduction="sum")

def train(epochs, callback=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-3)

    running_average_alpha = 0.9

    for epoch in range(epochs):

        running_loss = 0
        t = time.time()

        for i, data in enumerate(trainloader, 0):

            batch, labels = data
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i == 0:
                running_loss = loss.item()
            else:
                running_loss = loss.item() * (1-running_average_alpha) + running_average_alpha * running_loss

            if (i+1) % 12 == 0:
                print('Epoch', epoch + 1, "Batch", i+1, "Running loss:", running_loss)
                if callback is not None:
                    callback()

        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data in testloader:
                batch, labels = data
                outputs = model(batch)
                val_loss += val_criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = float(correct.item()) / total
            val_loss = val_loss.item() / total
            print('Epoch', epoch+1, 'Accuracy:', accuracy, 'Time:', time.time() - t, 'Running train loss', running_loss, 'Val loss', val_loss)

def prune():
    total_weights = 0.
    remaining_weights = 0.
    for module in model.children():
        if hasattr(module, 'weight'):
            # torchprune.ln_structured(module, 'weight', amount=1, n=1, dim=1)
            torchprune.l1_unstructured(module, 'weight', amount=0.0067)
            total_weights += module.weight_mask.numel()
            remaining_weights += module.weight_mask.sum().numpy()
    print("Density is now", remaining_weights/total_weights)

train(2, callback=prune)

model.reset_weights_to_init()
print("\nWeights reset to init")
train(2)

model.reset_weights_to_random()
print("\nWeights reset to random")
train(2)
