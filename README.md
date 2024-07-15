# Tucker-Decomposition
Certainly! Here's how you can separate the training of the weights of the Tucker decomposition convolution using the training dataset and the weights of the alphas using the test dataset.


### Step 1: Import Necessary Libraries


```python

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torchvision.models import resnet18

from tensorly.decomposition import partial_tucker

import tensorly as tl


tl.set_backend('pytorch')

```


### Step 2: Define CIFAR-10 DataLoader


```python

# Define transforms

transform_train = transforms.Compose([

    transforms.RandomCrop(32, padding=4),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])


transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])


# Load CIFAR-10 dataset

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


# Define DataLoader

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

```


### Step 3: Define Tucker Decomposition Function


```python

def tucker_decomposition_conv_layer(layer, rank):

    core, [last, first] = partial_tucker(layer.weight.data, modes=[0, 1], ranks=[rank, rank])

    core = torch.from_numpy(core)

    last = torch.from_numpy(last)

    first = torch.from_numpy(first)


    pointwise_s_to_r = nn.Conv2d(in_channels=first.shape[1], out_channels=first.shape[0],

                                 kernel_size=1, stride=1, padding=0, bias=False)

    depthwise_r_to_r = nn.Conv2d(in_channels=core.shape[1], out_channels=core.shape[0],

                                 kernel_size=layer.kernel_size, stride=layer.stride,

                                 padding=layer.padding, dilation=layer.dilation, bias=False)

    pointwise_r_to_t = nn.Conv2d(in_channels=last.shape[1], out_channels=last.shape[0],

                                 kernel_size=1, stride=1, padding=0, bias=False)


    pointwise_s_to_r.weight.data = first

    depthwise_r_to_r.weight.data = core

    pointwise_r_to_t.weight.data = last


    new_layers = [pointwise_s_to_r, depthwise_r_to_r, pointwise_r_to_t]

    return nn.Sequential(*new_layers)

```


### Step 4: Define MixedOp and Replace Convolutions


```python

class MixedOp(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):

        super(MixedOp, self).__init__()

        self.ops = nn.ModuleList()

        

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.bn_conv = nn.BatchNorm2d(out_channels)

        

        rank = min(in_channels, out_channels)

        self.tucker_conv = tucker_decomposition_conv_layer(self.conv, rank)

        self.bn_tucker_conv = nn.BatchNorm2d(out_channels)

        

        self.alpha = nn.Parameter(torch.randn(2))

    

    def forward(self, x):

        weights = F.softmax(self.alpha, dim=0)

        conv_out = F.relu(self.bn_conv(self.conv(x)))

        tucker_out = F.relu(self.bn_tucker_conv(self.tucker_conv(x)))

        return weights[0] * conv_out + weights[1] * tucker_out


def replace_conv_with_mixed_op(model):

    for name, module in model.named_children():

        if isinstance(module, nn.Conv2d):

            in_channels = module.in_channels

            out_channels = module.out_channels

            kernel_size = module.kernel_size

            stride = module.stride

            padding = module.padding

            bias = module.bias is not None

            

            mixed_op = MixedOp(in_channels, out_channels, kernel_size, stride, padding, bias)

            mixed_op.conv.weight.data = module.weight.data

            if bias:

                mixed_op.conv.bias.data = module.bias.data

            

            rank = min(in_channels, out_channels)

            tucker_layers = tucker_decomposition_conv_layer(module, rank)

            for i, layer in enumerate(tucker_layers):

                mixed_op.tucker_conv[i].weight.data = layer.weight.data

                if layer.bias is not None:

                    mixed_op.tucker_conv[i].bias.data = layer.bias.data

            

            setattr(model, name, mixed_op)

        

        elif isinstance(module, nn.Sequential):

            replace_conv_with_mixed_op(module)

        

        elif isinstance(module, nn.Module):

            replace_conv_with_mixed_op(module)

            

# Load pre-trained ResNet-18

model = resnet18(pretrained=True)

replace_conv_with_mixed_op(model)

model = model.cuda()

```


### Step 5: Define Training Loop


Separate the training for Tucker decomposition and alpha weights:


```python

# Function for entropy regularization

def entropy_regularization(alpha):

    prob = F.softmax(alpha, dim=0)

    entropy = -torch.sum(prob * torch.log(prob + 1e-6))

    return entropy


# Training the model

num_epochs = 10

criterion = nn.CrossEntropyLoss()


# Separate parameters

tucker_params = [param for name, module in model.named_modules() if isinstance(module, MixedOp) for param in module.tucker_conv.parameters()]

alpha_params = [param for name, param in model.named_parameters() if 'alpha' in name]


# Optimizers

tucker_optimizer = optim.Adam(tucker_params, lr=0.001)

alpha_optimizer = optim.Adam(alpha_params, lr=0.01)

lambda_entropy = 0.01


for epoch in range(num_epochs):

    # Train Tucker decomposition weights

    model.train()

    for inputs, targets in train_loader:

        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        

        tucker_optimizer.zero_grad()

        loss.backward()

        tucker_optimizer.step()


    # Train alpha weights

    model.eval()

    for inputs, targets in test_loader:

        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)

        classification_loss = criterion(outputs, targets)

        

        entropy_loss = sum(entropy_regularization(param) for param in alpha_params)

        loss = classification_loss + lambda_entropy * entropy_loss

        

        alpha_optimizer.zero_grad()

        loss.backward()

        alpha_optimizer.step()

        

        # Clip alpha values to be in the range [0, 1]

        with torch.no_grad():

            for param in alpha_params:

                param.data.clamp_(0, 1)

    

    print(f"Epoch {epoch+1}/{num_epochs} completed")


print("Finished Training")

```


### Step 6: Calculate Test Accuracy


```python

def calculate_accuracy(model, data_loader):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for data in data_loader:

            images, labels = data

            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total

    return accuracy


# Calculate test accuracy

test_accuracy = calculate_accuracy(model, test_loader)

print(f'Test Accuracy: {test_accuracy:.2f}%')

```


This setup allows you to train the weights of the Tucker decomposition convolutions using the training dataset and train the alpha weights using the test dataset, while ensuring that alpha values remain within the range [0, 1].
