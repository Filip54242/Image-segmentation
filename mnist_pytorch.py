




import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn as nn

from torch import optim





########################################################################################################################





transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])




transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])


##########################################################################################################################


train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True,
                                       transform=transform_train)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024,
                                           shuffle=True, num_workers=4)

test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True,
                                      transform=transform_test)

validation_loader = torch.utils.data.DataLoader(test_set, batch_size=1024,
                                                shuffle=False,
                                                num_workers=4)


##########################################################################################################################




net = nn.Sequential(nn.Linear(784, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                    ).cpu()





##########################################################################################################################







criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)






##########################################################################################################################



num_epochs = 10

for epoch in range(num_epochs):
    net.train()
    train_loss = 0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):            
            
        inputs, targets = inputs, targets
        inputs = inputs.view(-1, 784)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Results after epoch %d' % (epoch + 1))

    print('Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)' 
          % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))



##########################################################################################################################



net.eval()

valid_loss = 0
correct = 0
total = 0

for batch_idx, (inputs, targets) in enumerate(validation_loader):

    inputs, targets = inputs, targets
    inputs = inputs.view(-1, 784)

    outputs = net(inputs)

    loss = criterion(outputs, targets)

    valid_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

print('Validation Loss: %.3f | Validation Acc: %.3f%% (%d/%d)' 
      % (valid_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))





