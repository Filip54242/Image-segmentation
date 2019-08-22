from torch import optim
from torch.utils.data import DataLoader

from fnetv2 import *

network = FNET()

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_set = Voc2012Dataset(root_dir="/home/student/Documents/VOC2012", transform=transform, set_type=DatasetType.TRAIN)

train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=4)

file = open("Fnet_imgs_trained", "w")
for element in train_set.data_set:
    file.write(str(element) + "\n")
file.close()

num_epochs = 300
network.cuda()

# optimizer = optim.Adam(network.parameters(), lr=1e-3)
optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    network.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        outputs = network(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    if epoch > 0 and epoch % 10 == 0:
        torch.save(network, "FNET")

    print('Results after epoch %d' % (epoch + 1))

    print("Training Loss: %.10f " % (train_loss / (batch_idx + 1)))
torch.save(network, "FNET")
