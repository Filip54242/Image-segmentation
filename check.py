import matplotlib.pylab as plt

from fnetv2 import *

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

net = torch.load("FNET").cpu()
net.eval()

columns = 3
rows = 1

name_list = open("Fnet_imgs_trained", 'r').read().split("\n")
name_list.pop()
for name in name_list:
    fig = plt.figure(figsize=(224, 224))
    img_name = "/home/student/Documents/VOC2012/JPEGImages/" + name + ".jpg"
    image = Image.open(img_name)
    img_name = "/home/student/Documents/VOC2012/SegmentationClass/" + name + ".png"
    image2=Image.open(img_name)

    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)

    fig.add_subplot(rows, columns, 3)
    plt.imshow(image2)

    image = transform(image)

    output = net(image.unsqueeze(0))

    fig.add_subplot(rows, columns, 2)
    predicted = torch.argmax(output.data, 1)
    plt.imshow(predicted[0].data)

    plt.show()
