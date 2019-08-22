from torch import nn

from dataset_loader import *


class FNET(nn.Module):
    def __init__(self):
        super(FNET, self).__init__()

        self.down_phase_one = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)))

        self.down_phase_two = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)))

        self.down_phase_three = nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)))

        self.down_phase_four = nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)))

        self.down_phase_five = nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)))

        self.up_phase_one = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.ReLU())

        self.up_phase_two = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU())

        self.up_phase_three = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU())

        self.up_phase_four = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU())

        self.up_phase_five = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU())

        self.final = torch.nn.Conv2d(in_channels=32, out_channels=21, kernel_size=1, stride=1)

    def upscale(self, data):
        data = self.up_phase_one(data)
        data = self.up_phase_two(data)
        data = self.up_phase_three(data)
        data = self.up_phase_four(data)
        data = self.up_phase_five(data)
        return data

    def downscale(self, data):
        data = self.down_phase_one(data)
        data = self.down_phase_two(data)
        data = self.down_phase_three(data)
        data = self.down_phase_four(data)
        data = self.down_phase_five(data)
        return data

    def forward(self, data_input):
        o = self.downscale(data_input)
        o = self.upscale(o)
        o = self.final(o)
        return o
