import torch
import torch.nn as nn
import torch.nn.functional as F

class _G(nn.Module):

    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.args.z_dim, 512, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.args.z_dim, 1, 1, 1)
        #print(x.size())  # torch.Size([n, 200, 1, 1, 1])
        x = self.layer1(x)
        #print(x.size())  # torch.Size([n, 512, 4, 4, 4])
        x = self.layer2(x)
        #print(x.size())  # torch.Size([n, 256, 8, 8, 8])
        x = self.layer3(x)
        #print(x.size())  # torch.Size([n, 128, 16, 16, 16])
        x = self.layer4(x)
        #print(x.size())  # torch.Size([n, 64, 32, 32, 32])
        x = self.layer5(x)
        #print(x.size())  # torch.Size([n, 1, 64, 64, 64])

        return x


class _D(nn.Module):

    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),

        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),

        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        x = x.view(-1, 1, 64, 64, 64)
        # print(x.size())  # torch.Size([ n, 1, 64, 64, 64])
        x = self.layer1(x)
        # print(x.size())  # torch.Size([ 64, 32, 32, 32])
        x = self.layer2(x)
        # print(x.size())  # torch.Size([ 128, 16, 16, 16])
        x = self.layer3(x)
        # print(x.size())  # torch.Size([ 256, 8, 8, 8])
        x = self.layer4(x)
        # print(x.size())  # torch.Size([ 512, 4, 4, 4])
        x = self.layer5(x)
        # print(x.size())  # torch.Size([ 1, 1, 1, 1])
        x_after_sigmoid = F.sigmoid(x)

        return x_after_sigmoid, x

class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _, _ = x.size()
        return x.view(n, -1)


# custom weights initialization called on netG and netD
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        # nn.init.xavier_uniform(m.weight.data)
        nn.init.normal(m.weight.data, 0, 0.02)