import torch
import torch.nn as nn
from model.layers import Conv2dWithConstraint, LazyLinearWithConstraint


class DeepConvNet(nn.Module):
    def __init__(
            self,
            num_channels: int,
            first_conv_length=10,
            block_out_channels=[25, 25, 50, 100, 200],
            pool_size=3,
    ):
        super(DeepConvNet, self).__init__()
        self.first_conv_block = nn.Sequential(
            Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, first_conv_length), max_norm=2),
            Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(num_channels, 1), bias=False,
                                 max_norm=2),
            nn.BatchNorm2d(block_out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d((1, pool_size))
        )
        self.deep_block = nn.ModuleList(
            [self.default_block(block_out_channels[i - 1], block_out_channels[i], first_conv_length, pool_size) for i in
             range(2, 5)]
        )
        self.flatten = nn.Flatten()

    def default_block(self, in_channels, out_channels, T, P):
        default_block = nn.Sequential(
            nn.Dropout(0.5),
            Conv2dWithConstraint(in_channels, out_channels, (1, T), bias=False, max_norm=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d((1, P))
        )

        return default_block

    def forward(self, x):
        out = self.first_conv_block(x)
        for block in self.deep_block:
            out = block(out)
        X_output = self.flatten(out)
        return out, X_output


# class classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(classifier, self).__init__()
#
#         self.dense = LazyLinearWithConstraint(num_classes, max_norm=0.5)
#
#     def forward(self, x):
#         x = self.dense(x)
#         return x
class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(200, num_classes, (1, 4)),
            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = self.dense(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x


class Net(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_channels: int,
                 sampling_rate: int):
        super(Net, self).__init__()

        self.backbone = DeepConvNet(num_channels=num_channels)

        self.classifier = classifier(num_classes)

    def forward(self, x):
        x, X_output = self.backbone(x)
        x = self.classifier(x)
        return x


def get_model(args):
    model = Net(num_classes=args.num_classes,
                num_channels=args.num_channels,
                sampling_rate=args.sampling_rate)

    return model

class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))