import torch
import torch.nn as nn
from model.layers import Conv2dWithConstraint, LazyLinearWithConstraint


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            num_channels: int,
            sampling_rate: int,
            F1=40,
            T1=25,
            F2=40,
            P1_T=75,
            P1_S=15,
            drop_out=0.25,
            pool_mode= 'mean',
    ):
        super(ShallowConvNet, self).__init__()

        kernel_size = int(sampling_rate * 0.12)
        pooling_size = 0.3
        hop_size = 0.7
        pooling_kernel_size = int(sampling_rate * pooling_size)
        pooling_stride_size = int(sampling_rate * pooling_size * (1 - hop_size))


        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, kernel_size), padding='same', max_norm=2.),
            Conv2dWithConstraint(F1, F2, (num_channels, 1), padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, pooling_kernel_size), (1, pooling_stride_size)),
            ActLog(),
            nn.Dropout(drop_out),
            # nn.Flatten(),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.net(x)
        output = self.flatten(x)
        return x, output

class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(40, num_classes, (1, 31)),
            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = self.dense(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x

# class classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(classifier, self).__init__()
#
#         self.dense = LazyLinearWithConstraint(num_classes, max_norm=0.5)
#
#     def forward(self, x):
#         x = self.dense(x)
#         return x


class Net(nn.Module):
    def __init__(self,
                 num_classes: 2,
                 num_channels: int,
                 sampling_rate: int):
        super(Net, self).__init__()

        self.backbone = ShallowConvNet(num_channels=num_channels, sampling_rate=sampling_rate)

        self.classifier = classifier(num_classes)

    def forward(self, x):
        x, output = self.backbone(x)
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