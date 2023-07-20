import torch
import torch.nn as nn
from model.layers import Conv2dWithConstraint, LinearWithConstraint
from torch.nn import functional as F
class FBCNet(nn.Module):
    def __init__(
            self,
            num_channels: int,
            m=32,
            n_band=9,
            temporal_stride=4,
    ):
        super(FBCNet, self).__init__()
        self.temporal_stride = temporal_stride


        # SCB (Spatial Convolution Block)
        self.scb = nn.Sequential(
            Conv2dWithConstraint(n_band, m * n_band, (num_channels, 1), groups=n_band, max_norm=2),
            nn.BatchNorm2d(m * n_band),
            Swish()
        )

        # Temporal Layer
        self.temporal_layer = LogVarLayer(-1)


    def forward(self, x):
        x = torch.squeeze(x)
        x = self.scb(x)
        x = F.pad(x, (0, 1))
        x = x.reshape([*x.shape[:2], self.temporal_stride, int(x.shape[-1] / self.temporal_stride)])
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            LinearWithConstraint(32*9*4, num_classes, max_norm=0.5),
            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = self.dense(x)
        return x


class Net(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_channels: int,
                 sampling_rate: int):
        super(Net, self).__init__()

        self.backbone = FBCNet(num_channels=num_channels)

        self.classifier = classifier(num_classes)

    def forward(self, x):
        output = self.backbone(x)
        x = self.classifier(output)
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

class Swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''

    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))
