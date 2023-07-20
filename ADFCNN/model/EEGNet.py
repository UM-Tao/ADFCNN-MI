import torch
import torch.nn as nn
from model.layers import Conv2dWithConstraint, LazyLinearWithConstraint


class EEGNet(nn.Module):
    def __init__(self,
                num_channels: int,
                F1=8, D=2, F2= 'auto', T1= 125, T2=30, P1=4, P2=8, pool_mode= 'mean',
                drop_out=0.25):
        super(EEGNet, self).__init__()
    
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1),  padding=(0, T1//2), bias=False),
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (num_channels, 1), padding=0, groups=F1, bias=False, max_norm=1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1), stride=4),
            nn.Dropout(drop_out)
        )
        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2),  padding=(0, T2//2), groups=F1 * D),
            nn.Conv2d(F2, F2, 1,  stride=1, bias=False, padding=0),
            nn.BatchNorm2d(F2),
            # ActSquare(),
            nn.ELU(),
            pooling_layer((1, P2), stride=8),
            # ActLog(),
            nn.Dropout(drop_out)
        )
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm2d(F2)
    def forward(self, x):
        x = self.spectral(x)
        x = self.spatial(x)
        x = self.temporal(x)
        output = self.flatten(x)

        return x, output
    

class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(16, num_classes, (1, 23)),
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
#         self.dense = LazyLinearWithConstraint(num_classes, max_norm=0.25)
#
#     def forward(self, x):
#         x = self.dense(x)
#         return x

class Net(nn.Module):   
    def __init__(self,
                num_classes: int,
                num_channels: int,
                sampling_rate: int):
        super(Net, self).__init__()

        self.backbone = EEGNet(num_channels=num_channels)

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