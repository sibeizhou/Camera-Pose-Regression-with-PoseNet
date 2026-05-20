import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(f"inception_{key}/1x1", nn.Conv2d(in_channels, n1x1, kernel_size=1), weights),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(f"inception_{key}/3x3_reduce", nn.Conv2d(in_channels, n3x3red, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            init(f"inception_{key}/3x3", nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(f"inception_{key}/5x5_reduce", nn.Conv2d(in_channels, n5x5red, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            init(f"inception_{key}/5x5", nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2), weights),
            nn.ReLU(inplace=True)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            init(f"inception_{key}/pool_proj", nn.Conv2d(in_channels, pool_planes, kernel_size=1), weights),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate

        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class LossHeader(nn.Module):
    def __init__(self, key=None, weights=None):
        super(LossHeader, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(1024, 2048)
        self.dropout = nn.Dropout(0.4)
        self.fc_xyz = nn.Linear(2048, 3)
        self.fc_wpqr = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        return xyz, wpqr


class LossHeader2(nn.Module):
    def __init__(self, in_channels):
        super(LossHeader2, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((4,4))
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)  # 70% dropout
        self.fc_xyz = nn.Linear(1024, 3)
        self.fc_wpqr = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        x1 = self.fc_xyz(x)  # for xyz position
        x2 = self.fc_wpqr(x)  # for wpqr orientation

        return x1, x2

class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()

            # # Print all keys in the weights dictionary
            # print("keys in weights:")
            # print(weights.keys())

        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights)

        self.loss_header = LossHeader()
        self.loss_header_loss1 = LossHeader2(512)
        self.loss_header_loss2 = LossHeader2(528)

        print("PoseNet model created!")


    def forward(self, x):
        # TODO: Implement PoseNet forward

        x = self.pre_layers(x)
        x = self._3a(x)
        x = self._3b(x)
        x = self.max_pool1(x)
        x = self._4a(x)

        if self.training:
            loss1_xyz, loss1_wpqr = self.loss_header_loss1(x)

        x = self._4b(x)
        x = self._4c(x)
        x = self._4d(x)

        if self.training:
            loss2_xyz, loss2_wpqr = self.loss_header_loss2(x)

        x = self._4e(x)
        x = self.max_pool2(x)
        x = self._5a(x)
        x = self._5b(x)

        loss3_xyz, loss3_wpqr = self.loss_header(x)

        if self.training:
            return loss1_xyz, loss1_wpqr, loss2_xyz, loss2_wpqr, loss3_xyz, loss3_wpqr
        else:
            return loss3_xyz, loss3_wpqr


class PoseLoss(nn.Module):
    def __init__(self, w1, w2, w3, beta1, beta2, beta3):
        super(PoseLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        gt_xyz = poseGT[:, :3]  # First 3 entries: Ground truth xyz
        gt_wpqr = poseGT[:, 3:]  # Last 4 entries: Ground truth wpqr

        # Position loss
        loss_xyz_1 = F.mse_loss(p1_xyz, gt_xyz)
        loss_xyz_2 = F.mse_loss(p2_xyz, gt_xyz)
        loss_xyz_3 = F.mse_loss(p3_xyz, gt_xyz)

        # Orientation loss (quaternion)
        loss_wpqr_1 = F.mse_loss(p1_wpqr, gt_wpqr)
        loss_wpqr_2 = F.mse_loss(p2_wpqr, gt_wpqr)
        loss_wpqr_3 = F.mse_loss(p3_wpqr, gt_wpqr)

        # Combine losses for each header
        loss = self.w1 * (loss_xyz_1 + self.beta1 * loss_wpqr_1) + \
               self.w2 * (loss_xyz_2 + self.beta2 * loss_wpqr_2) + \
               self.w3 * (loss_xyz_3 + self.beta3 * loss_wpqr_3)

        return loss

