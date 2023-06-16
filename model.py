import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights, resnet50, ResNet50_Weights
import torch.nn.functional as F


def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class VGG(nn.Module):   # output relu4-1
    def __init__(self):
        super(VGG, self).__init__()
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.model = nn.Sequential(*model.features[:21])
        self._freeze_params()

    def forward(self, x):
        x = self.model(x)
        return x

    def _freeze_params(self):
        for p in self.model.parameters():
            p.requires_grad = False


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        model = resnet50()
        self.model = nn.Module()
        self.model.conv1 = model.conv1
        self.model.bn1 = model.bn1
        self.model.relu = model.relu
        self.model.maxpool = model.maxpool

        self.model.layer1 = model.layer1
        self.model.layer2 = model.layer2

        self.vgg = VGG()

        self._init_weights()

    def forward(self, x):
        sty = self.vgg(x)
        aes = self.model.conv1(x)
        aes = self.model.bn1(aes)
        aes = self.model.relu(aes)
        aes = self.model.maxpool(aes)

        aes = self.model.layer1(aes)
        aes = self.model.layer2(aes)

        output = adain(aes, sty)

        return output

    def _init_weights(self):
        self.model.load_state_dict(torch.load('checkpoint/ResNet_Pretrain/epoch_99.pth'),
                                    strict=False)


class GAB(nn.Module):
    def __init__(self):
        super(GAB, self).__init__()
        model = resnet50()
        self.model = nn.Module()
        self.model.conv1 = model.conv1
        self.model.bn1 = model.bn1
        self.model.relu = model.relu
        self.model.maxpool = model.maxpool

        self.model.layer1 = model.layer1
        self.model.layer2 = model.layer2
        self.model.layer3 = model.layer3

        self._init_weights()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        return x

    def _init_weights(self):
        self.model.load_state_dict(torch.load('checkpoint/ResNet_Pretrain/epoch_99.pth'),
                                    strict=False)


class SAAN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.GenAes = GAB()
        self.StyAes = SAB()

        self.NLB = NonLocalBlock(in_channels=1536)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(14, 14))

        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.BatchNorm2d(num_features=1536),
            nn.Flatten(),
            nn.Linear(1536 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(2048, num_classes),
            nn.Sigmoid(),
        )

        self._initial_weights()

    def forward(self, x):
        gen_aes = self.GenAes(x)
        sty_aes = self.StyAes(x)

        sty_aes = self.avg_pool(sty_aes)

        all_aes = torch.cat((sty_aes, gen_aes), 1)
        all_aes = self.NLB(all_aes)

        output = self.predictor(all_aes)

        return output

    def _initial_weights(self):
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)


class SAAN_AVA(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.GenAes = GAB()
        self.StyAes = SAB()

        self.NLB = NonLocalBlock(in_channels=1536)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(14, 14))

        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.BatchNorm2d(num_features=1536),
            nn.Flatten(),
            nn.Linear(1536*4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1),
        )

        self._initial_weights()

    def forward(self, x):
        gen_aes = self.GenAes(x)
        sty_aes = self.StyAes(x)

        sty_aes = self.avg_pool(sty_aes)

        all_aes = torch.cat((sty_aes, gen_aes), 1)
        all_aes = self.NLB(all_aes)

        output = self.predictor(all_aes)

        return output

    def _initial_weights(self):
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)