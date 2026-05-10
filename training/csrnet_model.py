import torch.nn as nn
from torchvision import models


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        conv2d = nn.Conv2d(
            in_channels,
            v,
            kernel_size=3,
            padding=d_rate,
            dilation=d_rate,
        )
        layers.extend(
            [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            if batch_norm else
            [conv2d, nn.ReLU(inplace=True)]
        )
        in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """Minimal CSRNet implementation compatible with current PyTorch."""

    def __init__(self, load_weights=False):
        super().__init__()
        self.seen = 0
        self.frontend_feat = [
            64, 64, "M", 128, 128, "M",
            256, 256, 256, "M", 512, 512, 512,
        ]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()
        if not load_weights:
            self._load_vgg_frontend_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _load_vgg_frontend_weights(self):
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except AttributeError:
            vgg = models.vgg16(pretrained=True)

        frontend_state = self.frontend.state_dict()
        vgg_state = vgg.features.state_dict()
        for key, value in frontend_state.items():
            if key in vgg_state:
                value.copy_(vgg_state[key])
