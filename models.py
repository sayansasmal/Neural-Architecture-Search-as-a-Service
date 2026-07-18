# models.py
import torch
import torch.nn as nn
from torchvision import models as tv_models


class TinyCNN(nn.Module):
    """
    Very small CNN for low-spec devices.
    ~100k parameters (depends slightly on num_classes).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SmallCNN(nn.Module):
    """
    Slightly larger CNN than TinyCNN, but still light.
    ~300k-500k parameters depending on classes.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/16
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PlainBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool(2). Simple, fast, few parameters."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.pool(x)


class ResidualBlock(nn.Module):
    """
    Two 3x3 convs with a skip connection (output = F(x) + shortcut(x)),
    followed by MaxPool(2). A 1x1 projection is used on the shortcut
    whenever the channel count changes.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return self.pool(out)


class BottleneckBlock(nn.Module):
    """
    1x1 squeeze -> 3x3 -> 1x1 expand, followed by MaxPool(2).
    Parameter-efficient, same idea as MobileNet/EfficientNet blocks.
    Squeeze channels = out_ch // 4 (minimum of 4).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = max(4, out_ch // 4)
        self.squeeze = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.expand = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.act(self.bn1(self.squeeze(x)))
        x = self.act(self.bn2(self.conv(x)))
        x = self.act(self.bn3(self.expand(x)))
        return self.pool(x)


_BLOCK_TYPES = {
    "plain": PlainBlock,
    "residual": ResidualBlock,
    "bottleneck": BottleneckBlock,
}


class ConfigurableCNN(nn.Module):
    """
    A CNN assembled dynamically from a NAS config dict:
        {"depth": int, "base_channels": int, "block_type": str, "dropout": float}

    Channel width doubles at each successive block, starting from
    base_channels (e.g. base_channels=32, depth=3 -> 32 -> 64 -> 128).
    """
    def __init__(self, config: dict, num_classes: int):
        super().__init__()
        depth = config["depth"]
        base_channels = config["base_channels"]
        block_type = config["block_type"]
        dropout = config["dropout"]

        if block_type not in _BLOCK_TYPES:
            raise ValueError(f"Unknown block_type: {block_type}")
        BlockCls = _BLOCK_TYPES[block_type]

        blocks = []
        in_ch = 3
        out_ch = base_channels
        for _ in range(depth):
            blocks.append(BlockCls(in_ch, out_ch))
            in_ch = out_ch
            out_ch = out_ch * 2

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model_from_config(config: dict, num_classes: int) -> nn.Module:
    """
    Factory for the "Configurable Space" NAS mode: builds a ConfigurableCNN
    from a {"depth","base_channels","block_type","dropout"} dict.
    """
    return ConfigurableCNN(config, num_classes)


def _make_resnet18(num_classes: int, pretrained: bool):
    # New torchvision uses "weights" instead of pretrained=True
    if pretrained:
        weights = tv_models.ResNet18_Weights.DEFAULT
    else:
        weights = None
    net = tv_models.resnet18(weights=weights)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net


def _make_mobilenet_v2(num_classes: int, pretrained: bool):
    if pretrained:
        weights = tv_models.MobileNet_V2_Weights.DEFAULT
    else:
        weights = None
    net = tv_models.mobilenet_v2(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def _make_efficientnet_b0(num_classes: int, pretrained: bool):
    if pretrained:
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None
    net = tv_models.efficientnet_b0(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def get_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Factory function to get a model by name.

    Supported names:
      - "tiny_cnn"      (custom small CNN)
      - "small_cnn"     (custom slightly larger CNN)
      - "resnet18"
      - "mobilenet_v2"
      - "efficientnet_b0"
    """
    name = name.lower()

    if name == "tiny_cnn":
        return TinyCNN(num_classes)
    elif name == "small_cnn":
        return SmallCNN(num_classes)
    elif name == "resnet18":
        return _make_resnet18(num_classes, pretrained=pretrained)
    elif name == "mobilenet_v2":
        return _make_mobilenet_v2(num_classes, pretrained=pretrained)
    elif name == "efficientnet_b0":
        return _make_efficientnet_b0(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {name}")