from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch is optional for non-learning utilities
    torch = None
    nn = None


if nn is not None:

    class TinyOccupancyCNN(nn.Module):
        """A small CPU-friendly CNN for STFT occupancy heatmaps."""

        def __init__(self, channels: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, kernel_size=1),
            )

        def forward(self, x):
            return self.net(x)


    class TinyRadioMLCNN(nn.Module):
        """Small 1-D CNN for RadioML I/Q modulation classification."""

        def __init__(self, n_classes: int, channels: int = 32, feature_dim: int = 64):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(2, channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels * 2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.embedding = nn.Linear(channels * 2, feature_dim)
            self.classifier = nn.Linear(feature_dim, n_classes)

        def forward(self, x, return_feature: bool = False):
            h = self.features(x).squeeze(-1)
            z = self.embedding(h)
            logits = self.classifier(torch.relu(z))
            if return_feature:
                return logits, z
            return logits


    class TinyRadDetCNN(nn.Module):
        """Small detector for 128x128 RadDet spectrogram images.

        It predicts at most one object per image: objectness, class logits and
        normalized YOLO-style box coordinates. This is deliberately simpler
        than YOLO; it is a first semantic-extraction baseline.
        """

        def __init__(self, n_classes: int = 11, channels: int = 24, feature_dim: int = 96):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.embedding = nn.Linear(channels * 4, feature_dim)
            self.objectness = nn.Linear(feature_dim, 1)
            self.classifier = nn.Linear(feature_dim, n_classes)
            self.box = nn.Linear(feature_dim, 4)

        def forward(self, x, return_feature: bool = False):
            h = self.features(x).flatten(1)
            z = torch.relu(self.embedding(h))
            obj_logit = self.objectness(z).squeeze(1)
            class_logits = self.classifier(z)
            box = torch.sigmoid(self.box(z))
            if return_feature:
                return obj_logit, class_logits, box, z
            return obj_logit, class_logits, box


    class TinyOccupancyClassCNN(nn.Module):
        """Small multi-task model for RadDet occupancy mask plus frame class.

        The mask head preserves spatial detail for localization, while the
        class head uses pooled encoder features. It is intended for the first
        box/class semantic baseline on mostly single-object RadDet frames.
        """

        def __init__(self, n_classes: int = 11, channels: int = 16, feature_dim: int = 64):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.down = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
            )
            self.mask_head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, kernel_size=1),
            )
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels * 2, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, n_classes),
            )

        def forward(self, x):
            h1 = self.enc1(x)
            h2 = self.enc2(self.down(h1))
            mask_logits = self.mask_head(h2)
            class_logits = self.class_head(h2)
            return mask_logits, class_logits


    class TinyGridDetectorCNN(nn.Module):
        """Tiny YOLO-style one-anchor grid detector for 128x128 RadDet images."""

        def __init__(self, n_classes: int = 11, channels: int = 24):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 16
                nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 8
                nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(channels * 4, 1 + 4 + n_classes, kernel_size=1)

        def forward(self, x):
            y = self.head(self.backbone(x))
            obj_logit = y[:, 0:1]
            box = torch.sigmoid(y[:, 1:5])
            class_logits = y[:, 5:]
            return obj_logit, box, class_logits


    class TinyGridDetector16CNN(nn.Module):
        """Tiny YOLO-style one-anchor detector with a 16x16 output grid.

        Compared with TinyGridDetectorCNN, this keeps one extra level of
        spatial detail. It is intended for small/narrow time-frequency objects
        where an 8x8 grid can be too coarse.
        """

        def __init__(self, n_classes: int = 11, channels: int = 20):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 16
                nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(channels * 4, 1 + 4 + n_classes, kernel_size=1)

        def forward(self, x):
            y = self.head(self.backbone(x))
            obj_logit = y[:, 0:1]
            box = torch.sigmoid(y[:, 1:5])
            class_logits = y[:, 5:]
            return obj_logit, box, class_logits

else:
    TinyOccupancyCNN = None
    TinyRadioMLCNN = None
    TinyRadDetCNN = None
    TinyOccupancyClassCNN = None
    TinyGridDetectorCNN = None
    TinyGridDetector16CNN = None
