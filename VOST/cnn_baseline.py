import torch.nn as nn
import torch
from torchvision import transforms, models

class CNN_Classifier(torch.nn.Module):
    def __init__(self, num_classes = 30, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = nn.Sequential( #MLP head, more parameters, higher accuracy
            nn.Linear(2048, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x):
        return self.classifier(x)


class CNN_With_Backbone(torch.nn.Module):
    def __init__(self, num_classes = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        in_features = self.backbone.fc.in_features # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        ) 
        # self.backbone.fc = nn.Sequential(
        #     nn.Linear(in_features, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x):
        return self.backbone(x)


class FrameClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FrameClassifier, self).__init__()
        # Load ResNet50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        #self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = models.resnet50(weights=weights)
        print("backbone created")
        # Remove original classifier
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # output: (B, 2048, 1, 1)
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-4])

        # 0 torch.Size([1, 64, 112, 112])
        # 1 torch.Size([1, 64, 112, 112])
        # 2 torch.Size([1, 64, 112, 112])
        # 3 torch.Size([1, 64, 56, 56]) -6
        # 4 torch.Size([1, 256, 56, 56]) -5
        # 5 torch.Size([1, 512, 28, 28]) -4
        # 6 torch.Size([1, 1024, 14, 14]) -3
        # 7 torch.Size([1, 2048, 7, 7]) -2????????

        self.fc = nn.Linear(512, num_classes) # was 2048 for -1
        
    def forward(self, x):
        # Resize input
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Forward through truncated backbone
        feature_maps = self.backbone(x)
        # print("feature maps dim", feature_maps.shape)  # e.g., (B, 512, 28, 28)
        # print("x shape", x.shape)

        # Global pooling for classification
        pooled = torch.nn.functional.adaptive_avg_pool2d(feature_maps, (1, 1))
        #print("pooled", pooled.shape)
        flattened = pooled.view(pooled.size(0), -1)  # (B, 512)
        #print("flattened shape", flattened.shape)

        # Linear classifier
        out = self.fc(flattened)

        return out, feature_maps
    