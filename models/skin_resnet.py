import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=7, pretrained=True):
    if pretrained:
        # Compatibility with older/newer torchvision versions
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18(pretrained=False)
        
    # Replace the classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
