import torch
import torchvision.models as models

class FaceRecognitionModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(FaceRecognitionModel, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        # Remove the last fully connected layer
        self.model.fc = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)

def load_model(device):
    model = FaceRecognitionModel(pretrained=True)
    model = model.to(device)
    model.eval()
    return model
