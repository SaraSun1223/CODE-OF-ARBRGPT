import torch



model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)