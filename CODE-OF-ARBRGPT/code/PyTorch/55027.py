import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {}".format(device.type))

    img = [torch.rand((3, 442, 500), dtype=torch.float32, device=device)]

    target = [{'boxes': torch.as_tensor([[122., 1., 230., 176.],
                                         [336., 1., 443., 150.]], dtype=torch.float32, device=device),
                'labels': torch.as_tensor([5, 5], dtype=torch.int64, device=device),
                'image_id': torch.as_tensor([1], dtype=torch.int64, device=device),
                'area': torch.as_tensor([18900., 15943.], dtype=torch.float32, device=device),
                'iscrowd': torch.as_tensor([0, 0], dtype=torch.int64, device=device)}]

    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
    model.train()

    res = model(img, target)
    print(res)


if __name__ == '__main__':
    main()