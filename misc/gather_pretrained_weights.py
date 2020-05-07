"""
This file converts PyTorch's pre-trained ImageNet state dictionaries into the appropriate 
dictionaries for our RetinaNet model (since our naming conventions differ a bit).
"""
import torch
from torchvision import models
from src.models import RetinaNet


if __name__ == "__main__":

    # == ResNet18
    ref_model = models.resnet18(pretrained=True)
    my_model = RetinaNet(device="cpu", num_classes=1, layers_config=RetinaNet.resnet18_layers)

    new_state_dict = {}
    for my_k, ref_k in zip(filter(lambda k: "shortcut" not in k, my_model.state_dict().keys()), 
                           filter(lambda k: "downsample" not in k, ref_model.state_dict().keys())):
        if ref_k == "fc.weight":
            break
        new_state_dict[my_k] = pretrained_state_dict[ref_k]

    for my_k, ref_k in zip(filter(lambda k: "shortcut" in k, self.state_dict().keys()),
                               filter(lambda k: "downsample" in k, ref_model.state_dict().keys())):
        new_state_dict[my_k] = pretrained_state_dict[ref_k]

    torch.save(new_state_dict, "src/lib/pretrained_resnet18.torch")

    # == ResNet50
    ref_model = models.resnet50(pretrained=True)
    my_model = RetinaNet(device="cpu", num_classes=1, layers_config=RetinaNet.resnet50_layers)

    new_state_dict = {}
    for my_k, ref_k in zip(filter(lambda k: "shortcut" not in k, my_model.state_dict().keys()), 
                           filter(lambda k: "downsample" not in k, ref_model.state_dict().keys())):
        if ref_k == "fc.weight":
            break
        new_state_dict[my_k] = pretrained_state_dict[ref_k]

    for my_k, ref_k in zip(filter(lambda k: "shortcut" in k, self.state_dict().keys()),
                               filter(lambda k: "downsample" in k, ref_model.state_dict().keys())):
        new_state_dict[my_k] = pretrained_state_dict[ref_k]

    torch.save(new_state_dict, "src/lib/pretrained_resnet50.torch")

