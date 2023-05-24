import os
import sys
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath('model'))
sys.path.insert(0, root)
from models import convnext, vision_transformer, resnet
import torch.nn as nn
import torchvision


def create_model(name_model, num_classes) : 
    if name_model == 'resnet50' : return resnet.resnet50(num_classes= num_classes)
    elif name_model =='resnet101' : return resnet.resnet101(num_classes= num_classes)
    elif name_model == 'resnet152' : return resnet.resnet152(num_classes=num_classes)
    elif name_model == 'resnext50_32x4d' : return resnet.resnext50_32x4d(num_classes=num_classes)

    if name_model == 'vit_l16' : return vision_transformer.vit_l16(num_classes = num_classes,pretrained=False)
    elif name_model == 'convnext_small' : return convnext.convnext_small(num_classes = num_classes)
    elif name_model == 'convnext_base' : return convnext.convnext_base(num_classes = num_classes)
    elif name_model == 'convnext_large' : return convnext.convnext_large(num_classes = num_classes)
    elif name_model == 'convnext_xlarge' : return convnext.convnext_xlarge(num_classes = num_classes)
    else : return 'name model error'

# model = create_model(name_model= 'resnet50', num_classes= 2)

