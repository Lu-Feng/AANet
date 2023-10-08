import torch
import logging
from torch import nn
from model.cct import cct_14_7x2_384
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=args.work_with_tokens), Flatten())
        self.maxpool = nn.MaxPool2d(3,3)
        
    def forward(self, x):
        x = self.backbone(x)
        x0 = x
        x1 = x

        x1 = x1.view(-1,24,24,384)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.aggregation(x1)
        x1 = torch.nn.functional.normalize(x1, p=2, dim=-1)

        x0 = x0.view(-1,24,24,384)
        x0 = x0.permute(0, 3, 1, 2)
        x0 = self.maxpool(x0)
        x0 = x0.permute(0, 2, 3, 1)
        x0 = torch.nn.functional.normalize(x0, p=2, dim=-1)
        return x0,x1


def get_backbone(args):
    args.work_with_tokens = False
    
    backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
    if args.trunc_te:
        logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
        backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
    if args.freeze_te:
        logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
        for p in backbone.parameters():
            p.requires_grad = False
        for name, child in backbone.classifier.blocks.named_children():
            if int(name) > args.freeze_te:
                for params in child.parameters():
                    params.requires_grad = True
    args.features_dim = 384
    return backbone

