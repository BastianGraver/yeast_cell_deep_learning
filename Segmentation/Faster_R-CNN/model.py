import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F

class Model_RPN(nn.Module):
    def __init__(self, num_anchors, **kwargs):
        super(Model_RPN, self).__init__()
        resnet34 = torchvision.models.resnet34(pretrained=True)
        resnet34.layer4[0].conv1.stride = (1, 1)
        resnet34.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet34.children())[:-2])

        self.num_anchors = num_anchors

        self.feat_dim = 512
        self.middle_dim = self.feat_dim // 4
        
        self.rpn_c1 = nn.Conv2d(self.feat_dim, self.middle_dim, 3, padding=1)
        self.relu1 = nn.ReLU()

        self.rpn_cls = nn.Conv2d(self.middle_dim, self.num_anchors, 1)
        self.rpn_reg = nn.Conv2d(self.middle_dim, 4 * self.num_anchors, 1)
        
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        base_x = self.base(x)
        
        x = self.relu1(self.rpn_c1(base_x))
        
        cls_k = self.sigmoid1(self.rpn_cls(x))
        cls_k = cls_k.permute(0, 2, 3, 1)

        reg_k = self.rpn_reg(x)
        reg_k = reg_k.permute(0, 2, 3, 1)

        return base_x, cls_k, reg_k


class Classifier(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(Classifier, self).__init__()
        
        self.pooling_regions = 7
        self.feat_dim = 512

        self.red_conv_roi = nn.Conv2d(self.feat_dim, self.feat_dim // 4, 1)

        self.d1 = nn.Linear(self.feat_dim // 4 * self.pooling_regions * self.pooling_regions, self.feat_dim * 2, bias=True)
        self.relu_d1 = nn.ReLU()
        self.drop_d1 = nn.Dropout(p=0.5, inplace=False)

        self.d2 = nn.Linear(self.feat_dim * 2, self.feat_dim, bias=True)
        self.relu_d2 = nn.ReLU()
        self.drop_d2 = nn.Dropout(p=0.5, inplace=False)

        self.d3 = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.softmax_d3 = nn.Softmax(dim=1)

        self.d4 = nn.Linear(self.feat_dim, 4 * (num_classes - 1), bias=False)

    def forward(self, base_x, rois):
        outputs = []

        b = base_x.size(0)
        f = base_x.size(1)
        h = base_x.size(2)
        w = base_x.size(3)
        
        for rid in range(rois.size(0)):
            x, y, h, w = rois[rid]
            x, y, h, w = x.int(), y.int(), h.int(), w.int()
            cropped_image = base_x[:, :, x:x+w, y:y+h]
            resized_image = F.adaptive_avg_pool2d(cropped_image, (self.pooling_regions, self.pooling_regions))
            outputs.append(resized_image)

        out_roi_pool = torch.cat(outputs, 0)
        out_roi_pool = self.red_conv_roi(out_roi_pool)
        out_roi_pool = out_roi_pool.view(rois.size(0), -1)

        out_roi_pool = self.drop_d1(self.relu_d1(self.d1(out_roi_pool)))
        out_roi_pool = self.drop_d2(self.relu_d2(self.d2(out_roi_pool)))

        out_class = self.softmax_d3(self.d3(out_roi_pool))
        out_regr = self.d4(out_roi_pool)

        return out_class, out_regr
