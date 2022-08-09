import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor
import numpy as np
from ltr.models.neck.position_encoding import build_position_encoding
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

class DiMPnet_DeT(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, feature_extractor_depth, classifier, bb_regressor, classification_layer, bb_regressor_layer,
                   merge_type='mean', W_rgb=0.6 ,W_depth=0.4):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_extractor_depth = feature_extractor_depth
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

        self.position_encoding = build_position_encoding(hidden_dim=256, position_embedding='sine')

        #template 
        self.featurefusion_layer2 = build_featurefusion_network(d_model=256,dropout=0.1, nhead=8, dim_feedforward=1024,num_featurefusion_layers=2)
        self.featurefusion_layer3 = build_featurefusion_network(d_model=256,dropout=0.1, nhead=8, dim_feedforward=1024,num_featurefusion_layers=2)
        self.RGB_layer3_conv = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.D_layer3_conv = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.layer3_up = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.RGB_layer2_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1,stride=1, padding=0)
        self.D_layer2_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1,stride=1, padding=0)
        self.layer2_up = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1,stride=1, padding=0)

        self.fuse_weight_layer2 = torch.nn.Parameter(torch.ones((512,1,1)), requires_grad=True)
        self.fuse_weight_layer3 = torch.nn.Parameter(torch.ones((1024,1,1)), requires_grad=True)
        self.fuse_weight_layer2.data.fill_(0.01)
        self.fuse_weight_layer3.data.fill_(0.01)

        


        self.merge_type = merge_type
        if self.merge_type == 'conv':
            self.merge_layer2 = nn.Conv2d(1024, 512, (1,1))
            self.merge_layer3 = nn.Conv2d(2048, 1024, (1,1))

        # self.id = 1

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        
        #change the shape of the input:torch.Size([3, 10, 6, 288, 288])-->torch.Size([30, 6, 288, 288])
        train_imgs = train_imgs.reshape(-1, *train_imgs.shape[-3:])
        test_imgs = test_imgs.reshape(-1, *test_imgs.shape[-3:])

        #get the NestTensor[Tensor, mask] : it should deal with the dim=6?
        if not isinstance(train_imgs, NestedTensor):
            train_imgs = nested_tensor_from_tensor(train_imgs) 
        if not isinstance(test_imgs, NestedTensor):
            test_imgs = nested_tensor_from_tensor(test_imgs)
        
        #get the feature: RGB+depth have been fusion
        train_feat = self.extract_backbone_features(train_imgs, model='train') #torch.Size([3, 10, 6, 288, 288])-->torch.Size([30, 6, 288, 288]) {'layer2': tensor([[[[1.4271e-0...Backward>), 'layer3': tensor([[[[3.1881e-0...Backward>)}
        test_feat = self.extract_backbone_features(test_imgs, model='test')

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat) #torch.Size([30, 1024, 18, 18]) 'layer3'
        test_feat_clf = self.get_backbone_clf_feat(test_feat)  #torch.Size([30, 1024, 18, 18])

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat) # list [tensor([[[[2.1355e-0...Backward>), tensor([[[[3.3256e-0...Backward>)]  torch.Size([30, 512, 36, 36]) torch.Size([30, 1024, 36, 36])
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]              # Song : layer2 and layer 3

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def merge2(self, color_feat, depth_feat):

        feat = {}

        if self.merge_type == 'conv':
            feat['layer2'] = self.merge_layer2(torch.cat((color_feat['layer2'], depth_feat['layer2']), 1))
            feat['layer3'] = self.merge_layer3(torch.cat((color_feat['layer3'], depth_feat['layer3']), 1))

        elif self.merge_type == 'max':
            # for Torch 1.7.1
            # feat['layer2'] = torch.maximum(color_feat['layer2'], depth_feat['layer2'])
            # feat['layer3'] = torch.maximum(color_feat['layer3'], depth_feat['layer3'])

            # for Torch 1.4.0
            feat['layer2'] = torch.max(color_feat['layer2'], depth_feat['layer2'])
            feat['layer3'] = torch.max(color_feat['layer3'], depth_feat['layer3'])

        elif self.merge_type == 'mul':
            feat['layer2'] = torch.mul(color_feat['layer2'], depth_feat['layer2'])
            feat['layer3'] = torch.mul(color_feat['layer3'], depth_feat['layer3'])

        elif self.merge_type == 'mean':
            feat['layer2'] = 0.5 * color_feat['layer2'] + 0.5 * depth_feat['layer2']
            feat['layer3'] = 0.5 * color_feat['layer3'] + 0.5 * depth_feat['layer3']

        elif self.merge_type == 'weightedSum':
            feat['layer2'] = self.W_rgb * color_feat['layer2'] + self.W_depth * depth_feat['layer2']
            feat['layer3'] = self.W_rgb * color_feat['layer3'] + self.W_depth * depth_feat['layer3']

        return feat
    def merge(self, color_feat, depth_feat):
        init_feat = {}
        feat = {}
        merged_feat = {}

        #the init fusion: layer2
        feat_RGB_layer2 = self.RGB_layer2_conv(color_feat['layer2'].tensors)
        feat_D_layer2 =  self.D_layer2_conv(depth_feat['layer2'].tensors)
        init_feat['layer2'] = NestedTensor(torch.max(feat_RGB_layer2,feat_D_layer2), color_feat['layer2'].mask)
        pos1_layer2 = []
        pos1_layer2.append(self.position_encoding(init_feat['layer2']).to(torch.float32))

        #RGB+Depth information: layer2
        infor_layer2 = torch.cat((feat_RGB_layer2, feat_D_layer2),dim=2)
        #clip mask 
        assert color_feat['layer2'].mask.equal(depth_feat['layer2'].mask)
        m = color_feat['layer2'].mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=infor_layer2.shape[-2:]).to(torch.bool)[0]
        feat['layer2'] = NestedTensor(infor_layer2, mask)
        #get position encoding
        pos2_layer2 = []
        pos2_layer2.append(self.position_encoding(feat['layer2']).to(torch.float32))

        #layer2 transformer fusion
        featurefusion_layer2 = self.featurefusion_layer2
        fusionfeat_layer2 = featurefusion_layer2(init_feat['layer2'].tensors, init_feat['layer2'].mask, feat['layer2'].tensors, feat['layer2'].mask, pos1_layer2[-1], pos2_layer2[-1])

        merged_feat['layer2'] = self.layer2_up(fusionfeat_layer2)

        #fusion layer3
        #down sampling 1024->512
        feat_RGB_layer3 = self.RGB_down(color_feat['layer3'].tensors)
        feat_D_layer3 = self.D_down(depth_feat['layer3'].tensors)
        RGBm = color_feat['layer3'].mask
        dm = color_feat['layer3'].mask
        assert RGBm is not None
        assert RGBm.equal(dm)
        # dm = color_feat['layer3'].mask
        # assert dm is not None
        # dmask = F.interpolate(dm[None].float(), size=infor_layer2.shape[-2:]).to(torch.bool)[0]

        init_feat['layer3'] = NestedTensor(torch.max(feat_RGB_layer3, feat_D_layer3), RGBm)
        pos1_layer3 = []
        pos1_layer3.append(self.position_encoding(init_feat['layer3']).to(torch.float32))

        #RGB+Depth information: layer3
        infor_layer3 = torch.cat((feat_RGB_layer3, feat_D_layer3),dim=2)
        #clip mask
        mask = F.interpolate(RGBm[None].float(), size=infor_layer3.shape[-2:]).to(torch.bool)[0]
        feat['layer3'] = NestedTensor(infor_layer3, mask)
        pos2_layer3 = []
        pos2_layer3.append(self.position_encoding(feat['layer3']).to(torch.float32))

        #layer3 transformer fusion
        featurefusion_layer3 = self.featurefusion_layer3
        fusionfeat_layer3 = featurefusion_layer3(init_feat['layer3'].tensors, init_feat['layer3'].mask, feat['layer3'].tensors, feat['layer3'].mask, pos1_layer3[-1], pos2_layer3[-1])
        #up sampling 512->1024
        merged_feat['layer3'] = self.layer3_up(fusionfeat_layer3)

        
        return merged_feat

    
    
    def fusion(self, color_feat, depth_feat):
        
        merged_feat = {}

        #layer2
        feat_RGB_layer2 = NestedTensor(self.RGB_layer2_conv(color_feat['layer2'].tensors), color_feat['layer2'].mask)
        feat_D_layer2 =  NestedTensor(self.D_layer2_conv(depth_feat['layer2'].tensors), depth_feat['layer2'].mask)
        
        pos1_layer2, pos2_layer2 = [], []
        pos1_layer2.append(self.position_encoding(feat_RGB_layer2).to(torch.float32))
        pos2_layer2.append(self.position_encoding(feat_D_layer2).to(torch.float32))
        #q:Depth, kv:RGB
        fusionfeat_layer2 = self.featurefusion_layer2(feat_D_layer2.tensors, feat_D_layer2.mask, feat_RGB_layer2.tensors, feat_RGB_layer2.mask, pos2_layer2[-1], pos1_layer2[-1])
        fusionfeat_layer2 = self.layer2_up(fusionfeat_layer2)
        fusionfeat_layer2 = depth_feat['layer2'].tensors + self.fuse_weight_layer2 * fusionfeat_layer2

        # fusionfeat_layer2 = torch.cat((color_feat['layer2'].tensors, fusionfeat_layer2),dim=1)
        fusionfeat_layer2 = 0.5* color_feat['layer2'].tensors + 0.5 * fusionfeat_layer2
        merged_feat['layer2'] = fusionfeat_layer2

        #layer3
        feat_RGB_layer3 = NestedTensor(self.RGB_layer3_conv(color_feat['layer3'].tensors), color_feat['layer3'].mask)
        feat_D_layer3 =  NestedTensor(self.D_layer3_conv(depth_feat['layer3'].tensors), depth_feat['layer3'].mask)
        
        pos1_layer3, pos2_layer3 = [], []
        pos1_layer3.append(self.position_encoding(feat_RGB_layer3).to(torch.float32))
        pos2_layer3.append(self.position_encoding(feat_D_layer3).to(torch.float32))
        #q:Depth, kv:RGB
        fusionfeat_layer3 = self.featurefusion_layer3(feat_D_layer3.tensors, feat_D_layer3.mask, feat_RGB_layer3.tensors, feat_RGB_layer3.mask, pos2_layer3[-1], pos1_layer3[-1])
        fusionfeat_layer3 = self.layer3_up(fusionfeat_layer3)
        fusionfeat_layer3 = depth_feat['layer3'].tensors + self.fuse_weight_layer3 * fusionfeat_layer3

        # fusionfeat_layer3 = torch.cat((color_feat['layer3'].tensors, fusionfeat_layer3),dim=1)
        fusionfeat_layer3 = 0.5 * color_feat['layer3'].tensors + 0.5 * fusionfeat_layer3
        merged_feat['layer3'] = fusionfeat_layer3

        return merged_feat




    # get the layer feature and RGB-depth fusion
    def extract_backbone_features(self, tensor_list, layers=None, model='train'):

        if layers is None:
            layers = self.output_layers

        im = tensor_list.tensors
        dims = im.shape
        if dims[1] == 6:
            color_feat_temporary = self.feature_extractor(im[:, :3, :, :], layers) #OrderedDict([('layer2', tensor([[[[2.1355e-0...ackward1>)), ('layer3', tensor([[[[3.3256e-0...ackward1>))]) #color_feat['layer2'].shape torch.Size([30, 512, 36, 36]) # layer3 torch.Size([30, 1024, 18, 18])
            depth_feat_temporary = self.feature_extractor_depth(im[:, 3:, :, :], layers)

            #change the mask by the feature tensor
            color_feat: Dict[str, NestedTensor] = {}
            for name, x in color_feat_temporary.items():
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                color_feat[name] = NestedTensor(x, mask)
            
            depth_feat: Dict[str, NestedTensor] = {}
            for name, x in depth_feat_temporary.items():
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                depth_feat[name] = NestedTensor(x, mask)


            merged_feat = self.fusion(color_feat, depth_feat)
            # if model == 'train':
            #     merged_feat = self.merge(color_feat, depth_feat)
            # else:
            #     merged_feat = self.merge1(color_feat, depth_feat)
            # self.id += 1
            return merged_feat
        else:
            return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        dims = im.shape
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            if dims[1] == 6:
                color_feat = self.feature_extractor(im[:, :3, :, :], layers)
                depth_feat = self.feature_extractor_depth(im[:, 3:, :, :], layers)
                return self.merge(color_feat, depth_feat)
            else:
                return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        if dims[1] == 6:
            color_feat = self.feature_extractor(im[:, :3, :, :], layers)
            depth_feat = self.feature_extractor_depth(im[:, 3:, :, :], layers)
            all_feat = self.merge(color_feat, depth_feat)
        else:
            all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})



@model_constructor
def dimp50_DeT(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=(),
              merge_type='max', W_rgb=0.6, W_depth=0.4):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, la='rgb', frozen_layers=frozen_backbone_layers)
    backbone_net_depth = backbones.resnet50(pretrained=backbone_pretrained, la='rgb', frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)
    

    # DiMP network
    net = DiMPnet_DeT(feature_extractor=backbone_net, feature_extractor_depth=backbone_net_depth, classifier=classifier, bb_regressor=bb_regressor,
                      classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'],
                      merge_type=merge_type, W_rgb=W_rgb, W_depth=W_depth)
    return net
