import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        super().__init__()

        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTOutputHead(nn.Module):
    """
    Standard Output Head for DPT.
    Projects features to half dimension, optionally upsamples, then projects to final output.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output_conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, size=None):
        x = self.output_conv1(x)
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        x = self.output_conv2(x)
        return x


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        final_out_channels=16,
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        # Multi-scale Heads
        # Scale 1/16 (uses path_4)
        self.head_1_16 = DPTOutputHead(features, final_out_channels)
        # Scale 1/8 (uses path_3)
        self.head_1_8 = DPTOutputHead(features, final_out_channels)
        # Scale 1/4 (uses path_2)
        self.head_1_4 = DPTOutputHead(features, final_out_channels)
        # Scale 1/2 (uses path_1)
        self.head_1_2 = DPTOutputHead(features, final_out_channels)
        # Scale 1 (uses path_1 + Upsample)
        self.head_1 = DPTOutputHead(features, final_out_channels)

    def forward(self, out_features, patch_size: tuple[int], out_size: tuple[int]):

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
                
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_size[0], patch_size[1]))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        # Project layers to feature dimension
        layer_1_rn = self.scratch.layer1_rn(layer_1) # 1/4
        layer_2_rn = self.scratch.layer2_rn(layer_2) # 1/8
        layer_3_rn = self.scratch.layer3_rn(layer_3) # 1/16
        layer_4_rn = self.scratch.layer4_rn(layer_4) # 1/32
        
        # Refine paths
        # path_4 size becomes 1/16 (matches layer_3_rn)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        
        # path_3 size becomes 1/8 (matches layer_2_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        
        # path_2 size becomes 1/4 (matches layer_1_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        
        # path_1 size becomes 1/2 (refinenet1 upsamples by 2 implicitly when size=None)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        outputs = {}
        
        # Output Generation
        outputs['scale_16'] = self.head_1_16(path_4)
        outputs['scale_8']  = self.head_1_8(path_3)
        outputs['scale_4']  = self.head_1_4(path_2)
        
        # path_1 is naturally Scale 1/2
        outputs['scale_2']  = self.head_1_2(path_1)
        
        # Scale 1 requires upsampling path_1 to original image size
        outputs['scale_1']    = self.head_1(path_1, size=out_size)
        
        return outputs