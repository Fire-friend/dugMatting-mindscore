import mindspore
import mindspore.nn as nn
import mindcv
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.ops as mop
import mindspore.numpy as mnp

from config.Base_option import Base_options
from utils.util import get_yaml_data, set_yaml_to_args


class IBNorm(nn.Cell):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def construct(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...])
        in_x = self.inorm(x[:, self.bnorm_channels:, ...])

        return mop.cat([bn_x, in_x], axis=1)


class Conv2dIBNormRelu(nn.Cell):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      group=groups, has_bias=bias, pad_mode='pad')
        ]
        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU())

        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)


class SEBlock(nn.Cell):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(in_channels, int(in_channels // reduction)),
            nn.ReLU(),
            nn.Dense(int(in_channels // reduction), out_channels),
            nn.Sigmoid()
        )

    def construct(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


class LRBranch(nn.Cell):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__()

        enc_channels = backbone.enc_channels

        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False,
                                        with_relu=False)

    def construct(self, img, inference):
        enc_features = self.backbone(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x = mop.interpolate(enc32x, scale_factor=2.0, recompute_scale_factor=True, mode="bilinear")
        lr16x = self.conv_lr16x(lr16x)
        lr8x = mop.interpolate(lr16x, scale_factor=2.0, recompute_scale_factor=True, mode='bilinear')
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = mop.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x]


class HRBranch(nn.Cell):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels, in_ch=None):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + in_ch, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.SequentialCell(
            Conv2dIBNormRelu(3 * hr_channels + in_ch, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.SequentialCell(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.SequentialCell(
            Conv2dIBNormRelu(hr_channels + in_ch, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def construct(self, img, enc2x, enc4x, lr8x, inference):
        img2x = mop.interpolate(img, scale_factor=1 / 2, recompute_scale_factor=True, mode='bilinear',
                                align_corners=False)
        img4x = mop.interpolate(img, scale_factor=1 / 4, recompute_scale_factor=True, mode='bilinear',
                                align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(mop.cat((img2x, enc2x), axis=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(mop.cat((hr4x, enc4x), axis=1))

        lr4x = mop.interpolate(lr8x, scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(mop.cat((hr4x, lr4x, img4x), axis=1))

        hr2x = mop.interpolate(hr4x, scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(mop.cat((hr2x, enc2x), axis=1))

        pred_detail = None
        if not inference:
            hr = mop.interpolate(hr2x, scale_factor=2.0, recompute_scale_factor=True, mode='bilinear',
                                 align_corners=False)
            hr = self.conv_hr(mop.cat((hr, img), axis=1))
            pred_detail = mop.sigmoid(hr)

        return pred_detail, hr2x


class FusionBranch(nn.Cell):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels, in_ch=None):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)

        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        # self.conv_f2x_f = Conv2dIBNormRelu(hr_channels + in_ch, int(hr_channels / 2), 3, stride=1, padding=1)

        self.conv_f = nn.SequentialCell(
            Conv2dIBNormRelu(hr_channels + in_ch, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )
        # self.conv_f = Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False,
        #                                with_relu=False)

        self.conv_lamda = nn.SequentialCell(
            Conv2dIBNormRelu(hr_channels + in_ch, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )
        # self.conv_lamda = Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False,
        #                                    with_relu=False)

        self.conv_alpha = nn.SequentialCell(
            Conv2dIBNormRelu(hr_channels + in_ch, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )
        # self.conv_alpha = Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False,
        #                                    with_relu=False)

        self.conv_beta = nn.SequentialCell(
            Conv2dIBNormRelu(hr_channels + in_ch, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )
        # self.conv_beta = Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False,
        #                                   with_relu=False)

    def construct(self, img, lr8x, hr2x):
        lr4x = mop.interpolate(lr8x, scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = mop.interpolate(lr4x, scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(mop.cat((lr2x, hr2x), axis=1))
        f = mop.interpolate(f2x, scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=False)
        last_in = mop.cat((f, img), axis=1)
        # last_in = self.conv_f2x_f(last_in)

        f = self.conv_f(last_in)
        pred_matte = mop.sigmoid(f)
        pred_la = mop.softplus(self.conv_lamda(last_in)) + 0.1
        pred_alpha = mop.softplus(self.conv_alpha(last_in)) + 2.1
        pred_beta = mop.softplus(self.conv_beta(last_in)) + 0.1
        return pred_matte, pred_la, pred_alpha, pred_beta


class MobileNetV2Backbone(nn.Cell):
    """ MobileNetV2 Backbone
    """

    def __init__(self):
        super().__init__()
        self.model = mindcv.create_model('mobilenet_v2_100', pretrained=True)
        # self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]
        self.out_channels = [16, 24, 32, 96, 1280]
        self.out_channels_sum = sum(self.enc_channels)
        self.output_size_num = 5

    def construct(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc2x = x
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        enc4x = x

        x = self.model.features[6](x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        enc8x = x

        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        enc16x = x

        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        x = self.model.features[19](x)
        x = self.model.features[20](x)
        x = self.model.features[21](x)
        x = self.model.features[22](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]


class ITMODNet_Net(nn.Cell):
    def __init__(self, args):
        super(ITMODNet_Net, self).__init__()
        in_channels = args.in_channels
        hr_channels = args.hr_channels
        backbone_arch = args.backbone

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        # self.match = nn.Conv2d(self.in_channels + 1, self.in_channels, 3, 1, 1)

        self.backbone = MobileNetV2Backbone()
        self.backbone.model.features[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2))

        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels, self.in_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels, self.in_channels)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         self._init_conv(m)
        #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        #         self._init_norm(m)
        # self.backbone.load_pretrained_ckpt()

    def construct(self, input, inference=False):
        # input = self.match(input)
        # show
        # show_tensor(img.permute([0, 2, 3, 1])[0])
        # input = input[:, :3] + 0 * self.match(input)
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(input, inference)
        pred_detail, hr2x = self.hr_branch(input, enc2x, enc4x, lr8x, inference)
        pred_matte, pred_la, pred_alpha, pred_beta = self.f_branch(input, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_la, pred_alpha, pred_beta, pred_matte


if __name__ == '__main__':
    from mindspore import context
    context.set_context(device_target="GPU")
    base_option = Base_options()
    args = base_option.get_args()
    yamls_dict = get_yaml_data('/hy-tmp/CycleGAN-main/train/config/' + args.model + '_config.yaml')
    set_yaml_to_args(args, yamls_dict)

    network = ITMODNet_Net(args)
    input = mindspore.Tensor(np.random.rand(1, 4, 512, 512).astype(np.float32))
    out = network(input)
    print()
