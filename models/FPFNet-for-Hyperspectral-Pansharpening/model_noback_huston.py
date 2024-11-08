import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


# conv 3x3
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(True))
    return blk


# deconv
def deconv_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True))
    return blk


# final conv 1x1
def conv1x1(in_channels, out_channels):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True))
    return blk

# resblock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        # conv 1x1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel*3, 192, kernel_size=1),
                                   nn.BatchNorm2d(192))

        # conv 3x3
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(True))
        # conv 5x5
        self.conv5 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(True))
        # conv 7x7
        self.conv7 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(True))
        # relu
        self.relu = nn.ReLU(inplace=True)

        self.out_conv3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),   # 2.3
                                       nn.BatchNorm2d(out_channel),
                                       nn.ReLU(True))

        if in_channel != out_channel:
            self.transition = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(True),
                nn.Conv2d(in_channel, out_channel, kernel_size=1)
            )
        else:
            self.transition = lambda x: x

    def forward(self, inputs):
        out1_conv3 = self.conv3(inputs)
        out1_conv5 = self.conv5(inputs)
        out1_conv7 = self.conv7(inputs)
        out1 = torch.cat([out1_conv3, out1_conv5, out1_conv7], 1)
        out1 = self.conv1(out1)
        out2 = self.transition(inputs)
        out = torch.add(out1, out2)
        out = self.relu(out)
        out = self.out_conv3(out)
        return out


# resblock
class HS_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel):
        super(HS_BasicBlock, self).__init__()
        # conv 1x1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel*3, 256, kernel_size=1),
                                   nn.BatchNorm2d(256))

        # conv 3x3
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(True))
        # out conv 3x3
        self.out_conv3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(out_channel),
                                       nn.ReLU(True))
        # conv 5x5
        self.conv5 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(True))
        # conv 7x7
        self.conv7 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(True))
        # relu
        self.relu = nn.ReLU(inplace=True)

        if in_channel != out_channel:
            self.transition = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(True),
                nn.Conv2d(in_channel, out_channel, kernel_size=1)
            )
        else:
            self.transition = lambda x: x

    def forward(self, inputs):
        out1_conv3 = self.conv3(inputs)
        out1_conv5 = self.conv5(inputs)
        out1_conv7 = self.conv7(inputs)
        out1 = torch.cat([out1_conv3, out1_conv5, out1_conv7], 1)
        out1 = self.conv1(out1)
        # 去掉hs的res
        out2 = self.transition(inputs)
        out = torch.add(out1, out2)
        out = self.relu(out)
        out = self.out_conv3(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, down, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.down = down
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)


    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion))

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            if self.down == 0:
                for j in range(num_branches):
                    if j > i:
                        fuse_layer.append(None)

                    elif j == i:
                        fuse_layer.append(None)
                    else:
                        conv3x3s = []
                        for k in range(i - j):
                            if k == i - j - 1:
                                num_outchannels_conv3x3 = num_inchannels[i]
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)))
                            else:
                                num_outchannels_conv3x3 = num_inchannels[j]
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=True)))
                        fuse_layer.append(nn.Sequential(*conv3x3s))
                fuse_layers.append(nn.ModuleList(fuse_layer))
            else:
                for j in range(num_branches):  # [320, 160]
                    if j > i:
                        fuse_layer.append(None)
                    elif j == i:
                        fuse_layer.append(None)
                    else:
                        conv3x3s = []
                        for k in range(i - j):
                            if k == i - j - 1:
                                num_outchannels_conv3x3 = num_inchannels[i]
                                conv3x3s.append(nn.Sequential(
                                    nn.ConvTranspose2d(num_inchannels[j], num_inchannels[i], kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)))
                            else:
                                num_outchannels_conv3x3 = num_inchannels[j]
                                conv3x3s.append(nn.Sequential(
                                    nn.ConvTranspose2d(num_inchannels[j], num_inchannels[i], kernel_size=3, stride=2, padding=1, bias=False, output_padding=1),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=True)))
                        fuse_layer.append(nn.Sequential(*conv3x3s))
                fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        if self.down == 1:
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                if i == 0:
                    y = x[0]
                    x_fuse.append(y)
                    continue
                for j in range(1, self.num_branches):
                    if i == j:
                        # y = y + x[j]
                        y = torch.cat([y, x[j]], 1)
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        # y = y + self.fuse_layers[i][j](x[j])
                        y = torch.cat([y, self.fuse_layers[i][j](x[j])], 1)
                    else:
                        # y = y + self.fuse_layers[i][j](x[j])
                        y = torch.cat([y, self.fuse_layers[i][j](x[j])], 1)
                x_fuse.append(self.relu(y))
        else:
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                if i == 0:
                    y = x[0]
                    x_fuse.append(y)
                    continue
                for j in range(1, self.num_branches):
                    if i == j:
                        # y = y + x[j]
                        y = torch.cat([y, x[j]], 1)
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        # y = y + F.interpolate(
                        #     self.fuse_layers[i][j](x[j]),
                        #     size=[height_output, width_output],
                        #     mode='bilinear')
                        y = torch.cat([y, F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear')], 1)
                    else:
                        # y = y + self.fuse_layers[i][j](x[j])
                        y = torch.cat([y, self.fuse_layers[i][j](x[j])], 1)
                x_fuse.append(self.relu(y))
        return x_fuse, x


blocks_dict = {
    'BASIC': BasicBlock,
    'HS_BASIC': HS_BasicBlock
}


class PAN_HighResolutionNet(nn.Module):
    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(PAN_HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]  # 64
        block = blocks_dict[self.stage1_cfg['BLOCK']]  # BASIC
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]  # 4
        self.layer1 = self._make_layer(block, 192, num_channels, num_blocks)  # [BASIC , 64, 64, 4]
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']  # [48, 96]
        block = blocks_dict[self.stage2_cfg['BLOCK']]  # BASIC
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels, 1)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)  # [48,96] * expansion

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, 1)

        self.conv3 = nn.Sequential(nn.Conv2d(384, 192, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU(True))

        self.conv3_0 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(True))

    # 对于第二层
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer, flag=1):
        num_branches_cur = len(num_channels_cur_layer)  # 3
        num_branches_pre = len(num_channels_pre_layer)  # 2

        transition_layers = []
        for i in range(num_branches_cur):  # i = [0, 1, 2]
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # 通道变化 前一层--->当前层
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1,
                                  padding=1),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for k in range(num_branches_pre):
                    # 下采样
                    if flag == 1:
                        for j in range(i + 1 - num_branches_pre):
                            inchannels = num_channels_pre_layer[k]
                            outchannels = num_channels_cur_layer[i] \
                                if j == i - num_branches_pre else inchannels  # 下采样
                            if k == 1:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(inplace=True)))
                            elif k == 0 and len(num_channels_pre_layer) == 2:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(True),
                                    nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(True)
                                ))
                            else:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2 * (num_branches_cur - 1 - k), padding=1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(inplace=True)))
                    # 上采样
                    else:
                        for j in range(i + 1 - num_branches_pre):
                            inchannels = num_channels_pre_layer[k]
                            outchannels = num_channels_cur_layer[i] \
                                if j == i - num_branches_pre else inchannels
                            if k == 1:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.ConvTranspose2d(inchannels, outchannels,
                                                           kernel_size=2 * (num_branches_cur - k) - 1,
                                                           stride=2 * (num_branches_cur - 1 - k),
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True)))
                            if k == 0 and num_channels_pre_layer == 2:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.ConvTranspose2d(inchannels, outchannels,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(inchannels, outchannels,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True)
                                    ))
                            else:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.ConvTranspose2d(inchannels, outchannels,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True))
                                )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']  # 1
        num_branches = layer_config['NUM_BRANCHES']  # 2
        num_blocks = layer_config['NUM_BLOCKS']  # [4, 4]
        num_channels = layer_config['NUM_CHANNELS']  # [48, 96]
        block = blocks_dict[layer_config['BLOCK']]  # BASIC
        fuse_method = layer_config['FUSE_METHOD']  # SUM
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,  # 2 分支数
                                     block,  # BASIC
                                     num_blocks,  # [4, 4] block数
                                     num_inchannels,  # [48,96] *
                                     num_channels,  # [48,96]
                                     fuse_method,  # SUM
                                     0,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list, x_input = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    for i in range(len(self.transition2)):
                        if self.transition2[i] is None:
                            continue
                        y1 = y_list[0] if i == 0 else self.transition2[i][0](x_input[0])
                        y2 = y_list[0] if i == 0 else self.transition2[i][1](x_input[1])
                        y = torch.cat([y1, y2], 1)
                        y = self.conv3(y)
                        x_list.append(y)
            else:
                y = self.conv3_0(y_list[i]) if i == 0 else self.conv3(y_list[i])
                x_list.append(y)
        return x_list


class HS_HighResolutionNet(nn.Module):
    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HS_HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(144, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['HS_NUM_CHANNELS'][0]  # 64
        block = blocks_dict[self.stage1_cfg['HS_BLOCK']]  # BASIC
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]  # 4
        self.layer1 = self._make_layer(block, 256, num_channels, num_blocks)  # [BASIC , 64, 64, 4]
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['HS_NUM_CHANNELS']  # [48, 96]
        block = blocks_dict[self.stage2_cfg['HS_BLOCK']]  # BASIC
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels, 0)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)  # [48,96] * expansion

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['HS_NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['HS_BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, 0)

        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))

        self.conv3_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))


    # 对于第二层
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer, flag=1):
        num_branches_cur = len(num_channels_cur_layer)  # 3
        num_branches_pre = len(num_channels_pre_layer)  # 2

        transition_layers = []
        for i in range(num_branches_cur):  # i = [0, 1, 2]
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # 通道变化 前一层--->当前层
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1,
                                  padding=1),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for k in range(num_branches_pre):
                    # 下采样
                    if flag == 1:
                        for j in range(i + 1 - num_branches_pre):
                            inchannels = num_channels_pre_layer[k]
                            outchannels = num_channels_cur_layer[i] \
                                if j == i - num_branches_pre else inchannels  # 下采样
                            if k == 1:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, 3, 2, 1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(inplace=True)))
                            if k == 0 and num_channels_pre_layer == 2:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, 3, 2, 1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(True),
                                    nn.Conv2d(outchannels, outchannels, 3, 2, 1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(True)
                                ))
                            else:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, 3, 2 * (num_branches_cur - 1 - k), 1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(inplace=True)))
                    # 上采样
                    else:
                        for j in range(i + 1 - num_branches_pre):
                            inchannels = num_channels_pre_layer[k]
                            outchannels = num_channels_cur_layer[i] \
                                if j == i - num_branches_pre else inchannels
                            if k == 1:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2 * (num_branches_cur - k) - 1,
                                                           stride=2 * (num_branches_cur - 1 - k),
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True)))
                            if k == 0 and num_channels_pre_layer == 2:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.ConvTranspose2d(inchannels, outchannels,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(inchannels, outchannels,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True)
                                    ))
                            else:
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.ConvTranspose2d(inchannels, outchannels,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1, output_padding=1),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(True))
                                )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']  # 1
        num_branches = layer_config['NUM_BRANCHES']  # 2
        num_blocks = layer_config['NUM_BLOCKS']  # [4, 4]
        num_channels = layer_config['HS_NUM_CHANNELS']  # [48, 96]
        block = blocks_dict[layer_config['HS_BLOCK']]  # BASIC
        fuse_method = layer_config['FUSE_METHOD']  # SUM

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,  # 2 分支数
                                     block,  # BASIC
                                     num_blocks,  # [4, 4] block数
                                     num_inchannels,  # [48,96] *
                                     num_channels,  # [48,96]
                                     fuse_method,   # SUM
                                     1,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list, x_input = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    for i in range(len(self.transition2)):
                        if self.transition2[i] is None:
                            continue
                        y1 = y_list[0] if i == 0 else self.transition2[i][1](self.transition2[i][0](x_input[0]))
                        y2 = y_list[0] if i == 0 else self.transition2[i][2](x_input[1])
                        y = torch.cat([y1, y2], 1)
                        y = self.conv3(y)
                        x_list.append(y)
            else:
                y = self.conv3_1(y_list[i]) if i == 0 else self.conv3(y_list[i])
                x_list.append(y)
        return x_list


class mergeModule(nn.Module):
    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(mergeModule, self).__init__()
        self.model1 = PAN_HighResolutionNet(config, **kwargs)
        self.model2 = HS_HighResolutionNet(config, **kwargs)
        self.stage3_cfg = extra['STAGE3']
        self.pan_num_channels = self.stage3_cfg['NUM_CHANNELS']
        self.hs_num_channels = self.stage3_cfg['HS_NUM_CHANNELS']
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(448, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(704, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True),
                                   nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))

        self.blk1 = conv1x1(486, 102)
        self.conv4 = nn.Sequential(nn.Conv2d(704, 512, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True),
                                   nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
        # self.conv3 = nn.Sequential(nn.Conv2d(256, 102, kernel_size=1, stride=1),
        #                            nn.BatchNorm2d(102))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 144, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(144),
                                   nn.ReLU(True))

        self.blk2 = conv1x1(256, 144)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=4)

    def forward(self, x1, x2):
        y1 = self.model1(x1)
        y2 = self.model2(x2)
        out = []
        j = len(y2) - 1
        for i in range(len(y1)):
            out.append(torch.cat([y1[i], y2[j]], 1))
            j = j - 1
        out1 = self.conv1(out[2])
        out1 = torch.cat([out1, out[1]], 1)
        out2 = self.conv2(out1)
        out = torch.cat([out2, out[0]], 1)
        out = self.conv4(out)
        out = self.blk2(out)
        # 去掉最后的res部分
        # x2 = self.up_sample(x2)
        # out = torch.add(out, x2)
        return out


def get_noback_model_huston(cfg, **kwargs):
    # b c h w

    # a = torch.randn(5, 1, 160, 160)
    # b = torch.randn(5, 144, 40, 40)
    # net = Net
    # c = net(a, b)
    # print(c.shape)
    # out1 = PAN_HighResolutionNet(cfg, **kwargs)
    # out2 = HS_HighResolutionNet(cfg, **kwargs)
    # print(out1)
    # print(out2)
    # c = out1(a)
    # d = out2(b)
    # 网络
    net = mergeModule(cfg, **kwargs)
    # 输入PAN和HS
    # out = net(a, b)

    return net
