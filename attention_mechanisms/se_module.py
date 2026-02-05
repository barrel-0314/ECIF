
""" 
PyTorch implementation of Squeeze-and-Excitation Networks

As described in https://arxiv.org/pdf/1709.01507

The SE block is composed of two main components: the squeeze layer and the excitation layer. 
The squeeze layer reduces the spatial dimensions of the input feature maps by taking the average 
value of each channel. This reduces the number of parameters in the network, making it more efficient. 
The excitation layer then applies a learnable gating mechanism to the squeezed feature maps, which helps
to select the most informative channels and amplifies their contribution to the final output.

"""

import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # 对 nn.Linear 层的权重进行 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # b, c, w, h = x.size()
        # avg_y = self.avg_pool(x).view(b, c)
        # max_y = self.avg_pool(x).view(b, c)
        # y = self.fc(avg_y + max_y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        attn = self.fc(x)
        return x * attn


class SELayerConv(nn.Module):
    def __init__(self, emb_dim=256, kernels_size=16):
        super(SELayerConv, self).__init__()

        self.conv2i = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernels_size, padding=0)
        self.conv3i = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=kernels_size, padding=0)
        self.fc_conv = nn.Sequential(
            nn.Linear(emb_dim - kernels_size + 1, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.Softmax()
        )
        # 对 nn.Linear 层的权重进行 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        b, c, e = x.size()
        if c == 2:
            y = self.conv2i(x)
            return x * self.fc_conv(y)
        if c == 3:
            y = self.conv3i(x)
            return x * self.fc_conv(y)

        # avg_y = self.avg_pool(x).view(b, c)
        # max_y = self.avg_pool(x).view(b, c)
        # y = self.fc(avg_y + max_y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        attn = self.fc(x)
        return x * attn

    
if __name__ == "__main__":
    # # x = torch.randn(2, 64, 32, 32) #(B, C, H, W)
    # # attn = SELayer(channel=64, reduction=16)
    # # y = attn(x)
    # # print(y.shape)
    x = torch.randn(10, 5)
    # attn = SELayer(5)
    channel, reduction = 5, 16
    l1 = nn.Linear(5, 5, bias=True)
    y1 = l1(x)
    print(y1.shape)
    r = nn.ReLU(inplace=True)
    y2 = r(y1)
    print(y2.shape)
    l2 = nn.Linear(5, 5, bias=True)
    y3 = l2(y2)
    print(y3.shape)
    s = nn.Sigmoid()
    y4 = s(y3)
    print(y4.shape)

    # # 定义输入张量 x
    # x = torch.tensor([[1.0, 2.0, 3.0]])
    #
    # # 创建 nn.Linear 层
    # linear_layer = nn.Linear(3, 2)  # 输入维度为3，输出维度为2
    #
    # # 使用 linear_layer 对象进行计算
    # output = linear_layer(x)
    #
    # print(output)