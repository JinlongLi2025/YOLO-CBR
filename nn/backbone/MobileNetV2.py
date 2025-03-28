from torch import nn


class ConvBNReLU(nn.Sequential):  # 该函数主要做卷积 池化 ReLU6激活操作
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2  # 池化 = （步长-1）整除2
        super(ConvBNReLU, self).__init__(  # 调用ConvBNReLU父类添加模块
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False, groups=groups),  # bias默认为False
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):  # 该模块主要实现了倒残差模块
    def __init__(self, inp, oup, stride, expand_ratio):  # inp 输入 oup 输出 stride步长 exoand_ratio 按比例扩张
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))  # 由于有到残差模块有1*1,3*3的卷积模块，所以可以靠expand_rarton来进行升维
        self.use_res_connect = self.stride == 1 and inp == oup  # 残差连接的判断条件：当步长=1且输入矩阵与输出矩阵的shape相同时进行
        layers = []
        if expand_ratio != 1:  # 如果expand_ratio不等于1，要做升维操作，对应图中的绿色模块
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))  # 这里添加的是1*1的卷积操作
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 这里做3*3的卷积操作，步长可能是1也可能是2,groups=hidden_dim表示这里使用了分组卷积的操作，对应图上的蓝色模块

            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  # 对应图中的黄色模块
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)  # 将layers列表中的元素解开依次传入nn.Sequential

    def forward(self, x):
        if self.use_res_connect:  # 如果使用了残差连接，就会进行一个x+的操作
            return x + self.conv(x)
        else:
            return self.conv(x)  # 否则不做操作
