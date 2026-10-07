# 模块说明：两个分支共用此编码器构造函数；下方保留 EEGNet 结构的原始学术来源。
"""EEGNet feature extraction used by the two NPFEEGNet-RD branches.

The temporal, depthwise spatial, and separable temporal convolutions follow
Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based
brain-computer interfaces", Journal of Neural Engineering 15 (2018), 056013.
DOI: 10.1088/1741-2552/aace8c. EEGNet is the architectural basis of these
encoders; it is not an original contribution of NPFEEGNet-RD.
"""

from torch import nn  # 导入 PyTorch 神经网络层，用于组合卷积、归一化、池化和正则化操作。


def eegnet_features(num_channels: int, temporal_kernel: int, dropout: float) -> nn.Sequential:  # 按通道数、时间卷积核长度和丢弃率构造编码器，输入为 [批量, 1, 通道, 时间]。
    # 函数说明：时间滤波器数为 8，空间深度倍数为 2，因此后续特征图数为 16。
    """Build the paper's eight-filter, depth-multiplier-two feature encoder."""
    return nn.Sequential(  # 按以下顺序执行各层，卷积和归一化的可训练参数由两个分支各自持有。
        nn.Conv2d(1, 8, (1, temporal_kernel), padding="same", bias=False),  # 每个电极沿时间提取 8 组特征，补零保持时间长度，输出 [批量, 8, 通道, 时间]。
        nn.BatchNorm2d(8, eps=1e-4),  # 对 8 张特征图分别归一化，分母加入 1e-4 保持数值稳定。
        nn.Conv2d(8, 16, (num_channels, 1), groups=8, bias=False),  # 每个时间滤波器学习 2 组跨电极空间权重，把电极维压到 1，输出 16 张特征图。
        nn.BatchNorm2d(16, eps=1e-4),  # 对空间卷积输出的 16 张特征图分别执行批归一化。
        nn.ELU(),  # 使用 ELU 非线性激活，保留正值并平滑压缩负值。
        nn.AvgPool2d((1, 4)),  # 每 4 个时间位置取均值，使时间长度变为原长度除以 4 后向下取整。
        nn.Dropout(dropout),  # 训练时按指定概率随机置零以减轻过拟合，评估模式下不丢弃元素。
        nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False),  # 在每张特征图内部独立执行长度 16 的时间卷积，并保持当前时间长度。
        nn.Conv2d(16, 16, 1, bias=False),  # 用 1×1 逐点卷积混合 16 张特征图，与上一层组成可分离时间卷积。
        nn.BatchNorm2d(16, eps=1e-4),  # 对可分离卷积后的特征图执行批归一化。
        nn.ELU(),  # 为混合后的时间特征加入非线性表达能力。
        nn.AvgPool2d((1, 8)),  # 再按 8 倍下采样时间维，最终时间长度为 时间长度 // 4 // 8。
        nn.Dropout(dropout),  # 在输出特征上应用同一丢弃率，输出形状为 [批量, 16, 1, 时间长度 // 4 // 8]。
    )  # 完成顺序编码器的构造并返回；每次调用都会创建一套新的网络参数。
