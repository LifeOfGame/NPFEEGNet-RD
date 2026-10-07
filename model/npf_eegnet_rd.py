# 模块说明：原始分支逐试次去均值，频谱分支使用原输入；固定频带和等权聚合不参与学习，残差系数受限但残差输出幅值没有绝对上界。
"""The fixed NPFEEGNet-RD architecture.

A temporally demeaned raw EEGNet branch is augmented by an original-input
spectral branch. Five response-energy-normalized soft frequency bands share
one EEGNet encoder, contribute uniformly, and form a bounded logit residual.
See eegnet_encoder.py for the EEGNet architecture's academic attribution.
"""

from __future__ import annotations  # 延迟解析类型注解，减少定义时对注解所引用类型的依赖。

import math  # 提供采样率有限性检查和残差系数初始化所需的自然对数。

import torch  # 导入张量运算、快速傅里叶变换以及模型参数初始化所需的 PyTorch 接口。
from torch import nn  # 导入神经网络模块基类、网络层和可训练参数类型。

from .eegnet_encoder import eegnet_features  # 复用 EEGNet 编码器构造函数，分别创建原始分支与共享频带分支。


FIXED_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 40.0))  # 五个预先指定的频带边界，单位为 Hz；这些边界不通过训练调整。


class FixedResponseFilterBank(nn.Module):  # 在频域应用五个固定软带通响应，并将各频带信号变换回时域。
    # 类说明：归一化的是每个滤波响应的平方积分近似值，不是每段 EEG 信号的能量。
    """Five fixed soft bands with unit squared-response energy per band."""

    def __init__(self, sampling_rate: float = 250.0):  # 接收采样率，默认每秒 250 个采样点。
        super().__init__()  # 初始化 nn.Module，使后续缓冲区能随模型保存和迁移设备。
        if not math.isfinite(sampling_rate) or sampling_rate <= 80.0:  # 要求采样率有限且奈奎斯特频率严格超过最高频带边界 40 Hz。
            raise ValueError("sampling_rate must be finite and greater than 80 Hz")  # 拒绝无法支持当前固定频带的采样率。
        self.sampling_rate = float(sampling_rate)  # 保存浮点形式的采样率，供频率轴和频率间隔计算使用。
        self.register_buffer("initial_edges", torch.tensor(FIXED_BANDS))  # 注册 [5, 2] 固定边界缓冲区，随状态字典保存但不属于可训练参数。
        # These zero buffers are part of the published model's weight schema. 中文：零偏移缓冲区是已发布模型权重格式的一部分。
        # They are never parameters and do not change the fixed band edges. 中文：它们始终不是可训练参数，也不改变固定频带边界。
        self.register_buffer("edge_offsets", torch.zeros(5, 2))  # 保留 [5, 2] 全零偏移缓冲区以匹配权重格式，实际响应计算不使用它。

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,  # 覆盖权重加载钩子，接收待加载字典、当前模块前缀和严格匹配设置。
                              missing_keys, unexpected_keys, error_msgs):  # 接收父级共享的缺失键、多余键和错误列表，便于统一报告加载问题。
        expected_edges = torch.tensor(FIXED_BANDS)  # 在 CPU 上构造当前结构要求的固定边界，作为权重检查基准。
        for name, expected in (("initial_edges", expected_edges),  # 首先核验已保存的频带上下边界是否与固定配置完全一致。
                               ("edge_offsets", torch.zeros_like(expected_edges))):  # 同时核验偏移缓冲区是否保持全零，禁止借权重文件改变频带定义。
            key = prefix + name  # 拼接嵌套模块前缀，得到状态字典中的完整缓冲区键名。
            if key in state_dict:  # 仅检查字典中已有的键，缺失键留给父类按加载规则处理。
                actual = state_dict[key].detach().to(device="cpu", dtype=torch.float32)  # 脱离计算图并统一到 CPU 浮点类型，进行不依赖原设备的数值比较。
                if not torch.equal(actual, expected):  # 要求形状与每个元素完全相等，避免加载已改变频带边界的权重。
                    error_msgs.append(f"{key} must match the fixed NPFEEGNet-RD bands")  # 收集明确的频带不匹配错误，让外层加载接口统一抛出。
        super()._load_from_state_dict(  # 完成额外校验后，交由 PyTorch 的标准逻辑加载当前模块状态。
            state_dict, prefix, local_metadata, strict,  # 原样传递待加载权重及模块层级相关的加载选项。
            missing_keys, unexpected_keys, error_msgs,  # 共享问题列表，使父类继续记录缺失、多余和不兼容的状态。
        )  # 结束父类权重加载调用，保留其原有错误处理行为。

    def effective_edges(self) -> torch.Tensor:  # 提供当前实际使用的 [5, 2] 频带边界，供分析或辅助输出读取。
        return self.initial_edges  # 直接返回固定边界缓冲区，不叠加任何偏移或可训练修正。

    def frequency_response(self, samples: int, *, device, dtype) -> torch.Tensor:  # 按试次长度在指定设备和精度下构造 [5, 频率点数] 滤波响应。
        # 函数说明：以 1 Hz 为软边界过渡尺度，用频率间隔加权的响应平方和进行归一化。
        """Return H / sqrt(delta_f * sum(H**2)); transition width is 1 Hz."""
        frequencies = torch.fft.rfftfreq(  # 生成实数 FFT 的非负频率轴，共 samples // 2 + 1 个频率点。
            samples, d=1.0 / self.sampling_rate, device=device,  # 使用采样点数和采样间隔确定 Hz 单位的频率坐标，并直接放到目标设备。
        ).to(dtype)  # 让频率轴与输入信号的浮点精度一致。
        edges = self.initial_edges.to(device=device, dtype=dtype)  # 将 [5, 2] 固定频带边界对齐到输入设备和精度。
        response = torch.sigmoid(frequencies - edges[:, 0, None])  # 频率轴与 [5, 1] 下边界广播相减，得到每个频带的平滑高通侧响应。
        response = response * torch.sigmoid(edges[:, 1, None] - frequencies)  # 再乘平滑低通侧响应，形成 [5, 频率点数] 软带通响应。
        energy = (response.square().sum(dim=-1) * (self.sampling_rate / samples)).clamp_min(1e-6)  # 按频率间隔 Δf=采样率/点数近似响应平方积分，并设下限防止后续除零。
        return response / energy.sqrt().unsqueeze(-1)  # 将每带能量平方根扩为 [5, 1] 后广播相除，统一滤波响应的能量尺度。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 将 [批量, 1, 电极, 时间] 输入转换为五带时域信号。
        if x.ndim != 4 or x.shape[1] != 1:  # 验证四维输入且第二维为单一输入特征图。
            raise ValueError("expected input shape (batch, 1, channels, samples)")  # 拒绝维度不符合频带滤波约定的输入。
        samples = x.shape[-1]  # 读取时间维长度，用于频率响应计算和逆变换的长度恢复。
        response = self.frequency_response(samples, device=x.device, dtype=x.dtype)  # 创建与当前输入长度、设备和精度相匹配的五个固定频率响应。
        spectrum = torch.fft.rfft(x.squeeze(1), dim=-1)  # 去除单例特征图维，沿时间做实数 FFT，得到 [批量, 电极, 频率点数] 复数频谱。
        return torch.fft.irfft(  # 对滤波后的各频带频谱执行逆变换，返回实值时域信号。
            spectrum[:, None, :, :] * response[None, :, None, :],  # 将 [批量, 1, 电极, 频率] 与 [1, 5, 1, 频率] 广播相乘，逐频率应用五带响应。
            n=samples, dim=-1,  # 指定原始点数并沿最后一维逆变换，保证输出为 [批量, 5, 电极, 时间]。
        )  # 返回全部频带信号，后续由共享编码器学习可训练特征。


class NPFEEGNetRD(nn.Module):  # 组合去均值原始分支与固定滤波频谱分支，在分类分数层进行残差融合。
    # 类说明：默认返回融合分类分数；辅助模式还返回分支分数与诊断特征，以便计算训练目标和分析模型。
    """Raw-demeaned EEGNet plus a fixed neurophysiological spectral residual.

    Inputs have shape ``(batch, 1, num_channels, n_times)``. The default return
    value is a tensor of class logits. ``return_aux=True`` additionally returns
    branch logits for the objective CE(fused) + 0.5 CE(raw) + 0.25 CE(spectral).
    Only data dimensions and sampling rate vary; all architecture settings are
    the fixed settings of the paper.
    """

    def __init__(self, num_channels: int, classes: int, n_times: int = 1000,  # 接收 EEG 电极数、类别数和每试次点数，默认 1000 点。
                 sampling_rate: float = 250.0):  # 接收采样率，默认 250 Hz 对应 4 秒的默认试次长度。
        super().__init__()  # 初始化模块容器，以便自动注册子网络和可训练参数。
        for name, value, minimum in (("num_channels", num_channels, 1),  # 为电极数配置至少为 1 的整数检查。
                                     ("classes", classes, 2), ("n_times", n_times, 32)):  # 类别数至少为 2，时间点数至少为 32，以支持两次时间池化。
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:  # 排除布尔值、非整数和低于各自最小值的维度参数。
                raise ValueError(f"{name} must be an integer of at least {minimum}")  # 指明哪个输入维度不合法及其最低要求。
        self.num_channels = num_channels  # 保存期望电极数，供空间卷积构造与输入形状检查使用。
        self.classes = classes  # 保存分类类别数，两个分支输出相同数量的分类分数。
        self.n_times = n_times  # 保存固定试次长度，决定原始分支展平后的特征维度。
        self.sampling_rate = float(sampling_rate)  # 保存浮点采样率，记录当前模型对应的数据采样设置。
        self.num_bands = 5  # 固定频带数量为 5，与预定义边界以及等权聚合保持一致。

        self.raw_features = eegnet_features(num_channels, temporal_kernel=64, dropout=0.5)  # 创建原始分支的独立 EEGNet 参数，首层时间核为 64，训练丢弃率为 0.5。
        self.raw_classifier = nn.Linear(16 * (n_times // 4 // 8), classes)  # 将 16 张池化后特征图展平后的向量映射为各类别的原始分支 logits。
        self.band_features = eegnet_features(num_channels, temporal_kernel=32, dropout=0.3)  # 创建五个频带共同使用的一套 EEGNet 参数，首层时间核为 32，丢弃率为 0.3。
        self.band_pool = nn.AdaptiveAvgPool2d((1, 8))  # 将每带的 16 张特征图统一池化为 [16, 1, 8]，展平后固定为 128 维。
        self.band_projection = nn.Sequential(  # 创建五带共享的嵌入投影层，统一各频带特征的表达维度。
            nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3),  # 依次归一化 128 维向量、映射为 64 维、执行 GELU 激活并在训练时丢弃 30% 元素。
        )  # 完成共享投影网络，其线性层和归一化层参数参与训练。
        self.filter_bank = FixedResponseFilterBank(sampling_rate)  # 根据采样率创建固定五带滤波器，频带边界和响应不包含可训练参数。
        self.spectral_classifier = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, classes))  # 对聚合后的 64 维频谱嵌入归一化，再映射为频谱分支 logits。
        nn.init.normal_(self.spectral_classifier[-1].weight, std=0.01)  # 用标准差 0.01 的零均值正态分布初始化频谱分类头权重。
        nn.init.zeros_(self.spectral_classifier[-1].bias)  # 将频谱分类头偏置初始化为零。
        self.residual_strength_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.4)))  # 注册可训练标量 a，使 0.5×sigmoid(a) 的初始值为 0.1。

    def forward(self, x: torch.Tensor, return_aux: bool = False):  # 计算分类分数，return_aux 控制是否附带分支结果和分析特征。
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, self.num_channels, self.n_times):  # 检查输入为 [批量, 1, 指定电极数, 指定时间点数]，批量大小可变。
            raise ValueError(  # 输入结构不符合当前模型配置时，抛出清晰的形状错误。
                f"expected input shape (batch, 1, {self.num_channels}, {self.n_times})"  # 在错误消息中展示本模型实例要求的电极数和试次长度。
            )  # 结束形状错误构造，防止不兼容输入继续进入卷积层。
        raw_input = x - x.mean(dim=-1, keepdim=True)  # 对每个试次和电极沿时间求均值，再通过 [批量, 1, 电极, 1] 广播相减，仅影响原始分支输入。
        raw_embedding = self.raw_features(raw_input).flatten(start_dim=1)  # 提取原始分支特征并保留批量维，将其余维度展平为分类向量。
        raw_logits = self.raw_classifier(raw_embedding)  # 生成 [批量, 类别数] 原始分支分类分数，此处尚未做 softmax。

        filtered = self.filter_bank(x)  # 对未经原始分支去均值的输入滤波，得到 [批量, 5, 电极, 时间] 五带信号。
        batch, bands, channels, samples = filtered.shape  # 解包批量、频带、电极和时间维，供合并与恢复维度时使用。
        feature_maps = self.band_features(filtered.reshape(batch * bands, 1, channels, samples))  # 将批量与频带合并为批次维，让五带共享同一套编码器并并行提取特征。
        band_vectors = self.band_pool(feature_maps).flatten(start_dim=1)  # 池化并展平每带特征，得到 [批量×5, 128] 向量。
        band_embeddings = self.band_projection(band_vectors).reshape(batch, bands, 64)  # 将各带投影为 64 维，再恢复 [批量, 5, 64] 嵌入布局。
        band_weights = filtered.new_full((batch, bands), 1.0 / self.num_bands)  # 在相同设备和精度下生成固定的每带 1/5 权重，这些权重不参与训练。
        spectral_embedding = torch.sum(band_weights.unsqueeze(-1) * band_embeddings, dim=1)  # 把权重扩为 [批量, 5, 1] 并广播相乘，沿频带求和得到 [批量, 64] 等权平均嵌入。
        spectral_logits = self.spectral_classifier(spectral_embedding)  # 对平均后的嵌入分类，产生 [批量, 类别数] 频谱分支分数。
        residual_strength = 0.5 * torch.sigmoid(self.residual_strength_logit)  # 将可训练标量映射为理论上介于 0 与 0.5 的融合系数，系数约束不限制频谱分数本身的幅值。
        logits = raw_logits + residual_strength * spectral_logits  # 计算 o=r+αs，在分类分数层把频谱分支的加权输出加到原始分支输出。
        if not return_aux:  # 默认推理接口只需要融合后的类别分数。
            return logits  # 返回 [批量, 类别数] logits，供交叉熵损失或外部预测逻辑使用。
        return logits, {  # 辅助模式同时返回融合结果和诊断字典，分支分数可用于额外的监督损失。
            "raw_logits": raw_logits,  # 保留原始分支 logits，支持独立计算原始分支交叉熵。
            "spectral_logits": spectral_logits,  # 保留频谱分支 logits，支持独立计算频谱分支交叉熵。
            "band_weights": band_weights,  # 输出 [批量, 5] 固定等权系数，明确当前聚合没有自适应频带加权。
            "band_features": band_embeddings,  # 输出 [批量, 5, 64] 各带嵌入，用于检查频带表征。
            "effective_band_edges": self.filter_bank.effective_edges(),  # 输出 [5, 2] 实际固定频带边界，便于记录和核验。
            "fusion_strength": residual_strength.expand(batch, 1),  # 将全模型共享的融合标量扩展为 [批量, 1] 视图，方便逐样本汇总。
            "embedding": torch.cat((raw_embedding, spectral_embedding), dim=-1),  # 沿特征维拼接原始与频谱嵌入供分析使用，该拼接不参与上面的分类融合。
        }  # 结束辅助信息字典，保持调用方使用的返回键名与张量内容不变。
