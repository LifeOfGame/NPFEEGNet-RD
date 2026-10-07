# 用合成输入检验当前固定模型的形状、数学定义、梯度和权重加载行为。
"""Checks of the fixed model's defining mathematical properties."""

import unittest  # 使用 Python 标准库的单元测试框架。

import torch  # 构造张量、执行模型并检查自动求导结果。
import torch.nn.functional as F  # 调用函数形式的交叉熵损失。

from model import NPFEEGNetRD  # 导入当前模型作为测试对象。
from model.npf_eegnet_rd import FIXED_BANDS, FixedResponseFilterBank  # 导入固定频带定义和滤波器组。


class ModelProperties(unittest.TestCase):  # 汇总固定架构的数学性质测试。
    @classmethod  # 在整个测试类开始前运行一次公共设置。
    def setUpClass(cls):  # 配置本组测试共享的执行环境。
        torch.set_num_threads(2)  # 限制 CPU 计算线程，减少小张量测试的调度开销。

    def test_shapes_auxiliary_loss_and_gradient(self):  # 检查多种输入维度下的输出、融合公式和反向传播。
        for channels, classes in ((3, 2), (20, 2), (22, 4)):  # 覆盖三种通道数以及二分类、四分类设置。
            with self.subTest(channels=channels, classes=classes):  # 失败时标出对应的数据维度。
                model = NPFEEGNetRD(channels, classes, n_times=128)  # 使用较短的 128 点试次缩短测试时间。
                x = torch.randn(2, 1, channels, 128, requires_grad=True)  # 构造两条试次，并开启输入梯度跟踪。
                logits, aux = model(x, return_aux=True)  # 同时获取融合得分和分支辅助信息。
                self.assertEqual(logits.shape, (2, classes))  # 每条试次必须得到一个长度为类别数的得分向量。
                self.assertEqual(aux["band_features"].shape, (2, 5, 64))  # 每条试次包含五个频带的 64 维嵌入。
                torch.testing.assert_close(aux["band_weights"], torch.full((2, 5), 0.2), rtol=0, atol=0)  # 五频带的权重必须严格均为 1/5。
                torch.testing.assert_close(  # 精确核对最终得分是否满足残差融合公式。
                    logits, aux["raw_logits"] + aux["fusion_strength"] * aux["spectral_logits"],  # 期望输出等于原始分支得分加上加权频谱得分。
                    rtol=0, atol=0,  # 不允许相对误差或绝对误差。
                )  # 完成融合公式断言。
                torch.testing.assert_close(aux["fusion_strength"], torch.full((2, 1), 0.1))  # 初始融合系数应为 0.1，按试次展开。
                targets = torch.tensor([0, 1])  # 为两条合成试次指定合法的零起始类别。
                loss = F.cross_entropy(logits, targets)  # 计算最终融合输出的主损失。
                loss = loss + 0.5 * F.cross_entropy(aux["raw_logits"], targets)  # 加上权重为 0.5 的原始分支监督。
                loss = loss + 0.25 * F.cross_entropy(aux["spectral_logits"], targets)  # 加上权重为 0.25 的频谱分支监督。
                loss.backward()  # 通过完整计算图反向传播。
                self.assertTrue(torch.isfinite(x.grad).all())  # 输入梯度不得出现 NaN 或无穷值。
                for name, parameter in model.named_parameters():  # 遍历所有可训练参数并保留参数名便于定位失败。
                    self.assertIsNotNone(parameter.grad, name)  # 每个参数都应参与损失计算并获得梯度。
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)  # 参数梯度也必须全部有限。

    def test_response_energy_and_fixed_edges(self):  # 核对频带边界固定以及离散频率响应能量归一化。
        bank = FixedResponseFilterBank()  # 使用默认采样率 250 Hz 创建五频带滤波器组。
        self.assertEqual(list(bank.parameters()), [])  # 滤波器组自身不应包含任何可训练参数。
        torch.testing.assert_close(bank.effective_edges(), torch.tensor(FIXED_BANDS))  # 有效边界必须对应预定义的五个频段。
        for samples in (128, 1000, 1001):  # 覆盖不同长度以及偶数、奇数时间点数。
            response = bank.frequency_response(samples, device="cpu", dtype=torch.float64)  # 用双精度计算每个频带的响应。
            energy = response.square().sum(-1) * 250.0 / samples  # 按频率间隔 Δf=250/T 对响应平方求离散积分。
            torch.testing.assert_close(energy, torch.ones(5, dtype=torch.float64), atol=1e-12, rtol=1e-12)  # 五个频带的响应能量都应接近 1。

    def test_only_raw_branch_receives_demeaned_input(self):  # 检查去均值处理是否仅作用于原始分支。
        model = NPFEEGNetRD(3, 2, n_times=128).eval()  # 切换推理模式，关闭 Dropout 并使用保存的归一化统计量。
        x = torch.randn(2, 1, 3, 128) + torch.tensor([10.0, -20.0, 30.0]).view(1, 1, 3, 1)  # 给三个通道分别添加明显的时间常量偏移。
        received = {}  # 保存两个分支真正收到的输入副本。
        handles = [  # 注册前向前钩子，以观测模块输入而不修改计算。
            model.raw_features.register_forward_pre_hook(  # 捕获原始 EEGNet 编码器收到的信号。
                lambda module, args: received.update(raw=args[0].detach().clone())),  # 脱离计算图并复制，避免后续操作影响记录。
            model.filter_bank.register_forward_pre_hook(  # 捕获频谱滤波器组收到的信号。
                lambda module, args: received.update(spectral=args[0].detach().clone())),  # 独立保存频谱分支的实际输入。
        ]  # 留存钩子句柄，供测试结束时移除。
        try:  # 即使模型调用抛错，也要执行后续清理。
            with torch.no_grad():  # 输入观察测试不需要构建梯度图。
                model(x)  # 执行一次前向计算，触发两个输入钩子。
        finally:  # 确保已注册钩子不会残留。
            for handle in handles:  # 逐个访问钩子句柄。
                handle.remove()  # 从模块上移除对应钩子。
        torch.testing.assert_close(received["raw"], x - x.mean(-1, keepdim=True), rtol=0, atol=0)  # 原始分支输入必须精确等于逐试次逐通道去时间均值的信号。
        torch.testing.assert_close(received["spectral"], x, rtol=0, atol=0)  # 频谱分支输入必须精确保留原始信号。

    def test_state_roundtrip_preserves_prediction(self):  # 验证状态字典复制到同构模型后预测完全一致。
        original = NPFEEGNetRD(20, 2).eval()  # 创建采用 OpenBMI 输入维度的原模型。
        restored = NPFEEGNetRD(20, 2).eval()  # 独立初始化用于加载权重的目标模型。
        restored.load_state_dict(original.state_dict(), strict=True)  # 严格匹配并加载全部参数与缓冲区。
        x = torch.randn(2, 1, 20, 1000)  # 构造两条 20 通道、1000 时间点的合成试次。
        with torch.no_grad():  # 仅比较前向预测，关闭梯度记录。
            torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)  # 同权重模型的 logits 应逐元素完全相同。

    def test_invalid_shapes_and_changed_frequency_state_are_rejected(self):  # 核对输入形状与固定频带状态的保护检查。
        model = NPFEEGNetRD(3, 2, n_times=128)  # 模型声明只接收三通道输入。
        with self.assertRaises(ValueError):  # 期望错误维度触发明确异常。
            model(torch.zeros(2, 1, 4, 128))  # 故意传入四通道数据，验证形状检查。
        state = model.state_dict()  # 获取完整权重与缓冲区字典。
        state["filter_bank.edge_offsets"] = torch.ones(5, 2)  # 故意把本应为零的频带偏移缓冲区改为 1。
        with self.assertRaisesRegex(RuntimeError, "fixed NPFEEGNet-RD bands"):  # 期望模型拒绝与固定频带定义不符的状态。
            model.load_state_dict(state, strict=True)  # 尝试严格加载被篡改的状态以触发检查。


if __name__ == "__main__":  # 支持直接运行此测试文件。
    unittest.main()  # 由标准测试运行器发现并执行测试方法。
