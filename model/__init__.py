# 包说明：向调用方提供用于运动想象脑电分类的当前模型。
"""NPFEEGNet-RD for motor-imagery EEG classification."""

from .npf_eegnet_rd import NPFEEGNetRD  # 从同一模型包中导入可直接实例化的 NPFEEGNetRD 类。

__all__ = ["NPFEEGNetRD"]  # 指定星号导入时公开的名称，统一对外使用当前模型类。
