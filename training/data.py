# 本模块读取已完成预处理的 EEG 数组，并检查数据形状、标签与来源。
"""Small, explicit readers for already-preprocessed EEG arrays."""

from dataclasses import dataclass  # 用数据类集中保存 EEG 数据及其来源信息。
import hashlib  # 计算输入文件的 SHA-256，用于核对实验数据是否发生变化。
from pathlib import Path  # 以跨平台路径对象定位并读取输入文件。

import numpy as np  # 处理 EEG、标签和分组数组，并执行数值检查。


TRIAL_FINGERPRINT_SCHEMA = "eeg-float32-le-shape-sha256-v1"  # 固定逐试次指纹格式，使不同容器、精度和字节序的同一 EEG 输入可比较。


def file_sha256(path):  # 计算文件内容的 SHA-256 摘要，供数据与源码溯源使用。
    digest = hashlib.sha256()  # 创建初始哈希状态。
    with Path(path).open("rb") as stream:  # 按二进制读取，避免文本编码或换行转换影响摘要。
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # 每次读取 1 MiB，遇到空字节串即到达文件末尾。
            digest.update(block)  # 将当前数据块累积到哈希状态中，无需一次性载入整个文件。
    return digest.hexdigest()  # 返回十六进制摘要，便于写入实验记录并进行精确比较。


def trial_fingerprints(data):  # 对模型实际接收的每条 EEG 生成独立内容指纹，不使用标签、分组或文件名。
    fingerprints = []  # 按原试次顺序保存摘要，既支持重叠检查，也能核验最终训练输入顺序。
    for trial in data.X:  # 逐试次处理，避免为了计算摘要而复制整份 EEG 数据。
        normalized = np.array(trial, dtype="<f4", order="C", copy=True)  # 统一为连续的小端 float32，消除原始存储精度和内存布局差异。
        normalized[normalized == 0] = 0.0  # 将数值相同的正零与负零编码统一；只修改当前临时副本。
        digest = hashlib.sha256(TRIAL_FINGERPRINT_SCHEMA.encode("ascii"))  # 将格式标识纳入摘要，避免不同规范之间误比较。
        digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())  # 包含单试次形状，防止相同字节被解释为不同通道或时间布局。
        digest.update(normalized.tobytes(order="C"))  # 按固定行优先顺序加入 EEG 数值，不受 NPZ 压缩或文件重命名影响。
        fingerprints.append(digest.hexdigest())  # 留存当前试次的 SHA-256，以检测整包、重排序和部分样本重复。
    return fingerprints  # 返回与输入试次一一对应的摘要列表。


def assert_disjoint_trials(left, right, left_name, right_name):  # 验证两个用途不同的数据集合没有完全相同的 EEG 试次。
    overlap = set(left).intersection(right)  # 按内容而非试次位置匹配，覆盖复制、硬链接、重新打包及部分重复。
    if overlap:  # 任意一个完全相同的试次跨越训练、验证或测试边界，都应在评估前拒绝。
        raise ValueError(f"{left_name} and {right_name} EEG content overlaps: {len(overlap)} identical trial(s).")  # 报告冲突的数据角色和重复内容数量。


@dataclass  # 自动生成初始化方法，统一管理数组及其元数据。
class EEGData:  # 表示一个会话或其子集，不混合其他会话的数据。
    X: np.ndarray  # EEG 输入统一为 [试次, 1, 通道, 时间点] 的浮点数组。
    y: np.ndarray | None  # 每个试次的整数类别标签；仅推理时可以没有标签。
    groups: np.ndarray | None  # 每个试次所属的组，用于训练会话内部的留组验证。
    files: dict  # 保存已读取文件的绝对路径及 SHA-256 摘要。
    label_offset: int = 0  # 记录读取时减去的标签偏移，确保选择和最终训练编码一致。

    def subset(self, indices):  # 按同一组试次索引提取 EEG、标签和分组，保持它们一一对应。
        return EEGData(self.X[indices], None if self.y is None else self.y[indices],  # 切出输入与可选标签，标签不存在时继续保留空值。
                       None if self.groups is None else self.groups[indices], self.files, self.label_offset)  # 同步截取组编号，并继承源文件摘要和标签偏移。


def load_data(path, labels=None, groups=None, *, require_labels=True, label_offset=0):  # 读取单个会话的数据；标签和分组仅在需要标签时载入。
    # 支持包含 X、y、可选 groups 的 NPZ，或带同会话配套文件的 NPY；推理不读取独立标签或分组文件。
    """Read NPZ (X, y, optional groups) or one *_data.npy session.

    NPY labels/groups can be supplied explicitly; otherwise their same-session
    *_label.npy and *_group.npy siblings are used. No other session is opened.
    Inference does not open separate label/group files.
    """
    path = Path(path).resolve(strict=True)  # 转为绝对路径，并立即拒绝不存在的输入文件。
    files = {str(path): file_sha256(path)}  # 记录 EEG 文件的内容指纹，供后续协议核验。
    y = group_values = None  # 默认不读取标签或分组，适用于只输入 EEG 的推理场景。
    if path.suffix.lower() == ".npz":  # NPZ 使用单个压缩包存放同一会话的多个数组。
        if labels is not None or groups is not None:  # NPZ 内部已约定标签和分组字段，不允许再混入独立文件。
            raise ValueError("NPZ inputs already contain labels/groups; omit separate files.")  # 报告重复或混合的数据来源配置。
        with np.load(path, allow_pickle=False) as arrays:  # 禁止反序列化 Python 对象，按普通数值数组读取。
            unexpected = set(arrays.files) - {"X", "y", "groups"}  # 仅接受当前会话的输入、标签和可选分组字段。
            if unexpected:  # 额外字段可能包含其他会话，因此直接拒绝。
                raise ValueError(f"NPZ must contain one session only: unexpected keys {unexpected}")  # 明确列出不受支持的字段。
            X = arrays["X"]  # 读取 EEG 试次数组，稍后统一到四维形状。
            if require_labels:  # 训练或评估需要标签，纯推理则跳过这些数组。
                y = arrays["y"]  # 读取每个试次的类别标签。
                group_values = arrays["groups"] if "groups" in arrays else None  # 存在分组字段时同时读取，缺省保持为空。
    elif path.suffix.lower() == ".npy":  # NPY 文件本身只保存 EEG 数组，配套信息来自独立文件。
        X = np.load(path, allow_pickle=False)  # 读取 EEG 数值数组，禁止加载对象型序列化内容。
        if require_labels:  # 只有需要标签时才访问独立标签与分组文件。
            prefix = str(path)[:-len("_data.npy")] if path.name.endswith("_data.npy") else None  # 从约定文件名提取同会话前缀。
            if labels is None and prefix is None:  # 没有显式标签路径，也无法按命名规则推断标签路径。
                raise ValueError("Supply --*-labels or use a *_data.npy filename.")  # 要求提供明确的同会话标签来源。
            label_path = Path(labels if labels is not None else prefix + "_label.npy").resolve(strict=True)  # 优先使用显式标签路径，否则使用同前缀标签文件。
            y = np.load(label_path, allow_pickle=False)  # 读取类别标签，不允许对象数组。
            files[str(label_path)] = file_sha256(label_path)  # 将标签文件指纹纳入训练数据身份记录。
            group_path = Path(groups) if groups is not None else (  # 优先使用显式分组文件，否则尝试同会话默认位置。
                Path(prefix + "_group.npy") if prefix is not None else None)  # 无法推断会话前缀时不自动寻找分组。
            if group_path is not None and (groups is not None or group_path.exists()):  # 显式分组必须存在；隐式分组仅在文件存在时读取。
                group_path = group_path.resolve(strict=True)  # 固定分组文件的绝对路径并校验存在性。
                group_values = np.load(group_path, allow_pickle=False)  # 读取各试次的组标识，供留组验证使用。
                files[str(group_path)] = file_sha256(group_path)  # 分组划分影响验证协议，因此同样记录内容摘要。
    else:  # 输入文件扩展名不属于支持的数组格式。
        raise ValueError("Input must be one .npz or .npy file.")  # 阻止不明确的格式被当作 EEG 数据处理。
    if X.ndim == 3:  # 三维输入默认按 [试次, 通道, 时间点] 解释。
        X = X[:, None, :, :]  # 插入单例特征图维度，得到 [试次, 1, 通道, 时间点]。
    if X.ndim != 4 or X.shape[1] != 1 or any(size < 1 for size in X.shape):  # 检查输入为非空四维张量，且特征图维度恰为 1。
        raise ValueError("EEG must have shape (trials, channels, time) or (trials, 1, channels, time).")  # 报告模型支持的两种输入形状。
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X) or not np.isfinite(X).all():  # 拒绝非数值、复数以及 NaN 或无穷大信号。
        raise ValueError("EEG must contain finite real numbers.")  # 确保 EEG 可用于实数神经网络运算。
    X = np.ascontiguousarray(X, dtype=np.float32)  # 将输入统一为连续内存的单精度数组，便于转换为模型张量。
    if not np.isfinite(X).all():  # 转换到单精度后再次检查，捕获过大数值产生的溢出。
        raise ValueError("EEG exceeds the finite float32 range.")  # 拒绝超出单精度表示范围的输入。
    if y is not None:  # 仅对实际读取到的标签执行编码与长度校验。
        y = np.asarray(y).reshape(-1)  # 将标签展平为 [试次]，兼容行向量或列向量存储。
        if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):  # 每个试次必须对应一个非负整数标签。
            raise ValueError("Labels must be one nonnegative integer per trial, encoded from zero.")  # 报告标签形状或类型不满足要求。
        y = np.ascontiguousarray(y, dtype=np.int64)  # 使用交叉熵损失所需的连续 64 位整数标签。
        if label_offset not in (0, 1):  # 只允许保持零起始标签，或把一起始标签减去 1。
            raise ValueError("label_offset must be 0 or 1.")  # 拒绝未定义的标签重编号方式。
        y = y - label_offset  # 将标签转换到模型约定的零起始类别编号。
        if np.any(y < 0):  # 检查用户指定的偏移是否造成负类别编号。
            raise ValueError("Labels are below the requested label offset.")  # 发现偏移与原始标签编码不匹配时终止读取。
    if group_values is not None:  # 有分组信息时校验其与试次数量及允许类型的一致性。
        group_values = np.asarray(group_values).reshape(-1)  # 将组标识统一展平为 [试次]。
        if group_values.shape != (len(X),) or group_values.dtype.kind not in "iuUS":  # 组标识必须逐试次对应，且为整数、Unicode 字符串或字节字符串。
            raise ValueError("Groups must be one integer or string per trial.")  # 拒绝无法安全用于分组划分的数据。
    return EEGData(X, y, group_values, files, label_offset)  # 返回已校验数组及其文件指纹和标签编码信息。


def validate_classes(data, classes, *, require_all=False):  # 对已读取标签核对模型类别范围，并可要求覆盖全部类别。
    if data.y is None or np.any(data.y >= classes):  # 标签不能为空，也不能出现超出类别总数的编号。
        raise ValueError("Labels are missing or outside the configured class range.")  # 阻止缺失标签或类别维度不一致的数据进入训练评估。
    if require_all and not np.array_equal(np.unique(data.y), np.arange(classes)):  # 训练集合要求恰好包含从 0 到类别数减 1 的全部类别。
        raise ValueError("Training labels must contain every class, encoded 0 through classes-1.")  # 避免某个训练折缺少类别却继续拟合。


def assert_shape(data, model_kwargs):  # 校验每个试次的特征维度与模型构造参数完全一致。
    expected = (1, model_kwargs["num_channels"], model_kwargs["n_times"])  # 单试次应具有 [1, 通道数, 时间点数] 形状。
    if data.X.shape[1:] != expected:  # 忽略可变的试次数量，仅比较模型实际接收的后三维。
        raise ValueError(f"EEG shape {data.X.shape[1:]} does not match model shape {expected}.")  # 明确报告数据形状与模型期望的差异。


def group_folds(data):  # 在同一训练会话内按组生成训练和验证索引，避免拆散组内样本。
    # 每个完整组依次作为验证组，其余组用于该折训练。
    """Leave each complete training-session group out once."""
    if data.groups is None or len(np.unique(data.groups)) < 2:  # 留组验证至少需要两个不同组，保证训练与验证都非空。
        raise ValueError("Group selection requires at least two groups in the training session.")  # 缺少有效分组时要求先补充训练会话的组信息。
    for group in np.unique(data.groups):  # 逐个遍历不同的组，确保每组恰好验证一次。
        yield str(group), np.flatnonzero(data.groups != group), np.flatnonzero(data.groups == group)  # 输出组名、其他组的训练索引和当前组的验证索引。
