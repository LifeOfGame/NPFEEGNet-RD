# 本模块实现有版本标识的训练流程，只使用验证数据选择训练轮数。
"""Versioned training recipe with validation-only epoch selection."""

from dataclasses import asdict, dataclass  # 定义不可变训练配置，并将配置转成可记录的字典。
from datetime import datetime, timezone  # 生成带 UTC 时区的实验创建时间。
import json  # 读取选择记录，并写入配置、指标和实验协议文件。
from pathlib import Path  # 管理数据、源码和实验输出的文件路径。
import platform  # 记录 Python 版本，便于复核实验环境。
import random  # 固定 Python 标准随机数生成器的种子。

import numpy as np  # 处理验证指标、概率数组和折间轮数汇总。
import torch  # 构造与训练模型，执行张量推理并保存权重。
from torch.nn import functional as F  # 使用交叉熵计算融合输出与两个分支的训练损失。
from torch.utils.data import DataLoader, TensorDataset  # 将 EEG 与标签打包成可复现的批次迭代器。

from model.npf_eegnet_rd import NPFEEGNetRD  # 引入当前 NPFEEGNet-RD 模型实现。
from .data import assert_shape, file_sha256, group_folds, validate_classes  # 复用输入维度、文件摘要、留组划分和类别校验。
from .data import TRIAL_FINGERPRINT_SCHEMA, assert_disjoint_trials, trial_fingerprints  # 按规范化 EEG 内容检查隔离，并保存可跨文件比较的逐试次指纹。


PROTOCOL = "npfeegnet-rd-clean-v1"  # 标识当前训练协议，防止不同流程的选择记录或权重混用。


@dataclass(frozen=True)  # 创建初始化后不可修改的配置对象，避免训练中途无记录地改变参数。
class TrainConfig:  # 集中定义选择阶段与最终训练共用的超参数。
    seed: int = 0  # 固定随机种子，控制初始化、随机失活及训练批次顺序。
    batch_size: int = 64  # 每个训练或推理批次最多包含 64 个试次。
    learning_rate: float = 1e-3  # Adam 优化器的初始学习率。
    weight_decay: float = 1e-4  # Adam 优化器使用的权重衰减系数。
    max_epochs: int = 300  # 选择阶段的最大轮数，也是两个阶段余弦调度的共同周期长度。
    patience: int = 40  # 验证标准连续多少轮未改善后停止该折选择。
    min_learning_rate: float = 1e-6  # 余弦学习率调度的最小学习率。

    def validate(self):  # 在启动训练前检查随机种子、训练长度和优化器参数是否合法。
        if self.seed < 0 or self.seed >= 2**32:  # 限制种子位于 NumPy 支持的无符号 32 位整数范围。
            raise ValueError("Seed must be between 0 and 2**32-1.")  # 拒绝超出各随机数生成器共同支持范围的种子。
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:  # 批大小、最大轮数和早停等待轮数必须为正。
            raise ValueError("Batch size, maximum epochs and patience must be positive.")  # 避免空批次、零轮训练或无意义的早停配置。
        values = [self.learning_rate, self.weight_decay, self.min_learning_rate]  # 汇总需要检查有限性的优化参数。
        if not all(np.isfinite(value) for value in values):  # 拒绝 NaN 和无穷大，避免优化过程直接产生无效数值。
            raise ValueError("Optimizer settings must be finite.")  # 报告不可用的优化器数值配置。
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.min_learning_rate <= self.learning_rate:  # 检查初始学习率为正、衰减非负，且最小学习率不超过初始值。
            raise ValueError("Invalid optimizer learning rate or weight decay.")  # 拒绝范围关系不合理的优化超参数。


def write_json(path, value):  # 以统一格式写入实验记录，便于人工阅读与版本比较。
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")  # 使用 UTF-8、排序键和缩进，并禁止写入不符合 JSON 标准的 NaN。


def new_output(path):  # 创建本次实验专属的输出目录。
    path = Path(path).resolve()  # 固定输出的绝对路径，便于记录和复核。
    path.mkdir(parents=True, exist_ok=False)  # 按需创建父目录；目标已存在时拒绝覆盖已有实验。
    return path  # 返回后续保存配置、权重和结果使用的目录对象。


def source_hashes():  # 对参与训练和推理的源码逐文件计算指纹。
    root = Path(__file__).resolve().parents[1]  # 由本文件位置确定项目根目录。
    paths = [root / "train.py", root / "predict.py"]  # 将训练入口与推理入口纳入源码身份检查。
    paths += sorted((root / "training").glob("*.py"))  # 纳入训练包中的全部 Python 源文件，并固定遍历顺序。
    paths += sorted((root / "model").glob("*.py"))  # 纳入模型包中的全部 Python 源文件。
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}  # 用相对路径记录内容摘要；注释变化也会改变文件哈希。


def manifest(stage, config, model_kwargs, device):  # 构造包含流程版本、模型参数、源码和环境的实验清单。
    return {  # 返回可直接序列化为 JSON 的实验元数据字典。
        "protocol": PROTOCOL, "stage": stage,  # 同时标识协议版本与当前属于选择还是最终训练阶段。
        "created_utc": datetime.now(timezone.utc).isoformat(),  # 使用带时区的 UTC 时间记录实验创建时刻。
        "model": "NPFEEGNet-RD", "model_kwargs": model_kwargs,  # 保存模型名称及输入维度等构造参数。
        "config": asdict(config), "device": str(device),  # 记录训练超参数与实际运行设备。
        "loss": {"fused_ce": 1.0, "raw_ce": 0.5, "spectral_ce": 0.25},  # 明确融合、原始分支和频谱分支交叉熵的权重。
        "scheduler": "CosineAnnealingLR; T_max=max_epochs in selection and final training",  # 声明两个训练阶段使用同样的余弦调度周期定义。
        "engine_scope": "Current implementation for new experiments; historical frozen results are not rerun or certified by this engine.",  # 限定当前引擎服务于新实验，不据此追认历史冻结结果。
        "source_sha256": source_hashes(),  # 保存全部相关源码指纹，确保选择和重新训练使用同一代码版本。
        "environment": {"python": platform.python_version(), "numpy": str(np.__version__),  # 记录 Python 与 NumPy 版本，帮助复核数值环境。
                        "torch": str(torch.__version__), "cuda": torch.version.cuda},  # 同时记录 PyTorch 及其 CUDA 构建版本。
    }  # 完成实验基础清单，后续阶段再补充数据、轮数和结果信息。


def seed_all(seed):  # 同步固定常用随机数生成器，降低重复运行的随机差异。
    random.seed(seed)  # 固定 Python 标准库的随机数序列。
    np.random.seed(seed)  # 固定 NumPy 全局随机数序列。
    torch.manual_seed(seed)  # 固定 PyTorch 随机状态，控制参数初始化等随机操作。
    if torch.cuda.is_available():  # 仅在当前环境可用 CUDA 时设置 GPU 随机状态。
        torch.cuda.manual_seed_all(seed)  # 对全部可见 CUDA 设备使用相同的种子值。
    torch.backends.cudnn.benchmark = False  # 禁止根据运行时测速自动选择卷积算法，减少算法选择波动。
    torch.backends.cudnn.deterministic = True  # 要求 cuDNN 优先采用确定性算法；不承诺不同硬件之间逐位相同。


def loss_value(logits, aux, labels):  # 计算融合输出和两个分支的联合监督损失。
    return (F.cross_entropy(logits, labels) + 0.5 * F.cross_entropy(aux["raw_logits"], labels)  # 融合分类损失权重为 1，原始分支辅助损失权重为 0.5。
            + 0.25 * F.cross_entropy(aux["spectral_logits"], labels))  # 再加权重为 0.25 的频谱分支辅助损失。


def loader(data, batch_size, *, shuffle, seed):  # 将 [试次, 1, 通道, 时间点] 输入及 [试次] 标签按批次配对。
    dataset = TensorDataset(torch.from_numpy(data.X), torch.from_numpy(data.y))  # 由 NumPy 数组创建张量数据集，确保输入和标签同步索引。
    generator = torch.Generator().manual_seed(seed)  # 使用独立随机生成器固定批次打乱过程。
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,  # 按指定批大小与是否打乱的设置构造迭代器。
                      num_workers=0, drop_last=False)  # 在当前进程读取数据，并保留末尾不足一个整批的试次。


def initialize(model_kwargs, config, device):  # 从固定种子重新构造模型、优化器和学习率调度器。
    seed_all(config.seed)  # 在参数初始化之前固定随机状态，让各折和最终训练遵循相同初始化规则。
    model = NPFEEGNetRD(**model_kwargs).to(device)  # 新建当前模型，并把参数移到目标设备。
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)  # 对所有可训练模型参数使用 Adam 更新。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # 使用余弦曲线按训练轮数降低学习率。
        optimizer, T_max=config.max_epochs, eta_min=config.min_learning_rate)  # 周期始终为最大轮数，最终训练不因所选轮数较短而改变曲线。
    return model, optimizer, scheduler  # 返回同一次初始化得到的三个训练对象。


def train_epoch(model, batches, optimizer, device):  # 遍历全部训练批次，执行一轮参数更新并返回按试次数加权的平均损失。
    model.train()  # 启用训练模式，使随机失活生效并更新批归一化统计量。
    total = count = 0  # 初始化累计损失总和与已经处理的试次数。
    for inputs, labels in batches:  # 逐批读取配对的 EEG 输入与类别标签。
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移到与模型相同的计算设备。
        optimizer.zero_grad(set_to_none=True)  # 清空上一批的梯度，避免跨批次意外累积。
        logits, aux = model(inputs, return_aux=True)  # 前向计算融合 logits，并取得两个分支的辅助输出。
        loss = loss_value(logits, aux, labels)  # 使用融合输出和分支输出共同计算监督损失。
        if not torch.isfinite(loss):  # 在反向传播前检查损失是否为有限数值。
            raise FloatingPointError("Training produced a non-finite loss.")  # 遇到 NaN 或无穷大时终止，避免保存损坏的模型状态。
        loss.backward()  # 根据联合损失计算所有参与训练参数的梯度。
        optimizer.step()  # 使用当前批次梯度更新模型参数。
        total += loss.item() * len(labels)  # 将批平均损失还原为按试次累加的量，避免小尾批被过度加权。
        count += len(labels)  # 累加当前批次实际包含的试次数量。
    return total / count  # 返回本轮每个训练试次的平均联合损失。


@torch.inference_mode()  # 推理过程关闭梯度记录，减少内存占用与计算开销。
def infer(model, X, batch_size, device):  # 按原始试次顺序生成 [试次, 类别数] 的融合 logits。
    model.eval()  # 关闭随机失活，并使用已保存的批归一化统计量。
    parts = []  # 暂存各批次在 CPU 上的输出数组。
    for start in range(0, len(X), batch_size):  # 按批次大小顺序切分输入，不打乱评估顺序。
        inputs = torch.from_numpy(X[start:start + batch_size]).to(device)  # 将当前 EEG 批次转换为目标设备上的张量。
        logits = model(inputs)  # 仅取得用于最终分类的融合输出。
        if not torch.isfinite(logits).all():  # 检查所有试次和类别的输出均为有限数值。
            raise FloatingPointError("Inference produced non-finite logits.")  # 无效输出不进入概率转换或指标计算。
        parts.append(logits.cpu().numpy())  # 将该批输出转回 CPU 数组，以便汇总和保存。
    return np.concatenate(parts)  # 沿试次维拼接各批，保持与输入和标签相同的顺序。


def probabilities(logits):  # 将每个试次的类别 logits 转换为归一化 softmax 概率。
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)  # 使用双精度并减去每行最大值，降低指数运算溢出的风险。
    exp = np.exp(shifted)  # 对平移后的各类别得分求指数，得到非负权重。
    return exp / exp.sum(axis=1, keepdims=True)  # 按类别维归一化，使每个试次的概率和为 1。


def metrics(logits, labels):  # 根据融合输出计算分类准确率、校正一致性及概率损失等指标。
    probs = probabilities(logits)  # 获得每个试次在所有类别上的预测概率。
    prediction = probs.argmax(axis=1)  # 选择概率最大的类别编号作为预测标签。
    classes = logits.shape[1]  # 从输出的第二维取得模型类别总数。
    matrix = np.zeros((classes, classes), dtype=np.int64)  # 创建混淆矩阵，行对应真实类别，列对应预测类别。
    np.add.at(matrix, (labels, prediction), 1)  # 按每个试次的真实与预测类别组合累加计数。
    support = matrix.sum(axis=1)  # 各行总和给出每个真实类别的试次数。
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(classes, dtype=float), where=support > 0)  # 用对角线正确数除以类别样本数，空类别暂记为零以避免除零。
    accuracy = float(np.mean(prediction == labels))  # 计算全部试次中预测正确的比例，取值范围为 0 到 1。
    expected = float(np.dot(support.astype(float), matrix.sum(axis=0).astype(float)) / len(labels)**2)  # 根据真实与预测边际分布计算随机一致率，供 Cohen's kappa 使用。
    return {"trials": len(labels), "accuracy": accuracy,  # 记录试次数量与普通准确率。
            "balanced_accuracy": float(recall[support > 0].mean()),  # 对实际出现的类别召回率取平均，减轻类别数量不均衡的影响。
            "kappa": float((accuracy - expected) / (1 - expected)) if expected < 1 else None,  # 扣除随机一致率；退化到分母为零时记为空值。
            "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-300, 1)).mean()),  # 计算真实类别概率的平均负对数，截断极小概率以避免对零取对数。
            "class_recall": [float(recall[i]) if support[i] else None for i in range(classes)],  # 逐类别报告召回率；无真实样本的类别不虚报为零表现。
            "confusion_matrix": matrix.tolist()}  # 将整数混淆矩阵转为列表，便于 JSON 序列化。


def save_predictions(path, logits, labels=None):  # 保存逐试次输出，支持后续独立复核分类与概率指标。
    payload = {"logits": logits, "probabilities": probabilities(logits),  # 同时保留原始得分和归一化概率，避免只保存汇总准确率。
               "predictions": logits.argmax(axis=1)}  # 保存每个试次得分最高的预测类别编号。
    if labels is not None:  # 评估阶段有真实标签时一并保存，纯推理则省略。
        payload["labels"] = labels  # 记录与预测顺序一致的真实类别。
    np.savez_compressed(path, **payload)  # 将相关数组写入单个压缩 NPZ 文件。


def select_epochs(train, output, config, model_kwargs, *, validation=None, device="cpu"):  # 只从训练与验证数据确定轮数，接口不接收测试数据。
    # 显式验证集与训练会话内的留一组交叉验证二选一，整个选择阶段不访问测试集。
    """Use explicit validation or leave-one-group-out CV; never accept test data."""
    config.validate()  # 首先检查随机种子与训练超参数，避免创建无效实验。
    assert_shape(train, model_kwargs)  # 确认训练试次的通道数和时间点数符合模型输入。
    validate_classes(train, model_kwargs["classes"], require_all=True)  # 训练数据必须包含模型定义的每一种类别。
    training_trials = trial_fingerprints(train)  # 先计算完整训练输入的逐试次指纹，供各折隔离与最终训练核验复用。
    validation_trials = []  # 留组验证的所有样本已包含在训练输入中，只有显式验证需要单独持久化指纹。
    if validation is not None:  # 用户显式提供验证数据时，采用单一训练与验证划分。
        assert_shape(validation, model_kwargs)  # 验证数据必须具有与训练模型相同的单试次形状。
        validate_classes(validation, model_kwargs["classes"])  # 检查验证标签存在且处于模型类别范围内。
        if set(train.files).intersection(validation.files):  # 检查训练与验证记录中是否出现相同文件路径。
            raise ValueError("Training and validation must use separate data files.")  # 阻止同一文件同时充当训练和显式验证来源。
        validation_trials = trial_fingerprints(validation)  # 读取已加载的验证 EEG 内容，不将标签或组编号混入指纹。
        assert_disjoint_trials(training_trials, validation_trials, "Training", "validation")  # 在模型初始化前拒绝不同文件路径下的相同或部分重复 EEG。
        folds = [("validation", train, validation)]  # 将显式验证包装为统一的单折结构。
    else:  # 未提供独立验证集时，只在训练会话内部划分组。
        folds = []  # 按组建立划分，并逐折检查是否有重复试次跨越训练与验证边界。
        for name, train_indices, val_indices in group_folds(train):  # 每次留一个完整组验证，其余组构成该折训练数据。
            assert_disjoint_trials([training_trials[i] for i in train_indices],  # 从完整训练指纹提取当前折的拟合样本，无需再次读取或哈希数据。
                                   [training_trials[i] for i in val_indices], "Training fold", "validation fold")  # 即使组标识不同，也拒绝两个集合中完全相同的 EEG。
            folds.append((name, train.subset(train_indices), train.subset(val_indices)))  # 内容隔离通过后，创建原有训练与验证子集。
    for _, fit_data, _ in folds:  # 在任何训练开始之前检查每一折的拟合数据。
        validate_classes(fit_data, model_kwargs["classes"], require_all=True)  # 确保留组后各训练折仍然覆盖全部类别。
    out = new_output(output)  # 创建新的选择结果目录，禁止覆盖已有输出。
    record = manifest("selection", config, model_kwargs, device)  # 建立选择阶段的版本、源码和环境清单。
    record.update({"training_files": train.files,  # 保存完整训练会话的文件路径与内容摘要。
                   "input_label_offset": train.label_offset,  # 记录标签从输入编码转换为零起始编号时使用的偏移。
                   "validation_files": validation.files if validation is not None else {},  # 显式验证记录其独立文件；留组验证沿用训练文件来源。
                   "trial_fingerprint_schema": TRIAL_FINGERPRINT_SCHEMA,  # 标记指纹规范；缺少该规范的旧选择记录不能直接继续最终训练。
                   "training_trial_sha256": training_trials,  # 按输入顺序保存完整训练会话每条 EEG 的内容指纹。
                   "validation_trial_sha256": validation_trials,  # 持久化显式验证指纹，使最终测试隔离不需要重新打开验证数据。
                   "selection_rule": "Highest validation accuracy, then lowest fused cross-entropy; earliest exact tie. Median fold epoch rounded to nearest integer, ties to even.",  # 固定准确率优先、负对数损失次之、完全并列保留最早轮的规则，并声明折间中位数舍入方式。
                   "fold_epochs": [], "folds": [], "test_accessed": False})  # 初始化各折选择记录，并明确此阶段未访问测试数据。
    write_json(out / "config.json", {"config": asdict(config), "model_kwargs": model_kwargs})  # 在训练前保存本次超参数与模型配置。
    write_json(out / "protocol_manifest.json", record)  # 先写基础协议记录，便于识别中途中断的实验。
    for index, (name, fit_data, val_data) in enumerate(folds):  # 依次完成每个训练与验证划分。
        model, optimizer, scheduler = initialize(model_kwargs, config, device)  # 每折重新固定种子并初始化模型，不沿用上一折权重。
        batches = loader(fit_data, config.batch_size, shuffle=True, seed=config.seed)  # 只对该折训练数据构造可复现的打乱批次。
        best_key = (-1.0, float("-inf"))  # 用低于正常验证指标的初值，确保第一轮有限结果可被选中。
        best_epoch = 0  # 在出现第一份有效验证结果前，尚未选择任何训练轮。
        history = []  # 保存该折逐轮损失、验证指标和学习率。
        fold_out = out / f"fold_{index + 1}"  # 为当前折分配从 1 开始编号的独立目录。
        fold_out.mkdir()  # 创建该折的权重与预测输出目录。
        for epoch in range(1, config.max_epochs + 1):  # 从第 1 轮迭代到最大轮数，期间可能触发早停。
            train_loss = train_epoch(model, batches, optimizer, device)  # 只使用该折训练数据完成一轮参数更新。
            val_logits = infer(model, val_data.X, config.batch_size, device)  # 用当前权重预测验证数据，过程不更新参数。
            score = metrics(val_logits, val_data.y)  # 计算用于轮数选择的验证准确率与概率损失。
            history.append({"epoch": epoch, "train_loss": train_loss,  # 记录当前轮编号与训练集平均联合损失。
                            "validation_accuracy": score["accuracy"], "validation_nll": score["nll"],  # 记录验证准确率和融合输出的平均负对数损失。
                            "learning_rate": optimizer.param_groups[0]["lr"]})  # 保存本轮实际使用的学习率，随后才推进调度器。
            key = (score["accuracy"], -score["nll"])  # 元组先比较准确率，准确率相同时再偏好较小的验证负对数损失。
            if key > best_key:  # 仅在严格改善时替换最佳轮，完全相同的指标保留最早轮。
                best_key, best_epoch = key, epoch  # 更新当前折最佳验证标准及对应训练轮数。
                torch.save({"protocol": PROTOCOL, "model_kwargs": model_kwargs,  # 将协议标识与模型结构参数一同写入验证最佳检查点。
                            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},  # 脱离梯度图并转为 CPU 保存模型参数及缓冲区。
                            "config": asdict(config), "selected_epoch": epoch, "stage": "selection"},  # 标明这份权重来自选择阶段及其对应轮数。
                           fold_out / "best.pt")  # 当前折只保留最新发现的最佳验证权重。
                save_predictions(fold_out / "validation_predictions.npz", val_logits, val_data.y)  # 保存所选轮的逐试次验证输出与真实标签。
                write_json(fold_out / "validation_metrics.json", score)  # 保存与最佳权重和预测文件相匹配的验证指标。
            scheduler.step()  # 每完成一轮训练后推进一次余弦学习率调度。
            if epoch - best_epoch >= config.patience:  # 若连续等待指定轮数仍无严格改善，停止当前折的选择。
                break  # 保留已找到的最佳轮，跳出本折训练循环。
        write_json(fold_out / "history.json", history)  # 保存当前折从开始到停止的完整逐轮记录。
        record["fold_epochs"].append(best_epoch)  # 收集各折最佳轮数，供最终汇总所需的训练长度。
        record["folds"].append({"group": name, "train_trials": len(fit_data.X),  # 记录该折名称与实际用于拟合的试次数。
                                "validation_trials": len(val_data.X), "selected_epoch": best_epoch})  # 同时记录验证规模和所选轮数，便于复核划分。
        print(f"Fold {index + 1}/{len(folds)} selected epoch {best_epoch}", flush=True)  # 即时报告当前折的选择进展。
    record["selected_epoch"] = int(np.rint(np.median(record["fold_epochs"])))  # 各折最佳轮数取中位数，再四舍六入五成双到最近整数。
    record["status"] = "complete"  # 仅在所有折完成后将选择实验标记为完成。
    write_json(out / "selection.json", record)  # 写出最终训练唯一接受的完整轮数选择记录。
    write_json(out / "protocol_manifest.json", record)  # 将协议清单同步更新为包含所有折结果的完成状态。
    return record  # 返回最终选择记录，供调用者查看所选轮数及数据来源。


def fit_selected(train, selection_path, output, *, device="cpu", test_loader=None):  # 按已冻结的轮数从头训练，保存最终权重之后才允许一次测试评估。
    # 最终训练重新初始化模型；测试数据以延迟调用函数提供，只在检查点写完后才被实际读取。
    """Freshly train at the selected epoch count, then optionally score test once.

    test_loader is a zero-argument callable. It is invoked only after the final
    checkpoint is written, so selection and training cannot inspect test data.
    """
    selection_path = Path(selection_path).resolve(strict=True)  # 固定选择记录的绝对路径，并要求文件已存在。
    selection = json.loads(selection_path.read_text(encoding="utf-8"))  # 读取选择阶段保存的模型参数、数据摘要与训练轮数。
    if selection.get("protocol") != PROTOCOL or selection.get("status") != "complete" or selection.get("stage") != "selection":  # 必须使用当前协议已经完成的选择阶段记录。
        raise ValueError("Supply a completed selection.json from this training protocol.")  # 拒绝其他协议、未完成实验或非选择阶段的文件。
    if selection.get("trial_fingerprint_schema") != TRIAL_FINGERPRINT_SCHEMA:  # 旧记录缺少内容隔离证据，不能把它静默当成已验证的新记录。
        raise ValueError("Selection record lacks current EEG trial fingerprints; run a new selection.")  # 明确要求重新选轮数，保持已有检查点的推理格式不变。
    for field, required in (("training_trial_sha256", True), ("validation_trial_sha256", bool(selection.get("validation_files")))):  # 核验训练指纹与可选显式验证指纹是否完整。
        values = selection.get(field)  # 读取持久化列表，不重新访问训练阶段用过的验证文件。
        if not isinstance(values, list) or (required and not values) or any(  # 拒绝缺失字段、应有却为空的记录以及不是 SHA-256 的条目。
                not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)  # 每个指纹必须为 64 位小写十六进制字符串。
                for value in values):  # 逐条检查列表内容，确保后续比较基于有效格式。
            raise ValueError("Selection record has invalid EEG trial fingerprints; run a new selection.")  # 无有效内容记录时要求重新执行选择阶段。
    if selection.get("source_sha256") != source_hashes():  # 逐文件比较当前源码与选择时保存的内容指纹，包括注释在内的任何字节变化。
        raise ValueError("Source files changed after selection; run a new selection under the changed code.")  # 源码不同则要求重新选择，避免跨代码版本直接沿用轮数。
    if selection["training_files"] != train.files:  # 最终训练必须使用选择阶段记录的相同路径及相同内容的训练文件。
        raise ValueError("Final training inputs differ from the hashed selection inputs.")  # 发现训练数据来源或字节内容变化时停止。
    if selection["input_label_offset"] != train.label_offset:  # 同一份数据也必须采用与选择时相同的标签偏移规则。
        raise ValueError("Label encoding differs from selection.")  # 防止最终训练把类别编号解释成不同含义。
    config = TrainConfig(**selection["config"])  # 完全从选择记录恢复训练超参数，不单独重新调参。
    config.validate()  # 校验恢复后的配置仍满足当前训练流程的范围要求。
    model_kwargs = selection["model_kwargs"]  # 恢复选择时使用的模型结构和输入维度。
    assert_shape(train, model_kwargs)  # 检查当前训练数组符合恢复后的模型输入要求。
    validate_classes(train, model_kwargs["classes"], require_all=True)  # 最终训练会话仍须覆盖全部类别。
    training_trials = trial_fingerprints(train)  # 重新对已加载训练数组计算摘要，也能识别文件读取后发生的内存内容变化。
    if selection["training_trial_sha256"] != training_trials:  # 数据内容和试次顺序都必须与选轮数时保存的指纹一致。
        raise ValueError("Final training EEG content differs from the selection trial fingerprints.")  # 保留文件摘要检查，并进一步拒绝模型实际输入的变化。
    validation_trials = selection["validation_trial_sha256"]  # 只使用选择记录保存的验证指纹，最终训练不回读验证文件。
    assert_disjoint_trials(training_trials, validation_trials, "Training", "validation")  # 防止不一致的选择记录将重复训练与验证样本带入最终阶段。
    epochs = selection["selected_epoch"]  # 使用验证阶段已经冻结的训练轮数。
    if not isinstance(epochs, int) or not 1 <= epochs <= config.max_epochs:  # 所选轮数必须为允许范围内的整数。
        raise ValueError("Selected epoch is invalid.")  # 拒绝缺乏有效训练长度的选择记录。
    out = new_output(output)  # 创建全新的最终训练目录，避免覆盖选择结果或已有模型。
    record = manifest("final", config, model_kwargs, device)  # 创建最终训练阶段的实验清单。
    record.update({"training_files": train.files, "selection_file": str(selection_path),  # 保存最终训练输入及其关联选择记录的位置。
                   "selection_sha256": file_sha256(selection_path), "selected_epoch": epochs,  # 对选择记录本身计算摘要，并保存固定轮数以追踪依赖关系。
                   "trial_fingerprint_schema": TRIAL_FINGERPRINT_SCHEMA,  # 在最终清单中保留内容隔离使用的指纹规范。
                   "training_trial_sha256": training_trials, "validation_trial_sha256": validation_trials,  # 保存训练及已冻结的验证内容依据，供测试隔离和后续审查。
                   "test_accessed": False, "test_evaluations": 0})  # 在参数拟合开始前，测试数据尚未读取且评估次数为零。
    write_json(out / "config.json", {"config": asdict(config), "model_kwargs": model_kwargs})  # 保存此次从选择记录恢复的配置。
    write_json(out / "protocol_manifest.json", record)  # 在训练前落盘协议状态，便于识别中断位置。
    model, optimizer, scheduler = initialize(model_kwargs, config, device)  # 从固定种子重新初始化模型，不载入任何验证折的最佳权重。
    batches = loader(train, config.batch_size, shuffle=True, seed=config.seed)  # 在传入的完整训练会话上构造可复现的打乱批次。
    history = []  # 记录最终训练每一轮的训练损失及学习率。
    for epoch in range(1, epochs + 1):  # 严格训练所选轮数，不再根据验证或测试表现改变停止位置。
        history.append({"epoch": epoch, "train_loss": train_epoch(model, batches, optimizer, device),  # 完成一轮训练并记录其按试次平均的联合损失。
                        "learning_rate": optimizer.param_groups[0]["lr"]})  # 保存本轮使用的学习率，供与选择阶段的调度定义对照。
        scheduler.step()  # 按共同的最大轮数周期推进余弦调度，而非按所选轮数重新缩放。
    torch.save({"protocol": PROTOCOL, "model_kwargs": model_kwargs,  # 保存最终模型并附带协议与结构信息。
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},  # 将参数和缓冲区从梯度图分离后转为 CPU 存储。
                "config": asdict(config), "selected_epoch": epochs, "stage": "final",  # 明确该权重来自固定轮数的最终训练阶段。
                "selection_sha256": record["selection_sha256"]}, out / "model.pt")  # 关联选择记录摘要，并在任何测试数据读取之前写出最终检查点。
    write_json(out / "history.json", history)  # 保存最终训练的逐轮记录。
    record["checkpoint_sha256"] = file_sha256(out / "model.pt")  # 对已经写出的模型文件计算摘要，标识评估时使用的确切权重。
    if test_loader is not None:  # 只有调用者提供延迟测试读取函数时，才执行可选的单次测试评估。
        record["test_accessed"] = True  # 在调用测试读取函数前记录即将发生的测试访问。
        write_json(out / "protocol_manifest.json", record)  # 先持久化访问标记，即使读取或评估中断也能追踪。
        test = test_loader()  # 此处才真正读取测试数据，选择轮数和参数训练此前均已完成。
        assert_shape(test, model_kwargs)  # 测试输入必须具有训练模型要求的通道数与时间点数。
        validate_classes(test, model_kwargs["classes"])  # 检查测试标签存在且没有超出配置类别范围。
        development_files = set(train.files) | set(selection.get("validation_files", {}))  # 汇总训练与显式验证使用过的文件路径。
        if development_files.intersection(test.files):  # 检查测试来源是否与开发阶段来源出现相同路径。
            raise ValueError("Test files overlap the training or validation files.")  # 拒绝使用训练或验证文件充当测试数据。
        test_trials = trial_fingerprints(test)  # 训练和模型保存完成后，才对延迟读取的测试 EEG 生成指纹。
        assert_disjoint_trials(training_trials, test_trials, "Training", "test")  # 在任何测试预测或指标写出之前，拒绝训练输入的副本或部分重复试次。
        assert_disjoint_trials(validation_trials, test_trials, "Validation", "test")  # 同时拒绝显式验证输入的副本，不需要重新读取验证文件。
        logits = infer(model, test.X, config.batch_size, device)  # 使用已保存的最终模型对测试集执行一次推理。
        save_predictions(out / "test_predictions.npz", logits, test.y)  # 保存逐试次测试得分、概率、预测和标签以便复核。
        write_json(out / "test_metrics.json", metrics(logits, test.y))  # 将本次测试分类指标写入独立结果文件。
        record["test_files"] = test.files  # 记录实际读取的测试文件及其 SHA-256 摘要。
        record["test_trial_sha256"] = test_trials  # 保存通过隔离检查的测试试次指纹，以便复核最终评估的数据内容。
        record["test_evaluations"] = 1  # 只有测试指标成功保存后，才将完成的测试评估次数记为一次。
    record["status"] = "complete"  # 最终训练及所请求的可选测试均完成后，标记整个阶段完成。
    write_json(out / "protocol_manifest.json", record)  # 持久化最终状态、模型摘要和实际测试访问记录。
    return record  # 返回最终训练的完整溯源与评估状态。


def load_checkpoint(path, device="cpu", model_kwargs=None):  # 从当前协议检查点或显式给定结构的纯权重字典恢复推理模型。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # 先将权重安全地映射到 CPU，并限制为权重加载模式。
    if checkpoint.get("protocol") == PROTOCOL:  # 当前协议检查点内含模型构造参数及训练元数据。
        if model_kwargs is not None and model_kwargs != checkpoint["model_kwargs"]:  # 若调用者同时指定结构，必须与检查点元数据完全一致。
            raise ValueError("Model dimensions differ from checkpoint metadata.")  # 阻止按错误输入维度或结构解释保存的权重。
        model_kwargs = checkpoint["model_kwargs"]  # 以检查点保存的参数恢复准确的模型结构。
        state = checkpoint["state_dict"]  # 从检查点中提取模型参数与缓冲区字典。
    elif model_kwargs is not None and checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):  # 纯权重字典必须非空、所有值为张量，且调用者显式提供模型结构。
        state = checkpoint  # 对纯权重格式，整个读取对象就是待恢复的状态字典。
        checkpoint = {"model_kwargs": model_kwargs, "format": "plain_state_dict"}  # 为没有训练元数据的权重补充结构与格式说明。
    else:  # 既不符合当前协议，也不满足明确结构的纯权重字典条件。
        raise ValueError("Use a current checkpoint, or provide model dimensions for a plain RD state dictionary.")  # 要求提供可明确解释的当前模型权重格式。
    model = NPFEEGNetRD(**model_kwargs).to(device)  # 构造与检查点对应的当前模型并移到推理设备。
    model.load_state_dict(state, strict=True)  # 严格匹配全部参数名和张量形状，拒绝缺失或多余的权重项。
    model.eval()  # 恢复后立即切换为评估模式，关闭随机失活并固定批归一化行为。
    return model, checkpoint  # 同时返回可推理的模型和其元数据，供调用者校验或记录。
