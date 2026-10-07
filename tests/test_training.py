# 用临时合成数据检查选轮数、重新初始化训练、测试读取时机和推理导出。
"""Synthetic end-to-end checks of selection, fresh fitting and inference."""
import json  # 读取并构造训练选择记录，测试其一致性保护。
from pathlib import Path  # 管理临时数据文件和模型输出路径。
import shutil  # 在临时目录构造字节相同但文件名不同的训练数据副本。
import tempfile  # 为每次测试创建自动清理的独立临时目录。
import unittest  # 使用标准单元测试框架组织断言。
from unittest.mock import patch  # 临时替换源码摘要函数，模拟代码发生变化的情况。

import numpy as np  # 生成合成 EEG 数组并读写 NPZ 数据。
import torch  # 配置计算线程并保存模型状态字典。

from predict import main as predict_main  # 直接调用推理命令入口，检验其参数与文件输出。
from training.data import load_data  # 使用正式数据加载逻辑读取合成训练和测试文件。
from training.engine import TrainConfig, fit_selected, infer, load_checkpoint, select_epochs  # 导入训练配置、选轮数、最终训练和推理函数。


class TrainingFlow(unittest.TestCase):  # 汇总训练流程及输入保护的集成测试。
    def test_group_selection_fit_and_prediction(self):  # 从合成数据选轮数开始，一直检查到预测文件导出。
        torch.set_num_threads(2)  # 限制线程数量，降低小规模测试的并行调度开销。
        with tempfile.TemporaryDirectory() as directory:  # 将所有测试产物隔离在自动清理的临时目录。
            root = Path(directory)  # 创建便于拼接文件名的目录对象。
            generator = np.random.default_rng(7)  # 使用固定 NumPy 随机种子，使合成数据可以复现。
            train_path, test_path = root / "train.npz", root / "test.npz"  # 分别指定训练和测试文件，保证二者独立。
            np.savez(train_path, X=generator.normal(size=(8, 3, 128)).astype("float32"),  # 保存 8 条三通道、128 时间点的训练试次。
                     y=np.array([1, 2] * 4), groups=np.repeat([0, 1], 4))  # 标签从 1 开始，并划分两个各有 4 条试次的组。
            np.savez(test_path, X=generator.normal(size=(4, 3, 128)).astype("float32"), y=np.array([1, 2] * 2))  # 另存 4 条平衡类别的测试试次。
            train = load_data(train_path, label_offset=1)  # 显式将一开始的标签转换为零开始的类别索引。
            settings = TrainConfig(batch_size=4, max_epochs=1, patience=1)  # 用一个训练轮次快速检验完整流程。
            dims = {"num_channels": 3, "classes": 2, "n_times": 128, "sampling_rate": 250.0}  # 定义与合成输入一致的模型维度。
            selected = select_epochs(train, root / "selection", settings, dims)  # 仅通过训练数据的分组交叉验证选择轮数。
            self.assertFalse(selected["test_accessed"])  # 选择记录必须明确表明没有访问测试集。
            self.assertEqual(selected["fold_epochs"], [1, 1])  # 两折在单轮预算下都只能选择第 1 轮。
            calls = []  # 记录测试数据读取回调的调用次数。
            def read_test():  # 定义延迟加载测试数据的回调，以观测测试集访问时机。
                self.assertTrue((root / "final/model.pt").is_file())  # 测试集被读取时，最终训练的模型应已保存。
                calls.append(1)  # 每次读取测试集时增加一条调用记录。
                return load_data(test_path, label_offset=1)  # 用与训练集一致的标签偏移读取测试数据。
            fitted = fit_selected(train, root / "selection/selection.json", root / "final", test_loader=read_test)  # 按固定轮数重新初始化训练，然后调用一次测试评估。
            self.assertEqual(calls, [1])  # 确认测试数据只读取了一次。
            self.assertEqual(fitted["test_evaluations"], 1)  # 最终结果也应记录一次测试评估。
            model, _ = load_checkpoint(root / "final/model.pt")  # 加载最终检查点，并忽略此处无需使用的附带元信息。
            logits = infer(model, train.X, 4, "cpu")  # 在 CPU 上以批大小 4 对训练输入做预测。
            self.assertEqual(logits.shape, (8, 2))  # 8 条试次应各输出两类得分。
            predict_main(["--checkpoint", str(root / "final/model.pt"), "--input", str(test_path),  # 通过实际 CLI 参数形式调用预测入口。
                          "--output", str(root / "predictions.npz")])  # 指定预测输出文件，检验端到端导出。
            with np.load(root / "predictions.npz") as saved:  # 打开预测文件检查其字段和概率。
                np.testing.assert_allclose(saved["probabilities"].sum(1), 1)  # 每条试次的类别概率之和应接近 1。
                self.assertNotIn("labels", saved.files)  # 推理结果文件不应包含输入数据的真实标签。
            # A plain saved RD state uses explicit dimensions and predicts identically. 中文：纯状态字典在显式提供维度后也应给出完全相同的预测。
            torch.save(model.state_dict(), root / "weights.pth")  # 保存不含模型维度元数据的纯状态字典。
            restored, _ = load_checkpoint(root / "weights.pth", model_kwargs=dims)  # 显式提供维度，检验旧式权重格式的加载。
            np.testing.assert_array_equal(infer(restored, train.X, 4, "cpu"), logits)  # 两种保存格式恢复出的预测必须精确一致。
            with self.assertRaises(FileExistsError):  # 重复使用已存在的最终输出目录应被拒绝。
                fit_selected(train, root / "selection/selection.json", root / "final")  # 故意复用同一路径，验证防覆盖检查。
            with patch("training.engine.source_hashes", return_value={"changed": "source"}):  # 模拟选轮数完成后源码发生变化。
                with self.assertRaisesRegex(ValueError, "Source files changed"):  # 期望训练入口拒绝不匹配的源码版本。
                    fit_selected(train, root / "selection/selection.json", root / "other")  # 在新输出路径下触发源码摘要校验。
            selection_path = root / "selection/selection.json"  # 定位测试生成的选择记录。
            changed = json.loads(selection_path.read_text())  # 将选择记录解析为可修改字典。
            changed["input_label_offset"] = 0  # 故意把原本为 1 的输入标签偏移改为 0。
            selection_path.write_text(json.dumps(changed))  # 仅覆盖本测试的临时记录，以构造不一致场景。
            with self.assertRaisesRegex(ValueError, "Label encoding"):  # 标签解释与选轮数阶段不一致时必须报错。
                fit_selected(train, selection_path, root / "different_labels")  # 使用被修改的记录触发标签编码检查。

    def test_no_implicit_label_guessing_or_single_group_cv(self):  # 检查错误标签偏移和单分组交叉验证都被明确拒绝。
        with tempfile.TemporaryDirectory() as directory:  # 独立创建本测试的临时目录。
            path = Path(directory) / "data.npz"  # 指定合成数据文件路径。
            np.savez(path, X=np.zeros((4, 3, 128), dtype="float32"), y=np.array([0, 1, 0, 1]), groups=np.zeros(4, dtype=int))  # 构造零起始标签且全部试次属于同一组的数据。
            with self.assertRaisesRegex(ValueError, "label offset"):  # 错误声明标签从 1 开始应导致非法负类别被拒绝。
                load_data(path, label_offset=1)  # 故意传入与数据不符的标签偏移。
            with self.assertRaisesRegex(ValueError, "at least two groups"):  # 留组交叉验证要求至少两个不同分组。
                select_epochs(load_data(path), Path(directory) / "out", TrainConfig(max_epochs=1),  # 对仅有一组的训练数据尝试选轮数。
                              {"num_channels": 3, "classes": 2, "n_times": 128})  # 模型维度合法，从而将失败原因限定为分组不足。


    def test_validation_copies_repacking_and_partial_overlap_are_rejected(self):  # 防止不同文件身份或部分重叠的 EEG 绕过训练与验证隔离。
        with tempfile.TemporaryDirectory() as directory:  # 所有原始文件和别名均限制在独立的临时测试目录。
            root = Path(directory)  # 统一管理本测试的输入和预期输出路径。
            generator = np.random.default_rng(17)  # 固定随机种子，生成互不重复且可复现的合成试次。
            X = generator.normal(size=(4, 3, 128)).astype("float32")  # 创建同时覆盖两类的四条 EEG 输入。
            labels = np.array([0, 1, 0, 1])  # 所有容器采用一致的零起始类别编码。
            np.savez(root / "train.npz", X=X, y=labels)  # 保存唯一的原始训练包。
            shutil.copyfile(root / "train.npz", root / "copied.npz")  # 构造文件路径不同而字节完全一致的整包副本。
            (root / "hardlinked.npz").hardlink_to(root / "train.npz")  # 构造不同路径但指向同一文件内容的硬链接。
            np.savez_compressed(root / "repacked.npz", y=labels[::-1], X=np.asfortranarray(X[::-1, None].astype("float64")))  # 改变压缩、字段顺序、试次顺序、形状写法、内存布局和精度，仍保留相同 EEG。
            rounded = np.nextafter(X.astype("float64"), np.inf)  # 在双精度中轻微扰动，使写盘字节不同但转换为模型 float32 后完全一致。
            np.savez(root / "rounded.npz", X=rounded, y=labels)  # 验证按模型实际输入精度识别重复，而不只比较原始文件字节。
            partial = generator.normal(size=X.shape).astype("float32")  # 先生成与训练集无关的新验证试次。
            partial[3] = X[1]  # 仅让一个验证试次复制某条训练 EEG，覆盖部分重叠场景。
            np.savez(root / "partial.npz", X=partial, y=labels)  # 保存混合新样本与重复样本的验证包。
            train = load_data(root / "train.npz")  # 使用正式加载器规范化训练数据。
            dims = {"num_channels": 3, "classes": 2, "n_times": 128}  # 保持模型维度合法，使失败原因限定为内容重叠。
            for name in ("copied", "hardlinked", "repacked", "rounded", "partial"):  # 分别覆盖复制、硬链接、重新打包、精度归一化与部分重复。
                with self.subTest(alias=name):  # 若某类输入漏检，明确报告其构造方式。
                    validation = load_data(root / f"{name}.npz")  # 通过实际文件加载被测验证数据。
                    with patch("training.engine.initialize") as initialize_model:  # 确认拒绝发生在任何模型初始化或训练之前。
                        with self.assertRaisesRegex(ValueError, "EEG content overlaps"):  # 必须明确指出 EEG 内容重叠，而非依赖无关的文件格式错误。
                            select_epochs(train, root / f"out_{name}", TrainConfig(max_epochs=1), dims, validation=validation)  # 通过正式选轮数入口触发隔离检查。
                        initialize_model.assert_not_called()  # 不允许先开始训练再发现验证数据重复。
                    self.assertFalse((root / f"out_{name}").exists())  # 输入被拒绝时也不应留下看似有效的选择结果目录。


    def test_duplicate_trials_across_groups_are_rejected(self):  # 组编号不同不能掩盖跨训练折与验证折的同一条 EEG。
        with tempfile.TemporaryDirectory() as directory:  # 将分组测试数据保存在临时目录。
            root = Path(directory)  # 构造会话文件和预期输出位置。
            X = np.random.default_rng(23).normal(size=(4, 3, 128)).astype("float32")  # 创建两组各两条、类别均完整的合成 EEG。
            X[2] = X[0]  # 把同一试次放入不同组，模拟复制样本被重新分组的情况。
            np.savez(root / "train.npz", X=X, y=np.array([0, 1, 0, 1]), groups=np.array([0, 0, 1, 1]))  # 每个训练折都有两类，避免类别检查掩盖内容隔离问题。
            with patch("training.engine.initialize") as initialize_model:  # 不允许错误分组进入真正训练。
                with self.assertRaisesRegex(ValueError, "EEG content overlaps"):  # 跨组重复必须按 EEG 内容被识别。
                    select_epochs(load_data(root / "train.npz"), root / "out", TrainConfig(max_epochs=1),  # 使用正常的留组交叉验证入口。
                                  {"num_channels": 3, "classes": 2, "n_times": 128})  # 设置与合成输入一致的模型维度。
                initialize_model.assert_not_called()  # 验证在第一折初始化之前就拒绝不独立的数据划分。


    def test_test_content_is_checked_after_fit_using_saved_validation_fingerprints(self):  # 检查测试延迟访问、训练及验证副本拒绝，以及相同标签不同 EEG 的正常路径。
        torch.set_num_threads(2)  # 限制合成单轮训练的线程开销。
        with tempfile.TemporaryDirectory() as directory:  # 本测试只创建临时合成数据和临时检查点。
            root = Path(directory)  # 统一管理多种测试输入与输出目录。
            generator = np.random.default_rng(31)  # 固定种子，保证训练、验证和独立测试的 EEG 可复现。
            labels, groups = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])  # 三个会话使用完全相同的标签及分组，检验它们不会导致误拒。
            inputs = {}  # 保存各会话的 EEG，用于构造重复和独立的测试输入。
            for role in ("train", "validation", "independent"):  # 为训练、显式验证和独立测试创建不同 EEG。
                inputs[role] = generator.normal(size=(4, 3, 128)).astype("float32")  # 每个会话独立采样四条合成试次。
                np.save(root / f"{role}_data.npy", inputs[role])  # 分别保存 EEG 文件，使实际训练加载路径可完整审查。
                np.save(root / f"{role}_label.npy", labels)  # 标签文件字节内容故意完全一致，但对应不同会话的合法标签。
                np.save(root / f"{role}_group.npy", groups)  # 分组文件字节也完全一致，不应单凭辅助文件内容判断 EEG 重叠。
            train = load_data(root / "train_data.npy")  # 加载正式训练会话及其配套标签和分组。
            validation = load_data(root / "validation_data.npy")  # 加载具有相同标签及分组、但不同 EEG 的显式验证会话。
            selected = select_epochs(train, root / "selection", TrainConfig(batch_size=4, max_epochs=1, patience=1),  # 实际执行一轮合成选轮数，确认正常的数据划分仍可用。
                                     {"num_channels": 3, "classes": 2, "n_times": 128}, validation=validation)  # 使用与输入相匹配的小模型完成单折验证。
            self.assertEqual(len(selected["training_trial_sha256"]), 4)  # 选择记录须保存每条训练 EEG 的指纹。
            self.assertEqual(len(selected["validation_trial_sha256"]), 4)  # 显式验证指纹也必须完整持久化，供之后的隔离检查。
            selection_path = root / "selection/selection.json"  # 保存多个最终训练场景共用的选择记录位置。
            (root / "validation_data.npy").unlink()  # 删除已加载的临时验证 EEG，证明最终阶段只依赖记录中的指纹而不回读它。
            cases = {"train_copy": inputs["train"], "validation_copy": inputs["validation"][::-1]}  # 构造整份训练副本及重排序后的验证副本。
            for role in ("train", "validation"):  # 分别构造仅有一条重复 EEG 的测试集合。
                partial = inputs["independent"].copy()  # 其余测试样本保持与开发数据独立。
                partial[2] = inputs[role][1]  # 将训练或验证的一条 EEG 混入测试集。
                cases[f"partial_{role}"] = partial  # 保存不同重复来源，供下面分别验证。
            for name, X in cases.items():  # 检查训练、验证整包副本与各自的部分重叠。
                with self.subTest(test_case=name):  # 失败时明确是哪种测试数据没有被正确隔离。
                    test_path, output = root / f"{name}.npz", root / f"final_{name}"  # 为每个场景分配独立文件与新训练目录。
                    np.savez_compressed(test_path, X=X.astype("float64"), y=labels, groups=groups)  # 换用压缩 NPZ 和不同精度，使文件路径或字节摘要比较无法替代 EEG 内容检查。
                    calls = []  # 记录测试文件的读取次数。
                    def read_test():  # 保留生产接口要求的延迟测试读取回调。
                        self.assertTrue((output / "model.pt").is_file())  # 读取测试数据时，最终训练及模型保存必须已经完成。
                        calls.append(1)  # 精确记录本次测试访问。
                        return load_data(test_path)  # 此处才加载被测的测试 EEG、标签和分组。
                    with patch("training.engine.infer") as test_infer:  # 观测测试推理是否在内容重叠被发现后仍被错误执行。
                        with self.assertRaisesRegex(ValueError, "EEG content overlaps"):  # 重复测试数据必须在正式预测和评分之前被拒绝。
                            fit_selected(train, selection_path, output, test_loader=read_test)  # 按已冻结轮数完成训练，再触发测试隔离。
                        test_infer.assert_not_called()  # 被拒绝的数据不得产生测试预测。
                    self.assertEqual(calls, [1])  # 失败场景也只允许按协议读取一次测试数据。
                    self.assertFalse((output / "test_predictions.npz").exists())  # 不得留下可被误当作有效测试结果的预测文件。
                    self.assertFalse((output / "test_metrics.json").exists())  # 不得为重叠数据写出准确率等测试指标。
                    rejected = json.loads((output / "protocol_manifest.json").read_text())  # 读取失败后保留的访问状态。
                    self.assertTrue(rejected["test_accessed"])  # 已发生的测试读取必须被如实记录。
                    self.assertEqual(rejected["test_evaluations"], 0)  # 没有完成有效测试评估时，计数必须仍为零。
            fitted = fit_selected(train, selection_path, root / "final_independent",  # 用独立 EEG 检验修复不会阻断正常训练与测试。
                                  test_loader=lambda: load_data(root / "independent_data.npy"))  # 测试标签和组文件内容与训练一致，但 EEG 内容独立，应正常通过。
            self.assertEqual(fitted["test_evaluations"], 1)  # 独立测试应完成且只完成一次正式评估。
            self.assertEqual(len(fitted["test_trial_sha256"]), 4)  # 成功评估需保存测试试次的完整内容指纹。
            legacy = dict(selected)  # 在临时目录构造旧格式选择记录，不修改真实实验记录。
            legacy.pop("trial_fingerprint_schema")  # 删除新指纹规范标记，模拟修复前生成的选择记录。
            legacy_path = root / "legacy_selection.json"  # 为兼容性回归分配单独的临时记录文件。
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")  # 保存缺少隔离依据的旧格式样本。
            with patch("training.engine.initialize") as initialize_model:  # 确认旧选择记录不会开始最终训练。
                with self.assertRaisesRegex(ValueError, "run a new selection"):  # 报错须明确说明旧记录需要重新执行选轮数。
                    fit_selected(train, legacy_path, root / "legacy_final")  # 尝试用旧记录启动最终阶段，验证兼容策略。
                initialize_model.assert_not_called()  # 兼容性校验必须在模型初始化前完成。


if __name__ == "__main__":  # 支持直接执行本测试文件。
    unittest.main()  # 启动标准单元测试运行器。
