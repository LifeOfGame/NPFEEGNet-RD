# 模块说明：从当前模型检查点或单独保存的参数字典加载模型，再对指定 EEG 文件做预测。
"""Predict from a current checkpoint or a saved NPFEEGNet-RD state dictionary."""
import argparse  # 解析检查点、输入输出路径、批量大小和计算设备等命令行参数。
from pathlib import Path  # 检查输出文件状态，创建父目录并显示绝对路径。

from train import model_config  # 复用训练入口的模型尺寸 JSON 读取与配置项检查逻辑。
from training.data import load_data  # 读取显式指定的 EEG 数组，并统一为 [试次, 1, 通道, 时间点]。
from training.engine import infer, load_checkpoint, save_predictions  # 导入批量推理、检查点加载和预测结果保存函数。


def main(argv=None):  # 运行预测命令；argv 为 None 时解析当前进程的命令行参数。
    parser = argparse.ArgumentParser(description=__doc__)  # 以模块说明生成预测命令的帮助简介。
    parser.add_argument("--checkpoint", required=True)  # 指定必需的当前协议检查点或纯模型参数字典文件。
    parser.add_argument("--input", required=True, help="One .npy or .npz EEG array; labels are not read.")  # 显式指定唯一的 EEG 输入文件，预测过程不读取标签。
    parser.add_argument("--output", required=True, help="New .npz file for logits/probabilities/predictions.")  # 指定尚不存在的 NPZ 文件，用于保存分类分数、概率和类别预测。
    parser.add_argument("--model-config", help="Required for plain .pth state dictionaries.")  # 纯参数字典缺少模型尺寸信息，因此需要另行提供模型配置。
    parser.add_argument("--batch-size", type=int, default=64)  # 设置每批推理的试次数，默认 64。
    parser.add_argument("--device", default="cpu")  # 指定模型推理设备，默认使用 CPU。
    args = parser.parse_args(argv)  # 解析参数，并由解析器检查必填项和整数类型。
    if args.batch_size < 1:  # 批量大小必须为正，才能按批遍历输入试次。
        parser.error("--batch-size must be positive")  # 输出参数错误并终止，避免进入无效的批处理循环。
    output = Path(args.output)  # 将输出路径转换为路径对象，便于检查文件与父目录。
    if output.exists() or output.suffix != ".npz":  # 拒绝覆盖已有结果，也拒绝非 .npz 后缀。
        parser.error("--output must be a new .npz file")  # 在加载模型或输入数据之前报告输出路径错误。
    kwargs = None if args.model_config is None else model_config(args.model_config)  # 可选读取外部模型配置；省略时由完整检查点提供配置。
    model, _ = load_checkpoint(args.checkpoint, args.device, kwargs)  # 按配置构建模型并严格加载权重；本入口不使用返回的检查点元数据。
    data = load_data(args.input, require_labels=False)  # 仅读取指定文件的 EEG，整理为 [N, 1, C, T]；不读取标签或分组，也不寻找其他测试会话。
    logits = infer(model, data.X, args.batch_size, args.device)  # 在评估模式下分批预测，返回形状为 [试次数, 类别数] 的未归一化分类分数。
    output.parent.mkdir(parents=True, exist_ok=True)  # 递归创建尚不存在的结果父目录，已有目录可复用。
    save_predictions(output, logits)  # 压缩保存 logits、按类别归一化的概率和从零开始的预测类别，不写入真实标签。
    print(f"Predicted {len(logits)} trials: {output.resolve()}")  # 在终端报告完成预测的试次数及结果文件绝对路径。


if __name__ == "__main__":  # 仅直接执行本文件时启动预测，作为模块导入时不读取模型或数据。
    main()  # 使用当前命令行参数执行预测流程。
