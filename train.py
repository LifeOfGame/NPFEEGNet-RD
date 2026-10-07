# 模块说明：先用训练会话及验证数据选择轮数，再重新初始化并训练最终模型。
"""Select epochs on training-session data, then fit a fresh NPFEEGNet-RD."""
import argparse  # 解析命令行中的子命令、文件路径和训练超参数。
import json  # 读取模型配置及已经完成的轮数选择记录。
from pathlib import Path  # 使用跨平台路径对象读取文件并输出绝对路径。

from training.data import load_data  # 显式读取指定会话，并将 EEG 整理为 [试次, 1, 通道, 时间点]。
from training.engine import TrainConfig, fit_selected, select_epochs  # 导入训练配置、固定轮数重训和验证选轮数入口。


def model_config(path):  # 从 JSON 文件取得模型输入尺寸、类别数与采样率。
    result = json.loads(Path(path).read_text(encoding="utf-8"))  # 按 UTF-8 解码文件，再解析为配置字典。
    allowed = {"num_channels", "classes", "n_times", "sampling_rate"}  # 只允许通道数、类别数、时间点数和采样率四种配置项。
    if set(result) - allowed or not {"num_channels", "classes"} <= set(result):  # 拒绝未知配置项，以及缺少通道数或类别数的配置。
        raise ValueError("Model config requires num_channels/classes and optional n_times/sampling_rate.")  # 明确报错，避免按不完整或拼错的配置构建模型。
    return {"n_times": 1000, "sampling_rate": 250.0, **result}  # 默认每试次 1000 点、250 Hz；文件内的同名值覆盖默认值。


def build_parser():  # 定义彼此分离的 select 选轮数流程与 fit 最终训练流程。
    parser = argparse.ArgumentParser(description=__doc__)  # 以模块说明作为命令行帮助的简介。
    commands = parser.add_subparsers(dest="command", required=True)  # 必须选择一个子命令，并将名称存入 args.command。
    select = commands.add_parser("select", help="Select using groups or separate validation data; no test argument.")  # 选轮数阶段仅接受训练及验证数据，不提供测试集参数。
    select.add_argument("--train", required=True)  # 指定必需的训练会话 NPY 或 NPZ 文件。
    select.add_argument("--validation")  # 可显式提供验证会话；省略时按训练会话内的完整分组做交叉验证。
    select.add_argument("--model-config", required=True)  # 指定必需的模型尺寸 JSON 文件。
    select.add_argument("--output", required=True)  # 指定用于保存各折记录及 selection.json 的新目录。
    select.add_argument("--label-offset", type=int, choices=(0, 1), default=1,  # 将存储标签减去 0 或 1，转换为模型要求的从零开始的类别编号。
                        help="Stored label minimum: supplied dataset arrays use 1; zero-based NPZ uses 0.")  # 帮助文本说明现有数据默认从 1 编号，零起始的 NPZ 应显式使用 0。
    select.add_argument("--seed", type=int, default=0)  # 设置模型初始化与训练批次顺序使用的随机种子。
    select.add_argument("--batch-size", type=int, default=64)  # 设置每个训练或推理批次包含的试次数，默认 64。
    select.add_argument("--epochs", type=int, default=300)  # 设置每折用于选择轮数的最大训练轮数，默认 300。
    select.add_argument("--patience", type=int, default=40)  # 验证指标连续指定轮数未改善时停止该折训练，默认 40。
    select.add_argument("--lr", type=float, default=0.001)  # 设置优化器初始学习率，默认 0.001。
    select.add_argument("--weight-decay", type=float, default=0.0001)  # 设置优化器权重衰减强度，默认 0.0001。
    select.add_argument("--device", default="cpu")  # 设置选轮数阶段使用的计算设备，默认 CPU。
    fit = commands.add_parser("fit", help="Fresh fixed-epoch fit from a completed selection record.")  # 根据已完成的选择记录重新初始化模型，并训练固定轮数。
    fit.add_argument("--train", required=True)  # 最终训练必须提供与选择阶段文件哈希一致的训练数据。
    fit.add_argument("--selection", required=True)  # 指定包含选定轮数、模型配置和数据记录的 selection.json。
    fit.add_argument("--output", required=True)  # 指定用于保存最终检查点、训练记录及可选评估结果的新目录。
    fit.add_argument("--test", help="Optional official evaluation session, opened only after fitting.")  # 仅显式提供此参数时，才在最终训练及保存检查点后读取测试会话。
    fit.add_argument("--device", default="cpu")  # 设置最终训练与可选测试推理使用的计算设备。
    return parser  # 返回完整解析器，供 main 解析命令行或调用者提供的参数列表。


def main(argv=None):  # 运行训练命令；argv 为 None 时读取进程命令行参数。
    args = build_parser().parse_args(argv)  # 解析并检查子命令、必填路径、数值类型和可选参数。
    if args.command == "select":  # 进入只使用训练会话及验证信息的轮数选择流程。
        train = load_data(args.train, label_offset=args.label_offset)  # 读取指定训练会话，得到 [N, 1, C, T] EEG、零起始标签和可选分组。
        validation = None if args.validation is None else load_data(args.validation, label_offset=args.label_offset)  # 仅在显式指定路径时读取验证数据，标签偏移与训练数据一致。
        settings = TrainConfig(seed=args.seed, batch_size=args.batch_size, learning_rate=args.lr,  # 将种子、批量大小与学习率汇总为训练配置。
                               weight_decay=args.weight_decay, max_epochs=args.epochs, patience=args.patience)  # 补充权重衰减、最大轮数和早停等待轮数。
        result = select_epochs(train, args.output, settings, model_config(args.model_config),  # 按模型配置启动验证选轮数，并将训练与选择记录写入输出目录。
                               validation=validation, device=args.device)  # 传入可选验证集和计算设备；此调用没有测试数据入口。
    else:  # 解析器只定义两个子命令，因此此处进入 fit 最终训练流程。
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))  # 读取此前保存的选择记录，以沿用确定的训练设置。
        offset = selection["input_label_offset"]  # 沿用选择阶段的标签偏移，避免最终训练改变类别编码。
        train = load_data(args.train, label_offset=offset)  # 读取指定训练会话，后续由训练引擎核验其与选择记录一致。
        test_reader = None if args.test is None else lambda: load_data(args.test, label_offset=offset)  # 创建延迟读取函数；此行不会打开测试数据，也不会自行寻找测试集。
        result = fit_selected(train, args.selection, args.output, device=args.device, test_loader=test_reader)  # 校验记录后从头训练固定轮数，并仅在检查点写完后调用可选测试读取函数。
    print(f"{args.command}: complete; epoch={result['selected_epoch']}; output={Path(args.output).resolve()}")  # 在终端报告完成的阶段、选定轮数和结果目录绝对路径。


if __name__ == "__main__":  # 仅直接运行本脚本时执行命令行入口，导入模块时不启动训练。
    main()  # 使用当前进程参数执行 select 或 fit。
