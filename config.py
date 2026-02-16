from copy import deepcopy
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "common": {
        "seed": 42,  # 全局随机种子（控制可复现）
        "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],  # 统一识别的图片后缀
        "video_exts": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"],  # 统一识别的视频后缀
    },
    "data": {
        "train_root": "./data/train",  # 训练集根目录
        "val_root": "./data/val",  # 验证集根目录
        "test_root": "./data/test",  # 测试集根目录
        "image_size": 224,  # 输入图像尺寸（会被 resize 到 image_size x image_size）
        "motion_stride": 2,  # 时间分支配对帧间隔（i 与 i+stride）
        "max_frames_per_video": 0,  # 每个视频最多使用帧数，0 表示不限制
        "real_class_names": ["0_real", "real", "0_reall"],  # 真实类目录名候选（按顺序匹配）
        "fake_class_names": ["1_fake", "fake"],  # 伪造类目录名候选（按顺序匹配）
    },
    "model": {
        "fusion_mode": "independent",  # 融合模式：independent（双头独立）或 cross_attention（交叉注意力）
        "feature_dim": 512,  # 空间/时间分支对齐后的特征维度
        "use_resnet_imagenet": False,  # 是否使用 ImageNet 预训练的 ResNet50
        "hfri_mode": "fft",  # 高频增强模式：fft 或 dct
        "flowformer_repo": "./FlowFormerPlusPlus-main",  # FlowFormer++ 源码目录
        "flowformer_ckpt": "./checkpoints/things.pth",  # FlowFormer++ 权重路径
        "require_flowformer": True,  # 是否强制要求 FlowFormer 加载成功（失败则报错）
    },
    "train": {
        "save_dir": "./data/exp/thesis_stf",  # 训练输出目录（checkpoint、日志等）
        "resume": "",  # 断点续训 checkpoint 路径，空字符串表示不续训
        "epochs": 20,  # 最大训练轮数
        "batch_size": 8,  # 训练批大小
        "num_workers": 4,  # DataLoader 读取进程数
        "lr": 1e-4,  # 学习率
        "weight_decay": 1e-4,  # AdamW 权重衰减
        "print_freq": 50,  # 每多少个 step 打印一次训练日志
        "save_every": 5,  # 每多少个 epoch 额外保存一次 epoch_xxx.pth（latest 每轮都会保存）
        "amp": False,  # 是否开启混合精度训练（CUDA 下生效）
        "aux_loss_weight": 0.5,  # independent 模式下辅助损失权重
        "fusion_loss_weight": 0.1,  # cross_attention 模式下融合正则损失权重
        "early_stop_metric": "auc",  # 早停指标：auc / acc / loss
        "early_stop_patience": 5,  # 早停容忍轮数（连续多少轮无提升后停止）
        "early_stop_min_delta": 0.0,  # 早停最小提升阈值（小于该提升不计为改进）
        "cache_val_predictions": True,  # 是否缓存验证集预测结果到 CSV（便于后续找阈值）
        "cache_val_every": 1,  # 每多少个验证 epoch 缓存一次预测
        "cache_val_threshold": 0.5,  # 缓存 CSV 里生成 pred 列时使用的阈值
    },
    "test": {
        "checkpoint": "./data/exp/thesis_stf/best.pth",  # 测试时加载的模型权重路径
        "output_csv": "./data/exp/thesis_stf/test_frame_predictions.csv",  # 帧级预测输出 CSV
        "video_output_csv": "",  # 视频级预测输出 CSV，空字符串表示自动命名
        "threshold": 0.5,  # 概率转标签阈值（prob >= threshold 判为 fake）
        "batch_size": 8,  # 测试推理批大小（按视频内部帧批处理）
        "num_workers": 4,  # 兼容参数（当前测试脚本主要按视频目录直接推理）
        "real_dir_name": "0_real",  # 测试集真实类目录主名称
        "fake_dir_name": "1_fake",  # 测试集伪造类目录主名称
        "balanced_1to1": True,  # 是否按每个 fake 生成模型分组做 real:fake=1:1 配平
        "cache_video_predictions": True,  # 是否开启视频级预测缓存（重复视频直接复用）
        "cache_dir": "./data/exp/thesis_stf/test_cache",  # 测试缓存根目录
        "refresh_cache": False,  # 是否忽略历史缓存并强制重算
        "fusion_mode": "",  # 测试时可选覆盖融合模式，空字符串表示沿用 checkpoint 配置
        "feature_dim": 0,  # 测试时可选覆盖特征维度，0 表示沿用 checkpoint 配置
        "hfri_mode": "",  # 测试时可选覆盖 HFRI 模式，空字符串表示沿用 checkpoint 配置
        "flowformer_repo": "",  # 测试时可选覆盖 FlowFormer 源码目录，空字符串表示沿用 checkpoint 配置
        "flowformer_ckpt": "",  # 测试时可选覆盖 FlowFormer 权重路径，空字符串表示沿用 checkpoint 配置
    },
    "process": {
        "output_root": "",  # 抽帧输出根目录，空字符串表示自动用 <input_root>_frames
        "image_ext": "jpg",  # 抽帧保存格式（jpg 或 png）
        "image_ext_choices": ["jpg", "png"],  # 可选输出格式列表（用于命令行 choices）
        "jpg_quality": 95,  # jpg 压缩质量（1~100）
        "zero_pad": 5,  # 帧序号补零宽度（例如 00000.jpg）
        "every_n": 1,  # 每隔 N 帧抽取 1 帧（1 表示逐帧保存）
        "overwrite": False,  # 输出目录已存在时是否覆盖
    },
}


def get_default_config() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy to avoid accidental in-place edits."""
    return deepcopy(DEFAULT_CONFIG)


def get_section(section: str) -> Dict[str, Any]:
    cfg = get_default_config()
    if section not in cfg:
        raise KeyError(f"Unknown config section: {section}")
    return cfg[section]
