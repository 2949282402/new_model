from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "common": {
        "seed": 42,  # 全局随机种子（控制可复现）
        "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],  # 统一识别的图像后缀
        "video_exts": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"],  # 统一识别的视频后缀
        "exp_root": "./data/exp",  # 实验输出根目录
        "exp_name": "thesis_stf",  # 实验名称（同名实验可复用数据，不同实验互不干扰）
        "use_exp_name_paths": True,  # 是否自动把 train/test/threshold 输出路径重定向到 exp_name 目录
    },
    "data": {
        "train_root": "./data/train",  # 训练集根目录
        "val_root": "./data/val",  # 验证集根目录
        "test_root": "./data/test",  # 测试集根目录
        "image_size": 224,  # 输入图像尺寸（会被 resize 到 image_size x image_size）
        "motion_stride": 2,  # 时序分支配对帧间隔（i 对应 i+stride）
        "max_frames_per_video": 0,  # 每个视频最多使用帧数，0 表示不限制
        "real_class_names": ["0_real", "real", "0_reall"],  # 真实类目录名称候选（按顺序匹配）
        "fake_class_names": ["1_fake", "fake"],  # 伪造类目录名称候选（按顺序匹配）
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
        "save_dir": "./data/exp/thesis_stf",  # 训练输出目录（会被 exp_name 自动重定向）
        "resume": "",  # 断点续训 checkpoint 路径，空字符串表示不续训
        "epochs": 20,  # 最大训练轮数
        "batch_size": 8,  # 训练批大小
        "num_workers": 4,  # DataLoader 读取进程数
        "lr": 1e-4,  # 学习率
        "weight_decay": 1e-4,  # AdamW 权重衰减
        "print_freq": 50,  # 每多少个 step 打印一次训练日志
        "save_every": 5,  # 每多少个 epoch 额外保存一次 epoch_xxx.pth（latest 每轮都保存）
        "amp": False,  # 是否开启混合精度训练（CUDA 下生效）
        "aux_loss_weight": 0.5,  # independent 模式下辅助损失权重
        "fusion_loss_weight": 0.1,  # cross_attention 模式下融合正则损失权重
        "early_stop_metric": "auc",  # 早停指标：auc / acc / loss
        "early_stop_patience": 5,  # 早停容忍轮数（连续多少轮无提升后停止）
        "early_stop_min_delta": 0.0,  # 早停最小提升阈值（小于该提升不计为改进）
        "cache_val_predictions": True,  # 是否缓存验证集预测结果到 CSV（便于后续找阈值）
        "cache_val_every": 1,  # 每多少个验证 epoch 缓存一次预测
        "cache_val_threshold": 0.5,  # 缓存 CSV 里生成 pred 列时使用的阈值
        "val_video_agg_method": "hybrid_topk_mean",  # 验证集视频级聚合策略：mean / topk_mean / hybrid_topk_mean
        "val_video_agg_topk_ratio": 0.2,  # top-k 聚合时使用的比例 k_ratio
        "val_video_agg_topk_min": 3,  # top-k 聚合的最小 k
        "val_video_agg_topk_max": 32,  # top-k 聚合的最大 k，0 表示不限制
        "val_video_agg_hybrid_alpha": 0.7,  # hybrid_topk_mean 权重：final=alpha*topk_mean+(1-alpha)*mean
        "use_motion_token_cache": True,  # 训练时是否优先读取 FlowFormer token 缓存
        "motion_token_cache_dir": "./data/exp/_shared/flowformer_token_cache",  # 训练读取/写入的 FlowFormer token 缓存目录
        "motion_token_cache_refresh": False,  # 是否忽略已有 token 缓存并强制重算
        "motion_token_cache_dtype": "float16",  # token 缓存保存精度：float16 或 float32
        "motion_token_cache_rgb_compress_quality": 0,  # 训练缓存 token 前的 RGB 压缩质量，0 表示不压缩
        "motion_token_cache_mem_videos": 64,  # 内存中最多保留多少个视频 token（LRU）
        "motion_token_cache_batch_size": 16,  # 单次计算一个视频 token 时的 batch 大小
        "motion_token_cache_print_freq": 20,  # 训练时每处理多少次缓存请求打印一次信息
        "motion_token_cache_for_val": True,  # 验证阶段是否也使用 token 缓存
        "motion_token_cache_force_deterministic_pairs": True,  # 使用 token 缓存时训练配对是否固定为 i->i+stride
    },
    "test": {
        "checkpoint": "./data/exp/thesis_stf/best.pth",  # 测试时加载的模型权重路径（会被 exp_name 自动重定向）
        "output_csv": "./data/exp/thesis_stf/test_frame_predictions.csv",  # 帧级预测输出 CSV（会被 exp_name 自动重定向）
        "video_output_csv": "",  # 视频级预测输出 CSV，空字符串表示自动命名
        "threshold": 0.5,  # 概率转标签阈值（prob >= threshold 判为 fake）
        "batch_size": 8,  # 测试推理批大小（按视频内部帧批处理）
        "num_workers": 4,  # 兼容参数（当前测试脚本主要按视频目录直接推理）
        "real_dir_name": "0_real",  # 测试集真实类目录主名称
        "fake_dir_name": "1_fake",  # 测试集伪造类目录主名称
        "balanced_1to1": True,  # 是否按每个 fake 生成模型分组做 real:fake=1:1 配平
        "cache_video_predictions": True,  # 是否开启视频级预测缓存（重复视频直接复用）
        "cache_dir": "./data/exp/thesis_stf/test_cache",  # 测试缓存根目录（会被 exp_name 自动重定向）
        "refresh_cache": False,  # 是否忽略历史缓存并强制重算
        "rgb_compress_quality": 0,  # 测试时 RGB 压缩质量，0 表示不压缩，1~100 表示 JPEG 质量
        "rgb_compress_quality_choices": [0, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30],  # 可选压缩质量
        "fusion_mode": "",  # 测试时可选覆盖融合模式，空字符串表示沿用 checkpoint 配置
        "feature_dim": 0,  # 测试时可选覆盖特征维度，0 表示沿用 checkpoint 配置
        "hfri_mode": "",  # 测试时可选覆盖 HFRI 模式，空字符串表示沿用 checkpoint 配置
        "flowformer_repo": "",  # 测试时可选覆盖 FlowFormer 源码目录，空字符串表示沿用 checkpoint 配置
        "flowformer_ckpt": "",  # 测试时可选覆盖 FlowFormer 权重路径，空字符串表示沿用 checkpoint 配置
        "video_agg_method": "topk_mean",  # 视频级概率聚合策略：mean / topk_mean / hybrid_topk_mean
        "video_agg_topk_ratio": 0.1,  # top-k 聚合时使用的比例 k_ratio
        "video_agg_topk_min": 3,  # top-k 聚合的最小 k
        "video_agg_topk_max": 32,  # top-k 聚合的最大 k，0 表示不限制
        "video_agg_hybrid_alpha": 0.7,  # hybrid_topk_mean 权重：final=alpha*topk_mean+(1-alpha)*mean
        "use_motion_token_cache": True,  # 测试时是否优先读取 FlowFormer token 缓存
        "motion_token_cache_dir": "./data/exp/_shared/flowformer_token_cache",  # 测试读取/写入的 FlowFormer token 缓存目录
        "motion_token_cache_refresh": False,  # 测试时是否忽略 token 缓存并强制重算
        "motion_token_cache_dtype": "float16",  # 测试 token 缓存保存精度：float16 或 float32
        "motion_token_cache_mem_videos": 128,  # 测试阶段内存中最多保留多少个视频 token（LRU）
        "motion_token_cache_batch_size": 16,  # 测试阶段 token 缓存 miss 时的计算 batch 大小
        "motion_token_cache_print_freq": 20,  # 测试阶段每处理多少次 token 缓存请求打印一次信息
    },
    "threshold_search": {
        "val_cache_dir": "./data/exp/thesis_stf/val_cache",  # 验证集预测缓存目录（包含 epoch_xxx_frame/video.csv，会被 exp_name 自动重定向）
        "input_csv": "",  # 显式指定输入 CSV；为空时自动从 val_cache_dir 选择
        "level": "video",  # 阈值搜索粒度：video 或 frame
        "epoch": 0,  # 选择的 epoch 编号；0 表示自动使用最新一轮
        "threshold_min": 0.0,  # 阈值搜索下界
        "threshold_max": 1.0,  # 阈值搜索上界
        "threshold_step": 0.001,  # 阈值搜索步长
        "opt_metric": "f1",  # 最优阈值目标指标：f1/acc/balanced_acc/youden_j/precision/recall
        "near_best_delta": 0.005,  # 阈值范围容差：保留 metric >= best - delta 的阈值
        "min_precision": 0.0,  # 约束条件：最小 precision
        "min_recall": 0.0,  # 约束条件：最小 recall
        "topk": 10,  # 输出前 K 个候选阈值
        "curve_output_csv": "./data/exp/thesis_stf/threshold_curve.csv",  # 阈值-指标曲线输出 CSV（会被 exp_name 自动重定向）
        "summary_output_json": "./data/exp/thesis_stf/threshold_summary.json",  # 最优阈值与推荐范围摘要 JSON（会被 exp_name 自动重定向）
    },
    "flow_cache": {
        "roots": ["./data/train", "./data/val", "./data/test"],  # 需要扫描并缓存的根目录列表（递归查找帧目录）
        "cache_dir": "./data/exp/_shared/flowformer_token_cache",  # FlowFormer token 共享缓存目录（跨实验复用）
        "image_size": 224,  # 预处理尺寸（需与训练/测试一致）
        "motion_stride": 2,  # 帧配对间隔：第 i 帧与 i+stride 帧组成时序对
        "max_frames_per_video": 0,  # 每个视频最多缓存多少帧对，0 表示不限制
        "batch_size": 16,  # 缓存阶段推理 batch 大小
        "rgb_compress_quality": 0,  # 缓存前 RGB 压缩质量：0 不压缩，1~100 为 JPEG 质量
        "flowformer_repo": "./FlowFormerPlusPlus-main",  # FlowFormer++ 源码目录
        "flowformer_ckpt": "./checkpoints/things.pth",  # FlowFormer++ 权重路径
        "require_flowformer": True,  # 是否强制必须加载 FlowFormer（失败则报错）
        "save_dtype": "float16",  # 缓存 token 保存精度：float16 或 float32
        "refresh_cache": False,  # 是否忽略已有缓存并强制重算
        "print_freq": 20,  # 每处理多少个视频打印一次进度
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


def _norm_path(path_obj: Path) -> str:
    return path_obj.as_posix()


def _apply_exp_name_paths(cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    common_cfg = cfg.get("common", {})
    if not bool(common_cfg.get("use_exp_name_paths", True)):
        return cfg

    exp_root = Path(str(common_cfg.get("exp_root", "./data/exp")))
    exp_name = str(common_cfg.get("exp_name", "thesis_stf")).strip() or "thesis_stf"
    exp_dir = exp_root / exp_name
    shared_cache_dir = exp_root / "_shared"

    common_cfg["exp_root"] = _norm_path(exp_root)
    common_cfg["exp_name"] = exp_name
    common_cfg["exp_dir"] = _norm_path(exp_dir)
    common_cfg["shared_cache_dir"] = _norm_path(shared_cache_dir)

    train_cfg = cfg.get("train", {})
    if train_cfg:
        train_cfg["save_dir"] = _norm_path(exp_dir)
        train_cfg["motion_token_cache_dir"] = _norm_path(shared_cache_dir / "flowformer_token_cache")

    test_cfg = cfg.get("test", {})
    if test_cfg:
        test_cfg["checkpoint"] = _norm_path(exp_dir / "best.pth")
        test_cfg["output_csv"] = _norm_path(exp_dir / "test_frame_predictions.csv")
        test_cfg["cache_dir"] = _norm_path(exp_dir / "test_cache")
        test_cfg["motion_token_cache_dir"] = _norm_path(shared_cache_dir / "flowformer_token_cache")
        if str(test_cfg.get("video_output_csv", "")).strip():
            test_cfg["video_output_csv"] = _norm_path(Path(str(test_cfg["video_output_csv"])))

    threshold_cfg = cfg.get("threshold_search", {})
    if threshold_cfg:
        threshold_cfg["val_cache_dir"] = _norm_path(exp_dir / "val_cache")
        threshold_cfg["curve_output_csv"] = _norm_path(exp_dir / "threshold_curve.csv")
        threshold_cfg["summary_output_json"] = _norm_path(exp_dir / "threshold_summary.json")

    flow_cfg = cfg.get("flow_cache", {})
    if flow_cfg:
        flow_cfg["cache_dir"] = _norm_path(shared_cache_dir / "flowformer_token_cache")

    return cfg


def get_default_config() -> Dict[str, Dict[str, Any]]:
    """返回默认配置的深拷贝，避免原地修改污染全局。"""
    cfg = deepcopy(DEFAULT_CONFIG)
    return _apply_exp_name_paths(cfg)


def get_section(section: str) -> Dict[str, Any]:
    cfg = get_default_config()
    if section not in cfg:
        raise KeyError(f"Unknown config section: {section}")
    return cfg[section]
