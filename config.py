from copy import deepcopy
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "common": {
        "seed": 42,
    },
    "data": {
        "train_root": "./data/train",
        "val_root": "./data/val",
        "test_root": "./data/test",
        "image_size": 224,
        "motion_stride": 2,
        "max_frames_per_video": 0,
    },
    "model": {
        "fusion_mode": "independent",  # options: independent, cross_attention
        "feature_dim": 512,
        "use_resnet_imagenet": False,
        "hfri_mode": "fft",  # options: fft, dct
        "flowformer_repo": "./FlowFormerPlusPlus-main",
        "flowformer_ckpt": "./checkpoints/things.pth",
    },
    "train": {
        "save_dir": "./runs/thesis_stf",
        "resume": "",
        "epochs": 20,
        "batch_size": 8,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "print_freq": 50,
        "save_every": 1,
        "amp": False,
        "aux_loss_weight": 0.5,
        "fusion_loss_weight": 0.1,
        "early_stop_metric": "auc",  # options: auc, acc, loss
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.0,
    },
    "test": {
        "checkpoint": "./runs/thesis_stf/best.pth",
        "output_csv": "./runs/thesis_stf/test_frame_predictions.csv",
        "video_output_csv": "",
        "threshold": 0.5,
        "batch_size": 8,
        "num_workers": 4,
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
