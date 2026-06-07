import os

CONFIG = {
    # 模型与数据路径
    "base_model": "google/vit-base-patch16-224",
    "train_csv": "../coco_person_car_train.csv",
    "val_csv": "../coco_person_car_val.csv",

    # 任务元信息
    "num_labels": 2,
    "class_names": ["person", "car"],
    "threshold": 0.5,

    # 核心训练超参数 (经过超参搜索优化后的最优值)
    "batch_size": 32,
    "epochs": 20,
    "patience": 5,
    "lr": 3e-4,

    # LoRA 结构微调参数
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # 进阶数据增强与正则化超参数
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    "aug_prob": 0.5,  # 触发 MixUp/CutMix 的总概率
    "max_drop_path": 0.1,  # 随机深度递增的最大丢弃率

    # 输出根目录
    "output_dir": "../outputs_pipeline",
}