import os
import random
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_logger(log_dir):
    """统一配置终端与本地文件的日志输出引擎"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed=42):
    """固定全局硬编码种子，确保代码执行结果的可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model, only_trainable=False):
    """精确度量和统计模型参数资产规模"""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_metrics(y_true, y_pred, mode="multilabel"):
    """多标签与单标签统一计算核心分类指标字典"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 计算准确率 (多标签下为 Subset Accuracy 严格全匹配)
    acc = accuracy_score(y_true, y_pred)

    # 宏平均分类指标
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 类级别精细指标
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_per_class": prec_per,
        "recall_per_class": rec_per,
        "f1_per_class": f1_per,
    }