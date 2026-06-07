import os
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模块组件
from config import CONFIG
from dataset import COCOSingleLabelDataset
from modules import MultiLabelBatchAugmenter, apply_stochastic_depth
from utils import set_seed, setup_logger, count_parameters, compute_metrics

# 🚀 控制开关：True 激活进阶组合增强（MixUp/CutMix/随机深度），False 走经典纯净训练
USE_ADVANCED = True


class TrainingTracker:
    """高度自律的指标状态监视器，掌控早停与极值更替"""

    def __init__(self, patience=5):
        self.patience = patience
        self.state = {
            'epoch': 0, 'best_val_f1': 0.0, 'best_epoch': 0, 'patience_counter': 0,
            'train_losses': [], 'val_losses': [], 'val_f1s': [], 'lr_history': []
        }

    def step(self, epoch, train_loss, val_loss, val_f1, lr):
        self.state['epoch'] = epoch
        self.state['train_losses'].append(train_loss)
        self.state['val_losses'].append(val_loss)
        self.state['val_f1s'].append(val_f1)
        self.state['lr_history'].append(lr)

        improved = False
        # 统一强制采用业界更合理的 Val Macro F1 作为考核早停的黄金准则
        if val_f1 > self.state['best_val_f1']:
            self.state['best_val_f1'] = val_f1
            self.state['best_epoch'] = epoch
            self.state['patience_counter'] = 0
            improved = True
        else:
            self.state['patience_counter'] += 1

        return improved, self.state['patience_counter'] >= self.patience


def analyze_and_plot_errors(all_labels, all_probs, output_dir):
    """【核心分析】对混淆不确定区域进行深度扫描并落盘分布热图"""
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    all_preds = (all_probs >= CONFIG["threshold"]).astype(int)
    class_names = CONFIG["class_names"]

    summary = {name: {"FP": 0, "FN": 0, "Hard": 0} for name in class_names}
    hard_samples = []

    for i in range(len(all_labels)):
        for j, name in enumerate(class_names):
            lbl, pred, prob = all_labels[i][j], all_preds[i][j], all_probs[i][j]
            if lbl == 0 and pred == 1: summary[name]["FP"] += 1
            if lbl == 1 and pred == 0: summary[name]["FN"] += 1
            # 犹豫混沌带样本拦截 [0.3, 0.7]
            if 0.3 <= prob <= 0.7:
                summary[name]["Hard"] += 1
                hard_samples.append({"index": i, "class": name, "prob": float(prob)})

    # 可视化错误谱图分布
    plots_path = os.path.join(output_dir, "plots")
    os.makedirs(plots_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(class_names))
    width = 0.2

    ax.bar(x - width, [summary[c]["FP"] for c in class_names], width, label='FP', color='#E24B4A')
    ax.bar(x, [summary[c]["FN"] for c in class_names], width, label='FN', color='#378ADD')
    ax.bar(x + width, [summary[c]["Hard"] for c in class_names], width, label='Hard', color='#EF9F27')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_title("Error Topology Analytics")
    plt.savefig(os.path.join(plots_path, "error_distribution.png"), dpi=150)
    plt.close()
    return summary, hard_samples


def main():
    start_time = time.time()
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 初始化骨架输出目录结构
    out_dir = CONFIG["output_dir"]
    for sub in ["logs", "plots", "best_model", "checkpoints"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    logger = setup_logger(os.path.join(out_dir, "logs"))

    logger.info(f"⚙️ 模式宣告: 当前全量训练流水线启动 [进阶增强模块={USE_ADVANCED}]")

    # 数据就绪
    processor = AutoImageProcessor.from_pretrained(CONFIG["base_model"])
    train_loader = DataLoader(COCOSingleLabelDataset(pd.read_csv(CONFIG["train_csv"]).to_dict("records"), processor),
                              batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(COCOSingleLabelDataset(pd.read_csv(CONFIG["val_csv"]).to_dict("records"), processor),
                            batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 架构实例化
    model = AutoModelForImageClassification.from_pretrained(CONFIG["base_model"], num_labels=CONFIG["num_labels"],
                                                            ignore_mismatched_sizes=True)
    lora_cfg = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], target_modules=["query", "value"],
                          task_type="SEQ_CLS")
    model = get_peft_model(model, lora_cfg)

    # 根据开关执行高能正则化动态编织注入
    if USE_ADVANCED:
        model = apply_stochastic_depth(model, max_drop_path_rate=CONFIG["max_drop_path"])
        augmenter = MultiLabelBatchAugmenter(mixup_alpha=CONFIG["mixup_alpha"], cutmix_alpha=CONFIG["cutmix_alpha"],
                                             p=CONFIG["aug_prob"])

    trainable_params = count_parameters(model, True)
    total_params = count_parameters(model, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * len(train_loader) * CONFIG["epochs"]),
                                                len(train_loader) * CONFIG["epochs"])

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader,
                                                                                val_loader, scheduler)
    criterion = torch.nn.BCEWithLogitsLoss()
    tracker = TrainingTracker(patience=CONFIG["patience"])

    for epoch in range(CONFIG["epochs"]):
        # ---------- 循环训练域 ----------
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}"):
            images, labels = batch["pixel_values"], batch["labels"].float()

            if USE_ADVANCED:
                images, labels = augmenter(images, labels)

            optimizer.zero_grad()
            loss = criterion(model(pixel_values=images).logits, labels)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # ---------- 闭环验证域 ----------
        model.eval()
        val_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch["pixel_values"], batch["labels"].float()
                logits = model(pixel_values=imgs).logits
                val_loss += criterion(logits, labels).item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 指标度量汇聚
        metrics = compute_metrics(all_labels, (np.array(all_probs) >= CONFIG["threshold"]).astype(int),
                                  mode="multilabel")
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        improved, stop = tracker.step(epoch, avg_train, avg_val, metrics["f1_macro"], optimizer.param_groups[0]['lr'])
        logger.info(
            f"Epoch {epoch + 1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val F1: {metrics['f1_macro']:.4f}")

        # 持久化轮次 Checkpoint
        accelerator.unwrap_model(model).save_pretrained(os.path.join(out_dir, "checkpoints", f"epoch_{epoch + 1}"))

        if improved:
            accelerator.unwrap_model(model).save_pretrained(os.path.join(out_dir, "best_model"))
            logger.info("🏆 核心验证集 F1 指标刷新，记录安全入库。")

        if stop:
            logger.warning("🚨 触发容忍度上界，多轮性能未见突破，启动提早刹车机制。")
            break

    # ===================== 全盘深度复盘报告生成 =====================
    logger.info("全链闭幕，正在进行全面复盘及硬件开销审计...")
    err_summary, hard_list = analyze_and_plot_errors(all_labels, all_probs, out_dir)

    # 算力显存审计
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    auc_scores = [roc_auc_score(np.array(all_labels)[:, i], np.array(all_probs)[:, i]) for i in
                  range(CONFIG["num_labels"])]
    mcm = multilabel_confusion_matrix(all_labels, (np.array(all_probs) >= CONFIG["threshold"]).astype(int))

    final_report = {
        'meta': {
            'experiment_id': f"EXP_COCO_{datetime.now().strftime('%Y%m%d_%H%M')}",
            'advanced_mode_enabled': USE_ADVANCED,
            'best_epoch': tracker.state['best_epoch'] + 1,
            'time_cost_hours': round((time.time() - start_time) / 3600, 4)
        },
        'hardware_and_weights': {
            'peak_gpu_memory_gb': round(peak_mem_gb, 3),
            'trainable_params_count': trainable_params,
            'total_params_count': total_params,
            'trainable_ratio': f"{100 * trainable_params / total_params:.2f}%"
        },
        'final_performance': {
            'subset_accuracy': metrics["accuracy"],
            'f1_macro': metrics["f1_macro"],
            'auc_roc_per_class': dict(zip(CONFIG["class_names"], auc_scores)),
            'confusion_matrix': {CONFIG["class_names"][i]: mcm[i].tolist() for i in range(CONFIG["num_labels"])}
        },
        'error_analysis': {
            'class_error_counts': err_summary,
            'total_hard_samples': len(hard_list)
        }
    }

    with open(os.path.join(out_dir, "final_pipeline_report.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    logger.info(f"🎉 恭喜！流水线完整关闭。全量度量及可视化文件均安全存储于: {out_dir}")


if __name__ == "__main__":
    main()