import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from config import CONFIG
from dataset import COCOSingleLabelDataset
from utils import set_seed, setup_logger, compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tune_model(r, alpha):
    """根据指定的超参动态配置构建微调 LoRA 模型"""
    base = AutoModelForImageClassification.from_pretrained(
        CONFIG["base_model"], num_labels=CONFIG["num_labels"], ignore_mismatched_sizes=True
    )
    lora_cfg = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=["query", "key", "value", "output.dense"],
        lora_dropout=CONFIG["lora_dropout"], bias="none", task_type="SEQ_CLS"
    )
    return get_peft_model(base, lora_cfg)


def get_subset_loader(dataset, ratio, batch_size, shuffle=True):
    """高灵活性按配额截取轻量化子集加载器"""
    n = max(1, int(len(dataset) * ratio))
    indices = torch.randperm(len(dataset))[:n].tolist()
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, num_workers=4)


def run_lr_finder(loader, logger):
    """【阶段2】启动指数缩放单向步进的学习率探测器 (LR Finder)"""
    logger.info(">>> 启动 LR Finder 探测器流程...")
    model = build_tune_model(16, 32).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-7)

    num_iter = 100
    gamma = (10.0 / 1e-7) ** (1.0 / (num_iter - 1))
    lrs, losses = [], []
    best_loss, avg_loss, beta = float("inf"), 0.0, 0.98

    loader_iter = iter(loader)
    model.train()

    for i in range(num_iter):
        lr = 1e-7 * (gamma ** i)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader); batch = next(loader_iter)

        pv, lbl = batch["pixel_values"].to(DEVICE), batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        loss = nn.BCEWithLogitsLoss()(model(pixel_values=pv).logits, lbl.float())
        loss.backward()
        optimizer.step()

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smooth = avg_loss / (1 - beta ** (i + 1))

        if i > 5 and smooth > 4 * best_loss: break
        if smooth < best_loss: best_loss = smooth

        lrs.append(lr)
        losses.append(smooth)

    grads = np.gradient(np.array(losses))
    suggest_lr = lrs[np.argmin(grads)]
    logger.info(f"✅ LR Finder 完毕。最优梯度下降切点 LR 推荐值为: {suggest_lr:.2e}")
    return lrs, losses, suggest_lr


def run_grid_search(train_ds, val_ds, logger):
    """【阶段3】基于离散核心超参数组合的小规模网格调优搜索"""
    logger.info(">>> 启动多组合超参数对比微型网格搜索...")
    tune_combinations = [
        {"lora_r": 8, "lora_alpha": 16, "lr": 3e-4},
        {"lora_r": 16, "lora_alpha": 32, "lr": 3e-4},
        {"lora_r": 32, "lora_alpha": 64, "lr": 1e-3},
        {"lora_r": 64, "lora_alpha": 128, "lr": 3e-3},
    ]

    t_loader = get_subset_loader(train_ds, 0.05, CONFIG["batch_size"])
    v_loader = get_subset_loader(val_ds, 0.05, CONFIG["batch_size"], shuffle=False)
    results = []

    for idx, params in enumerate(tune_combinations):
        logger.info(f"正在测试组合 [{idx + 1}/{len(tune_combinations)}]: r={params['lora_r']}, lr={params['lr']}")
        model = build_tune_model(params["lora_r"], params["lora_alpha"]).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=params["lr"])
        criterion = nn.BCEWithLogitsLoss()

        t_losses, v_f1s = [], []
        for epoch in range(3):  # 每组超参快速跑3个轮次
            model.train()
            epoch_loss = 0.0
            for batch in t_loader:
                pv, lbl = batch["pixel_values"].to(DEVICE), batch["labels"].to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(pixel_values=pv).logits, lbl.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in v_loader:
                    pv, lbl = batch["pixel_values"].to(DEVICE), batch["labels"].to(DEVICE)
                    logits = model(pixel_values=pv).logits
                    all_preds.extend((torch.sigmoid(logits) >= CONFIG["threshold"]).int().cpu().numpy())
                    all_labels.extend(lbl.cpu().numpy())

            metrics = compute_metrics(all_labels, all_preds, mode="multilabel")
            t_losses.append(epoch_loss / len(t_loader))
            v_f1s.append(metrics["f1_macro"])

        results.append({**params, "best_f1": max(v_f1s), "t_losses": t_losses, "v_f1s": v_f1s})
    return results


def main():
    set_seed(42)
    log_dir = os.path.join(CONFIG["output_dir"], "logs")
    plot_dir = os.path.join(CONFIG["output_dir"], "plots")
    os.makedirs(plot_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    processor = AutoImageProcessor.from_pretrained(CONFIG["base_model"])
    train_ds = COCOSingleLabelDataset(pd.read_csv(CONFIG["train_csv"]).to_dict("records"), processor)
    val_ds = COCOSingleLabelDataset(pd.read_csv(CONFIG["val_csv"]).to_dict("records"), processor)

    # 【阶段1】流水线连通性回归测试
    logger.info("【阶段1】启动极小规模快速回归连通性测试...")
    quick_loader = get_subset_loader(train_ds, 0.01, CONFIG["batch_size"])
    model = build_tune_model(8, 16).to(DEVICE)
    batch = next(iter(quick_loader))
    out = model(pixel_values=batch["pixel_values"].to(DEVICE))
    assert out.logits.shape[1] == CONFIG["num_labels"], "模型输出维度异常！"
    logger.info("✅ 阶段1：代码连通性完美验证通过。")

    # 执行阶段 2 & 3
    finder_loader = get_subset_loader(train_ds, 0.02, CONFIG["batch_size"])
    lrs, losses, suggest_lr = run_lr_finder(finder_loader, logger)
    tune_results = run_grid_search(train_ds, val_ds, logger)

    # 输出整合后的报告图表
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lrs, losses, color='#1D9E75')
    plt.axvline(suggest_lr, color='#E24B4A', linestyle='--', label=f'Rec LR: {suggest_lr:.1e}')
    plt.xscale('log')
    plt.title("LR Finder Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    labels = [f"r={r['lora_r']}\nlr={r['lr']:.0e}" for r in tune_results]
    f1s = [r["best_f1"] for r in tune_results]
    plt.bar(labels, f1s, color='#378ADD', width=0.4)
    plt.title("Hyperparameter Best F1 Comparison")
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "tuning_summary_report.png"), dpi=150)
    plt.close()
    logger.info("🎉 搜索完毕！超参检索全链路汇总图表已成功落盘存盘。")


if __name__ == "__main__":
    main()