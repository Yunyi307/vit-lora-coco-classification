import os
import argparse
import pandas as pd
import torch
import torch.nn as nn


from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
from dataset import COCOSingleLabelDataset
from utils import (
    setup_logger, create_dirs, set_seed, count_parameters,
    get_classification_report, plot_confusion_matrix,
    compute_multilabel_metrics, compute_singlelabel_metrics
)

# ===================== 参数解析 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="ViT LoRA 图像分类训练脚本")
    parser.add_argument("--mode", type=str, default="multilabel",
                        choices=["singlelabel", "multilabel"],
                        help="分类模式：singlelabel（单标签）或 multilabel（多标签）")
    return parser.parse_args()

# ===================== 配置 =====================
CONFIG = {
    "base_model": "google/vit-base-patch16-224",
    "train_csv": "../coco_person_car_train.csv",
    "val_csv": "../coco_person_car_val.csv",
    "num_labels": 2,       # person 和 car 两个标签
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-4,
    "base_dir": "../outputs",
    "save_dir": "../best_lora_model",
    "log_dir": "../outputs/logs",
    "multilabel_threshold": 0.5,   # 多标签分类的阈值
}

# ===================== 主训练流程 =====================
def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    args = parse_args()

    set_seed(42)
    create_dirs(CONFIG["base_dir"], ["models", "logs"])
    logger = setup_logger(CONFIG["log_dir"])
    accelerator = Accelerator(mixed_precision="fp16")

    logger.info(f"🚀 训练模式: {args.mode}")

    # ---- 图像处理器 ----
    image_processor = AutoImageProcessor.from_pretrained(CONFIG["base_model"])

    # ---- 加载CSV ----
    train_df = pd.read_csv(CONFIG["train_csv"])
    val_df   = pd.read_csv(CONFIG["val_csv"])

    train_dataset = COCOSingleLabelDataset(
        train_df.to_dict("records"), image_processor, transform=None, mode="train"
    )
    val_dataset = COCOSingleLabelDataset(
        val_df.to_dict("records"), image_processor, transform=None, mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4
    )

    # ---- 模型 ----
    model = AutoModelForImageClassification.from_pretrained(
        CONFIG["base_model"],
        num_labels=CONFIG["num_labels"],
        ignore_mismatched_sizes=True
    )

    # ---- LoRA 配置 ----
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    peft_model = get_peft_model(model, lora_config)
    count_parameters(peft_model, only_trainable=True)

    # ---- 损失函数 ----
    if args.mode == "multilabel":
        criterion = nn.BCEWithLogitsLoss()
        logger.info("损失函数: BCEWithLogitsLoss（多标签）")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("损失函数: CrossEntropyLoss（单标签）")

    # ---- 优化器 & 调度器 ----
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=CONFIG["lr"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # ---- Accelerator 准备 ----
    peft_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        peft_model, optimizer, train_loader, val_loader, scheduler
    )

    # ======训练循环===
    best_acc = 0.0
    best_f1  = 0.0

    for epoch in range(CONFIG["epochs"]):
        # ---------- 训练阶段 ----------
        peft_model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{CONFIG['epochs']}"):
            with accelerator.accumulate(peft_model):
                pixel_values = batch["pixel_values"]
                labels       = batch["labels"]

                # 前向传播
                outputs = peft_model(pixel_values=pixel_values)

                # 计算损失
                if args.mode == "multilabel":
                    # 多标签：labels 需为 float 类型 (BCEWithLogitsLoss)
                    loss = criterion(outputs.logits, labels.float())
                else:
                    # 单标签：labels 需为 long 类型 (CrossEntropyLoss)
                    loss = criterion(outputs.logits, labels.long())

                # 反向传播
                accelerator.backward(loss)

                # 优化器步进
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ==验证阶段==
        peft_model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val   Epoch {epoch+1}/{CONFIG['epochs']}"):
                pixel_values = batch["pixel_values"]
                labels       = batch["labels"]

                outputs = peft_model(pixel_values=pixel_values)

                # 计算验证损失
                if args.mode == "multilabel":
                    loss = criterion(outputs.logits, labels.float())
                else:
                    loss = criterion(outputs.logits, labels.long())

                val_loss += loss.item()

                # 收集预测结果
                if args.mode == "multilabel":
                    # 多标签：sigmoid + 阈值
                    probs = torch.sigmoid(outputs.logits)
                    preds = (probs >= CONFIG["multilabel_threshold"]).int()
                else:
                    # 单标签：argmax
                    preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(
                    accelerator.gather(preds).cpu().numpy()
                )
                all_labels.extend(
                    accelerator.gather(labels).cpu().numpy()
                )

        avg_val_loss = val_loss / len(val_loader)

        #计算验证指标
        if args.mode == "multilabel":
            metrics = compute_multilabel_metrics(all_labels, all_preds)
            monitor_metric = metrics["f1_macro"]
            logger.info(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc(subset): {metrics['accuracy']:.4f} | "
                f"Val Precision: {metrics['precision_macro']:.4f} | "
                f"Val Recall: {metrics['recall_macro']:.4f} | "
                f"Val F1(macro): {metrics['f1_macro']:.4f}"
            )
        else:
            metrics = compute_singlelabel_metrics(all_labels, all_preds)
            monitor_metric = metrics["accuracy"]
            logger.info(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {metrics['accuracy']:.4f} | "
                f"Val Precision(macro): {metrics['precision_macro']:.4f} | "
                f"Val Recall(macro): {metrics['recall_macro']:.4f} | "
                f"Val F1(macro): {metrics['f1_macro']:.4f}"
            )

        # 保存最佳模型
        improved = (
            (args.mode == "multilabel"  and monitor_metric > best_f1) or
            (args.mode == "singlelabel" and monitor_metric > best_acc)
        )
        if improved:
            best_acc = monitor_metric if args.mode == "singlelabel" else best_acc
            best_f1  = monitor_metric if args.mode == "multilabel"  else best_f1
            peft_model.save_pretrained(CONFIG["save_dir"])
            image_processor.save_pretrained(CONFIG["save_dir"])
            logger.info(f"✅ Best model saved | metric: {monitor_metric:.4f}")

    # =====训练结束===
    logger.info(f"\n🎉 Training finished!")
    if args.mode == "multilabel":
        logger.info(f"Best F1(macro): {best_f1:.4f}")
    else:
        logger.info(f"Best Acc: {best_acc:.4f}")

    # 输出分类报告
    get_classification_report(all_labels, all_preds, mode=args.mode)
    if args.mode == "singlelabel":
        plot_confusion_matrix(
            all_labels, all_preds,
            save_path=os.path.join(CONFIG["base_dir"], "confusion.png")
        )


if __name__ == "__main__":
    main()