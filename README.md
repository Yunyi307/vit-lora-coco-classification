本项目是一个基于 Hugging Face `transformers`、`peft` (LoRA) 以及 `accelerate` 构建的高性能多标签图像分类系统。项目主要用于对 COCO 数据集中的 `person`（人）和 `car`（车）进行高效微调与鲁棒性识别。
数据路径校验：在运行任何代码前，请打开 config.py，确保 "train_csv" 和 "val_csv" 的路径正确指向了你的 COCO 数据集索引 CSV 文件。
1.config.py: 定义了预训练模型路径、训练数据源（train_csv, val_csv）、训练超参数（Batch Size, Epochs, LR）以及 LoRA 配置参数（lora_r, lora_alpha）。
2.dataset.py: 实现 COCOSingleLabelDataset。可自动识别 CSV 文件中的 label（单标签整型）或 label_list（多标签逗号分隔字符串），并调用 AutoImageProcessor 进行标准张量化转换。
3.modules.py:MultiLabelBatchAugmenter: 实现批次级别的 MixUp 与 CutMix。apply_stochastic_depth: 动态解包 PEFT 封装，并在 ViT 主干网络的 Encoder 层中无缝注入线性递增的随机深度。
4.hyperparam_tune.py: 包含三个阶段：回归测试验证连通性 $\rightarrow$ LR Finder 寻找最优学习率 $\rightarrow$ 遍历 r 与 alpha 的网格搜索。最终保存双子图可视化分析图表。
5.train_pipeline.py: 生产级完整训练管线。提供开关 USE_ADVANCED 以一键激活进阶数据增强。自带 TrainingTracker 实施早停与指标监控，并在结束后导出详尽的 JSON 格式复盘实验报告。
