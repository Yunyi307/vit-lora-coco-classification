import torch
import numpy as np

class MultiLabelBatchAugmenter:
    """多标签 Batch 级别混合增强引擎 (支持变体 MixUp 与 CutMix)"""
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, p=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.p = p

    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def __call__(self, x, y):
        if np.random.rand() > self.p:
            return x, y  # 概率不命中则直接返回原图

        batch_size = x.size(0)
        rand_index = torch.randperm(batch_size).to(x.device)

        if np.random.rand() < 0.5:
            # 执行 MixUp 线性特征混合
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = lam * x + (1 - lam) * x[rand_index]
            y = lam * y + (1 - lam) * y[rand_index]
        else:
            # 执行 CutMix 区域掩码局部剪切
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # 依据实际裁剪边界面积重新修正标签权重系数 lam
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
            y = lam * y + (1 - lam) * y[rand_index]

        return x, y


class DropPath(torch.nn.Module):
    """随机深度 (Stochastic Depth) 核心算子"""
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化掩码矩阵
        return x.div(keep_prob) * random_tensor


def apply_stochastic_depth(model, max_drop_path_rate=0.1):
    """动态在 Hugging Face ViT 主干网络中注入线性递增的 DropPath 概率"""
    if hasattr(model, 'base_model'):  # 兼容解包经过 PEFT 包装的模型
        layers = model.base_model.model.vit.encoder.layer
    else:
        layers = model.vit.encoder.layer

    num_layers = len(layers)
    dp_rates = [x.item() for x in torch.linspace(0, max_drop_path_rate, num_layers)]

    for i, layer in enumerate(layers):
        drop_rate = dp_rates[i]
        if drop_rate > 0:
            drop_path = DropPath(drop_prob=drop_rate)
            old_output_forward = layer.output.forward

            # 动态改写前向传播函数以阻断非残差支路
            def patched_output_forward(hidden_states, input_tensor, old_fn=old_output_forward, dp=drop_path):
                raw_output = old_fn(hidden_states, torch.zeros_like(input_tensor))
                return dp(raw_output) + input_tensor

            layer.output.forward = patched_output_forward
    return model