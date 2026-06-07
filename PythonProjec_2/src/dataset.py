import torch
from torch.utils.data import Dataset
from PIL import Image

class COCOSingleLabelDataset(Dataset):
    """
    COCO 图像分类数据集封装。
    支持单标签（label列，整数）和多标签（label_list列，逗号分隔字符串）两种格式。
    """
    def __init__(self, csv_data, image_processor):
        self.csv_data = csv_data
        self.image_processor = image_processor

        # 自动感知标签列模式
        first_item = csv_data[0]
        if "label_list" in first_item:
            self.label_mode = "multilabel"
        elif "label" in first_item:
            self.label_mode = "singlelabel"
        else:
            raise ValueError("CSV数据中未包含 'label' 或 'label_list' 列，请检查输入源！")

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = self.csv_data[idx]

        # 读取并转换图像通道
        img_path = data["file_path"]
        image = Image.open(img_path).convert("RGB")

        # 借助 HuggingFace Processor 转换标准化张量
        inputs = self.image_processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # 根据模式解析标签
        if self.label_mode == "multilabel":
            # 例如 "1,0" → [1.0, 0.0]
            label_str = str(data["label_list"])
            label = torch.tensor(
                [int(x) for x in label_str.split(",")],
                dtype=torch.float32
            )
        else:
            # 单标签整数模式
            label = torch.tensor(int(data["label"]), dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": label}