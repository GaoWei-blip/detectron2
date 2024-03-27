import random
import cv2
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer



# 注册数据集
register_coco_instances("dior_train", {}, "data/annotations/train.json", "DIOR/train")
register_coco_instances("dior_val", {}, "data/annotations/val.json", "DIOR/val")
register_coco_instances("dior_test", {}, "data/annotations/test.json", "DIOR/test")


# 获取 DIOR 训练集的元数据
metadata = MetadataCatalog.get("dior_train")

# 获取 DIOR 训练集的样本列表
dataset_dicts = DatasetCatalog.get("dior_train")

# 随机选择几个样本进行可视化
for d in random.sample(dataset_dicts, 3):  # 这里选择可视化 3 个样本
    # 读取图像
    img = cv2.imread(d["file_name"])

    # 使用 Visualizer 进行可视化
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)

    # 显示图像
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.axis("off")
    plt.show()
