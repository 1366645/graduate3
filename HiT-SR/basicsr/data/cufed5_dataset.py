import os
from os import path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from basicsr.utils.registry import DATASET_REGISTRY

def default_loader(path):
    return Image.open(path).convert('RGB')

@DATASET_REGISTRY.register()
class CUFED5Dataset(Dataset):
    """
    CUFED5 数据集，用于参考图像超分任务。

    文件夹结构示例：
        datasets/CUFED5/GT/       -> 高分辨率图像
        datasets/CUFED5/LQ/       -> 低分辨率图像
        datasets/CUFED5/REF/      -> 参考图像 (文件名格式: '0001_ref.png' 与 GT '0001.png' 对应)

    构造函数接收配置字典 opt，其中需包含以下键：
        dataroot_gt (str): GT 图像目录路径。
        dataroot_lq (str): 低分辨率图像目录路径。
        dataroot_ref (str): 参考图像目录路径。
        filename_tmpl (str): 文件名模板，如 '{}.png' （默认）。
        transform (callable, optional): 数据预处理变换。
        loader (callable, optional): 图像读取函数，默认为 PIL.Image 的加载。
    """
    def __init__(self, opt):
        super(CUFED5Dataset, self).__init__()
        # 从配置字典中读取必要参数
        self.dataroot_gt = opt['dataroot_gt']
        self.dataroot_lq = opt['dataroot_lq']
        self.dataroot_ref = opt['dataroot_ref']
        self.filename_tmpl = opt.get('filename_tmpl', '{}.png')
        self.transform = opt.get('transform', None)
        self.loader = opt.get('loader', default_loader)

        # 获取所有 GT 文件（假设扩展名为 png 或 jpg）
        self.gt_files = sorted([
            f for f in os.listdir(self.dataroot_gt)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if len(self.gt_files) == 0:
            raise ValueError(f"No image files found in GT folder: {self.dataroot_gt}")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        gt_filename = self.gt_files[index]
        basename, ext = os.path.splitext(gt_filename)
        gt_path = osp.join(self.dataroot_gt, gt_filename)
        lq_path = osp.join(self.dataroot_lq, self.filename_tmpl.format(basename))
        ref_filename = basename + ext
        ref_path = osp.join(self.dataroot_ref, ref_filename)

        gt = self.loader(gt_path)
        lq = self.loader(lq_path)
        ref = self.loader(ref_path)

        # 统一调整尺寸（例如全部 resize 到 128x128）
        desired_size = (128, 128)
        gt = gt.resize(desired_size, Image.BICUBIC)
        lq = lq.resize(desired_size, Image.BICUBIC)
        ref = ref.resize(desired_size, Image.BICUBIC)

        if self.transform:
            gt = self.transform(gt)
            lq = self.transform(lq)
            ref = self.transform(ref)
        else:
            gt = np.array(gt).astype(np.float32) / 255.
            lq = np.array(lq).astype(np.float32) / 255.
            ref = np.array(ref).astype(np.float32) / 255.
            gt = torch.from_numpy(gt.transpose(2, 0, 1))
            lq = torch.from_numpy(lq.transpose(2, 0, 1))
            ref = torch.from_numpy(ref.transpose(2, 0, 1))
        return {'gt': gt, 'lq': lq, 'ref': ref,
                'gt_path': gt_path, 'lq_path': lq_path, 'ref_path': ref_path}
