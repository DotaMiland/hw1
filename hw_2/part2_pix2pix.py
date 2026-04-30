from __future__ import annotations  # 让类型注解可以更灵活地引用后面定义的类型

import argparse  # 用来读取命令行参数
import json  # 用来保存训练历史等 JSON 文件
import random  # 用来做随机打乱和数据增强中的随机裁剪
from dataclasses import dataclass  # 用来写更简洁的数据类
from pathlib import Path  # 更方便地处理文件路径
from typing import List, Sequence  # 类型提示，帮助读代码

import cv2  # OpenCV：负责读图、缩放、存图
import numpy as np  # NumPy：负责数组处理
import torch  # PyTorch：负责张量、网络、梯度和优化
import torch.nn as nn  # 神经网络模块，卷积层、BatchNorm、损失函数都在这里
from torch.utils.data import DataLoader, Dataset  # Dataset 负责定义数据集，DataLoader 负责按批读取


def seed_everything(seed: int) -> None:  # 固定随机种子，尽量让实验更可复现
    random.seed(seed)  # 固定 Python 自带 random 的随机性
    np.random.seed(seed)  # 固定 NumPy 的随机性
    torch.manual_seed(seed)  # 固定 PyTorch 的随机性


@dataclass(frozen=True)  # dataclass 会自动帮我们生成初始化方法；frozen=True 表示创建后不允许修改
class PairRecord:  # 用来表示一组配对样本的信息
    stem: str  # 文件名主干，比如 cmp_b0001
    input_path: Path  # 输入图路径，比如 cmp_b0001.png
    target_path: Path  # 目标图路径，比如 cmp_b0001.jpg


class CMPFacadeDataset(Dataset):  # 自定义数据集类，专门读取 CMP Facades 数据
    def __init__(self, records: Sequence[PairRecord], image_size: int = 256, load_size: int = 286, augment: bool = False):  # 初始化数据集
        self.records = list(records)  # 把输入的样本记录转成列表保存下来
        self.image_size = image_size  # 最终喂给网络的图像尺寸
        self.load_size = load_size  # 做随机裁剪前，先缩放到的尺寸
        self.augment = augment  # 是否启用数据增强

    def __len__(self) -> int:  # 告诉 DataLoader 这个数据集里有多少样本
        return len(self.records)  # 样本数量就是 records 列表长度

    def __getitem__(self, index: int) -> dict:  # DataLoader 取第 index 个样本时，会调用这里
        record = self.records[index]  # 先拿到这条样本的路径信息
        input_image = cv2.imread(str(record.input_path), cv2.IMREAD_COLOR)  # 读取输入图
        target_image = cv2.imread(str(record.target_path), cv2.IMREAD_COLOR)  # 读取目标图
        if input_image is None or target_image is None:  # 如果任意一张图读取失败
            raise FileNotFoundError(f"Failed to read pair: {record.stem}")  # 就直接报错，提示是哪组样本有问题

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # OpenCV 默认读出来是 BGR，这里转成 RGB
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)  # 目标图同理也转成 RGB

        if self.augment:  # 如果启用了数据增强
            input_image = cv2.resize(input_image, (self.load_size, self.load_size), interpolation=cv2.INTER_NEAREST)  # 输入图先缩放到 load_size
            target_image = cv2.resize(target_image, (self.load_size, self.load_size), interpolation=cv2.INTER_AREA)  # 目标图也缩放到 load_size

            max_offset = self.load_size - self.image_size  # 随机裁剪时，左上角最多能偏移多少像素
            offset_x = random.randint(0, max_offset)  # 随机决定裁剪框的横向起点
            offset_y = random.randint(0, max_offset)  # 随机决定裁剪框的纵向起点
            input_image = input_image[offset_y : offset_y + self.image_size, offset_x : offset_x + self.image_size]  # 按随机框裁剪输入图
            target_image = target_image[offset_y : offset_y + self.image_size, offset_x : offset_x + self.image_size]  # 目标图必须用同一个裁剪框，才能保持配对关系

            if random.random() < 0.5:  # 50% 概率做水平翻转
                input_image = np.ascontiguousarray(input_image[:, ::-1])  # 对输入图做左右翻转，并确保内存连续
                target_image = np.ascontiguousarray(target_image[:, ::-1])  # 目标图也做同样翻转
        else:  # 如果不做增强
            input_image = cv2.resize(input_image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)  # 直接缩放到最终尺寸
            target_image = cv2.resize(target_image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)  # 目标图也直接缩放到最终尺寸

        input_tensor = torch.from_numpy(input_image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)  # 把输入图转成 float 张量，并归一化到 [-1, 1]，再从 HWC 变成 CHW
        target_tensor = torch.from_numpy(target_image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)  # 目标图也做同样处理
        return {  # 返回一个字典，DataLoader 会把同名字段自动拼成 batch
            "input": input_tensor,  # 输入图张量
            "target": target_tensor,  # 目标图张量
            "stem": record.stem,  # 文件主干，后面保存预测图时会用到
        }  # 字典结束


def discover_pairs(data_dir: Path) -> List[PairRecord]:  # 在数据目录里自动寻找成对的 png / jpg 样本
    records: List[PairRecord] = []  # 准备一个空列表，用来存放所有找到的样本对
    for image_path in sorted(data_dir.glob("cmp_b*.png")):  # 找到所有形如 cmp_bXXXX.png 的输入图
        stem = image_path.stem  # 取文件主干，比如 cmp_b0001
        target_path = data_dir / f"{stem}.jpg"  # 假设对应目标图文件名相同，只是扩展名变成 .jpg
        if target_path.exists():  # 如果这个目标图真的存在
            records.append(PairRecord(stem=stem, input_path=image_path, target_path=target_path))  # 就把这对样本记录下来
    if not records:  # 如果一对都没找到
        raise FileNotFoundError(f"No paired cmp_b*.png/jpg samples found in {data_dir}")  # 直接报错提醒数据目录不对
    return records  # 返回所有配对样本


def split_records(records: Sequence[PairRecord], train_ratio: float, seed: int) -> tuple[list[PairRecord], list[PairRecord]]:  # 把全部样本切成训练集和验证集
    shuffled = list(records)  # 先复制一份，避免改动原始列表
    rng = random.Random(seed)  # 创建一个带固定种子的随机数生成器
    rng.shuffle(shuffled)  # 原地随机打乱样本顺序
    train_count = max(1, int(len(shuffled) * train_ratio))  # 按比例算出训练集应该有多少样本，至少保留 1 个
    train_records = shuffled[:train_count]  # 前半部分作为训练集
    val_records = shuffled[train_count:]  # 后半部分作为验证集
    if not val_records:  # 如果验证集被分成了空列表
        val_records = shuffled[-max(1, len(shuffled) // 10) :]  # 就至少拿最后 10% 或 1 个样本出来当验证集
        train_records = shuffled[: len(shuffled) - len(val_records)]  # 剩下的再作为训练集
    return train_records, val_records  # 返回训练集和验证集


class DownBlock(nn.Module):  # U-Net 编码器里的一个下采样模块
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):  # 初始化一个下采样块
        super().__init__()  # 先初始化父类 nn.Module
        layers: List[nn.Module] = [  # 准备一组按顺序执行的层
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize),  # 4x4 卷积，步长 2，相当于做一次下采样
        ]  # 列表暂时结束
        if normalize:  # 如果需要归一化
            layers.append(nn.BatchNorm2d(out_channels))  # 就加一层 BatchNorm
        layers.append(nn.LeakyReLU(0.2, inplace=True))  # 再加一个 LeakyReLU 激活函数
        self.block = nn.Sequential(*layers)  # 把这些层打包成一个顺序网络

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播
        return self.block(x)  # 输入 x 依次经过上面的层并返回结果


class UpBlock(nn.Module):  # U-Net 解码器里的一个上采样模块
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):  # 初始化一个上采样块
        super().__init__()  # 初始化父类 nn.Module
        layers: List[nn.Module] = [  # 准备上采样阶段的层
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),  # 反卷积，把特征图尺寸放大 2 倍
            nn.BatchNorm2d(out_channels),  # 做 BatchNorm 稳定训练
            nn.ReLU(inplace=True),  # 再接一个普通 ReLU 激活
        ]  # 列表暂时结束
        if dropout:  # 如果这一层需要随机失活
            layers.append(nn.Dropout(0.5))  # 就加一个 Dropout
        self.block = nn.Sequential(*layers)  # 打包成顺序网络

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播
        return self.block(x)  # 输入 x 经过上采样块并返回结果


class UNetGenerator(nn.Module):  # 生成器：一个轻量版 U-Net
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):  # 初始化生成器
        super().__init__()  # 初始化父类 nn.Module
        self.down1 = DownBlock(in_channels, base_channels, normalize=False)  # 第 1 层下采样，不做 BatchNorm
        self.down2 = DownBlock(base_channels, base_channels * 2)  # 第 2 层下采样，通道数翻倍
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)  # 第 3 层下采样
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)  # 第 4 层下采样
        self.down5 = DownBlock(base_channels * 8, base_channels * 8)  # bottleneck 前最后一层下采样

        self.up1 = UpBlock(base_channels * 8, base_channels * 8, dropout=True)  # 第 1 层上采样，带 Dropout
        self.up2 = UpBlock(base_channels * 16, base_channels * 4, dropout=True)  # 第 2 层上采样，输入通道翻倍是因为要拼接跳跃连接
        self.up3 = UpBlock(base_channels * 8, base_channels * 2)  # 第 3 层上采样
        self.up4 = UpBlock(base_channels * 4, base_channels)  # 第 4 层上采样
        self.final = nn.Sequential(  # 最后再做一次上采样并输出 3 通道图像
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),  # 最后一层反卷积，把尺寸恢复到原图大小
            nn.Tanh(),  # 用 Tanh 把输出限制在 [-1, 1]
        )  # 输出层结束

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义生成器的前向传播
        d1 = self.down1(x)  # 第 1 次下采样
        d2 = self.down2(d1)  # 第 2 次下采样
        d3 = self.down3(d2)  # 第 3 次下采样
        d4 = self.down4(d3)  # 第 4 次下采样
        bottleneck = self.down5(d4)  # 最底部瓶颈层

        u1 = self.up1(bottleneck)  # 第 1 次上采样
        u2 = self.up2(torch.cat([u1, d4], dim=1))  # 把上采样结果和对应编码层特征在通道维拼接，再做第 2 次上采样
        u3 = self.up3(torch.cat([u2, d3], dim=1))  # 同理继续拼接跳跃连接
        u4 = self.up4(torch.cat([u3, d2], dim=1))  # 同理继续拼接跳跃连接
        return self.final(torch.cat([u4, d1], dim=1))  # 最后再拼一次最浅层特征，输出生成图像


class PatchDiscriminator(nn.Module):  # 判别器：PatchGAN，不是判整张图真伪，而是判很多局部 patch 的真伪
    def __init__(self, in_channels: int = 6, base_channels: int = 32):  # 初始化判别器
        super().__init__()  # 初始化父类 nn.Module
        self.model = nn.Sequential(  # 按顺序堆几层卷积
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),  # 第一层卷积，输入通道是 source 和 target 拼起来的 6 通道
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 第二层卷积
            nn.BatchNorm2d(base_channels * 2),  # BatchNorm
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 第三层卷积
            nn.BatchNorm2d(base_channels * 4),  # BatchNorm
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),  # 第四层卷积，步长改成 1
            nn.BatchNorm2d(base_channels * 8),  # BatchNorm
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),  # 最后一层输出 1 通道“真假得分图”
        )  # 顺序网络结束

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # 定义判别器前向传播
        return self.model(torch.cat([source, target], dim=1))  # 先把条件图和目标图沿通道维拼起来，再送进判别器


def init_weights(module: nn.Module) -> None:  # 按 GAN 里常见的方式初始化卷积层参数
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):  # 如果当前模块是卷积层或反卷积层
        nn.init.normal_(module.weight.data, 0.0, 0.02)  # 权重用均值 0、标准差 0.02 的正态分布初始化
        if module.bias is not None:  # 如果这一层有 bias
            nn.init.zeros_(module.bias.data)  # 就把 bias 初始化为 0
    elif isinstance(module, nn.BatchNorm2d):  # 如果当前模块是 BatchNorm
        nn.init.normal_(module.weight.data, 1.0, 0.02)  # gamma 用均值 1、标准差 0.02 的正态分布初始化
        nn.init.zeros_(module.bias.data)  # beta 初始化为 0


def denormalize(image: torch.Tensor) -> torch.Tensor:  # 把 [-1, 1] 范围的图像张量恢复成 [0, 255] 的 uint8 像素
    return ((image.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)  # 先截断到 [-1, 1]，再映射到 [0, 255]，最后转 uint8


def save_tensor_image(image: torch.Tensor, image_path: Path) -> None:  # 把单张张量图像保存到磁盘
    image_path.parent.mkdir(parents=True, exist_ok=True)  # 如果输出目录不存在，就先创建
    image_np = denormalize(image.cpu()).permute(1, 2, 0).numpy()  # 张量搬回 CPU，再从 CHW 变成 HWC，转成 NumPy 数组
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # OpenCV 保存前需要先转成 BGR
    cv2.imwrite(str(image_path), bgr)  # 把图像写到磁盘


def save_triptych(source: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, image_path: Path) -> None:  # 把输入图、预测图、目标图拼成一张对比图保存
    source_np = denormalize(source.cpu()).permute(1, 2, 0).numpy()  # 输入图转成可保存的 NumPy 图像
    prediction_np = denormalize(prediction.cpu()).permute(1, 2, 0).numpy()  # 预测图转成可保存的 NumPy 图像
    target_np = denormalize(target.cpu()).permute(1, 2, 0).numpy()  # 目标图转成可保存的 NumPy 图像
    merged = np.concatenate([source_np, prediction_np, target_np], axis=1)  # 沿水平方向把三张图拼起来
    image_path.parent.mkdir(parents=True, exist_ok=True)  # 如果输出目录不存在，就先创建
    cv2.imwrite(str(image_path), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))  # 转成 BGR 并保存


def evaluate_model(  # 在验证集上评估当前生成器的表现
    generator: nn.Module,  # 生成器模型
    dataloader: DataLoader,  # 验证集 DataLoader
    device: torch.device,  # 运行设备
    l1_loss: nn.Module,  # 这里复用训练时定义的 L1Loss
) -> dict:  # 返回一个字典，里面有验证集指标
    generator.eval()  # 切换到评估模式，关闭 Dropout，BatchNorm 也使用统计值
    total_l1 = 0.0  # 用来累计所有 batch 的 L1 损失
    total_psnr = 0.0  # 用来累计所有 batch 的 PSNR
    count = 0  # 统计一共评估了多少个 batch
    with torch.no_grad():  # 验证阶段不需要梯度，节省显存和计算
        for batch in dataloader:  # 遍历验证集
            source = batch["input"].to(device)  # 把输入图放到设备上
            target = batch["target"].to(device)  # 把目标图放到设备上
            prediction = generator(source)  # 生成预测图
            total_l1 += l1_loss(prediction, target).item()  # 累加这个 batch 的平均 L1 损失

            mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))  # 对每张图计算整体均方误差
            psnr = 10.0 * torch.log10(4.0 / mse.clamp_min(1e-8))  # 因为像素范围是 [-1, 1]，动态范围平方是 4，所以按这个公式算 PSNR
            total_psnr += psnr.mean().item()  # 累加这个 batch 的平均 PSNR
            count += 1  # batch 数加 1
    generator.train()  # 评估完后切回训练模式，方便外面的训练循环继续用
    return {  # 返回验证指标
        "val_l1": total_l1 / max(count, 1),  # 验证集平均 L1
        "val_psnr": total_psnr / max(count, 1),  # 验证集平均 PSNR
    }  # 字典结束


def train(args: argparse.Namespace) -> None:  # 执行完整训练流程
    seed_everything(args.seed)  # 先固定随机种子

    data_dir = Path(args.data_dir)  # 数据目录路径
    output_dir = Path(args.output_dir)  # 输出目录路径
    output_dir.mkdir(parents=True, exist_ok=True)  # 如果输出目录不存在，就先创建
    checkpoints_dir = output_dir / "checkpoints"  # 存模型权重的目录
    samples_dir = output_dir / "samples"  # 存每个 epoch 可视化样例图的目录
    predictions_dir = output_dir / "predictions"  # 存验证集预测结果的目录
    checkpoints_dir.mkdir(parents=True, exist_ok=True)  # 创建 checkpoints 目录
    samples_dir.mkdir(parents=True, exist_ok=True)  # 创建 samples 目录
    predictions_dir.mkdir(parents=True, exist_ok=True)  # 创建 predictions 目录

    records = discover_pairs(data_dir)  # 自动发现所有配对样本
    train_records, val_records = split_records(records, train_ratio=args.train_ratio, seed=args.seed)  # 把样本切分为训练集和验证集
    if args.max_train_samples:  # 如果用户限制了训练样本数
        train_records = train_records[: args.max_train_samples]  # 就只保留前面这些训练样本
    if args.max_val_samples:  # 如果用户限制了验证样本数
        val_records = val_records[: args.max_val_samples]  # 就只保留前面这些验证样本

    train_dataset = CMPFacadeDataset(  # 创建训练集 Dataset
        train_records,  # 训练样本记录
        image_size=args.image_size,  # 最终训练尺寸
        load_size=args.load_size,  # 先缩放到这个尺寸再随机裁剪
        augment=not args.disable_augment,  # 默认开增强，除非用户显式关闭
    )  # Dataset 初始化结束
    val_dataset = CMPFacadeDataset(val_records, image_size=args.image_size, load_size=args.image_size, augment=False)  # 验证集不做随机增强，直接缩放到固定尺寸
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)  # 创建训练集 DataLoader，开启打乱
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)  # 创建验证集 DataLoader，不打乱

    device = torch.device(args.device)  # 把字符串设备名转成 PyTorch 设备对象
    generator = UNetGenerator(base_channels=args.base_channels).to(device)  # 创建生成器并搬到设备上
    discriminator = PatchDiscriminator(base_channels=args.base_channels).to(device)  # 创建判别器并搬到设备上
    generator.apply(init_weights)  # 对生成器所有子模块应用权重初始化函数
    discriminator.apply(init_weights)  # 对判别器所有子模块应用权重初始化函数

    adv_loss = nn.BCEWithLogitsLoss()  # 对抗损失：内部自带 sigmoid，更适合判别器原始输出
    l1_loss = nn.L1Loss()  # 像素级重建损失
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))  # 生成器优化器
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))  # 判别器优化器
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(  # 给生成器设置学习率调度器
        optimizer_g,  # 绑定到生成器优化器上
        lr_lambda=lambda epoch: 1.0  # 前半程保持原学习率
        if epoch < max(args.epochs // 2, 1)  # 判断是否还在前半程
        else max(0.0, 1.0 - (epoch - max(args.epochs // 2, 1)) / max(args.epochs - max(args.epochs // 2, 1), 1)),  # 后半程线性下降到 0
    )  # 调度器定义结束
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(  # 给判别器也设置同样的调度器
        optimizer_d,  # 绑定到判别器优化器上
        lr_lambda=lambda epoch: 1.0  # 前半程保持原学习率
        if epoch < max(args.epochs // 2, 1)  # 判断是否还在前半程
        else max(0.0, 1.0 - (epoch - max(args.epochs // 2, 1)) / max(args.epochs - max(args.epochs // 2, 1), 1)),  # 后半程线性下降到 0
    )  # 调度器定义结束

    history: List[dict] = []  # 用来保存每个 epoch 的训练和验证指标
    best_val_l1 = float("inf")  # 记录到目前为止最好的验证集 L1，初始设为无穷大

    for epoch in range(1, args.epochs + 1):  # 从第 1 个 epoch 开始，一直训练到 args.epochs
        generator.train()  # 生成器切到训练模式
        discriminator.train()  # 判别器切到训练模式
        epoch_g = 0.0  # 累计当前 epoch 的生成器损失
        epoch_d = 0.0  # 累计当前 epoch 的判别器损失
        step_count = 0  # 记录当前 epoch 一共训练了多少个 batch

        for batch in train_loader:  # 遍历训练集的所有 batch
            source = batch["input"].to(device)  # 把输入图搬到设备上
            target = batch["target"].to(device)  # 把目标图搬到设备上

            fake = generator(source)  # 生成器根据输入图生成假图像

            optimizer_d.zero_grad()  # 清空判别器上一次反向传播留下的梯度
            pred_real = discriminator(source, target)  # 判别器看“输入图 + 真图”这对组合
            pred_fake = discriminator(source, fake.detach())  # 判别器看“输入图 + 假图”这对组合；detach 表示这里不让梯度回传到生成器
            real_labels = torch.ones_like(pred_real)  # 真样本标签全设为 1，形状和判别器输出一致
            fake_labels = torch.zeros_like(pred_fake)  # 假样本标签全设为 0
            loss_d = 0.5 * (adv_loss(pred_real, real_labels) + adv_loss(pred_fake, fake_labels))  # 判别器损失 = 判真损失和判假损失的平均
            loss_d.backward()  # 反向传播，计算判别器参数的梯度
            optimizer_d.step()  # 更新判别器参数

            optimizer_g.zero_grad()  # 清空生成器上一次反向传播留下的梯度
            pred_fake_for_g = discriminator(source, fake)  # 让判别器重新看一遍假图，这次不 detach，因为我们要把梯度传回生成器
            loss_g_adv = adv_loss(pred_fake_for_g, real_labels)  # 生成器的对抗目标：希望判别器把假图判成真图
            loss_g_l1 = l1_loss(fake, target)  # 生成器的重建目标：希望假图在像素上接近真图
            loss_g = loss_g_adv + args.lambda_l1 * loss_g_l1  # 总生成器损失 = 对抗损失 + lambda * L1 损失
            loss_g.backward()  # 反向传播，计算生成器参数的梯度
            optimizer_g.step()  # 更新生成器参数

            epoch_g += loss_g.item()  # 把当前 batch 的生成器损失累计起来
            epoch_d += loss_d.item()  # 把当前 batch 的判别器损失累计起来
            step_count += 1  # batch 数加 1

        metrics = evaluate_model(generator, val_loader, device, l1_loss)  # 每个 epoch 结束后，在验证集上评估一次
        epoch_summary = {  # 整理出本轮 epoch 的主要指标
            "epoch": epoch,  # 当前是第几个 epoch
            "train_g_loss": epoch_g / max(step_count, 1),  # 平均生成器训练损失
            "train_d_loss": epoch_d / max(step_count, 1),  # 平均判别器训练损失
            **metrics,  # 把验证集指标也一并展开进来
        }  # 字典结束
        history.append(epoch_summary)  # 存入历史记录
        print(json.dumps(epoch_summary, ensure_ascii=False))  # 打印这一轮结果，方便在终端里看训练进度

        sample_batch = next(iter(val_loader))  # 从验证集里拿一个 batch，当作可视化样例
        source = sample_batch["input"].to(device)  # 取样例输入图
        target = sample_batch["target"].to(device)  # 取样例目标图
        with torch.no_grad():  # 保存样例图时不需要梯度
            prediction = generator(source)  # 生成一组预测结果
        save_triptych(source[0], prediction[0], target[0], samples_dir / f"epoch_{epoch:03d}.png")  # 只取 batch 里第一张图，拼成对比图保存

        if metrics["val_l1"] < best_val_l1:  # 如果这轮验证集 L1 比之前最好结果还小
            best_val_l1 = metrics["val_l1"]  # 就更新“最好成绩”
            torch.save(  # 把当前最好的模型存盘
                {
                    "generator": generator.state_dict(),  # 生成器权重
                    "discriminator": discriminator.state_dict(),  # 判别器权重
                    "args": vars(args),  # 训练用到的参数
                    "history": history,  # 到当前为止的训练历史
                },  # 字典结束
                checkpoints_dir / "best.pt",  # 保存路径
            )  # 保存结束

        scheduler_g.step()  # epoch 结束后更新一次生成器学习率
        scheduler_d.step()  # epoch 结束后更新一次判别器学习率

    torch.save(  # 整个训练全部结束后，再保存一个 last.pt
        {
            "generator": generator.state_dict(),  # 最后一轮生成器权重
            "discriminator": discriminator.state_dict(),  # 最后一轮判别器权重
            "args": vars(args),  # 训练参数
            "history": history,  # 完整训练历史
        },  # 字典结束
        checkpoints_dir / "last.pt",  # 保存路径
    )  # 保存结束

    with torch.no_grad():  # 最后再对整个验证集做一次预测并存图
        generator.eval()  # 生成器切到评估模式
        for batch in val_loader:  # 遍历验证集
            source = batch["input"].to(device)  # 取输入图
            stems = batch["stem"]  # 取对应文件名主干
            predictions = generator(source)  # 一次生成整个 batch 的预测图
            for stem, prediction in zip(stems, predictions):  # 把每张预测图和它对应的文件名配对起来
                save_tensor_image(prediction, predictions_dir / f"{stem}.png")  # 单张保存到 predictions 目录里

    with (output_dir / "train_history.json").open("w", encoding="utf-8") as file:  # 用 UTF-8 打开一个 JSON 文件准备写训练历史
        json.dump(history, file, ensure_ascii=False, indent=2)  # 把 history 美化后写进去


def predict(args: argparse.Namespace) -> None:  # 用训练好的 checkpoint 对单张图做推理
    device = torch.device(args.device)  # 把设备字符串转成 PyTorch 设备对象
    checkpoint = torch.load(args.checkpoint, map_location=device)  # 加载模型权重文件，并映射到指定设备

    generator = UNetGenerator(base_channels=checkpoint["args"].get("base_channels", 32)).to(device)  # 根据 checkpoint 里的参数重建同样结构的生成器
    generator.load_state_dict(checkpoint["generator"])  # 把生成器权重加载进去
    generator.eval()  # 切到评估模式

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)  # 读取待预测的输入图
    if image is None:  # 如果读取失败
        raise FileNotFoundError(f"Failed to load {args.input}")  # 直接报错
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转成 RGB
    image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)  # 缩放到训练时使用的尺寸
    input_tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)  # 转成形状为 [1, C, H, W] 的输入张量，并归一化到 [-1, 1]

    with torch.no_grad():  # 推理阶段不需要梯度
        prediction = generator(input_tensor)[0]  # 生成结果，并取出 batch 里的第 1 张图
    save_tensor_image(prediction, Path(args.output))  # 把结果保存到指定路径


def build_parser() -> argparse.ArgumentParser:  # 定义这个脚本支持的命令行参数
    parser = argparse.ArgumentParser(description="Pix2Pix training on CMP facades.")  # 创建总解析器
    subparsers = parser.add_subparsers(dest="command", required=True)  # 创建子命令解析器，这个脚本支持 train 和 predict 两个子命令

    train_parser = subparsers.add_parser("train")  # 创建 train 子命令
    train_parser.add_argument("--data-dir", type=str, default="base")  # 数据目录
    train_parser.add_argument("--output-dir", type=str, default="outputs/part2")  # 输出目录
    train_parser.add_argument("--epochs", type=int, default=10)  # 训练轮数
    train_parser.add_argument("--batch-size", type=int, default=4)  # batch size
    train_parser.add_argument("--image-size", type=int, default=256)  # 最终训练图像尺寸
    train_parser.add_argument("--load-size", type=int, default=286)  # 随机裁剪前先缩放到的尺寸
    train_parser.add_argument("--lr", type=float, default=2e-4)  # 学习率
    train_parser.add_argument("--lambda-l1", type=float, default=100.0)  # L1 损失前面的权重
    train_parser.add_argument("--train-ratio", type=float, default=0.9)  # 训练集比例
    train_parser.add_argument("--seed", type=int, default=42)  # 随机种子
    train_parser.add_argument("--device", type=str, default="cpu")  # 运行设备
    train_parser.add_argument("--base-channels", type=int, default=32)  # U-Net 和判别器的基础通道数
    train_parser.add_argument("--max-train-samples", type=int, default=None)  # 可选：限制训练样本数，方便快速调试
    train_parser.add_argument("--max-val-samples", type=int, default=None)  # 可选：限制验证样本数
    train_parser.add_argument("--disable-augment", action="store_true")  # 如果写上这个参数，就关闭数据增强

    predict_parser = subparsers.add_parser("predict")  # 创建 predict 子命令
    predict_parser.add_argument("--checkpoint", type=str, required=True)  # 要加载的模型权重路径
    predict_parser.add_argument("--input", type=str, required=True)  # 输入图路径
    predict_parser.add_argument("--output", type=str, required=True)  # 输出图路径
    predict_parser.add_argument("--image-size", type=int, default=256)  # 推理时要缩放到的尺寸
    predict_parser.add_argument("--device", type=str, default="cpu")  # 推理设备

    return parser  # 返回配置好的解析器


def main() -> None:  # 程序入口
    parser = build_parser()  # 先构建命令行参数解析器
    args = parser.parse_args()  # 真正读取命令行参数
    if args.command == "train":  # 如果子命令是 train
        train(args)  # 就进入训练流程
    elif args.command == "predict":  # 如果子命令是 predict
        predict(args)  # 就进入推理流程
    else:  # 理论上不会走到这里，因为上面已经限制了子命令
        raise ValueError(f"Unsupported command: {args.command}")  # 这里留一个保险报错


if __name__ == "__main__":  # 只有当这个文件被直接运行时，才会执行下面这一行
    main()  # 调用主函数，启动脚本
