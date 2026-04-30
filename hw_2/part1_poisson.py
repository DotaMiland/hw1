from __future__ import annotations  # 让类型注解可以更灵活地引用后面才会出现的类型

import argparse  # 用来读取命令行参数，比如 --source、--target
from pathlib import Path  # 更方便地处理文件路径
from typing import List, Sequence, Tuple  # 这些只是类型提示，帮助我们读懂代码

import cv2  # OpenCV：负责读图、画图、存图
import numpy as np  # NumPy：负责数组处理
import torch  # PyTorch：负责张量、梯度和优化
import torch.nn.functional as F  # 这里主要用 F.conv2d 来做卷积


Point = Tuple[int, int]  # 一个点用 (x, y) 表示，两个值都是整数


DEFAULT_POINTS_A1_TO_A2: List[Point] = [  # 这是把 a1 的某块区域贴到 a2 时默认使用的多边形顶点
    (69, 101),  # 第 1 个顶点
    (103, 78),  # 第 2 个顶点
    (167, 77),  # 第 3 个顶点
    (201, 99),  # 第 4 个顶点
    (194, 149),  # 第 5 个顶点
    (179, 214),  # 第 6 个顶点
    (132, 262),  # 第 7 个顶点
    (93, 226),  # 第 8 个顶点
    (72, 162),  # 第 9 个顶点
]  # 列表结束

DEFAULT_POINTS_A2_TO_A1: List[Point] = [  # 这是把 a2 的某块区域贴到 a1 时默认使用的多边形顶点
    (57, 95),  # 第 1 个顶点
    (92, 74),  # 第 2 个顶点
    (160, 77),  # 第 3 个顶点
    (194, 104),  # 第 4 个顶点
    (190, 154),  # 第 5 个顶点
    (171, 217),  # 第 6 个顶点
    (124, 264),  # 第 7 个顶点
    (84, 229),  # 第 8 个顶点
    (61, 167),  # 第 9 个顶点
]  # 列表结束


def parse_points(raw_points: str | None, fallback: Sequence[Point]) -> List[Point]:  # 把命令行里的点字符串解析成 Python 列表
    if not raw_points:  # 如果用户没有传 --points
        return list(fallback)  # 就复制一份默认点返回，避免直接改原来的默认列表
    points: List[Point] = []  # 准备一个空列表，后面把解析好的点放进去
    for chunk in raw_points.split(";"):  # 先按分号切开，比如 "1,2;3,4" 会变成 ["1,2", "3,4"]
        x_str, y_str = chunk.split(",")  # 再把每一小段按逗号拆成 x 和 y 两部分
        points.append((int(x_str), int(y_str)))  # 把字符串转成整数，再作为一个点加进列表
    if len(points) < 3:  # 少于 3 个点没法围成一个多边形
        raise ValueError("Polygon needs at least three points.")  # 直接报错，提醒用户输入不合法
    return points  # 返回解析好的点列表


def load_image(image_path: Path) -> np.ndarray:  # 从磁盘读取图片，并返回 RGB 格式的 NumPy 数组
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # OpenCV 读取彩色图，默认读出来是 BGR 顺序
    if image is None:  # 如果读取失败，cv2.imread 会返回 None
        raise FileNotFoundError(f"Failed to load image: {image_path}")  # 主动抛出更清楚的错误信息
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 把 BGR 转成更常见的 RGB 再返回


def save_image(image_path: Path, image: np.ndarray) -> None:  # 把 RGB 图片保存到指定路径
    image_path.parent.mkdir(parents=True, exist_ok=True)  # 如果输出目录不存在，就顺手创建出来
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV 保存图片时更习惯 BGR，所以先转回去
    cv2.imwrite(str(image_path), bgr)  # 真正把图片写到磁盘


def create_mask_from_points(points: Sequence[Point], img_h: int, img_w: int) -> np.ndarray:  # 根据多边形顶点生成二值 mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)  # 先创建一张全黑图，大小和原图一致
    polygon = np.asarray(points, dtype=np.int32)  # 把点列表转成 OpenCV 需要的 int32 数组
    cv2.fillPoly(mask, [polygon], 255)  # 把多边形内部填成白色（255）
    return mask  # 返回这张黑白 mask


def overlay_polygon(image: np.ndarray, points: Sequence[Point], color: Tuple[int, int, int]) -> np.ndarray:  # 在图像上画出多边形轮廓，方便可视化
    canvas = image.copy()  # 复制一份图像，避免直接修改原图
    polygon = np.asarray(points, dtype=np.int32)  # 把点列表转成 OpenCV 画线需要的格式
    cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=2)  # 在图上画出闭合多边形边框
    return canvas  # 返回画好线的图像


def crop_to_mask(image: np.ndarray, mask: np.ndarray, margin: int = 12) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:  # 只裁出 mask 周围的小区域，减少后续优化的计算量
    ys, xs = np.where(mask > 0)  # 找出 mask 中所有白色像素的位置
    if len(xs) == 0 or len(ys) == 0:  # 如果一个白点都没有，说明 mask 是空的
        raise ValueError("Mask is empty.")  # 直接报错，因为后面没法继续做裁剪
    left = max(int(xs.min()) - margin, 0)  # 左边界：最小 x 再往左留一点余量，但不能小于 0
    right = min(int(xs.max()) + margin + 1, image.shape[1])  # 右边界：最大 x 再往右留一点余量，但不能超过图像宽度
    top = max(int(ys.min()) - margin, 0)  # 上边界：最小 y 再往上留一点余量，但不能小于 0
    bottom = min(int(ys.max()) + margin + 1, image.shape[0])  # 下边界：最大 y 再往下留一点余量，但不能超过图像高度
    return image[top:bottom, left:right], mask[top:bottom, left:right], (left, top, right, bottom)  # 返回裁好的图、裁好的 mask、以及裁剪框坐标


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:  # 把 HWC 的 NumPy 图像转成 PyTorch 需要的 BCHW 张量
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)  # 先转成 float32，并把像素从 0~255 归一化到 0~1
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW，再补一个 batch 维度，变成 1CHW
    return tensor.to(device)  # 把张量放到指定设备上，比如 CPU 或 GPU


def mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:  # 把二维 mask 转成形状为 11HW 的张量
    tensor = torch.from_numpy((mask.astype(np.float32) / 255.0)).unsqueeze(0).unsqueeze(0)  # 先归一化到 0~1，再补通道维和 batch 维
    return tensor.to(device)  # 把 mask 张量放到指定设备上


def cal_laplacian_loss(  # 计算 Poisson blending 里最关键的损失函数
    foreground_img: torch.Tensor,  # 前景图像块，形状大致是 [1, 3, H, W]
    foreground_mask: torch.Tensor,  # 前景 mask，形状大致是 [1, 1, H, W]
    blended_img: torch.Tensor,  # 当前正在优化的融合结果图像块
    background_mask: torch.Tensor,  # 背景侧参与计算的 mask，这里实际也传的是同一个 mask
) -> torch.Tensor:  # 返回一个标量损失
    kernel = torch.tensor(  # 这里手写了一个 3x3 的拉普拉斯卷积核
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],  # 这个核会强调图像中的边缘和变化
        dtype=foreground_img.dtype,  # 卷积核的数据类型和输入图像保持一致
        device=foreground_img.device,  # 卷积核放在和输入图像相同的设备上，避免 CPU/GPU 混用报错
    ).view(1, 1, 3, 3)  # 把它整理成卷积层需要的 4 维形状
    kernel = kernel.repeat(foreground_img.shape[1], 1, 1, 1)  # 按通道复制，比如 RGB 三通道就复制 3 份

    fg_laplacian = F.conv2d(foreground_img, kernel, padding=1, groups=foreground_img.shape[1])  # 对前景图每个通道分别做拉普拉斯卷积
    blended_laplacian = F.conv2d(blended_img, kernel, padding=1, groups=foreground_img.shape[1])  # 对当前融合图也做同样的卷积

    valid_mask = foreground_mask * background_mask  # 有效区域就是两个 mask 的交集，这里基本就是多边形内部
    denominator = valid_mask.sum().clamp_min(1.0) * foreground_img.shape[1]  # 统计有效像素数量，防止后面除以 0
    laplacian_term = ((fg_laplacian - blended_laplacian) ** 2) * valid_mask  # 让融合结果的梯度结构尽量接近前景
    color_term = ((foreground_img - blended_img) ** 2) * valid_mask  # 再加一点颜色约束，避免颜色漂得太离谱
    return laplacian_term.sum() / denominator + 0.05 * color_term.sum() / denominator  # 总损失 = 结构项 + 0.05 倍颜色项


def poisson_blend(  # 执行一次完整的 Poisson 融合
    source_image: np.ndarray,  # 前景图
    target_image: np.ndarray,  # 背景图
    polygon_points: Sequence[Point],  # 要融合的多边形区域
    iterations: int = 1200,  # 优化迭代次数，越大通常越慢但可能更自然
    learning_rate: float = 1e-2,  # Adam 优化器的学习率
    device: str = "cpu",  # 默认在 CPU 上运行
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # 返回融合结果图、完整 mask、以及直接复制粘贴的对照图
    if source_image.shape != target_image.shape:  # 如果前景图和背景图大小不一致
        target_image = cv2.resize(  # 就先把背景图缩放到和前景图一样大
            target_image,  # 要被缩放的是目标图
            (source_image.shape[1], source_image.shape[0]),  # OpenCV 的尺寸顺序是 (宽, 高)
            interpolation=cv2.INTER_AREA,  # 这里选择更适合缩小图像的插值方式
        )  # resize 调用结束

    full_mask = create_mask_from_points(polygon_points, source_image.shape[0], source_image.shape[1])  # 根据多边形生成整张图大小的 mask
    source_crop, mask_crop, (left, top, right, bottom) = crop_to_mask(source_image, full_mask)  # 只裁出要处理的小区域，减少计算量
    target_crop = target_image[top:bottom, left:right]  # 在背景图上裁出同样位置的小块

    device_obj = torch.device(device)  # 把 "cpu" 或 "cuda" 这样的字符串转成 PyTorch 设备对象
    fg_tensor = image_to_tensor(source_crop, device_obj)  # 前景小块转成张量
    bg_tensor = image_to_tensor(target_crop, device_obj)  # 背景小块转成张量
    mask_tensor = mask_to_tensor(mask_crop, device_obj)  # mask 小块转成张量

    blended = bg_tensor.clone().detach().requires_grad_(True)  # 把当前要优化的图像初始化为背景图，并告诉 PyTorch 需要对它求梯度
    optimizer = torch.optim.Adam([blended], lr=learning_rate)  # 注意这里优化的不是模型参数，而是 blended 这张图本身

    for step in range(iterations):  # 开始迭代优化，总共跑 iterations 次
        blended_for_loss = blended * mask_tensor + bg_tensor.detach() * (1.0 - mask_tensor)  # mask 内用当前结果，mask 外强制保持背景不变
        loss = cal_laplacian_loss(fg_tensor, mask_tensor, blended_for_loss, mask_tensor)  # 计算当前融合结果和前景之间的损失

        optimizer.zero_grad()  # 先把上一次迭代残留的梯度清空
        loss.backward()  # 自动求出 loss 对 blended 的梯度
        optimizer.step()  # 用 Adam 根据梯度更新 blended

        with torch.no_grad():  # 下面这步只是数值裁剪，不需要记录进计算图
            blended.clamp_(0.0, 1.0)  # 把像素值强行限制在 0~1，避免出现非法范围

        if step == iterations // 2:  # 当训练跑到一半时
            optimizer.param_groups[0]["lr"] *= 0.2  # 把学习率缩小到原来的 0.2 倍，让后半程更稳定

    result_crop = (  # 下面把优化好的结果从 PyTorch 张量转回 NumPy 图像
        blended.detach()  # 先和计算图断开，后面就只是导出结果了
        .clamp(0.0, 1.0)  # 再保险地限制一次像素范围
        .cpu()  # 如果在 GPU 上，就先搬回 CPU
        .squeeze(0)  # 去掉 batch 维度，形状从 [1, C, H, W] 变成 [C, H, W]
        .permute(1, 2, 0)  # 再从 CHW 变回 HWC，方便转成图片
        .numpy()  # 最后转成 NumPy 数组
    )  # 转换链结束
    result_crop = (result_crop * 255.0).astype(np.uint8)  # 把 0~1 的浮点像素恢复成 0~255 的整数像素

    naive = target_image.copy()  # 先复制一份背景图，作为最朴素的“直接粘贴”结果
    naive[top:bottom, left:right][mask_crop > 0] = source_crop[mask_crop > 0]  # 在 mask 内直接把前景像素复制到背景上

    result = target_image.copy()  # 再复制一份背景图，准备写入优化后的融合结果
    result[top:bottom, left:right][mask_crop > 0] = result_crop[mask_crop > 0]  # 只在 mask 内写入优化后的结果像素
    return result, full_mask, naive  # 返回最终融合图、完整 mask、以及直接粘贴对照图


def run_case(  # 跑一组完整实验，并把中间结果和最终结果都保存下来
    source_path: Path,  # 前景图路径
    target_path: Path,  # 背景图路径
    points: Sequence[Point],  # 多边形顶点
    output_dir: Path,  # 输出目录
    prefix: str,  # 输出文件名前缀
    iterations: int,  # 优化次数
    learning_rate: float,  # 学习率
    device: str,  # 设备名
) -> None:  # 这个函数没有返回值，只负责保存文件
    source = load_image(source_path)  # 读取前景图
    target = load_image(target_path)  # 读取背景图
    blended, mask, naive = poisson_blend(  # 调用核心函数做 Poisson 融合
        source_image=source,  # 传入前景图
        target_image=target,  # 传入背景图
        polygon_points=points,  # 传入多边形顶点
        iterations=iterations,  # 传入迭代次数
        learning_rate=learning_rate,  # 传入学习率
        device=device,  # 传入设备
    )  # 融合结束后得到三样结果

    save_image(output_dir / f"{prefix}_source.png", source)  # 保存原始前景图
    save_image(output_dir / f"{prefix}_target.png", target)  # 保存原始背景图
    save_image(output_dir / f"{prefix}_source_polygon.png", overlay_polygon(source, points, (255, 0, 0)))  # 保存“前景图 + 多边形边框”的可视化结果
    save_image(output_dir / f"{prefix}_target_polygon.png", overlay_polygon(target, points, (0, 255, 0)))  # 保存“背景图 + 多边形边框”的可视化结果
    save_image(output_dir / f"{prefix}_mask.png", np.repeat(mask[:, :, None], 3, axis=2))  # 把单通道 mask 复制成三通道，方便当普通图片保存
    save_image(output_dir / f"{prefix}_naive_clone.png", naive)  # 保存直接复制粘贴的结果
    save_image(output_dir / f"{prefix}_poisson.png", blended)  # 保存 Poisson 融合后的结果


def build_parser() -> argparse.ArgumentParser:  # 定义这个脚本支持哪些命令行参数
    parser = argparse.ArgumentParser(description="Poisson image editing with PyTorch.")  # 创建参数解析器，并写一段简短说明
    parser.add_argument("--source", type=Path, default=Path("base/a1.png"))  # 前景图路径，默认是 base/a1.png
    parser.add_argument("--target", type=Path, default=Path("base/a2.png"))  # 背景图路径，默认是 base/a2.png
    parser.add_argument("--points", type=str, default=None, help="Format: x1,y1;x2,y2;...")  # 手动输入多边形点，格式像 1,2;3,4;5,6
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part1"))  # 输出目录，默认写到 outputs/part1
    parser.add_argument("--prefix", type=str, default="a1_to_a2")  # 输出文件名前缀
    parser.add_argument("--iterations", type=int, default=1200)  # 迭代次数
    parser.add_argument("--lr", type=float, default=1e-2)  # 学习率
    parser.add_argument("--device", type=str, default="cpu")  # 运行设备，默认 CPU
    parser.add_argument(  # 下面这个参数是一个布尔开关
        "--run-both-examples",  # 参数名
        action="store_true",  # 只要命令行里写了它，值就会变成 True
        help="Run a1->a2 and a2->a1 with built-in polygon points.",  # 给用户看的帮助说明
    )  # 参数定义结束
    return parser  # 返回配置好的参数解析器


def main() -> None:  # 程序入口函数，脚本运行时会从这里开始
    parser = build_parser()  # 先创建参数解析器
    args = parser.parse_args()  # 真正读取命令行参数，并把结果放进 args

    if args.run_both_examples:  # 如果用户指定了 --run-both-examples
        run_case(  # 先跑 a1 -> a2 这一组示例
            source_path=Path("base/a1.png"),  # 前景图是 a1
            target_path=Path("base/a2.png"),  # 背景图是 a2
            points=DEFAULT_POINTS_A1_TO_A2,  # 用预设好的点
            output_dir=args.output_dir,  # 输出目录沿用命令行参数
            prefix="a1_to_a2",  # 结果文件名前缀
            iterations=args.iterations,  # 迭代次数沿用命令行参数
            learning_rate=args.lr,  # 学习率沿用命令行参数
            device=args.device,  # 设备沿用命令行参数
        )  # 第一组示例结束
        run_case(  # 再跑 a2 -> a1 这一组示例
            source_path=Path("base/a2.png"),  # 前景图换成 a2
            target_path=Path("base/a1.png"),  # 背景图换成 a1
            points=DEFAULT_POINTS_A2_TO_A1,  # 用另一套预设点
            output_dir=args.output_dir,  # 输出目录不变
            prefix="a2_to_a1",  # 结果文件名前缀
            iterations=args.iterations,  # 迭代次数不变
            learning_rate=args.lr,  # 学习率不变
            device=args.device,  # 设备不变
        )  # 第二组示例结束
        return  # 两组都跑完就直接结束程序

    points = parse_points(args.points, DEFAULT_POINTS_A1_TO_A2)  # 如果不是跑双示例，就先解析用户输入的点；没传的话就用默认点
    run_case(  # 跑单组实验
        source_path=args.source,  # 前景图路径来自命令行
        target_path=args.target,  # 背景图路径来自命令行
        points=points,  # 多边形点用上面解析好的结果
        output_dir=args.output_dir,  # 输出目录来自命令行
        prefix=args.prefix,  # 文件名前缀来自命令行
        iterations=args.iterations,  # 迭代次数来自命令行
        learning_rate=args.lr,  # 学习率来自命令行
        device=args.device,  # 设备来自命令行
    )  # 单组实验结束


if __name__ == "__main__":  # 只有当这个文件被直接运行时，才会进入下面这一行
    main()  # 调用主函数，启动整个脚本
