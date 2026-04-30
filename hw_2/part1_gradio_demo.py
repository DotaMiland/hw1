from __future__ import annotations  # 让类型注解的写法更灵活

from pathlib import Path  # 用来处理文件路径
from typing import List, Sequence, Tuple  # 这些是类型提示，帮助我们读懂变量大概是什么

import cv2  # OpenCV：负责画点、画线、做简单图像处理
import gradio as gr  # Gradio：快速搭一个网页交互界面
import numpy as np  # NumPy：负责数组运算

from part1_poisson import load_image, poisson_blend  # 直接复用 part1 里的读图函数和 Poisson 融合核心函数


Point = Tuple[int, int]  # 一个点就是 (x, y)


def ensure_rgb(image: np.ndarray | None) -> np.ndarray | None:  # 保证输入图片是 3 通道 RGB 图
    if image is None:  # 如果图片还没有上传
        return None  # 就直接返回 None
    if image.ndim == 2:  # 如果是灰度图，只有高和宽两个维度
        return np.repeat(image[:, :, None], 3, axis=2)  # 复制成 3 通道，变成伪 RGB
    if image.shape[2] == 4:  # 如果是 RGBA，说明多了一个透明度通道
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # 把透明度去掉，转成普通 RGB
    return image  # 如果本来就是 RGB，就原样返回


def draw_points(image: np.ndarray | None, points: Sequence[Point], closed: bool) -> np.ndarray | None:  # 在图像上画出当前已选的点和多边形边框
    image = ensure_rgb(image)  # 先确保图像是 RGB 格式
    if image is None:  # 如果没有图像
        return None  # 就没法画，直接返回

    canvas = image.copy()  # 复制一份图像，避免改动原图
    for idx, (x, y) in enumerate(points):  # 枚举所有点，同时拿到点的序号
        cv2.circle(canvas, (int(x), int(y)), 4, (255, 0, 0), -1)  # 在点的位置画一个实心小圆点
        cv2.putText(canvas, str(idx + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 在点旁边标上编号

    if len(points) >= 2:  # 至少要有两个点，连线才有意义
        poly = np.asarray(points, dtype=np.int32)  # 转成 OpenCV 需要的整数数组
        cv2.polylines(canvas, [poly], isClosed=closed, color=(255, 0, 0), thickness=2)  # 画线；closed=True 时会自动把最后一个点和第一个点连起来
    return canvas  # 返回画好标记的图像


def points_to_text(points: Sequence[Point], closed: bool) -> str:  # 把点列表转成右侧文本框里更容易读的说明文字
    if not points:  # 如果还没有点
        return "No points yet. Click on the foreground image to add polygon vertices."  # 给用户一个提示
    chunks = [f"{i + 1}. ({x}, {y})" for i, (x, y) in enumerate(points)]  # 把每个点整理成 “序号. (x, y)” 的文本
    status = "Closed polygon" if closed else "Polygon is open"  # 根据是否闭合，给出状态描述
    return status + "\n" + "\n".join(chunks)  # 状态放在第一行，后面逐行列出所有点


def get_mask_and_crop(source_image: np.ndarray, points: Sequence[Point]) -> tuple[np.ndarray, tuple[int, int, int, int]]:  # 根据多边形点算出 mask 和最小包围盒
    mask = np.zeros(source_image.shape[:2], dtype=np.uint8)  # 先创建一张和图像同样高宽的黑色 mask
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)  # 把多边形内部填成白色
    ys, xs = np.where(mask > 0)  # 找出所有白色像素的位置
    if len(xs) == 0 or len(ys) == 0:  # 如果一个白色像素都没有
        raise ValueError("Polygon is empty.")  # 说明多边形无效，直接报错
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1  # 算出包围盒，右边和下边加 1 是为了 Python 切片时包含到边界
    return mask, (left, top, right, bottom)  # 返回 mask 和包围盒坐标


def compose_preview(  # 生成“把前景区域半透明贴到背景上”的预览图
    foreground_image: np.ndarray | None,  # 前景图
    background_image: np.ndarray | None,  # 背景图
    points: Sequence[Point],  # 当前已选的多边形点
    closed: bool,  # 当前多边形是否已经闭合
    offset_x: float,  # 在背景图上的水平偏移
    offset_y: float,  # 在背景图上的垂直偏移
) -> np.ndarray | None:  # 返回预览图；如果条件不够则返回 None 或背景图本身
    background_image = ensure_rgb(background_image)  # 确保背景图是 RGB
    foreground_image = ensure_rgb(foreground_image)  # 确保前景图是 RGB
    if background_image is None:  # 如果连背景图都没有
        return None  # 就无法生成预览

    preview = background_image.copy()  # 从背景图复制一份开始画预览
    if foreground_image is None or len(points) < 3:  # 如果前景图没上传，或者点数不足 3 个
        return preview  # 那就只能返回原背景图

    try:  # 下面尝试根据点生成 mask 和裁剪框
        mask, (left, top, right, bottom) = get_mask_and_crop(foreground_image, points)  # 得到前景选区的 mask 和包围盒
    except ValueError:  # 如果多边形无效
        return preview  # 还是返回背景图，不让界面崩掉

    crop = foreground_image[top:bottom, left:right]  # 裁出前景选区所在的小块
    crop_mask = mask[top:bottom, left:right]  # 裁出与之对应的小 mask

    paste_x = int(np.clip(round(offset_x), 0, max(preview.shape[1] - crop.shape[1], 0)))  # 把水平偏移限制在合法范围内，避免越界
    paste_y = int(np.clip(round(offset_y), 0, max(preview.shape[0] - crop.shape[0], 0)))  # 把垂直偏移限制在合法范围内，避免越界
    end_x = paste_x + crop.shape[1]  # 计算贴到背景上的右边界
    end_y = paste_y + crop.shape[0]  # 计算贴到背景上的下边界

    roi = preview[paste_y:end_y, paste_x:end_x].copy()  # 取出背景图上将要被覆盖的那块区域
    alpha = (crop_mask.astype(np.float32) / 255.0)[:, :, None] * 0.55  # 把 mask 变成 0~1 的透明度，并乘 0.55 让预览更像半透明叠加
    preview[paste_y:end_y, paste_x:end_x] = (crop * alpha + roi * (1.0 - alpha)).astype(np.uint8)  # 做简单 alpha 混合，形成预览图

    shifted_points = [(x - left + paste_x, y - top + paste_y) for x, y in points]  # 把原本在前景图坐标系里的点，平移到背景图坐标系里
    if len(shifted_points) >= 2:  # 至少有两点时才画边框
        cv2.polylines(  # 在背景预览图上再把多边形边框画出来
            preview,  # 画到预览图上
            [np.asarray(shifted_points, dtype=np.int32)],  # 点坐标列表
            isClosed=closed,  # 是否闭合由当前状态决定
            color=(0, 255, 0),  # 用绿色画出来，和前景图里的蓝色区分开
            thickness=2,  # 线宽 2
        )  # 画线结束
    return preview  # 返回最终预览图


def add_point(  # 用户在前景预览图上点一下时，会触发这个函数
    foreground_image: np.ndarray | None,  # 当前前景图
    points: List[Point],  # 之前已经记录的点
    closed: bool,  # 当前多边形是否已闭合
    evt: gr.SelectData,  # Gradio 传进来的点击事件对象
):
    foreground_image = ensure_rgb(foreground_image)  # 先保证前景图是 RGB
    if foreground_image is None:  # 如果还没有图像
        return None, points, closed, points_to_text(points, closed), gr.update(), gr.update()  # 返回默认结果，不做任何修改
    if closed:  # 如果多边形已经闭合
        return (  # 提示用户不能继续加点，除非先撤销或清空
            draw_points(foreground_image, points, closed),  # 保持当前画面不变
            points,  # 点列表不变
            closed,  # 闭合状态不变
            "Polygon is already closed. Use Clear or Undo to modify it.",  # 给出提示信息
            gr.update(),  # 滑块设置不变
            gr.update(),  # 滑块设置不变
        )  # 返回结束

    index = getattr(evt, "index", None)  # 从点击事件里取出坐标信息
    if index is None or not isinstance(index, (tuple, list)) or len(index) < 2:  # 如果事件里没有合法坐标
        return (  # 那就什么都不改，直接返回当前状态
            draw_points(foreground_image, points, closed),  # 重新画一下当前状态
            points,  # 点列表不变
            closed,  # 闭合状态不变
            points_to_text(points, closed),  # 文本也保持当前内容
            gr.update(),  # 滑块不变
            gr.update(),  # 滑块不变
        )  # 返回结束

    x, y = int(index[0]), int(index[1])  # 取出点击位置的 x 和 y
    points = list(points) + [(x, y)]  # 在原来的点列表后面追加一个新点
    max_x = max(foreground_image.shape[1] - 1, 0)  # 根据前景图宽度设置水平滑块最大值
    max_y = max(foreground_image.shape[0] - 1, 0)  # 根据前景图高度设置垂直滑块最大值
    return (  # 返回更新后的前景图、点状态、文本和滑块范围
        draw_points(foreground_image, points, closed),  # 重新绘制带点的前景图
        points,  # 返回新的点列表
        closed,  # 闭合状态暂时不变
        points_to_text(points, closed),  # 更新文本框里的点列表
        gr.update(maximum=max_x),  # 更新 X 滑块的最大值
        gr.update(maximum=max_y),  # 更新 Y 滑块的最大值
    )  # 返回结束


def close_polygon(foreground_image: np.ndarray | None, points: List[Point]):  # 用户点击 “Close Polygon” 按钮时触发
    foreground_image = ensure_rgb(foreground_image)  # 先确保图像格式正确
    closed = len(points) >= 3  # 只有点数至少为 3 时，才允许闭合
    message = points_to_text(points, closed)  # 默认生成一份状态文本
    if len(points) < 3:  # 如果点数还不够
        message = "Need at least 3 points before closing the polygon."  # 给出提示
    return draw_points(foreground_image, points, closed), closed, message  # 返回更新后的前景图、闭合状态和文本


def undo_point(foreground_image: np.ndarray | None, points: List[Point], closed: bool):  # 撤销最后一个点
    foreground_image = ensure_rgb(foreground_image)  # 保证图像格式正确
    if points:  # 如果点列表非空
        points = list(points[:-1])  # 删掉最后一个点
    closed = False  # 只要撤销过点，就重新视为未闭合
    return draw_points(foreground_image, points, closed), points, closed, points_to_text(points, closed)  # 返回更新后的状态


def clear_points(foreground_image: np.ndarray | None):  # 清空所有点
    foreground_image = ensure_rgb(foreground_image)  # 保证图像格式正确
    points: List[Point] = []  # 重置为空列表
    closed = False  # 闭合状态也重置为 False
    return foreground_image, points, closed, points_to_text(points, closed)  # 返回清空后的状态


def on_foreground_upload(foreground_image: np.ndarray | None):  # 上传前景图后触发
    foreground_image = ensure_rgb(foreground_image)  # 保证图像格式正确
    if foreground_image is None:  # 如果上传失败或用户清空了图像
        return None, [], False, points_to_text([], False), gr.update(maximum=1, value=0), gr.update(maximum=1, value=0)  # 返回最初始状态
    return (  # 如果上传成功，就初始化一套和图像尺寸匹配的界面状态
        foreground_image,  # 前景预览图先显示原图
        [],  # 点列表清空
        False,  # 闭合状态设为 False
        points_to_text([], False),  # 文本框显示默认提示
        gr.update(maximum=max(foreground_image.shape[1] - 1, 1), value=0),  # X 滑块最大值设成图像宽度减 1
        gr.update(maximum=max(foreground_image.shape[0] - 1, 1), value=0),  # Y 滑块最大值设成图像高度减 1
    )  # 返回结束


def on_background_upload(background_image: np.ndarray | None):  # 上传背景图后触发
    background_image = ensure_rgb(background_image)  # 保证图像格式正确
    if background_image is None:  # 如果背景图无效
        return None  # 就返回空
    return background_image  # 否则直接把背景图显示出来


def update_background_preview(foreground_image, background_image, points, closed, offset_x, offset_y):  # 当点、闭合状态或偏移滑块变化时，刷新背景预览图
    return compose_preview(foreground_image, background_image, points, closed, offset_x, offset_y)  # 直接调用上面的预览生成函数


def run_blending(foreground_image, background_image, points, closed, offset_x, offset_y, iterations):  # 用户点击 “Blend” 按钮时，真正执行一次融合
    foreground_image = ensure_rgb(foreground_image)  # 保证前景图格式正确
    background_image = ensure_rgb(background_image)  # 保证背景图格式正确

    if foreground_image is None or background_image is None:  # 如果前景或背景缺失
        return None, None, "Please upload both foreground and background images first."  # 提示用户先上传图片
    if len(points) < 3 or not closed:  # 如果点不够，或者多边形还没闭合
        return None, None, "Please finish the polygon before blending."  # 提示用户先完成选区

    mask, (left, top, right, bottom) = get_mask_and_crop(foreground_image, points)  # 根据点生成 mask 和包围盒
    crop = foreground_image[top:bottom, left:right]  # 裁出前景图中被选中的那一块
    crop_points = [(x - left, y - top) for x, y in points]  # 把原图坐标转换成“裁剪后小块”里的局部坐标
    crop_mask = mask[top:bottom, left:right]  # 裁出与小块对应的 mask

    paste_x = int(np.clip(round(offset_x), 0, max(background_image.shape[1] - crop.shape[1], 0)))  # 计算合法的粘贴横坐标
    paste_y = int(np.clip(round(offset_y), 0, max(background_image.shape[0] - crop.shape[0], 0)))  # 计算合法的粘贴纵坐标

    source_canvas = np.zeros_like(background_image)  # 创建一张和背景图一样大的全黑图
    source_canvas[paste_y : paste_y + crop.shape[0], paste_x : paste_x + crop.shape[1]] = crop  # 把前景小块放到这张黑图的目标位置
    shifted_points = [(x + paste_x, y + paste_y) for x, y in crop_points]  # 把小块内部的点坐标再平移回整张背景图坐标系

    blended, _, naive = poisson_blend(  # 调用 part1 里的核心融合函数
        source_image=source_canvas,  # 源图就是那张“只有目标区域有内容，其余都是黑色”的画布
        target_image=background_image,  # 目标图就是背景图
        polygon_points=shifted_points,  # 多边形点是已经平移到背景坐标系里的点
        iterations=int(iterations),  # 把滑块值转成整数迭代次数
        learning_rate=1e-2,  # 这里固定学习率为 1e-2
        device="cpu",  # demo 中固定用 CPU，避免环境差异太大
    )  # 融合结束

    return naive, blended, "Blending complete."  # 返回直接粘贴图、Poisson 融合图，以及状态文本


def load_examples():  # 点击 “Use Example Images” 时，自动加载仓库里的示例图片
    fg = load_image(Path("base/a1.png"))  # 加载示例前景图 a1
    bg = load_image(Path("base/a2.png"))  # 加载示例背景图 a2
    preview = fg.copy()  # 前景预览图一开始就显示前景图本身
    return (  # 返回一整套能把界面恢复到示例状态的值
        fg,  # 前景上传组件的值
        bg,  # 背景上传组件的值
        preview,  # 前景预览组件的值
        bg,  # 背景预览组件的值
        [],  # 初始点列表为空
        False,  # 初始闭合状态为 False
        points_to_text([], False),  # 文本框显示默认提示
        gr.update(maximum=max(fg.shape[1] - 1, 1), value=0),  # X 滑块范围按前景图宽度设置
        gr.update(maximum=max(fg.shape[0] - 1, 1), value=0),  # Y 滑块范围按前景图高度设置
    )  # 返回结束


def build_demo() -> gr.Blocks:  # 搭建整个 Gradio 网页界面
    with gr.Blocks(title="Poisson Image Editing Demo") as demo:  # 创建一个 Gradio Blocks 应用
        gr.Markdown(  # 在页面顶部放一段说明文字
            """
            # Poisson Image Editing Demo
            1. Upload a foreground and a background image.
            2. Click on the foreground image to add polygon vertices.
            3. Use `Close Polygon` after selecting the region.
            4. Move the selected region on the background with the sliders.
            5. Click `Blend` to compare naive copy-paste and Poisson blending.
            """
        )  # Markdown 文本结束

        points_state = gr.State([])  # 用来在界面内部存储点列表
        closed_state = gr.State(False)  # 用来在界面内部存储“是否闭合”的状态

        with gr.Row():  # 第一行：两个上传框
            foreground_input = gr.Image(label="Foreground Image", type="numpy", image_mode="RGB", interactive=True)  # 上传前景图
            background_input = gr.Image(label="Background Image", type="numpy", image_mode="RGB", interactive=True)  # 上传背景图

        with gr.Row():  # 第二行：两个预览框
            foreground_preview = gr.Image(label="Foreground With Polygon", type="numpy", image_mode="RGB", interactive=True)  # 显示带点和边框的前景图
            background_preview = gr.Image(label="Background Preview", type="numpy", image_mode="RGB", interactive=False)  # 显示叠加预览的背景图

        with gr.Row():  # 第三行：三个滑块
            offset_x = gr.Slider(label="Move X", minimum=0, maximum=1, step=1, value=0)  # 控制前景区域在背景上的水平位置
            offset_y = gr.Slider(label="Move Y", minimum=0, maximum=1, step=1, value=0)  # 控制前景区域在背景上的垂直位置
            iterations = gr.Slider(label="Optimization Iterations", minimum=100, maximum=3000, step=100, value=1200)  # 控制 Poisson 优化迭代次数

        with gr.Row():  # 第四行：操作按钮
            load_button = gr.Button("Use Example Images")  # 一键加载示例图
            close_button = gr.Button("Close Polygon")  # 闭合多边形
            undo_button = gr.Button("Undo Last Point")  # 撤销最后一个点
            clear_button = gr.Button("Clear Points")  # 清空所有点
            blend_button = gr.Button("Blend")  # 执行融合

        status_box = gr.Textbox(label="Status / Polygon Points", lines=10, interactive=False)  # 用来显示状态信息和点坐标

        with gr.Row():  # 第五行：输出结果
            naive_output = gr.Image(label="Naive Copy-Paste", type="numpy", image_mode="RGB", interactive=False)  # 显示直接复制粘贴结果
            blend_output = gr.Image(label="Poisson Blending Result", type="numpy", image_mode="RGB", interactive=False)  # 显示 Poisson 融合结果

        foreground_input.upload(  # 当前景图上传完成时
            on_foreground_upload,  # 调用这个回调函数
            inputs=[foreground_input],  # 输入来自前景上传组件
            outputs=[foreground_preview, points_state, closed_state, status_box, offset_x, offset_y],  # 更新这些组件
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上也不显示 API 信息
        )  # 事件绑定结束
        background_input.upload(  # 当背景图上传完成时
            on_background_upload,  # 调用这个回调函数
            inputs=[background_input],  # 输入来自背景上传组件
            outputs=[background_preview],  # 只更新背景预览组件
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 事件绑定结束

        load_button.click(  # 点击 “Use Example Images” 按钮时
            load_examples,  # 调用示例加载函数
            outputs=[  # 它会一次性更新很多组件
                foreground_input,  # 前景图上传框
                background_input,  # 背景图上传框
                foreground_preview,  # 前景预览
                background_preview,  # 背景预览
                points_state,  # 点状态
                closed_state,  # 闭合状态
                status_box,  # 文本框
                offset_x,  # X 滑块
                offset_y,  # Y 滑块
            ],  # 输出列表结束
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 事件绑定结束

        foreground_preview.select(  # 当用户在前景预览图上点击时
            add_point,  # 调用加点函数
            inputs=[foreground_input, points_state, closed_state],  # 输入包括当前前景图、点状态、闭合状态
            outputs=[foreground_preview, points_state, closed_state, status_box, offset_x, offset_y],  # 更新这些组件
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 事件绑定结束

        close_button.click(  # 点击闭合按钮时
            close_polygon,  # 先闭合多边形
            inputs=[foreground_input, points_state],  # 输入是前景图和点列表
            outputs=[foreground_preview, closed_state, status_box],  # 更新前景图、闭合状态和文本
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        ).then(  # 上一步做完后，再继续做下一步
            update_background_preview,  # 重新生成背景预览图
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],  # 输入包括前景、背景、点、闭合状态和偏移
            outputs=[background_preview],  # 只更新背景预览图
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 链式事件绑定结束

        undo_button.click(  # 点击撤销按钮时
            undo_point,  # 先撤销最后一个点
            inputs=[foreground_input, points_state, closed_state],  # 输入前景图、点状态和闭合状态
            outputs=[foreground_preview, points_state, closed_state, status_box],  # 更新前景图和状态
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        ).then(  # 再刷新背景预览
            update_background_preview,  # 调用预览更新函数
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],  # 输入相关状态
            outputs=[background_preview],  # 更新背景预览图
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 链式事件绑定结束

        clear_button.click(  # 点击清空按钮时
            clear_points,  # 清空点
            inputs=[foreground_input],  # 输入只需要前景图
            outputs=[foreground_preview, points_state, closed_state, status_box],  # 更新前景图和状态
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        ).then(  # 再刷新背景预览
            update_background_preview,  # 调用预览更新函数
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],  # 输入相关状态
            outputs=[background_preview],  # 更新背景预览图
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 链式事件绑定结束

        offset_x.change(  # 当 X 偏移滑块变化时
            update_background_preview,  # 重新生成背景预览图
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],  # 输入相关状态
            outputs=[background_preview],  # 更新背景预览图
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 事件绑定结束
        offset_y.change(  # 当 Y 偏移滑块变化时
            update_background_preview,  # 重新生成背景预览图
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],  # 输入相关状态
            outputs=[background_preview],  # 更新背景预览图
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 事件绑定结束
        background_input.change(  # 当背景图变化时
            update_background_preview,  # 也重新生成背景预览图
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y],  # 输入相关状态
            outputs=[background_preview],  # 更新背景预览图
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 事件绑定结束

        blend_button.click(  # 点击 Blend 按钮时
            run_blending,  # 执行真正的融合
            inputs=[foreground_input, background_input, points_state, closed_state, offset_x, offset_y, iterations],  # 输入前景、背景、点、偏移和迭代次数
            outputs=[naive_output, blend_output, status_box],  # 输出直接粘贴结果、Poisson 结果和状态文本
            api_name=False,  # 不暴露成 API
            show_api=False,  # 页面上不显示 API 信息
        )  # 事件绑定结束

    return demo  # 返回这个已经搭好的 Gradio 应用


if __name__ == "__main__":  # 只有直接运行这个文件时，才启动网页服务
    build_demo().launch(server_name="0.0.0.0", server_port=7860, show_api=False, share=False)  # 构建并启动 demo，监听 7860 端口
