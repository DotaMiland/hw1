import gradio as gr
import cv2
import numpy as np


def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    image = np.array(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        dtype=np.uint8,
    ) + np.array((255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)
    image_new[pad_size:pad_size + image.shape[0], pad_size:pad_size + image.shape[1]] = image
    image = np.array(image_new)

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    T_to_origin = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0],
    ])
    T_back = np.array([
        [1.0, 0.0, cx],
        [0.0, 1.0, cy],
        [0.0, 0.0, 1.0],
    ])

    S = np.array([
        [scale, 0.0, 0.0],
        [0.0, scale, 0.0],
        [0.0, 0.0, 1.0],
    ])

    theta = np.deg2rad(rotation)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    if flip_horizontal:
        F = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    else:
        F = np.eye(3)

    T_shift = np.array([
        [1.0, 0.0, float(translation_x)],
        [0.0, 1.0, float(translation_y)],
        [0.0, 0.0, 1.0],
    ])

    M = T_shift @ T_back @ R @ S @ F @ T_to_origin
    transformed_image = cv2.warpAffine(
        image,
        M[:2, :],
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return transformed_image


def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            image_output = gr.Image(label="Transformed Image")

        inputs = [image_input, scale, rotation, translation_x, translation_y, flip_horizontal]
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)
    return demo


if __name__ == "__main__":
    interactive_transform().launch()
