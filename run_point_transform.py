import cv2
import numpy as np
import gradio as gr

points_src = []
points_dst = []
image = None


def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img


def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 3, (255, 0, 0), -1)
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 3, (0, 0, 255), -1)
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)
    return marked_image


def bilinear_sample(image, map_x, map_y):
    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    RBF-based image warping using inverse mapping.
    source_pts: points on original image
    target_pts: desired locations in warped image
    """
    if image is None:
        return None

    img = np.array(image)
    if len(source_pts) == 0 or len(target_pts) == 0:
        return img
    if len(source_pts) != len(target_pts):
        return img

    source_pts = source_pts.astype(np.float64)
    target_pts = target_pts.astype(np.float64)
    n = source_pts.shape[0]

    # We solve inverse mapping: at destination control points, map back to source points.
    displacement = source_pts - target_pts

    diff = target_pts[:, None, :] - target_pts[None, :, :]
    r = np.linalg.norm(diff, axis=2)

    # Gaussian RBF for numerical stability.
    sigma = max(np.mean(r[r > 0]) if np.any(r > 0) else 20.0, 1.0)
    K = np.exp(-(r ** 2) / (2.0 * (alpha * sigma) ** 2 + eps))
    K += eps * np.eye(n)

    wx = np.linalg.solve(K, displacement[:, 0])
    wy = np.linalg.solve(K, displacement[:, 1])

    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

    rg = np.linalg.norm(grid[:, None, :] - target_pts[None, :, :], axis=2)
    Phi = np.exp(-(rg ** 2) / (2.0 * (alpha * sigma) ** 2 + eps))

    disp_x = (Phi @ wx).reshape(h, w)
    disp_y = (Phi @ wy).reshape(h, w)

    map_x = grid_x + disp_x
    map_y = grid_y + disp_y

    warped_image = bilinear_sample(img, map_x, map_y)
    return warped_image


def run_warping():
    global points_src, points_dst, image
    return point_guided_deformation(image, np.array(points_src), np.array(points_dst))


def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)
        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)


if __name__ == "__main__":
    demo.launch()
