import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import traceback

# ========================
# Part 1: Utility Functions
# ========================

def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """
    Convert Euler angles to rotation matrix (same interface as pytorch3d).
    euler_angles: (*, 3) tensor in radians
    convention: "XYZ" means R = Rz(gamma) @ Ry(beta) @ Rx(alpha)
    """
    alpha, beta, gamma = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
    batch_shape = euler_angles.shape[:-1]
    device = euler_angles.device
    dtype = euler_angles.dtype

    zeros = torch.zeros(batch_shape, device=device, dtype=dtype)
    ones = torch.ones(batch_shape, device=device, dtype=dtype)

    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
    cos_b, sin_b = torch.cos(beta), torch.sin(beta)
    cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)

    Rx = torch.stack([
        ones, zeros, zeros,
        zeros, cos_a, -sin_a,
        zeros, sin_a, cos_a
    ], dim=-1).reshape(*batch_shape, 3, 3)

    Ry = torch.stack([
        cos_b, zeros, sin_b,
        zeros, ones, zeros,
        -sin_b, zeros, cos_b
    ], dim=-1).reshape(*batch_shape, 3, 3)

    Rz = torch.stack([
        cos_g, -sin_g, zeros,
        sin_g, cos_g, zeros,
        zeros, zeros, ones
    ], dim=-1).reshape(*batch_shape, 3, 3)

    R = Rz @ Ry @ Rx
    return R


def project(points_3d, R, T, f, cx, cy):
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) 3D points in world coordinates
        R: (V, 3, 3) rotation matrices (world -> camera)
        T: (V, 3) translation vectors
        f: scalar, focal length
        cx, cy: scalar, principal point

    Returns:
        projected_2d: (V, N, 2) 2D pixel coordinates
    """
    X_cam = torch.einsum('vij,nj->vni', R, points_3d) + T.unsqueeze(1)
    Xc, Yc, Zc = X_cam[..., 0], X_cam[..., 1], X_cam[..., 2]
    Zc_safe = Zc + 1e-8 * torch.sign(Zc)
    u = -f * Xc / Zc_safe + cx
    v = f * Yc / Zc_safe + cy
    return torch.stack([u, v], dim=-1)


# ========================
# Part 2: Data Loading
# ========================

def generate_synthetic_data(num_views=50, num_points=20000, image_size=1024, device='cpu'):
    """
    Generate synthetic data for Bundle Adjustment.
    Cameras are placed in front of the object (+Z direction, ±70° range).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    cx = cy = image_size / 2.0
    f_gt_target = image_size / (2 * np.tan(np.deg2rad(45) / 2))

    points_3d_gt = torch.randn(num_points, 3, device=device) * 0.4
    colors = torch.rand(num_points, 3, device=device)

    R_gt_list, T_gt_list = [], []
    for i in range(num_views):
        yaw = (torch.rand(1).item() * 2 - 1) * 70.0 * np.pi / 180.0
        pitch = (torch.rand(1).item() * 2 - 1) * 70.0 * np.pi / 180.0
        roll = (torch.rand(1).item() * 2 - 1) * 5.0 * np.pi / 180.0
        euler = torch.tensor([roll, pitch, yaw], device=device)
        R = euler_angles_to_matrix(euler.unsqueeze(0)).squeeze(0)
        R_gt_list.append(R)

        d = 2.5 + torch.rand(1).item() * 0.5
        T = torch.tensor([0.0, 0.0, -d], device=device)
        T_gt_list.append(T)

    R_gt = torch.stack(R_gt_list)
    T_gt = torch.stack(T_gt_list)

    projections_2d = project(points_3d_gt, R_gt, T_gt, f_gt_target, cx, cy)

    noise = torch.randn_like(projections_2d) * 0.2
    projections_2d = projections_2d + noise

    mask = torch.ones(num_views, num_points, device=device)

    return {
        'points_3d_gt': points_3d_gt,
        'R_gt': R_gt,
        'T_gt': T_gt,
        'f_gt': f_gt_target,
        'cx': cx,
        'cy': cy,
        'projections_2d': projections_2d,
        'mask': mask,
        'colors': colors,
        'image_size': image_size,
        'data_type': 'synthetic',
    }


def load_real_data(data_dir, device='cpu'):
    """
    Load real data from the assignment dataset.

    Args:
        data_dir: path to directory containing points2d.npz and points3d_colors.npy
        device: torch device

    Returns:
        data dict with keys: cx, cy, projections_2d, mask, colors, image_size
    """
    npz_path = os.path.join(data_dir, 'points2d.npz')
    colors_path = os.path.join(data_dir, 'points3d_colors.npy')

    npz = np.load(npz_path)
    keys = sorted(npz.keys(), key=lambda k: int(k.split('_')[-1]))
    num_views = len(keys)

    projections_list = []
    mask_list = []
    for key in keys:
        arr = npz[key]
        uv = arr[:, :2]
        flag = arr[:, 2]
        projections_list.append(uv)
        mask_list.append(flag)

    projections_np = np.stack(projections_list, axis=0)
    mask_np = np.stack(mask_list, axis=0)

    projections_2d = torch.tensor(projections_np, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    colors_np = np.load(colors_path)
    colors = torch.tensor(colors_np.astype(np.float32), device=device)

    num_points = projections_2d.shape[1]
    image_size = 1024
    cx = cy = image_size / 2.0

    print(f"  Loaded {num_views} views, {num_points} points")
    print(f"  Mask: {mask.sum().item():.0f} visible / {mask.numel():.0f} total "
          f"({100 * mask.sum().item() / mask.numel():.1f}%)")

    return {
        'cx': cx,
        'cy': cy,
        'projections_2d': projections_2d,
        'mask': mask,
        'colors': colors,
        'image_size': image_size,
        'data_type': 'real',
    }


# ========================
# Part 3: Bundle Adjustment Optimization
# ========================

class BundleAdjustment:
    """
    Bundle Adjustment: jointly optimize camera parameters (R, T, f) and 3D points
    from 2D observations.
    """
    def __init__(self, data, device='cpu'):
        self.device = device

        self.cx = data['cx']
        self.cy = data['cy']
        self.image_size = data['image_size']

        self.observed_2d = data['projections_2d'].to(device)
        self.mask = data['mask'].to(device)
        self.colors = data['colors'].to(device)
        self.data_type = data.get('data_type', 'synthetic')

        self.V = self.observed_2d.shape[0]
        self.N = self.observed_2d.shape[1]

        if 'f_gt' in data:
            self.f_gt = data['f_gt']
        else:
            self.f_gt = None

        self._init_params(data)

    def _init_params(self, data):
        euler_angles_init = torch.zeros(self.V, 3, device=self.device)
        self.euler_angles = torch.nn.Parameter(euler_angles_init)

        T_init = torch.zeros(self.V, 3, device=self.device)
        T_init[:, 2] = -2.5
        self.T = torch.nn.Parameter(T_init)

        fov_guess = 45.0 * np.pi / 180.0
        f_init_val = self.image_size / (2.0 * np.tan(fov_guess / 2.0))
        f_init = torch.tensor([f_init_val], device=self.device)
        self.focal = torch.nn.Parameter(f_init)

        points_init = 0.2 * torch.randn(self.N, 3, device=self.device)
        self.points_3d = torch.nn.Parameter(points_init)

    def forward(self):
        R = euler_angles_to_matrix(self.euler_angles)
        predicted_2d = project(self.points_3d, R, self.T, self.focal, self.cx, self.cy)
        return predicted_2d

    def compute_loss(self, predicted_2d):
        diff = predicted_2d - self.observed_2d
        sq_error = (diff ** 2).sum(dim=-1)
        masked_sq = sq_error * self.mask
        visible_count = self.mask.sum() + 1e-8
        loss = masked_sq.sum() / visible_count
        return loss


def run_optimization(data, num_epochs=3000, lr=0.01, device='cpu'):
    """
    Run Bundle Adjustment optimization and return results.
    """
    ba = BundleAdjustment(data, device=device)
    
    params = [ba.euler_angles, ba.T, ba.focal, ba.points_3d]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-5)

    loss_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predicted_2d = ba.forward()
        loss = ba.compute_loss(predicted_2d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        loss_val = loss.item()
        scheduler.step(loss_val)
        loss_history.append(loss_val)

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            elapsed = time.time() - start_time
            msg = f"Epoch {epoch:5d}/{num_epochs} | Loss: {loss_val:.6f} | f: {ba.focal.item():.2f} | Time: {elapsed:.1f}s"
            print(msg, flush=True)
            with open('run_log.txt', 'a', encoding='utf-8') as lf:
                lf.write(msg + '\n')

    elapsed_total = time.time() - start_time
    msg = f"\nOptimization finished in {elapsed_total:.1f}s"
    print(msg, flush=True)
    with open('run_log.txt', 'a', encoding='utf-8') as lf:
        lf.write(msg + '\n')
    msg = f"Final loss: {loss_history[-1]:.6f}"
    print(msg, flush=True)
    with open('run_log.txt', 'a', encoding='utf-8') as lf:
        lf.write(msg + '\n')
    msg = f"Estimated focal: {ba.focal.item():.2f}"
    print(msg, flush=True)
    with open('run_log.txt', 'a', encoding='utf-8') as lf:
        lf.write(msg + '\n')
    if ba.f_gt is not None:
        msg = f"GT focal: {ba.f_gt:.2f}, Error: {abs(ba.focal.item()-ba.f_gt)/ba.f_gt*100:.2f}%"
        print(msg, flush=True)
        with open('run_log.txt', 'a', encoding='utf-8') as lf:
            lf.write(msg + '\n')

    return {
        'loss_history': loss_history,
        'euler_angles': ba.euler_angles.detach(),
        'T': ba.T.detach(),
        'focal': ba.focal.detach(),
        'points_3d': ba.points_3d.detach(),
        'colors': ba.colors,
    }, ba


# ========================
# Part 4: Visualization & Export
# ========================

def compute_visible_reproj_error(points_3d, R_est, T_est, focal, data, device='cpu'):
    mask = data['mask'].to(device)
    cx, cy = data['cx'], data['cy']
    with torch.no_grad():
        predicted = project(points_3d.to(device), R_est.to(device),
                           T_est.to(device), focal.to(device), cx, cy)
        observed = data['projections_2d'].to(device)
        sq_diff = ((predicted - observed) ** 2).sum(dim=-1)
        masked_sum = (sq_diff * mask).sum()
        visible_count = mask.sum() + 1e-8
        rmse = torch.sqrt(masked_sum / visible_count)
    return rmse.item()


def plot_loss_curve(loss_history, save_path='task1/loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=1)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Bundle Adjustment - Reprojection Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def export_colored_obj(points_3d, colors, save_path='task1/point_cloud.obj'):
    points = points_3d.cpu().numpy()
    cols = colors.cpu().numpy()
    cols = np.clip(cols, 0.0, 1.0)

    with open(save_path, 'w') as f:
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = cols[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

    print(f"Colored OBJ saved to {save_path} ({points.shape[0]} points)")


def plot_camera_positions(R_est, T_est, save_path='task1/cameras.png'):
    R = R_est.cpu().numpy()
    T = T_est.cpu().numpy()
    V = R.shape[0]

    cam_positions = []
    for i in range(V):
        C = -R[i].T @ T[i]
        cam_positions.append(C)
    cam_positions = np.array(cam_positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(0, 0, 0, c='red', s=100, marker='*', label='Origin')
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
               c='blue', s=30, alpha=0.7, label='Cameras')
    
    for i in range(V):
        C = cam_positions[i]
        direction = R[i].T @ np.array([0, 0, 1])
        ax.quiver(C[0], C[1], C[2], 
                  -direction[0]*0.3, -direction[1]*0.3, -direction[2]*0.3,
                  color='green', alpha=0.4, linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated Camera Positions')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Camera positions plot saved to {save_path}")


# ========================
# Part 5: Main Entry Point
# ========================

def log_print(msg):
    print(msg, flush=True)
    with open('run_log.txt', 'a', encoding='utf-8') as lf:
        lf.write(msg + '\n')


def main():
    device = 'cpu'
    task_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(task_dir)

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = None

    if os.path.exists('run_log.txt'):
        os.remove('run_log.txt')

    try:
        log_print("=" * 50)
        log_print("Task 1: Bundle Adjustment with PyTorch")
        log_print("=" * 50)
        log_print(f"Python: {sys.version}")
        log_print(f"PyTorch: {torch.__version__}")
        log_print(f"Device: {device}")

        if data_dir is not None:
            log_print(f"\n[1/4] Loading real data from: {data_dir}")
            data = load_real_data(data_dir, device=device)
        else:
            log_print("\n[1/4] Generating synthetic data...")
            log_print(f"  Views=50, Points=20000, Image=1024x1024")
            data = generate_synthetic_data(num_views=50, num_points=20000, image_size=1024, device=device)
            np.save('points3d_colors.npy', data['colors'].cpu().numpy())
            np.save('projections_2d.npy', data['projections_2d'].cpu().numpy())
            log_print(f"  - {data['projections_2d'].shape[0]} views, {data['projections_2d'].shape[1]} points")
            log_print(f"  - GT focal length: {data['f_gt']:.2f}")
            log_print(f"  - Data saved to projections_2d.npy, points3d_colors.npy")

        log_print(f"  - Projections tensor: {data['projections_2d'].shape}, "
                  f"memory: {data['projections_2d'].element_size() * data['projections_2d'].numel() / 1024/1024:.1f} MB")

        log_print("\n[2/4] Running Bundle Adjustment optimization...")
        log_print(f"  Epochs=500, LR=0.01")
        results, ba_model = run_optimization(data, num_epochs=500, lr=0.01, device=device)
        log_print(f"  Final loss: {results['loss_history'][-1]:.6f}")

        log_print("\n[3/4] Visualizing results...")
        R_est = euler_angles_to_matrix(results['euler_angles'])
        plot_loss_curve(results['loss_history'], save_path='loss_curve.png')
        export_colored_obj(results['points_3d'], results['colors'], save_path='point_cloud.obj')
        plot_camera_positions(R_est.cpu(), results['T'], save_path='cameras.png')

        log_print("\n[4/4] Computing errors...")
        reproj_error = compute_visible_reproj_error(
            results['points_3d'], R_est, results['T'],
            results['focal'], data, device)
        log_print(f"  Visible reprojection error: {reproj_error:.4f} px")

        if data.get('data_type') == 'synthetic':
            with torch.no_grad():
                points_gt = data['points_3d_gt'].cpu()
                points_est = results['points_3d'].cpu()
                mean_3d_error = torch.mean(torch.norm(points_gt - points_est, dim=1)).item()
                f_error = abs(results['focal'].item() - data['f_gt']) / data['f_gt'] * 100
            log_print(f"  Mean 3D point error: {mean_3d_error:.4f}")
            log_print(f"  Focal length error: {f_error:.2f}%")
            log_print(f"  GT focal: {data['f_gt']:.2f}, Estimated: {results['focal'].item():.2f}")

        log_print("\n" + "=" * 50)
        log_print("Task 1 Complete! Output files:")
        log_print("  - loss_curve.png")
        log_print("  - point_cloud.obj")
        log_print("  - cameras.png")
        log_print("  - run_log.txt")
        log_print("=" * 50)

    except Exception as e:
        log_print(f"\nERROR: {e}")
        log_print(traceback.format_exc())


if __name__ == '__main__':
    main()
