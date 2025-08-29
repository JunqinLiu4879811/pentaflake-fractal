"""
Pentaflake – PyTorch implementation with analysis
- Tensor-parallel placement of child pentagons (GPU-friendly)
- Safe rendering (default draws all layers, avoids blank output)
- Multiple visualization options (colormap)
- Box-counting method to estimate fractal dimension (log–log fit)
Please document your AI prompts and modification records in prompts.md to satisfy assignment requirements.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Geometry utilities
# ---------------------------
@torch.inference_mode()
def regular_pentagon_vertices(center: torch.Tensor,
                              scale: torch.Tensor,
                              angle0: float = math.pi/2) -> torch.Tensor:
    """
    Batch-generate vertices of regular pentagons.
    center: (B,) complex tensor (cx + i cy)
    scale : (B,) radius (circumcircle radius)
    Returns: (B, 5, 2) float (x,y) coordinates
    """
    B = center.shape[0]
    k = torch.arange(5, device=center.device, dtype=torch.float32)
    angles = angle0 + 2*math.pi*k/5.0  # (5,)
    unit = torch.exp(1j * angles)      # (5,)
    verts_c = center.unsqueeze(1) + (scale.unsqueeze(1) * unit.unsqueeze(0))  # (B,5)
    verts = torch.stack([verts_c.real, verts_c.imag], dim=-1)  # (B,5,2)
    return verts

@torch.inference_mode()
def pentaflake_iterate(n_iter=5,
                       base_scale=1.0,
                       scale_ratio=None,
                       spacing=None,
                       angle0=math.pi/2):
    """
    Iteratively generate all levels of pentagon centers and scales:
    Each pentagon → 6 child pentagons (1 at center + 5 at vertex directions).
    scale_ratio: scaling factor for each level; default theoretical s = 1/(1+phi) ≈ 0.381966
    spacing: displacement factor for outer child pentagons (multiplied by current scale);
             default 1.10 for clearer visualization
    """
    if scale_ratio is None:
        phi = (1 + math.sqrt(5)) / 2.0
        scale_ratio = 1.0 / (1.0 + phi)  # ≈ 0.381966

    if spacing is None:
        spacing = 1.10  # Slightly spread out to avoid overlap and improve clarity

    centers = [torch.zeros(1, dtype=torch.complex64, device=device)]           # (1,)
    scales  = [torch.tensor([base_scale], dtype=torch.float32, device=device)] # (1,)

    k = torch.arange(5, device=device, dtype=torch.float32)
    dir5 = torch.exp(1j * (angle0 + 2*math.pi*k/5.0))  # (5,)

    for _ in range(n_iter):
        c = centers[-1]  # (B,)
        s = scales[-1]   # (B,)

        new_scale = s * scale_ratio
        c0 = c.unsqueeze(1)  # (B,1)

        cout = c.unsqueeze(1) + dir5.unsqueeze(0) * (spacing * s).unsqueeze(1)  # (B,5)

        c_next = torch.cat([c0, cout], dim=1).reshape(-1)             # (B*6,)
        s_next = new_scale.unsqueeze(1).expand(-1, 6).reshape(-1)     # (B*6,)

        centers.append(c_next)
        scales.append(s_next)

    all_centers = torch.cat(centers, dim=0)
    all_scales  = torch.cat(scales,  dim=0)
    return all_centers, all_scales, (scale_ratio, spacing)

# ---------------------------
# Safe rendering (avoid blank output)
# ---------------------------
@torch.inference_mode()
def render_pentaflake_safe(n_iter=3,
                           base_scale=1.0,
                           scale_ratio=None,
                           spacing=None,
                           cmap="viridis",
                           linewidth=0.7,
                           face_alpha=0.95):
    """
    Robust Pentaflake visualization: 
    - Default draws all levels
    - Automatically adjusts margins
    - Prevents blank images
    """
    centers, scales, params = pentaflake_iterate(
        n_iter=n_iter, base_scale=base_scale,
        scale_ratio=scale_ratio, spacing=spacing
    )
    sratio, sp = params

    verts = regular_pentagon_vertices(centers, scales)  # (B,5,2)
    verts_np = verts.detach().cpu().numpy()

    # Level-based coloring for better distinction
    levels = torch.log(scales/base_scale + 1e-12)/math.log(sratio + 1e-12)
    levels = (-levels).round().to(torch.int64).clamp(min=0, max=n_iter)
    colors = (levels.detach().cpu().numpy() / max(1, n_iter)).astype(float)

    fig = plt.figure(figsize=(8, 8))
    coll = PolyCollection(
        verts_np,
        array=colors, cmap=cmap,
        edgecolors="k", linewidths=linewidth,
        closed=True
    )
    coll.set_alpha(face_alpha)

    ax = plt.gca()
    ax.add_collection(coll)
    ax.autoscale()
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    pad_x = 0.08 * (xlim[1]-xlim[0] + 1e-9)
    pad_y = 0.08 * (ylim[1]-ylim[0] + 1e-9)
    ax.set_xlim(xlim[0]-pad_x, xlim[1]+pad_x)
    ax.set_ylim(ylim[0]-pad_y, ylim[1]+pad_y)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.title(f"Pentaflake (n_iter={n_iter}, s={sratio:.4f}, d×s={sp:.2f})")
    plt.tight_layout()
    plt.show()

    return centers, scales, params

# ---------------------------
# Boundary sampling & box-counting dimension
# ---------------------------
@torch.inference_mode()
def sample_boundary_points(centers: torch.Tensor,
                           scales: torch.Tensor,
                           points_per_edge=12,
                           angle0: float = math.pi/2) -> np.ndarray:
    """
    Sample boundary points from all pentagons for box-counting dimension estimation.
    Returns: (N,2) numpy array, normalized to [0,1]^2
    """
    verts = regular_pentagon_vertices(centers, scales, angle0=angle0)  # (B,5,2)
    B = verts.shape[0]
    segs = []
    t = torch.linspace(0, 1, points_per_edge, device=verts.device).view(1, -1, 1)  # (1,P,1)
    for i in range(5):
        v0 = verts[:, i, :]
        v1 = verts[:, (i+1) % 5, :]
        pts = v0.unsqueeze(1)*(1-t) + v1.unsqueeze(1)*t   # (B,P,2)
        segs.append(pts)
    pts = torch.cat(segs, dim=1).reshape(-1, 2)  # (N,2)

    pts_np = pts.detach().cpu().numpy()
    mn = pts_np.min(axis=0); mx = pts_np.max(axis=0)
    pts01 = (pts_np - mn) / (mx - mn + 1e-12)
    return pts01

def box_count_dimension(points01: np.ndarray,
                        grid_sizes=(8, 16, 32, 64, 128, 256)):
    """
    Box-counting method: count occupied grid cells at different scales.
    Returns slope D and data (x,y).
    """
    counts, inv_eps = [], []
    for n in grid_sizes:
        grid = np.zeros((n, n), dtype=bool)
        ix = np.clip((points01[:,0]*n).astype(int), 0, n-1)
        iy = np.clip((points01[:,1]*n).astype(int), 0, n-1)
        grid[iy, ix] = True
        counts.append(grid.sum())
        inv_eps.append(n)

    x = np.log(np.array(inv_eps, dtype=float))
    y = np.log(np.array(counts,  dtype=float) + 1e-9)
    a, b = np.polyfit(x, y, 1)
    return a, (x, y)

# ---------------------------
# Main program: Visualization & analysis
# ---------------------------
if __name__ == "__main__":
    # Start with lower iteration count to verify correctness
    render_pentaflake_safe(n_iter=2, cmap="viridis")
    render_pentaflake_safe(n_iter=3, cmap="Blues")

    # Higher iterations (slower but more detailed)
    render_pentaflake_safe(n_iter=4, cmap="plasma")

    # —— Fractal dimension (box-counting) analysis ——
    centers, scales, _ = pentaflake_iterate(n_iter=4)
    pts01 = sample_boundary_points(centers, scales, points_per_edge=14)
    D, (lx, ly) = box_count_dimension(pts01)
    print(f"Estimated box-counting dimension D ≈ {D:.3f}")

    plt.figure(figsize=(5,4))
    plt.plot(lx, ly, "o", label="data")
    a,b = np.polyfit(lx, ly, 1)
    plt.plot(lx, a*lx+b, "-", label=f"fit: D={a:.3f}")
    plt.xlabel("log(1/epsilon)")
    plt.ylabel("log N_boxes")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Theoretical reference: D_th = log(6)/log(1+phi)
    phi = (1 + math.sqrt(5)) / 2.0
    D_th = math.log(6) / math.log(1 + phi)
    print(f"Theoretical dimension D_th = {D_th:.4f}")

