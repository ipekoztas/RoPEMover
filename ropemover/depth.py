"""
Depth utilities for RoPEMover.

The model consumes a *synthesized target depth map*: the source depth with the
object removed (inpainted background) and re-inserted at its new location with
a depth offset ``delta_d``. This map is normalized to the PE-field range [1, 2]
and used as the depth coordinate of the 3D RoPE for both source and target
tokens.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (binary_dilation, binary_erosion,
                           distance_transform_edt, gaussian_filter)
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

DEPTH_RES = 512


# ─────────────────────────────────────────────────────────────────────────────
# Monocular depth (MoGe-2)
# ─────────────────────────────────────────────────────────────────────────────

class MoGeDepthEstimator:
    """Metric depth with MoGe-2 (``Ruicheng/moge-2-vitl-normal``)."""

    def __init__(self, model_id="Ruicheng/moge-2-vitl-normal", device="cuda"):
        from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image: Image.Image, resolution: int = DEPTH_RES) -> np.ndarray:
        """Returns raw MoGe depth (H=W=resolution) for ``image`` resized to resolution²."""
        image = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        x = torch.tensor(np.array(image) / 255.0, dtype=torch.float32,
                         device=self.device).permute(2, 0, 1)
        depth = self.model.infer(x)["depth"].cpu().numpy()
        finite = depth[np.isfinite(depth)]
        depth[~np.isfinite(depth)] = finite.max()
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        return depth.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Background inpainting
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_inpaint(D, hole_mask):
    """Solve ∇²D = 0 inside the hole via sparse linear system."""
    H, W = D.shape
    hole_idx = np.flatnonzero(hole_mask)
    n = len(hole_idx)

    idx_map = np.full(H * W, -1, dtype=np.int32)
    idx_map[hole_idx] = np.arange(n)

    D_flat    = D.ravel().astype(np.float64)
    hole_flat = hole_mask.ravel()

    A = lil_matrix((n, n))
    b = np.zeros(n, dtype=np.float64)

    for eq, fi in enumerate(hole_idx):
        r, c = divmod(int(fi), W)
        neighbours = []
        if r > 0:     neighbours.append((r - 1) * W + c)
        if r < H - 1: neighbours.append((r + 1) * W + c)
        if c > 0:     neighbours.append(r * W + c - 1)
        if c < W - 1: neighbours.append(r * W + c + 1)

        A[eq, eq] = len(neighbours)
        for nb in neighbours:
            if hole_flat[nb]:
                A[eq, idx_map[nb]] -= 1.0
            else:
                b[eq] += D_flat[nb]

    x = spsolve(A.tocsr(), b)
    D_out = D.copy()
    D_out[hole_mask] = x.astype(D.dtype)
    return D_out


def gaussian_diffusion_inpaint(D, hole_mask, iterations=200):
    """Iterative Gaussian blur, only updating hole pixels."""
    D_out = D.copy()
    _, idx = distance_transform_edt(hole_mask, return_indices=True)
    D_out[hole_mask] = D[idx[0][hole_mask], idx[1][hole_mask]]
    for _ in range(iterations):
        D_smooth = gaussian_filter(D_out, sigma=1.0)
        D_out[hole_mask] = D_smooth[hole_mask]
    return D_out.astype(np.float32)


def opencv_inpaint(D, hole_mask, method="telea", radius=5):
    """OpenCV Telea / Navier-Stokes inpainting (uint16 quantised)."""
    d_min, d_max = D.min(), D.max()
    D_norm = ((D - d_min) / (d_max - d_min + 1e-8) * 65535).astype(np.uint16)
    mask_u8 = hole_mask.astype(np.uint8) * 255
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    D_filled = cv2.inpaint(D_norm, mask_u8, radius, flag)
    return (D_filled.astype(np.float32) / 65535.0 * (d_max - d_min) + d_min)


def safe_border_inpaint(D, hole_mask, safety_px=3, method="laplace"):
    """
    Inpaint with a safety margin: dilate hole by safety_px, replace the
    contaminated border ring with nearest clean background values, then
    run the chosen inpainting method on the original hole.
    """
    expanded_hole = binary_dilation(hole_mask, iterations=safety_px)
    contaminated  = expanded_hole & ~hole_mask

    if contaminated.sum() > 0:
        clean_bg = ~expanded_hole
        _, idx = distance_transform_edt(~clean_bg, return_indices=True)
        D_clean = D.copy()
        D_clean[contaminated] = D[idx[0][contaminated], idx[1][contaminated]]
    else:
        D_clean = D

    if method == "gauss":
        return gaussian_diffusion_inpaint(D_clean, hole_mask)
    elif method == "ns":
        return opencv_inpaint(D_clean, hole_mask, method="ns")
    elif method == "telea":
        return opencv_inpaint(D_clean, hole_mask, method="telea")
    return laplacian_inpaint(D_clean, hole_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Target depth synthesis
# ─────────────────────────────────────────────────────────────────────────────

def derive_tgt_mask(M_src, dx_px, dy_px):
    H, W = M_src.shape
    M_tgt = np.zeros_like(M_src)
    rows, cols = np.where(M_src)
    new_rows = np.clip(rows + dy_px, 0, H - 1)
    new_cols = np.clip(cols + dx_px, 0, W - 1)
    M_tgt[new_rows, new_cols] = True
    return M_tgt


def blend_boundaries(D_out, M_tgt, D_bg, radius=3):
    dilated  = binary_dilation(M_tgt, iterations=radius)
    eroded   = binary_erosion(M_tgt,  iterations=radius)
    boundary = dilated & ~eroded
    dist     = distance_transform_edt(M_tgt)
    weight   = np.clip(dist / (radius + 1e-8), 0.0, 1.0)
    obj_in_front = M_tgt & (D_out < D_bg)
    blend_region = boundary & obj_in_front
    D_blended = D_out.copy()
    if blend_region.sum() > 0:
        w = weight[blend_region]
        D_blended[blend_region] = (w * D_out[blend_region]
                                   + (1 - w) * D_bg[blend_region])
    return D_blended


def synthesize_target_depth(D_s, M_src, dx_px, dy_px, delta_d):
    """
    Per-pixel reverse-warp with safety-margin inpainting.
    Each target pixel gets its source pixel's depth + delta_d.

    Args:
        D_s:     (H, W) raw source depth.
        M_src:   (H, W) bool source object mask (same resolution as D_s).
        dx_px:   horizontal displacement in D_s pixels.
        dy_px:   vertical displacement in D_s pixels.
        delta_d: depth offset added to the object (same units as D_s;
                 positive = away from camera).
    Returns:
        D_out:   (H, W) synthesized target depth.
        M_tgt:   (H, W) bool target object mask.
    """
    H, W = D_s.shape
    M_tgt = derive_tgt_mask(M_src, dx_px, dy_px)

    D_bg = safe_border_inpaint(D_s, hole_mask=M_src, safety_px=3, method="laplace")

    # Reverse-warp: for each target pixel, find its source pixel
    tgt_rows, tgt_cols = np.where(M_tgt)
    rev_rows = (tgt_rows - dy_px).astype(np.int64)
    rev_cols = (tgt_cols - dx_px).astype(np.int64)

    valid = ((rev_rows >= 0) & (rev_rows < H) &
             (rev_cols >= 0) & (rev_cols < W))
    valid[valid] &= M_src[rev_rows[valid], rev_cols[valid]]

    D_obj = np.full((H, W), np.nan, dtype=np.float32)
    D_obj[tgt_rows[valid],  tgt_cols[valid]]  = D_s[rev_rows[valid], rev_cols[valid]] + delta_d
    D_obj[tgt_rows[~valid], tgt_cols[~valid]] = float(np.median(D_s[M_src])) + delta_d

    # z-test: the object only overwrites background that lies behind it
    D_out = D_bg.copy()
    obj_d = D_obj[M_tgt]
    bg_d  = D_bg[M_tgt]
    D_out[M_tgt] = np.where(obj_d < bg_d, obj_d, bg_d)

    D_out = blend_boundaries(D_out, M_tgt, D_bg, radius=3)
    return D_out, M_tgt


def to_pefield(depth):
    """Min-max normalize depth to the PE-field range [1, 2]."""
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    return (depth_norm + 1.0).astype(np.float32)


def depth_to_uint8(depth):
    d_min, d_max = depth.min(), depth.max()
    return ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
