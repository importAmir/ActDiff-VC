"""Budget-aware sparse trajectory selection (paper Sec. 4.4) and the uniform / high-motion ablations."""
import math

import numpy as np
import torch

from actdiff.point_selection.hed import HEDDetector

# RBF bandwidths tried during the sigma fit, in units of the initial grid cell size.
SIGMA_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
MAX_ITERATIONS = 100


def Grid_Layout(H, W, count):
    """Square-cell grid of about `count` cells, centred in the image. Returns (R, C, cell, y0, x0)."""
    aspect = H / max(W, 1)
    R = max(1, int(round(math.sqrt(count * aspect))))
    C = max(1, int(math.ceil(count / R)))
    R, C = min(R, H), min(C, W)
    if R * C < count:
        if W / max(H, 1) >= 1.0:
            C = min(W, C + (count - R * C + R - 1) // R)
        else:
            R = min(H, R + (count - R * C + C - 1) // C)
    cell = max(1, min(H // R, W // C))
    return R, C, cell, max(0, (H - R * cell) // 2), max(0, (W - C * cell) // 2)


def Grid_Maxima(map_2d, count):
    """Per-cell maximum of `map_2d`: (ys, xs, values) flattened over the R x C grid, and the layout."""
    H, W = map_2d.shape
    R, C, cell, y_start, x_start = layout = Grid_Layout(H, W, count)
    cropped = map_2d[y_start : y_start + R * cell, x_start : x_start + C * cell]
    cells = cropped.reshape(R, cell, C, cell).permute(0, 2, 1, 3).reshape(R, C, cell * cell)
    values, idx = torch.max(cells, dim=2)
    y0 = y_start + torch.arange(R, device=map_2d.device).unsqueeze(1) * cell
    x0 = x_start + torch.arange(C, device=map_2d.device).unsqueeze(0) * cell
    ys = (y0 + torch.div(idx, cell, rounding_mode='floor')).reshape(-1)
    xs = (x0 + idx % cell).reshape(-1)
    return ys, xs, values.reshape(-1), layout


def Gather_Tracks(trajectories, points_yx):
    """trajectories [1, T, 2, H, W], points [N, 2] (y, x) -> tracks [T, N, 2] (x, y)."""
    return trajectories[0][:, :, points_yx[:, 0].long(), points_yx[:, 1].long()].permute(0, 2, 1).contiguous()


def To_Int16(tracks):
    return np.rint(tracks.detach().cpu().numpy()).astype(np.int16)


def L2_Distances(points_yx, H, W, device):
    """Euclidean distance from every point to every pixel: [N, H, W]."""
    points_yx = points_yx.to(device=device, dtype=torch.float32)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    dy = yy - points_yx[:, 0].view(-1, 1, 1)
    dx = xx - points_yx[:, 1].view(-1, 1, 1)
    return torch.sqrt(dy * dy + dx * dx + 1e-8)


def Rbf_Scores(distances, sigma):
    """Unnormalized RBF kernel values exp(-d^2 / 2 sigma^2)."""
    return torch.exp(-(distances * distances) / (2.0 * (float(sigma) ** 2) + 1e-8))


def Interpolate_Displacements(scores, displacements):
    """Normalized-RBF interpolation of point displacements [T, N, 2] to a dense field [T, 2, H, W]."""
    weights = scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return torch.einsum('tnc,nhw->tchw', displacements, weights)


def Weighted_Error(estimate, dense, weight):
    """Per-pixel sketch-weighted motion error, [T, H, W]."""
    diff = estimate - dense
    return torch.sqrt((diff * diff).sum(dim=1) + 1e-8) * weight


class PointSelector:
    def __init__(self, args):
        self.device = args.device
        self.hed = HEDDetector(args.hed_checkpoint, args.device)

    def Get_Sketch(self, image):
        """HED edge map of a [3, H, W] image in [0, 1], as HxW uint8."""
        return self.hed(image.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def Uniform_Selection(self, trajectories, num_points):
        """Ablation: trajectories on a regular grid. Returns int16 tracks [T, N, 2]."""
        _, _, _, H, W = trajectories.shape
        num_points = min(num_points, H * W)
        num_cols = int(np.ceil(np.sqrt(num_points)))
        num_rows = int(np.ceil(num_points / num_cols))
        ys = torch.linspace(0, H - 1, steps=num_rows, device=trajectories.device).round().long()
        xs = torch.linspace(0, W - 1, steps=num_cols, device=trajectories.device).round().long()
        points = torch.cartesian_prod(ys, xs)[:num_points]
        return To_Int16(Gather_Tracks(trajectories, points))

    @torch.no_grad()
    def High_Motion_Selection(self, trajectories, refine_grid_cells, num_trajectories):
        """Ablation: the largest-motion pixel of each grid cell, keeping the top `num_trajectories`."""
        _, T, _, H, W = trajectories.shape
        traj = trajectories[0]
        deltas = traj[1:] - traj[:-1]
        motion = torch.sqrt((deltas * deltas).sum(dim=1) + 1e-8).mean(dim=0)
        ys, xs, values, _ = Grid_Maxima(motion, refine_grid_cells)
        num_points = min(num_trajectories, len(values))
        if num_points == 0:
            return np.empty((T, 0, 2), dtype=np.int16)
        top = torch.argsort(values, descending=True)[:num_points]
        points = torch.stack([ys[top].long().clamp(0, H - 1), xs[top].long().clamp(0, W - 1)], dim=1)
        return To_Int16(Gather_Tracks(trajectories, points))

    @torch.no_grad()
    def Sparse_Selection(
        self,
        sketch,
        trajectories,
        num_anchor_cells,
        refine_grid_cells,
        anchor_threshold,
        num_trajectories,
        points_per_step=10,
        use_uniform_weights=False,
    ):
        """Greedy RBF-residual point selection.

        1. Anchors: the strongest sketch pixel of each coarse grid cell (if above threshold).
        2. Fit the RBF bandwidth on the anchors only.
        3. Repeatedly add the largest-residual pixel of each empty sub-grid cell until the budget is used.
        Returns (int16 tracks [T, N, 2], sigma index), or (empty tracks, None) when no anchor is found.
        """
        device = self.device
        H, W = sketch.shape
        if use_uniform_weights:
            weight = torch.ones((H, W), device=device, dtype=torch.float32)
        else:
            weight = torch.from_numpy(sketch).to(device=device, dtype=torch.float32) / 255.0

        ys, xs, values, _ = Grid_Maxima(weight, num_anchor_cells)
        keep = values >= float(anchor_threshold)
        points = torch.stack([ys[keep], xs[keep]], dim=1).to(dtype=torch.int16)
        if points.numel() == 0:
            return np.empty((trajectories.shape[1], 0, 2), dtype=np.int16), None

        tracks = Gather_Tracks(trajectories, points)
        dense = (trajectories[0] - trajectories[0][0:1]).to(device=device, dtype=tracks.dtype)
        weight = weight.to(dtype=dense.dtype)

        sigma, sigma_index, estimate, scores = self.Fit_Sigma(points, tracks - tracks[0:1], dense, weight)

        for _ in range(MAX_ITERATIONS):
            budget = int(num_trajectories) - int(points.shape[0])
            if budget <= 0:
                break
            residual = Weighted_Error(estimate, dense, weight).mean(dim=0)
            added = self.Select_Additional_Points(
                residual, refine_grid_cells, min(budget, points_per_step), points
            )
            if added.numel() == 0:
                break
            tracks = torch.cat([tracks, Gather_Tracks(trajectories, added)], dim=1)
            points = torch.cat([points, added], dim=0)
            scores = torch.cat([scores, Rbf_Scores(L2_Distances(added.long(), H, W, device), sigma)], dim=0)
            estimate = Interpolate_Displacements(scores, tracks - tracks[0:1])

        return To_Int16(tracks), sigma_index

    @torch.no_grad()
    def Fit_Sigma(self, points, displacements, dense, weight):
        """Grid-search the RBF bandwidth minimizing the weighted motion error of the anchor interpolation."""
        H, W = weight.shape
        _, _, cell, _, _ = Grid_Layout(H, W, max(1, int(math.sqrt(H * W) // 8)))
        distances = L2_Distances(points.long(), H, W, self.device)

        best = (float('inf'), None, None, None, None)
        for index, factor in enumerate(SIGMA_GRID):
            sigma = factor * max(1.0, float(cell))
            scores = Rbf_Scores(distances, sigma)
            estimate = Interpolate_Displacements(scores, displacements)
            loss = Weighted_Error(estimate, dense, weight).mean().item()
            if loss < best[0]:
                best = (loss, sigma, index, estimate, scores)
        return best[1:]

    @torch.no_grad()
    def Select_Additional_Points(self, residual, refine_grid_cells, max_points, points):
        """Top-`max_points` residual maxima among sub-grid cells that do not contain a point yet."""
        ys, xs, values, (R, C, cell, y_start, x_start) = Grid_Maxima(residual, refine_grid_cells)
        occupied = torch.zeros((R, C), dtype=torch.bool, device=residual.device)
        rows = ((points[:, 0].long() - y_start).clamp(min=0) // cell).clamp(0, R - 1)
        cols = ((points[:, 1].long() - x_start).clamp(min=0) // cell).clamp(0, C - 1)
        occupied[rows, cols] = True

        keep = (~occupied).reshape(-1)
        ys, xs, values = ys[keep], xs[keep], values[keep]
        keep = values > 1e-6
        ys, xs, values = ys[keep], xs[keep], values[keep]
        if ys.numel() == 0:
            return torch.empty((0, 2), dtype=torch.int16, device=residual.device)
        order = torch.argsort(values, descending=True)[:max_points]
        return torch.stack([ys[order], xs[order]], dim=1).to(dtype=torch.int16)
