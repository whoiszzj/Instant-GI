from symbol import pass_stmt

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sympy import ceiling
import torch.nn as nn
from torch_dither import torch_image_dither
from torch_kdtree import build_kd_tree
from pytorch_msssim import ms_ssim, ssim
from generalizable_model.ssim import SSIM

def dither_image(position_field, kernel_size=3, image=None):
    B, oH, oW = position_field.shape
    k = kernel_size
    # position_field_ori = position_field.clone()
    position_field = F.pad(position_field, (0, (k - oW % k) % k, 0, (k - oH % k) % k), "replicate")
    B, H, W = position_field.shape
    position_field = F.unfold(position_field.unsqueeze(1), kernel_size=(k, k), stride=(k, k))
    position_field = position_field.max(dim=1)[0].view(B, H // k, W // k)
    sampled_xy = torch_image_dither(position_field[0]) * k + k / 2
    sampled_xy = sampled_xy[(sampled_xy[:, 0] < oW) & (sampled_xy[:, 1] < oH)]

    # print(f"sampled_xy:{sampled_xy.shape}")
    # sampled_xy_np = sampled_xy.cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(position_field_ori[0].cpu().numpy(), cmap='gray')
    # plt.scatter(sampled_xy_np[:, 0], sampled_xy_np[:, 1], c='r', s=1)
    # plt.show()
    # exit(0)
    return sampled_xy


def min_bounding_ellipse(vertices):
    midpoints = (vertices[:3] + np.roll(vertices[:3], -1, axis=0)) / 2
    vertices_added = np.vstack((vertices[:3], midpoints))
    vertices_added = vertices_added.reshape(-1, 1, 2).astype(np.float32)
    ellipse = cv2.fitEllipse(vertices_added)
    center, axes, angle = ellipse
    return center, axes, angle


def add_boundary_points(points, H, W):
    tree = build_kd_tree(points)
    dist, idx = tree.query(points, 2)
    dist = torch.sqrt(dist[:, 1])
    mean_dist = dist.mean()
    add_point_interval = 3 * mean_dist
    points = points.cpu()
    x_interval_num = int(ceiling(W / add_point_interval).evalf())
    y_interval_num = int(ceiling(H / add_point_interval).evalf())
    x_sample = torch.linspace(0, W - 1, steps=x_interval_num)
    y_sample = torch.linspace(0, H - 1, steps=y_interval_num)
    x_sample = x_sample.unsqueeze(1)
    y_sample = y_sample.unsqueeze(1)
    new_points = torch.cat([x_sample, torch.zeros_like(x_sample)], dim=1)
    new_points = torch.cat([new_points, torch.cat([x_sample, torch.ones_like(x_sample) * (H - 1)], dim=1)], dim=0)
    new_points = torch.cat([new_points, torch.cat([torch.zeros_like(y_sample), y_sample], dim=1)], dim=0)
    new_points = torch.cat([new_points, torch.cat([torch.ones_like(y_sample) * (W - 1), y_sample], dim=1)], dim=0)
    points = torch.cat([points, new_points], dim=0)
    points = torch.unique(points, dim=0)
    points = points.numpy()
    return points


def tri_area(tri):
    x1, y1 = tri[:, 0, 0], tri[:, 0, 1]
    x2, y2 = tri[:, 1, 0], tri[:, 1, 1]
    x3, y3 = tri[:, 2, 0], tri[:, 2, 1]
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return areas


def neighbors_process(neighbors):
    mask = neighbors == -1
    self_index = np.arange(neighbors.shape[0]).reshape(-1, 1).repeat(3, axis=1)
    neighbors[mask] = self_index[mask]
    return neighbors


def calculate_circumcenters_and_radii(vertices):
    A = vertices[:, 0, :]  # [N, 2]
    B = vertices[:, 1, :]  # [N, 2]
    C = vertices[:, 2, :]  # [N, 2]

    mid_AB = (A + B) / 2  # [N, 2]
    mid_BC = (B + C) / 2  # [N, 2]

    d_AB = B - A  # [N, 2]
    d_BC = C - B  # [N, 2]

    perp_AB = torch.stack([-d_AB[:, 1], d_AB[:, 0]], dim=1)  # [N, 2]
    perp_BC = torch.stack([-d_BC[:, 1], d_BC[:, 0]], dim=1)  # [N, 2]

    A_matrix = torch.stack([perp_AB, -perp_BC], dim=1).transpose(1, 2)  # [N, 2, 2]
    b_vector = (mid_BC - mid_AB).unsqueeze(2)  # [N, 2, 1]

    try:
        A_inv = torch.linalg.inv(A_matrix)  # [N, 2, 2]
        t_s = torch.bmm(A_inv, b_vector).squeeze(2)  # [N, 2]
        circumcenters = mid_AB + t_s[:, 0:1] * perp_AB  # [N, 2]
    except RuntimeError:
        circumcenters = torch.full((vertices.size(0), 2), float('nan'), device=vertices.device)

    radii = torch.norm(circumcenters - A, dim=1)  # [N]

    return circumcenters, radii


class FocalMSELoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        error = torch.abs(y_true - y_pred)
        focal_weight = self.alpha * (1 - torch.exp(-error)) ** self.gamma
        loss = focal_weight * (error ** 2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RGBLoss(nn.Module):
    def __init__(self, lambda_val=0.7):
        super(RGBLoss, self).__init__()
        self.lambda_val = lambda_val
        self.ssim = SSIM(size_average=False)

    def forward(self, pred, target, weight):
        mse_loss = F.mse_loss(pred, target, reduction='none')
        mse_loss = mse_loss * weight.unsqueeze(1)
        mse_loss = mse_loss.mean()

        ssim_loss = 1 - self.ssim(pred, target)
        ssim_loss = ssim_loss * weight
        ssim_loss = ssim_loss.mean()

        loss = self.lambda_val * mse_loss + (1 - self.lambda_val) * ssim_loss
        return loss
