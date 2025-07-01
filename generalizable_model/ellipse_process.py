import os
import numpy as np
import torch
from scipy.spatial import Delaunay
from generalizable_model.utils import add_boundary_points, min_bounding_ellipse, dither_image, neighbors_process
from ellipse_fit import fit_ellipses
import cupy as cp
from cupyx.scipy.spatial import Delaunay as DelaunayGPU
import time

# pool = None
# def init_pool(cpu_num):
#     global pool
#     pool = torch.multiprocessing.Pool(cpu_num)

def ellipse_filter(ellipses_size):
    # ellipse_size: [N, 2]
    less_mask = ellipses_size < 1
    less_mask = less_mask.any(dim=1)
    # great mask
    mean = ellipses_size.mean()
    std = ellipses_size.std()
    great_mask = ellipses_size > (mean + std)
    great_mask = great_mask.any(dim=1)
    mask = less_mask | great_mask
    return mask
    
    
def simplices_to_neighbors(simplices):
    """
    Compute the neighbors for each triangle in the triangulation using torch (CUDA).
    Returns an array of shape (n_tri, 3), where each entry is the index of the neighboring triangle
    sharing the corresponding edge, or the triangle's own index if there is no neighbor.
    """
    device = simplices.device
    n_tri = simplices.shape[0]
    # 1. Construct three edges for each triangle (n_tri, 3, 2)
    edges = torch.stack([
        torch.stack([simplices[:, 0], simplices[:, 1]], dim=1),
        torch.stack([simplices[:, 1], simplices[:, 2]], dim=1),
        torch.stack([simplices[:, 2], simplices[:, 0]], dim=1)
    ], dim=1)
    # 2. Sort to ensure undirected edge uniqueness
    edges, _ = torch.sort(edges, dim=2)
    # 3. Flatten to (n_tri*3, 2)
    flat_edges = edges.reshape(-1, 2)
    tri_ids = torch.arange(n_tri, device=device).repeat_interleave(3)
    edge_ids = torch.arange(3, device=device).repeat(n_tri)
    # 4. Use a unique key for each edge
    edge_keys = flat_edges[:, 0].to(torch.int64) * (simplices.max()+1) + flat_edges[:, 1].to(torch.int64)
    unique_edges, inverse_indices, counts = torch.unique(edge_keys, return_inverse=True, return_counts=True)
    # 5. Find all triangles corresponding to each edge
    mask = counts[inverse_indices] == 2
    idxs = torch.nonzero(mask, as_tuple=False).flatten()
    # 6. For each edge with neighbors, find its two triangles
    sort_idx = torch.argsort(inverse_indices[idxs])
    idxs_sorted = idxs[sort_idx]
    # Group every two
    t0 = tri_ids[idxs_sorted][0::2].to(torch.int32)
    e0 = edge_ids[idxs_sorted][0::2].to(torch.int32)
    t1 = tri_ids[idxs_sorted][1::2].to(torch.int32)
    e1 = edge_ids[idxs_sorted][1::2].to(torch.int32)
    # 7. Construct adjacency matrix
    neighbors = torch.arange(n_tri, device=device).unsqueeze(1).repeat(1, 3).to(torch.int32)
    neighbors[t0, e0] = t1
    neighbors[t1, e1] = t0
    return neighbors
    
class EllipseProcess:
    def __init__(self):
        # self.cpu_num = len(os.sched_getaffinity(0))
        self.triangles = None
        # init_pool(self.cpu_num)
        
        
    def fit_ellipse_cuda(self):
        # self.triangles [N, 3, 2]
        # Add midpoints of each triangle's edges as new points
        midpoints = (self.triangles + torch.roll(self.triangles, -1, dims=1)) / 2
        points = torch.cat([self.triangles, midpoints], dim=1)  # [N, 6, 2]
        # to tensor
        fit_result = fit_ellipses(points)
        return fit_result[:, :2], fit_result[:, 2:4], fit_result[:, 4]

    # def fit_task_batch(self, start, end):
    #     centers, axes, angles = [], [], []
    #     for i in range(start, end):
    #         tri_ver = self.triangles[i]
    #         center, axis, angle = min_bounding_ellipse(tri_ver)
    #         centers.append(center)
    #         axes.append(axis)
    #         angles.append(angle)
    #     return np.array(centers), np.array(axes), np.array(angles)

    # def fit_ellipse_parallel(self):
    #     global pool
    #     task_num = len(self.triangles) // self.cpu_num
    #     results = []
    #     for i in range(self.cpu_num):
    #         start = i * task_num
    #         end = (i + 1) * task_num if i != self.cpu_num - 1 else len(self.triangles)
    #         results.append(pool.apply_async(
    #             self.fit_task_batch,
    #             args=(start, end)
    #         ))
    #     centers, axes, angles = [], [], []
    #     for res in results:
    #         center, axis, angle = res.get()
    #         centers.append(center)
    #         axes.append(axis)
    #         angles.append(angle)
    #     centers = np.concatenate(centers, axis=0)
    #     axes = np.concatenate(axes, axis=0)
    #     angles = np.concatenate(angles, axis=0)
    #     return centers, axes, angles

    @torch.no_grad()
    def post_process(self, points, H, W):
        points = add_boundary_points(points, H, W)

        # CPU version
        # tri = Delaunay(points, incremental=True)
        # simplices = tri.simplices
        
        # GPU version with cupy
        points = cp.asarray(points)
        tri = DelaunayGPU(points)
        simplices = tri.simplices
        # There are some bugs with tri.neighbors, so compute neighbors manually
        
        self.triangles = points[simplices]  # [N, 3, 2]
        self.triangles = torch.tensor(self.triangles, dtype=torch.float32).cuda()

        # cpu version
        # ellipses_center, ellipses_size, ellipses_angle = self.fit_ellipse_parallel()
        # ellipses_center = torch.tensor(ellipses_center, dtype=torch.float32).cuda()
        # ellipses_size = torch.tensor(ellipses_size, dtype=torch.float32).cuda()
        # ellipses_angle = torch.tensor(ellipses_angle, dtype=torch.float32).cuda()
        
        # cuda version
        ellipses_center, ellipses_size, ellipses_angle = self.fit_ellipse_cuda()
        
        mask = ~ellipse_filter(ellipses_size)
        self.triangles = self.triangles[mask]
        ellipses_center = ellipses_center[mask]
        ellipses_size = ellipses_size[mask]
        ellipses_angle = ellipses_angle[mask]
        
        # simplices to neighbors
        simplices = torch.tensor(simplices, dtype=torch.int32).cuda()
        simplices = simplices[mask]
        self.neighbors = simplices_to_neighbors(simplices)
        
        scale_param = torch.tensor([W, H], dtype=torch.float32, device=self.triangles.device)

        self.triangles = self.triangles[:, :, :2] / scale_param * 2 - 1
        ellipses_center = ellipses_center[:, :2] / scale_param * 2 - 1
        ellipses_center = torch.clamp(ellipses_center, -0.99999, 0.99999)

        ellipses_size = ellipses_size * 0.5
        # ellipses_angle to [0, 1]
        ellipses_angle = ellipses_angle / 360
        ellipses_angle = torch.clamp(ellipses_angle, 0.00001, 0.99999)

        # neighbors = tri.neighbors
        # neighbors = neighbors_process(neighbors)
        # neighbors = torch.tensor(neighbors, dtype=torch.int32, device=self.triangles.device)

        return self.triangles, ellipses_center, ellipses_size, ellipses_angle, self.neighbors

    @torch.no_grad()
    def process(self, pf, kernel_size):
        B, H, W = pf.shape[0], pf.shape[1], pf.shape[2]
        sampled_xy = dither_image(pf, kernel_size=kernel_size)
        elements = self.post_process(sampled_xy, H, W)
        return elements
