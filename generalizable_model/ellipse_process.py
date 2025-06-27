import os
import numpy as np
import torch
from scipy.spatial import Delaunay
from generalizable_model.utils import add_boundary_points, min_bounding_ellipse, dither_image, neighbors_process
from ellipse_fit import fit_ellipses

# pool = None
# def init_pool(cpu_num):
#     global pool
#     pool = torch.multiprocessing.Pool(cpu_num)

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
        points = torch.tensor(points, dtype=torch.float32).cuda()
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
        # Delaunay is the most time consumption on CPU, however I can't find a way to accelerate it.
        tri = Delaunay(points, incremental=True)
        simplices = tri.simplices
        self.triangles = points[simplices]  # [N, 3, 2]
        self.triangles = torch.tensor(self.triangles, dtype=torch.float32).cuda()

        # cpu version
        # ellipses_center, ellipses_size, ellipses_angle = self.fit_ellipse_parallel()
        # ellipses_center = torch.tensor(ellipses_center, dtype=torch.float32).cuda()
        # ellipses_size = torch.tensor(ellipses_size, dtype=torch.float32).cuda()
        # ellipses_angle = torch.tensor(ellipses_angle, dtype=torch.float32).cuda()
        
        # cuda version
        ellipses_center, ellipses_size, ellipses_angle = self.fit_ellipse_cuda()
        
        scale_param = torch.tensor([W, H], dtype=torch.float32, device=self.triangles.device)

        self.triangles = self.triangles[:, :, :2] / scale_param * 2 - 1
        ellipses_center = ellipses_center[:, :2] / scale_param * 2 - 1
        ellipses_center = torch.clamp(ellipses_center, -0.99999, 0.99999)

        ellipses_size = ellipses_size * 0.5
        # ellipses_angle to [0, 1]
        ellipses_angle = ellipses_angle / 360
        ellipses_angle = torch.clamp(ellipses_angle, 0.00001, 0.99999)

        neighbors = tri.neighbors
        neighbors = neighbors_process(neighbors)
        neighbors = torch.tensor(neighbors, dtype=torch.int32, device=self.triangles.device)

        return self.triangles, ellipses_center, ellipses_size, ellipses_angle, neighbors

    @torch.no_grad()
    def process(self, pf, kernel_size):
        B, H, W = pf.shape[0], pf.shape[1], pf.shape[2]
        sampled_xy = dither_image(pf, kernel_size=kernel_size)
        elements = self.post_process(sampled_xy, H, W)
        return elements
