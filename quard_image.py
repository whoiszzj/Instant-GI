import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
from collections import deque

from networkx.algorithms.cluster import triangles
from scipy.integrate import simps
from scipy.spatial import Delaunay
import cv2
from skimage.draw import ellipse


# Define quadtree node class
class QuadTreeNode:
    def __init__(self, x, y, x_size, y_size):
        self.x = x  # x-coordinate of the top-left corner of the region
        self.y = y  # y-coordinate of the top-left corner of the region
        self.x_size = x_size  # size of the region
        self.y_size = y_size


def load_image(image_path):
    image = Image.open(image_path)
    # Add a GaussianBlur
    image = image.filter(ImageFilter.GaussianBlur(1))
    ori_size = image.size
    max_size = max(image.size)
    ori_image = image.copy()
    # Ensure the image is square
    image = image.crop((0, 0, max_size, max_size))
    return np.array(image) / 255.0, ori_size, ori_image


def min_bounding_ellipse(vertices):
    midpoints = (vertices[:3] + np.roll(vertices[:3], -1, axis=0)) / 2
    vertices = np.vstack((vertices[:3], midpoints))
    # OpenCV requires input point set as 2D array
    vertices = vertices.reshape(-1, 1, 2).astype(np.float32)
    ellipse = cv2.fitEllipse(vertices)
    # Return ellipse parameters: center coordinates, major and minor axes, rotation angle
    center, axes, angle = ellipse
    return center, axes, angle


class QuardImage:
    def __init__(self, image_path, threshold=0.02):
        self.pad_image, self.ori_size, self.ori_image = load_image(image_path)
        self.threshold = threshold
        self.height, self.width, _ = self.pad_image.shape
        self.image_sum = None
        self.image_sumsq = None
        self.magic_num = 0.27
        self.pre_compute()

    def pre_compute(self):
        self.image_sum = np.cumsum(np.cumsum(self.pad_image, axis=0), axis=1).astype(np.double)
        self.image_sumsq = np.cumsum(np.cumsum(self.pad_image ** 2, axis=0), axis=1).astype(np.double)
        self.image_sum = np.pad(self.image_sum, ((1, 0), (1, 0), (0, 0)), mode='constant')
        self.image_sumsq = np.pad(self.image_sumsq, ((1, 0), (1, 0), (0, 0)), mode='constant')

    def query_std(self, x1, y1, x2, y2):
        """Query the standard deviation of a region"""
        sum_sq = self.image_sumsq[y2, x2] - self.image_sumsq[y1, x2] - self.image_sumsq[y2, x1] \
                 + self.image_sumsq[y1, x1]
        sum_ = self.image_sum[y2, x2] - self.image_sum[y1, x2] - self.image_sum[y2, x1] + self.image_sum[y1, x1]
        mean = sum_ / ((y2 - y1) * (x2 - x1))
        diff = sum_sq / ((y2 - y1) * (x2 - x1)) - mean ** 2
        diff[diff < 0] = 0
        return np.sqrt(diff)

    def get_dynamic_threshold(self, x):
        # return -0.0003021 * x + 0.0312084  # x = 100  y=0.001; x = 4  y = 0.03
        return -0.0001979 * x + 0.0207916 # x = 100  y=0.001; x = 4  y = 0.02
        # return -0.00009375 * x + 0.010375  # x = 100  y=0.001; x = 4  y = 0.01

    def is_uniform(self, node: QuadTreeNode):
        """Determine if the current region is uniform enough, if uniform then stop splitting"""
        std = self.query_std(node.x, node.y, node.x + node.x_size, node.y + node.y_size)
        return std.max() < self.get_dynamic_threshold(max(node.x_size, node.y_size))

    def split(self, get_triangle=False):
        start_time = time.time()
        nodes = []
        root = QuadTreeNode(0, 0, self.height, self.width)
        queue = deque([root])  # Initialize queue
        while queue:
            node = queue.popleft()  # Take a node from the queue
            if min(node.x_size, node.y_size) <= 4 or self.is_uniform(node):
                nodes.append(node)
                continue  # Region is small enough or uniform enough, skip

            x_half_size = node.x_size // 2
            y_half_size = node.y_size // 2

            children = [
                QuadTreeNode(node.x, node.y, x_half_size, y_half_size),
                QuadTreeNode(node.x + x_half_size, node.y, node.x_size - x_half_size, y_half_size),
                QuadTreeNode(node.x, node.y + y_half_size, x_half_size, node.y_size - y_half_size),
                QuadTreeNode(node.x + x_half_size, node.y + y_half_size, node.x_size - x_half_size,
                             node.y_size - y_half_size),
            ]

            x_half_split_left = QuadTreeNode(node.x, node.y, x_half_size, node.y_size)
            x_half_split_right = QuadTreeNode(node.x + x_half_size, node.y, node.x_size - x_half_size, node.y_size)
            mask = [0, 0, 0, 0]
            if self.is_uniform(x_half_split_left):
                nodes.append(x_half_split_left)
                mask[0] = 1
                mask[2] = 1
            if self.is_uniform(x_half_split_right):
                nodes.append(x_half_split_right)
                mask[1] = 1
                mask[3] = 1
            if mask == [1, 1, 1, 1]:
                continue
            y_half_split_top = QuadTreeNode(node.x, node.y, node.x_size, y_half_size)
            y_half_split_bottom = QuadTreeNode(node.x, node.y + y_half_size, node.x_size, node.y_size - y_half_size)
            if mask[0] == 0 and mask[1] == 0 and self.is_uniform(y_half_split_top):
                nodes.append(y_half_split_top)
                mask[0] = 1
                mask[1] = 1
            if mask[2] == 0 and mask[3] == 0 and self.is_uniform(y_half_split_bottom):
                nodes.append(y_half_split_bottom)
                mask[2] = 1
                mask[3] = 1

            for i in range(4):
                if mask[i] == 1:
                    continue
                queue.append(children[i])
        # Convert nodes to points
        points = []
        for i, node in enumerate(nodes):
            half_x_size = node.x_size / 2
            half_y_size = node.y_size / 2
            x_center = node.x + half_x_size
            y_center = node.y + half_y_size
            if x_center >= self.ori_size[0] or y_center >= self.ori_size[1]:
                continue
            # x_center = (x_center / self.ori_size[0] - 0.5) * 2
            # y_center = (y_center / self.ori_size[1] - 0.5) * 2
            points.append([x_center, y_center, half_x_size, half_y_size])
        points = np.array(points)
        # # Draw the points after splitting
        # plt.figure()
        # plt.imshow(self.ori_image)
        # plt.scatter(points[:, 0], points[:, 1], c='r', s=1)
        # plt.show()
        results = self.post_process(points[:, :2]) if not get_triangle else self.post_process_triangle(points[:, :2])
        end_time = time.time()
        return results[:, :5], end_time - start_time

    def compute_triangle_area(self, points):
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        return 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)

    def compute_ellipse_area(self, axes):
        a, b = axes
        return np.pi * a * b

    def point_in_triangle(self, p, triangle):
        # Determine if point p is inside the triangle
        x1, y1 = triangle[0]
        x2, y2 = triangle[1]
        x3, y3 = triangle[2]
        x, y = p
        area = self.compute_triangle_area(triangle)
        area1 = self.compute_triangle_area([(x, y), (x2, y2), (x3, y3)])
        area2 = self.compute_triangle_area([(x1, y1), (x, y), (x3, y3)])
        area3 = self.compute_triangle_area([(x1, y1), (x2, y2), (x, y)])
        return area == area1 + area2 + area3

    def post_process(self, points):
        # Add the four corners of the image to points
        points = np.concatenate(
            [
                points, [[0, 0], [self.ori_size[0], 0], [0, self.ori_size[1]], [self.ori_size[0], self.ori_size[1]]]
            ], axis=0
        )
        tri = Delaunay(points)
        simplices = tri.simplices
        ellipses = []
        ellipses_area = np.zeros(len(simplices)).astype(np.float32)
        points_total_area = np.zeros(len(points)).astype(np.float32)
        # error_count = 0
        # Later use cv2.fitEllipse() to fit ellipses
        # debug_point = np.array([252, 514]).astype(np.float32)
        for i in range(len(simplices)):
            triangle = simplices[i]
            vertices = points[triangle]
            center, axes, angle = min_bounding_ellipse(vertices)
            if center[0] < 0 or center[0] > self.ori_size[0] or center[1] < 0 or center[1] > self.ori_size[1]:
                # error_count += 1
                continue
            if max(axes) / min(axes) > 10:
                # error_count += 1
                continue

            # if self.point_in_triangle(debug_point, vertices):
            #     print(f"idx:{ellipses.__len__()}, center:{center}, axes:{axes}, angle:{angle}")
            ellipses.append([center[0], center[1], axes[0], axes[1], angle])
            area = self.compute_ellipse_area(axes)
            ellipses_area[i] = area
            for j in range(3):
                points_total_area[triangle[j]] += area
        ellipses = np.array(ellipses).astype(np.float32)
        # print(f"error count: {error_count}")
        # self.show_triangle_and_ellipses(points, simplices, ellipses)
        ellipses_area_mean = np.mean(ellipses_area)
        points_total_area_mean = np.mean(points_total_area)
        need_add_threshold = points_total_area_mean * 2
        need_add_idxs = np.where(points_total_area > need_add_threshold)[0]
        added_point_areas = (points_total_area[need_add_idxs] / points_total_area_mean) * ellipses_area_mean
        added_point_radius = np.sqrt(added_point_areas / np.pi) * 0.4
        added_circle = np.zeros((len(need_add_idxs), 5))
        added_circle[:, :2] = points[need_add_idxs]
        added_circle[:, 2:4] = added_point_radius[:, None]
        added_circle[:, 4] = 0
        ellipses = np.concatenate([ellipses, added_circle], axis=0)
        # Convert pixel to xy coordinates
        ellipses[:, :2] = ellipses[:, :2] / np.array([self.ori_size[0], self.ori_size[1]]) * 2 - 1
        ellipses[:, 2:4] = ellipses[:, 2:4] * self.magic_num
        ellipses[:, 4] = np.maximum(ellipses[:, 4], 0.001) / 360
        return ellipses

    def post_process_triangle(self, points):
        # Add the four corners of the image to points
        points = np.concatenate(
            [
                points, [[0, 0], [self.ori_size[0], 0], [0, self.ori_size[1]], [self.ori_size[0], self.ori_size[1]]]
            ], axis=0
        )
        tri = Delaunay(points)
        simplices = tri.simplices
        ellipses = []
        triangles = []
        tri_area = []
        for i in range(len(simplices)):
            triangle = simplices[i]
            vertices = points[triangle]
            center, axes, angle = min_bounding_ellipse(vertices)
            if center[0] < 0 or center[0] > self.ori_size[0] or center[1] < 0 or center[1] > self.ori_size[1]:
                # error_count += 1
                continue
            if max(axes) / min(axes) > 10:
                # error_count += 1
                continue
            triangles.append(vertices)
            tri_area.append(self.compute_triangle_area(vertices))
            # if self.point_in_triangle(debug_point, vertices):
            #     print(f"idx:{ellipses.__len__()}, center:{center}, axes:{axes}, angle:{angle}")
            ellipses.append([center[0], center[1], axes[0], axes[1], angle])

        ellipses = np.array(ellipses).astype(np.float32)
        triangles = np.array(triangles).astype(np.float32)  # [N, 3, 2]
        tri_area = np.array(tri_area).astype(np.float32)
        # Convert pixel to xy coordinates
        triangles = triangles / np.array([self.ori_size[0], self.ori_size[1]]) * 2 - 1
        ellipses[:, :2] = ellipses[:, :2] / np.array([self.ori_size[0], self.ori_size[1]]) * 2 - 1
        ellipses[:, 2:4] = ellipses[:, 2:4] * self.magic_num
        ellipses[:, 4] = np.maximum(ellipses[:, 4], 0.001) / 360
        return ellipses, triangles, tri_area

    def show_triangle_and_ellipses(self, points, tri, ellipses):
        plt.figure()
        ax = plt.gca()
        ax.invert_yaxis()
        plt.imshow(self.ori_image)
        # Draw triangles
        for i in range(len(tri)):
            triangle = tri[i]
            t = plt.Polygon(points[triangle], fill=None, edgecolor='red')
            plt.gca().add_patch(t)
            # e = ellipses[i]
            # center, axes, angle = e[:2], e[2:4], e[4]
            # ell = Ellipse(center, axes[0], axes[1], angle=angle, edgecolor='blue', fill=None)
            # plt.gca().add_patch(ell)

        plt.show()
        print("Done!")
