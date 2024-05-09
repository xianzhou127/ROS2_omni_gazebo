import os
os.environ['QT_QPA_PLATFORM'] = 'minimal'

import faulthandler
faulthandler.enable()

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import triangle as tr
import time

# def map_preprocess(cvmap, threshold, safe_radius):


# def map_find_opti_obstacles(obstacle_map, epsilon):
#     def find_non_extern_contours(hierarchy):
#         # 找到每一行中第一个不为-1的元素
#         valid_values = [row[np.where(row != -1)[0][0]] for row in hierarchy[0]]
#         return valid_values

#     contours, hierarchy = cv2.findContours(obstacle_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     approx_contours = []
#     for i in find_non_extern_contours(hierarchy):
#         contour         = contours[i]
#         approx_contour  = cv2.approxPolyDP(contour, epsilon, True)
#         approx_contours.append(approx_contour)
#     return approx_contours



# def map_obstacles_triangulate(approx_contours):
#     def create_cyclic_array(n):
#         # 创建一个空数组来存放边缘
#         edges = []
#         # 生成边缘
#         for i in range(n):
#             edges.append([i, (i + 1) % n])
#         # 转换成 numpy 数组
#         return np.array(edges)

#     ob_triangles = []
#     for approx_contour in approx_contours:
#         approx_contour_reshape        = approx_contour.reshape(-1,2)
#         point_len, _                  = approx_contour_reshape.shape
#         approx_contour_approx_contour = tr.triangulate(
#                 dict(vertices=approx_contour_reshape, segments=create_cyclic_array(point_len)),
#                 'p'
#             )
#         for t in approx_contour_approx_contour['triangles']:
#             x1, y1 = approx_contour_approx_contour['vertices'][t[0]]
#             x2, y2 = approx_contour_approx_contour['vertices'][t[1]]
#             x3, y3 = approx_contour_approx_contour['vertices'][t[2]]
#             ob_triangles.append((x1, y1, x2, y2, x3, y3))
#     return ob_triangles


class OptMap(object):
    def __init__(self, grid_map:np.ndarray) -> None:
        self.grid_map       = grid_map
        self.binary_map     = None
        self.triangle_map   = None
        self.nn_map         = None

    def create_binary_map(self, threshold:int=250, safe_radius:int=8) -> None:
        """ 从灰度图构建二值图，并对障碍物区域进行膨胀

        Args:
            threshold：二值化的阈值，大于此被设为255（可行域），小于次会被设置为0（障碍物）
            safe_radius：安全边距，即膨胀的大小（单位cm）
        """
        # 二值化
        self.grid_map[self.grid_map >= threshold] = 255
        self.grid_map[self.grid_map <  threshold] = 0
        # 利用erode操作扩大黑色（障碍物）的面积
        self.binary_map = cv2.erode(self.grid_map, np.ones((safe_radius, safe_radius), dtype=np.uint8), iterations=2)

    def create_triangle_map(self, max_triangle_num:float = 128) -> None:
        """ 将障碍物用三角形进行切分

        Args:
            max_triangle_num：最大的三角形个数
        """

        # 判断二值图是否存在
        if self.binary_map is None:
            self.create_binary_map()

        # 将二值图的边缘改成白色，方便切割
        tmp_binary_map = self.binary_map.copy()
        tmp_binary_map[0,:]  = 255
        tmp_binary_map[:,0]  = 255
        tmp_binary_map[-1,:] = 255
        tmp_binary_map[:,-1] = 255
        w,h                  = tmp_binary_map.shape
        tmp_binary_map[int(0.5*w),0:int(0.1*w)]   = 255
        tmp_binary_map[int(0.5*w),-int(0.1*w):-1] = 255

        # 定义一些工具函数

        def find_contours(tmp_binary_map, epsilon:int):
            """对障碍物的轮廓用多边形进行拟合

            Args:
                epsilon：运行的误差范围，单位为cm
            """

            def find_non_extern_contours(hierarchy):
                """找到每一行中第一个不为-1的元素"""
                valid_values = [row[np.where(row != -1)[0][0]] for row in hierarchy[0]]
                return valid_values

            approx_contours = []
            contours, hierarchy = cv2.findContours(tmp_binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i in find_non_extern_contours(hierarchy):
                contour         = contours[i]
                approx_contour  = cv2.approxPolyDP(contour, epsilon, True)
                approx_contours.append(approx_contour)
            return approx_contours

        def counters_to_triangles(approx_contours):
            """将障碍物的多边形轮廓分割为三角形"""

            def create_cyclic_array(n):
                """创建一个空数组来存放边缘"""
                edges = []
                # 生成边缘
                for i in range(n):
                    edges.append([i, (i + 1) % n])
                # 转换成 numpy 数组
                return np.array(edges)

            ob_triangles = []
            for approx_contour in approx_contours:
                approx_contour_reshape        = approx_contour.reshape(-1,2)
                point_len, _                  = approx_contour_reshape.shape
                approx_contour_approx_contour = tr.triangulate(
                        dict(vertices=approx_contour_reshape, segments=create_cyclic_array(point_len)),
                        'p'
                    )
                for t in approx_contour_approx_contour['triangles']:
                    x1, y1 = approx_contour_approx_contour['vertices'][t[0]]
                    x2, y2 = approx_contour_approx_contour['vertices'][t[1]]
                    x3, y3 = approx_contour_approx_contour['vertices'][t[2]]
                    ob_triangles.append((x1, y1, x2, y2, x3, y3))
            return ob_triangles

        # 初始化 epsilon 范围
        triangles       = None
        triangles_last  = None


        for epsilon in range(2,10):
            try:
                contours  = find_contours(tmp_binary_map, epsilon)
                triangles = counters_to_triangles(contours)
                triangles_last = triangles
            except:
                break

            print("epsilon = ", epsilon, " len(triangles) = ", len(triangles))
            if ( len(triangles) >= max_triangle_num) :
                triangles_last = triangles
                epsilon += 1
            else:
                break

        self.triangle_map = triangles_last

    def show_triangle_map(self):
        if self.triangle_map is None:
            self.create_triangle_map(128)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # Loop over the list of triangles and plot each one
        for triangle in self.triangle_map:
            x1, y1, x2, y2, x3, y3 = triangle
            triangle_x = [x1, x2, x3, x1]  # Close the triangle by returning to the first point
            triangle_y = [y1, y2, y3, y1]
            ax.plot(triangle_x, triangle_y, 'k-')  # 'k-' sets the color to black and the line style to solid
        # Setting labels and showing the plot
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Plot of Triangles')
        plt.show()

    def is_in_ob_by_binary(self, points):
        """ 通过二值图判断是否位于障碍物

        Args:
            points = [(x1,y1) , (x2,y2), ...]
        """

        # 判断二值图是否存在
        if self.binary_map is None:
            self.create_binary_map()

        points = np.array(points)  # 转换列表为NumPy数组
        x_coords, y_coords = points[:, 0], points[:, 1]  # 分割x和y坐标

        # 使用NumPy的高级索引一次性获取所有点的像素值
        pixel_values = self.binary_map[x_coords.astype(int), y_coords.astype(int)]

        # 返回一个布尔数组，表示每个点是否位于障碍物
        return pixel_values > 100



