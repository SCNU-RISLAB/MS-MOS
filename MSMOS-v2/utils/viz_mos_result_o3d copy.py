#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# @author: Jiadai Sun

import numpy as np
import open3d as o3d
from kitti_utils import load_vertex, load_labels
from sklearn.cluster import DBSCAN
import random

if __name__ == '__main__':

    seq = "08"
    frame_id = [98, 218, 222, 1630, 1641, 4016]

    data_path = "/data/datasets/dataset/kitti_dataset/"
    path = {
        # "gtlabel": data_path,
        # "method1": "/the prediction result of method_1",
        # "method2": "/the prediction result of method_2",
        "ours": "/data/zlf/mos3d/mos3d_refine_e3_l1.5/2023-10-11-18:09_MotionRV_raw_valid_08"
    }

    for f_id in frame_id:
        str_fid = "%06d" % (f_id)
        print(str_fid)

        scan_path = f'{data_path}/sequences/{seq}/velodyne/{str_fid}.bin'
        scan = load_vertex(scan_path)

        for key, value in path.items():
            if key == 'gtlabel':
                label_path = f'{value}/sequences/{seq}/labels/{str_fid}.label'
            else:
                label_path = f'{value}/sequences/{seq}/predictions/{str_fid}.label'

            print(key)
            label, _ = load_labels(label_path)
            xyz_label = np.concatenate([scan[:, :3], label[:, np.newaxis]], axis=1)
            mov_xyz_label = xyz_label[xyz_label[:, 3] > 200]

            clustering = DBSCAN(eps=0.5, min_samples=6).fit(mov_xyz_label[:, :3])
            instance_labels = clustering.labels_
            moving_mask = (label > 200)
            instance_labels_full = -1 * np.ones_like(label)
            instance_labels_full[moving_mask] = instance_labels

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
            pcd.paint_uniform_color([0.25, 0.25, 0.25])
            colors = np.array(pcd.colors)
            # 获取实例标签的唯一值，忽略噪声点 (-1)
            unique_instances = np.unique(instance_labels_full[instance_labels_full != -1])
            # 为每个实例分配随机颜色
            for instance in unique_instances:
                # 为每个实例生成随机颜色
                color = [random.random(), random.random(), random.random()]
                
                # 将属于该实例的所有点赋予相同的颜色
                colors[instance_labels_full == instance] = color

            pcd.colors = o3d.utility.Vector3dVector(colors)

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f'{key}_seq{seq}_frame{f_id}', width=1000, height=1000)
            vis.add_geometry(pcd)
            # parameters = o3d.io.read_pinhole_camera_parameters("/home/user/Repo/LiDAR-MOS/ScreenCamera_2022-02-20-21-03-42.json")
            # ctr = vis.get_view_control()
            # ctr.convert_from_pinhole_camera_parameters(parameters)
            vis.run()
            vis.destroy_window()
