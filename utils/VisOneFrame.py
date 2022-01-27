# -*- coding: utf-8 -*-
"""
Created on  16.09.2020

visualization one from ground truth, must check before training

--voxel can show voxel visualization, not use

@author: lihan
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
import argparse
from numpy.lib.function_base import corrcoef
import pandas as pd
import os
# from mayavi import mlab
from math import cos, sin
import numba
import open3d


def Readoneframe(PathName):
    points = np.fromfile(PathName, dtype=np.float32).reshape(-1, 4)
    print(points)
    return points[:, 0:4].astype(np.float32)


def ReadSquenceFrame(PathName, nNext):
    import yaml
    TransformPath = "/media/liangdao/DATA/waymoTest/datasetNew/training/Transform.yaml"
    with open(TransformPath, 'r') as f:
        TransformMatrix = yaml.full_load(f.read())
    PathNameBase = "/".join(PathName.split("/")[:-1])
    PathNameShort = PathName.split("/")[-1]
    PathNameInt = int(PathNameShort.split(".")[0])
    points = Readoneframe(PathName)
    rot_mat_base = np.array(
        TransformMatrix[str(PathNameInt).zfill(6)]).reshape((4, 4))[:3, :3]
    points[:, :3] = points[:, :3] @ np.linalg.inv(rot_mat_base)
    tran_mat_base = np.array(
        TransformMatrix[str(PathNameInt).zfill(6)]).reshape((4, 4))[:3, 3]
    for i in range(nNext):
        rot_mat = np.array(TransformMatrix[str(PathNameInt -
                                               1).zfill(6)]).reshape(
                                                   (4, 4))[:3, :3]
        tran_mat = np.array(
            TransformMatrix[str(PathNameInt - 1).zfill(6)]).reshape((4, 4))[:3,
                                                                            3]
        tran_mat = tran_mat - tran_mat_base
        points_next = Readoneframe(PathNameBase + '/' +
                                   str(PathNameInt - 1).zfill(6) + ".bin")
        points_next[:, :
                    3] = points_next[:, :3] @ np.linalg.inv(rot_mat) + tran_mat
        points = np.concatenate((points, points_next), axis=0)
        PathNameInt -= 1
    points[:, :3] = points[:, :3] @ rot_mat_base
    return points[:, 0:4].astype(np.float32)


@numba.njit
def creat_coordinate(voxel_x, voxel_y, voxel_z):

    voxel_xyz = np.zeros(
        (voxel_x.shape[0] * voxel_y.shape[0] * voxel_z.shape[0], 3))

    count = 0
    for x in voxel_x:
        for y in voxel_y:
            for z in voxel_z:
                voxel_xyz[count] = [x, y, z]
                count += 1
    return voxel_xyz


if __name__ == "__main__":

    #save_path = os.path.abspath('.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='.bin')

    parser.add_argument('--label_path', type=str, help='.txt')

    parser.add_argument('--save_path', type=str, help='network checkpoint')

    parser.add_argument('--show', action='store_true', help='show result')

    parser.add_argument('--voxel', action='store_true', help='show result')

    parser.add_argument('--intensity', action='store_true', help='show result')

    parser.add_argument('--sequence',
                        type=int,
                        help='an integer for the sequence')

    args = parser.parse_args()

    if args.sequence:
        data = ReadSquenceFrame(args.file_path, args.sequence)
    else:
        data = Readoneframe(args.file_path)

    print('numbers of points : {}'.format(data.shape))
    print(f'max intensity: {np.max(data[:,3])}')
    print(f'max x: {np.max(data[:,0])}')
    print(f'min x: {np.min(data[:,0])}')
    print(f'max y: {np.max(data[:,1])}')
    print(f'min y: {np.min(data[:,1])}')
    print(f'max z: {np.max(data[:,2])}')
    print(f'min z: {np.min(data[:,2])}')

    if args.intensity:
        import matplotlib.pyplot as plt
        intensity = data[:, 3] * 100
        plt.hist(intensity, bins=120)  #,bins=120)
        plt.show()

    voxel_x = np.arange(-88, 88, 1)
    voxel_y = np.arange(-32, 32, 1)
    voxel_z = np.arange(-3, 3, 1)

    voxel_xyz = creat_coordinate(voxel_x, voxel_y, voxel_z)

    if args.label_path:
        label = pd.read_csv(args.label_path,
                            sep=' ',
                            header=None,
                            index_col=None)
        label = label[label.iloc[:, 0] != "DontCare"]
        print(label)

        n = len(label)
        box_points = []

        for i in range(n):
            box = label.iloc[i]
            points = np.zeros((8, 3))

            x = np.float(box[11])
            y = np.float(box[12])
            z = np.float(box[13])
            angle = box[14]  #-np.pi/2
            depth = np.float(box[8])
            width = np.float(box[9])
            lang = np.float(box[10])

            A_top = [
                x + lang / 2 * cos(angle) + width / 2 * sin(angle),
                y + lang / 2 * sin(angle) - width / 2 * cos(angle),
                z + depth / 2
            ]
            A_under = [
                x + lang / 2 * cos(angle) + width / 2 * sin(angle),
                y + lang / 2 * sin(angle) - width / 2 * cos(angle),
                z - depth / 2
            ]

            B_top = [
                x + lang / 2 * cos(angle) - width / 2 * sin(angle),
                y + width / 2 * cos(angle) + lang / 2 * sin(angle),
                z + depth / 2
            ]
            B_under = [
                x + lang / 2 * cos(angle) - width / 2 * sin(angle),
                y + width / 2 * cos(angle) + lang / 2 * sin(angle),
                z - depth / 2
            ]

            C_top = [
                2 * x - (x + lang / 2 * cos(angle) + width / 2 * sin(angle)),
                2 * y - (y + lang / 2 * sin(angle) - width / 2 * cos(angle)),
                z + depth / 2
            ]
            C_under = [
                2 * x - (x + lang / 2 * cos(angle) + width / 2 * sin(angle)),
                2 * y - (y + lang / 2 * sin(angle) - width / 2 * cos(angle)),
                z - depth / 2
            ]

            D_top = [
                2 * x - (x + lang / 2 * cos(angle) - width / 2 * sin(angle)),
                2 * y - (y + width / 2 * cos(angle) + lang / 2 * sin(angle)),
                z + depth / 2
            ]
            D_under = [
                2 * x - (x + lang / 2 * cos(angle) - width / 2 * sin(angle)),
                2 * y - (y + width / 2 * cos(angle) + lang / 2 * sin(angle)),
                z - depth / 2
            ]

            points[0] = A_top
            points[1] = B_top
            points[2] = B_under
            points[3] = A_under
            points[4] = D_top
            points[5] = C_top
            points[6] = C_under
            points[7] = D_under

            box_points.append(points)

    if args.show:
        #for i in labels:
        #pts = extract_points(data_npy,i)
        #pts = mlab.points3d(pts[:,0],pts[:,1],pts[:,2],colors=(pts[:,3]/max_labels,pts[:,3]/max_labels,pts[:,3]/max_labels),
        #mode="sphere", scale_factor=0.3)
        vis = open3d.visualization.Visualizer()
        vis.create_window(window_name="Cali")
        vis.get_render_option().point_size = 6
        vis.get_render_option().line_width = 10
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        colors = plt.get_cmap('gist_rainbow')(data[:, 3])
        pc_all = open3d.geometry.PointCloud()
        pc_all.points = open3d.utility.Vector3dVector(data[:, :-1])
        pc_all.colors = open3d.utility.Vector3dVector(colors[:, :-1])
        if args.label_path:

            lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                     [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

            corr = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
            lines1 = [[0, 1], [0, 2], [0, 3]]

            CSLine = open3d.geometry.LineSet(
                points=open3d.utility.Vector3dVector(corr),
                lines=open3d.utility.Vector2iVector(
                    lines1),  #索引转化为open3d中的线的表示
            )
            vis.add_geometry(CSLine)

            for points__ in box_points:
                cornersLine = open3d.geometry.LineSet(
                    points=open3d.utility.Vector3dVector(points__),
                    lines=open3d.utility.Vector2iVector(
                        lines),  #索引转化为open3d中的线的表示
                )
                vis.add_geometry(cornersLine)

        vis.add_geometry(pc_all)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        """
        for i in range(len(box_points)):
            x = [np.append(box_points[i][:4,0],box_points[i][0,0]),np.append(box_points[i][4:,0],box_points[i][4,0])]
            y = [np.append(box_points[i][:4,1],box_points[i][0,1]),np.append(box_points[i][4:,1],box_points[i][4,1])]
            z = [np.append(box_points[i][:4,2],box_points[i][0,2]),np.append(box_points[i][4:,2],box_points[i][4,2])]
            
            mlab.mesh(x, y, z, line_width=4.5 ,representation='wireframe', color=(0, 1, 0))
    

    mlab.plot3d([0,0], [0,0], [0,1] ,line_width=1,representation='wireframe',color=(0,1,0))
    mlab.plot3d([0,0], [0,1], [0,0], representation='wireframe', line_width=1,color=(0,1,0))
    mlab.plot3d([0,1], [0,0], [0,0], representation='wireframe', line_width=1,color=(0,1,0))
    mlab.points3d(data[:,0],data[:,1],data[:,2],data[:,3],scale_mode="none",scale_factor=0.2,colormap="rainbow")

    if args.voxel:
        mlab.points3d(voxel_xyz[:,0],voxel_xyz[:,1],voxel_xyz[:,2],mode="cube",scale_factor=0.8,opacity=0.1)

    mlab.orientation_axes()
    mlab.show()
    """
