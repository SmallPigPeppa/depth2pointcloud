import numpy as np
import math
import cv2
import sys
import os
import os.path as osp
import logging
from utils import colormaps

def depth_image_to_point_cloud(rgb, depth, scale, vFov,hFov, pose):
    '''
    1. convert vF hF to camera intrinsics(3×3 matrix)
    2. convert pose(world2pose) to pose(camera2world)
    '''

    # vF hF to intrinsics(3×3 matrix)
    W = rgb.shape[1] / 2.0
    H = rgb.shape[0] / 2.0

    K00 = W / math.tan(hFov / 2.0)
    K11= H / math.tan(vFov / 2.0)
    # K00 = 1.
    # K11= 1.
    K02 = W
    K12 = H

    # pose(world2pose) to pose(camera2world)
    pose = np.linalg.inv(pose)

    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale

    # X = (u - K[0, 2]) * Z / K[0, 0]
    # Y = (v - K[1, 2]) * Z / K[1, 1]

    X = (u - K02) * Z / K00
    Y = (v - K12) * Z / K11

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    # print(pose)
    # pose_i=np.linalg.inv(pose)
    # print(pose_i)
    # print(np.dot(pose,pose_i))
    position = np.dot(pose,position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return points




def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


def visualize_depth(depth, depth_min=None, depth_max=None):
    """Visualize the depth map with colormap.

    Rescales the values so that depth_min and depth_max map to 0 and 1,
    respectively.
    """
    if depth_min is None:
        depth_min = np.nanmin(depth)

    if depth_max is None:
        depth_max = np.nanmax(depth)

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled = depth_scaled ** 0.5
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_img=((cv2.applyColorMap(
        depth_scaled_uint8, colormaps.cm_magma) / 255) ** 2.2) * 255
    cv2.imwrite("/home/wendy.liu/depth.png", depth_scaled_uint8)
    return depth_scaled_uint8

if __name__=='__main__':
    #
    # import matplotlib.pyplot as plt
    # from utils import image_io
    # output_dir = "family_run_output"
    # paramters=np.load('family_run_output/camera_params/frame_000000.npz')
    # color_dir = osp.join("/tmp/cvd2", output_dir, "color_down_png")
    # vF,hF,pose_world2cam=paramters['vF'],paramters['hF'],paramters['world2cam']
    # rgb = plt.imread(osp.join(color_dir,'frame_000000.png'))
    # src_dir=osp.join("/tmp/cvd2", output_dir,
    #          "R0-10_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/depth_e0000/e0000_filtered/depth")
    # src_file='frame_000000.raw'
    # disparity = image_io.load_raw_float32_image(f"{src_dir}/{src_file}")
    # print(disparity.shape)
    # # print(vF,hF,pose_world2cam)
    # print(rgb.shape)
    #
    # points=depth_image_to_point_cloud(rgb=rgb,depth=disparity,scale=1, vFov=vF,hFov=hF, pose=pose_world2cam)
    # # print(points)
    # print(points[0])
    #
    #
    # write_point_cloud(ply_filename='/home/wendy.liu/frame_000000.ply', points=points)

    import matplotlib.pyplot as plt
    from utils import image_io
    root_dir='/tmp/cvd2'
    # root_dir = '/home/wendy.liu/cvd2'
    output_dir = "family_run_output"
    frame_no=1
    paramters=np.load(f'{root_dir}/{output_dir}/camera_params/frame_00000{frame_no}.npz')
    vF,hF,pose_world2cam=paramters['vF'],paramters['hF'],paramters['world2cam']
    rgb = cv2.imread((f'{root_dir}/{output_dir}/color_down_png/frame_00000{frame_no}.png'))
    depth_dir=osp.join(f"{root_dir}/{output_dir}/R0-10_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/depth_e0000/e0000_filtered/depth")
    depth_file=f'frame_00000{frame_no}.raw'

    d_min = sys.float_info.max
    d_max = sys.float_info.min
    print("reading '%s'." % depth_file)
    disparity = image_io.load_raw_float32_image(f"{depth_dir}/{depth_file}")
    # disparity=1./disparity
    d_range=disparity.max()
    disparity=2.+d_range-disparity
    # disparity=disparity *0.05

    ix = np.isfinite(disparity)

    if np.sum(ix) == 0:
        logging.warning(f"{depth_file} has 0 valid depth")

    valid_disp = disparity[ix]
    min_percentile=0.
    max_percentile=100.
    d_min = min(d_min, np.percentile(valid_disp, min_percentile))
    d_max = max(d_max, np.percentile(valid_disp, max_percentile))
    _ = visualize_depth(disparity, d_min, d_max)


    print(disparity[0])
    # print(vF,hF,pose_world2cam)
    print(rgb.shape)

    points=depth_image_to_point_cloud(rgb=rgb,depth=disparity,scale=1, vFov=vF,hFov=hF, pose=pose_world2cam)
    # print(points)
    print(points[0])


    write_point_cloud(ply_filename=f'/home/wendy.liu/frame_00000{frame_no}.ply', points=points)