import cv2
import numpy as np
import open3d as o3d
import torch

import depth_pro


if __name__=="__main__":
    input_img_path = "data/1.jpg"
    output_depth_path = "results/depth.npy"
    output_depth_vis_path = "results/depth_vis.jpg"
    output_ply_path = "results/point_cloud.ply"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.float16)
    depth_model.eval()

    input_img, _, focal_px = depth_pro.load_rgb(input_img_path)
    input_img = transform(input_img)

    result = depth_model.infer(input_img, f_px=focal_px)

    output_depth = result["depth"].cpu().numpy()
    output_depth = output_depth * 1000.0
    output_focal_px = result["focallength_px"].cpu().numpy()

    depth_vis_img = np.uint8(cv2.normalize(output_depth, None, 0, 255, cv2.NORM_MINMAX))

    color_raw = o3d.io.read_image(input_img_path)
    depth_raw = o3d.geometry.Image(output_depth)
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    height, width = output_depth.shape
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, output_focal_px, output_focal_px, width * 0.5, height * 0.5)
    cam_intrinsic.intrinsic_matrix = [[output_focal_px, 0, width * 0.5],
                                      [0, output_focal_px, height * 0.5],
                                      [0, 0, 1]]
    
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.intrinsic = cam_intrinsic
    cam_param.extrinsic = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])

    output_ply = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, cam_param.intrinsic, cam_param.extrinsic)

    np.save(output_depth_path, output_depth)
    cv2.imwrite(output_depth_vis_path, depth_vis_img)
    o3d.io.write_point_cloud(output_ply_path, output_ply)
