import numpy as np

import pose_estimator


def triangulate_points(P_src, P_dst, src_kpts, dst_kpts):
    pts4D = np.zeros((4, src_kpts.shape[0]))

    for i in range(src_kpts.shape[0]):
        A = np.array([
            src_kpts[i, 0] * P_src[2, :] - P_src[0, :],
            src_kpts[i, 1] * P_src[2, :] - P_src[1, :],
            dst_kpts[i, 0] * P_dst[2, :] - P_dst[0, :],
            dst_kpts[i, 1] * P_dst[2, :] - P_dst[1, :]
        ])

        _, _, Vt = np.linalg.svd(A)
        pts4D[:, i] = Vt[-1]

    pts3D = pts4D[:3, :]/ pts4D[3, :]

    return pts3D.T

def calculate_reprojection_error(pts3D, pts2D, K, R, t):
    if pts3D.shape[0] == 3:
        pts3D = pts3D.T

    pts3D_cam = R @ pts3D.T + t
    pts2D_proj_h = K @ pts3D_cam
    pts2D_proj = pts2D_proj_h[:2, :] / pts2D_proj_h[2, :]
    pts2D_proj = pts2D_proj.T

    errors = np.sqrt(np.sum((pts2D - pts2D_proj) ** 2, axis=1))

    threshold_index = int(len(errors) * 0.1)
    top_errors = np.partition(errors, threshold_index)[:threshold_index]

    mean_error = np.mean(top_errors)

    return mean_error

def get_avg_error(src_kpts, dst_kpts, K, R, t):
    P_src = np.hstack((K, np.zeros((3, 1))))
    P_dst = K @ np.hstack((R, t))

    pts3D = triangulate_points(P_src, P_dst, src_kpts, dst_kpts)

    error_src = calculate_reprojection_error(pts3D, src_kpts, K, np.eye(3), np.zeros((3, 1)))
    error_dst = calculate_reprojection_error(pts3D, dst_kpts, K, R, t)

    avg_error = (error_src + error_dst) / 2

    return avg_error

def get_optimal_pose(src_kpts, dst_kpts, K, error_threshold=5.0):
    min_avg_error = float('inf')
    best_R = None
    best_t = None

    while (min_avg_error > error_threshold):
        R1, R2, t = pose_estimator.compute_pose(src_kpts, dst_kpts, K)

        avg_first = get_avg_error(src_kpts, dst_kpts, K, R1, t)
        avg_second = get_avg_error(src_kpts, dst_kpts, K, R2, t)

        if (avg_first < avg_second):
            avg_error = avg_first
            R = R1
        else:
            avg_error = avg_second
            R = R2

        if (avg_error < min_avg_error):
            min_avg_error = avg_error
            best_R = R
            best_t = t

    return best_R, best_t
