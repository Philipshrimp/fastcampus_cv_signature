import numpy as np


def normalize_points(input_pts):
    mean = np.mean(input_pts, axis=0)
    std = np.std(input_pts)

    T = np.array([[1/std, 0, -mean[0]/std],
                  [0, 1/std, -mean[1]/std],
                  [0, 0, 1]])
    
    input_homogeneous = np.hstack([input_pts, np.ones((input_pts.shape[0], 1))])
    input_normalized = (T @ input_homogeneous.T).T

    return input_normalized, T

def construct_A_matrix(src_kpts, dst_kpts):
    A = []

    for i in range(src_kpts.shape[0]):
        x1, y1 = src_kpts[i, 0], src_kpts[i, 1]
        x2, y2 = dst_kpts[i, 0], dst_kpts[i, 1]
        A.append([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])

    return np.array(A)

def compute_fundamental_matrix(src_kpts, dst_kpts):
    src_norm, T_src = normalize_points(src_kpts)
    dst_norm, T_dst = normalize_points(dst_kpts)

    A = construct_A_matrix(src_norm[:, :2], dst_norm[:, :2])

    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F_norm)
    S[-1] = 0
    F_norm = U @ np.diag(S) @ V

    F = T_dst.T @ F_norm @ T_src

    return F

def compute_m_estimator(src_kpts, dst_kpts, max_iterations=2000):
    best_F = None
    best_error = float('inf')

    for _ in range(max_iterations):
        indices = np.random.choice(len(src_kpts), 8, replace=False)
        src_sampled = src_kpts[indices]
        dst_sampled = dst_kpts[indices]

        F = compute_fundamental_matrix(src_sampled, dst_sampled)

        src_kpts_h = np.hstack((src_kpts, np.ones((src_kpts.shape[0], 1))))
        dst_kpts_h = np.hstack((dst_kpts, np.ones((dst_kpts.shape[0], 1))))
        errors = np.abs(np.einsum('ij,jk->i', src_kpts_h, F @ dst_kpts_h.T))

        threshold_index = int(len(errors) * 0.1)
        top_errors = np.partition(errors, threshold_index)[:threshold_index]

        top_mean_error = np.mean(top_errors)
        mean_error = np.mean(errors)

        if mean_error < best_error:
            best_error = mean_error
            best_top_error = top_mean_error
            best_F = F
        
        if (best_top_error < 2.0):
            return best_F
        
    return best_F

def compute_pose(src_kpts, dst_kpts, K):
    F = compute_m_estimator(src_kpts, dst_kpts)
    E = K.T @ F @ K

    U, _ ,Vt = np.linalg.svd(E)

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    R1 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0:
        R1 = -R1

    R2 = U @ W @ Vt
    if np.linalg.det(R2) < 0:
        R2 = -R2

    t_x = -U[:, 2]
    t = (t_x / np.linalg.norm(t_x)).reshape(-1, 1)

    return R1, R2, t
