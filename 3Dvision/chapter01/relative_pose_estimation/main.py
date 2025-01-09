import numpy as np

import matcher
import optimizer


if __name__=="__main__":
    src_path = "phototourism/british_museum/00350405_2611802704.jpg"
    dst_path = "phototourism/british_museum/01858319_78150445.jpg"

    src_kpts, dst_kpts = matcher.compute_correspondence_matching(src_path, dst_path)
    matcher.visualize(src_path, dst_path, src_kpts, dst_kpts)
    src_kpts, dst_kpts = matcher.reject_outliers(src_kpts, dst_kpts)

    K = np.array([[1024.0, 0, 531.0], 
         [0, 1024.0, 391.0], 
         [0, 0, 1]])
    
    R, t = optimizer.get_optimal_pose(src_kpts, dst_kpts, K)

    print(R)
    print(t)
