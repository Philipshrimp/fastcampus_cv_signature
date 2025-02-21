from matplotlib import pyplot as plt
import numpy as np

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid


if __name__=='__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_224_linear"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=224)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=batch_size)

    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    scene_imgs = scene.imgs
    scene_focals = scene.get_focals()
    scene_poses = scene.get_im_poses()
    scene_pts3d = scene.get_pts3d()
    scene_conf_mask = scene.get_masks()

    scene.show()

    pts2d_list, pts3d_list = [], []

    for i in range(2):
        conf_i = scene_conf_mask[i].cpu().numpy()
        pts2d_list.append(xy_grid(*scene_imgs[i].shape[:2][::-1])[conf_i])
        pts3d_list.append(scene_pts3d[i].detach().cpu().numpy()[conf_i])

    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)

    matches_img1 = pts2d_list[1][reciprocal_in_P2]
    matches_img0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    n_visualize = 10

    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_visualize)).astype(int)
    viz_matches_img0, viz_matches_img1 = matches_img0[match_idx_to_viz], matches_img1[match_idx_to_viz]

    H0, W0, H1, W1 = *scene_imgs[0].shape[:2], *scene_imgs[1].shape[:2]

    result_img0 = np.pad(scene_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    result_img1 = np.pad(scene_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)

    result_img = np.concatenate((result_img0, result_img1), axis=1)

    plt.figure()
    plt.imshow(result_img)

    color_map = plt.get_cmap('jet')

    for i in range(n_visualize):
        (x0, y0), (x1, y1) = viz_matches_img0[i].T, viz_matches_img1[i].T
        plt.plot([x0, x1 + W0], [y0, y1], '-+', color=color_map(i / (n_visualize - 1)), scalex=False, scaley=False)

    plt.show(block=True)
