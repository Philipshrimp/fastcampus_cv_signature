from matplotlib import pyplot as plt
import numpy as np
import torch
import sys

sys.path.append("dust3r/")

from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R


if __name__=='__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    images = load_images(['data/9.jpg', 'data/32.jpg'], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    matches_img0, matches_img1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13)

    H0, W0 = view1['true_shape'][0]
    H1, W1 = view2['true_shape'][0]

    valid_matches_img0 = (matches_img0[:, 0] >= 3) & (matches_img0[:, 0] < int(W0) - 3) & (matches_img0[:, 1] >= 3) & (matches_img0 [:, 1] < int(H0) - 3)
    valid_matches_img1 = (matches_img1[:, 0] >= 3) & (matches_img1[:, 0] < int(W1) - 3) & (matches_img1[:, 1] >= 3) & (matches_img1 [:, 1] < int(H1) - 3)
    valid_matches = valid_matches_img0 & valid_matches_img1

    matches_img0, matches_img1 = matches_img0[valid_matches], matches_img1[valid_matches]

    num_viz = 20
    num_matches = matches_img0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, num_viz)).astype(int)

    viz_matches_img0, viz_matches_img1 = matches_img0[match_idx_to_viz], matches_img1[match_idx_to_viz]

    img_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    img_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []

    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * img_std + img_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]

    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0,), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0,), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)

    plt.figure()
    plt.imshow(img)

    color_map = plt.get_cmap('jet')

    for i in range(num_viz):
        (x0, y0), (x1, y1) = viz_matches_img0[i].T, viz_matches_img1[i].T
        plt.plot([x0, x1 + W0], [y0, y1], '-+', color=color_map(i / (num_viz - 1)), scalex=False, scaley=False)

    plt.show(block=True)
