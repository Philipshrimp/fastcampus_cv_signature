import os
import cv2
import numpy as np
import torch

# MMOCR
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox
# SAM
from segment_anything import SamPredictor, sam_model_registry


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

if __name__ == '__main__':
    inputs_dir = ''
    outputs_dir = ''

    text_det_cfg_path = 'configs/dbnetpp_swinv2_base_w16_in21k.py'
    text_det_ckpt_path = 'path../db_swin_mix_pretrain.pth'
    text_recog_cfg_path = 'mmocr_dev/configs/textrecog/abinet/abinet_20e_st-an_mj.py'
    text_recog_ckpt_path = 'path../abinet_20e_st-an_mj_20221005_012617-ead8c139.pth'
    sam_model_type = ''
    sam_ckpt_path = ''

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    mmocr_inferencer = MMOCRInferencer(
        text_det_cfg_path,
        text_det_ckpt_path,
        text_recog_cfg_path,
        text_recog_ckpt_path,
        device=device)

    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt_path)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    original_inputs = mmocr_inferencer._inputs_to_list(inputs_dir)

    for _, original_input in enumerate(original_inputs):
        input_img = cv2.imread(original_input)

        result = mmocr_inferencer(input_img, save_vis=True, out_dir=outputs_dir)['predictions'][0]
    
        rec_texts = result['rec_texts']
        det_polygons = result['det_polygons']

        det_bboxes = torch.tensor(np.array([poly2bbox(poly) for poly in det_polygons]), device=device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(det_bboxes, input_img.shape[:2])

        sam_predictor.set_image(input_img, image_format='BGR')

        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        for mask, rec_text, polygon in zip(masks, rec_texts, det_polygons):
            show_mask(mask.cpu(), plt.gca(), random_color=True)
            polygon = np.array(polygon).reshape(-1, 2)
            polygon = np.concatenate([polygon, polygon[:1]], axis=0)
