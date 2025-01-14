import cv2
import random
import torch

import detector, sam


if __name__=='__main__':
    input_path = 'COCO-128-2/train/000000000034_jpg.rf.a33e87d94b16c1112e8d9946fee784b9.jpg'
    output_path = 'result.png'

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    detect_cfg_path = 'configs/detr_r50_8xb2-150e_coco.py'
    detect_ckpt_path = 'weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
    sam_model_type = 'vit_b'
    sam_ckpt_path = 'weights/sam_vit_b_01ec64.pth'

    detect_model = detector.initialize(detect_cfg_path, detect_ckpt_path, device)
    sam_model = sam.initialize(sam_model_type, sam_ckpt_path)

    bboxes = detector.inference(input_path, detect_model)

    input_img = cv2.imread(input_path)

    for bbox in bboxes:
        random_color = [random.randint(0, 255) for _ in range(3)]
        sam_mask = sam.inference(input_path, bbox, sam_model)
        input_img = sam.visualize(input_img, sam_mask, random_color)

    cv2.imwrite(output_path, input_img)
